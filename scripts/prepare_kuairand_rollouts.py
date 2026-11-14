#!/usr/bin/env python3
"""Convert KuaiSim rollout traces into public S2T-PO critic/context contracts."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_sid_mapping(path: str, item_id_offset: int) -> Tuple[Dict[int, List[int]], List[List[int]]]:
    mapping: Dict[int, List[int]] = {}
    valid = []
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        sid_columns = sorted(
            (name for name in reader.fieldnames or [] if name.startswith("sid_")),
            key=lambda name: int(name.split("_")[-1]),
        )
        if not sid_columns or "video_id" not in (reader.fieldnames or []):
            raise ValueError("SID mapping requires video_id and sid_* columns")
        for row in reader:
            # Public TIGER reserves zero for PAD, hence the +1 token offset.
            sid = [int(row[column]) + 1 for column in sid_columns]
            environment_item_id = int(row["video_id"]) - int(item_id_offset)
            mapping[environment_item_id] = sid
            valid.append(sid)
    return mapping, valid


def discounted_returns(rows: List[Dict[str, Any]], gamma: float) -> List[float]:
    result = [0.0] * len(rows)
    running = 0.0
    for index in range(len(rows) - 1, -1, -1):
        running = float(rows[index].get("step_reward", 0.0)) + gamma * running
        result[index] = running
    return result


def context_tokens(row: Dict[str, Any], mapping: Dict[int, List[int]], max_history: int) -> List[int]:
    if row.get("context_input_ids"):
        return [int(token) for token in row["context_input_ids"]]
    history = [int(item) for item in row.get("history_items", [])][-max_history:]
    tokens: List[int] = []
    for item in history:
        sid = mapping.get(item)
        if sid is not None:
            tokens.extend(sid)
    if not tokens:
        raise ValueError("rollout row has no mappable history/context tokens")
    return tokens


def page_sids(row: Dict[str, Any], mapping: Dict[int, List[int]]) -> List[List[int]]:
    raw = row.get("selected_sid_tokens_list")
    if raw:
        return [[int(token) for token in sid] for sid in raw]
    result = []
    for item in row.get("selected_item_ids", []):
        sid = mapping.get(int(item))
        if sid is None:
            raise ValueError(f"selected item {item} is absent from SID mapping")
        result.append(sid)
    return result


def item_evidence(row: Dict[str, Any], page_size: int) -> List[float]:
    values = row.get("selected_item_rewards", [])
    if len(values) != page_size:
        values = [sum(float(value) for value in response) for response in row.get("selected_responses", [])]
    if len(values) != page_size:
        values = [1.0] * page_size
    return [max(float(value), 0.0) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--sid-mapping", required=True)
    parser.add_argument("--critic-sessions", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--valid-sids", required=True)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument(
        "--item-id-offset",
        type=int,
        default=-1,
        help="video_id = simulator item_id + offset; KuaiSim commonly uses -1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping, valid_sids = load_sid_mapping(args.sid_mapping, args.item_id_offset)
    episodes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with Path(args.trace).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                episodes[str(row["episode_id"])].append(row)
    if not episodes:
        raise ValueError("rollout trace is empty")

    session_path = Path(args.critic_sessions)
    context_path = Path(args.contexts)
    session_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with session_path.open("w", encoding="utf-8") as sessions, context_path.open("w", encoding="utf-8") as contexts:
        for episode_rows in episodes.values():
            episode_rows.sort(key=lambda row: int(row.get("page_index", 0)))
            returns = discounted_returns(episode_rows, args.gamma)
            for row, session_return in zip(episode_rows, returns):
                context = context_tokens(row, mapping, args.max_history)
                page = page_sids(row, mapping)
                evidence = item_evidence(row, len(page))
                sessions.write(
                    json.dumps(
                        {
                            "schema_version": "s2tpo_critic_session_v1",
                            "context_input_ids": context,
                            "context_attention_mask": [1] * len(context),
                            "page_sids": page,
                            "session_return": session_return,
                            "item_evidence": evidence,
                        }
                    )
                    + "\n"
                )
                contexts.write(
                    json.dumps(
                        {
                            "context_input_ids": context,
                            "context_attention_mask": [1] * len(context),
                        }
                    )
                    + "\n"
                )
                count += 1
    Path(args.valid_sids).write_text(
        json.dumps({"valid_sids": valid_sids}), encoding="utf-8"
    )
    print(f"prepared pages={count} episodes={len(episodes)} gamma={args.gamma}")


if __name__ == "__main__":
    main()
