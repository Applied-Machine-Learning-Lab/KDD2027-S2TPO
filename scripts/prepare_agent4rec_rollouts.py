#!/usr/bin/env python3
"""Convert Agent4Rec behavior sessions into S2T-PO critic and context JSONL."""

import argparse
import json
from pathlib import Path

from s2tpo.agent4rec import (
    discounted_returns,
    encode_history,
    load_behavior_sessions,
    load_cf_sequences,
    load_sid_mapping,
    page_feedback,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--train-sequences", required=True)
    parser.add_argument("--sid-mapping", required=True)
    parser.add_argument("--critic-sessions", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--valid-sids", required=True)
    parser.add_argument("--page-size", type=int, default=6)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    histories = load_cf_sequences(args.train_sequences)
    item_to_sid, valid_sids = load_sid_mapping(args.sid_mapping)
    sessions = load_behavior_sessions(args.run_dirs)
    critic_rows = []
    context_rows = []
    for session in sessions:
        user_id = int(session["user_id"])
        history = list(histories.get(user_id, []))
        retained = []
        rewards = []
        for page in session["pages"]:
            recommended = [int(item) for item in page.get("recommended_id", [])]
            if len(recommended) != args.page_size:
                raise ValueError(
                    f"session {session['run']}:{user_id} contains a "
                    f"{len(recommended)}-item page; expected {args.page_size}"
                )
            if any(item not in item_to_sid for item in recommended):
                raise ValueError(
                    f"session {session['run']}:{user_id} contains an unmapped item"
                )
            context = encode_history(history, item_to_sid, max_history=args.max_history)
            if not context:
                raise ValueError(
                    f"session {session['run']}:{user_id} has no mappable history"
                )
            evidence, utility = page_feedback(page)
            retained.append(
                {
                    "context": context,
                    "page_sids": [item_to_sid[item] for item in recommended],
                    "item_evidence": evidence,
                }
            )
            rewards.append(utility)
            aligned = [int(item) for item in page.get("align_id", [])]
            history.extend(item for item in aligned if item in item_to_sid)

        returns = discounted_returns(rewards, args.gamma)
        for page_index, (row, session_return) in enumerate(zip(retained, returns), start=1):
            critic_rows.append(
                {
                    "schema_version": "s2tpo_critic_session_v1",
                    "context_input_ids": row["context"],
                    "context_attention_mask": [1] * len(row["context"]),
                    "page_sids": row["page_sids"],
                    "session_return": session_return,
                    "item_evidence": row["item_evidence"],
                    "source": "agent4rec",
                    "session_id": f"{session['run']}:{user_id}",
                    "page_index": page_index,
                }
            )
            context_rows.append(
                {
                    "context_input_ids": row["context"],
                    "context_attention_mask": [1] * len(row["context"]),
                }
            )

    if not critic_rows:
        raise ValueError(
            "no paper-aligned pages were found; verify that Agent4Rec used the requested page size"
        )
    session_count = write_jsonl(critic_rows, args.critic_sessions)
    context_count = write_jsonl(context_rows, args.contexts)
    valid_path = Path(args.valid_sids)
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.write_text(json.dumps({"valid_sids": valid_sids}), encoding="utf-8")
    print(
        f"prepared Agent4Rec pages={session_count} contexts={context_count} "
        f"gamma={args.gamma}"
    )


if __name__ == "__main__":
    main()
