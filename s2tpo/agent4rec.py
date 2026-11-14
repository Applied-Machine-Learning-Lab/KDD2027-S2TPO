"""Data adapters shared by the external Agent4Rec integration."""

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def load_cf_sequences(path: str) -> Dict[int, List[int]]:
    """Load Agent4Rec ``cf_data`` lines: user ID followed by item IDs."""
    sequences: Dict[int, List[int]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            values = [int(value) for value in line.strip().split()]
            if len(values) >= 2:
                sequences[values[0]] = values[1:]
    if not sequences:
        raise ValueError(f"no interaction sequences found in {path}")
    return sequences


def load_sid_mapping(path: str) -> Tuple[Dict[int, List[int]], List[List[int]]]:
    """Load zero-based SID codes and shift them by one to reserve PAD=0."""
    item_to_sid: Dict[int, List[int]] = {}
    valid_sids: List[List[int]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        item_column = "item_id" if "item_id" in names else "video_id"
        sid_columns = sorted(
            (name for name in names if name.startswith("sid_")),
            key=lambda name: int(name.split("_")[-1]),
        )
        if item_column not in names or not sid_columns:
            raise ValueError("SID mapping requires item_id/video_id and sid_* columns")
        for row in reader:
            item_id = int(row[item_column])
            sid = [int(row[column]) + 1 for column in sid_columns]
            item_to_sid[item_id] = sid
            valid_sids.append(sid)
    if len({tuple(sid) for sid in valid_sids}) != len(valid_sids):
        raise ValueError("SID mapping must be injective")
    return item_to_sid, valid_sids


def encode_history(
    history_items: Sequence[int],
    item_to_sid: Dict[int, List[int]],
    *,
    max_history: int,
) -> List[int]:
    """Flatten the most recent item SIDs into one TIGER encoder context."""
    tokens: List[int] = []
    for item_id in history_items[-max_history:]:
        sid = item_to_sid.get(int(item_id))
        if sid is not None:
            tokens.extend(sid)
    return tokens


def build_next_sid_samples(
    sequences: Dict[int, List[int]],
    item_to_sid: Dict[int, List[int]],
    *,
    max_history: int,
    max_samples: int = 0,
) -> List[Tuple[List[int], List[int]]]:
    """Construct next-SID imitation examples from historical sequences."""
    samples: List[Tuple[List[int], List[int]]] = []
    for sequence in sequences.values():
        history: List[int] = []
        for item_id in sequence:
            target = item_to_sid.get(item_id)
            context = encode_history(history, item_to_sid, max_history=max_history)
            if context and target is not None:
                samples.append((context, target))
                if max_samples > 0 and len(samples) >= max_samples:
                    return samples
            history.append(item_id)
    if not samples:
        raise ValueError("no valid TIGER training samples were constructed")
    return samples


def discounted_returns(rewards: Sequence[float], gamma: float) -> List[float]:
    returns = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + float(gamma) * running
        returns[index] = running
    return returns


def load_behavior_sessions(run_dirs: Iterable[str]) -> List[Dict[str, Any]]:
    """Load Agent4Rec per-user behavior pickle files from simulation runs."""
    sessions: List[Dict[str, Any]] = []
    for raw_dir in run_dirs:
        run_dir = Path(raw_dir)
        behavior_dir = run_dir / "behavior"
        for behavior_path in sorted(behavior_dir.glob("*.pkl")):
            with behavior_path.open("rb") as handle:
                behavior = pickle.load(handle)
            if not isinstance(behavior, dict) or not behavior:
                continue
            pages = [behavior[key] for key in sorted(behavior, key=lambda value: int(value))]
            sessions.append(
                {
                    "run": run_dir.name,
                    "user_id": int(behavior_path.stem),
                    "pages": pages,
                }
            )
    if not sessions:
        raise ValueError("no Agent4Rec behavior sessions were found")
    return sessions


def page_feedback(page: Dict[str, Any]) -> Tuple[List[float], float]:
    """Return item evidence and transparent page utility from Agent4Rec signals."""
    recommended = [int(item) for item in page.get("recommended_id", [])]
    aligned = {int(item) for item in page.get("align_id", [])}
    watched = {int(item) for item in page.get("watch_id", [])}
    ratings = {
        int(item): float(rating) / 5.0
        for item, rating in zip(page.get("rating_id", []), page.get("rating", []))
    }
    evidence = [
        float(item in aligned) + float(item in watched) + ratings.get(item, 0.0)
        for item in recommended
    ]
    page_size = max(len(recommended), 1)
    click_rate = sum(float(item in watched) for item in recommended) / page_size
    like_rate = sum(float(item in aligned) for item in recommended) / page_size
    rating_mean = sum(ratings.values()) / max(len(ratings), 1)
    continuation = 0.0 if bool(page.get("exit_flag", False)) else 1.0
    utility = click_rate + like_rate + rating_mean + continuation
    return evidence, utility


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
