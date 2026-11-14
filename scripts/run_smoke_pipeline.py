#!/usr/bin/env python3
"""Run critic training, complete-page rollout, and actor update on tiny local data."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from transformers import T5Config, T5ForConditionalGeneration


def run(command: list) -> None:
    print("+", " ".join(str(value) for value in command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    model_dir = root / "tiny_tiger"
    model = T5ForConditionalGeneration(
        T5Config(
            vocab_size=24,
            d_model=16,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            d_kv=8,
            pad_token_id=0,
            eos_token_id=0,
            decoder_start_token_id=0,
        )
    )
    model.save_pretrained(model_dir)

    valid_sids = [[1, 5], [1, 6], [2, 7], [2, 8], [3, 9], [3, 10], [4, 11], [4, 12]]
    (root / "valid_sids.json").write_text(
        json.dumps({"valid_sids": valid_sids}), encoding="utf-8"
    )
    contexts = [
        {"context_input_ids": [3, 4, 5], "context_attention_mask": [1, 1, 1]},
        {"context_input_ids": [6, 7, 8], "context_attention_mask": [1, 1, 1]},
    ]
    with (root / "contexts.jsonl").open("w", encoding="utf-8") as handle:
        for row in contexts:
            handle.write(json.dumps(row) + "\n")
    with (root / "critic_sessions.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(8):
            row = {
                "schema_version": "s2tpo_critic_session_v1",
                **contexts[index % 2],
                "page_sids": [valid_sids[index % 4], valid_sids[4 + index % 4]],
                "session_return": 0.5 + 0.1 * index,
                "item_evidence": [1.0 + index % 2, 2.0 - index % 2],
            }
            handle.write(json.dumps(row) + "\n")

    critic_dir = root / "critic"
    groups_path = root / "groups.jsonl"
    actor_dir = root / "actor"
    run(
        [
            sys.executable,
            "-m",
            "scripts.train_s2tpo_critic",
            "--sessions",
            str(root / "critic_sessions.jsonl"),
            "--output-dir",
            str(critic_dir),
            "--vocab-size",
            "24",
            "--sid-depth",
            "2",
            "--ensemble-size",
            "2",
            "--hidden-dim",
            "16",
            "--max-page-size",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "scripts.build_s2tpo_groups",
            "--model",
            str(model_dir),
            "--critic",
            str(critic_dir),
            "--contexts",
            str(root / "contexts.jsonl"),
            "--valid-sids",
            str(root / "valid_sids.json"),
            "--output",
            str(groups_path),
            "--group-size",
            "2",
            "--page-size",
            "2",
            "--proposal-count",
            "8",
            "--device",
            "cpu",
        ]
    )
    run([sys.executable, "-m", "scripts.validate_groups", str(groups_path)])
    run(
        [
            sys.executable,
            "-m",
            "scripts.train_s2tpo_actor",
            "--model",
            str(model_dir),
            "--groups",
            str(groups_path),
            "--output-dir",
            str(actor_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ]
    )
    print(f"smoke pipeline complete: {root}")


if __name__ == "__main__":
    main()
