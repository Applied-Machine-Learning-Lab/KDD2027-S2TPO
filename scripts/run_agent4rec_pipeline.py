#!/usr/bin/env python3
"""Run one paper-default S2T-PO round from Agent4Rec behavior sessions."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def command(module: str, *arguments: object) -> None:
    values = [sys.executable, "-m", module, *(str(value) for value in arguments)]
    print("+", " ".join(values), flush=True)
    subprocess.run(values, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--train-sequences", required=True)
    parser.add_argument("--sid-mapping", required=True)
    parser.add_argument("--model", required=True, help="Frozen Hugging Face TIGER checkpoint")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/agent4rec_paper.json")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output_dir).resolve()
    prepared = output / "prepared"
    critic = output / "critic"
    groups = output / "candidate_groups.jsonl"
    actor = output / "actor"
    prepared.mkdir(parents=True, exist_ok=True)
    sessions = prepared / "critic_sessions.jsonl"
    contexts = prepared / "contexts.jsonl"
    valid_sids = prepared / "valid_sids.json"

    command(
        "scripts.prepare_agent4rec_rollouts",
        "--run-dirs", *args.run_dirs,
        "--train-sequences", args.train_sequences,
        "--sid-mapping", args.sid_mapping,
        "--critic-sessions", sessions,
        "--contexts", contexts,
        "--valid-sids", valid_sids,
        "--page-size", config["page_size"],
        "--max-history", config["max_history"],
        "--gamma", config["session_discount"],
    )
    command(
        "scripts.train_s2tpo_critic",
        "--sessions", sessions,
        "--output-dir", critic,
        "--vocab-size", args.vocab_size,
        "--sid-depth", config["sid_depth"],
        "--ensemble-size", config["critic_ensemble_size"],
        "--epochs", config["critic_epochs"],
        "--batch-size", config["critic_batch_size"],
        "--learning-rate", config["critic_learning_rate"],
        "--item-coefficient", config["critic_item_coefficient"],
        "--prefix-coefficient", config["critic_prefix_coefficient"],
        "--rank-coefficient", config["critic_rank_coefficient"],
        "--device", args.device,
    )
    command(
        "scripts.build_s2tpo_groups",
        "--model", args.model,
        "--critic", critic,
        "--contexts", contexts,
        "--valid-sids", valid_sids,
        "--output", groups,
        "--group-size", config["group_size"],
        "--page-size", config["page_size"],
        "--proposal-count", config["proposal_count"],
        "--device", args.device,
    )
    command("scripts.validate_groups", groups)
    command(
        "scripts.train_s2tpo_actor",
        "--model", args.model,
        "--groups", groups,
        "--output-dir", actor,
        "--epochs", config["actor_epochs"],
        "--batch-size", config["actor_batch_size"],
        "--learning-rate", config["actor_learning_rate"],
        "--beta-q", config["beta_q"],
        "--clip-epsilon", config["clip_epsilon"],
        "--kl-coefficient", config["kl_coefficient"],
        "--ema-retention", config["ema_retention"],
        "--device", args.device,
    )
    print(f"Agent4Rec S2T-PO round complete: {output}")


if __name__ == "__main__":
    main()
