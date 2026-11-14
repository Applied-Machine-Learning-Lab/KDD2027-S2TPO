#!/usr/bin/env python3
"""Install the S2T-PO adapter and evaluate it with an external Agent4Rec Arena."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def parse_metrics(path: Path):
    metrics = {}
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        try:
            metrics[name.strip()] = float(value.strip())
        except ValueError:
            continue
    return metrics


def install_adapter(agent4rec_root: Path, dataset: str) -> None:
    source = Path(__file__).resolve().parents[1] / "integrations" / "agent4rec" / "S2TPO.py"
    model_dir = agent4rec_root / "recommenders" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, model_dir / "S2TPO.py")
    weights = agent4rec_root / "recommenders" / "weights" / dataset / "S2TPO" / "Saved"
    weights.mkdir(parents=True, exist_ok=True)
    args_payload = {
        "candidate": False,
        "batch_size": 2048,
        "neg_sample": 1,
        "IPStype": "cn",
        "num_workers": 0,
        "nodrop": False,
        "infonce": 0,
        "modeltype": "S2TPO",
        "dataset": dataset,
        "cuda": 0,
    }
    (weights / "args.txt").write_text(json.dumps(args_payload, indent=2), encoding="utf-8")
    torch.save({"epoch": 0, "state_dict": {}}, weights / "epoch=0.checkpoint.pth.tar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent4rec-root", required=True)
    parser.add_argument("--dataset", required=True, choices=("ml-1m", "all-beauty"))
    parser.add_argument("--model", required=True, help="S2T-PO learner or rollout_ema directory")
    parser.add_argument("--sid-mapping", required=True)
    parser.add_argument("--eval-name", default="s2tpo")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--n-avatars", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--page-size", type=int, default=6)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--execution-mode", choices=("serial", "parallel"), default="serial")
    parser.add_argument("--output", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.agent4rec_root).resolve()
    main_script = root / "main.py"
    if not main_script.exists():
        raise ValueError("Agent4Rec root must contain main.py")
    install_adapter(root, args.dataset)
    env = os.environ.copy()
    env["S2TPO_MODEL"] = str(Path(args.model).resolve())
    env["S2TPO_SID_MAP"] = str(Path(args.sid_mapping).resolve())
    env["S2TPO_PROPOSAL_COUNT"] = "192"
    results = []
    for seed in args.seeds:
        simulation_name = f"{args.eval_name}_seed{seed}"
        command = [
            sys.executable,
            str(main_script),
            "--dataset", args.dataset,
            "--modeltype", "S2TPO",
            "--model_path", "Saved",
            "--simulation_name", simulation_name,
            "--seed", str(seed),
            "--n_avatars", str(args.n_avatars),
            "--max_pages", str(args.max_pages),
            "--items_per_page", str(args.page_size),
            "--execution_mode", args.execution_mode,
            "--cuda", str(args.cuda),
        ]
        print("+", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, env=env, check=True)
        metrics_path = root / "storage" / args.dataset / "S2TPO" / simulation_name / "metrics.txt"
        results.append({"seed": seed, "metrics": parse_metrics(metrics_path)})
    output = Path(args.output) if args.output else Path("outputs") / f"agent4rec_{args.dataset}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"config": vars(args), "results": results}, indent=2), encoding="utf-8")
    print(f"Agent4Rec evaluation summary: {output}")


if __name__ == "__main__":
    main()
