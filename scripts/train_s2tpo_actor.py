#!/usr/bin/env python3
"""Train a TIGER actor from audited S2T-PO candidate-group JSONL data."""

import argparse
import json
import random
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM

from s2tpo.groups import collate_groups, load_groups
from s2tpo.objectives import soft_update
from s2tpo.tiger_actor import s2tpo_actor_loss, set_decoder_trainable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="TIGER/Hugging Face checkpoint")
    parser.add_argument("--groups", required=True, help="Candidate-group JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=7e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--beta-q", type=float, default=0.5)
    parser.add_argument(
        "--advantage-clip",
        type=float,
        default=0.0,
        help="positive symmetric clip value; 0 disables advantage clipping",
    )
    parser.add_argument("--clip-epsilon", type=float, default=0.20)
    parser.add_argument("--kl-coefficient", type=float, default=0.03)
    parser.add_argument("--ema-retention", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.ema_retention) < 1.0:
        raise ValueError("ema-retention must be in [0, 1)")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    groups = load_groups(args.groups)
    loader = DataLoader(
        groups,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=partial(collate_groups, pad_token_id=args.pad_token_id),
    )
    actor = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    old_policy = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()
    for parameter in old_policy.parameters():
        parameter.requires_grad = False
    set_decoder_trainable(actor)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in actor.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    actor.train()
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = s2tpo_actor_loss(
                actor,
                old_policy,
                batch,
                beta_q=args.beta_q,
                group_advantage_clip=(
                    args.advantage_clip if args.advantage_clip > 0.0 else None
                ),
                clip_epsilon=args.clip_epsilon,
                kl_coefficient=args.kl_coefficient,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in actor.parameters() if parameter.requires_grad),
                args.grad_clip,
            )
            optimizer.step()
            metrics["epoch"] = epoch + 1
            history.append(metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the frozen old policy as the EMA target instead of loading a third model.
    actor.to("cpu")
    old_policy.to("cpu")
    soft_update(old_policy, actor, tau=1.0 - args.ema_retention)
    actor.save_pretrained(output_dir / "learner")
    old_policy.save_pretrained(output_dir / "rollout_ema")
    (output_dir / "training_metrics.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
