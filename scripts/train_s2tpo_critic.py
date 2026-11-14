#!/usr/bin/env python3
"""Train the paper-aligned bootstrap page-critic ensemble from session JSONL."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from s2tpo.critic import SIDPageCriticEnsemble, save_sid_page_critic
from s2tpo.critic_objective import (
    critic_calibration_objective,
    online_bootstrap_weights,
)
from s2tpo.page_rollout import build_unique_path_states, restore_item_paths


class SessionPageDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.rows: List[Dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema_version") != "s2tpo_critic_session_v1":
                    raise ValueError(f"invalid critic schema at line {line_number}")
                self.rows.append(row)
        if not self.rows:
            raise ValueError("critic dataset is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]


def collate_sessions(rows: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_context = max(len(row["context_input_ids"]) for row in rows)
    context = torch.full((len(rows), max_context), pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(context)
    for index, row in enumerate(rows):
        tokens = torch.tensor(row["context_input_ids"], dtype=torch.long)
        context[index, : len(tokens)] = tokens
        raw_mask = row.get("context_attention_mask", [1] * len(tokens))
        attention[index, : len(tokens)] = torch.tensor(raw_mask, dtype=torch.long)
    pages = torch.tensor([row["page_sids"] for row in rows], dtype=torch.long)
    returns = torch.tensor([row["session_return"] for row in rows], dtype=torch.float32)
    evidence = torch.tensor([row["item_evidence"] for row in rows], dtype=torch.float32)
    if pages.ndim != 3 or evidence.shape != pages.shape[:2]:
        raise ValueError("page_sids [B,N,L] and item_evidence [B,N] must align")
    return {
        "context_input_ids": context,
        "context_attention_mask": attention,
        "page_sids": pages,
        "session_returns": returns,
        "item_evidence": evidence,
    }


def evaluate_training_paths(
    model: SIDPageCriticEnsemble,
    context_input_ids: torch.Tensor,
    context_attention_mask: torch.Tensor,
    pages: torch.Tensor,
) -> torch.Tensor:
    """Evaluate every displayed page's N masked-to-complete paths."""
    batch_size, page_size, sid_depth = pages.shape
    states = torch.cat(
        [build_unique_path_states(pages[index : index + 1]) for index in range(batch_size)],
        dim=0,
    )
    state_count = states.shape[1]
    flat_states = states.reshape(batch_size * state_count, page_size, sid_depth)
    context = context_input_ids[:, None, :].expand(-1, state_count, -1)
    context = context.reshape(batch_size * state_count, -1)
    mask = context_attention_mask[:, None, :].expand(-1, state_count, -1)
    mask = mask.reshape(batch_size * state_count, -1)
    values = model.member_values(context, mask, flat_states)
    return restore_item_paths(
        values.reshape(batch_size, state_count, model.size),
        page_size=page_size,
        sid_depth=sid_depth,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--sid-depth", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-page-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--item-coefficient", type=float, default=0.75)
    parser.add_argument("--prefix-coefficient", type=float, default=0.50)
    parser.add_argument("--rank-coefficient", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = SessionPageDataset(args.sessions)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=lambda rows: collate_sessions(rows, args.pad_token_id),
    )
    model = SIDPageCriticEnsemble(
        vocab_size=args.vocab_size,
        sid_depth=args.sid_depth,
        hidden_dim=args.hidden_dim,
        max_page_size=args.max_page_size,
        dropout=args.dropout,
        size=args.ensemble_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    model.train()
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            paths = evaluate_training_paths(
                model,
                batch["context_input_ids"],
                batch["context_attention_mask"],
                batch["page_sids"],
            )
            weights = online_bootstrap_weights(
                len(paths), model.size, device=device
            )
            losses = critic_calibration_objective(
                paths,
                batch["session_returns"],
                batch["item_evidence"],
                bootstrap_weights=weights,
                item_coefficient=args.item_coefficient,
                prefix_coefficient=args.prefix_coefficient,
                rank_coefficient=args.rank_coefficient,
            )
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(losses.total.item()),
                    "page_loss": float(losses.page.item()),
                    "item_loss": float(losses.item.item()),
                    "prefix_loss": float(losses.prefix.item()),
                    "rank_loss": float(losses.rank.item()),
                }
            )
    save_sid_page_critic(model.to("cpu"), args.output_dir)
    output = Path(args.output_dir)
    (output / "training_metrics.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"trained critic members={model.size} samples={len(dataset)} output={output}")


if __name__ == "__main__":
    main()
