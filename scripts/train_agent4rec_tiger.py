#!/usr/bin/env python3
"""Train a Hugging Face TIGER NTP checkpoint on Agent4Rec-format sequences."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Config, T5ForConditionalGeneration

from s2tpo.agent4rec import build_next_sid_samples, load_cf_sequences, load_sid_mapping


class NextSIDDataset(Dataset):
    def __init__(
        self,
        sequences: Dict[int, List[int]],
        item_to_sid: Dict[int, List[int]],
        max_history: int,
        max_samples: int,
    ) -> None:
        self.rows = build_next_sid_samples(
            sequences,
            item_to_sid,
            max_history=max_history,
            max_samples=max_samples,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def collate(rows):
    max_context = max(len(row[0]) for row in rows)
    input_ids = torch.zeros((len(rows), max_context), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.tensor([row[1] for row in rows], dtype=torch.long)
    for index, (context, _) in enumerate(rows):
        input_ids[index, -len(context) :] = torch.tensor(context, dtype=torch.long)
        attention_mask[index, -len(context) :] = 1
    return input_ids, attention_mask, labels


def model_config(size: str, vocab_size: int) -> T5Config:
    dimensions = {
        "mini": (3, 128, 512, 4, 32),
        "medium": (4, 256, 1024, 8, 32),
        "large": (6, 512, 2048, 8, 64),
    }
    layers, hidden, feedforward, heads, key_value = dimensions[size]
    return T5Config(
        vocab_size=vocab_size,
        d_model=hidden,
        d_ff=feedforward,
        num_layers=layers,
        num_decoder_layers=layers,
        num_heads=heads,
        d_kv=key_value,
        dropout_rate=0.1,
        pad_token_id=0,
        eos_token_id=0,
        decoder_start_token_id=0,
        feed_forward_proj="relu",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sequences", required=True, help="Agent4Rec cf_data/train.txt")
    parser.add_argument("--sid-mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-size", choices=("mini", "medium", "large"), default="mini")
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    sequences = load_cf_sequences(args.train_sequences)
    item_to_sid, valid_sids = load_sid_mapping(args.sid_mapping)
    dataset = NextSIDDataset(sequences, item_to_sid, args.max_history, args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collate,
    )
    vocab_size = max(token for sid in valid_sids for token in sid) + 1
    model = T5ForConditionalGeneration(model_config(args.model_size, vocab_size)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for input_ids, attention_mask, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
                use_cache=False,
                return_dict=True,
            )
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(output.loss.item()))
        mean_loss = sum(losses) / max(len(losses), 1)
        history.append({"epoch": epoch + 1, "ntp_loss": mean_loss})
        print(f"epoch={epoch + 1} ntp_loss={mean_loss:.6f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to("cpu").save_pretrained(output_dir)
    (output_dir / "valid_sids.json").write_text(
        json.dumps({"valid_sids": valid_sids}), encoding="utf-8"
    )
    (output_dir / "training_metrics.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"trained TIGER samples={len(dataset)} output={output_dir}")


if __name__ == "__main__":
    main()
