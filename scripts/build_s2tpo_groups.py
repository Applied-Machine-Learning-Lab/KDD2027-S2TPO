#!/usr/bin/env python3
"""Generate complete candidate pages and critic paths for S2T-PO actor training."""

import argparse
import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForSeq2SeqLM

from s2tpo.critic import load_sid_page_critic
from s2tpo.export import write_groups_jsonl
from s2tpo.page_rollout import (
    build_sid_trie,
    decode_ranked_sids,
    evaluate_candidate_paths,
    make_candidate_group,
    sample_complete_pages,
)


def load_valid_sids(path: str) -> List[List[int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("valid_sids") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError("SID vocabulary must be a list or contain valid_sids")
    return [[int(token) for token in sid] for sid in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Frozen TIGER/Hugging Face checkpoint")
    parser.add_argument("--critic", required=True, help="Critic checkpoint directory")
    parser.add_argument("--contexts", required=True, help="KuaiRand/simulator context JSONL")
    parser.add_argument("--valid-sids", required=True, help="JSON SID vocabulary")
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=6)
    parser.add_argument("--proposal-count", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--critic-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    valid_sids = load_valid_sids(args.valid_sids)
    sid_depth = len(valid_sids[0])
    trie = build_sid_trie(valid_sids)
    policy = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()
    policy.requires_grad_(False)
    critic = load_sid_page_critic(args.critic, device=device).eval()
    critic.requires_grad_(False)
    generator = torch.Generator().manual_seed(args.seed)

    groups = []
    with Path(args.contexts).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            context = torch.tensor(row["context_input_ids"], dtype=torch.long, device=device).view(1, -1)
            raw_mask = row.get("context_attention_mask", [1] * context.shape[1])
            attention = torch.tensor(raw_mask, dtype=torch.long, device=device).view(1, -1)
            ranked = decode_ranked_sids(
                policy,
                context,
                attention,
                trie,
                sid_depth=sid_depth,
                proposal_count=max(args.proposal_count, args.group_size * args.page_size * 4),
                decoder_start_token_id=int(policy.config.decoder_start_token_id or 0),
            )
            pages = sample_complete_pages(
                ranked,
                page_size=args.page_size,
                group_size=args.group_size,
                temperature=args.temperature,
                generator=generator,
            )
            paths = evaluate_candidate_paths(
                critic,
                context,
                attention,
                pages,
                batch_size=args.critic_batch_size,
            )
            groups.append(make_candidate_group(context, attention, pages, paths))
    count = write_groups_jsonl(groups, args.output)
    print(f"generated groups={count} W={args.group_size} N={args.page_size} output={args.output}")


if __name__ == "__main__":
    main()
