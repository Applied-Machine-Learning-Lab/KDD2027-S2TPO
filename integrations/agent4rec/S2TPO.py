"""Drop-in Agent4Rec recommender backed by a trained S2T-PO actor."""

import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

from s2tpo.page_rollout import build_sid_trie, decode_ranked_sids


class S2TPO(nn.Module):
    """Agent4Rec model interface using static TIGER rankings per user."""

    def __init__(self, args, data) -> None:
        super().__init__()
        model_path = os.environ.get("S2TPO_MODEL", "").strip()
        sid_path = os.environ.get("S2TPO_SID_MAP", "").strip()
        if not model_path or not sid_path:
            raise ValueError("S2TPO_MODEL and S2TPO_SID_MAP must be set")
        self.device = torch.device(args.cuda)
        self.data = data
        self.n_items = int(data.n_items)
        self.item_to_sid, self.sid_to_item = _load_mapping(sid_path)
        self.trie = build_sid_trie(list(self.sid_to_item))
        self.sid_depth = len(next(iter(self.sid_to_item)))
        self.proposal_count = int(os.environ.get("S2TPO_PROPOSAL_COUNT", "192"))
        policy = AutoModelForSeq2SeqLM.from_pretrained(model_path).eval()
        # Keep the external Arena checkpoint hook empty; the actor owns weights.
        self.__dict__["_s2tpo_policy"] = policy.to(self.device)

    @property
    def policy(self):
        return self.__dict__["_s2tpo_policy"]

    def cuda(self, device):
        self.device = torch.device(device)
        self.__dict__["_s2tpo_policy"] = self.policy.to(self.device)
        return self

    def state_dict(self, *args, **kwargs):
        return {}

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict({}, strict=False)

    def predict(self, users, items=None):
        item_ids = np.arange(self.n_items, dtype=np.int64) if items is None else np.asarray(items)
        scores = np.full((len(users), len(item_ids)), -1e9, dtype=np.float32)
        item_positions = {int(item): index for index, item in enumerate(item_ids.tolist())}
        for row_index, user_id in enumerate(users):
            history = list(self.data.train_user_list.get(int(user_id), []))
            context = []
            for item_id in history[-50:]:
                context.extend(self.item_to_sid.get(int(item_id), []))
            if not context:
                continue
            input_ids = torch.tensor(context, dtype=torch.long, device=self.device).view(1, -1)
            attention_mask = torch.ones_like(input_ids)
            ranked = decode_ranked_sids(
                self.policy,
                input_ids,
                attention_mask,
                self.trie,
                sid_depth=self.sid_depth,
                proposal_count=self.proposal_count,
                decoder_start_token_id=int(self.policy.config.decoder_start_token_id or 0),
            )
            for sid, score in ranked:
                item_id = self.sid_to_item.get(tuple(sid))
                position = item_positions.get(int(item_id)) if item_id is not None else None
                if position is not None:
                    scores[row_index, position] = float(score)
        return scores


def _load_mapping(path: str) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, ...], int]]:
    item_to_sid: Dict[int, List[int]] = {}
    sid_to_item: Dict[Tuple[int, ...], int] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        item_column = "item_id" if "item_id" in names else "video_id"
        sid_columns = sorted(
            (name for name in names if name.startswith("sid_")),
            key=lambda name: int(name.split("_")[-1]),
        )
        for row in reader:
            item_id = int(row[item_column])
            sid = [int(row[column]) + 1 for column in sid_columns]
            item_to_sid[item_id] = sid
            sid_to_item[tuple(sid)] = item_id
    return item_to_sid, sid_to_item
