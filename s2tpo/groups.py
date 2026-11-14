"""Validated data contract for complete-page S2T-PO candidate groups."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch import Tensor


SCHEMA_VERSION = "s2tpo_complete_page_group_v1"


@dataclass(frozen=True)
class CandidateGroup:
    """Complete pages generated independently for one request context."""

    context_input_ids: List[int]
    context_attention_mask: List[int]
    candidate_sids: List[List[List[int]]]
    critic_member_path_values: List[List[List[List[float]]]]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CandidateGroup":
        schema_version = payload.get("schema_version")
        if schema_version != SCHEMA_VERSION:
            if "target_tokens" in payload:
                raise ValueError(
                    "page-wise S2T-PO records must be converted with "
                    "scripts.export_complete_page_groups"
                )
            raise ValueError(
                f"expected schema_version={SCHEMA_VERSION!r}, "
                f"got {schema_version!r}"
            )
        context = [int(value) for value in payload["context_input_ids"]]
        attention = [int(value) for value in payload.get("context_attention_mask", [])]
        if not attention:
            attention = [1] * len(context)

        raw_candidates = payload["candidate_sids"]
        if raw_candidates and raw_candidates[0] and isinstance(raw_candidates[0][0], int):
            raise ValueError(
                "legacy slot-conditioned candidate_sids [W,L] are not supported; "
                "expected independently generated complete pages [W,N,L]"
            )
        candidates = [
            [[int(token) for token in item] for item in page]
            for page in raw_candidates
        ]
        member_paths = [
            [
                [
                    [float(value) for value in member_values]
                    for member_values in item_path
                ]
                for item_path in page_paths
            ]
            for page_paths in payload["critic_member_path_values"]
        ]
        group = cls(context, attention, candidates, member_paths)
        group.validate()
        return group

    def validate(self) -> None:
        if not self.context_input_ids:
            raise ValueError("context_input_ids cannot be empty")
        if len(self.context_input_ids) != len(self.context_attention_mask):
            raise ValueError("context_input_ids and context_attention_mask must align")
        if any(value not in (0, 1) for value in self.context_attention_mask):
            raise ValueError("context_attention_mask must contain only 0 or 1")
        if any(value < 0 for value in self.context_input_ids):
            raise ValueError("context_input_ids must be non-negative")
        if len(self.candidate_sids) < 2:
            raise ValueError("each group must contain at least two candidate pages")
        if len(self.candidate_sids) != len(self.critic_member_path_values):
            raise ValueError("candidate pages and critic paths must align")
        page_keys = {
            tuple(tuple(item) for item in page) for page in self.candidate_sids
        }
        if len(page_keys) != len(self.candidate_sids):
            raise ValueError("candidate pages within a group must be distinct")

        page_size = len(self.candidate_sids[0])
        if page_size < 1:
            raise ValueError("candidate pages cannot be empty")
        sid_depth = len(self.candidate_sids[0][0])
        if sid_depth < 1:
            raise ValueError("candidate item SIDs cannot be empty")
        first_path = self.critic_member_path_values[0][0]
        if len(first_path) != sid_depth + 1:
            raise ValueError("each item path must contain PAD plus one value per SID token")
        member_count = len(first_path[0])
        if member_count < 2:
            raise ValueError("critic paths must contain at least two ensemble members")

        for page, page_paths in zip(
            self.candidate_sids, self.critic_member_path_values
        ):
            if len(page) != page_size or len(page_paths) != page_size:
                raise ValueError("all candidate pages must share one page size")
            if len({tuple(item) for item in page}) != page_size:
                raise ValueError("items within a candidate page must be unique")
            for item, path in zip(page, page_paths):
                if len(item) != sid_depth:
                    raise ValueError("all candidate item SIDs must share one depth")
                if any(token < 0 for token in item):
                    raise ValueError("candidate SID token IDs must be non-negative")
                if len(path) != sid_depth + 1:
                    raise ValueError(
                        "each item path must contain PAD plus one value per SID token"
                    )
                for member_values in path:
                    if len(member_values) != member_count:
                        raise ValueError("critic ensemble width must be constant")
                    if not all(math.isfinite(value) for value in member_values):
                        raise ValueError("critic values must be finite")

            endpoint = page_paths[0][-1]
            for path in page_paths[1:]:
                if any(
                    not math.isclose(value, reference, rel_tol=1e-5, abs_tol=1e-6)
                    for value, reference in zip(path[-1], endpoint)
                ):
                    raise ValueError(
                        "all item paths in one page must share the complete-page endpoint"
                    )

    def to_dict(self) -> Dict[str, Any]:
        """Return the canonical paper-aligned JSON representation."""
        return {
            "schema_version": SCHEMA_VERSION,
            "context_input_ids": self.context_input_ids,
            "context_attention_mask": self.context_attention_mask,
            "candidate_sids": self.candidate_sids,
            "critic_member_path_values": self.critic_member_path_values,
        }


def load_groups(path: str) -> List[CandidateGroup]:
    """Load one validated complete-page candidate group per JSONL line."""
    groups: List[CandidateGroup] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                groups.append(CandidateGroup.from_dict(json.loads(text)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid group at line {line_number}: {exc}") from exc
    if not groups:
        raise ValueError(f"no candidate groups found in {path}")

    shape = _group_shape(groups[0])
    for group in groups[1:]:
        if _group_shape(group) != shape:
            raise ValueError(
                "all groups in one file must share candidate count, page size, "
                "SID depth, and critic ensemble width"
            )
    return groups


def _group_shape(group: CandidateGroup) -> tuple[int, int, int, int]:
    return (
        len(group.candidate_sids),
        len(group.candidate_sids[0]),
        len(group.candidate_sids[0][0]),
        len(group.critic_member_path_values[0][0][0]),
    )


def collate_groups(groups: Iterable[CandidateGroup], pad_token_id: int) -> Dict[str, Tensor]:
    """Collate groups into candidate pages with shape ``[B,W,N,L]``."""
    rows = list(groups)
    if not rows:
        raise ValueError("cannot collate an empty batch")
    expected_shape = _group_shape(rows[0])
    for row in rows:
        if _group_shape(row) != expected_shape:
            raise ValueError("complete-page group shape must be constant within a batch")
        if any(
            token == int(pad_token_id)
            for page in row.candidate_sids
            for item in page
            for token in item
        ):
            raise ValueError("PAD cannot appear inside a complete candidate SID")

    max_context = max(len(row.context_input_ids) for row in rows)
    input_ids = torch.full(
        (len(rows), max_context), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros((len(rows), max_context), dtype=torch.long)
    for index, row in enumerate(rows):
        length = len(row.context_input_ids)
        input_ids[index, :length] = torch.tensor(row.context_input_ids, dtype=torch.long)
        attention_mask[index, :length] = torch.tensor(
            row.context_attention_mask, dtype=torch.long
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "candidate_sids": torch.tensor(
            [row.candidate_sids for row in rows], dtype=torch.long
        ),
        "critic_member_path_values": torch.tensor(
            [row.critic_member_path_values for row in rows], dtype=torch.float32
        ),
    }
