"""Export page-wise records to the canonical complete-page group contract."""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

from .groups import CandidateGroup


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects with line-numbered validation errors."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} must contain one JSON object")
            yield payload


def assemble_complete_page_group(
    page_records: Sequence[Dict[str, Any]],
) -> CandidateGroup:
    """Combine one page-wise record per candidate into one paper-aligned group."""
    if len(page_records) < 2:
        raise ValueError("a candidate group requires at least two complete pages")

    indexed_records = list(enumerate(page_records))
    candidate_indices = [record.get("candidate_page_index") for record in page_records]
    if all(index is not None for index in candidate_indices):
        numeric_indices = [int(index) for index in candidate_indices]
        if len(set(numeric_indices)) != len(numeric_indices):
            raise ValueError("candidate_page_index values must be distinct")
        indexed_records.sort(key=lambda pair: int(pair[1]["candidate_page_index"]))

    ordered_records = [record for _, record in indexed_records]
    context = _required_vector(ordered_records[0], "context_input_ids", "input_ids")
    attention = _optional_vector(
        ordered_records[0], "context_attention_mask", "attention_mask"
    )
    if not attention:
        attention = [1] * len(context)

    candidate_sids: List[List[List[int]]] = []
    member_paths: List[List[List[List[float]]]] = []
    for record in ordered_records:
        record_context = _required_vector(record, "context_input_ids", "input_ids")
        record_attention = _optional_vector(
            record, "context_attention_mask", "attention_mask"
        )
        if not record_attention:
            record_attention = [1] * len(record_context)
        if record_context != context or record_attention != attention:
            raise ValueError("all candidate pages in one group must share one context")

        raw_page = record.get("candidate_sids", record.get("target_tokens"))
        if raw_page is None:
            raise ValueError("page record is missing candidate_sids/target_tokens")
        raw_paths = record.get("critic_member_path_values")
        if raw_paths is None:
            raise ValueError("page record is missing critic_member_path_values")
        candidate_sids.append(
            [[int(token) for token in item] for item in raw_page]
        )
        member_paths.append(
            [
                [
                    [float(value) for value in member_values]
                    for member_values in item_path
                ]
                for item_path in raw_paths
            ]
        )

    group = CandidateGroup(context, attention, candidate_sids, member_paths)
    group.validate()
    return group


def group_page_records(records: Iterable[Dict[str, Any]]) -> List[CandidateGroup]:
    """Group page-wise builder records by context-level group ID."""
    buckets: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for record in records:
        group_id = record.get("group_id")
        if group_id is None:
            raise ValueError("every page record must contain group_id")
        buckets.setdefault(str(group_id), []).append(record)
    if not buckets:
        raise ValueError("no page records were provided")
    return [assemble_complete_page_group(rows) for rows in buckets.values()]


def write_groups_jsonl(groups: Iterable[CandidateGroup], path: str) -> int:
    """Write canonical candidate groups and return the number of records."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for group in groups:
            group.validate()
            handle.write(json.dumps(group.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError("cannot write an empty candidate-group file")
    return count


def _required_vector(
    record: Dict[str, Any], canonical_name: str, page_name: str
) -> List[int]:
    values = record.get(canonical_name, record.get(page_name))
    if values is None:
        raise ValueError(f"page record is missing {canonical_name}/{page_name}")
    return [int(value) for value in values]


def _optional_vector(
    record: Dict[str, Any], canonical_name: str, page_name: str
) -> List[int]:
    values = record.get(canonical_name, record.get(page_name, []))
    return [int(value) for value in values]
