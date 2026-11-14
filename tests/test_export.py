import json

import pytest

from s2tpo.export import (
    assemble_complete_page_group,
    group_page_records,
    iter_jsonl,
    write_groups_jsonl,
)
from s2tpo.groups import load_groups


def page_record(index: int) -> dict:
    endpoints = [[1.0 + index, 1.2 + index], [1.0 + index, 1.2 + index]]
    return {
        "group_id": "episode-1:page-2",
        "candidate_page_index": index,
        "input_ids": [10, 11],
        "attention_mask": [1, 1],
        "target_tokens": [
            [2 + 4 * index, 3 + 4 * index],
            [4 + 4 * index, 5 + 4 * index],
        ],
        "critic_member_path_values": [
            [[0.0, 0.1], [0.5, 0.6], endpoints[0]],
            [[0.2, 0.3], [0.7, 0.8], endpoints[1]],
        ],
    }


def test_page_records_form_one_canonical_group(tmp_path) -> None:
    group = assemble_complete_page_group([page_record(1), page_record(0)])
    assert group.candidate_sids[0] == page_record(0)["target_tokens"]
    assert len(group.candidate_sids) == 2

    output = tmp_path / "groups.jsonl"
    assert write_groups_jsonl([group], str(output)) == 1
    loaded = load_groups(str(output))
    assert loaded == [group]


def test_grouping_preserves_first_group_appearance() -> None:
    second_group = page_record(0)
    second_group["group_id"] = "second"
    second_group_peer = page_record(1)
    second_group_peer["group_id"] = "second"
    groups = group_page_records(
        [second_group, second_group_peer, page_record(0), page_record(1)]
    )
    assert groups[0].candidate_sids[0] == second_group["target_tokens"]
    assert groups[1].candidate_sids[0] == page_record(0)["target_tokens"]


def test_export_rejects_context_mismatch() -> None:
    first, second = page_record(0), page_record(1)
    second["input_ids"] = [99]
    second["attention_mask"] = [1]
    with pytest.raises(ValueError, match="share one context"):
        assemble_complete_page_group([first, second])


def test_export_rejects_duplicate_candidate_index() -> None:
    first, second = page_record(0), page_record(1)
    second["candidate_page_index"] = 0
    with pytest.raises(ValueError, match="must be distinct"):
        assemble_complete_page_group([first, second])


def test_jsonl_reader_reports_line_number(tmp_path) -> None:
    path = tmp_path / "pages.jsonl"
    path.write_text(json.dumps(page_record(0)) + "\n[]\n", encoding="utf-8")
    rows = iter_jsonl(str(path))
    assert next(rows)["candidate_page_index"] == 0
    with pytest.raises(ValueError, match="line 2"):
        next(rows)
