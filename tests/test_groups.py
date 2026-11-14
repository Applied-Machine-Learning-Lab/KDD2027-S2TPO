import json

import pytest
import torch

from s2tpo.groups import CandidateGroup, collate_groups, load_groups


def valid_payload() -> dict:
    return {
        "schema_version": "s2tpo_complete_page_group_v1",
        "context_input_ids": [10, 11],
        "context_attention_mask": [1, 1],
        "candidate_sids": [
            [[2, 3], [4, 5]],
            [[6, 7], [8, 9]],
        ],
        "critic_member_path_values": [
            [
                [[0.0, 0.2], [0.4, 0.6], [0.8, 1.0]],
                [[0.1, 0.3], [0.5, 0.7], [0.8, 1.0]],
            ],
            [
                [[0.0, 0.2], [0.2, 0.4], [0.3, 0.5]],
                [[0.1, 0.3], [0.2, 0.4], [0.3, 0.5]],
            ],
        ],
    }


def test_group_collation_shapes_and_padding() -> None:
    first = CandidateGroup.from_dict(valid_payload())
    second_payload = valid_payload()
    second_payload["context_input_ids"] = [12]
    second_payload["context_attention_mask"] = [1]
    second = CandidateGroup.from_dict(second_payload)

    batch = collate_groups([first, second], pad_token_id=0)
    assert batch["input_ids"].shape == (2, 2)
    assert batch["candidate_sids"].shape == (2, 2, 2, 2)
    assert batch["critic_member_path_values"].shape == (2, 2, 2, 3, 2)
    torch.testing.assert_close(batch["input_ids"][1], torch.tensor([12, 0]))
    torch.testing.assert_close(batch["attention_mask"][1], torch.tensor([1, 0]))


def test_complete_sid_rejects_pad() -> None:
    payload = valid_payload()
    payload["candidate_sids"][0][0][1] = 0
    group = CandidateGroup.from_dict(payload)
    with pytest.raises(ValueError, match="PAD cannot appear"):
        collate_groups([group], pad_token_id=0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda data: data["critic_member_path_values"][0][0][0].pop(),
            "two ensemble",
        ),
        (
            lambda data: data["critic_member_path_values"][0][0][0].append(0.3),
            "ensemble width",
        ),
        (
            lambda data: data["critic_member_path_values"][0][0][0].__setitem__(0, float("nan")),
            "must be finite",
        ),
        (lambda data: data["context_attention_mask"].__setitem__(0, 2), "only 0 or 1"),
    ],
)
def test_group_rejects_malformed_values(mutation, message: str) -> None:
    payload = valid_payload()
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        CandidateGroup.from_dict(payload)


def test_jsonl_reports_bad_line(tmp_path) -> None:
    path = tmp_path / "groups.jsonl"
    path.write_text(json.dumps(valid_payload()) + "\n{bad json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_groups(str(path))


def test_legacy_slot_schema_is_rejected() -> None:
    payload = valid_payload()
    payload["candidate_sids"] = [[2, 3], [4, 5]]
    with pytest.raises(ValueError, match="legacy slot-conditioned"):
        CandidateGroup.from_dict(payload)


def test_page_wise_schema_requires_exporter() -> None:
    payload = {
        "schema_version": "s2tpo_complete_page_v1",
        "target_tokens": [[2, 3], [4, 5]],
    }
    with pytest.raises(ValueError, match="export_complete_page_groups"):
        CandidateGroup.from_dict(payload)


def test_duplicate_complete_pages_are_rejected() -> None:
    payload = valid_payload()
    payload["candidate_sids"][1] = payload["candidate_sids"][0]
    with pytest.raises(ValueError, match="must be distinct"):
        CandidateGroup.from_dict(payload)


def test_item_paths_share_complete_page_endpoint() -> None:
    payload = valid_payload()
    payload["critic_member_path_values"][0][1][-1][0] += 0.1
    with pytest.raises(ValueError, match="complete-page endpoint"):
        CandidateGroup.from_dict(payload)
