import csv
import pickle

from s2tpo.agent4rec import (
    build_next_sid_samples,
    discounted_returns,
    load_behavior_sessions,
    load_cf_sequences,
    load_sid_mapping,
    page_feedback,
)


def test_agent4rec_data_and_feedback_contract(tmp_path) -> None:
    sequence_path = tmp_path / "train.txt"
    sequence_path.write_text("0 0 1 2\n1 1 2 3\n", encoding="utf-8")
    sequences = load_cf_sequences(str(sequence_path))
    assert sequences[0] == [0, 1, 2]

    mapping_path = tmp_path / "sid.csv"
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "sid_1", "sid_2"])
        for item in range(8):
            writer.writerow([item, item // 4, item % 4])
    item_to_sid, valid_sids = load_sid_mapping(str(mapping_path))
    assert item_to_sid[0] == [1, 1]
    assert len(valid_sids) == 8

    run_dir = tmp_path / "run"
    behavior_dir = run_dir / "behavior"
    behavior_dir.mkdir(parents=True)
    page = {
        "page": 1,
        "recommended_id": [0, 1, 2, 3, 4, 5],
        "align_id": [0, 1],
        "watch_id": [0],
        "rating_id": [0],
        "rating": [5],
        "exit_flag": False,
    }
    with (behavior_dir / "0.pkl").open("wb") as handle:
        pickle.dump({1: page}, handle)
    sessions = load_behavior_sessions([str(run_dir)])
    assert sessions[0]["user_id"] == 0
    evidence, utility = page_feedback(page)
    assert evidence == [3.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert utility > 1.0
    assert discounted_returns([1.0, 2.0], 0.9) == [2.8, 2.0]

    samples = build_next_sid_samples(
        sequences, item_to_sid, max_history=2, max_samples=0
    )
    assert samples[0] == ([1, 1], [1, 2])
