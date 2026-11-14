#!/usr/bin/env python3
"""Validate the public S2T-PO candidate-group JSONL contract."""

import argparse

from s2tpo.groups import load_groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups")
    args = parser.parse_args()
    groups = load_groups(args.groups)
    candidate_counts = sorted({len(group.candidate_sids) for group in groups})
    page_sizes = sorted({len(group.candidate_sids[0]) for group in groups})
    sid_depths = sorted({len(group.candidate_sids[0][0]) for group in groups})
    print(
        f"valid groups={len(groups)} "
        f"candidate_counts={candidate_counts} page_sizes={page_sizes} "
        f"sid_depths={sid_depths}"
    )


if __name__ == "__main__":
    main()
