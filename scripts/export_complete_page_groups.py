#!/usr/bin/env python3
"""Convert page-wise S2T-PO records to the canonical complete-page schema."""

import argparse

from s2tpo.export import group_page_records, iter_jsonl, write_groups_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Page-wise builder JSONL")
    parser.add_argument("--output", required=True, help="Canonical group JSONL")
    args = parser.parse_args()

    groups = group_page_records(iter_jsonl(args.input))
    count = write_groups_jsonl(groups, args.output)
    print(f"exported groups={count} output={args.output}")


if __name__ == "__main__":
    main()
