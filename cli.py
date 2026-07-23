"""Command-line smoke test for the vehicle Re-ID pipeline.

Runs the full pipeline (detect + embed + match) over two folders and prints the
matched A->B pairs. Useful to verify the model weights download and the whole
stack works without opening the GUI.

Example:
    python cli.py --dir-a samples/A --dir-b samples/B --threshold 0.6
"""
from __future__ import annotations

import argparse
import sys
import os

# Make src/ importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import config


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vehicle Re-ID A/B matcher (CLI).")
    p.add_argument("--dir-a", required=True, help="Folder of point-A frames")
    p.add_argument("--dir-b", required=True, help="Folder of point-B frames")
    p.add_argument(
        "--threshold",
        type=float,
        default=config.SIMILARITY_THRESHOLD,
        help="Cosine-similarity threshold (default: %(default)s)",
    )
    p.add_argument(
        "--min-travel",
        type=float,
        default=config.MIN_TRAVEL_SECONDS,
        help="Minimum travel time A->B in seconds (default: %(default)s)",
    )
    p.add_argument(
        "--max-travel",
        type=float,
        default=config.MAX_TRAVEL_SECONDS,
        help="Maximum travel time A->B in seconds (default: %(default)s)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=config.TOP_K,
        help="Candidates per A vehicle (default: %(default)s)",
    )
    p.add_argument(
        "--one-to-one",
        action="store_true",
        help="Use Hungarian one-to-one assignment instead of top-k ranking",
    )
    p.add_argument(
        "--regex",
        default=config.TIMESTAMP_REGEX,
        help="Timestamp regex for filenames",
    )
    p.add_argument(
        "--format",
        dest="fmt",
        default=config.TIMESTAMP_FORMAT,
        help="strptime format for the timestamp",
    )
    return p


def _progress(done: int, total: int, message: str) -> None:
    print(f"  ({done}/{total}) {message}")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Import heavy modules only after args parse (so --help stays fast).
    from mash_reid.pipeline import Pipeline, stack_embeddings, record_timestamps
    from mash_reid import matcher

    pipeline = Pipeline()

    print(f"Processing point A: {args.dir_a}")
    records_a = pipeline.process_folder(args.dir_a, "A", args.regex, args.fmt, _progress)
    print(f"Processing point B: {args.dir_b}")
    records_b = pipeline.process_folder(args.dir_b, "B", args.regex, args.fmt, _progress)

    print(f"\nDetected {len(records_a)} vehicles at A, {len(records_b)} at B.")
    if not records_a or not records_b:
        print("Nothing to match.")
        return 0

    emb_a = stack_embeddings(records_a)
    emb_b = stack_embeddings(records_b)
    times_a = record_timestamps(records_a)
    times_b = record_timestamps(records_b)

    print("\n=== Matches (A -> B) ===")
    if args.one_to_one:
        matches = matcher.match_one_to_one(
            emb_a, emb_b, times_a, times_b,
            similarity_threshold=args.threshold,
            min_travel_seconds=args.min_travel,
            max_travel_seconds=args.max_travel,
        )
        if not matches:
            print("(no matches above threshold within the time window)")
        for m in matches:
            ra, rb = records_a[m.a_index], records_b[m.b_index]
            print(
                f"A[{m.a_index}] {ra.frame_name} ({ra.class_name}) "
                f"-> B[{m.b_index}] {rb.frame_name} ({rb.class_name})  "
                f"sim={m.similarity:.3f}  dt={m.delta_seconds:.0f}s"
            )
    else:
        result = matcher.match(
            emb_a, emb_b, times_a, times_b,
            similarity_threshold=args.threshold,
            min_travel_seconds=args.min_travel,
            max_travel_seconds=args.max_travel,
            top_k=args.top_k,
        )
        any_match = False
        for a_idx, cands in result.items():
            if not cands:
                continue
            any_match = True
            ra = records_a[a_idx]
            print(f"\nA[{a_idx}] {ra.frame_name} ({ra.class_name}):")
            for c in cands:
                rb = records_b[c.b_index]
                print(
                    f"    -> B[{c.b_index}] {rb.frame_name} ({rb.class_name})  "
                    f"sim={c.similarity:.3f}  dt={c.delta_seconds:.0f}s"
                )
        if not any_match:
            print("(no matches above threshold within the time window)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
