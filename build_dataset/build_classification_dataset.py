"""
Dataset construction for 5-label multi-label prompt classification.

Reads the Stage 2 classified JSONL and produces two pools:

    1. **Weakly-labeled pool** – stratified sample from rows with one or more
         keyword-assigned labels among the 5 target labels. Each row retains its
         multi-label annotation.
    2. **Unlabeled candidate pool** – length-matched sample from rows whose
         keyword-assigned label set is empty. These rows are exported for LLM
         relabeling and may remain all-zero examples after that step.

The LLM relabeling step (Script 2) will then:
    - Assign one or more of the 5 labels to some rows
    - Leave the rest as all-zero samples with no labels

Output: two JSONL files (weak_labeled.jsonl, unlabeled_for_relabel.jsonl)
plus a manifest JSON with sampling statistics.

Sampling strategy
-----------------
* Weakly-labeled pool: for each of the 5 labels, sample up to N rows that
    carry that label. A single row can appear under multiple labels.
    Deduplication ensures each row appears once in the output with all its
    labels preserved.
* Unlabeled pool: sample M rows from the zero-label pool, matching the
    text-length distribution of the weakly-labeled pool to avoid length-based
    confounds.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_LABELS: tuple[str, ...] = (
    "PERSISTENCE",
    "AUTHORITY",
    "RECOMMENDATION",
    "CITATION",
    "SUMMARY",
)

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

_IO_BUFFER_SIZE = 8 * 1024 * 1024


def _open_read(path: Path) -> BinaryIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb", buffering=_IO_BUFFER_SIZE)


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with _open_read(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _print(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _normalize_labels(raw_labels: Any) -> list[str]:
    if not isinstance(raw_labels, list):
        return []
    valid = set(CLASS_LABELS)
    return sorted(
        {
            label.upper()
            for label in raw_labels
            if isinstance(label, str) and label.upper() in valid
        }
    )


# ---------------------------------------------------------------------------
# Length-matched negative sampling
# ---------------------------------------------------------------------------

def _compute_length_buckets(
    texts: list[str],
    num_buckets: int = 10,
) -> list[tuple[int, int]]:
    """
    Compute equal-frequency length buckets from a list of texts.
    Returns a list of (min_len, max_len) inclusive bounds.
    """
    lengths = sorted(len(t) for t in texts)
    if not lengths:
        return []
    bucket_size = max(1, math.ceil(len(lengths) / num_buckets))
    buckets: list[tuple[int, int]] = []
    for i in range(0, len(lengths), bucket_size):
        chunk = lengths[i : i + bucket_size]
        buckets.append((chunk[0], chunk[-1]))
    return buckets


def _sample_length_matched(
    candidates: list[dict[str, Any]],
    target_distribution: list[tuple[int, int]],
    target_counts: list[int],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Sample from *candidates* to match a target length distribution.

    *target_distribution* is a list of (min_len, max_len) buckets.
    *target_counts* is the number of samples desired per bucket.
    """
    # Index candidates into buckets.
    bucket_pools: list[list[dict[str, Any]]] = [[] for _ in target_distribution]
    for row in candidates:
        text = row.get("primary_prompt_text", "")
        tlen = len(text) if isinstance(text, str) else 0
        for i, (lo, hi) in enumerate(target_distribution):
            if lo <= tlen <= hi:
                bucket_pools[i].append(row)
                break  # assign to first matching bucket

    sampled: list[dict[str, Any]] = []
    for i, desired in enumerate(target_counts):
        pool = bucket_pools[i]
        if len(pool) <= desired:
            sampled.extend(pool)
        else:
            sampled.extend(rng.sample(pool, desired))
    return sampled


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    weak_labeled_per_label: int,
    unlabeled_total: int,
    length_buckets: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)

    # ---- Pass 1: partition rows into weak-labeled and unlabeled pools ----
    _print("Pass 1: reading and partitioning rows...")

    weak_labeled_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unlabeled_candidates: list[dict[str, Any]] = []
    total_rows = 0
    label_counts_all: Counter[str] = Counter()

    for row in _iter_rows(input_path):
        total_rows += 1
        labels = _normalize_labels(row.get("prompt_labels", []))

        if labels:
            for label in labels:
                weak_labeled_by_label[label].append(row)
                label_counts_all[label] += 1
        else:
            unlabeled_candidates.append(row)

    _print(f"  Total rows: {total_rows:,}")
    _print(f"  Rows with one or more labels: {total_rows - len(unlabeled_candidates):,}")
    _print(f"  Unlabeled candidates: {len(unlabeled_candidates):,}")
    for label in CLASS_LABELS:
        _print(f"  {label}: {label_counts_all[label]:,} rows")

    # ---- Step 2: stratified weak-label sampling ----
    _print(f"\nStep 2: sampling up to {weak_labeled_per_label:,} per label...")

    sampled_labeled_ids: set[str] = set()
    sampled_weak_labeled: list[dict[str, Any]] = []
    per_label_sampled: dict[str, int] = {}

    for label in CLASS_LABELS:
        pool = weak_labeled_by_label.get(label, [])
        n = min(weak_labeled_per_label, len(pool))
        selected = rng.sample(pool, n) if n < len(pool) else pool

        added = 0
        for row in selected:
            row_id = f"{row.get('source_url', '')}|{row.get('target_url', '')}"
            if row_id not in sampled_labeled_ids:
                sampled_labeled_ids.add(row_id)
                entry = dict(row)
                entry["prompt_labels"] = _normalize_labels(row.get("prompt_labels", []))
                entry["classification"] = ";".join(entry["prompt_labels"])
                entry["dataset_role"] = "weak_labeled"
                entry["dataset_source"] = "keyword_weak_label"
                sampled_weak_labeled.append(entry)
            added += 1
        per_label_sampled[label] = added

    rng.shuffle(sampled_weak_labeled)

    _print(f"  Deduplicated weak-labeled samples: {len(sampled_weak_labeled):,}")
    for label in CLASS_LABELS:
        _print(f"  {label}: {per_label_sampled.get(label, 0):,} sampled")

    # ---- Step 3: length-matched unlabeled sampling ----
    _print(f"\nStep 3: sampling {unlabeled_total:,} length-matched unlabeled rows...")

    weak_labeled_texts = [
        row.get("primary_prompt_text", "")
        for row in sampled_weak_labeled
        if isinstance(row.get("primary_prompt_text"), str)
    ]

    buckets = _compute_length_buckets(weak_labeled_texts, num_buckets=length_buckets)

    # Compute how many weak-labeled rows fall in each bucket.
    bucket_labeled_counts: list[int] = [0] * len(buckets)
    for text in weak_labeled_texts:
        tlen = len(text)
        for i, (lo, hi) in enumerate(buckets):
            if lo <= tlen <= hi:
                bucket_labeled_counts[i] += 1
                break

    sampled_unlabeled: list[dict[str, Any]] = []
    target_counts: list[int] = []

    # Scale bucket counts to sum to unlabeled_total.
    total_labeled = sum(bucket_labeled_counts)
    if total_labeled > 0 and buckets:
        target_counts = [
            max(1, round(c / total_labeled * unlabeled_total))
            for c in bucket_labeled_counts
        ]
        sampled_unlabeled = _sample_length_matched(
            unlabeled_candidates, buckets, target_counts, rng
        )
    else:
        n = min(unlabeled_total, len(unlabeled_candidates))
        sampled_unlabeled = rng.sample(unlabeled_candidates, n) if n < len(unlabeled_candidates) else list(unlabeled_candidates)

    # Add dataset metadata.
    for row in sampled_unlabeled:
        row["dataset_role"] = "unlabeled_candidate"
        row["dataset_source"] = "length_matched_unlabeled"
        row["prompt_labels"] = []
        row["classification"] = ""

    rng.shuffle(sampled_unlabeled)

    _print(f"  Unlabeled rows sampled: {len(sampled_unlabeled):,}")
    _print(f"  Length buckets used: {len(buckets)}")
    for i, (lo, hi) in enumerate(buckets):
        _print(
            f"    bucket {i}: len [{lo}, {hi}]  "
            f"labeled={bucket_labeled_counts[i]}  "
            f"target={target_counts[i]}  "
        )

    # ---- Step 4: write outputs ----
    weak_labeled_path = output_dir / "weak_labeled.jsonl"
    unlabeled_path = output_dir / "unlabeled_for_relabel.jsonl"

    n_labeled = _write_jsonl(sampled_weak_labeled, weak_labeled_path)
    n_unlabeled = _write_jsonl(sampled_unlabeled, unlabeled_path)

    _print(f"\nWritten: {weak_labeled_path} ({n_labeled:,} rows)")
    _print(f"Written: {unlabeled_path} ({n_unlabeled:,} rows)")

    # ---- Manifest ----
    manifest = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "seed": seed,
        "total_input_rows": total_rows,
        "weak_labeled_per_label_requested": weak_labeled_per_label,
        "unlabeled_total_requested": unlabeled_total,
        "weak_labeled_written": n_labeled,
        "unlabeled_written": n_unlabeled,
        "per_label_sampled": per_label_sampled,
        "label_counts_in_data": dict(label_counts_all.most_common()),
        "length_buckets": [
            {"min": lo, "max": hi, "labeled_count": c, "unlabeled_target": t}
            for (lo, hi), c, t in zip(buckets, bucket_labeled_counts, target_counts)
        ],
        "weak_labeled_file": str(weak_labeled_path),
        "unlabeled_file": str(unlabeled_path),
    }
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build multi-label classification dataset from classified prompt links. "
            "Produces a weakly-labeled pool and a length-matched unlabeled "
            "candidate pool for LLM relabeling."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input classified JSONL from classify_prompt_links.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for weak_labeled.jsonl, unlabeled_for_relabel.jsonl, and manifest.",
    )
    parser.add_argument(
        "--weak-labeled-per-label",
        "--positive-per-label",
        dest="weak_labeled_per_label",
        type=int,
        default=2000,
        help="Max samples per class label in the weak-labeled pool (default: 2000).",
    )
    parser.add_argument(
        "--unlabeled-total",
        "--negative-total",
        dest="unlabeled_total",
        type=int,
        default=5000,
        help="Total unlabeled candidates to sample for LLM relabeling (default: 5000).",
    )
    parser.add_argument(
        "--length-buckets",
        type=int,
        default=10,
        help="Number of length buckets for negative sampling (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_dataset(
        input_path,
        output_dir,
        weak_labeled_per_label=args.weak_labeled_per_label,
        unlabeled_total=args.unlabeled_total,
        length_buckets=args.length_buckets,
        seed=args.seed,
    )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    _print(f"\nManifest: {manifest_path}")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())