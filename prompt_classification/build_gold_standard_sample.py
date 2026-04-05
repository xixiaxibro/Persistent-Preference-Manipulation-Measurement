"""
Build a stratified sample for human annotation (gold standard).

Draws a stratified sample from each of the 5 labels plus an unlabeled stratum,
producing a JSONL file ready for dual-annotator labeling.

Strategy (from Section 4.1 of the design doc):
    - 200 rows per label drawn from Tier 1 positives.
    - 200 rows with no Tier 1 labels (to measure false negatives).
    - Target: ~1200 rows (with deduplication, will be ~1000–1200).
    - This set is used *only* for evaluation, never for training.

Output:
    - gold_standard_sample.jsonl   (rows to annotate)
    - gold_sample_manifest.json    (sampling statistics)
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_LABELS: tuple[str, ...] = (
    "PERSIST",
    "AUTHORITY",
    "RECOMMEND",
    "CITE",
    "SUMMARIZE",
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
# Sampling
# ---------------------------------------------------------------------------

def build_gold_sample(
    input_path: Path,
    output_dir: Path,
    *,
    per_label: int = 200,
    unlabeled_count: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Draw a stratified sample for human gold-standard annotation.

    Returns a manifest dictionary with sampling statistics.
    """
    rng = random.Random(seed)

    # ---- Pass 1: partition rows by label ----
    _print("Pass 1: reading and partitioning rows...")

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unlabeled_pool: list[dict[str, Any]] = []
    total_rows = 0
    label_counts: Counter[str] = Counter()

    for row in _iter_rows(input_path):
        total_rows += 1
        labels = _normalize_labels(row.get("prompt_labels", []))
        text = row.get("primary_prompt_text", "")

        # Skip rows with empty text.
        if not isinstance(text, str) or len(text.strip()) < 5:
            continue

        if labels:
            for label in labels:
                by_label[label].append(row)
                label_counts[label] += 1
        else:
            unlabeled_pool.append(row)

    _print(f"  Total rows: {total_rows:,}")
    _print(f"  Unlabeled pool: {len(unlabeled_pool):,}")
    for label in CLASS_LABELS:
        _print(f"  {label}: {label_counts[label]:,} rows")

    # ---- Step 2: stratified sampling ----
    _print(f"\nStep 2: sampling {per_label} per label + {unlabeled_count} unlabeled...")

    sampled_ids: set[str] = set()
    sampled_rows: list[dict[str, Any]] = []
    per_label_sampled: dict[str, int] = {}

    for label in CLASS_LABELS:
        pool = by_label.get(label, [])
        n = min(per_label, len(pool))
        selected = rng.sample(pool, n) if n < len(pool) else list(pool)

        added = 0
        for row in selected:
            row_id = f"{row.get('source_url', '')}|{row.get('target_url', '')}"
            if row_id not in sampled_ids:
                sampled_ids.add(row_id)
                entry = dict(row)
                entry["gold_tier1_labels"] = _normalize_labels(
                    row.get("prompt_labels", [])
                )
                entry["gold_stratum"] = label
                entry["annotator_1_labels"] = []
                entry["annotator_2_labels"] = []
                entry["resolved_labels"] = []
                sampled_rows.append(entry)
                added += 1
        per_label_sampled[label] = added

    # Unlabeled stratum.
    n_unlabeled = min(unlabeled_count, len(unlabeled_pool))
    unlabeled_selected = (
        rng.sample(unlabeled_pool, n_unlabeled)
        if n_unlabeled < len(unlabeled_pool)
        else list(unlabeled_pool)
    )
    unlabeled_added = 0
    for row in unlabeled_selected:
        row_id = f"{row.get('source_url', '')}|{row.get('target_url', '')}"
        if row_id not in sampled_ids:
            sampled_ids.add(row_id)
            entry = dict(row)
            entry["gold_tier1_labels"] = []
            entry["gold_stratum"] = "UNLABELED"
            entry["annotator_1_labels"] = []
            entry["annotator_2_labels"] = []
            entry["resolved_labels"] = []
            sampled_rows.append(entry)
            unlabeled_added += 1

    rng.shuffle(sampled_rows)

    _print(f"  Total deduplicated samples: {len(sampled_rows):,}")
    for label in CLASS_LABELS:
        _print(f"  {label}: {per_label_sampled.get(label, 0):,}")
    _print(f"  UNLABELED: {unlabeled_added:,}")

    # ---- Write output ----
    output_path = output_dir / "gold_standard_sample.jsonl"
    n_written = _write_jsonl(sampled_rows, output_path)

    _print(f"\nWritten: {output_path} ({n_written:,} rows)")

    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "seed": seed,
        "per_label_requested": per_label,
        "unlabeled_requested": unlabeled_count,
        "total_input_rows": total_rows,
        "total_sampled": n_written,
        "per_label_sampled": per_label_sampled,
        "unlabeled_sampled": unlabeled_added,
        "label_counts_in_input": dict(label_counts.most_common()),
    }
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a stratified gold-standard sample for human annotation. "
            "Draws 200 rows per label + 200 unlabeled rows."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input classified JSONL (output of classify_prompt_links.py with --include-benign).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for gold_standard_sample.jsonl and manifest.",
    )
    parser.add_argument(
        "--per-label",
        type=int,
        default=200,
        help="Rows per label stratum (default: 200).",
    )
    parser.add_argument(
        "--unlabeled-count",
        type=int,
        default=200,
        help="Rows from unlabeled stratum (default: 200).",
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

    manifest = build_gold_sample(
        input_path,
        output_dir,
        per_label=args.per_label,
        unlabeled_count=args.unlabeled_count,
        seed=args.seed,
    )

    manifest_path = output_dir / "gold_sample_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    _print(f"Manifest: {manifest_path}")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
