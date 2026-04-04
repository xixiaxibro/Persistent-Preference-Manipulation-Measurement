"""
Assemble the final 5-label multi-label training dataset.

Merges the outputs of the two previous steps:
    weak_labeled.jsonl + llm_labeled.jsonl + llm_all_zero.jsonl

Produces:
    - train.jsonl / val.jsonl / test.jsonl  (stratified split)
    - dataset_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

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

def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> int:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _print(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """
    Produce a uniform training record from an enriched + classified row.

    Output schema:
      text:           str      – the prompt text
      labels:         list[str] – list of class label names (may be empty)
      source_url:     str
      target_url:     str
      target_platform: str
      dataset_role:   str      – provenance label for this sample
      dataset_source: str      – source of the labels or unlabeled sampling
    """
    text = row.get("primary_prompt_text", "")
    if not isinstance(text, str):
        text = ""

    labels = row.get("prompt_labels", [])
    if "llm_assigned_labels" in row:
        labels = row.get("llm_assigned_labels", [])

    valid = set(CLASS_LABELS)
    if not isinstance(labels, list):
        labels = []
    labels = sorted(
        {
            label.upper()
            for label in labels
            if isinstance(label, str) and label.upper() in valid
        }
    )

    return {
        "text": text,
        "labels": labels,
        "source_url": row.get("source_url", ""),
        "target_url": row.get("target_url", ""),
        "target_platform": row.get("target_platform", ""),
        "dataset_role": row.get("dataset_role", ""),
        "dataset_source": row.get("dataset_source", ""),
    }


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def _stratified_split(
    rows: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split rows into train/val/test preserving approximate label distribution.

    Uses the label-combination as the stratification key.
    """
    strata: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = ";".join(row.get("labels", [])) or "NONE"
        strata.setdefault(key, []).append(row)

    train, val, test = [], [], []
    for key, group in strata.items():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio)) if n > 2 else 0
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def assemble(
    labeled_paths: list[Path],
    all_zero_paths: list[Path],
    output_dir: Path,
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)

    # Load and normalize.
    all_rows: list[dict[str, Any]] = []
    for path in labeled_paths:
        for row in _iter_rows(path):
            all_rows.append(_normalize_row(row))
    for path in all_zero_paths:
        for row in _iter_rows(path):
            all_rows.append(_normalize_row(row))

    _print(f"Total rows loaded: {len(all_rows):,}")

    # Deduplicate by text.
    seen_texts: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    for row in all_rows:
        text = row["text"].strip().lower()
        if text and text not in seen_texts:
            seen_texts.add(text)
            unique_rows.append(row)
        elif not text:
            # Keep empty-text negatives (they carry structural signal).
            unique_rows.append(row)

    _print(f"After dedup: {len(unique_rows):,}")

    # Split.
    train, val, test = _stratified_split(unique_rows, train_ratio, val_ratio, rng)

    # Write.
    output_dir.mkdir(parents=True, exist_ok=True)
    n_train = _write_jsonl(train, output_dir / "train.jsonl")
    n_val = _write_jsonl(val, output_dir / "val.jsonl")
    n_test = _write_jsonl(test, output_dir / "test.jsonl")

    # Stats.
    label_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    for row in unique_rows:
        for label in row["labels"]:
            label_counts[label] += 1
        role_counts[row.get("dataset_role", "unknown")] += 1

    stats = {
        "total_unique_rows": len(unique_rows),
        "train": n_train,
        "val": n_val,
        "test": n_test,
        "label_distribution": dict(label_counts.most_common()),
        "role_distribution": dict(role_counts.most_common()),
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
    }

    _print(f"\nTrain: {n_train:,}  Val: {n_val:,}  Test: {n_test:,}")
    _print(f"Labels: {dict(label_counts.most_common())}")
    _print(f"Roles:  {dict(role_counts.most_common())}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble final 5-label multi-label training dataset from weakly-labeled, LLM-labeled, and all-zero pools.",
    )
    parser.add_argument(
        "--labeled",
        "--positives",
        nargs="+",
        required=True,
        dest="labeled",
        help="Labeled JSONL files (weak_labeled.jsonl, llm_labeled.jsonl).",
    )
    parser.add_argument(
        "--all-zero",
        "--negatives",
        nargs="+",
        required=True,
        dest="all_zero",
        help="All-zero JSONL files (llm_all_zero.jsonl).",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default: 0.8).")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio (default: 0.1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    stats = assemble(
        labeled_paths=[Path(p) for p in args.labeled],
        all_zero_paths=[Path(p) for p in args.all_zero],
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    stats_path = output_dir / "dataset_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())