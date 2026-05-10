#!/usr/bin/env python3
"""
Export a prompt dataset by directly random-sampling Stage 01 filtered rows.

The exporter reads the platform-filtered JSONL from Stage 01, derives a prompt
text and target platform for each row, and performs deterministic reservoir
sampling without additional filtering or label assignment.

Output schema per row:
    - text
    - labels
    - source_url
    - target_url
    - target_platform

`labels` is always written as an empty list in this workflow.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, TextIO

from platform_signatures import (
    extract_prompt_parameters,
    flatten_prompt_parameters,
    match_platform_with_exclusion,
)


_IO_BUFFER_SIZE = 8 * 1024 * 1024
_PROGRESS_ROW_INTERVAL = 200_000


def _print_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _open_read(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(
        "r",
        encoding="utf-8",
        buffering=_IO_BUFFER_SIZE,
        errors="replace",
    )


def _open_write(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8", buffering=_IO_BUFFER_SIZE)


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with _open_read(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def _coerce_string(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _normalized_export_row(row: dict[str, Any]) -> dict[str, Any] | None:
    target_url = _coerce_string(row.get("target_url"))
    if not target_url:
        return None

    signature, excluded = match_platform_with_exclusion(target_url)
    if excluded or signature is None:
        return None

    prompt_parameters = row.get("prompt_parameters")
    if not isinstance(prompt_parameters, dict):
        prompt_parameters = extract_prompt_parameters(target_url, signature)

    decoded_candidates = flatten_prompt_parameters(prompt_parameters)
    text = ""
    if decoded_candidates:
        first_value = decoded_candidates[0]
        if isinstance(first_value, str):
            text = first_value.strip()

    return {
        "text": text,
        "labels": [],
        "source_url": _coerce_string(row.get("source_url")),
        "target_url": target_url,
        "target_platform": signature.name,
    }


def _reservoir_sample(
    input_path: Path,
    *,
    total_samples: int,
    seed: int,
    progress_interval: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    counters: Counter[str] = Counter()
    platform_counts_seen: Counter[str] = Counter()
    platform_counts_sampled: Counter[str] = Counter()
    reservoir: list[tuple[int, dict[str, Any]]] = []

    start_time = time.monotonic()
    last_report = start_time

    for row in _iter_rows(input_path):
        counters["rows_seen"] += 1

        normalized = _normalized_export_row(row)
        if normalized is None:
            counters["rows_skipped_unmatched"] += 1
            continue

        counters["rows_eligible"] += 1
        platform_counts_seen[normalized["target_platform"]] += 1

        eligible_index = counters["rows_eligible"]
        if len(reservoir) < total_samples:
            reservoir.append((eligible_index, normalized))
        else:
            replacement_index = rng.randrange(eligible_index)
            if replacement_index < total_samples:
                reservoir[replacement_index] = (eligible_index, normalized)

        if counters["rows_seen"] % _PROGRESS_ROW_INTERVAL == 0:
            now = time.monotonic()
            if now - last_report >= progress_interval:
                elapsed = now - start_time
                rate = counters["rows_seen"] / elapsed if elapsed > 0 else 0.0
                _print_progress(
                    f"[{elapsed:>5.0f}s] rows: {counters['rows_seen']:>10,}  "
                    f"eligible: {counters['rows_eligible']:>10,}  "
                    f"reservoir: {len(reservoir):>7,}  "
                    f"rate: {rate:,.0f} rows/s"
                )
                last_report = now

    reservoir.sort(key=lambda item: item[0])
    sampled_rows = [row for _, row in reservoir]
    for row in sampled_rows:
        platform_counts_sampled[row["target_platform"]] += 1

    return sampled_rows, {
        "counts": {
            "rows_seen": counters["rows_seen"],
            "rows_eligible": counters["rows_eligible"],
            "rows_skipped_unmatched": counters["rows_skipped_unmatched"],
        },
        "platform_counts_seen": dict(platform_counts_seen.most_common()),
        "platform_counts_sampled": dict(platform_counts_sampled.most_common()),
    }


def _write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> int:
    with _open_write(output_path) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def export_prompt_dataset(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    total_samples: int,
    seed: int,
    progress_interval: float,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")
    if manifest_path.exists():
        raise FileExistsError(f"Manifest file already exists: {manifest_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    rows, sampling_stats = _reservoir_sample(
        input_path,
        total_samples=total_samples,
        seed=seed,
        progress_interval=progress_interval,
    )
    written = _write_jsonl(rows, output_path)

    elapsed = time.monotonic() - start_time
    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "manifest": str(manifest_path),
        "requested_total": total_samples,
        "written_total": written,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 2),
        "schema": [
            "text",
            "labels",
            "source_url",
            "target_url",
            "target_platform",
        ],
        "sampling_mode": "direct_random_sample_from_stage01",
        "counts": sampling_stats["counts"],
        "sampling": {
            "method": "reservoir",
            "platform_counts_seen": sampling_stats["platform_counts_seen"],
            "platform_counts_sampled": sampling_stats["platform_counts_sampled"],
        },
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a prompt dataset by directly random-sampling Stage 01 "
            "platform-filtered prompt-link JSONL."
        ),
    )
    parser.add_argument("--input", required=True, help="Input Stage 01 prompt_links.jsonl or .jsonl.gz.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional manifest JSON path. Default: <output_dir>/export_manifest.json.",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=20_000,
        help="Requested sample count (default: 20000).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="Seconds between progress reports (default: 10).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_path.parent / "export_manifest.json"

    _print_progress(f"Input:           {input_path}")
    _print_progress(f"Output:          {output_path}")
    _print_progress(f"Manifest:        {manifest_path}")
    _print_progress(f"Requested rows:  {args.total_samples:,}")
    _print_progress(f"Sampling seed:   {args.seed}")
    _print_progress("")

    manifest = export_prompt_dataset(
        input_path,
        output_path,
        manifest_path,
        total_samples=args.total_samples,
        seed=args.seed,
        progress_interval=args.progress_interval,
    )

    _print_progress("")
    _print_progress(
        f"Done. {manifest['counts']['rows_seen']:,} rows seen, "
        f"{manifest['counts']['rows_eligible']:,} eligible rows, "
        f"{manifest['written_total']:,} rows written."
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())