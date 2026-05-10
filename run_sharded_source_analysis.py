#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import heapq
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

from source_url_analysis_common import (
    TRANCO_BUCKET_ORDER,
    counter_to_sorted_rows,
    ensure_directory,
    extract_root_domain,
    iso_now_epoch,
    make_domain_extractor,
    ordered_bucket_rows,
    write_csv,
    write_json,
)


try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


def _normalize_string(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _as_int(value: object) -> int:
    if value in (None, ""):
        return 0
    return int(value)


def _shard_key(source_url: str, source_domain: str, extractor, line_number: int) -> str:
    root_domain = extract_root_domain(source_domain or source_url, extractor)
    if root_domain:
        return root_domain
    if source_domain:
        return source_domain.lower()
    if source_url:
        return source_url
    return f"missing:{line_number}"


def _stable_bucket(key: str, shard_count: int) -> int:
    digest = hashlib.md5(key.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest[:8], 16) % shard_count


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def _merge_sorted_rows(paths: list[Path], sort_key):
    handles: list[Any] = []
    readers: list[csv.DictReader] = []
    heap: list[tuple[tuple[Any, ...], int, dict[str, str]]] = []

    try:
        for path in paths:
            handle = path.open("r", encoding="utf-8", newline="")
            reader = csv.DictReader(handle)
            index = len(readers)
            handles.append(handle)
            readers.append(reader)
            try:
                row = next(reader)
            except StopIteration:
                continue
            heapq.heappush(heap, (sort_key(row), index, row))

        while heap:
            _key, index, row = heapq.heappop(heap)
            yield row
            try:
                next_row = next(readers[index])
            except StopIteration:
                continue
            heapq.heappush(heap, (sort_key(next_row), index, next_row))
    finally:
        for handle in handles:
            handle.close()


def _source_url_sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (-_as_int(row.get("rows")), -_as_int(row.get("unique_target_platforms")), row.get("source_url", ""))


def _source_domain_sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (-_as_int(row.get("rows")), -_as_int(row.get("unique_source_urls")), row.get("source_domain", ""))


def _root_domain_sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (-_as_int(row.get("rows")), -_as_int(row.get("unique_source_urls")), row.get("root_domain", ""))


def _path_pattern_sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (-_as_int(row.get("rows")), row.get("root_domain", ""), row.get("path_template", ""))


def _source_platform_reuse_sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (-_as_int(row.get("unique_target_platforms")), -_as_int(row.get("rows")), row.get("source_url", ""))


def _top_tranco_sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (-_as_int(row.get("rows")), _as_int(row.get("tranco_rank")), row.get("root_domain", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source URL analysis in root-domain shards and merge the final outputs."
    )
    parser.add_argument("--input", required=True, help="Input classified_prompt_links.jsonl or .jsonl.gz")
    parser.add_argument("--output-dir", required=True, help="Final output directory for merged analysis artifacts")
    parser.add_argument("--crawl-name", default="", help="Override crawl name in outputs")
    parser.add_argument("--top-n", type=int, default=200, help="Rows to keep in reviewer-facing top tables")
    parser.add_argument("--examples-per-group", type=int, default=2, help="Examples retained by the underlying shard analyzer")
    parser.add_argument("--only-nonempty-prompt", action="store_true", help="Analyze only rows with non-empty primary_prompt_text")
    parser.add_argument("--suspicious-only", action="store_true", help="Analyze only rows where is_suspicious is true")
    parser.add_argument("--tranco-cache", default="tranco_top1m.csv", help="Tranco cache path")
    parser.add_argument(
        "--tranco-mode",
        choices=("fixed", "download-if-missing"),
        default="fixed",
        help="Use only local Tranco data or download when missing",
    )
    parser.add_argument("--shards", type=int, default=32, help="Number of root-domain shards to use")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary shard inputs and outputs after merge")
    parser.add_argument("--reuse-existing-shards", action="store_true", help="Skip partitioning and shard analysis, and merge the existing shard outputs in _sharded_work")
    return parser.parse_args()


def _partition_input(input_path: Path, shard_input_dir: Path, shard_count: int) -> list[dict[str, Any]]:
    ensure_directory(shard_input_dir)
    extractor = make_domain_extractor()
    shard_paths = [shard_input_dir / f"shard_{index:02d}.jsonl" for index in range(shard_count)]
    handles = [path.open("w", encoding="utf-8") for path in shard_paths]
    shard_rows = [0 for _ in range(shard_count)]

    try:
        with _open_text(input_path) as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"Line {line_number} is not a JSON object.")
                source_url = _normalize_string(payload.get("source_url"))
                source_domain = _normalize_string(payload.get("source_domain"))
                key = _shard_key(source_url, source_domain, extractor, line_number)
                bucket = _stable_bucket(key, shard_count)
                handles[bucket].write(stripped)
                handles[bucket].write("\n")
                shard_rows[bucket] += 1
                if line_number % 200_000 == 0:
                    print(json.dumps({"stage": "partition", "rows_seen": line_number}, ensure_ascii=False), flush=True)
    finally:
        for handle in handles:
            handle.close()

    shard_specs: list[dict[str, Any]] = []
    for index, row_count in enumerate(shard_rows):
        if row_count == 0:
            try:
                shard_paths[index].unlink()
            except FileNotFoundError:
                pass
            continue
        shard_specs.append({"index": index, "input": shard_paths[index], "rows": row_count})
    return shard_specs


def _run_shard_analyzers(args: argparse.Namespace, shard_specs: list[dict[str, Any]], shard_output_dir: Path) -> list[dict[str, Any]]:
    script_path = Path(__file__).with_name("analyze_source_urls.py")
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing analyzer script: {script_path}")

    ensure_directory(shard_output_dir)
    completed: list[dict[str, Any]] = []

    for position, spec in enumerate(shard_specs, start=1):
        shard_dir = shard_output_dir / f"shard_{spec['index']:02d}"
        cmd = [
            sys.executable,
            str(script_path),
            "--input",
            str(spec["input"]),
            "--output-dir",
            str(shard_dir),
            "--top-n",
            str(args.top_n),
            "--examples-per-group",
            str(args.examples_per_group),
            "--tranco-cache",
            str(args.tranco_cache),
            "--tranco-mode",
            args.tranco_mode,
        ]
        if args.crawl_name:
            cmd.extend(["--crawl-name", args.crawl_name])
        if args.only_nonempty_prompt:
            cmd.append("--only-nonempty-prompt")
        if args.suspicious_only:
            cmd.append("--suspicious-only")

        print(
            json.dumps(
                {
                    "stage": "shard_start",
                    "shard": spec["index"],
                    "position": position,
                    "total_shards": len(shard_specs),
                    "rows": spec["rows"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        subprocess.run(cmd, check=True)
        summary_path = shard_dir / "summary.json"
        summary = _read_json(summary_path)
        completed.append({"index": spec["index"], "rows": spec["rows"], "output_dir": shard_dir, "summary": summary})
        print(
            json.dumps(
                {
                    "stage": "shard_done",
                    "shard": spec["index"],
                    "rows_analyzed": summary.get("quality", {}).get("rows_analyzed", 0),
                    "unique_source_urls": summary.get("counts", {}).get("unique_source_urls", 0),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return completed


def _load_existing_shards(work_dir: Path) -> list[dict[str, Any]]:
    output_root = work_dir / "outputs"
    completed: list[dict[str, Any]] = []

    for summary_path in sorted(output_root.glob("shard_*/summary.json")):
        shard_dir = summary_path.parent
        shard_name = shard_dir.name
        try:
            shard_index = int(shard_name.split("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"Unexpected shard directory name: {shard_name}") from exc
        summary = _read_json(summary_path)
        completed.append(
            {
                "index": shard_index,
                "rows": _as_int(summary.get("quality", {}).get("rows_seen")),
                "output_dir": shard_dir,
                "summary": summary,
            }
        )

    if not completed:
        raise RuntimeError(f"No completed shard outputs found under: {output_root}")

    return completed


def _aggregate_distribution(summaries: list[dict[str, Any]], distribution_name: str, key_name: str) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    for summary in summaries:
        for row in summary.get("distributions", {}).get(distribution_name, []):
            key = str(row.get(key_name, ""))
            counts[key] += _as_int(row.get("count"))
    return counts


def _sum_quality_counts(summaries: list[dict[str, Any]]) -> dict[str, int]:
    quality: collections.Counter[str] = collections.Counter()
    for summary in summaries:
        quality.update({key: _as_int(value) for key, value in summary.get("quality", {}).items()})
    return dict(quality)


def _sum_count_block(summaries: list[dict[str, Any]], field_names: list[str]) -> dict[str, int]:
    totals = {field_name: 0 for field_name in field_names}
    for summary in summaries:
        counts = summary.get("counts", {})
        for field_name in field_names:
            totals[field_name] += _as_int(counts.get(field_name))
    return totals


def _aggregate_bucket_summary(output_dirs: list[Path]) -> list[dict[str, Any]]:
    numeric_fields = [
        "rows",
        "rows_with_prompt_text",
        "suspicious_rows",
        "high_rows",
        "medium_rows",
        "low_rows",
        "ioc_rows",
        "unique_root_domains",
        "unique_source_domains",
        "unique_source_urls",
        "unique_target_platforms",
    ]
    bucket_totals: dict[str, dict[str, Any]] = {
        bucket: {field_name: 0 for field_name in numeric_fields} for bucket in TRANCO_BUCKET_ORDER
    }

    for output_dir in output_dirs:
        path = output_dir / "tables" / "tranco_bucket_summary.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                bucket = row.get("tranco_bucket", "")
                if bucket not in bucket_totals:
                    bucket_totals[bucket] = {field_name: 0 for field_name in numeric_fields}
                for field_name in numeric_fields:
                    bucket_totals[bucket][field_name] += _as_int(row.get(field_name))

    rows: list[dict[str, Any]] = []
    for bucket in TRANCO_BUCKET_ORDER:
        payload = {"tranco_bucket": bucket}
        payload.update(bucket_totals.get(bucket, {field_name: 0 for field_name in numeric_fields}))
        rows.append(payload)
    return rows


def _aggregate_bucket_platform_rows(
    output_dirs: list[Path],
    crawl_name: str,
    platform_totals: collections.Counter[str],
    bucket_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bucket_totals = {row["tranco_bucket"]: _as_int(row["rows"]) for row in bucket_rows}
    counts: collections.Counter[tuple[str, str]] = collections.Counter()

    for output_dir in output_dirs:
        path = output_dir / "tables" / "tranco_bucket_by_platform.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                bucket = row.get("tranco_bucket", "")
                platform = row.get("target_platform", "")
                counts[(bucket, platform)] += _as_int(row.get("rows"))

    rows: list[dict[str, Any]] = []
    for bucket in TRANCO_BUCKET_ORDER:
        for platform in sorted(platform_totals):
            count = counts.get((bucket, platform), 0)
            rows.append(
                {
                    "crawl": crawl_name,
                    "tranco_bucket": bucket,
                    "target_platform": platform,
                    "rows": count,
                    "share_within_bucket": round((count / bucket_totals[bucket]), 6) if bucket_totals.get(bucket) else 0.0,
                    "share_within_platform": round((count / platform_totals[platform]), 6) if platform_totals.get(platform) else 0.0,
                }
            )
    return rows


def _apply_bucket_platform_uniques(
    bucket_rows: list[dict[str, Any]], bucket_platform_rows: list[dict[str, Any]]
) -> None:
    platform_counts: collections.Counter[str] = collections.Counter()
    for row in bucket_platform_rows:
        if _as_int(row.get("rows")) > 0:
            platform_counts[str(row.get("tranco_bucket", ""))] += 1

    for row in bucket_rows:
        row["unique_target_platforms"] = int(platform_counts.get(str(row.get("tranco_bucket", "")), 0))


def _aggregate_bucket_label_rows(
    output_dirs: list[Path],
    crawl_name: str,
    label_totals: collections.Counter[str],
    bucket_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bucket_totals = {row["tranco_bucket"]: _as_int(row["rows"]) for row in bucket_rows}
    counts: collections.Counter[tuple[str, str]] = collections.Counter()

    for output_dir in output_dirs:
        path = output_dir / "tables" / "tranco_bucket_by_label.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                bucket = row.get("tranco_bucket", "")
                label = row.get("label", "")
                counts[(bucket, label)] += _as_int(row.get("rows"))

    rows: list[dict[str, Any]] = []
    for bucket in TRANCO_BUCKET_ORDER:
        for label in sorted(label_totals):
            count = counts.get((bucket, label), 0)
            rows.append(
                {
                    "crawl": crawl_name,
                    "tranco_bucket": bucket,
                    "label": label,
                    "rows": count,
                    "share_within_bucket": round((count / bucket_totals[bucket]), 6) if bucket_totals.get(bucket) else 0.0,
                    "share_within_label": round((count / label_totals[label]), 6) if label_totals.get(label) else 0.0,
                }
            )
    return rows


def _merge_simple_csv(paths: list[Path], output_path: Path, sort_key) -> None:
    if not paths:
        return
    fieldnames = _read_fieldnames(paths[0])
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in _merge_sorted_rows(paths, sort_key):
            writer.writerow(row)


def _merge_source_url_tables(output_dirs: list[Path], tables_dir: Path, review_dir: Path, top_n: int) -> None:
    stats_paths = [output_dir / "tables" / "source_url_stats.csv" for output_dir in output_dirs]
    review_rows: list[dict[str, Any]] = []
    fieldnames = _read_fieldnames(stats_paths[0])
    final_path = tables_dir / "source_url_stats.csv"

    with final_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in _merge_sorted_rows(stats_paths, _source_url_sort_key):
            writer.writerow(row)
            if len(review_rows) < top_n:
                review_rows.append(
                    {
                        "crawl": row.get("crawl", ""),
                        "rank": len(review_rows) + 1,
                        "source_url": row.get("source_url", ""),
                        "source_domain": row.get("source_domain", ""),
                        "root_domain": row.get("root_domain", ""),
                        "tranco_rank": row.get("tranco_rank", ""),
                        "tranco_bucket": row.get("tranco_bucket", ""),
                        "rows": _as_int(row.get("rows")),
                        "unique_target_platforms": _as_int(row.get("unique_target_platforms")),
                        "target_platforms": row.get("target_platforms", ""),
                        "unique_target_urls": _as_int(row.get("unique_target_urls")),
                        "top_labels": row.get("top_labels", ""),
                        "example_target_url": row.get("example_target_url", ""),
                        "example_prompt_text": row.get("example_prompt_text", ""),
                    }
                )

    write_csv(
        review_dir / "top_source_urls.csv",
        review_rows,
        [
            "crawl",
            "rank",
            "source_url",
            "source_domain",
            "root_domain",
            "tranco_rank",
            "tranco_bucket",
            "rows",
            "unique_target_platforms",
            "target_platforms",
            "unique_target_urls",
            "top_labels",
            "example_target_url",
            "example_prompt_text",
        ],
    )

    reuse_paths = [output_dir / "tables" / "source_platform_reuse.csv" for output_dir in output_dirs]
    _merge_simple_csv(reuse_paths, tables_dir / "source_platform_reuse.csv", _source_platform_reuse_sort_key)


def _merge_source_domain_tables(
    output_dirs: list[Path],
    tables_dir: Path,
    review_dir: Path,
    figure_dir: Path,
    top_n: int,
    total_rows_analyzed: int,
) -> None:
    stats_paths = [output_dir / "tables" / "source_domain_stats.csv" for output_dir in output_dirs]
    fieldnames = _read_fieldnames(stats_paths[0])
    review_rows: list[dict[str, Any]] = []
    final_path = tables_dir / "source_domain_stats.csv"
    rank_path = figure_dir / "source_domain_rank.csv"
    cumulative_rows = 0

    with final_path.open("w", encoding="utf-8", newline="") as handle, rank_path.open(
        "w", encoding="utf-8", newline=""
    ) as rank_handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        rank_writer = csv.DictWriter(
            rank_handle,
            fieldnames=[
                "crawl",
                "rank",
                "source_domain",
                "root_domain",
                "tranco_bucket",
                "rows",
                "cumulative_rows",
                "cumulative_share_rows",
            ],
        )
        rank_writer.writeheader()

        for index, row in enumerate(_merge_sorted_rows(stats_paths, _source_domain_sort_key), start=1):
            writer.writerow(row)
            rows = _as_int(row.get("rows"))
            cumulative_rows += rows
            rank_writer.writerow(
                {
                    "crawl": row.get("crawl", ""),
                    "rank": index,
                    "source_domain": row.get("source_domain", ""),
                    "root_domain": row.get("root_domain", ""),
                    "tranco_bucket": row.get("tranco_bucket", ""),
                    "rows": rows,
                    "cumulative_rows": cumulative_rows,
                    "cumulative_share_rows": round((cumulative_rows / total_rows_analyzed), 6) if total_rows_analyzed else 0.0,
                }
            )
            if len(review_rows) < top_n:
                review_rows.append(
                    {
                        "crawl": row.get("crawl", ""),
                        "rank": len(review_rows) + 1,
                        "source_domain": row.get("source_domain", ""),
                        "root_domain": row.get("root_domain", ""),
                        "tranco_rank": row.get("tranco_rank", ""),
                        "tranco_bucket": row.get("tranco_bucket", ""),
                        "rows": rows,
                        "unique_source_urls": _as_int(row.get("unique_source_urls")),
                        "unique_target_platforms": _as_int(row.get("unique_target_platforms")),
                        "target_platforms": row.get("target_platforms", ""),
                        "top_labels": row.get("top_labels", ""),
                        "top_source_url_1": row.get("top_source_url_1", ""),
                        "top_source_url_2": row.get("top_source_url_2", ""),
                        "example_target_url": row.get("example_target_url", ""),
                        "example_prompt_text": row.get("example_prompt_text", ""),
                    }
                )

    write_csv(
        review_dir / "top_source_domains.csv",
        review_rows,
        [
            "crawl",
            "rank",
            "source_domain",
            "root_domain",
            "tranco_rank",
            "tranco_bucket",
            "rows",
            "unique_source_urls",
            "unique_target_platforms",
            "target_platforms",
            "top_labels",
            "top_source_url_1",
            "top_source_url_2",
            "example_target_url",
            "example_prompt_text",
        ],
    )


def _merge_root_domain_tables(output_dirs: list[Path], tables_dir: Path, figure_dir: Path) -> None:
    stats_paths = [output_dir / "tables" / "root_domain_stats.csv" for output_dir in output_dirs]
    fieldnames = _read_fieldnames(stats_paths[0])
    final_path = tables_dir / "root_domain_stats.csv"
    intensity_path = figure_dir / "root_domain_intensity.csv"

    with final_path.open("w", encoding="utf-8", newline="") as handle, intensity_path.open(
        "w", encoding="utf-8", newline=""
    ) as intensity_handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        intensity_writer = csv.DictWriter(
            intensity_handle,
            fieldnames=[
                "crawl",
                "root_domain",
                "tranco_rank",
                "tranco_bucket",
                "rows",
                "unique_source_domains",
                "unique_source_urls",
            ],
        )
        intensity_writer.writeheader()

        for row in _merge_sorted_rows(stats_paths, _root_domain_sort_key):
            writer.writerow(row)
            intensity_writer.writerow(
                {
                    "crawl": row.get("crawl", ""),
                    "root_domain": row.get("root_domain", ""),
                    "tranco_rank": row.get("tranco_rank", ""),
                    "tranco_bucket": row.get("tranco_bucket", ""),
                    "rows": _as_int(row.get("rows")),
                    "unique_source_domains": _as_int(row.get("unique_source_domains")),
                    "unique_source_urls": _as_int(row.get("unique_source_urls")),
                }
            )

    top_tranco_paths = [output_dir / "tables" / "top_tranco_abused_domains.csv" for output_dir in output_dirs]
    _merge_simple_csv(top_tranco_paths, tables_dir / "top_tranco_abused_domains.csv", _top_tranco_sort_key)


def _merge_path_pattern_tables(output_dirs: list[Path], tables_dir: Path) -> None:
    stats_paths = [output_dir / "tables" / "source_path_patterns.csv" for output_dir in output_dirs]
    _merge_simple_csv(stats_paths, tables_dir / "source_path_patterns.csv", _path_pattern_sort_key)


def _write_bucket_outputs(
    crawl_name: str,
    bucket_rows: list[dict[str, Any]],
    bucket_platform_rows: list[dict[str, Any]],
    bucket_label_rows: list[dict[str, Any]],
    tables_dir: Path,
    figure_dir: Path,
) -> None:
    table_rows = [{"crawl": crawl_name, **row} for row in bucket_rows]
    write_csv(
        tables_dir / "tranco_bucket_summary.csv",
        table_rows,
        [
            "crawl",
            "tranco_bucket",
            "rows",
            "rows_with_prompt_text",
            "suspicious_rows",
            "high_rows",
            "medium_rows",
            "low_rows",
            "ioc_rows",
            "unique_root_domains",
            "unique_source_domains",
            "unique_source_urls",
            "unique_target_platforms",
        ],
    )
    write_csv(
        tables_dir / "tranco_bucket_by_platform.csv",
        bucket_platform_rows,
        ["crawl", "tranco_bucket", "target_platform", "rows", "share_within_bucket", "share_within_platform"],
    )
    write_csv(
        tables_dir / "tranco_bucket_by_label.csv",
        bucket_label_rows,
        ["crawl", "tranco_bucket", "label", "rows", "share_within_bucket", "share_within_label"],
    )
    write_csv(
        figure_dir / "tranco_bucket_rows.csv",
        [{"crawl": crawl_name, "tranco_bucket": row["tranco_bucket"], "rows": row["rows"]} for row in bucket_rows],
        ["crawl", "tranco_bucket", "rows"],
    )
    write_csv(
        figure_dir / "tranco_bucket_unique_root_domains.csv",
        [
            {
                "crawl": crawl_name,
                "tranco_bucket": row["tranco_bucket"],
                "unique_root_domains": row["unique_root_domains"],
            }
            for row in bucket_rows
        ],
        ["crawl", "tranco_bucket", "unique_root_domains"],
    )
    write_csv(
        figure_dir / "platform_by_tranco_bucket.csv",
        bucket_platform_rows,
        ["crawl", "tranco_bucket", "target_platform", "rows", "share_within_bucket", "share_within_platform"],
    )
    write_csv(
        figure_dir / "label_by_tranco_bucket.csv",
        bucket_label_rows,
        ["crawl", "tranco_bucket", "label", "rows", "share_within_bucket", "share_within_label"],
    )


def _build_summary(
    args: argparse.Namespace,
    crawl_name: str,
    input_path: Path,
    output_dir: Path,
    summaries: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    platform_counts: collections.Counter[str],
    severity_counts: collections.Counter[str],
    label_counts: collections.Counter[str],
    scheme_counts: collections.Counter[str],
    page_kind_counts: collections.Counter[str],
) -> dict[str, Any]:
    counts = _sum_count_block(
        summaries,
        [
            "rows_with_prompt_text",
            "rows_suspicious",
            "rows_with_ioc_keywords",
            "unique_source_urls",
            "unique_source_domains",
            "unique_root_domains",
            "multi_platform_source_urls",
            "ranked_source_domains",
            "ranked_root_domains",
        ],
    )
    quality = _sum_quality_counts(summaries)
    rows_analyzed = quality.get("rows_analyzed", 0)
    first_summary = summaries[0] if summaries else {}

    return {
        "generated_at_epoch": iso_now_epoch(),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "crawl": crawl_name,
        "filters": {
            "only_nonempty_prompt": args.only_nonempty_prompt,
            "suspicious_only": args.suspicious_only,
        },
        "tranco": dict(first_summary.get("tranco", {})),
        "quality": quality,
        "counts": counts,
        "distributions": {
            "platform_distribution": counter_to_sorted_rows(dict(platform_counts), total=rows_analyzed, key_name="target_platform"),
            "severity_distribution": counter_to_sorted_rows(dict(severity_counts), total=rows_analyzed, key_name="severity"),
            "label_distribution": counter_to_sorted_rows(dict(label_counts), total=rows_analyzed, key_name="label"),
            "scheme_distribution": counter_to_sorted_rows(dict(scheme_counts), total=rows_analyzed, key_name="scheme"),
            "page_kind_distribution": counter_to_sorted_rows(dict(page_kind_counts), total=rows_analyzed, key_name="page_kind"),
            "tranco_bucket_summary": ordered_bucket_rows([{"crawl": crawl_name, **row} for row in bucket_rows]),
        },
        "files": {
            "summary_json": str(output_dir / "summary.json"),
            "manifest_json": str(output_dir / "manifest.json"),
            "source_url_stats_csv": str(output_dir / "tables" / "source_url_stats.csv"),
            "source_domain_stats_csv": str(output_dir / "tables" / "source_domain_stats.csv"),
            "root_domain_stats_csv": str(output_dir / "tables" / "root_domain_stats.csv"),
            "source_path_patterns_csv": str(output_dir / "tables" / "source_path_patterns.csv"),
            "source_platform_reuse_csv": str(output_dir / "tables" / "source_platform_reuse.csv"),
            "tranco_bucket_summary_csv": str(output_dir / "tables" / "tranco_bucket_summary.csv"),
            "tranco_bucket_by_platform_csv": str(output_dir / "tables" / "tranco_bucket_by_platform.csv"),
            "tranco_bucket_by_label_csv": str(output_dir / "tables" / "tranco_bucket_by_label.csv"),
            "top_tranco_abused_domains_csv": str(output_dir / "tables" / "top_tranco_abused_domains.csv"),
            "top_source_urls_csv": str(output_dir / "review" / "top_source_urls.csv"),
            "top_source_domains_csv": str(output_dir / "review" / "top_source_domains.csv"),
        },
    }


def _build_manifest(
    output_dir: Path,
    crawl_name: str,
    input_path: Path,
    tranco_source: str,
    shard_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    tables_dir = output_dir / "tables"
    review_dir = output_dir / "review"
    figure_dir = output_dir / "figure_data"
    return {
        "script": "run_sharded_source_analysis.py",
        "version": 1,
        "crawl": crawl_name,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "tables": sorted(str(path) for path in tables_dir.iterdir()),
        "review": sorted(str(path) for path in review_dir.iterdir()),
        "figure_data": sorted(str(path) for path in figure_dir.iterdir()),
        "summary_json": str(output_dir / "summary.json"),
        "tranco_source": tranco_source,
        "shards": [
            {
                "index": spec["index"],
                "rows": spec["rows"],
                "output_dir": str(spec["output_dir"]),
            }
            for spec in shard_specs
        ],
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    review_dir = output_dir / "review"
    figure_dir = output_dir / "figure_data"
    work_dir = output_dir / "_sharded_work"
    shard_input_dir = work_dir / "inputs"
    shard_output_dir = work_dir / "outputs"

    ensure_directory(output_dir)
    ensure_directory(tables_dir)
    ensure_directory(review_dir)
    ensure_directory(figure_dir)

    if args.reuse_existing_shards:
        print(json.dumps({"stage": "reuse_shards", "work_dir": str(work_dir)}, ensure_ascii=False), flush=True)
        completed_specs = _load_existing_shards(work_dir)
    else:
        if work_dir.exists():
            shutil.rmtree(work_dir)

        print(json.dumps({"stage": "partition_start", "input": str(input_path), "shards": args.shards}, ensure_ascii=False), flush=True)
        shard_specs = _partition_input(input_path, shard_input_dir, args.shards)
        if not shard_specs:
            raise RuntimeError("No non-empty shards were produced.")
        print(
            json.dumps(
                {
                    "stage": "partition_done",
                    "nonempty_shards": len(shard_specs),
                    "rows": sum(int(spec["rows"]) for spec in shard_specs),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        completed_specs = _run_shard_analyzers(args, shard_specs, shard_output_dir)
    summaries = [spec["summary"] for spec in completed_specs]
    output_dirs = [spec["output_dir"] for spec in completed_specs]
    crawl_name = args.crawl_name.strip() or summaries[0].get("crawl", "") or input_path.stem

    print(json.dumps({"stage": "merge_start", "shards": len(output_dirs)}, ensure_ascii=False), flush=True)

    _merge_source_url_tables(output_dirs, tables_dir, review_dir, max(args.top_n, 1))
    total_rows_analyzed = _sum_quality_counts(summaries).get("rows_analyzed", 0)
    _merge_source_domain_tables(output_dirs, tables_dir, review_dir, figure_dir, max(args.top_n, 1), total_rows_analyzed)
    _merge_root_domain_tables(output_dirs, tables_dir, figure_dir)
    _merge_path_pattern_tables(output_dirs, tables_dir)

    platform_counts = _aggregate_distribution(summaries, "platform_distribution", "target_platform")
    severity_counts = _aggregate_distribution(summaries, "severity_distribution", "severity")
    label_counts = _aggregate_distribution(summaries, "label_distribution", "label")
    scheme_counts = _aggregate_distribution(summaries, "scheme_distribution", "scheme")
    page_kind_counts = _aggregate_distribution(summaries, "page_kind_distribution", "page_kind")

    bucket_rows = _aggregate_bucket_summary(output_dirs)
    bucket_platform_rows = _aggregate_bucket_platform_rows(output_dirs, crawl_name, platform_counts, bucket_rows)
    _apply_bucket_platform_uniques(bucket_rows, bucket_platform_rows)
    bucket_label_rows = _aggregate_bucket_label_rows(output_dirs, crawl_name, label_counts, bucket_rows)
    _write_bucket_outputs(crawl_name, bucket_rows, bucket_platform_rows, bucket_label_rows, tables_dir, figure_dir)

    summary = _build_summary(
        args,
        crawl_name,
        input_path,
        output_dir,
        summaries,
        bucket_rows,
        platform_counts,
        severity_counts,
        label_counts,
        scheme_counts,
        page_kind_counts,
    )
    tranco_source = str(summaries[0].get("tranco", {}).get("source", "")) if summaries else ""
    manifest = _build_manifest(output_dir, crawl_name, input_path, tranco_source, completed_specs)

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "manifest.json", manifest)

    if not args.keep_temp:
        shutil.rmtree(work_dir)

    print(
        json.dumps(
            {
                "stage": "merge_done",
                "crawl": crawl_name,
                "rows_seen": summary.get("quality", {}).get("rows_seen", 0),
                "rows_analyzed": summary.get("quality", {}).get("rows_analyzed", 0),
                "unique_source_urls": summary.get("counts", {}).get("unique_source_urls", 0),
                "unique_source_domains": summary.get("counts", {}).get("unique_source_domains", 0),
                "unique_root_domains": summary.get("counts", {}).get("unique_root_domains", 0),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())