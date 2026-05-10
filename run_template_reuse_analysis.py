#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from analyze_template_reuse import REVIEW_FIELDS, build_concentration_rows, build_reuse_bucket_rows
from risk_analysis_common import resolve_classified_input
from source_url_analysis_common import ensure_directory, iso_now_epoch, write_csv, write_json

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_BASE = Path(os.environ.get("RUNS_BASE") or os.environ.get("ARTIFACT_ROOT") or PROJECT_ROOT / "runs")
DEFAULT_REPORT_DIR = Path(os.environ.get("REPORTS_DIR") or PROJECT_ROOT / "analysis_reports")

DEFAULT_RUNS: tuple[tuple[str, str], ...] = (
    ("CC-MAIN-2025-51", str(DEFAULT_RUNS_BASE / "collect_ccmain2025_51")),
    ("CC-MAIN-2026-04", str(DEFAULT_RUNS_BASE / "collect_ccmain2026_04")),
    ("CC-MAIN-2026-08", str(DEFAULT_RUNS_BASE / "collect_ccmain2026_08")),
    ("CC-MAIN-2026-12", str(DEFAULT_RUNS_BASE / "collect_ccmain2026_12")),
)
DEFAULT_COMPARISON_ROOT = str(DEFAULT_RUNS_BASE / "template_reuse_analysis")
DEFAULT_REPORT_PATH = str(DEFAULT_REPORT_DIR / "template_reuse_analysis.md")
AGGREGATED_TEMPLATE_FIELDS: list[str] = [
    "template_hash",
    "row_count",
    "row_share",
    "risky_row_count",
    "risky_share_of_template",
    "medium_row_count",
    "high_row_count",
    "active_crawls",
    "crawls",
    "first_crawl",
    "last_crawl",
    "unique_source_domains",
    "top_source_domain",
    "top_source_domain_rows",
    "top_source_domain_share",
    "source_domain_head_json",
    "unique_target_platforms",
    "top_target_platform",
    "top_target_platform_rows",
    "top_target_platform_share",
    "target_platform_head_json",
    "unique_labels",
    "top_label",
    "top_label_rows",
    "top_label_share",
    "label_head_json",
    "prompt_length",
    "sample_prompt",
]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _iter_csv(path: Path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def _run_command(command: list[str]) -> None:
    print(json.dumps({"stage": "exec", "command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True)


def _ensure_output_dir_available(path: Path, *, allow_existing_output: bool) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise NotADirectoryError(f"Expected directory path: {path}")
    if any(path.iterdir()) and not allow_existing_output:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use --allow-existing-output if you want to reuse it."
        )


def _resolve_run_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.run_root:
        run_roots = [Path(value) for value in args.run_root]
        if args.crawl_name and len(args.crawl_name) != len(run_roots):
            raise ValueError("When --crawl-name is provided, it must appear exactly once per --run-root.")
        crawl_names = args.crawl_name or [path.name for path in run_roots]
        pairs = zip(crawl_names, run_roots)
    else:
        if args.crawl_name:
            raise ValueError("--crawl-name requires matching --run-root values.")
        pairs = ((crawl, Path(run_root)) for crawl, run_root in DEFAULT_RUNS)
    specs: list[dict[str, Any]] = []
    for crawl, run_root in pairs:
        specs.append(
            {
                "crawl": crawl,
                "run_root": run_root,
                "input": resolve_classified_input(run_root),
                "output_dir": run_root / args.output_dirname,
            }
        )
    return specs


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run exact prompt-template reuse analysis across completed Stage 02 runs.")
    parser.add_argument("--run-root", action="append", default=[], help="Optional run root. Repeatable.")
    parser.add_argument("--crawl-name", action="append", default=[], help="Optional crawl name matching each run root.")
    parser.add_argument("--script-dir", default=str(script_dir), help=f"Repository script directory (default: {script_dir}).")
    parser.add_argument("--comparison-root", default=DEFAULT_COMPARISON_ROOT, help=f"Shared root for combined outputs (default: {DEFAULT_COMPARISON_ROOT}).")
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH, help=f"Markdown report path (default: {DEFAULT_REPORT_PATH}).")
    parser.add_argument("--output-dirname", default="03d_template_reuse_analysis", help="Per-crawl output directory name.")
    parser.add_argument("--review-limit", type=int, default=500, help="Per-crawl and aggregate review CSV limit.")
    parser.add_argument("--allow-existing-output", action="store_true", help="Allow non-empty per-crawl output directories to exist before execution.")
    return parser.parse_args()


def _to_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key, 0)
    if value in (None, ""):
        return 0
    return int(value)


def _to_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    if value in (None, ""):
        return 0.0
    return float(value)


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _append_head(head: list[dict[str, Any]], payload: dict[str, Any], limit: int = 5) -> None:
    if len(head) < limit:
        head.append(payload)

def _aggregate_template_rows(
    template_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    platform_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    crawl_order: dict[str, int],
) -> list[dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    for row in template_rows:
        hash_value = str(row.get("template_hash", ""))
        if not hash_value:
            continue
        crawl = str(row.get("crawl", ""))
        entry = aggregate.get(hash_value)
        if entry is None:
            entry = {
                "template_hash": hash_value,
                "row_count": 0,
                "risky_row_count": 0,
                "medium_row_count": 0,
                "high_row_count": 0,
                "crawls_set": set(),
                "unique_source_domains": 0,
                "top_source_domain": "",
                "top_source_domain_rows": 0,
                "top_source_domain_share": 0.0,
                "source_domain_head": [],
                "unique_target_platforms": 0,
                "top_target_platform": "",
                "top_target_platform_rows": 0,
                "top_target_platform_share": 0.0,
                "target_platform_head": [],
                "unique_labels": 0,
                "top_label": "",
                "top_label_rows": 0,
                "top_label_share": 0.0,
                "label_head": [],
                "prompt_length": _to_int(row, "prompt_length"),
                "sample_prompt": str(row.get("sample_prompt", "")),
            }
            aggregate[hash_value] = entry
        entry["row_count"] += _to_int(row, "row_count")
        entry["risky_row_count"] += _to_int(row, "risky_row_count")
        entry["medium_row_count"] += _to_int(row, "medium_row_count")
        entry["high_row_count"] += _to_int(row, "high_row_count")
        entry["prompt_length"] = max(entry["prompt_length"], _to_int(row, "prompt_length"))
        if not entry["sample_prompt"] and row.get("sample_prompt"):
            entry["sample_prompt"] = str(row.get("sample_prompt", ""))
        entry["crawls_set"].add(crawl)

    source_aggregate: dict[tuple[str, str], dict[str, int]] = {}
    for row in source_rows:
        key = (str(row.get("template_hash", "")), str(row.get("source_domain", "")))
        bucket = source_aggregate.setdefault(key, {"row_count": 0, "risky_row_count": 0})
        bucket["row_count"] += _to_int(row, "row_count")
        bucket["risky_row_count"] += _to_int(row, "risky_row_count")

    platform_aggregate: dict[tuple[str, str], dict[str, int]] = {}
    for row in platform_rows:
        key = (str(row.get("template_hash", "")), str(row.get("target_platform", "")))
        bucket = platform_aggregate.setdefault(key, {"row_count": 0, "risky_row_count": 0})
        bucket["row_count"] += _to_int(row, "row_count")
        bucket["risky_row_count"] += _to_int(row, "risky_row_count")

    label_aggregate: dict[tuple[str, str], dict[str, int]] = {}
    for row in label_rows:
        key = (str(row.get("template_hash", "")), str(row.get("label", "")))
        bucket = label_aggregate.setdefault(key, {"row_count": 0, "risky_row_count": 0, "medium_row_count": 0, "high_row_count": 0})
        bucket["row_count"] += _to_int(row, "row_count")
        bucket["risky_row_count"] += _to_int(row, "risky_row_count")
        bucket["medium_row_count"] += _to_int(row, "medium_row_count")
        bucket["high_row_count"] += _to_int(row, "high_row_count")

    for (hash_value, source_domain), payload in sorted(source_aggregate.items(), key=lambda item: (item[0][0], -item[1]["row_count"], item[0][1])):
        entry = aggregate.get(hash_value)
        if entry is None:
            continue
        entry["unique_source_domains"] += 1
        if not entry["top_source_domain"]:
            entry["top_source_domain"] = source_domain
            entry["top_source_domain_rows"] = payload["row_count"]
        _append_head(entry["source_domain_head"], {"source_domain": source_domain, **payload})

    for (hash_value, target_platform), payload in sorted(platform_aggregate.items(), key=lambda item: (item[0][0], -item[1]["row_count"], item[0][1])):
        entry = aggregate.get(hash_value)
        if entry is None:
            continue
        entry["unique_target_platforms"] += 1
        if not entry["top_target_platform"]:
            entry["top_target_platform"] = target_platform
            entry["top_target_platform_rows"] = payload["row_count"]
        _append_head(entry["target_platform_head"], {"target_platform": target_platform, **payload})

    for (hash_value, label), payload in sorted(label_aggregate.items(), key=lambda item: (item[0][0], -item[1]["row_count"], item[0][1])):
        entry = aggregate.get(hash_value)
        if entry is None:
            continue
        entry["unique_labels"] += 1
        if not entry["top_label"]:
            entry["top_label"] = label
            entry["top_label_rows"] = payload["row_count"]
        _append_head(entry["label_head"], {"label": label, **payload})

    total_rows = sum(entry["row_count"] for entry in aggregate.values())
    materialized: list[dict[str, Any]] = []
    for entry in aggregate.values():
        crawls = sorted(entry["crawls_set"], key=lambda crawl: crawl_order.get(crawl, 10**9))
        row_count = entry["row_count"]
        materialized.append(
            {
                "template_hash": entry["template_hash"],
                "row_count": row_count,
                "row_share": round((row_count / total_rows), 6) if total_rows else 0.0,
                "risky_row_count": entry["risky_row_count"],
                "risky_share_of_template": round((entry["risky_row_count"] / row_count), 6) if row_count else 0.0,
                "medium_row_count": entry["medium_row_count"],
                "high_row_count": entry["high_row_count"],
                "active_crawls": len(crawls),
                "crawls": " | ".join(crawls),
                "first_crawl": crawls[0] if crawls else "",
                "last_crawl": crawls[-1] if crawls else "",
                "unique_source_domains": entry["unique_source_domains"],
                "top_source_domain": entry["top_source_domain"],
                "top_source_domain_rows": entry["top_source_domain_rows"],
                "top_source_domain_share": round((entry["top_source_domain_rows"] / row_count), 6) if row_count else 0.0,
                "source_domain_head_json": _json_dumps(entry["source_domain_head"]),
                "unique_target_platforms": entry["unique_target_platforms"],
                "top_target_platform": entry["top_target_platform"],
                "top_target_platform_rows": entry["top_target_platform_rows"],
                "top_target_platform_share": round((entry["top_target_platform_rows"] / row_count), 6) if row_count else 0.0,
                "target_platform_head_json": _json_dumps(entry["target_platform_head"]),
                "unique_labels": entry["unique_labels"],
                "top_label": entry["top_label"],
                "top_label_rows": entry["top_label_rows"],
                "top_label_share": round((entry["top_label_rows"] / row_count), 6) if row_count else 0.0,
                "label_head_json": _json_dumps(entry["label_head"]),
                "prompt_length": entry["prompt_length"],
                "sample_prompt": entry["sample_prompt"],
            }
        )
    materialized.sort(key=lambda row: (-_to_int(row, "row_count"), -_to_int(row, "active_crawls"), row.get("template_hash", "")))
    return materialized

def _render_markdown_report(
    *,
    run_summaries: list[dict[str, Any]],
    aggregated_rows: list[dict[str, Any]],
    report_path: Path,
    comparison_root: Path,
    n_crawls: int,
) -> None:
    total_rows = sum(_to_int(row, "row_count") for row in aggregated_rows)
    unique_templates = len(aggregated_rows)
    singleton_count = sum(1 for row in aggregated_rows if _to_int(row, "row_count") == 1)
    rows_from_singletons = sum(_to_int(row, "row_count") for row in aggregated_rows if _to_int(row, "row_count") == 1)
    rows_from_reused = total_rows - rows_from_singletons
    distributed_rows = sum(_to_int(row, "row_count") for row in aggregated_rows if _to_int(row, "unique_source_domains") >= 5)
    cross_platform_rows = sum(_to_int(row, "row_count") for row in aggregated_rows if _to_int(row, "unique_target_platforms") >= 3)
    persistent_rows = sum(_to_int(row, "row_count") for row in aggregated_rows if _to_int(row, "active_crawls") >= 2)
    all_four_rows = sum(_to_int(row, "row_count") for row in aggregated_rows if _to_int(row, "active_crawls") == n_crawls)
    concentration = {row["scope"]: row for row in build_concentration_rows(aggregated_rows, "ALL")}
    top_reused = aggregated_rows[:10]
    distributed_templates = [row for row in aggregated_rows if _to_int(row, "unique_source_domains") >= 5][:10]
    persistent_templates = [row for row in aggregated_rows if _to_int(row, "active_crawls") >= 2][:10]

    lines: list[str] = []
    lines.append("# Template Reuse Analysis")
    lines.append("")
    lines.append(f"Generated at epoch: {iso_now_epoch()}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(f"- Input: Stage 02 classified prompt-link outputs from {n_crawls} completed crawls")
    lines.append("- Template definition: exact normalized prompt after HTML unescape, whitespace collapse, and URL replacement with `<URL>`")
    lines.append("- Goal: measure how much of the corpus is driven by repeated exact prompt templates rather than one-off prompts")
    lines.append("- Key spread dimensions: unique source domains, unique target platforms, and cross-crawl persistence")
    lines.append("")
    lines.append("## Overview By Crawl")
    lines.append("")
    lines.append("| Crawl | Prompt Rows | Exact Templates | Avg Reuse | Reused Row Share | Top-10 Share | Top-100 Share |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for summary in run_summaries:
        shares = summary.get("shares", {})
        lines.append(
            f"| {summary.get('crawl', '')} | {int(summary.get('rows_with_prompt', 0)):,} | {int(summary.get('unique_templates', 0)):,} | {float(shares.get('avg_rows_per_template', 0.0)):.2f} | {_format_pct(float(shares.get('rows_from_reused_templates_share', 0.0)))} | {_format_pct(float(shares.get('top_10_row_share', 0.0)))} | {_format_pct(float(shares.get('top_100_row_share', 0.0)))} |"
        )
    lines.append("")
    lines.append("## All-Crawl Combined")
    lines.append("")
    lines.append(f"- Combined prompt rows with extracted prompt text: {total_rows:,}")
    lines.append(f"- Combined exact templates: {unique_templates:,}")
    lines.append(f"- Average rows per exact template: {(total_rows / unique_templates):.2f}" if unique_templates else "- Average rows per exact template: 0.00")
    lines.append(f"- Singleton templates: {singleton_count:,} ({_format_pct(singleton_count / unique_templates) if unique_templates else '0.00%'}) of templates but only {_format_pct(rows_from_singletons / total_rows) if total_rows else '0.00%'} of rows")
    lines.append(f"- Reused templates (count >= 2) drive {_format_pct(rows_from_reused / total_rows) if total_rows else '0.00%'} of rows")
    lines.append(f"- Top 10 exact templates cover {_format_pct(_to_float(concentration['top_10'], 'row_share'))}; top 100 cover {_format_pct(_to_float(concentration['top_100'], 'row_share'))}; top 1000 cover {_format_pct(_to_float(concentration['top_1000'], 'row_share'))}")
    lines.append(f"- Templates seen on >=5 source domains cover {_format_pct(distributed_rows / total_rows) if total_rows else '0.00%'} of rows")
    lines.append(f"- Templates seen on >=3 target platforms cover {_format_pct(cross_platform_rows / total_rows) if total_rows else '0.00%'} of rows")
    lines.append(f"- Templates persisting across >=2 crawls cover {_format_pct(persistent_rows / total_rows) if total_rows else '0.00%'} of rows; templates present in all {n_crawls} crawls cover {_format_pct(all_four_rows / total_rows) if total_rows else '0.00%'}")
    lines.append("")
    lines.append("## Top Exact Templates")
    lines.append("")
    for row in top_reused:
        lines.append(
            f"- rows={_to_int(row, 'row_count'):,}; crawls={row.get('crawls', '')}; source_domains={_to_int(row, 'unique_source_domains')}; platforms={_to_int(row, 'unique_target_platforms')}; top_label={row.get('top_label', '')}; prompt={row.get('sample_prompt', '')}"
        )
    lines.append("")
    lines.append("## Distributed Templates")
    lines.append("")
    for row in distributed_templates:
        lines.append(
            f"- rows={_to_int(row, 'row_count'):,}; source_domains={_to_int(row, 'unique_source_domains')}; top_source_share={_format_pct(_to_float(row, 'top_source_domain_share'))}; platforms={_to_int(row, 'unique_target_platforms')}; prompt={row.get('sample_prompt', '')}"
        )
    lines.append("")
    lines.append("## Persistent Templates")
    lines.append("")
    for row in persistent_templates:
        lines.append(
            f"- rows={_to_int(row, 'row_count'):,}; active_crawls={_to_int(row, 'active_crawls')}; crawls={row.get('crawls', '')}; top_label={row.get('top_label', '')}; prompt={row.get('sample_prompt', '')}"
        )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Comparison root: `{comparison_root}`")
    lines.append(f"- Report path: `{report_path}`")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_fields(rows: list[dict[str, Any]], fieldnames: list[str]) -> list[dict[str, Any]]:
    return [{field: row.get(field, "") for field in fieldnames} for row in rows]

def main() -> int:
    args = parse_args()
    script_dir = Path(args.script_dir)
    analyzer_script = script_dir / "analyze_template_reuse.py"
    run_specs = _resolve_run_specs(args)
    crawl_order = {spec["crawl"]: index for index, spec in enumerate(run_specs)}

    run_summaries: list[dict[str, Any]] = []
    overview_by_crawl: list[dict[str, Any]] = []
    all_template_rows: list[dict[str, Any]] = []
    all_source_rows: list[dict[str, Any]] = []
    all_platform_rows: list[dict[str, Any]] = []
    all_label_rows: list[dict[str, Any]] = []

    for spec in run_specs:
        _ensure_output_dir_available(spec["output_dir"], allow_existing_output=args.allow_existing_output)
        _run_command(
            [
                sys.executable,
                str(analyzer_script),
                "--input", str(spec["input"]),
                "--output-dir", str(spec["output_dir"]),
                "--crawl-name", str(spec["crawl"]),
                "--review-limit", str(args.review_limit),
            ]
        )
        summary = _read_json(spec["output_dir"] / "summary.json")
        summary["crawl"] = spec["crawl"]
        run_summaries.append(summary)
        shares = summary.get("shares", {})
        overview_by_crawl.append(
            {
                "crawl": spec["crawl"],
                "rows_seen": int(summary.get("rows_seen", 0)),
                "rows_with_prompt": int(summary.get("rows_with_prompt", 0)),
                "unique_templates": int(summary.get("unique_templates", 0)),
                "avg_rows_per_template": round(float(shares.get("avg_rows_per_template", 0.0)), 6),
                "rows_from_reused_templates_share": round(float(shares.get("rows_from_reused_templates_share", 0.0)), 6),
                "rows_from_singletons_share": round(float(shares.get("rows_from_singletons_share", 0.0)), 6),
                "distributed_rows_ge_5_domains_share": round(float(shares.get("distributed_rows_ge_5_domains_share", 0.0)), 6),
                "cross_platform_rows_ge_3_platforms_share": round(float(shares.get("cross_platform_rows_ge_3_platforms_share", 0.0)), 6),
                "top_10_row_share": round(float(shares.get("top_10_row_share", 0.0)), 6),
                "top_100_row_share": round(float(shares.get("top_100_row_share", 0.0)), 6),
                "top_1000_row_share": round(float(shares.get("top_1000_row_share", 0.0)), 6),
                "elapsed_seconds": round(float(summary.get("runtime", {}).get("elapsed_seconds", 0.0)), 3),
                "rows_per_second": round(float(summary.get("runtime", {}).get("rows_per_second", 0.0)), 3),
                "templates_per_second": round(float(summary.get("runtime", {}).get("templates_per_second", 0.0)), 3),
            }
        )
        all_template_rows.extend(_iter_csv(spec["output_dir"] / "tables" / "template_overview.csv"))
        all_source_rows.extend(_iter_csv(spec["output_dir"] / "tables" / "template_source_stats.csv"))
        all_platform_rows.extend(_iter_csv(spec["output_dir"] / "tables" / "template_platform_stats.csv"))
        all_label_rows.extend(_iter_csv(spec["output_dir"] / "tables" / "template_label_stats.csv"))

    comparison_root = Path(args.comparison_root)
    tables_dir = comparison_root / "tables"
    review_dir = comparison_root / "review"
    ensure_directory(comparison_root)
    ensure_directory(tables_dir)
    ensure_directory(review_dir)

    aggregated_rows = _aggregate_template_rows(all_template_rows, all_source_rows, all_platform_rows, all_label_rows, crawl_order)
    aggregated_concentration = build_concentration_rows(aggregated_rows, "ALL")
    aggregated_buckets = build_reuse_bucket_rows(aggregated_rows, "ALL")

    write_csv(
        tables_dir / "overview_by_crawl.csv",
        overview_by_crawl,
        [
            "crawl", "rows_seen", "rows_with_prompt", "unique_templates", "avg_rows_per_template",
            "rows_from_reused_templates_share", "rows_from_singletons_share",
            "distributed_rows_ge_5_domains_share", "cross_platform_rows_ge_3_platforms_share",
            "top_10_row_share", "top_100_row_share", "top_1000_row_share",
            "elapsed_seconds", "rows_per_second", "templates_per_second",
        ],
    )
    write_csv(tables_dir / "template_overview_all_crawls.csv", all_template_rows, list(all_template_rows[0].keys()) if all_template_rows else [])
    write_csv(tables_dir / "template_overview_aggregated.csv", aggregated_rows, AGGREGATED_TEMPLATE_FIELDS)
    write_csv(tables_dir / "template_source_stats_all_crawls.csv", all_source_rows, list(all_source_rows[0].keys()) if all_source_rows else [])
    write_csv(tables_dir / "template_platform_stats_all_crawls.csv", all_platform_rows, list(all_platform_rows[0].keys()) if all_platform_rows else [])
    write_csv(tables_dir / "template_label_stats_all_crawls.csv", all_label_rows, list(all_label_rows[0].keys()) if all_label_rows else [])
    write_csv(tables_dir / "concentration_aggregated.csv", aggregated_concentration, ["crawl", "scope", "template_count", "row_count", "row_share", "risky_row_count", "risky_row_share"])
    write_csv(tables_dir / "reuse_buckets_aggregated.csv", aggregated_buckets, ["crawl", "reuse_bucket", "template_count", "template_share", "row_count", "row_share", "risky_row_count", "risky_row_share"])

    top_reused = aggregated_rows[: max(args.review_limit, 1)]
    distributed_templates = [row for row in aggregated_rows if _to_int(row, "unique_source_domains") >= 5][: max(args.review_limit, 1)]
    cross_platform_templates = [row for row in aggregated_rows if _to_int(row, "unique_target_platforms") >= 3][: max(args.review_limit, 1)]
    persistent_templates = [row for row in aggregated_rows if _to_int(row, "active_crawls") >= 2][: max(args.review_limit, 1)]
    all_four_templates = [row for row in aggregated_rows if _to_int(row, "active_crawls") == len(run_specs)][: max(args.review_limit, 1)]
    aggregated_review_fields = REVIEW_FIELDS + ["active_crawls", "crawls", "first_crawl", "last_crawl"]
    write_csv(review_dir / "top_reused_templates_aggregated.csv", _select_fields(top_reused, aggregated_review_fields), aggregated_review_fields)
    write_csv(review_dir / "distributed_templates_aggregated.csv", _select_fields(distributed_templates, aggregated_review_fields), aggregated_review_fields)
    write_csv(review_dir / "cross_platform_templates_aggregated.csv", _select_fields(cross_platform_templates, aggregated_review_fields), aggregated_review_fields)
    write_csv(review_dir / "persistent_templates.csv", _select_fields(persistent_templates, aggregated_review_fields), aggregated_review_fields)
    write_csv(review_dir / "all_four_crawls_templates.csv", _select_fields(all_four_templates, aggregated_review_fields), aggregated_review_fields)

    report_path = Path(args.report_path)
    _render_markdown_report(
        run_summaries=run_summaries,
        aggregated_rows=aggregated_rows,
        report_path=report_path,
        comparison_root=comparison_root,
        n_crawls=len(run_specs),
    )

    total_rows = sum(_to_int(row, "row_count") for row in aggregated_rows)
    unique_templates = len(aggregated_rows)
    summary = {
        "generated_at_epoch": iso_now_epoch(),
        "comparison_root": str(comparison_root),
        "report_path": str(report_path),
        "rows_with_prompt": total_rows,
        "unique_templates": unique_templates,
        "shares": {
            "avg_rows_per_template": round((total_rows / unique_templates), 6) if unique_templates else 0.0,
            "rows_from_reused_templates_share": round(
                (sum(_to_int(row, 'row_count') for row in aggregated_rows if _to_int(row, 'row_count') >= 2) / total_rows), 6
            ) if total_rows else 0.0,
            "persistent_rows_share": round(
                (sum(_to_int(row, 'row_count') for row in aggregated_rows if _to_int(row, 'active_crawls') >= 2) / total_rows), 6
            ) if total_rows else 0.0,
        },
        "runs": overview_by_crawl,
        "files": {
            "overview_by_crawl_csv": str(tables_dir / "overview_by_crawl.csv"),
            "template_overview_aggregated_csv": str(tables_dir / "template_overview_aggregated.csv"),
            "concentration_aggregated_csv": str(tables_dir / "concentration_aggregated.csv"),
            "reuse_buckets_aggregated_csv": str(tables_dir / "reuse_buckets_aggregated.csv"),
            "top_reused_templates_aggregated_csv": str(review_dir / "top_reused_templates_aggregated.csv"),
            "distributed_templates_aggregated_csv": str(review_dir / "distributed_templates_aggregated.csv"),
            "persistent_templates_csv": str(review_dir / "persistent_templates.csv"),
            "all_four_crawls_templates_csv": str(review_dir / "all_four_crawls_templates.csv"),
        },
    }
    write_json(comparison_root / "summary.json", summary)
    print(json.dumps({"comparison_root": str(comparison_root), "report_path": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
