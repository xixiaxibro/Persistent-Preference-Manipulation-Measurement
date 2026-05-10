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
DEFAULT_COMPARISON_ROOT = str(DEFAULT_RUNS_BASE / "prompt_language_analysis")
DEFAULT_REPORT_PATH = str(DEFAULT_REPORT_DIR / "prompt_language_analysis.md")


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
    parser = argparse.ArgumentParser(
        description="Run prompt-language analysis across completed Stage 02 classified prompt-link runs."
    )
    parser.add_argument("--run-root", action="append", default=[], help="Optional run root. Repeatable.")
    parser.add_argument("--crawl-name", action="append", default=[], help="Optional crawl name matching each run root.")
    parser.add_argument("--script-dir", default=str(script_dir), help=f"Repository script directory (default: {script_dir}).")
    parser.add_argument(
        "--comparison-root",
        default=DEFAULT_COMPARISON_ROOT,
        help=f"Shared root for combined outputs (default: {DEFAULT_COMPARISON_ROOT}).",
    )
    parser.add_argument(
        "--report-path",
        default=DEFAULT_REPORT_PATH,
        help=f"Markdown report path (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--output-dirname",
        default="03c_language_analysis",
        help="Per-crawl output directory name.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.80,
        help="Minimum normalized langid confidence to accept a prediction directly.",
    )
    parser.add_argument(
        "--min-alpha-chars",
        type=int,
        default=4,
        help="Short-prompt threshold used by the per-crawl analyzer.",
    )
    parser.add_argument("--review-limit", type=int, default=500, help="Per-crawl review CSV limit.")
    parser.add_argument("--top-lang-limit", type=int, default=25, help="Per-crawl summary top-language limit.")
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow non-empty per-crawl output directories to exist before execution.",
    )
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


def _top_langs_text(rows: list[dict[str, Any]], *, limit: int = 5, key: str = "row_count") -> str:
    selected = sorted(rows, key=lambda row: (-_to_int(row, key), row.get("lang", "")))[: max(limit, 1)]
    return " | ".join(f"{row.get('lang', '')}:{_format_pct(_to_float(row, 'row_share'))}" for row in selected)


def _aggregate_language_overview(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    for row in rows:
        lang = str(row.get("lang", "und"))
        bucket = aggregate.setdefault(
            lang,
            {
                "crawl": "ALL",
                "lang": lang,
                "row_count": 0,
                "risky_row_count": 0,
                "unique_prompts": 0,
                "risky_unique_prompts": 0,
                "confidence_weighted_sum": 0.0,
            },
        )
        bucket["row_count"] += _to_int(row, "row_count")
        bucket["risky_row_count"] += _to_int(row, "risky_row_count")
        bucket["unique_prompts"] += _to_int(row, "unique_prompts")
        bucket["risky_unique_prompts"] += _to_int(row, "risky_unique_prompts")
        bucket["confidence_weighted_sum"] += _to_float(row, "avg_confidence") * _to_int(row, "unique_prompts")
    total_rows = sum(bucket["row_count"] for bucket in aggregate.values())
    total_risky_rows = sum(bucket["risky_row_count"] for bucket in aggregate.values())
    total_unique_prompts = sum(bucket["unique_prompts"] for bucket in aggregate.values())
    total_risky_unique_prompts = sum(bucket["risky_unique_prompts"] for bucket in aggregate.values())
    materialized: list[dict[str, Any]] = []
    for bucket in aggregate.values():
        unique_prompts = bucket["unique_prompts"]
        materialized.append(
            {
                "crawl": "ALL",
                "lang": bucket["lang"],
                "row_count": bucket["row_count"],
                "row_share": round((bucket["row_count"] / total_rows), 6) if total_rows else 0.0,
                "risky_row_count": bucket["risky_row_count"],
                "risky_row_share": round((bucket["risky_row_count"] / total_risky_rows), 6) if total_risky_rows else 0.0,
                "unique_prompts": unique_prompts,
                "unique_prompt_share": round((unique_prompts / total_unique_prompts), 6) if total_unique_prompts else 0.0,
                "risky_unique_prompts": bucket["risky_unique_prompts"],
                "risky_unique_prompt_share": round((bucket["risky_unique_prompts"] / total_risky_unique_prompts), 6)
                if total_risky_unique_prompts else 0.0,
                "avg_confidence": round((bucket["confidence_weighted_sum"] / unique_prompts), 6) if unique_prompts else 0.0,
            }
        )
    materialized.sort(key=lambda row: (-_to_int(row, "row_count"), row.get("lang", "")))
    return materialized

def _aggregate_dimension_rows(rows: list[dict[str, Any]], dimension_key: str) -> list[dict[str, Any]]:
    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        dimension_value = str(row.get(dimension_key, ""))
        lang = str(row.get("lang", "und"))
        key = (dimension_value, lang)
        bucket = aggregate.setdefault(
            key,
            {
                "crawl": "ALL",
                dimension_key: dimension_value,
                "lang": lang,
                "row_count": 0,
                "risky_row_count": 0,
                "unique_prompts": 0,
                "risky_unique_prompts": 0,
                "medium_row_count": 0,
                "high_row_count": 0,
            },
        )
        bucket["row_count"] += _to_int(row, "row_count")
        bucket["risky_row_count"] += _to_int(row, "risky_row_count")
        bucket["unique_prompts"] += _to_int(row, "unique_prompts")
        bucket["risky_unique_prompts"] += _to_int(row, "risky_unique_prompts")
        bucket["medium_row_count"] += _to_int(row, "medium_row_count")
        bucket["high_row_count"] += _to_int(row, "high_row_count")
        total_bucket = totals.setdefault(
            dimension_value,
            {"row_count": 0, "risky_row_count": 0, "unique_prompts": 0, "risky_unique_prompts": 0},
        )
        total_bucket["row_count"] += _to_int(row, "row_count")
        total_bucket["risky_row_count"] += _to_int(row, "risky_row_count")
        total_bucket["unique_prompts"] += _to_int(row, "unique_prompts")
        total_bucket["risky_unique_prompts"] += _to_int(row, "risky_unique_prompts")
    materialized: list[dict[str, Any]] = []
    for bucket in aggregate.values():
        total_bucket = totals[bucket[dimension_key]]
        materialized.append(
            {
                "crawl": "ALL",
                dimension_key: bucket[dimension_key],
                "lang": bucket["lang"],
                "row_count": bucket["row_count"],
                f"row_share_within_{dimension_key}": round((bucket["row_count"] / total_bucket["row_count"]), 6)
                if total_bucket["row_count"] else 0.0,
                "risky_row_count": bucket["risky_row_count"],
                f"risky_row_share_within_{dimension_key}": round((bucket["risky_row_count"] / total_bucket["risky_row_count"]), 6)
                if total_bucket["risky_row_count"] else 0.0,
                "unique_prompts": bucket["unique_prompts"],
                f"unique_prompt_share_within_{dimension_key}": round((bucket["unique_prompts"] / total_bucket["unique_prompts"]), 6)
                if total_bucket["unique_prompts"] else 0.0,
                "risky_unique_prompts": bucket["risky_unique_prompts"],
                f"risky_unique_prompt_share_within_{dimension_key}": round(
                    (bucket["risky_unique_prompts"] / total_bucket["risky_unique_prompts"]), 6
                ) if total_bucket["risky_unique_prompts"] else 0.0,
                "medium_row_count": bucket["medium_row_count"],
                "high_row_count": bucket["high_row_count"],
            }
        )
    materialized.sort(key=lambda row: (row[dimension_key], -_to_int(row, "row_count"), row.get("lang", "")))
    return materialized


def _aggregate_severity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        severity = str(row.get("severity", ""))
        lang = str(row.get("lang", "und"))
        key = (severity, lang)
        bucket = aggregate.setdefault(key, {"crawl": "ALL", "severity": severity, "lang": lang, "row_count": 0, "unique_prompts": 0})
        bucket["row_count"] += _to_int(row, "row_count")
        bucket["unique_prompts"] += _to_int(row, "unique_prompts")
        total_bucket = totals.setdefault(severity, {"row_count": 0, "unique_prompts": 0})
        total_bucket["row_count"] += _to_int(row, "row_count")
        total_bucket["unique_prompts"] += _to_int(row, "unique_prompts")
    materialized: list[dict[str, Any]] = []
    for bucket in aggregate.values():
        total_bucket = totals[bucket["severity"]]
        materialized.append(
            {
                "crawl": "ALL",
                "severity": bucket["severity"],
                "lang": bucket["lang"],
                "row_count": bucket["row_count"],
                "row_share_within_severity": round((bucket["row_count"] / total_bucket["row_count"]), 6)
                if total_bucket["row_count"] else 0.0,
                "unique_prompts": bucket["unique_prompts"],
                "unique_prompt_share_within_severity": round((bucket["unique_prompts"] / total_bucket["unique_prompts"]), 6)
                if total_bucket["unique_prompts"] else 0.0,
            }
        )
    materialized.sort(key=lambda row: (row["severity"], -_to_int(row, "row_count"), row.get("lang", "")))
    return materialized


def _render_markdown_report(
    *,
    run_summaries: list[dict[str, Any]],
    aggregated_overview: list[dict[str, Any]],
    aggregated_label_rows: list[dict[str, Any]],
    aggregated_platform_rows: list[dict[str, Any]],
    report_path: Path,
    comparison_root: Path,
    config: dict[str, Any],
) -> None:
    overall_total_rows = sum(_to_int(row, "row_count") for row in aggregated_overview)
    overall_total_unique_prompts = sum(_to_int(row, "unique_prompts") for row in aggregated_overview)
    non_english_rows = [row for row in aggregated_overview if row.get("lang") not in {"en", "und"}]
    top_non_english_rows = sorted(non_english_rows, key=lambda row: (-_to_int(row, "row_count"), row.get("lang", "")))[:10]
    top_unique_non_english_rows = sorted(non_english_rows, key=lambda row: (-_to_int(row, "unique_prompts"), row.get("lang", "")))[:10]
    lines: list[str] = []
    lines.append("# Prompt Language Analysis")
    lines.append("")
    lines.append(f"Generated at epoch: {iso_now_epoch()}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(f"- Input: Stage 02 classified prompt-link outputs from {len(run_summaries)} completed crawls")
    lines.append("- Unit of analysis: normalized `primary_prompt_text`/prompt text with URL tokens replaced by `<URL>`")
    lines.append("- Deduplication: SHA1 over normalized prompt text, then language inferred once per unique prompt")
    lines.append(
        f"- Detector: langid with confidence threshold {config['confidence_threshold']:.2f} and short-prompt threshold {config['min_alpha_chars']} alpha chars"
    )
    lines.append("- Reporting: both row-weighted exposure scale and unique-prompt template diversity")
    lines.append("- Caveat: label tables are multi-label, so label totals should be read per label, not summed across labels")
    lines.append("")
    lines.append("## Overview By Crawl")
    lines.append("")
    lines.append("| Crawl | Prompt Rows | Unique Prompts | English Rows | Non-English Rows | Unknown Rows | Top Row Languages |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in run_summaries:
        shares = summary.get("shares", {})
        top_text = "; ".join(
            f"{item.get('lang', '')}:{_format_pct(float(item.get('row_share', 0.0)))}"
            for item in summary.get("top_languages_by_rows", [])[:5]
        )
        lines.append(
            f"| {summary.get('crawl', '')} | {int(summary.get('rows_with_prompt', 0)):,} | {int(summary.get('unique_normalized_prompts', 0)):,} | {_format_pct(float(shares.get('english_row_share', 0.0)))} | {_format_pct(float(shares.get('non_english_row_share', 0.0)))} | {_format_pct(float(shares.get('unknown_row_share', 0.0)))} | {top_text} |"
        )
    lines.append("")
    lines.append("## All-Crawl Combined")
    lines.append("")
    lines.append(f"- Combined prompt rows with extracted prompt text: {overall_total_rows:,}")
    lines.append(f"- Combined unique normalized prompts: {overall_total_unique_prompts:,}")
    lines.append("- Top languages by row-weighted share:")
    for row in aggregated_overview[:10]:
        lines.append(
            f"  - {row.get('lang', '')}: {_format_pct(_to_float(row, 'row_share'))} rows, {_format_pct(_to_float(row, 'unique_prompt_share'))} unique prompts"
        )
    lines.append("- Top non-English languages by rows:")
    for row in top_non_english_rows:
        lines.append(
            f"  - {row.get('lang', '')}: {_format_pct(_to_float(row, 'row_share'))} rows, {_format_pct(_to_float(row, 'risky_row_share'))} of risky rows"
        )
    lines.append("- Top non-English languages by unique prompts:")
    for row in top_unique_non_english_rows:
        lines.append(
            f"  - {row.get('lang', '')}: {_format_pct(_to_float(row, 'unique_prompt_share'))} unique prompts"
        )
    lines.append("")
    lines.append("## By Label (Rows)")
    lines.append("")
    for label in sorted({row.get('label', '') for row in aggregated_label_rows}):
        label_rows = [row for row in aggregated_label_rows if row.get("label") == label][:5]
        if not label_rows:
            continue
        lines.append(f"- {label}: " + "; ".join(
            f"{row.get('lang', '')} {_format_pct(_to_float(row, 'row_share_within_label'))}" for row in label_rows
        ))
    lines.append("")
    lines.append("## By Platform (Rows)")
    lines.append("")
    for platform in sorted({row.get('target_platform', '') for row in aggregated_platform_rows}):
        platform_rows = [row for row in aggregated_platform_rows if row.get("target_platform") == platform][:5]
        if not platform_rows:
            continue
        lines.append(f"- {platform}: " + "; ".join(
            f"{row.get('lang', '')} {_format_pct(_to_float(row, 'row_share_within_target_platform'))}" for row in platform_rows
        ))
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Comparison root: `{comparison_root}`")
    lines.append(f"- Report path: `{report_path}`")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    script_dir = Path(args.script_dir)
    analyzer_script = script_dir / "analyze_prompt_language.py"
    run_specs = _resolve_run_specs(args)
    overview_by_crawl: list[dict[str, Any]] = []
    all_overview_rows: list[dict[str, Any]] = []
    all_severity_rows: list[dict[str, Any]] = []
    all_label_rows: list[dict[str, Any]] = []
    all_platform_rows: list[dict[str, Any]] = []
    all_uncertain_rows: list[dict[str, Any]] = []
    all_non_english_review_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    for spec in run_specs:
        _ensure_output_dir_available(spec["output_dir"], allow_existing_output=args.allow_existing_output)
        _run_command(
            [
                sys.executable,
                str(analyzer_script),
                "--input", str(spec["input"]),
                "--output-dir", str(spec["output_dir"]),
                "--crawl-name", str(spec["crawl"]),
                "--confidence-threshold", str(args.confidence_threshold),
                "--min-alpha-chars", str(args.min_alpha_chars),
                "--review-limit", str(args.review_limit),
                "--top-lang-limit", str(args.top_lang_limit),
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
                "rows_missing_prompt": int(summary.get("rows_missing_prompt", 0)),
                "unique_normalized_prompts": int(summary.get("unique_normalized_prompts", 0)),
                "english_row_share": round(float(shares.get("english_row_share", 0.0)), 6),
                "non_english_row_share": round(float(shares.get("non_english_row_share", 0.0)), 6),
                "unknown_row_share": round(float(shares.get("unknown_row_share", 0.0)), 6),
                "english_unique_prompt_share": round(float(shares.get("english_unique_prompt_share", 0.0)), 6),
                "non_english_unique_prompt_share": round(float(shares.get("non_english_unique_prompt_share", 0.0)), 6),
                "unknown_unique_prompt_share": round(float(shares.get("unknown_unique_prompt_share", 0.0)), 6),
                "elapsed_seconds": round(float(summary.get("runtime", {}).get("elapsed_seconds", 0.0)), 3),
                "rows_per_second": round(float(summary.get("runtime", {}).get("rows_per_second", 0.0)), 3),
                "unique_prompts_per_second": round(float(summary.get("runtime", {}).get("unique_prompts_per_second", 0.0)), 3),
                "top_row_languages": " | ".join(
                    f"{item.get('lang', '')}:{_format_pct(float(item.get('row_share', 0.0)))}"
                    for item in summary.get("top_languages_by_rows", [])[:5]
                ),
            }
        )
        all_overview_rows.extend(_iter_csv(spec["output_dir"] / "tables" / "language_overview.csv"))
        all_severity_rows.extend(_iter_csv(spec["output_dir"] / "tables" / "language_by_severity.csv"))
        all_label_rows.extend(_iter_csv(spec["output_dir"] / "tables" / "language_by_label.csv"))
        all_platform_rows.extend(_iter_csv(spec["output_dir"] / "tables" / "language_by_platform.csv"))
        all_uncertain_rows.extend(_iter_csv(spec["output_dir"] / "review" / "uncertain_or_unknown_prompts.csv"))
        all_non_english_review_rows.extend(_iter_csv(spec["output_dir"] / "review" / "top_non_english_prompts.csv"))

    comparison_root = Path(args.comparison_root)
    tables_dir = comparison_root / "tables"
    review_dir = comparison_root / "review"
    ensure_directory(comparison_root)
    ensure_directory(tables_dir)
    ensure_directory(review_dir)

    aggregated_overview = _aggregate_language_overview(all_overview_rows)
    aggregated_severity = _aggregate_severity_rows(all_severity_rows)
    aggregated_label = _aggregate_dimension_rows(all_label_rows, "label")
    aggregated_platform = _aggregate_dimension_rows(all_platform_rows, "target_platform")

    write_csv(
        tables_dir / "overview_by_crawl.csv",
        overview_by_crawl,
        [
            "crawl", "rows_seen", "rows_with_prompt", "rows_missing_prompt", "unique_normalized_prompts",
            "english_row_share", "non_english_row_share", "unknown_row_share",
            "english_unique_prompt_share", "non_english_unique_prompt_share", "unknown_unique_prompt_share",
            "elapsed_seconds", "rows_per_second", "unique_prompts_per_second", "top_row_languages",
        ],
    )
    write_csv(tables_dir / "language_overview_all_crawls.csv", all_overview_rows, list(all_overview_rows[0].keys()) if all_overview_rows else [])
    write_csv(tables_dir / "language_overview_aggregated.csv", aggregated_overview, list(aggregated_overview[0].keys()) if aggregated_overview else [])
    write_csv(tables_dir / "language_by_severity_all_crawls.csv", all_severity_rows, list(all_severity_rows[0].keys()) if all_severity_rows else [])
    write_csv(tables_dir / "language_by_severity_aggregated.csv", aggregated_severity, list(aggregated_severity[0].keys()) if aggregated_severity else [])
    write_csv(tables_dir / "language_by_label_all_crawls.csv", all_label_rows, list(all_label_rows[0].keys()) if all_label_rows else [])
    write_csv(tables_dir / "language_by_label_aggregated.csv", aggregated_label, list(aggregated_label[0].keys()) if aggregated_label else [])
    write_csv(tables_dir / "language_by_platform_all_crawls.csv", all_platform_rows, list(all_platform_rows[0].keys()) if all_platform_rows else [])
    write_csv(tables_dir / "language_by_platform_aggregated.csv", aggregated_platform, list(aggregated_platform[0].keys()) if aggregated_platform else [])
    write_csv(review_dir / "uncertain_or_unknown_prompts_all_crawls.csv", all_uncertain_rows, list(all_uncertain_rows[0].keys()) if all_uncertain_rows else [])
    write_csv(review_dir / "top_non_english_prompts_all_crawls.csv", all_non_english_review_rows, list(all_non_english_review_rows[0].keys()) if all_non_english_review_rows else [])

    report_path = Path(args.report_path)
    _render_markdown_report(
        run_summaries=run_summaries,
        aggregated_overview=aggregated_overview,
        aggregated_label_rows=aggregated_label,
        aggregated_platform_rows=aggregated_platform,
        report_path=report_path,
        comparison_root=comparison_root,
        config={"confidence_threshold": args.confidence_threshold, "min_alpha_chars": args.min_alpha_chars},
    )

    summary = {
        "generated_at_epoch": iso_now_epoch(),
        "comparison_root": str(comparison_root),
        "report_path": str(report_path),
        "configuration": {
            "confidence_threshold": args.confidence_threshold,
            "min_alpha_chars": args.min_alpha_chars,
            "review_limit": args.review_limit,
            "top_lang_limit": args.top_lang_limit,
        },
        "runs": overview_by_crawl,
        "files": {
            "overview_by_crawl_csv": str(tables_dir / "overview_by_crawl.csv"),
            "language_overview_all_crawls_csv": str(tables_dir / "language_overview_all_crawls.csv"),
            "language_overview_aggregated_csv": str(tables_dir / "language_overview_aggregated.csv"),
            "language_by_label_aggregated_csv": str(tables_dir / "language_by_label_aggregated.csv"),
            "language_by_platform_aggregated_csv": str(tables_dir / "language_by_platform_aggregated.csv"),
            "uncertain_or_unknown_prompts_all_crawls_csv": str(review_dir / "uncertain_or_unknown_prompts_all_crawls.csv"),
            "top_non_english_prompts_all_crawls_csv": str(review_dir / "top_non_english_prompts_all_crawls.csv"),
        },
    }
    write_json(comparison_root / "summary.json", summary)
    print(json.dumps({"comparison_root": str(comparison_root), "report_path": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
