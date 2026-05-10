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

DEFAULT_RUNS: tuple[tuple[str, str], ...] = (
    ("CC-MAIN-2025-51", str(DEFAULT_RUNS_BASE / "collect_ccmain2025_51")),
    ("CC-MAIN-2026-04", str(DEFAULT_RUNS_BASE / "collect_ccmain2026_04")),
    ("CC-MAIN-2026-08", str(DEFAULT_RUNS_BASE / "collect_ccmain2026_08")),
    ("CC-MAIN-2026-12", str(DEFAULT_RUNS_BASE / "collect_ccmain2026_12")),
)
DEFAULT_COMPARISON_ROOT = str(DEFAULT_RUNS_BASE / "simplified_risk_analysis")


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
                "source_output_dir": run_root / args.source_output_dirname,
                "target_output_dir": run_root / args.target_output_dirname,
            }
        )
    return specs


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run simplified medium/high-risk source and target analysis across completed Stage 02 runs."
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
        "--source-output-dirname",
        default="03_source_url_analysis",
        help="Per-crawl simplified source output directory name.",
    )
    parser.add_argument(
        "--target-output-dirname",
        default="03b_target_analysis",
        help="Per-crawl simplified target output directory name.",
    )
    parser.add_argument("--source-top-n", type=int, default=200, help="Top-N source domains for review output.")
    parser.add_argument(
        "--tranco-cache",
        default=str(script_dir / "tranco_top1m.csv"),
        help="Tranco cache path for source analysis.",
    )
    parser.add_argument(
        "--tranco-mode",
        choices=("fixed", "download-if-missing"),
        default="fixed",
        help="Tranco loading mode for source analysis.",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow non-empty output directories to exist before execution.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(args.script_dir)
    if not script_dir.is_dir():
        raise FileNotFoundError(f"Script directory does not exist: {script_dir}")

    source_script = script_dir / "analyze_source_risk.py"
    target_script = script_dir / "analyze_target_risk.py"
    run_specs = _resolve_run_specs(args)

    source_overview_rows: list[dict[str, Any]] = []
    target_overview_rows: list[dict[str, Any]] = []
    combined_source_rows: list[dict[str, Any]] = []
    combined_target_rows: list[dict[str, Any]] = []

    for spec in run_specs:
        _ensure_output_dir_available(spec["source_output_dir"], allow_existing_output=args.allow_existing_output)
        _ensure_output_dir_available(spec["target_output_dir"], allow_existing_output=args.allow_existing_output)

        _run_command(
            [
                sys.executable,
                str(source_script),
                "--input",
                str(spec["input"]),
                "--output-dir",
                str(spec["source_output_dir"]),
                "--crawl-name",
                str(spec["crawl"]),
                "--top-n",
                str(args.source_top_n),
                "--tranco-cache",
                args.tranco_cache,
                "--tranco-mode",
                args.tranco_mode,
            ]
        )
        _run_command(
            [
                sys.executable,
                str(target_script),
                "--input",
                str(spec["input"]),
                "--output-dir",
                str(spec["target_output_dir"]),
                "--crawl-name",
                str(spec["crawl"]),
            ]
        )

        source_summary = _read_json(spec["source_output_dir"] / "summary.json")
        target_summary = _read_json(spec["target_output_dir"] / "summary.json")
        source_overview_rows.append(
            {
                "crawl": spec["crawl"],
                "rows_seen": source_summary.get("rows_seen", 0),
                "risky_rows": source_summary.get("risky_rows", 0),
                "risky_share_of_all_rows": source_summary.get("risky_share_of_all_rows", 0.0),
                "unique_risky_source_domains": source_summary.get("unique_risky_source_domains", 0),
                "high_rows": source_summary.get("severity_counts", {}).get("high", 0),
                "medium_rows": source_summary.get("severity_counts", {}).get("medium", 0),
            }
        )
        target_overview_rows.append(
            {
                "crawl": spec["crawl"],
                "rows_seen": target_summary.get("rows_seen", 0),
                "risky_rows": target_summary.get("risky_rows", 0),
                "risky_share_of_all_rows": target_summary.get("risky_share_of_all_rows", 0.0),
                "high_rows": target_summary.get("severity_counts", {}).get("high", 0),
                "medium_rows": target_summary.get("severity_counts", {}).get("medium", 0),
            }
        )
        combined_source_rows.extend(_iter_csv(spec["source_output_dir"] / "tables" / "source_domain_risk.csv"))
        combined_target_rows.extend(_iter_csv(spec["target_output_dir"] / "tables" / "target_platform_risk.csv"))

    comparison_root = Path(args.comparison_root)
    ensure_directory(comparison_root)
    source_comparison_dir = comparison_root / "source_risk"
    target_comparison_dir = comparison_root / "target_risk"
    ensure_directory(source_comparison_dir)
    ensure_directory(target_comparison_dir)

    combined_source_rows.sort(
        key=lambda row: (row.get("crawl", ""), -int(row.get("rows", 0)), row.get("source_domain", ""))
    )
    combined_target_rows.sort(
        key=lambda row: (row.get("crawl", ""), -int(row.get("rows", 0)), row.get("target_platform", ""))
    )

    write_csv(
        source_comparison_dir / "overview_by_crawl.csv",
        source_overview_rows,
        ["crawl", "rows_seen", "risky_rows", "risky_share_of_all_rows", "unique_risky_source_domains", "high_rows", "medium_rows"],
    )
    write_csv(
        source_comparison_dir / "source_domain_risk_all_crawls.csv",
        combined_source_rows,
        [
            "crawl",
            "source_domain",
            "root_domain",
            "tranco_rank",
            "tranco_matched_domain",
            "tranco_bucket",
            "tranco_match_type",
            "rows",
            "share_of_risky_rows",
            "high_rows",
            "medium_rows",
            "unique_target_platforms",
            "target_platforms",
            "top_platform",
            "platform_features",
            "platform_distribution_json",
        ],
    )
    write_csv(
        target_comparison_dir / "overview_by_crawl.csv",
        target_overview_rows,
        ["crawl", "rows_seen", "risky_rows", "risky_share_of_all_rows", "high_rows", "medium_rows"],
    )
    write_csv(
        target_comparison_dir / "target_platform_risk_all_crawls.csv",
        combined_target_rows,
        [
            "crawl",
            "target_platform",
            "rows",
            "share_of_risky_rows",
            "high_rows",
            "medium_rows",
            "unique_target_domains",
        ],
    )

    summary = {
        "generated_at_epoch": iso_now_epoch(),
        "script": "run_simplified_risk_analysis.py",
        "crawl_count": len(run_specs),
        "comparison_root": str(comparison_root),
        "runs": [
            {
                "crawl": spec["crawl"],
                "run_root": str(spec["run_root"]),
                "input": str(spec["input"]),
                "source_output_dir": str(spec["source_output_dir"]),
                "target_output_dir": str(spec["target_output_dir"]),
            }
            for spec in run_specs
        ],
        "files": {
            "source_overview_by_crawl_csv": str(source_comparison_dir / "overview_by_crawl.csv"),
            "source_domain_risk_all_crawls_csv": str(source_comparison_dir / "source_domain_risk_all_crawls.csv"),
            "target_overview_by_crawl_csv": str(target_comparison_dir / "overview_by_crawl.csv"),
            "target_platform_risk_all_crawls_csv": str(target_comparison_dir / "target_platform_risk_all_crawls.csv"),
        },
    }
    write_json(comparison_root / "summary.json", summary)

    print(
        json.dumps(
            {
                "crawl_count": len(run_specs),
                "comparison_root": str(comparison_root),
                "source_output_dirname": args.source_output_dirname,
                "target_output_dirname": args.target_output_dirname,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
