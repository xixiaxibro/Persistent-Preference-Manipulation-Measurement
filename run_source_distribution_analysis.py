#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_BASE = Path(os.environ.get("RUNS_BASE") or os.environ.get("ARTIFACT_ROOT") or PROJECT_ROOT / "runs")
DEFAULT_REPORT_DIR = Path(os.environ.get("REPORTS_DIR") or PROJECT_ROOT / "analysis_reports")

DEFAULT_SOURCE_RISK_CSV = str(
    DEFAULT_RUNS_BASE / "simplified_risk_analysis" / "source_risk" / "source_domain_risk_all_crawls.csv"
)
DEFAULT_OUTPUT_DIR = str(DEFAULT_RUNS_BASE / "source_distribution_analysis")
DEFAULT_REPORT_PATH = str(DEFAULT_REPORT_DIR / "source_distribution_analysis.md")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run source-site distribution analysis on cross-crawl source-risk outputs.")
    parser.add_argument("--script-dir", default=str(script_dir), help=f"Repository script directory (default: {script_dir}).")
    parser.add_argument("--source-risk-csv", default=DEFAULT_SOURCE_RISK_CSV, help="Cross-crawl source risk CSV input.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH, help="Markdown report path.")
    parser.add_argument("--review-limit", type=int, default=300, help="Review CSV row limit.")
    parser.add_argument("--fetch-timeout", type=float, default=10.0, help="Per-request timeout in seconds.")
    parser.add_argument("--max-workers", type=int, default=16, help="Concurrent homepage fetch workers.")
    parser.add_argument("--max-bytes", type=int, default=131072, help="Maximum response bytes to inspect per homepage.")
    parser.add_argument("--refresh-profiles", action="store_true", help="Ignore cached domain profiles and refetch.")
    parser.add_argument("--max-roots", type=int, default=0, help="Optional cap on unique root domains for testing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(args.script_dir) / "analyze_source_distribution.py"
    command = [
        sys.executable,
        str(script_path),
        "--source-risk-csv",
        args.source_risk_csv,
        "--output-dir",
        args.output_dir,
        "--report-path",
        args.report_path,
        "--review-limit",
        str(args.review_limit),
        "--fetch-timeout",
        str(args.fetch_timeout),
        "--max-workers",
        str(args.max_workers),
        "--max-bytes",
        str(args.max_bytes),
    ]
    if args.refresh_profiles:
        command.append("--refresh-profiles")
    if args.max_roots > 0:
        command.extend(["--max-roots", str(args.max_roots)])
    print(json.dumps({"stage": "exec", "command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
