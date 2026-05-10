#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from .risk_analysis_common import (
    iter_jsonl_rows,
    is_risky_row,
    normalize_string,
    row_severity,
    row_target_domain,
    row_target_platform,
)
from .source_url_analysis_common import ensure_directory, iso_now_epoch, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize medium/high-risk targets with counts, shares, and domain spread."
    )
    parser.add_argument("--input", required=True, help="Input classified JSONL or JSONL.GZ")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--crawl-name", default="", help="Optional crawl name override")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    ensure_directory(output_dir)
    ensure_directory(tables_dir)

    rows_seen = 0
    risky_rows = 0
    severity_counts: collections.Counter[str] = collections.Counter()
    target_platform_stats: dict[str, dict[str, Any]] = {}
    detected_crawl_name = ""

    for row in iter_jsonl_rows(input_path):
        rows_seen += 1
        if not detected_crawl_name:
            detected_crawl_name = normalize_string(row.get("crawl"))

        if not is_risky_row(row):
            continue

        platform = row_target_platform(row)
        target_domain = row_target_domain(row)
        severity = row_severity(row)

        risky_rows += 1
        severity_counts[severity] += 1

        aggregate = target_platform_stats.get(platform)
        if aggregate is None:
            aggregate = {
                "target_platform": platform,
                "rows": 0,
                "medium_rows": 0,
                "high_rows": 0,
                "target_domains": set(),
            }
            target_platform_stats[platform] = aggregate

        aggregate["rows"] += 1
        if severity == "high":
            aggregate["high_rows"] += 1
        else:
            aggregate["medium_rows"] += 1
        if target_domain:
            aggregate["target_domains"].add(target_domain)

    crawl_name = args.crawl_name.strip() or detected_crawl_name or input_path.stem

    target_rows: list[dict[str, Any]] = []
    for aggregate in target_platform_stats.values():
        target_rows.append(
            {
                "crawl": crawl_name,
                "target_platform": aggregate["target_platform"],
                "rows": aggregate["rows"],
                "share_of_risky_rows": round((aggregate["rows"] / risky_rows), 6) if risky_rows else 0.0,
                "high_rows": aggregate["high_rows"],
                "medium_rows": aggregate["medium_rows"],
                "unique_target_domains": len(aggregate["target_domains"]),
            }
        )

    target_rows.sort(key=lambda row: (-int(row["rows"]), row["target_platform"]))

    write_csv(
        tables_dir / "target_platform_risk.csv",
        target_rows,
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
        "input": str(input_path),
        "output_dir": str(output_dir),
        "crawl": crawl_name,
        "rows_seen": rows_seen,
        "risky_rows": risky_rows,
        "risky_share_of_all_rows": round((risky_rows / rows_seen), 6) if rows_seen else 0.0,
        "severity_counts": dict(severity_counts),
        "files": {
            "target_platform_risk_csv": str(tables_dir / "target_platform_risk.csv"),
        },
    }
    manifest = {
        "script": "analyze_target_risk.py",
        "version": 1,
        "crawl": crawl_name,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "tables": sorted(str(path) for path in tables_dir.iterdir()),
        "summary_json": str(output_dir / "summary.json"),
    }

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "manifest.json", manifest)

    print(
        json.dumps(
            {
                "crawl": crawl_name,
                "rows_seen": rows_seen,
                "risky_rows": risky_rows,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
