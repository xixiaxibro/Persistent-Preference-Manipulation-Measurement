#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from source_url_analysis_common import ensure_directory, iso_now_epoch, write_csv, write_json

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_BASE = Path(os.environ.get("RUNS_BASE") or os.environ.get("ARTIFACT_ROOT") or PROJECT_ROOT / "runs")
DEFAULT_REPORT_DIR = Path(os.environ.get("REPORTS_DIR") or PROJECT_ROOT / "analysis_reports")

DEFAULT_AGGREGATE_CSV = str(
    DEFAULT_RUNS_BASE / "source_distribution_analysis" / "tables" / "root_domain_distribution_all_crawls.csv"
)
DEFAULT_MAPPING_CSV = str(PROJECT_ROOT / "top30_source_industry_mapping.csv")
DEFAULT_OUTPUT_DIR = str(DEFAULT_RUNS_BASE / "source_distribution_analysis" / "top30_refined")
DEFAULT_REPORT_PATH = str(DEFAULT_REPORT_DIR / "source_distribution_top30.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a refined top-30 source industry table.")
    parser.add_argument("--aggregate-csv", default=DEFAULT_AGGREGATE_CSV, help="Aggregate root-domain distribution CSV.")
    parser.add_argument("--mapping-csv", default=DEFAULT_MAPPING_CSV, help="Manual top-30 refined industry mapping CSV.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH, help="Markdown report path.")
    return parser.parse_args()


def _iter_csv(path: Path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


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


def main() -> int:
    args = parse_args()
    aggregate_csv = Path(args.aggregate_csv)
    mapping_csv = Path(args.mapping_csv)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)
    tables_dir = output_dir / "tables"
    ensure_directory(output_dir)
    ensure_directory(tables_dir)

    aggregate_rows = list(_iter_csv(aggregate_csv))
    mapping_rows = list(_iter_csv(mapping_csv))
    aggregate_by_domain = {str(row.get("root_domain", "")): row for row in aggregate_rows}
    mapping_by_domain = {str(row.get("root_domain", "")): row for row in mapping_rows}

    missing = [domain for domain in mapping_by_domain if domain not in aggregate_by_domain]
    if missing:
        raise ValueError(f"Mapped domains missing from aggregate CSV: {missing}")

    total_rows = sum(_to_int(row, "rows") for row in aggregate_rows)
    top30_rows: list[dict[str, Any]] = []
    for rank, aggregate_row in enumerate(aggregate_rows[:30], start=1):
        root_domain = str(aggregate_row.get("root_domain", ""))
        mapping = mapping_by_domain.get(root_domain)
        if mapping is None:
            raise ValueError(f"Top-30 domain missing refined mapping: {root_domain}")
        top30_rows.append(
            {
                "rank": rank,
                "root_domain": root_domain,
                "rows": _to_int(aggregate_row, "rows"),
                "row_share": _to_float(aggregate_row, "row_share"),
                "tranco_rank": aggregate_row.get("tranco_rank", ""),
                "tranco_bucket": aggregate_row.get("tranco_bucket", ""),
                "active_crawls": _to_int(aggregate_row, "active_crawls"),
                "coarse_industry": aggregate_row.get("industry_label", ""),
                "broad_sector": mapping.get("broad_sector", ""),
                "refined_industry": mapping.get("refined_industry", ""),
                "site_type": mapping.get("site_type", ""),
                "evidence_url": mapping.get("evidence_url", ""),
                "evidence_note": mapping.get("evidence_note", ""),
                "confidence": mapping.get("confidence", ""),
            }
        )

    top30_total_rows = sum(row["rows"] for row in top30_rows)
    sector_rows: dict[str, dict[str, Any]] = {}
    for row in top30_rows:
        sector = str(row.get("broad_sector", ""))
        bucket = sector_rows.setdefault(sector, {"broad_sector": sector, "row_count": 0, "domain_count": 0})
        bucket["row_count"] += _to_int(row, "rows")
        bucket["domain_count"] += 1
    sector_summary: list[dict[str, Any]] = []
    for bucket in sector_rows.values():
        sector_summary.append(
            {
                "broad_sector": bucket["broad_sector"],
                "row_count": bucket["row_count"],
                "row_share_within_top30": round(bucket["row_count"] / top30_total_rows, 6) if top30_total_rows else 0.0,
                "domain_count": bucket["domain_count"],
                "domain_share_within_top30": round(bucket["domain_count"] / len(top30_rows), 6) if top30_rows else 0.0,
            }
        )
    sector_summary.sort(key=lambda row: (-_to_int(row, "row_count"), row.get("broad_sector", "")))

    write_csv(
        tables_dir / "top30_source_industry_table.csv",
        top30_rows,
        ["rank", "root_domain", "rows", "row_share", "tranco_rank", "tranco_bucket", "active_crawls", "coarse_industry", "broad_sector", "refined_industry", "site_type", "evidence_url", "evidence_note", "confidence"],
    )
    write_csv(
        tables_dir / "top30_source_industry_sector_summary.csv",
        sector_summary,
        ["broad_sector", "row_count", "row_share_within_top30", "domain_count", "domain_share_within_top30"],
    )

    summary = {
        "generated_at_epoch": iso_now_epoch(),
        "aggregate_csv": str(aggregate_csv),
        "mapping_csv": str(mapping_csv),
        "top30_rows": len(top30_rows),
        "top30_risky_rows": top30_total_rows,
        "top30_share_of_all_risky_rows": round(top30_total_rows / total_rows, 6) if total_rows else 0.0,
        "files": {
            "top30_table_csv": str(tables_dir / "top30_source_industry_table.csv"),
            "sector_summary_csv": str(tables_dir / "top30_source_industry_sector_summary.csv"),
            "report_md": str(report_path),
        },
    }
    manifest = {
        "script": "build_top30_source_industry_table.py",
        "version": 1,
        "aggregate_csv": str(aggregate_csv),
        "mapping_csv": str(mapping_csv),
        "output_dir": str(output_dir),
        "tables": sorted(str(path) for path in tables_dir.iterdir()),
        "summary_json": str(output_dir / "summary.json"),
        "report_md": str(report_path),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "manifest.json", manifest)

    lines: list[str] = []
    lines.append("# Top-30 Source Industry Review")
    lines.append("")
    lines.append(f"Generated at epoch: {iso_now_epoch()}")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- The top 30 source roots account for `{top30_total_rows:,}` risky rows, or `{_format_pct(top30_total_rows / total_rows)}` of all risky rows.")
    lines.append("- Every top-30 root has a refined industry label and evidence URL in the output table.")
    lines.append("")
    lines.append("## Broad-Sector Summary")
    lines.append("")
    lines.append("| Broad sector | Risky rows | Share within top-30 | Domains |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in sector_summary:
        lines.append(
            f"| {row['broad_sector']} | {_to_int(row, 'row_count'):,} | {_format_pct(_to_float(row, 'row_share_within_top30'))} | {_to_int(row, 'domain_count')} |"
        )
    lines.append("")
    lines.append("## Top-30 Table")
    lines.append("")
    lines.append("| Rank | Root domain | Rows | Tranco | Broad sector | Refined industry |")
    lines.append("| --- | --- | ---: | --- | --- | --- |")
    for row in top30_rows:
        lines.append(
            f"| {_to_int(row, 'rank')} | {row['root_domain']} | {_to_int(row, 'rows'):,} | {row['tranco_bucket']} | {row['broad_sector']} | {row['refined_industry']} |"
        )
    lines.append("")
    lines.append("## Output Directory")
    lines.append("")
    lines.append(f"- `{output_dir}`")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "report_path": str(report_path), "top30_share": round(top30_total_rows / total_rows, 6) if total_rows else 0.0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
