#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from source_url_analysis_common import ensure_directory, iso_now_epoch, write_csv, write_json


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


def _as_int(value: object) -> int:
    if value in (None, ""):
        return 0
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple single-crawl TARGET_URL analysis snapshots.")
    parser.add_argument("--input-dirs", nargs="+", required=True, help="One or more output directories from analyze_target_urls.py")
    parser.add_argument("--output-dir", required=True, help="Output directory for cross-crawl comparison artifacts")
    parser.add_argument("--top-n", type=int, default=200, help="Rows to keep in focused output tables")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figure_data"
    ensure_directory(output_dir)
    ensure_directory(figure_dir)

    crawls: list[str] = []
    crawl_order: dict[str, int] = {}
    summary_by_crawl: dict[str, dict[str, Any]] = {}
    target_domain_sets_by_crawl: dict[str, set[str]] = {}
    target_domain_persistence: dict[str, dict[str, Any]] = {}

    overview_by_crawl_rows: list[dict[str, Any]] = []
    target_platform_shift_rows: list[dict[str, Any]] = []
    label_shift_rows: list[dict[str, Any]] = []
    severity_shift_rows: list[dict[str, Any]] = []
    session_entry_shift_rows: list[dict[str, Any]] = []

    for index, input_dir_value in enumerate(args.input_dirs):
        input_dir = Path(input_dir_value)
        summary = _read_json(input_dir / "summary.json")
        crawl = str(summary.get("crawl") or input_dir.name)
        crawls.append(crawl)
        crawl_order[crawl] = index
        summary_by_crawl[crawl] = summary

        counts = summary.get("counts", {})
        quality = summary.get("quality", {})
        overview_by_crawl_rows.append(
            {
                "crawl": crawl,
                "rows_seen": quality.get("rows_seen", 0),
                "rows_analyzed": quality.get("rows_analyzed", 0),
                "rows_with_prompt_text": counts.get("rows_with_prompt_text", 0),
                "rows_suspicious": counts.get("rows_suspicious", 0),
                "rows_with_ioc_keywords": counts.get("rows_with_ioc_keywords", 0),
                "unique_target_platforms": counts.get("unique_target_platforms", 0),
                "unique_target_domains": counts.get("unique_target_domains", 0),
                "unique_source_domains": counts.get("unique_source_domains", 0),
                "unique_source_urls": counts.get("unique_source_urls", 0),
            }
        )

        for row in summary.get("distributions", {}).get("platform_distribution", []):
            target_platform_shift_rows.append(
                {
                    "crawl": crawl,
                    "target_platform": row.get("target_platform", ""),
                    "count": row.get("count", 0),
                    "share": row.get("share", 0.0),
                }
            )
        for row in summary.get("distributions", {}).get("label_distribution", []):
            label_shift_rows.append(
                {
                    "crawl": crawl,
                    "label": row.get("label", ""),
                    "count": row.get("count", 0),
                    "share": row.get("share", 0.0),
                }
            )
        for row in summary.get("distributions", {}).get("severity_distribution", []):
            severity_shift_rows.append(
                {
                    "crawl": crawl,
                    "severity": row.get("severity", ""),
                    "count": row.get("count", 0),
                    "share": row.get("share", 0.0),
                }
            )
        for row in summary.get("distributions", {}).get("session_entry_distribution", []):
            session_entry_shift_rows.append(
                {
                    "crawl": crawl,
                    "session_entry_reason": row.get("session_entry_reason", ""),
                    "count": row.get("count", 0),
                    "share": row.get("share", 0.0),
                }
            )

        target_domain_rows = list(_iter_csv(input_dir / "tables" / "target_domain_stats.csv"))
        target_domain_sets_by_crawl[crawl] = {
            row["target_domain"] for row in target_domain_rows if row.get("target_domain")
        }

        for row in target_domain_rows:
            target_domain = row.get("target_domain", "")
            if not target_domain:
                continue
            aggregate = target_domain_persistence.get(target_domain)
            if aggregate is None:
                aggregate = {
                    "target_domain": target_domain,
                    "target_platform": row.get("target_platform", ""),
                    "active_crawls": [],
                    "rows_total": 0,
                    "max_rows_single_crawl": 0,
                    "max_unique_source_domains_single_crawl": 0,
                    "max_unique_source_urls_single_crawl": 0,
                    "rows_by_crawl": {},
                }
                target_domain_persistence[target_domain] = aggregate
            aggregate["active_crawls"].append(crawl)
            rows = _as_int(row.get("rows"))
            aggregate["rows_total"] += rows
            aggregate["max_rows_single_crawl"] = max(aggregate["max_rows_single_crawl"], rows)
            aggregate["max_unique_source_domains_single_crawl"] = max(
                aggregate["max_unique_source_domains_single_crawl"],
                _as_int(row.get("unique_source_domains")),
            )
            aggregate["max_unique_source_urls_single_crawl"] = max(
                aggregate["max_unique_source_urls_single_crawl"],
                _as_int(row.get("unique_source_urls")),
            )
            aggregate["rows_by_crawl"][crawl] = rows

    new_vs_retained_target_domain_rows: list[dict[str, Any]] = []
    seen_target_domains: set[str] = set()
    for crawl in crawls:
        current = target_domain_sets_by_crawl[crawl]
        new_domains = current - seen_target_domains
        retained_domains = current & seen_target_domains
        new_vs_retained_target_domain_rows.append(
            {
                "crawl": crawl,
                "total_target_domains": len(current),
                "new_target_domains": len(new_domains),
                "retained_target_domains": len(retained_domains),
            }
        )
        seen_target_domains.update(current)

    target_domain_persistence_rows: list[dict[str, Any]] = []
    for aggregate in target_domain_persistence.values():
        active_crawls = sorted(aggregate["active_crawls"], key=lambda crawl: crawl_order[crawl])
        target_domain_persistence_rows.append(
            {
                "target_domain": aggregate["target_domain"],
                "target_platform": aggregate["target_platform"],
                "active_crawl_count": len(active_crawls),
                "first_seen_crawl": active_crawls[0],
                "last_seen_crawl": active_crawls[-1],
                "rows_total": aggregate["rows_total"],
                "max_rows_single_crawl": aggregate["max_rows_single_crawl"],
                "max_unique_source_domains_single_crawl": aggregate["max_unique_source_domains_single_crawl"],
                "max_unique_source_urls_single_crawl": aggregate["max_unique_source_urls_single_crawl"],
                "crawls": " | ".join(active_crawls),
                "rows_by_crawl_json": json.dumps(aggregate["rows_by_crawl"], ensure_ascii=False, sort_keys=True),
            }
        )
    target_domain_persistence_rows.sort(
        key=lambda row: (-int(row["active_crawl_count"]), -int(row["rows_total"]), row["target_domain"])
    )

    persistent_target_domain_rows = [
        row for row in target_domain_persistence_rows if int(row["active_crawl_count"]) > 1
    ]

    target_domain_overlap_rows: list[dict[str, Any]] = []
    for crawl_a in crawls:
        for crawl_b in crawls:
            set_a = target_domain_sets_by_crawl[crawl_a]
            set_b = target_domain_sets_by_crawl[crawl_b]
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            target_domain_overlap_rows.append(
                {
                    "crawl_a": crawl_a,
                    "crawl_b": crawl_b,
                    "intersection": intersection,
                    "union": union,
                    "jaccard": round((intersection / union), 6) if union else 0.0,
                }
            )

    write_csv(
        output_dir / "overview_by_crawl.csv",
        overview_by_crawl_rows,
        [
            "crawl",
            "rows_seen",
            "rows_analyzed",
            "rows_with_prompt_text",
            "rows_suspicious",
            "rows_with_ioc_keywords",
            "unique_target_platforms",
            "unique_target_domains",
            "unique_source_domains",
            "unique_source_urls",
        ],
    )
    write_csv(
        output_dir / "target_platform_shift_by_crawl.csv",
        target_platform_shift_rows,
        ["crawl", "target_platform", "count", "share"],
    )
    write_csv(
        output_dir / "label_shift_by_crawl.csv",
        label_shift_rows,
        ["crawl", "label", "count", "share"],
    )
    write_csv(
        output_dir / "severity_shift_by_crawl.csv",
        severity_shift_rows,
        ["crawl", "severity", "count", "share"],
    )
    write_csv(
        output_dir / "session_entry_shift_by_crawl.csv",
        session_entry_shift_rows,
        ["crawl", "session_entry_reason", "count", "share"],
    )
    write_csv(
        output_dir / "new_vs_retained_target_domains.csv",
        new_vs_retained_target_domain_rows,
        ["crawl", "total_target_domains", "new_target_domains", "retained_target_domains"],
    )
    write_csv(
        output_dir / "target_domain_persistence.csv",
        target_domain_persistence_rows,
        [
            "target_domain",
            "target_platform",
            "active_crawl_count",
            "first_seen_crawl",
            "last_seen_crawl",
            "rows_total",
            "max_rows_single_crawl",
            "max_unique_source_domains_single_crawl",
            "max_unique_source_urls_single_crawl",
            "crawls",
            "rows_by_crawl_json",
        ],
    )
    write_csv(
        output_dir / "persistent_target_domains.csv",
        persistent_target_domain_rows[: max(args.top_n, 1)],
        [
            "target_domain",
            "target_platform",
            "active_crawl_count",
            "first_seen_crawl",
            "last_seen_crawl",
            "rows_total",
            "max_rows_single_crawl",
            "max_unique_source_domains_single_crawl",
            "max_unique_source_urls_single_crawl",
            "crawls",
            "rows_by_crawl_json",
        ],
    )

    write_csv(
        figure_dir / "overview_by_crawl.csv",
        overview_by_crawl_rows,
        [
            "crawl",
            "rows_seen",
            "rows_analyzed",
            "rows_with_prompt_text",
            "rows_suspicious",
            "rows_with_ioc_keywords",
            "unique_target_platforms",
            "unique_target_domains",
            "unique_source_domains",
            "unique_source_urls",
        ],
    )
    write_csv(
        figure_dir / "target_platform_shift_by_crawl.csv",
        target_platform_shift_rows,
        ["crawl", "target_platform", "count", "share"],
    )
    write_csv(
        figure_dir / "label_shift_by_crawl.csv",
        label_shift_rows,
        ["crawl", "label", "count", "share"],
    )
    write_csv(
        figure_dir / "severity_shift_by_crawl.csv",
        severity_shift_rows,
        ["crawl", "severity", "count", "share"],
    )
    write_csv(
        figure_dir / "session_entry_shift_by_crawl.csv",
        session_entry_shift_rows,
        ["crawl", "session_entry_reason", "count", "share"],
    )
    write_csv(
        figure_dir / "new_vs_retained_target_domains.csv",
        new_vs_retained_target_domain_rows,
        ["crawl", "total_target_domains", "new_target_domains", "retained_target_domains"],
    )
    write_csv(
        figure_dir / "target_domain_overlap_matrix.csv",
        target_domain_overlap_rows,
        ["crawl_a", "crawl_b", "intersection", "union", "jaccard"],
    )

    summary = {
        "generated_at_epoch": iso_now_epoch(),
        "input_dirs": [str(Path(path)) for path in args.input_dirs],
        "output_dir": str(output_dir),
        "crawls": crawls,
        "counts": {
            "crawl_count": len(crawls),
            "persistent_target_domains": len(persistent_target_domain_rows),
        },
        "files": {
            "overview_by_crawl_csv": str(output_dir / "overview_by_crawl.csv"),
            "target_platform_shift_by_crawl_csv": str(output_dir / "target_platform_shift_by_crawl.csv"),
            "label_shift_by_crawl_csv": str(output_dir / "label_shift_by_crawl.csv"),
            "severity_shift_by_crawl_csv": str(output_dir / "severity_shift_by_crawl.csv"),
            "session_entry_shift_by_crawl_csv": str(output_dir / "session_entry_shift_by_crawl.csv"),
            "new_vs_retained_target_domains_csv": str(output_dir / "new_vs_retained_target_domains.csv"),
            "target_domain_persistence_csv": str(output_dir / "target_domain_persistence.csv"),
            "persistent_target_domains_csv": str(output_dir / "persistent_target_domains.csv"),
        },
    }
    manifest = {
        "script": "compare_target_url_snapshots.py",
        "version": 1,
        "crawls": crawls,
        "input_dirs": [str(Path(path)) for path in args.input_dirs],
        "output_dir": str(output_dir),
        "files": sorted(str(path) for path in output_dir.iterdir() if path.is_file()),
        "figure_data": sorted(str(path) for path in figure_dir.iterdir()),
        "summary_json": str(output_dir / "summary.json"),
    }

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "manifest.json", manifest)

    print(
        json.dumps(
            {
                "crawl_count": len(crawls),
                "persistent_target_domains": len(persistent_target_domain_rows),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())