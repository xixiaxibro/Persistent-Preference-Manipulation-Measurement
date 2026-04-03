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


def _as_float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple single-crawl SOURCE_URL analysis snapshots.")
    parser.add_argument("--input-dirs", nargs="+", required=True, help="One or more output directories from analyze_source_urls.py")
    parser.add_argument("--output-dir", required=True, help="Output directory for cross-crawl comparison artifacts")
    parser.add_argument("--top-n", type=int, default=200, help="Rows to keep in focused output tables")
    parser.add_argument("--skip-source-url-persistence", action="store_true", help="Skip the heaviest persistence table for source_url")
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
    root_sets_by_crawl: dict[str, set[str]] = {}

    root_persistence: dict[str, dict[str, Any]] = {}
    source_domain_persistence: dict[str, dict[str, Any]] = {}
    source_url_persistence: dict[str, dict[str, Any]] = {}

    tranco_bucket_overview_rows: list[dict[str, Any]] = []

    for index, input_dir_value in enumerate(args.input_dirs):
        input_dir = Path(input_dir_value)
        summary = _read_json(input_dir / "summary.json")
        crawl = str(summary.get("crawl") or input_dir.name)
        crawls.append(crawl)
        crawl_order[crawl] = index
        summary_by_crawl[crawl] = summary

        root_domain_rows = list(_iter_csv(input_dir / "tables" / "root_domain_stats.csv"))
        source_domain_rows = list(_iter_csv(input_dir / "tables" / "source_domain_stats.csv"))
        root_sets_by_crawl[crawl] = {row["root_domain"] for row in root_domain_rows if row.get("root_domain")}

        tranco_bucket_overview_rows.extend(_iter_csv(input_dir / "tables" / "tranco_bucket_summary.csv"))

        for row in root_domain_rows:
            root_domain = row.get("root_domain", "")
            if not root_domain:
                continue
            aggregate = root_persistence.get(root_domain)
            if aggregate is None:
                aggregate = {
                    "root_domain": root_domain,
                    "tranco_rank": row.get("tranco_rank", ""),
                    "tranco_bucket": row.get("tranco_bucket", "unranked"),
                    "active_crawls": [],
                    "rows_total": 0,
                    "max_rows_single_crawl": 0,
                    "max_unique_source_urls_single_crawl": 0,
                    "rows_by_crawl": {},
                }
                root_persistence[root_domain] = aggregate
            aggregate["active_crawls"].append(crawl)
            rows = _as_int(row.get("rows"))
            unique_source_urls = _as_int(row.get("unique_source_urls"))
            aggregate["rows_total"] += rows
            aggregate["max_rows_single_crawl"] = max(aggregate["max_rows_single_crawl"], rows)
            aggregate["max_unique_source_urls_single_crawl"] = max(
                aggregate["max_unique_source_urls_single_crawl"], unique_source_urls
            )
            aggregate["rows_by_crawl"][crawl] = rows

        for row in source_domain_rows:
            source_domain = row.get("source_domain", "")
            if not source_domain:
                continue
            aggregate = source_domain_persistence.get(source_domain)
            if aggregate is None:
                aggregate = {
                    "source_domain": source_domain,
                    "root_domain": row.get("root_domain", ""),
                    "tranco_rank": row.get("tranco_rank", ""),
                    "tranco_bucket": row.get("tranco_bucket", "unranked"),
                    "active_crawls": [],
                    "rows_total": 0,
                    "max_rows_single_crawl": 0,
                    "rows_by_crawl": {},
                }
                source_domain_persistence[source_domain] = aggregate
            aggregate["active_crawls"].append(crawl)
            rows = _as_int(row.get("rows"))
            aggregate["rows_total"] += rows
            aggregate["max_rows_single_crawl"] = max(aggregate["max_rows_single_crawl"], rows)
            aggregate["rows_by_crawl"][crawl] = rows

        if not args.skip_source_url_persistence:
            for row in _iter_csv(input_dir / "tables" / "source_url_stats.csv"):
                source_url = row.get("source_url", "")
                if not source_url:
                    continue
                aggregate = source_url_persistence.get(source_url)
                if aggregate is None:
                    aggregate = {
                        "source_url": source_url,
                        "source_domain": row.get("source_domain", ""),
                        "root_domain": row.get("root_domain", ""),
                        "tranco_rank": row.get("tranco_rank", ""),
                        "tranco_bucket": row.get("tranco_bucket", "unranked"),
                        "active_crawls": [],
                        "rows_total": 0,
                        "max_rows_single_crawl": 0,
                        "max_unique_target_platforms_single_crawl": 0,
                        "rows_by_crawl": {},
                    }
                    source_url_persistence[source_url] = aggregate
                aggregate["active_crawls"].append(crawl)
                rows = _as_int(row.get("rows"))
                platforms = _as_int(row.get("unique_target_platforms"))
                aggregate["rows_total"] += rows
                aggregate["max_rows_single_crawl"] = max(aggregate["max_rows_single_crawl"], rows)
                aggregate["max_unique_target_platforms_single_crawl"] = max(
                    aggregate["max_unique_target_platforms_single_crawl"], platforms
                )
                aggregate["rows_by_crawl"][crawl] = rows

    overview_by_crawl_rows: list[dict[str, Any]] = []
    label_shift_rows: list[dict[str, Any]] = []
    platform_shift_rows: list[dict[str, Any]] = []

    for crawl in crawls:
        summary = summary_by_crawl[crawl]
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
                "unique_source_urls": counts.get("unique_source_urls", 0),
                "unique_source_domains": counts.get("unique_source_domains", 0),
                "unique_root_domains": counts.get("unique_root_domains", 0),
                "multi_platform_source_urls": counts.get("multi_platform_source_urls", 0),
                "ranked_root_domains": counts.get("ranked_root_domains", 0),
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

        for row in summary.get("distributions", {}).get("platform_distribution", []):
            platform_shift_rows.append(
                {
                    "crawl": crawl,
                    "target_platform": row.get("target_platform", ""),
                    "count": row.get("count", 0),
                    "share": row.get("share", 0.0),
                }
            )

    new_vs_retained_root_rows: list[dict[str, Any]] = []
    seen_root_domains: set[str] = set()
    for crawl in crawls:
        current = root_sets_by_crawl[crawl]
        new_domains = current - seen_root_domains
        retained_domains = current & seen_root_domains
        new_vs_retained_root_rows.append(
            {
                "crawl": crawl,
                "total_root_domains": len(current),
                "new_root_domains": len(new_domains),
                "retained_root_domains": len(retained_domains),
            }
        )
        seen_root_domains.update(current)

    root_persistence_rows: list[dict[str, Any]] = []
    for aggregate in root_persistence.values():
        active_crawls = sorted(aggregate["active_crawls"], key=lambda crawl: crawl_order[crawl])
        root_persistence_rows.append(
            {
                "root_domain": aggregate["root_domain"],
                "tranco_rank": aggregate["tranco_rank"],
                "tranco_bucket": aggregate["tranco_bucket"],
                "active_crawl_count": len(active_crawls),
                "first_seen_crawl": active_crawls[0],
                "last_seen_crawl": active_crawls[-1],
                "rows_total": aggregate["rows_total"],
                "max_rows_single_crawl": aggregate["max_rows_single_crawl"],
                "max_unique_source_urls_single_crawl": aggregate["max_unique_source_urls_single_crawl"],
                "crawls": " | ".join(active_crawls),
                "rows_by_crawl_json": json.dumps(aggregate["rows_by_crawl"], ensure_ascii=False, sort_keys=True),
            }
        )
    root_persistence_rows.sort(
        key=lambda row: (-int(row["active_crawl_count"]), -int(row["rows_total"]), row["root_domain"])
    )

    source_domain_persistence_rows: list[dict[str, Any]] = []
    for aggregate in source_domain_persistence.values():
        active_crawls = sorted(aggregate["active_crawls"], key=lambda crawl: crawl_order[crawl])
        source_domain_persistence_rows.append(
            {
                "source_domain": aggregate["source_domain"],
                "root_domain": aggregate["root_domain"],
                "tranco_rank": aggregate["tranco_rank"],
                "tranco_bucket": aggregate["tranco_bucket"],
                "active_crawl_count": len(active_crawls),
                "first_seen_crawl": active_crawls[0],
                "last_seen_crawl": active_crawls[-1],
                "rows_total": aggregate["rows_total"],
                "max_rows_single_crawl": aggregate["max_rows_single_crawl"],
                "crawls": " | ".join(active_crawls),
                "rows_by_crawl_json": json.dumps(aggregate["rows_by_crawl"], ensure_ascii=False, sort_keys=True),
            }
        )
    source_domain_persistence_rows.sort(
        key=lambda row: (-int(row["active_crawl_count"]), -int(row["rows_total"]), row["source_domain"])
    )

    source_url_persistence_rows: list[dict[str, Any]] = []
    if not args.skip_source_url_persistence:
        for aggregate in source_url_persistence.values():
            active_crawls = sorted(aggregate["active_crawls"], key=lambda crawl: crawl_order[crawl])
            source_url_persistence_rows.append(
                {
                    "source_url": aggregate["source_url"],
                    "source_domain": aggregate["source_domain"],
                    "root_domain": aggregate["root_domain"],
                    "tranco_rank": aggregate["tranco_rank"],
                    "tranco_bucket": aggregate["tranco_bucket"],
                    "active_crawl_count": len(active_crawls),
                    "first_seen_crawl": active_crawls[0],
                    "last_seen_crawl": active_crawls[-1],
                    "rows_total": aggregate["rows_total"],
                    "max_rows_single_crawl": aggregate["max_rows_single_crawl"],
                    "max_unique_target_platforms_single_crawl": aggregate["max_unique_target_platforms_single_crawl"],
                    "crawls": " | ".join(active_crawls),
                    "rows_by_crawl_json": json.dumps(aggregate["rows_by_crawl"], ensure_ascii=False, sort_keys=True),
                }
            )
        source_url_persistence_rows.sort(
            key=lambda row: (-int(row["active_crawl_count"]), -int(row["rows_total"]), row["source_url"])
        )

    persistent_popular_domain_rows = [
        row
        for row in root_persistence_rows
        if row["tranco_rank"] not in ("", None) and int(row["active_crawl_count"]) > 1
    ]
    persistent_popular_domain_rows.sort(
        key=lambda row: (-int(row["active_crawl_count"]), -int(row["rows_total"]), int(row["tranco_rank"]), row["root_domain"])
    )

    root_overlap_rows: list[dict[str, Any]] = []
    for crawl_a in crawls:
        for crawl_b in crawls:
            set_a = root_sets_by_crawl[crawl_a]
            set_b = root_sets_by_crawl[crawl_b]
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            root_overlap_rows.append(
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
            "crawl", "rows_seen", "rows_analyzed", "rows_with_prompt_text", "rows_suspicious",
            "rows_with_ioc_keywords", "unique_source_urls", "unique_source_domains", "unique_root_domains",
            "multi_platform_source_urls", "ranked_root_domains",
        ],
    )
    write_csv(
        output_dir / "tranco_bucket_overview_by_crawl.csv",
        tranco_bucket_overview_rows,
        [
            "crawl", "tranco_bucket", "rows", "rows_with_prompt_text", "suspicious_rows", "high_rows",
            "medium_rows", "low_rows", "ioc_rows", "unique_root_domains", "unique_source_domains",
            "unique_source_urls", "unique_target_platforms",
        ],
    )
    write_csv(
        output_dir / "label_shift_by_crawl.csv",
        label_shift_rows,
        ["crawl", "label", "count", "share"],
    )
    write_csv(
        output_dir / "platform_shift_by_crawl.csv",
        platform_shift_rows,
        ["crawl", "target_platform", "count", "share"],
    )
    write_csv(
        output_dir / "new_vs_retained_root_domains.csv",
        new_vs_retained_root_rows,
        ["crawl", "total_root_domains", "new_root_domains", "retained_root_domains"],
    )
    write_csv(
        output_dir / "root_domain_persistence.csv",
        root_persistence_rows,
        [
            "root_domain", "tranco_rank", "tranco_bucket", "active_crawl_count", "first_seen_crawl",
            "last_seen_crawl", "rows_total", "max_rows_single_crawl", "max_unique_source_urls_single_crawl",
            "crawls", "rows_by_crawl_json",
        ],
    )
    write_csv(
        output_dir / "source_domain_persistence.csv",
        source_domain_persistence_rows,
        [
            "source_domain", "root_domain", "tranco_rank", "tranco_bucket", "active_crawl_count",
            "first_seen_crawl", "last_seen_crawl", "rows_total", "max_rows_single_crawl", "crawls",
            "rows_by_crawl_json",
        ],
    )
    if not args.skip_source_url_persistence:
        write_csv(
            output_dir / "source_url_persistence.csv",
            source_url_persistence_rows,
            [
                "source_url", "source_domain", "root_domain", "tranco_rank", "tranco_bucket",
                "active_crawl_count", "first_seen_crawl", "last_seen_crawl", "rows_total",
                "max_rows_single_crawl", "max_unique_target_platforms_single_crawl", "crawls", "rows_by_crawl_json",
            ],
        )
    write_csv(
        output_dir / "persistent_popular_domains.csv",
        persistent_popular_domain_rows[: max(args.top_n, 1)],
        [
            "root_domain", "tranco_rank", "tranco_bucket", "active_crawl_count", "first_seen_crawl",
            "last_seen_crawl", "rows_total", "max_rows_single_crawl", "max_unique_source_urls_single_crawl",
            "crawls", "rows_by_crawl_json",
        ],
    )

    write_csv(
        figure_dir / "overview_by_crawl.csv",
        overview_by_crawl_rows,
        [
            "crawl", "rows_seen", "rows_analyzed", "rows_with_prompt_text", "rows_suspicious",
            "rows_with_ioc_keywords", "unique_source_urls", "unique_source_domains", "unique_root_domains",
            "multi_platform_source_urls", "ranked_root_domains",
        ],
    )
    write_csv(
        figure_dir / "tranco_bucket_rows_by_crawl.csv",
        [
            {"crawl": row["crawl"], "tranco_bucket": row["tranco_bucket"], "rows": row["rows"]}
            for row in tranco_bucket_overview_rows
        ],
        ["crawl", "tranco_bucket", "rows"],
    )
    write_csv(
        figure_dir / "tranco_bucket_unique_root_domains_by_crawl.csv",
        [
            {
                "crawl": row["crawl"],
                "tranco_bucket": row["tranco_bucket"],
                "unique_root_domains": row["unique_root_domains"],
            }
            for row in tranco_bucket_overview_rows
        ],
        ["crawl", "tranco_bucket", "unique_root_domains"],
    )
    write_csv(
        figure_dir / "label_shift_by_crawl.csv",
        label_shift_rows,
        ["crawl", "label", "count", "share"],
    )
    write_csv(
        figure_dir / "platform_shift_by_crawl.csv",
        platform_shift_rows,
        ["crawl", "target_platform", "count", "share"],
    )
    write_csv(
        figure_dir / "new_vs_retained_root_domains.csv",
        new_vs_retained_root_rows,
        ["crawl", "total_root_domains", "new_root_domains", "retained_root_domains"],
    )
    write_csv(
        figure_dir / "root_domain_overlap_matrix.csv",
        root_overlap_rows,
        ["crawl_a", "crawl_b", "intersection", "union", "jaccard"],
    )

    summary = {
        "generated_at_epoch": iso_now_epoch(),
        "input_dirs": [str(Path(path)) for path in args.input_dirs],
        "output_dir": str(output_dir),
        "crawls": crawls,
        "skip_source_url_persistence": args.skip_source_url_persistence,
        "counts": {
            "crawl_count": len(crawls),
            "persistent_root_domains": sum(1 for row in root_persistence_rows if int(row["active_crawl_count"]) > 1),
            "persistent_source_domains": sum(1 for row in source_domain_persistence_rows if int(row["active_crawl_count"]) > 1),
            "persistent_source_urls": 0 if args.skip_source_url_persistence else sum(1 for row in source_url_persistence_rows if int(row["active_crawl_count"]) > 1),
            "persistent_popular_domains": len(persistent_popular_domain_rows),
        },
        "files": {
            "overview_by_crawl_csv": str(output_dir / "overview_by_crawl.csv"),
            "tranco_bucket_overview_by_crawl_csv": str(output_dir / "tranco_bucket_overview_by_crawl.csv"),
            "label_shift_by_crawl_csv": str(output_dir / "label_shift_by_crawl.csv"),
            "platform_shift_by_crawl_csv": str(output_dir / "platform_shift_by_crawl.csv"),
            "new_vs_retained_root_domains_csv": str(output_dir / "new_vs_retained_root_domains.csv"),
            "root_domain_persistence_csv": str(output_dir / "root_domain_persistence.csv"),
            "source_domain_persistence_csv": str(output_dir / "source_domain_persistence.csv"),
            "source_url_persistence_csv": "" if args.skip_source_url_persistence else str(output_dir / "source_url_persistence.csv"),
            "persistent_popular_domains_csv": str(output_dir / "persistent_popular_domains.csv"),
        },
    }
    manifest = {
        "script": "compare_source_url_snapshots.py",
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
                "persistent_root_domains": summary["counts"]["persistent_root_domains"],
                "persistent_source_domains": summary["counts"]["persistent_source_domains"],
                "persistent_popular_domains": summary["counts"]["persistent_popular_domains"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())