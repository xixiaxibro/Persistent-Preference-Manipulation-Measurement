#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from risk_analysis_common import iter_jsonl_rows, is_risky_row, normalize_string, row_severity, row_source_domain, row_target_platform
from source_url_analysis_common import (
    ensure_directory,
    extract_root_domain,
    iso_now_epoch,
    load_tranco_ranking,
    lookup_tranco,
    make_domain_extractor,
    write_csv,
    write_json,
)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _top_items(counter: collections.Counter[str], limit: int = 3) -> str:
    items = counter.most_common(limit)
    return " | ".join(f"{name}:{count}" for name, count in items)


def _select_fields(rows: list[dict[str, Any]], fieldnames: list[str]) -> list[dict[str, Any]]:
    return [{field: row.get(field, "") for field in fieldnames} for row in rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize medium/high-risk source domains with Tranco rank and target-platform features."
    )
    parser.add_argument("--input", required=True, help="Input classified JSONL or JSONL.GZ")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--crawl-name", default="", help="Optional crawl name override")
    parser.add_argument("--top-n", type=int, default=200, help="Top-N review rows")
    parser.add_argument("--tranco-csv", default="", help="Optional fixed Tranco CSV file")
    parser.add_argument("--tranco-cache", default="tranco_top1m.csv", help="Tranco cache path")
    parser.add_argument(
        "--tranco-mode",
        choices=("fixed", "download-if-missing"),
        default="fixed",
        help="Use only local Tranco data or download when missing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    review_dir = output_dir / "review"
    ensure_directory(output_dir)
    ensure_directory(tables_dir)
    ensure_directory(review_dir)

    extractor = make_domain_extractor()
    tranco_csv = Path(args.tranco_csv) if args.tranco_csv else None
    tranco_cache = Path(args.tranco_cache) if args.tranco_cache else None
    tranco_ranking, tranco_source = load_tranco_ranking(
        tranco_csv=tranco_csv,
        tranco_cache=tranco_cache,
        mode=args.tranco_mode,
    )

    rows_seen = 0
    risky_rows = 0
    severity_counts: collections.Counter[str] = collections.Counter()
    source_domain_stats: dict[str, dict[str, Any]] = {}
    detected_crawl_name = ""

    for row in iter_jsonl_rows(input_path):
        rows_seen += 1
        if not detected_crawl_name:
            detected_crawl_name = normalize_string(row.get("crawl"))

        if not is_risky_row(row):
            continue

        source_domain = row_source_domain(row)
        if not source_domain:
            continue

        severity = row_severity(row)
        platform = row_target_platform(row)
        risky_rows += 1
        severity_counts[severity] += 1

        root_domain = extract_root_domain(source_domain, extractor) or source_domain
        tranco_match = lookup_tranco(root_domain, tranco_ranking)

        aggregate = source_domain_stats.get(source_domain)
        if aggregate is None:
            aggregate = {
                "source_domain": source_domain,
                "root_domain": root_domain,
                "tranco_rank": tranco_match.rank,
                "tranco_matched_domain": tranco_match.matched_domain,
                "tranco_bucket": tranco_match.bucket,
                "tranco_match_type": tranco_match.match_type,
                "rows": 0,
                "medium_rows": 0,
                "high_rows": 0,
                "target_platforms": set(),
                "target_platform_counts": collections.Counter(),
            }
            source_domain_stats[source_domain] = aggregate

        aggregate["rows"] += 1
        if severity == "high":
            aggregate["high_rows"] += 1
        else:
            aggregate["medium_rows"] += 1
        aggregate["target_platforms"].add(platform)
        aggregate["target_platform_counts"][platform] += 1

    crawl_name = args.crawl_name.strip() or detected_crawl_name or input_path.stem

    source_rows: list[dict[str, Any]] = []
    for aggregate in source_domain_stats.values():
        source_rows.append(
            {
                "crawl": crawl_name,
                "source_domain": aggregate["source_domain"],
                "root_domain": aggregate["root_domain"],
                "tranco_rank": aggregate["tranco_rank"] or "",
                "tranco_matched_domain": aggregate["tranco_matched_domain"],
                "tranco_bucket": aggregate["tranco_bucket"],
                "tranco_match_type": aggregate["tranco_match_type"],
                "rows": aggregate["rows"],
                "share_of_risky_rows": round((aggregate["rows"] / risky_rows), 6) if risky_rows else 0.0,
                "high_rows": aggregate["high_rows"],
                "medium_rows": aggregate["medium_rows"],
                "unique_target_platforms": len(aggregate["target_platforms"]),
                "target_platforms": " | ".join(sorted(aggregate["target_platforms"])),
                "top_platform": aggregate["target_platform_counts"].most_common(1)[0][0] if aggregate["target_platform_counts"] else "",
                "platform_features": _top_items(aggregate["target_platform_counts"]),
                "platform_distribution_json": _json_dumps(dict(aggregate["target_platform_counts"].most_common())),
            }
        )

    source_rows.sort(
        key=lambda row: (-int(row["rows"]), -int(row["unique_target_platforms"]), row["source_domain"])
    )

    write_csv(
        tables_dir / "source_domain_risk.csv",
        source_rows,
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
    review_fieldnames = [
        "crawl",
        "source_domain",
        "root_domain",
        "tranco_rank",
        "rows",
        "share_of_risky_rows",
        "high_rows",
        "medium_rows",
        "unique_target_platforms",
        "target_platforms",
        "platform_features",
    ]
    write_csv(
        review_dir / "top_source_domains.csv",
        _select_fields(source_rows[: max(args.top_n, 1)], review_fieldnames),
        review_fieldnames,
    )

    summary = {
        "generated_at_epoch": iso_now_epoch(),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "crawl": crawl_name,
        "rows_seen": rows_seen,
        "risky_rows": risky_rows,
        "risky_share_of_all_rows": round((risky_rows / rows_seen), 6) if rows_seen else 0.0,
        "unique_risky_source_domains": len(source_rows),
        "severity_counts": dict(severity_counts),
        "tranco": {
            "mode": args.tranco_mode,
            "source": tranco_source,
            "domains_loaded": len(tranco_ranking),
        },
        "files": {
            "source_domain_risk_csv": str(tables_dir / "source_domain_risk.csv"),
            "top_source_domains_csv": str(review_dir / "top_source_domains.csv"),
        },
    }
    manifest = {
        "script": "analyze_source_risk.py",
        "version": 1,
        "crawl": crawl_name,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "tables": sorted(str(path) for path in tables_dir.iterdir()),
        "review": sorted(str(path) for path in review_dir.iterdir()),
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
                "unique_risky_source_domains": len(source_rows),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())