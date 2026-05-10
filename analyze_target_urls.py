#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from platform_signatures import parse_domain
from source_url_analysis_common import (
    counter_to_sorted_rows,
    ensure_directory,
    iso_now_epoch,
    iter_jsonl,
    write_csv,
    write_json,
)


SEVERITY_ORDER: tuple[str, ...] = ("high", "medium", "low")


def _normalize_string(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _normalize_labels(row: dict[str, Any]) -> list[str]:
    raw_labels = row.get("prompt_labels")
    if isinstance(raw_labels, list):
        labels = [item.strip().upper() for item in raw_labels if isinstance(item, str) and item.strip()]
        if labels:
            return sorted(set(labels))

    classification = _normalize_string(row.get("classification"))
    if not classification:
        return []

    labels = [item.strip().upper() for item in classification.split(";") if item.strip()]
    return sorted(set(labels))


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return ""
    if not parsed.netloc:
        return ""
    path = parsed.path or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


def _target_url_features(target_url: str) -> dict[str, Any] | None:
    target_url_norm = _normalize_url(target_url)
    if not target_url_norm:
        return None

    try:
        parsed = urlsplit(target_url_norm)
    except ValueError:
        return None

    target_domain = parsed.netloc.lower()
    if not target_domain:
        return None

    query_keys = sorted({key.lower() for key in parsed.query.split("&") if key})
    return {
        "target_url_norm": target_url_norm,
        "target_domain": target_domain,
        "scheme": parsed.scheme.lower(),
        "path": parsed.path or "/",
        "query_key_count": len(query_keys),
    }


def _new_metrics() -> dict[str, Any]:
    return {
        "rows": 0,
        "rows_with_prompt_text": 0,
        "suspicious_rows": 0,
        "high_rows": 0,
        "medium_rows": 0,
        "low_rows": 0,
        "ioc_rows": 0,
        "label_counts": collections.Counter(),
    }


def _update_metrics(
    metrics: dict[str, Any],
    *,
    labels: list[str],
    severity: str,
    has_prompt_text: bool,
    is_suspicious: bool,
    has_ioc_keywords: bool,
) -> None:
    metrics["rows"] += 1
    if has_prompt_text:
        metrics["rows_with_prompt_text"] += 1
    if is_suspicious:
        metrics["suspicious_rows"] += 1
    if severity == "high":
        metrics["high_rows"] += 1
    elif severity == "medium":
        metrics["medium_rows"] += 1
    else:
        metrics["low_rows"] += 1
    if has_ioc_keywords:
        metrics["ioc_rows"] += 1
    for label in labels:
        metrics["label_counts"][label] += 1


def _top_items(counter: collections.Counter[str], limit: int = 3) -> str:
    items = counter.most_common(limit)
    return " | ".join(f"{name}:{count}" for name, count in items)


def _join_sorted(values: set[str]) -> str:
    return " | ".join(sorted(value for value in values if value))


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _severity_distribution_json(metrics: dict[str, Any]) -> str:
    payload = {
        "high": metrics["high_rows"],
        "medium": metrics["medium_rows"],
        "low": metrics["low_rows"],
    }
    return _json_dumps(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze TARGET_URL concentration, platform mix, and label/severity distributions for one classified crawl."
    )
    parser.add_argument("--input", required=True, help="Input classified_prompt_links.jsonl or .jsonl.gz")
    parser.add_argument("--output-dir", required=True, help="Output directory for analysis artifacts")
    parser.add_argument("--crawl-name", default="", help="Override crawl name in outputs")
    parser.add_argument("--top-n", type=int, default=100, help="Rows to keep in review tables")
    parser.add_argument(
        "--only-nonempty-prompt",
        action="store_true",
        help="Analyze only rows with non-empty primary_prompt_text",
    )
    parser.add_argument(
        "--suspicious-only",
        action="store_true",
        help="Analyze only rows where is_suspicious is true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    review_dir = output_dir / "review"
    figure_dir = output_dir / "figure_data"
    ensure_directory(output_dir)
    ensure_directory(tables_dir)
    ensure_directory(review_dir)
    ensure_directory(figure_dir)

    rows_seen = 0
    rows_analyzed = 0
    rows_missing_target_url = 0
    rows_invalid_target_url = 0
    rows_target_domain_missing = 0
    rows_target_domain_mismatch = 0
    rows_filtered_empty_prompt = 0
    rows_filtered_not_suspicious = 0
    rows_with_prompt_text = 0
    rows_suspicious = 0
    rows_with_ioc_keywords = 0

    platform_counts: collections.Counter[str] = collections.Counter()
    severity_counts: collections.Counter[str] = collections.Counter()
    label_counts: collections.Counter[str] = collections.Counter()
    session_entry_counts: collections.Counter[str] = collections.Counter()

    platform_label_counts: collections.Counter[tuple[str, str]] = collections.Counter()
    platform_severity_counts: collections.Counter[tuple[str, str]] = collections.Counter()
    platform_session_entry_counts: collections.Counter[tuple[str, str]] = collections.Counter()

    target_platform_stats: dict[str, dict[str, Any]] = {}
    target_domain_stats: dict[str, dict[str, Any]] = {}
    all_source_domains: set[str] = set()
    all_source_urls: set[str] = set()

    detected_crawl_name = ""

    for row in iter_jsonl(input_path):
        rows_seen += 1

        target_url_raw = _normalize_string(row.get("target_url"))
        if not target_url_raw:
            rows_missing_target_url += 1
            continue

        features = _target_url_features(target_url_raw)
        if features is None:
            rows_invalid_target_url += 1
            continue

        prompt_text = _normalize_string(row.get("primary_prompt_text"))
        if args.only_nonempty_prompt and not prompt_text:
            rows_filtered_empty_prompt += 1
            continue

        is_suspicious = bool(row.get("is_suspicious"))
        if args.suspicious_only and not is_suspicious:
            rows_filtered_not_suspicious += 1
            continue

        if not detected_crawl_name:
            detected_crawl_name = _normalize_string(row.get("crawl"))

        rows_analyzed += 1

        target_domain_row = _normalize_string(row.get("target_domain"))
        if not target_domain_row:
            rows_target_domain_missing += 1
        elif target_domain_row.lower() != features["target_domain"]:
            rows_target_domain_mismatch += 1

        platform = _normalize_string(row.get("target_platform")) or "(unknown)"
        source_url = _normalize_string(row.get("source_url"))
        source_domain = _normalize_string(row.get("source_domain")) or parse_domain(source_url) or "(unknown)"
        severity = _normalize_string(row.get("severity")) or "low"
        labels = _normalize_labels(row)
        session_entry_reason = _normalize_string(row.get("session_entry_reason")) or "(none)"
        has_ioc_keywords = bool(row.get("has_ioc_keywords"))

        if source_domain and source_domain != "(unknown)":
            all_source_domains.add(source_domain)
        if source_url:
            all_source_urls.add(source_url)

        if prompt_text:
            rows_with_prompt_text += 1
        if is_suspicious:
            rows_suspicious += 1
        if has_ioc_keywords:
            rows_with_ioc_keywords += 1

        platform_counts[platform] += 1
        severity_counts[severity] += 1
        session_entry_counts[session_entry_reason] += 1
        platform_severity_counts[(platform, severity)] += 1
        platform_session_entry_counts[(platform, session_entry_reason)] += 1
        for label in labels:
            label_counts[label] += 1
            platform_label_counts[(platform, label)] += 1

        platform_agg = target_platform_stats.get(platform)
        if platform_agg is None:
            platform_agg = {
                "target_platform": platform,
                "target_domains": set(),
                "target_urls": set(),
                "source_domains": set(),
                "source_urls": set(),
                "target_domain_counts": collections.Counter(),
                "source_domain_counts": collections.Counter(),
                "session_entry_counts": collections.Counter(),
                "example_target_url": "",
                "example_prompt_text": "",
                "metrics": _new_metrics(),
            }
            target_platform_stats[platform] = platform_agg
        platform_agg["target_domains"].add(features["target_domain"])
        platform_agg["target_urls"].add(features["target_url_norm"])
        platform_agg["source_domains"].add(source_domain)
        if source_url:
            platform_agg["source_urls"].add(source_url)
        platform_agg["target_domain_counts"][features["target_domain"]] += 1
        platform_agg["source_domain_counts"][source_domain] += 1
        platform_agg["session_entry_counts"][session_entry_reason] += 1
        if not platform_agg["example_target_url"]:
            platform_agg["example_target_url"] = features["target_url_norm"]
        if prompt_text and not platform_agg["example_prompt_text"]:
            platform_agg["example_prompt_text"] = prompt_text
        _update_metrics(
            platform_agg["metrics"],
            labels=labels,
            severity=severity,
            has_prompt_text=bool(prompt_text),
            is_suspicious=is_suspicious,
            has_ioc_keywords=has_ioc_keywords,
        )

        target_domain_key = features["target_domain"]
        domain_agg = target_domain_stats.get(target_domain_key)
        if domain_agg is None:
            domain_agg = {
                "target_domain": target_domain_key,
                "target_platform_counts": collections.Counter(),
                "target_urls": set(),
                "source_domains": set(),
                "source_urls": set(),
                "source_domain_counts": collections.Counter(),
                "session_entry_counts": collections.Counter(),
                "example_target_url": "",
                "example_prompt_text": "",
                "metrics": _new_metrics(),
            }
            target_domain_stats[target_domain_key] = domain_agg
        domain_agg["target_platform_counts"][platform] += 1
        domain_agg["target_urls"].add(features["target_url_norm"])
        domain_agg["source_domains"].add(source_domain)
        if source_url:
            domain_agg["source_urls"].add(source_url)
        domain_agg["source_domain_counts"][source_domain] += 1
        domain_agg["session_entry_counts"][session_entry_reason] += 1
        if not domain_agg["example_target_url"]:
            domain_agg["example_target_url"] = features["target_url_norm"]
        if prompt_text and not domain_agg["example_prompt_text"]:
            domain_agg["example_prompt_text"] = prompt_text
        _update_metrics(
            domain_agg["metrics"],
            labels=labels,
            severity=severity,
            has_prompt_text=bool(prompt_text),
            is_suspicious=is_suspicious,
            has_ioc_keywords=has_ioc_keywords,
        )

    crawl_name = args.crawl_name.strip() or detected_crawl_name or input_path.stem

    target_platform_rows: list[dict[str, Any]] = []
    for aggregate in target_platform_stats.values():
        metrics = aggregate["metrics"]
        top_target_domains = [name for name, _count in aggregate["target_domain_counts"].most_common(2)]
        top_source_domains = [name for name, _count in aggregate["source_domain_counts"].most_common(2)]
        top_session_entry = aggregate["session_entry_counts"].most_common(1)
        target_platform_rows.append(
            {
                "crawl": crawl_name,
                "target_platform": aggregate["target_platform"],
                "rows": metrics["rows"],
                "rows_with_prompt_text": metrics["rows_with_prompt_text"],
                "suspicious_rows": metrics["suspicious_rows"],
                "high_rows": metrics["high_rows"],
                "medium_rows": metrics["medium_rows"],
                "low_rows": metrics["low_rows"],
                "ioc_rows": metrics["ioc_rows"],
                "unique_target_domains": len(aggregate["target_domains"]),
                "unique_target_urls": len(aggregate["target_urls"]),
                "unique_source_domains": len(aggregate["source_domains"]),
                "unique_source_urls": len(aggregate["source_urls"]),
                "top_target_domain_1": top_target_domains[0] if len(top_target_domains) > 0 else "",
                "top_target_domain_2": top_target_domains[1] if len(top_target_domains) > 1 else "",
                "top_source_domain_1": top_source_domains[0] if len(top_source_domains) > 0 else "",
                "top_source_domain_2": top_source_domains[1] if len(top_source_domains) > 1 else "",
                "top_session_entry_reason": top_session_entry[0][0] if top_session_entry else "",
                "top_labels": _top_items(metrics["label_counts"]),
                "label_distribution_json": _json_dumps(dict(metrics["label_counts"].most_common())),
                "severity_distribution_json": _severity_distribution_json(metrics),
                "example_target_url": aggregate["example_target_url"],
                "example_prompt_text": aggregate["example_prompt_text"],
            }
        )
    target_platform_rows.sort(
        key=lambda row: (-int(row["rows"]), -int(row["unique_target_domains"]), row["target_platform"])
    )

    target_domain_rows: list[dict[str, Any]] = []
    for aggregate in target_domain_stats.values():
        metrics = aggregate["metrics"]
        top_source_domains = [name for name, _count in aggregate["source_domain_counts"].most_common(2)]
        top_platform = aggregate["target_platform_counts"].most_common(1)
        top_session_entry = aggregate["session_entry_counts"].most_common(1)
        target_domain_rows.append(
            {
                "crawl": crawl_name,
                "target_domain": aggregate["target_domain"],
                "target_platform": top_platform[0][0] if top_platform else "",
                "rows": metrics["rows"],
                "rows_with_prompt_text": metrics["rows_with_prompt_text"],
                "suspicious_rows": metrics["suspicious_rows"],
                "high_rows": metrics["high_rows"],
                "medium_rows": metrics["medium_rows"],
                "low_rows": metrics["low_rows"],
                "ioc_rows": metrics["ioc_rows"],
                "unique_target_urls": len(aggregate["target_urls"]),
                "unique_source_domains": len(aggregate["source_domains"]),
                "unique_source_urls": len(aggregate["source_urls"]),
                "top_source_domain_1": top_source_domains[0] if len(top_source_domains) > 0 else "",
                "top_source_domain_2": top_source_domains[1] if len(top_source_domains) > 1 else "",
                "top_session_entry_reason": top_session_entry[0][0] if top_session_entry else "",
                "top_labels": _top_items(metrics["label_counts"]),
                "label_distribution_json": _json_dumps(dict(metrics["label_counts"].most_common())),
                "severity_distribution_json": _severity_distribution_json(metrics),
                "example_target_url": aggregate["example_target_url"],
                "example_prompt_text": aggregate["example_prompt_text"],
            }
        )
    target_domain_rows.sort(
        key=lambda row: (-int(row["rows"]), -int(row["unique_source_domains"]), row["target_domain"])
    )

    platform_totals = {platform: count for platform, count in platform_counts.items()}
    label_totals = {label: count for label, count in label_counts.items()}
    severity_totals = {severity: count for severity, count in severity_counts.items()}
    session_entry_totals = {reason: count for reason, count in session_entry_counts.items()}

    platform_by_label_rows: list[dict[str, Any]] = []
    for platform in sorted(platform_counts):
        for label in sorted(label_totals):
            count = platform_label_counts.get((platform, label), 0)
            platform_by_label_rows.append(
                {
                    "crawl": crawl_name,
                    "target_platform": platform,
                    "label": label,
                    "rows": count,
                    "share_within_platform": round((count / platform_totals[platform]), 6) if platform_totals.get(platform) else 0.0,
                    "share_within_label": round((count / label_totals[label]), 6) if label_totals.get(label) else 0.0,
                }
            )

    platform_by_severity_rows: list[dict[str, Any]] = []
    for platform in sorted(platform_counts):
        for severity in SEVERITY_ORDER:
            count = platform_severity_counts.get((platform, severity), 0)
            platform_by_severity_rows.append(
                {
                    "crawl": crawl_name,
                    "target_platform": platform,
                    "severity": severity,
                    "rows": count,
                    "share_within_platform": round((count / platform_totals[platform]), 6) if platform_totals.get(platform) else 0.0,
                    "share_within_severity": round((count / severity_totals[severity]), 6) if severity_totals.get(severity) else 0.0,
                }
            )

    platform_by_session_entry_rows: list[dict[str, Any]] = []
    for platform in sorted(platform_counts):
        for reason in sorted(session_entry_totals):
            count = platform_session_entry_counts.get((platform, reason), 0)
            platform_by_session_entry_rows.append(
                {
                    "crawl": crawl_name,
                    "target_platform": platform,
                    "session_entry_reason": reason,
                    "rows": count,
                    "share_within_platform": round((count / platform_totals[platform]), 6) if platform_totals.get(platform) else 0.0,
                    "share_within_reason": round((count / session_entry_totals[reason]), 6) if session_entry_totals.get(reason) else 0.0,
                }
            )

    label_distribution_rows = counter_to_sorted_rows(dict(label_counts), total=rows_analyzed, key_name="label")
    severity_distribution_rows = counter_to_sorted_rows(dict(severity_counts), total=rows_analyzed, key_name="severity")
    session_entry_distribution_rows = counter_to_sorted_rows(
        dict(session_entry_counts),
        total=rows_analyzed,
        key_name="session_entry_reason",
    )

    review_top_target_platforms = [
        {
            "crawl": row["crawl"],
            "rank": index,
            "target_platform": row["target_platform"],
            "rows": row["rows"],
            "suspicious_rows": row["suspicious_rows"],
            "unique_target_domains": row["unique_target_domains"],
            "unique_source_domains": row["unique_source_domains"],
            "top_target_domain_1": row["top_target_domain_1"],
            "top_target_domain_2": row["top_target_domain_2"],
            "top_source_domain_1": row["top_source_domain_1"],
            "top_source_domain_2": row["top_source_domain_2"],
            "top_labels": row["top_labels"],
            "example_target_url": row["example_target_url"],
            "example_prompt_text": row["example_prompt_text"],
        }
        for index, row in enumerate(target_platform_rows[: max(args.top_n, 1)], start=1)
    ]

    review_top_target_domains = [
        {
            "crawl": row["crawl"],
            "rank": index,
            "target_domain": row["target_domain"],
            "target_platform": row["target_platform"],
            "rows": row["rows"],
            "suspicious_rows": row["suspicious_rows"],
            "unique_source_domains": row["unique_source_domains"],
            "unique_source_urls": row["unique_source_urls"],
            "top_source_domain_1": row["top_source_domain_1"],
            "top_source_domain_2": row["top_source_domain_2"],
            "top_labels": row["top_labels"],
            "example_target_url": row["example_target_url"],
            "example_prompt_text": row["example_prompt_text"],
        }
        for index, row in enumerate(target_domain_rows[: max(args.top_n, 1)], start=1)
    ]

    target_domain_rank_rows: list[dict[str, Any]] = []
    cumulative_rows = 0
    for index, row in enumerate(target_domain_rows, start=1):
        cumulative_rows += int(row["rows"])
        target_domain_rank_rows.append(
            {
                "crawl": crawl_name,
                "rank": index,
                "target_domain": row["target_domain"],
                "target_platform": row["target_platform"],
                "rows": row["rows"],
                "cumulative_rows": cumulative_rows,
                "cumulative_share_rows": round((cumulative_rows / rows_analyzed), 6) if rows_analyzed else 0.0,
            }
        )

    target_platform_rows_figure = [
        {"crawl": row["crawl"], "target_platform": row["target_platform"], "rows": row["rows"]}
        for row in target_platform_rows
    ]
    target_platform_suspicious_share_rows = [
        {
            "crawl": row["crawl"],
            "target_platform": row["target_platform"],
            "rows": row["rows"],
            "suspicious_rows": row["suspicious_rows"],
            "high_rows": row["high_rows"],
            "ioc_rows": row["ioc_rows"],
            "suspicious_share": round((int(row["suspicious_rows"]) / int(row["rows"])), 6) if int(row["rows"]) else 0.0,
            "high_share": round((int(row["high_rows"]) / int(row["rows"])), 6) if int(row["rows"]) else 0.0,
            "ioc_share": round((int(row["ioc_rows"]) / int(row["rows"])), 6) if int(row["rows"]) else 0.0,
        }
        for row in target_platform_rows
    ]

    target_platform_fieldnames = [
        "crawl", "target_platform", "rows", "rows_with_prompt_text", "suspicious_rows", "high_rows",
        "medium_rows", "low_rows", "ioc_rows", "unique_target_domains", "unique_target_urls",
        "unique_source_domains", "unique_source_urls", "top_target_domain_1", "top_target_domain_2",
        "top_source_domain_1", "top_source_domain_2", "top_session_entry_reason", "top_labels",
        "label_distribution_json", "severity_distribution_json", "example_target_url", "example_prompt_text",
    ]
    target_domain_fieldnames = [
        "crawl", "target_domain", "target_platform", "rows", "rows_with_prompt_text", "suspicious_rows",
        "high_rows", "medium_rows", "low_rows", "ioc_rows", "unique_target_urls", "unique_source_domains",
        "unique_source_urls", "top_source_domain_1", "top_source_domain_2", "top_session_entry_reason",
        "top_labels", "label_distribution_json", "severity_distribution_json", "example_target_url",
        "example_prompt_text",
    ]

    write_csv(tables_dir / "target_platform_stats.csv", target_platform_rows, target_platform_fieldnames)
    write_csv(tables_dir / "target_domain_stats.csv", target_domain_rows, target_domain_fieldnames)
    write_csv(
        tables_dir / "platform_by_label.csv",
        platform_by_label_rows,
        ["crawl", "target_platform", "label", "rows", "share_within_platform", "share_within_label"],
    )
    write_csv(
        tables_dir / "platform_by_severity.csv",
        platform_by_severity_rows,
        ["crawl", "target_platform", "severity", "rows", "share_within_platform", "share_within_severity"],
    )
    write_csv(
        tables_dir / "platform_by_session_entry.csv",
        platform_by_session_entry_rows,
        [
            "crawl",
            "target_platform",
            "session_entry_reason",
            "rows",
            "share_within_platform",
            "share_within_reason",
        ],
    )
    write_csv(tables_dir / "label_distribution.csv", label_distribution_rows, ["label", "count", "share"])
    write_csv(tables_dir / "severity_distribution.csv", severity_distribution_rows, ["severity", "count", "share"])
    write_csv(
        tables_dir / "session_entry_distribution.csv",
        session_entry_distribution_rows,
        ["session_entry_reason", "count", "share"],
    )

    write_csv(
        review_dir / "top_target_platforms.csv",
        review_top_target_platforms,
        [
            "crawl",
            "rank",
            "target_platform",
            "rows",
            "suspicious_rows",
            "unique_target_domains",
            "unique_source_domains",
            "top_target_domain_1",
            "top_target_domain_2",
            "top_source_domain_1",
            "top_source_domain_2",
            "top_labels",
            "example_target_url",
            "example_prompt_text",
        ],
    )
    write_csv(
        review_dir / "top_target_domains.csv",
        review_top_target_domains,
        [
            "crawl",
            "rank",
            "target_domain",
            "target_platform",
            "rows",
            "suspicious_rows",
            "unique_source_domains",
            "unique_source_urls",
            "top_source_domain_1",
            "top_source_domain_2",
            "top_labels",
            "example_target_url",
            "example_prompt_text",
        ],
    )

    write_csv(
        figure_dir / "target_platform_rows.csv",
        target_platform_rows_figure,
        ["crawl", "target_platform", "rows"],
    )
    write_csv(
        figure_dir / "target_platform_suspicious_share.csv",
        target_platform_suspicious_share_rows,
        [
            "crawl",
            "target_platform",
            "rows",
            "suspicious_rows",
            "high_rows",
            "ioc_rows",
            "suspicious_share",
            "high_share",
            "ioc_share",
        ],
    )
    write_csv(
        figure_dir / "target_domain_rank.csv",
        target_domain_rank_rows,
        ["crawl", "rank", "target_domain", "target_platform", "rows", "cumulative_rows", "cumulative_share_rows"],
    )
    write_csv(
        figure_dir / "label_by_target_platform.csv",
        platform_by_label_rows,
        ["crawl", "target_platform", "label", "rows", "share_within_platform", "share_within_label"],
    )
    write_csv(
        figure_dir / "severity_by_target_platform.csv",
        platform_by_severity_rows,
        ["crawl", "target_platform", "severity", "rows", "share_within_platform", "share_within_severity"],
    )
    write_csv(
        figure_dir / "session_entry_by_target_platform.csv",
        platform_by_session_entry_rows,
        [
            "crawl",
            "target_platform",
            "session_entry_reason",
            "rows",
            "share_within_platform",
            "share_within_reason",
        ],
    )

    summary = {
        "generated_at_epoch": iso_now_epoch(),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "crawl": crawl_name,
        "filters": {
            "only_nonempty_prompt": args.only_nonempty_prompt,
            "suspicious_only": args.suspicious_only,
        },
        "quality": {
            "rows_seen": rows_seen,
            "rows_analyzed": rows_analyzed,
            "rows_missing_target_url": rows_missing_target_url,
            "rows_invalid_target_url": rows_invalid_target_url,
            "rows_target_domain_missing": rows_target_domain_missing,
            "rows_target_domain_mismatch": rows_target_domain_mismatch,
            "rows_filtered_empty_prompt": rows_filtered_empty_prompt,
            "rows_filtered_not_suspicious": rows_filtered_not_suspicious,
        },
        "counts": {
            "rows_with_prompt_text": rows_with_prompt_text,
            "rows_suspicious": rows_suspicious,
            "rows_with_ioc_keywords": rows_with_ioc_keywords,
            "unique_target_platforms": len(target_platform_rows),
            "unique_target_domains": len(target_domain_rows),
            "unique_source_domains": len(all_source_domains),
            "unique_source_urls": len(all_source_urls),
        },
        "distributions": {
            "platform_distribution": counter_to_sorted_rows(dict(platform_counts), total=rows_analyzed, key_name="target_platform"),
            "severity_distribution": severity_distribution_rows,
            "label_distribution": label_distribution_rows,
            "session_entry_distribution": session_entry_distribution_rows,
        },
        "files": {
            "summary_json": str(output_dir / "summary.json"),
            "manifest_json": str(output_dir / "manifest.json"),
            "target_platform_stats_csv": str(tables_dir / "target_platform_stats.csv"),
            "target_domain_stats_csv": str(tables_dir / "target_domain_stats.csv"),
            "platform_by_label_csv": str(tables_dir / "platform_by_label.csv"),
            "platform_by_severity_csv": str(tables_dir / "platform_by_severity.csv"),
            "platform_by_session_entry_csv": str(tables_dir / "platform_by_session_entry.csv"),
            "label_distribution_csv": str(tables_dir / "label_distribution.csv"),
            "severity_distribution_csv": str(tables_dir / "severity_distribution.csv"),
            "session_entry_distribution_csv": str(tables_dir / "session_entry_distribution.csv"),
            "top_target_platforms_csv": str(review_dir / "top_target_platforms.csv"),
            "top_target_domains_csv": str(review_dir / "top_target_domains.csv"),
        },
    }
    manifest = {
        "script": "analyze_target_urls.py",
        "version": 1,
        "crawl": crawl_name,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "tables": sorted(str(path) for path in tables_dir.iterdir()),
        "review": sorted(str(path) for path in review_dir.iterdir()),
        "figure_data": sorted(str(path) for path in figure_dir.iterdir()),
        "summary_json": str(output_dir / "summary.json"),
    }

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "manifest.json", manifest)

    print(
        json.dumps(
            {
                "crawl": crawl_name,
                "rows_seen": rows_seen,
                "rows_analyzed": rows_analyzed,
                "unique_target_platforms": len(target_platform_rows),
                "unique_target_domains": len(target_domain_rows),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())