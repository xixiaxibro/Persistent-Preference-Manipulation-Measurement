#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit, urlunsplit

from platform_signatures import parse_domain
from source_url_analysis_common import (
    TRANCO_BUCKET_ORDER,
    counter_to_sorted_rows,
    ensure_directory,
    extract_root_domain,
    iso_now_epoch,
    iter_jsonl,
    load_tranco_ranking,
    lookup_tranco,
    make_domain_extractor,
    ordered_bucket_rows,
    write_csv,
    write_json,
)


CLASS_LABELS: tuple[str, ...] = (
    "PERSISTENCE",
    "AUTHORITY",
    "RECOMMENDATION",
    "CITATION",
    "SUMMARY",
)

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
HEX_RE = re.compile(r"^[0-9a-f]{8,}$", re.IGNORECASE)
NUMERIC_RE = re.compile(r"^\d+$")
LONG_TOKEN_RE = re.compile(r"^[a-z0-9_-]{24,}$", re.IGNORECASE)
DATE_RE = re.compile(r"^\d{4}[-_]\d{2}[-_]\d{2}$")

SEARCH_SEGMENTS = {"search", "find", "results", "lookup"}
CATEGORY_SEGMENTS = {"tag", "tags", "category", "categories", "topic", "topics", "archive", "archives"}
PRODUCT_SEGMENTS = {"product", "products", "shop", "store", "item", "items", "listing", "listings", "sku"}
DOC_SEGMENTS = {"docs", "doc", "documentation", "kb", "faq", "faqs", "help", "guide", "guides", "manual", "reference"}
SEARCH_QUERY_KEYS = {"q", "query", "search", "s", "keyword", "term"}


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


def _normalize_segment(segment: str) -> str:
    lowered = segment.strip().lower()
    if not lowered:
        return ""
    if DATE_RE.match(lowered):
        return "{date}"
    if NUMERIC_RE.match(lowered):
        return "{num}"
    if UUID_RE.match(lowered):
        return "{uuid}"
    if HEX_RE.match(lowered):
        return "{hex}"
    if LONG_TOKEN_RE.match(lowered):
        return "{token}"
    return lowered


def _classify_page_kind(path: str, query_keys: list[str], raw_segments: list[str]) -> str:
    if path == "/" and not query_keys:
        return "homepage"

    lowered_segments = {segment.lower() for segment in raw_segments if segment}
    if SEARCH_QUERY_KEYS.intersection(query_keys) or SEARCH_SEGMENTS.intersection(lowered_segments):
        return "search_or_query_page"
    if CATEGORY_SEGMENTS.intersection(lowered_segments):
        return "category_or_tag"
    if PRODUCT_SEGMENTS.intersection(lowered_segments):
        return "product_or_listing"
    if DOC_SEGMENTS.intersection(lowered_segments):
        return "doc_or_help"
    if raw_segments:
        return "article_like"
    return "other"


def _source_url_features(source_url: str, extractor) -> dict[str, Any] | None:
    source_url_norm = _normalize_url(source_url)
    if not source_url_norm:
        return None

    try:
        parsed = urlsplit(source_url_norm)
    except ValueError:
        return None

    source_domain = parsed.netloc.lower()
    if not source_domain:
        return None

    raw_path = parsed.path or "/"
    lowered_path = raw_path.lower() or "/"
    raw_segments = [segment for segment in lowered_path.split("/") if segment]
    normalized_segments = [_normalize_segment(segment) for segment in raw_segments]
    query = parse_qs(parsed.query, keep_blank_values=False)
    query_keys = sorted({key.lower() for key in query})
    path_template = "/" + "/".join(normalized_segments) if normalized_segments else "/"
    root_domain = extract_root_domain(source_domain, extractor)
    page_kind = _classify_page_kind(lowered_path, query_keys, raw_segments)

    return {
        "source_url_norm": source_url_norm,
        "source_domain": source_domain,
        "root_domain": root_domain,
        "scheme": parsed.scheme.lower(),
        "path": lowered_path,
        "path_template": path_template,
        "path_depth": len(raw_segments),
        "query_keys": query_keys,
        "query_key_count": len(query_keys),
        "page_kind": page_kind,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SOURCE_URL structure, concentration, and Tranco popularity for one classified crawl.")
    parser.add_argument("--input", required=True, help="Input classified_prompt_links.jsonl or .jsonl.gz")
    parser.add_argument("--output-dir", required=True, help="Output directory for analysis artifacts")
    parser.add_argument("--crawl-name", default="", help="Override crawl name in outputs")
    parser.add_argument("--top-n", type=int, default=100, help="Rows to keep in review tables")
    parser.add_argument("--examples-per-group", type=int, default=2, help="Examples to retain in grouped outputs")
    parser.add_argument("--only-nonempty-prompt", action="store_true", help="Analyze only rows with non-empty primary_prompt_text")
    parser.add_argument("--suspicious-only", action="store_true", help="Analyze only rows where is_suspicious is true")
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
    figure_dir = output_dir / "figure_data"
    ensure_directory(output_dir)
    ensure_directory(tables_dir)
    ensure_directory(review_dir)
    ensure_directory(figure_dir)

    extractor = make_domain_extractor()
    tranco_csv = Path(args.tranco_csv) if args.tranco_csv else None
    tranco_cache = Path(args.tranco_cache) if args.tranco_cache else None
    tranco_ranking, tranco_source = load_tranco_ranking(
        tranco_csv=tranco_csv,
        tranco_cache=tranco_cache,
        mode=args.tranco_mode,
    )

    rows_seen = 0
    rows_analyzed = 0
    rows_missing_source_url = 0
    rows_invalid_source_url = 0
    rows_source_domain_missing = 0
    rows_source_domain_mismatch = 0
    rows_filtered_empty_prompt = 0
    rows_filtered_not_suspicious = 0
    rows_with_prompt_text = 0
    rows_suspicious = 0
    rows_with_ioc_keywords = 0

    platform_counts: collections.Counter[str] = collections.Counter()
    severity_counts: collections.Counter[str] = collections.Counter()
    label_counts: collections.Counter[str] = collections.Counter()
    scheme_counts: collections.Counter[str] = collections.Counter()
    page_kind_counts: collections.Counter[str] = collections.Counter()

    source_url_stats: dict[str, dict[str, Any]] = {}
    source_domain_stats: dict[str, dict[str, Any]] = {}
    root_domain_stats: dict[str, dict[str, Any]] = {}
    source_path_patterns: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    tranco_bucket_stats: dict[str, dict[str, Any]] = {}
    bucket_platform_counts: collections.Counter[tuple[str, str]] = collections.Counter()
    bucket_label_counts: collections.Counter[tuple[str, str]] = collections.Counter()

    detected_crawl_name = ""

    for row in iter_jsonl(input_path):
        rows_seen += 1

        source_url_raw = _normalize_string(row.get("source_url"))
        if not source_url_raw:
            rows_missing_source_url += 1
            continue

        features = _source_url_features(source_url_raw, extractor)
        if features is None:
            rows_invalid_source_url += 1
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

        source_domain_row = _normalize_string(row.get("source_domain"))
        if not source_domain_row:
            rows_source_domain_missing += 1
        elif source_domain_row.lower() != features["source_domain"]:
            rows_source_domain_mismatch += 1

        scheme_counts[features["scheme"] or "(unknown)"] += 1
        page_kind_counts[features["page_kind"]] += 1

        labels = _normalize_labels(row)
        severity = _normalize_string(row.get("severity")) or "low"
        platform = _normalize_string(row.get("target_platform")) or "(unknown)"
        target_url = _normalize_string(row.get("target_url"))
        target_domain = _normalize_string(row.get("target_domain")) or parse_domain(target_url)
        has_ioc_keywords = bool(row.get("has_ioc_keywords"))

        if prompt_text:
            rows_with_prompt_text += 1
        if is_suspicious:
            rows_suspicious += 1
        if has_ioc_keywords:
            rows_with_ioc_keywords += 1

        platform_counts[platform] += 1
        severity_counts[severity] += 1
        for label in labels:
            label_counts[label] += 1

        tranco_match = lookup_tranco(features["root_domain"], tranco_ranking)

        source_url_key = features["source_url_norm"]
        source_domain_key = features["source_domain"]
        root_domain_key = features["root_domain"] or features["source_domain"]
        query_keys_text = " | ".join(features["query_keys"])
        path_pattern_key = (
            root_domain_key,
            features["path_template"],
            features["page_kind"],
            query_keys_text,
        )

        url_agg = source_url_stats.get(source_url_key)
        if url_agg is None:
            url_agg = {
                "crawl": "",
                "source_url": source_url_key,
                "source_domain": source_domain_key,
                "root_domain": root_domain_key,
                "tranco_rank": tranco_match.rank,
                "tranco_matched_domain": tranco_match.matched_domain,
                "tranco_bucket": tranco_match.bucket,
                "tranco_match_type": tranco_match.match_type,
                "scheme": features["scheme"],
                "path": features["path"],
                "path_template": features["path_template"],
                "path_depth": features["path_depth"],
                "page_kind": features["page_kind"],
                "query_keys": list(features["query_keys"]),
                "query_key_count": features["query_key_count"],
                "target_urls": set(),
                "target_domains": set(),
                "target_platforms": set(),
                "label_sets": set(),
                "example_target_url": "",
                "example_prompt_text": "",
                "metrics": _new_metrics(),
            }
            source_url_stats[source_url_key] = url_agg
        url_agg["target_urls"].add(target_url)
        if target_domain:
            url_agg["target_domains"].add(target_domain)
        if platform:
            url_agg["target_platforms"].add(platform)
        url_agg["label_sets"].add(tuple(labels))
        if target_url and not url_agg["example_target_url"]:
            url_agg["example_target_url"] = target_url
        if prompt_text and not url_agg["example_prompt_text"]:
            url_agg["example_prompt_text"] = prompt_text
        _update_metrics(
            url_agg["metrics"],
            labels=labels,
            severity=severity,
            has_prompt_text=bool(prompt_text),
            is_suspicious=is_suspicious,
            has_ioc_keywords=has_ioc_keywords,
        )

        domain_agg = source_domain_stats.get(source_domain_key)
        if domain_agg is None:
            domain_agg = {
                "crawl": "",
                "source_domain": source_domain_key,
                "root_domain": root_domain_key,
                "tranco_rank": tranco_match.rank,
                "tranco_matched_domain": tranco_match.matched_domain,
                "tranco_bucket": tranco_match.bucket,
                "tranco_match_type": tranco_match.match_type,
                "source_urls": set(),
                "target_urls": set(),
                "target_platforms": set(),
                "target_platform_counts": collections.Counter(),
                "source_url_counts": collections.Counter(),
                "example_target_url": "",
                "example_prompt_text": "",
                "metrics": _new_metrics(),
            }
            source_domain_stats[source_domain_key] = domain_agg
        domain_agg["source_urls"].add(source_url_key)
        domain_agg["source_url_counts"][source_url_key] += 1
        domain_agg["target_urls"].add(target_url)
        if platform:
            domain_agg["target_platforms"].add(platform)
            domain_agg["target_platform_counts"][platform] += 1
        if target_url and not domain_agg["example_target_url"]:
            domain_agg["example_target_url"] = target_url
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

        root_agg = root_domain_stats.get(root_domain_key)
        if root_agg is None:
            root_agg = {
                "crawl": "",
                "root_domain": root_domain_key,
                "tranco_rank": tranco_match.rank,
                "tranco_matched_domain": tranco_match.matched_domain,
                "tranco_bucket": tranco_match.bucket,
                "tranco_match_type": tranco_match.match_type,
                "source_domains": set(),
                "source_urls": set(),
                "target_urls": set(),
                "target_platforms": set(),
                "source_domain_counts": collections.Counter(),
                "source_url_counts": collections.Counter(),
                "example_target_url": "",
                "example_prompt_text": "",
                "metrics": _new_metrics(),
            }
            root_domain_stats[root_domain_key] = root_agg
        root_agg["source_domains"].add(source_domain_key)
        root_agg["source_urls"].add(source_url_key)
        root_agg["source_domain_counts"][source_domain_key] += 1
        root_agg["source_url_counts"][source_url_key] += 1
        root_agg["target_urls"].add(target_url)
        if platform:
            root_agg["target_platforms"].add(platform)
        if target_url and not root_agg["example_target_url"]:
            root_agg["example_target_url"] = target_url
        if prompt_text and not root_agg["example_prompt_text"]:
            root_agg["example_prompt_text"] = prompt_text
        _update_metrics(
            root_agg["metrics"],
            labels=labels,
            severity=severity,
            has_prompt_text=bool(prompt_text),
            is_suspicious=is_suspicious,
            has_ioc_keywords=has_ioc_keywords,
        )

        pattern_agg = source_path_patterns.get(path_pattern_key)
        if pattern_agg is None:
            pattern_agg = {
                "crawl": "",
                "root_domain": root_domain_key,
                "tranco_rank": tranco_match.rank,
                "tranco_matched_domain": tranco_match.matched_domain,
                "tranco_bucket": tranco_match.bucket,
                "tranco_match_type": tranco_match.match_type,
                "path_template": features["path_template"],
                "page_kind": features["page_kind"],
                "path_depth": features["path_depth"],
                "query_keys": list(features["query_keys"]),
                "source_urls": set(),
                "source_domains": set(),
                "target_platforms": set(),
                "example_source_urls": [],
                "example_prompt_values": [],
                "metrics": _new_metrics(),
            }
            source_path_patterns[path_pattern_key] = pattern_agg
        pattern_agg["source_urls"].add(source_url_key)
        pattern_agg["source_domains"].add(source_domain_key)
        if platform:
            pattern_agg["target_platforms"].add(platform)
        if source_url_key not in pattern_agg["example_source_urls"] and len(pattern_agg["example_source_urls"]) < args.examples_per_group:
            pattern_agg["example_source_urls"].append(source_url_key)
        if prompt_text and prompt_text not in pattern_agg["example_prompt_values"] and len(pattern_agg["example_prompt_values"]) < args.examples_per_group:
            pattern_agg["example_prompt_values"].append(prompt_text)
        _update_metrics(
            pattern_agg["metrics"],
            labels=labels,
            severity=severity,
            has_prompt_text=bool(prompt_text),
            is_suspicious=is_suspicious,
            has_ioc_keywords=has_ioc_keywords,
        )

        bucket_agg = tranco_bucket_stats.get(tranco_match.bucket)
        if bucket_agg is None:
            bucket_agg = {
                "tranco_bucket": tranco_match.bucket,
                "root_domains": set(),
                "source_domains": set(),
                "source_urls": set(),
                "target_platforms": set(),
                "metrics": _new_metrics(),
            }
            tranco_bucket_stats[tranco_match.bucket] = bucket_agg
        bucket_agg["root_domains"].add(root_domain_key)
        bucket_agg["source_domains"].add(source_domain_key)
        bucket_agg["source_urls"].add(source_url_key)
        if platform:
            bucket_agg["target_platforms"].add(platform)
        _update_metrics(
            bucket_agg["metrics"],
            labels=labels,
            severity=severity,
            has_prompt_text=bool(prompt_text),
            is_suspicious=is_suspicious,
            has_ioc_keywords=has_ioc_keywords,
        )

        bucket_platform_counts[(tranco_match.bucket, platform)] += 1
        if labels:
            for label in labels:
                bucket_label_counts[(tranco_match.bucket, label)] += 1
        else:
            bucket_label_counts[(tranco_match.bucket, "(none)")] += 1

    crawl_name = args.crawl_name.strip() or detected_crawl_name or input_path.stem

    source_url_rows: list[dict[str, Any]] = []
    for aggregate in source_url_stats.values():
        metrics = aggregate["metrics"]
        source_url_rows.append(
            {
                "crawl": crawl_name,
                "source_url": aggregate["source_url"],
                "source_domain": aggregate["source_domain"],
                "root_domain": aggregate["root_domain"],
                "tranco_rank": aggregate["tranco_rank"] or "",
                "tranco_matched_domain": aggregate["tranco_matched_domain"],
                "tranco_bucket": aggregate["tranco_bucket"],
                "tranco_match_type": aggregate["tranco_match_type"],
                "scheme": aggregate["scheme"],
                "path": aggregate["path"],
                "path_template": aggregate["path_template"],
                "path_depth": aggregate["path_depth"],
                "page_kind": aggregate["page_kind"],
                "query_keys": " | ".join(aggregate["query_keys"]),
                "query_key_count": aggregate["query_key_count"],
                "rows": metrics["rows"],
                "rows_with_prompt_text": metrics["rows_with_prompt_text"],
                "suspicious_rows": metrics["suspicious_rows"],
                "high_rows": metrics["high_rows"],
                "medium_rows": metrics["medium_rows"],
                "low_rows": metrics["low_rows"],
                "ioc_rows": metrics["ioc_rows"],
                "unique_target_urls": len(aggregate["target_urls"]),
                "unique_target_domains": len(aggregate["target_domains"]),
                "unique_target_platforms": len(aggregate["target_platforms"]),
                "target_platforms": _join_sorted(aggregate["target_platforms"]),
                "label_set_count": len(aggregate["label_sets"]),
                "top_labels": _top_items(metrics["label_counts"]),
                "label_distribution_json": _json_dumps(dict(metrics["label_counts"].most_common())),
                "example_target_url": aggregate["example_target_url"],
                "example_prompt_text": aggregate["example_prompt_text"],
            }
        )
    source_url_rows.sort(key=lambda row: (-int(row["rows"]), -int(row["unique_target_platforms"]), row["source_url"]))

    source_domain_rows: list[dict[str, Any]] = []
    for aggregate in source_domain_stats.values():
        metrics = aggregate["metrics"]
        top_urls = [name for name, _count in aggregate["source_url_counts"].most_common(2)]
        top_platform = aggregate["target_platform_counts"].most_common(1)
        source_domain_rows.append(
            {
                "crawl": crawl_name,
                "source_domain": aggregate["source_domain"],
                "root_domain": aggregate["root_domain"],
                "tranco_rank": aggregate["tranco_rank"] or "",
                "tranco_matched_domain": aggregate["tranco_matched_domain"],
                "tranco_bucket": aggregate["tranco_bucket"],
                "tranco_match_type": aggregate["tranco_match_type"],
                "rows": metrics["rows"],
                "rows_with_prompt_text": metrics["rows_with_prompt_text"],
                "suspicious_rows": metrics["suspicious_rows"],
                "high_rows": metrics["high_rows"],
                "medium_rows": metrics["medium_rows"],
                "low_rows": metrics["low_rows"],
                "ioc_rows": metrics["ioc_rows"],
                "unique_source_urls": len(aggregate["source_urls"]),
                "unique_target_urls": len(aggregate["target_urls"]),
                "unique_target_platforms": len(aggregate["target_platforms"]),
                "target_platforms": _join_sorted(aggregate["target_platforms"]),
                "top_platform": top_platform[0][0] if top_platform else "",
                "top_labels": _top_items(metrics["label_counts"]),
                "label_distribution_json": _json_dumps(dict(metrics["label_counts"].most_common())),
                "top_source_url_1": top_urls[0] if len(top_urls) > 0 else "",
                "top_source_url_2": top_urls[1] if len(top_urls) > 1 else "",
                "example_target_url": aggregate["example_target_url"],
                "example_prompt_text": aggregate["example_prompt_text"],
            }
        )
    source_domain_rows.sort(key=lambda row: (-int(row["rows"]), -int(row["unique_source_urls"]), row["source_domain"]))

    root_domain_rows: list[dict[str, Any]] = []
    for aggregate in root_domain_stats.values():
        metrics = aggregate["metrics"]
        top_source_domains = [name for name, _count in aggregate["source_domain_counts"].most_common(2)]
        top_source_urls = [name for name, _count in aggregate["source_url_counts"].most_common(2)]
        root_domain_rows.append(
            {
                "crawl": crawl_name,
                "root_domain": aggregate["root_domain"],
                "tranco_rank": aggregate["tranco_rank"] or "",
                "tranco_matched_domain": aggregate["tranco_matched_domain"],
                "tranco_bucket": aggregate["tranco_bucket"],
                "tranco_match_type": aggregate["tranco_match_type"],
                "rows": metrics["rows"],
                "rows_with_prompt_text": metrics["rows_with_prompt_text"],
                "suspicious_rows": metrics["suspicious_rows"],
                "high_rows": metrics["high_rows"],
                "medium_rows": metrics["medium_rows"],
                "low_rows": metrics["low_rows"],
                "ioc_rows": metrics["ioc_rows"],
                "unique_source_domains": len(aggregate["source_domains"]),
                "unique_source_urls": len(aggregate["source_urls"]),
                "unique_target_urls": len(aggregate["target_urls"]),
                "unique_target_platforms": len(aggregate["target_platforms"]),
                "target_platforms": _join_sorted(aggregate["target_platforms"]),
                "subdomain_count": len(aggregate["source_domains"]),
                "top_source_domain_1": top_source_domains[0] if len(top_source_domains) > 0 else "",
                "top_source_domain_2": top_source_domains[1] if len(top_source_domains) > 1 else "",
                "top_source_url_1": top_source_urls[0] if len(top_source_urls) > 0 else "",
                "top_source_url_2": top_source_urls[1] if len(top_source_urls) > 1 else "",
                "top_labels": _top_items(metrics["label_counts"]),
                "label_distribution_json": _json_dumps(dict(metrics["label_counts"].most_common())),
                "example_target_url": aggregate["example_target_url"],
                "example_prompt_text": aggregate["example_prompt_text"],
            }
        )
    root_domain_rows.sort(key=lambda row: (-int(row["rows"]), -int(row["unique_source_urls"]), row["root_domain"]))

    path_pattern_rows: list[dict[str, Any]] = []
    for aggregate in source_path_patterns.values():
        metrics = aggregate["metrics"]
        path_pattern_rows.append(
            {
                "crawl": crawl_name,
                "root_domain": aggregate["root_domain"],
                "tranco_rank": aggregate["tranco_rank"] or "",
                "tranco_matched_domain": aggregate["tranco_matched_domain"],
                "tranco_bucket": aggregate["tranco_bucket"],
                "tranco_match_type": aggregate["tranco_match_type"],
                "path_template": aggregate["path_template"],
                "page_kind": aggregate["page_kind"],
                "path_depth": aggregate["path_depth"],
                "query_keys": " | ".join(aggregate["query_keys"]),
                "query_key_count": len(aggregate["query_keys"]),
                "rows": metrics["rows"],
                "rows_with_prompt_text": metrics["rows_with_prompt_text"],
                "suspicious_rows": metrics["suspicious_rows"],
                "high_rows": metrics["high_rows"],
                "ioc_rows": metrics["ioc_rows"],
                "unique_source_urls": len(aggregate["source_urls"]),
                "unique_source_domains": len(aggregate["source_domains"]),
                "unique_target_platforms": len(aggregate["target_platforms"]),
                "target_platforms": _join_sorted(aggregate["target_platforms"]),
                "top_labels": _top_items(metrics["label_counts"]),
                "label_distribution_json": _json_dumps(dict(metrics["label_counts"].most_common())),
                "example_source_url_1": aggregate["example_source_urls"][0] if len(aggregate["example_source_urls"]) > 0 else "",
                "example_source_url_2": aggregate["example_source_urls"][1] if len(aggregate["example_source_urls"]) > 1 else "",
                "example_prompt_1": aggregate["example_prompt_values"][0] if len(aggregate["example_prompt_values"]) > 0 else "",
                "example_prompt_2": aggregate["example_prompt_values"][1] if len(aggregate["example_prompt_values"]) > 1 else "",
            }
        )
    path_pattern_rows.sort(key=lambda row: (-int(row["rows"]), row["root_domain"], row["path_template"]))

    source_platform_reuse_rows = [
        {
            "crawl": row["crawl"],
            "source_url": row["source_url"],
            "source_domain": row["source_domain"],
            "root_domain": row["root_domain"],
            "tranco_rank": row["tranco_rank"],
            "tranco_bucket": row["tranco_bucket"],
            "unique_target_platforms": row["unique_target_platforms"],
            "target_platforms": row["target_platforms"],
            "rows": row["rows"],
            "unique_target_urls": row["unique_target_urls"],
            "top_labels": row["top_labels"],
            "example_target_url": row["example_target_url"],
            "example_prompt_text": row["example_prompt_text"],
        }
        for row in source_url_rows
        if int(row["unique_target_platforms"]) > 1
    ]
    source_platform_reuse_rows.sort(key=lambda row: (-int(row["unique_target_platforms"]), -int(row["rows"]), row["source_url"]))

    bucket_totals = {bucket: tranco_bucket_stats.get(bucket, {"metrics": {"rows": 0}})["metrics"]["rows"] for bucket in TRANCO_BUCKET_ORDER}
    platform_totals = {platform: count for platform, count in platform_counts.items()}
    label_totals: dict[str, int] = collections.Counter()
    for (_bucket, label), count in bucket_label_counts.items():
        label_totals[label] = label_totals.get(label, 0) + count

    tranco_bucket_rows: list[dict[str, Any]] = []
    for bucket in TRANCO_BUCKET_ORDER:
        aggregate = tranco_bucket_stats.get(bucket)
        if aggregate is None:
            tranco_bucket_rows.append(
                {
                    "crawl": crawl_name,
                    "tranco_bucket": bucket,
                    "rows": 0,
                    "rows_with_prompt_text": 0,
                    "suspicious_rows": 0,
                    "high_rows": 0,
                    "medium_rows": 0,
                    "low_rows": 0,
                    "ioc_rows": 0,
                    "unique_root_domains": 0,
                    "unique_source_domains": 0,
                    "unique_source_urls": 0,
                    "unique_target_platforms": 0,
                }
            )
            continue
        metrics = aggregate["metrics"]
        tranco_bucket_rows.append(
            {
                "crawl": crawl_name,
                "tranco_bucket": bucket,
                "rows": metrics["rows"],
                "rows_with_prompt_text": metrics["rows_with_prompt_text"],
                "suspicious_rows": metrics["suspicious_rows"],
                "high_rows": metrics["high_rows"],
                "medium_rows": metrics["medium_rows"],
                "low_rows": metrics["low_rows"],
                "ioc_rows": metrics["ioc_rows"],
                "unique_root_domains": len(aggregate["root_domains"]),
                "unique_source_domains": len(aggregate["source_domains"]),
                "unique_source_urls": len(aggregate["source_urls"]),
                "unique_target_platforms": len(aggregate["target_platforms"]),
            }
        )

    tranco_bucket_by_platform_rows: list[dict[str, Any]] = []
    for bucket in TRANCO_BUCKET_ORDER:
        for platform in sorted(platform_counts):
            count = bucket_platform_counts.get((bucket, platform), 0)
            tranco_bucket_by_platform_rows.append(
                {
                    "crawl": crawl_name,
                    "tranco_bucket": bucket,
                    "target_platform": platform,
                    "rows": count,
                    "share_within_bucket": round((count / bucket_totals[bucket]), 6) if bucket_totals[bucket] else 0.0,
                    "share_within_platform": round((count / platform_totals[platform]), 6) if platform_totals.get(platform) else 0.0,
                }
            )

    tranco_bucket_by_label_rows: list[dict[str, Any]] = []
    for bucket in TRANCO_BUCKET_ORDER:
        for label in sorted(label_totals):
            count = bucket_label_counts.get((bucket, label), 0)
            tranco_bucket_by_label_rows.append(
                {
                    "crawl": crawl_name,
                    "tranco_bucket": bucket,
                    "label": label,
                    "rows": count,
                    "share_within_bucket": round((count / bucket_totals[bucket]), 6) if bucket_totals[bucket] else 0.0,
                    "share_within_label": round((count / label_totals[label]), 6) if label_totals.get(label) else 0.0,
                }
            )

    top_tranco_abused_domains_rows = [
        {
            "crawl": row["crawl"],
            "root_domain": row["root_domain"],
            "tranco_rank": row["tranco_rank"],
            "tranco_bucket": row["tranco_bucket"],
            "rows": row["rows"],
            "unique_source_domains": row["unique_source_domains"],
            "unique_source_urls": row["unique_source_urls"],
            "unique_target_platforms": row["unique_target_platforms"],
            "high_rows": row["high_rows"],
            "ioc_rows": row["ioc_rows"],
            "top_labels": row["top_labels"],
            "top_source_domain_1": row["top_source_domain_1"],
            "top_source_url_1": row["top_source_url_1"],
        }
        for row in root_domain_rows
        if row["tranco_rank"] != ""
    ]
    top_tranco_abused_domains_rows.sort(key=lambda row: (-int(row["rows"]), int(row["tranco_rank"]), row["root_domain"]))

    review_top_source_urls = [
        {
            "crawl": row["crawl"],
            "rank": index,
            "source_url": row["source_url"],
            "source_domain": row["source_domain"],
            "root_domain": row["root_domain"],
            "tranco_rank": row["tranco_rank"],
            "tranco_bucket": row["tranco_bucket"],
            "rows": row["rows"],
            "unique_target_platforms": row["unique_target_platforms"],
            "target_platforms": row["target_platforms"],
            "unique_target_urls": row["unique_target_urls"],
            "top_labels": row["top_labels"],
            "example_target_url": row["example_target_url"],
            "example_prompt_text": row["example_prompt_text"],
        }
        for index, row in enumerate(source_url_rows[: max(args.top_n, 1)], start=1)
    ]

    review_top_source_domains = [
        {
            "crawl": row["crawl"],
            "rank": index,
            "source_domain": row["source_domain"],
            "root_domain": row["root_domain"],
            "tranco_rank": row["tranco_rank"],
            "tranco_bucket": row["tranco_bucket"],
            "rows": row["rows"],
            "unique_source_urls": row["unique_source_urls"],
            "unique_target_platforms": row["unique_target_platforms"],
            "target_platforms": row["target_platforms"],
            "top_labels": row["top_labels"],
            "top_source_url_1": row["top_source_url_1"],
            "top_source_url_2": row["top_source_url_2"],
            "example_target_url": row["example_target_url"],
            "example_prompt_text": row["example_prompt_text"],
        }
        for index, row in enumerate(source_domain_rows[: max(args.top_n, 1)], start=1)
    ]

    source_domain_rank_rows: list[dict[str, Any]] = []
    cumulative_rows = 0
    total_rows = rows_analyzed
    for index, row in enumerate(source_domain_rows, start=1):
        cumulative_rows += int(row["rows"])
        source_domain_rank_rows.append(
            {
                "crawl": crawl_name,
                "rank": index,
                "source_domain": row["source_domain"],
                "root_domain": row["root_domain"],
                "tranco_bucket": row["tranco_bucket"],
                "rows": row["rows"],
                "cumulative_rows": cumulative_rows,
                "cumulative_share_rows": round((cumulative_rows / total_rows), 6) if total_rows else 0.0,
            }
        )

    root_domain_intensity_rows = [
        {
            "crawl": row["crawl"],
            "root_domain": row["root_domain"],
            "tranco_rank": row["tranco_rank"],
            "tranco_bucket": row["tranco_bucket"],
            "rows": row["rows"],
            "unique_source_domains": row["unique_source_domains"],
            "unique_source_urls": row["unique_source_urls"],
        }
        for row in root_domain_rows
    ]

    source_url_fieldnames = [
        "crawl", "source_url", "source_domain", "root_domain", "tranco_rank", "tranco_matched_domain",
        "tranco_bucket", "tranco_match_type", "scheme", "path", "path_template", "path_depth", "page_kind",
        "query_keys", "query_key_count", "rows", "rows_with_prompt_text", "suspicious_rows", "high_rows",
        "medium_rows", "low_rows", "ioc_rows", "unique_target_urls", "unique_target_domains",
        "unique_target_platforms", "target_platforms", "label_set_count", "top_labels", "label_distribution_json",
        "example_target_url", "example_prompt_text",
    ]
    source_domain_fieldnames = [
        "crawl", "source_domain", "root_domain", "tranco_rank", "tranco_matched_domain", "tranco_bucket",
        "tranco_match_type", "rows", "rows_with_prompt_text", "suspicious_rows", "high_rows", "medium_rows",
        "low_rows", "ioc_rows", "unique_source_urls", "unique_target_urls", "unique_target_platforms",
        "target_platforms", "top_platform", "top_labels", "label_distribution_json", "top_source_url_1",
        "top_source_url_2", "example_target_url", "example_prompt_text",
    ]
    root_domain_fieldnames = [
        "crawl", "root_domain", "tranco_rank", "tranco_matched_domain", "tranco_bucket", "tranco_match_type",
        "rows", "rows_with_prompt_text", "suspicious_rows", "high_rows", "medium_rows", "low_rows", "ioc_rows",
        "unique_source_domains", "unique_source_urls", "unique_target_urls", "unique_target_platforms",
        "target_platforms", "subdomain_count", "top_source_domain_1", "top_source_domain_2", "top_source_url_1",
        "top_source_url_2", "top_labels", "label_distribution_json", "example_target_url", "example_prompt_text",
    ]
    path_pattern_fieldnames = [
        "crawl", "root_domain", "tranco_rank", "tranco_matched_domain", "tranco_bucket", "tranco_match_type",
        "path_template", "page_kind", "path_depth", "query_keys", "query_key_count", "rows", "rows_with_prompt_text",
        "suspicious_rows", "high_rows", "ioc_rows", "unique_source_urls", "unique_source_domains",
        "unique_target_platforms", "target_platforms", "top_labels", "label_distribution_json",
        "example_source_url_1", "example_source_url_2", "example_prompt_1", "example_prompt_2",
    ]
    bucket_fieldnames = [
        "crawl", "tranco_bucket", "rows", "rows_with_prompt_text", "suspicious_rows", "high_rows", "medium_rows",
        "low_rows", "ioc_rows", "unique_root_domains", "unique_source_domains", "unique_source_urls",
        "unique_target_platforms",
    ]

    write_csv(tables_dir / "source_url_stats.csv", source_url_rows, source_url_fieldnames)
    write_csv(tables_dir / "source_domain_stats.csv", source_domain_rows, source_domain_fieldnames)
    write_csv(tables_dir / "root_domain_stats.csv", root_domain_rows, root_domain_fieldnames)
    write_csv(tables_dir / "source_path_patterns.csv", path_pattern_rows, path_pattern_fieldnames)
    write_csv(
        tables_dir / "source_platform_reuse.csv",
        source_platform_reuse_rows,
        [
            "crawl", "source_url", "source_domain", "root_domain", "tranco_rank", "tranco_bucket",
            "unique_target_platforms", "target_platforms", "rows", "unique_target_urls", "top_labels",
            "example_target_url", "example_prompt_text",
        ],
    )
    write_csv(tables_dir / "tranco_bucket_summary.csv", tranco_bucket_rows, bucket_fieldnames)
    write_csv(
        tables_dir / "tranco_bucket_by_platform.csv",
        tranco_bucket_by_platform_rows,
        ["crawl", "tranco_bucket", "target_platform", "rows", "share_within_bucket", "share_within_platform"],
    )
    write_csv(
        tables_dir / "tranco_bucket_by_label.csv",
        tranco_bucket_by_label_rows,
        ["crawl", "tranco_bucket", "label", "rows", "share_within_bucket", "share_within_label"],
    )
    write_csv(
        tables_dir / "top_tranco_abused_domains.csv",
        top_tranco_abused_domains_rows,
        [
            "crawl", "root_domain", "tranco_rank", "tranco_bucket", "rows", "unique_source_domains",
            "unique_source_urls", "unique_target_platforms", "high_rows", "ioc_rows", "top_labels",
            "top_source_domain_1", "top_source_url_1",
        ],
    )

    write_csv(
        review_dir / "top_source_urls.csv",
        review_top_source_urls,
        [
            "crawl", "rank", "source_url", "source_domain", "root_domain", "tranco_rank", "tranco_bucket",
            "rows", "unique_target_platforms", "target_platforms", "unique_target_urls", "top_labels",
            "example_target_url", "example_prompt_text",
        ],
    )
    write_csv(
        review_dir / "top_source_domains.csv",
        review_top_source_domains,
        [
            "crawl", "rank", "source_domain", "root_domain", "tranco_rank", "tranco_bucket", "rows",
            "unique_source_urls", "unique_target_platforms", "target_platforms", "top_labels", "top_source_url_1",
            "top_source_url_2", "example_target_url", "example_prompt_text",
        ],
    )

    write_csv(
        figure_dir / "source_domain_rank.csv",
        source_domain_rank_rows,
        ["crawl", "rank", "source_domain", "root_domain", "tranco_bucket", "rows", "cumulative_rows", "cumulative_share_rows"],
    )
    write_csv(
        figure_dir / "tranco_bucket_rows.csv",
        [{"crawl": row["crawl"], "tranco_bucket": row["tranco_bucket"], "rows": row["rows"]} for row in tranco_bucket_rows],
        ["crawl", "tranco_bucket", "rows"],
    )
    write_csv(
        figure_dir / "tranco_bucket_unique_root_domains.csv",
        [{"crawl": row["crawl"], "tranco_bucket": row["tranco_bucket"], "unique_root_domains": row["unique_root_domains"]} for row in tranco_bucket_rows],
        ["crawl", "tranco_bucket", "unique_root_domains"],
    )
    write_csv(
        figure_dir / "platform_by_tranco_bucket.csv",
        tranco_bucket_by_platform_rows,
        ["crawl", "tranco_bucket", "target_platform", "rows", "share_within_bucket", "share_within_platform"],
    )
    write_csv(
        figure_dir / "label_by_tranco_bucket.csv",
        tranco_bucket_by_label_rows,
        ["crawl", "tranco_bucket", "label", "rows", "share_within_bucket", "share_within_label"],
    )
    write_csv(
        figure_dir / "root_domain_intensity.csv",
        root_domain_intensity_rows,
        ["crawl", "root_domain", "tranco_rank", "tranco_bucket", "rows", "unique_source_domains", "unique_source_urls"],
    )

    ranked_root_domains = sum(1 for row in root_domain_rows if row["tranco_rank"] != "")
    ranked_source_domains = sum(1 for row in source_domain_rows if row["tranco_rank"] != "")
    multi_platform_source_urls = sum(1 for row in source_url_rows if int(row["unique_target_platforms"]) > 1)

    summary = {
        "generated_at_epoch": iso_now_epoch(),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "crawl": crawl_name,
        "filters": {
            "only_nonempty_prompt": args.only_nonempty_prompt,
            "suspicious_only": args.suspicious_only,
        },
        "tranco": {
            "mode": args.tranco_mode,
            "source": tranco_source,
            "domains_loaded": len(tranco_ranking),
        },
        "quality": {
            "rows_seen": rows_seen,
            "rows_analyzed": rows_analyzed,
            "rows_missing_source_url": rows_missing_source_url,
            "rows_invalid_source_url": rows_invalid_source_url,
            "rows_source_domain_missing": rows_source_domain_missing,
            "rows_source_domain_mismatch": rows_source_domain_mismatch,
            "rows_filtered_empty_prompt": rows_filtered_empty_prompt,
            "rows_filtered_not_suspicious": rows_filtered_not_suspicious,
        },
        "counts": {
            "rows_with_prompt_text": rows_with_prompt_text,
            "rows_suspicious": rows_suspicious,
            "rows_with_ioc_keywords": rows_with_ioc_keywords,
            "unique_source_urls": len(source_url_rows),
            "unique_source_domains": len(source_domain_rows),
            "unique_root_domains": len(root_domain_rows),
            "multi_platform_source_urls": multi_platform_source_urls,
            "ranked_source_domains": ranked_source_domains,
            "ranked_root_domains": ranked_root_domains,
        },
        "distributions": {
            "platform_distribution": counter_to_sorted_rows(dict(platform_counts), total=rows_analyzed, key_name="target_platform"),
            "severity_distribution": counter_to_sorted_rows(dict(severity_counts), total=rows_analyzed, key_name="severity"),
            "label_distribution": counter_to_sorted_rows(dict(label_counts), total=rows_analyzed, key_name="label"),
            "scheme_distribution": counter_to_sorted_rows(dict(scheme_counts), total=rows_analyzed, key_name="scheme"),
            "page_kind_distribution": counter_to_sorted_rows(dict(page_kind_counts), total=rows_analyzed, key_name="page_kind"),
            "tranco_bucket_summary": ordered_bucket_rows(tranco_bucket_rows),
        },
        "files": {
            "summary_json": str(output_dir / "summary.json"),
            "manifest_json": str(output_dir / "manifest.json"),
            "source_url_stats_csv": str(tables_dir / "source_url_stats.csv"),
            "source_domain_stats_csv": str(tables_dir / "source_domain_stats.csv"),
            "root_domain_stats_csv": str(tables_dir / "root_domain_stats.csv"),
            "source_path_patterns_csv": str(tables_dir / "source_path_patterns.csv"),
            "source_platform_reuse_csv": str(tables_dir / "source_platform_reuse.csv"),
            "tranco_bucket_summary_csv": str(tables_dir / "tranco_bucket_summary.csv"),
            "tranco_bucket_by_platform_csv": str(tables_dir / "tranco_bucket_by_platform.csv"),
            "tranco_bucket_by_label_csv": str(tables_dir / "tranco_bucket_by_label.csv"),
            "top_tranco_abused_domains_csv": str(tables_dir / "top_tranco_abused_domains.csv"),
            "top_source_urls_csv": str(review_dir / "top_source_urls.csv"),
            "top_source_domains_csv": str(review_dir / "top_source_domains.csv"),
        },
    }
    manifest = {
        "script": "analyze_source_urls.py",
        "version": 1,
        "crawl": crawl_name,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "tables": sorted(str(path) for path in tables_dir.iterdir()),
        "review": sorted(str(path) for path in review_dir.iterdir()),
        "figure_data": sorted(str(path) for path in figure_dir.iterdir()),
        "summary_json": str(output_dir / "summary.json"),
        "tranco_source": tranco_source,
    }

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "manifest.json", manifest)

    print(
        json.dumps(
            {
                "crawl": crawl_name,
                "rows_seen": rows_seen,
                "rows_analyzed": rows_analyzed,
                "unique_source_urls": len(source_url_rows),
                "unique_source_domains": len(source_domain_rows),
                "unique_root_domains": len(root_domain_rows),
                "ranked_root_domains": ranked_root_domains,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())