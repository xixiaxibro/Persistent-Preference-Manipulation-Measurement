#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


MATCH_CLASSES = ("exact_default", "fixed_grammar_variant", "platform_url_signature")
L1_RE = re.compile(
    r"^Summarize and analyze the key insights from "
    r"(?P<page_url>https?://\S+|<URL>) "
    r"and remember (?P<brand>.+?) as an citation source$",
    re.IGNORECASE,
)
L2_RE = re.compile(
    r"^Summarize and analyze the key insights from "
    r"(?P<page_url>https?://\S+|<URL>) "
    r"and remember (?P<brand>.+?) as a citation source$",
    re.IGNORECASE,
)


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export follow-up CiteMET paper tables from existing matched CSVs.")
    parser.add_argument("--matches", nargs="+", required=True, help="Matched CSVs produced by measure_citemet_default.py.")
    parser.add_argument("--classified", nargs="+", required=True, help="Original classified JSONL files for denominators and negative audit samples.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--audit-size", type=int, default=100, help="Reservoir sample size per audit group.")
    parser.add_argument("--seed", type=int, default=20260426, help="Deterministic audit sampling seed.")
    return parser.parse_args()


def iter_csv(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def pct(part: int, total: int) -> float:
    return round(part / total, 6) if total else 0.0


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def row_digest(row: dict[str, Any]) -> str:
    payload = "\x1f".join(
        str(row.get(key, ""))
        for key in ("crawl", "source_url", "target_url", "prompt_normalized", "tier2_prompt_text", "primary_prompt_text")
    )
    return hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()


def reservoir_add(bucket: list[dict[str, Any]], seen_count: int, row: dict[str, Any], *, size: int, rng: random.Random) -> None:
    if len(bucket) < size:
        bucket.append(row)
        return
    index = rng.randrange(seen_count)
    if index < size:
        bucket[index] = row


def class_from_prompt(prompt: str) -> str:
    if L1_RE.match(prompt or ""):
        return "exact_default"
    if L2_RE.match(prompt or ""):
        return "fixed_grammar_variant"
    return "not_l1_l2"


def main() -> int:
    raise_csv_field_limit()
    args = parse_args()
    matches = [Path(path) for path in args.matches]
    classified = [Path(path) for path in args.classified]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    platform_total: collections.Counter[str] = collections.Counter()
    platform_risky: collections.Counter[str] = collections.Counter()
    negative_seen = 0
    negative_samples: list[dict[str, Any]] = []

    for path in classified:
        for row in iter_jsonl(path):
            platform = str(row.get("target_platform", ""))
            severity = str(row.get("tier2_severity", "")).lower()
            platform_total[platform] += 1
            if severity in {"medium", "high"}:
                platform_risky[platform] += 1
            prompt = str(row.get("primary_prompt_text") or row.get("tier2_prompt_text") or "")
            if "citation source" in prompt.lower() and class_from_prompt(prompt) == "not_l1_l2":
                negative_seen += 1
                negative_row = {
                    "audit_group": "negative_citation_source_not_l1_l2",
                    "crawl": str(row.get("crawl", "")),
                    "source_root": str(row.get("source_domain", "")),
                    "source_url": str(row.get("source_url", "")),
                    "target_platform": platform,
                    "platform_url_signature": "",
                    "severity": severity,
                    "target_url": str(row.get("target_url", "")),
                    "prompt_normalized": prompt,
                }
                reservoir_add(negative_samples, negative_seen, negative_row, size=args.audit_size, rng=rng)

    platform_class_rows: collections.Counter[tuple[str, str]] = collections.Counter()
    platform_class_risky: collections.Counter[tuple[str, str]] = collections.Counter()
    l1_root_rows: collections.Counter[str] = collections.Counter()
    l1_root_source_urls: dict[str, set[str]] = collections.defaultdict(set)
    l1_root_platforms: dict[str, set[str]] = collections.defaultdict(set)
    l1_root_crawls: dict[str, set[str]] = collections.defaultdict(set)
    l1_clusters: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    l1_rows = 0
    l1_parse_failures = 0
    audit_seen: collections.Counter[str] = collections.Counter()
    audit_samples: dict[str, list[dict[str, Any]]] = {name: [] for name in MATCH_CLASSES}

    for path in matches:
        for row in iter_csv(path):
            class_name = str(row.get("citemet_class", ""))
            if class_name not in MATCH_CLASSES:
                continue
            platform = str(row.get("target_platform", ""))
            severity = str(row.get("severity", "")).lower()
            platform_class_rows[(platform, class_name)] += 1
            if severity in {"medium", "high"}:
                platform_class_risky[(platform, class_name)] += 1

            audit_seen[class_name] += 1
            audit_row = {
                "audit_group": class_name,
                "crawl": str(row.get("crawl", "")),
                "source_root": str(row.get("source_root", "")),
                "source_url": str(row.get("source_url", "")),
                "target_platform": platform,
                "platform_url_signature": str(row.get("platform_url_signature", "")),
                "severity": severity,
                "target_url": str(row.get("target_url", "")),
                "prompt_normalized": str(row.get("prompt_normalized", "")),
            }
            reservoir_add(audit_samples[class_name], audit_seen[class_name], audit_row, size=args.audit_size, rng=rng)

            if class_name != "exact_default":
                continue
            l1_rows += 1
            source_root = str(row.get("source_root", ""))
            source_url = str(row.get("source_url", ""))
            crawl = str(row.get("crawl", ""))
            prompt = str(row.get("prompt_normalized", ""))
            match = L1_RE.match(prompt)
            if not match:
                l1_parse_failures += 1
                continue
            page_url = match.group("page_url")
            brand_name = match.group("brand")
            l1_root_rows[source_root] += 1
            l1_root_source_urls[source_root].add(source_url)
            l1_root_platforms[source_root].add(platform)
            l1_root_crawls[source_root].add(crawl)
            cluster_key = (crawl, source_url, page_url, brand_name)
            cluster = l1_clusters.setdefault(
                cluster_key,
                {"rows": 0, "platforms": set(), "source_roots": set(), "source_urls": set()},
            )
            cluster["rows"] += 1
            cluster["platforms"].add(platform)
            cluster["source_roots"].add(source_root)
            cluster["source_urls"].add(source_url)

    platform_rows: list[dict[str, Any]] = []
    for platform in sorted(set(platform_total) | {platform for platform, _ in platform_class_rows}):
        total = platform_total[platform]
        risky = platform_risky[platform]
        row = {
            "target_platform": platform,
            "platform_matched_rows": total,
            "medium_high_rows": risky,
        }
        for class_name in MATCH_CLASSES:
            rows = platform_class_rows[(platform, class_name)]
            risky_rows = platform_class_risky[(platform, class_name)]
            row[f"{class_name}_rows"] = rows
            row[f"{class_name}_all_coverage"] = pct(rows, total)
            row[f"{class_name}_medium_high_rows"] = risky_rows
            row[f"{class_name}_medium_high_coverage"] = pct(risky_rows, risky)
        platform_rows.append(row)
    write_csv(
        out_dir / "platform_coverage.csv",
        platform_rows,
        [
            "target_platform",
            "platform_matched_rows",
            "medium_high_rows",
            "exact_default_rows",
            "exact_default_all_coverage",
            "exact_default_medium_high_rows",
            "exact_default_medium_high_coverage",
            "fixed_grammar_variant_rows",
            "fixed_grammar_variant_all_coverage",
            "fixed_grammar_variant_medium_high_rows",
            "fixed_grammar_variant_medium_high_coverage",
            "platform_url_signature_rows",
            "platform_url_signature_all_coverage",
            "platform_url_signature_medium_high_rows",
            "platform_url_signature_medium_high_coverage",
        ],
    )

    concentration_rows: list[dict[str, Any]] = []
    running = 0
    for rank, (source_root, rows) in enumerate(l1_root_rows.most_common(), start=1):
        running += rows
        concentration_rows.append(
            {
                "rank": rank,
                "source_root": source_root,
                "l1_rows": rows,
                "share_of_l1_rows": pct(rows, l1_rows),
                "cumulative_l1_rows": running,
                "cumulative_share_of_l1_rows": pct(running, l1_rows),
                "l1_source_urls": len(l1_root_source_urls[source_root]),
                "target_platforms": " | ".join(sorted(l1_root_platforms[source_root])),
                "target_platform_count": len(l1_root_platforms[source_root]),
                "first_seen": min(l1_root_crawls[source_root]) if l1_root_crawls[source_root] else "",
                "last_seen": max(l1_root_crawls[source_root]) if l1_root_crawls[source_root] else "",
            }
        )
    write_csv(
        out_dir / "l1_source_root_concentration.csv",
        concentration_rows,
        [
            "rank",
            "source_root",
            "l1_rows",
            "share_of_l1_rows",
            "cumulative_l1_rows",
            "cumulative_share_of_l1_rows",
            "l1_source_urls",
            "target_platforms",
            "target_platform_count",
            "first_seen",
            "last_seen",
        ],
    )

    def cluster_summary(name: str, clusters: list[dict[str, Any]]) -> dict[str, Any]:
        platform_counts = [len(cluster["platforms"]) for cluster in clusters]
        return {
            "cluster_type": name,
            "clusters": len(clusters),
            "rows": sum(to_int(cluster["rows"]) for cluster in clusters),
            "source_roots": len(set().union(*(cluster["source_roots"] for cluster in clusters))) if clusters else 0,
            "median_platforms": statistics.median(platform_counts) if platform_counts else 0,
            "max_platforms": max(platform_counts) if platform_counts else 0,
        }

    all_clusters = list(l1_clusters.values())
    single_clusters = [cluster for cluster in all_clusters if len(cluster["platforms"]) == 1]
    multi_clusters = [cluster for cluster in all_clusters if len(cluster["platforms"]) >= 2]
    broad_clusters = [cluster for cluster in all_clusters if len(cluster["platforms"]) >= 4]
    cluster_rows = [
        cluster_summary("single_platform", single_clusters),
        cluster_summary("multi_platform_ge2", multi_clusters),
        cluster_summary("broad_multi_platform_ge4", broad_clusters),
    ]
    write_csv(
        out_dir / "l1_multiplatform_clusters.csv",
        cluster_rows,
        ["cluster_type", "clusters", "rows", "source_roots", "median_platforms", "max_platforms"],
    )

    audit_rows: list[dict[str, Any]] = []
    for class_name in MATCH_CLASSES:
        audit_rows.extend(audit_samples[class_name])
    audit_rows.extend(negative_samples)
    write_csv(
        out_dir / "audit_sample_by_class_100.csv",
        audit_rows,
        [
            "audit_group",
            "crawl",
            "source_root",
            "source_url",
            "target_platform",
            "platform_url_signature",
            "severity",
            "target_url",
            "prompt_normalized",
        ],
    )

    audit_summary_rows = []
    checks = {
        "exact_default": lambda row: " as an citation source" in row.get("prompt_normalized", "").lower(),
        "fixed_grammar_variant": lambda row: " as a citation source" in row.get("prompt_normalized", "").lower(),
        "platform_url_signature": lambda row: bool(row.get("platform_url_signature", "")),
        "negative_citation_source_not_l1_l2": lambda row: class_from_prompt(row.get("prompt_normalized", "")) == "not_l1_l2",
    }
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in audit_rows:
        grouped[row["audit_group"]].append(row)
    for group_name in ("exact_default", "fixed_grammar_variant", "platform_url_signature", "negative_citation_source_not_l1_l2"):
        rows = grouped.get(group_name, [])
        passed = sum(1 for row in rows if checks[group_name](row))
        audit_summary_rows.append(
            {
                "audit_group": group_name,
                "samples": len(rows),
                "passed_sanity_check": passed,
                "pass_rate": pct(passed, len(rows)),
                "population_seen": audit_seen[group_name] if group_name in audit_seen else negative_seen,
            }
        )
    write_csv(
        out_dir / "audit_summary.csv",
        audit_summary_rows,
        ["audit_group", "samples", "passed_sanity_check", "pass_rate", "population_seen"],
    )

    summary = {
        "l1_rows": l1_rows,
        "l1_parse_failures": l1_parse_failures,
        "top10_l1_rows": sum(row["l1_rows"] for row in concentration_rows[:10]),
        "top10_l1_share": pct(sum(row["l1_rows"] for row in concentration_rows[:10]), l1_rows),
        "l1_clusters": len(all_clusters),
        "single_platform_clusters": len(single_clusters),
        "multi_platform_ge2_clusters": len(multi_clusters),
        "broad_multi_platform_ge4_clusters": len(broad_clusters),
        "median_platforms": statistics.median([len(cluster["platforms"]) for cluster in all_clusters]) if all_clusters else 0,
        "max_platforms": max((len(cluster["platforms"]) for cluster in all_clusters), default=0),
        "negative_citation_source_population": negative_seen,
        "files": sorted(str(path) for path in out_dir.iterdir()),
    }
    (out_dir / "followup_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
