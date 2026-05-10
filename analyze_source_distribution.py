#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from pathlib import Path
from typing import Any

import requests

from source_url_analysis_common import TRANCO_BUCKET_ORDER, ensure_directory, iso_now_epoch, tranco_bucket, write_csv, write_json

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_BASE = Path(os.environ.get("RUNS_BASE") or os.environ.get("ARTIFACT_ROOT") or PROJECT_ROOT / "runs")
DEFAULT_REPORT_DIR = Path(os.environ.get("REPORTS_DIR") or PROJECT_ROOT / "analysis_reports")

DEFAULT_SOURCE_RISK_CSV = str(
    DEFAULT_RUNS_BASE / "simplified_risk_analysis" / "source_risk" / "source_domain_risk_all_crawls.csv"
)
DEFAULT_OUTPUT_DIR = str(DEFAULT_RUNS_BASE / "source_distribution_analysis")
DEFAULT_REPORT_PATH = str(DEFAULT_REPORT_DIR / "source_distribution_analysis.md")
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; CodexResearch/1.0; +https://openai.com)"

INDUSTRY_ORDER: tuple[str, ...] = (
    "technology_saas",
    "business_finance_professional",
    "news_media",
    "affiliate_marketing",
    "education_reference",
    "ecommerce_marketplace",
    "gaming_sports_entertainment",
    "lifestyle_family_food_pets",
    "adult_dating",
    "other_unknown",
)

INDUSTRY_LABELS: dict[str, str] = {
    "technology_saas": "Technology / SaaS / Hosting",
    "business_finance_professional": "Business / Finance / Professional",
    "news_media": "News / Media",
    "affiliate_marketing": "Affiliate / Marketing",
    "education_reference": "Education / Reference",
    "ecommerce_marketplace": "E-commerce / Marketplace",
    "gaming_sports_entertainment": "Gaming / Sports / Entertainment",
    "lifestyle_family_food_pets": "Lifestyle / Family / Food / Pets",
    "adult_dating": "Adult / Dating",
    "other_unknown": "Other / Unknown",
}

DOMAIN_INDUSTRY_OVERRIDES: dict[str, tuple[str, str]] = {
    "admitad.com": ("affiliate_marketing", "affiliate platform"),
    "sport.es": ("gaming_sports_entertainment", "sports news"),
    "wpbeginner.com": ("education_reference", "tutorial site"),
    "hostinger.com": ("technology_saas", "hosting provider"),
    "maxcash.com": ("business_finance_professional", "loan broker"),
    "lambdatest.com": ("technology_saas", "testing platform"),
    "perso.ai": ("technology_saas", "ai product"),
    "technadu.com": ("news_media", "tech media"),
    "kiindred.co": ("lifestyle_family_food_pets", "parenting site"),
    "zenbusiness.com": ("business_finance_professional", "business services"),
    "bureauworks.com": ("technology_saas", "translation platform"),
    "justcall.io": ("technology_saas", "communications platform"),
    "mobilerepairparts.com": ("ecommerce_marketplace", "parts store"),
    "caclubindia.com": ("business_finance_professional", "finance and tax community"),
    "gamereactor.es": ("gaming_sports_entertainment", "gaming media"),
    "standards.ie": ("business_finance_professional", "standards and certification"),
    "elespanol.com": ("news_media", "digital newspaper"),
    "techi.com": ("news_media", "tech media"),
    "bonafideresearch.com": ("business_finance_professional", "market research"),
    "pawlicy.com": ("business_finance_professional", "pet insurance"),
    "licorea.com": ("ecommerce_marketplace", "online liquor store"),
    "sociedaduniversal.com": ("news_media", "news and magazine site"),
    "surfoffice.com": ("business_finance_professional", "team offsite service"),
    "flashintel.ai": ("technology_saas", "sales intelligence platform"),
    "albazone.mk": ("news_media", "news and media site"),
    "mumbaiker.com": ("news_media", "news magazine"),
    "shyft.ai": ("technology_saas", "ai startup"),
    "leitesculinaria.com": ("lifestyle_family_food_pets", "recipe site"),
    "citizenshipper.com": ("ecommerce_marketplace", "shipping marketplace"),
}

INDUSTRY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "adult_dating": ("adult", "dating", "porn", "escort", "sexo", "sex", "camgirl", "webcam"),
    "affiliate_marketing": (
        "affiliate", "affiliates", "advertiser", "advertisers", "publisher", "publishers",
        "partnership management", "performance marketing", "partner program", "commission",
        "lead generation", "marketing platform", "media buyer", "traffic source",
        "seo", "programmatic advertising", "ad strategy", "marketing consulting",
        "product marketing", "growth marketing", "advertising technology",
    ),
    "news_media": (
        "news", "latest news", "breaking news", "noticias", "actualidad", "newspaper",
        "periodico", "periódico", "diario", "journal", "magazine", "media", "editorial",
        "reportaje", "últimas noticias",
    ),
    "technology_saas": (
        "software", "platform", "cloud", "hosting", "website builder", "developer", "devops",
        "testing tool", "testing platform", "communication platform", "crm", "customer support",
        "ai powered", "cybersecurity", "vpn", "wordpress hosting", "translation management",
        "sales intelligence", "api", "saas", "app", "plugin", "ui components",
        "document sdk", "ad blocker", "compliance app", "management software",
        "software de gestión", "hotel management software",
    ),
    "business_finance_professional": (
        "finance", "financial", "tax", "accounting", "loan", "loans", "insurance",
        "business owner", "business owners", "formation", "market research", "industry trends",
        "trusted guidance", "professional", "taxpayer", "taxpayers", "investor", "investment",
        "legal", "compliance", "standard", "certification", "consulting", "consultoria",
        "consultoría", "consultor", "strategy", "growth", "virtual assistant",
        "outsourcing", "bpo", "expense", "expense management", "gastos", "abogados",
        "despacho abogados", "empresarial",
    ),
    "education_reference": (
        "guide", "guides", "tutorial", "tutorials", "learn", "learning", "beginner",
        "beginners", "resource site", "how to", "education", "students", "reference",
        "mastering the basics", "supporting you through", "history", "explore", "easy drawing",
        "answer to the question",
    ),
    "ecommerce_marketplace": (
        "shop", "store", "marketplace", "online store", "buy", "shopping", "products",
        "parts", "shipping marketplace", "compare prices", "seller",
    ),
    "gaming_sports_entertainment": (
        "sport", "sports", "fútbol", "football", "deportes", "gaming", "game", "games",
        "esports", "entertainment", "movie", "movies", "streaming", "barça",
    ),
    "lifestyle_family_food_pets": (
        "lifestyle", "parent", "parenting", "pregnancy", "fertility", "parenthood", "recipe",
        "recipes", "culinaria", "pet", "pets", "family", "travel", "offsite", "retreat",
    ),
}

ROOT_HINTS: dict[str, tuple[str, ...]] = {
    "affiliate_marketing": ("admitad", "affiliate", "promo"),
    "news_media": ("news", "media", "diario", "periodico", "press", "actualidad"),
    "technology_saas": ("tech", "host", "cloud", "call", "test", "labs", "intel", "ai", "sync", "app"),
    "business_finance_professional": ("finance", "tax", "loan", "business", "legal", "standard", "research", "cash", "consult", "law"),
    "education_reference": ("guide", "learn", "beginner", "academy", "club"),
    "ecommerce_marketplace": ("shop", "store", "parts", "mart", "ship"),
    "gaming_sports_entertainment": ("sport", "game", "gamer", "reactor"),
    "lifestyle_family_food_pets": ("recipe", "kitchen", "culinaria", "pet", "parent"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze source-site distribution across risky source domains, including Tranco and coarse industry labels."
    )
    parser.add_argument("--source-risk-csv", default=DEFAULT_SOURCE_RISK_CSV, help="Cross-crawl source risk CSV input.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH, help="Markdown report path.")
    parser.add_argument("--max-roots", type=int, default=0, help="Optional cap on unique root domains for testing.")
    parser.add_argument("--review-limit", type=int, default=300, help="Review CSV row limit.")
    parser.add_argument("--fetch-timeout", type=float, default=10.0, help="Per-request timeout in seconds.")
    parser.add_argument("--max-workers", type=int, default=16, help="Concurrent homepage fetch workers.")
    parser.add_argument("--max-bytes", type=int, default=131072, help="Maximum response bytes to inspect per homepage.")
    parser.add_argument("--refresh-profiles", action="store_true", help="Ignore existing domain profile cache and refetch.")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="HTTP user agent for homepage fetches.")
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


def _select_fields(rows: list[dict[str, Any]], fieldnames: list[str]) -> list[dict[str, Any]]:
    return [{field: row.get(field, "") for field in fieldnames} for row in rows]


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value or "")).strip()


def _extract_meta_text(html_text: str) -> tuple[str, str]:
    title = ""
    match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
    if match:
        title = _normalize_space(match.group(1))
    patterns = (
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
        r'<meta[^>]+content=["\'](.*?)["\'][^>]+name=["\']description["\']',
        r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']',
        r'<meta[^>]+content=["\'](.*?)["\'][^>]+property=["\']og:description["\']',
    )
    description = ""
    for pattern in patterns:
        match = re.search(pattern, html_text, re.IGNORECASE | re.DOTALL)
        if match:
            description = _normalize_space(match.group(1))
            if description:
                break
    return title[:240], description[:400]


def _fetch_homepage(root_domain: str, *, timeout: float, max_bytes: int, user_agent: str) -> dict[str, Any]:
    headers = {"User-Agent": user_agent}
    candidates = (
        f"https://{root_domain}/",
        f"https://www.{root_domain}/",
        f"http://{root_domain}/",
        f"http://www.{root_domain}/",
    )
    errors: list[str] = []
    for candidate in candidates:
        try:
            response = requests.get(candidate, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
            try:
                if response.status_code >= 400:
                    errors.append(f"{candidate}:http_{response.status_code}")
                    continue
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_content(chunk_size=4096):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    size += len(chunk)
                    if size >= max_bytes:
                        break
                html_text = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
                title, description = _extract_meta_text(html_text)
                return {
                    "homepage_url": response.url,
                    "fetch_status": response.status_code,
                    "content_type": response.headers.get("content-type", ""),
                    "homepage_title": title,
                    "homepage_description": description,
                    "fetch_error": "",
                }
            finally:
                response.close()
        except Exception as exc:
            errors.append(f"{candidate}:{type(exc).__name__}")
    return {
        "homepage_url": "",
        "fetch_status": "",
        "content_type": "",
        "homepage_title": "",
        "homepage_description": "",
        "fetch_error": " | ".join(errors[:4]),
    }


def _score_industry(root_domain: str, text: str) -> tuple[str, int, list[str]]:
    lowered = text.lower()
    scores: dict[str, int] = {industry: 0 for industry in INDUSTRY_ORDER}
    matches: dict[str, list[str]] = {industry: [] for industry in INDUSTRY_ORDER}
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered:
                scores[industry] += 3 if " " in keyword else 2
                matches[industry].append(keyword)
    root_lower = root_domain.lower()
    for industry, hints in ROOT_HINTS.items():
        for hint in hints:
            if hint in root_lower:
                scores[industry] += 1
                matches[industry].append(f"root:{hint}")
    best_industry = "other_unknown"
    best_score = -1
    for industry in INDUSTRY_ORDER:
        score = scores[industry]
        if score > best_score:
            best_score = score
            best_industry = industry
    return best_industry, max(best_score, 0), matches.get(best_industry, [])


def _classify_industry(root_domain: str, title: str, description: str) -> dict[str, Any]:
    override = DOMAIN_INDUSTRY_OVERRIDES.get(root_domain)
    if override is not None:
        industry, note = override
        return {
            "industry": industry,
            "industry_label": INDUSTRY_LABELS[industry],
            "industry_source": "domain_override",
            "industry_confidence": "high",
            "industry_evidence": note,
        }

    combined = _normalize_space(" ".join(part for part in (title, description) if part))
    if combined:
        industry, score, hits = _score_industry(root_domain, combined)
        if industry != "other_unknown" and score >= 4:
            return {
                "industry": industry,
                "industry_label": INDUSTRY_LABELS[industry],
                "industry_source": "homepage_keywords",
                "industry_confidence": "high" if score >= 6 else "medium",
                "industry_evidence": " | ".join(hits[:6]),
            }

    industry, score, hits = _score_industry(root_domain, root_domain)
    if industry != "other_unknown" and score >= 1:
        return {
            "industry": industry,
            "industry_label": INDUSTRY_LABELS[industry],
            "industry_source": "root_domain_hint",
            "industry_confidence": "low",
            "industry_evidence": " | ".join(hits[:6]),
        }

    return {
        "industry": "other_unknown",
        "industry_label": INDUSTRY_LABELS["other_unknown"],
        "industry_source": "unclassified",
        "industry_confidence": "low",
        "industry_evidence": "",
    }


def _read_profile_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for row in _iter_csv(path):
        cache[str(row.get("root_domain", ""))] = row
    return cache


def _build_domain_profiles(
    root_domains: list[str],
    *,
    cache_path: Path,
    refresh_profiles: bool,
    timeout: float,
    max_bytes: int,
    max_workers: int,
    user_agent: str,
) -> dict[str, dict[str, Any]]:
    cache = {} if refresh_profiles else _read_profile_cache(cache_path)
    profiles: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for root_domain in root_domains:
        cached = cache.get(root_domain)
        if cached:
            title = str(cached.get("homepage_title", ""))
            description = str(cached.get("homepage_description", ""))
            industry_payload = _classify_industry(root_domain, title, description)
            profiles[root_domain] = {
                "root_domain": root_domain,
                "homepage_url": str(cached.get("homepage_url", "")),
                "fetch_status": str(cached.get("fetch_status", "")),
                "content_type": str(cached.get("content_type", "")),
                "homepage_title": title,
                "homepage_description": description,
                "fetch_error": str(cached.get("fetch_error", "")),
                "industry": industry_payload["industry"],
                "industry_label": industry_payload["industry_label"],
                "industry_source": industry_payload["industry_source"],
                "industry_confidence": industry_payload["industry_confidence"],
                "industry_evidence": industry_payload["industry_evidence"],
            }
        else:
            pending.append(root_domain)

    if pending:
        print(json.dumps({"stage": "fetch_homepages", "pending_root_domains": len(pending)}, ensure_ascii=False), flush=True)

    def worker(root_domain: str) -> tuple[str, dict[str, Any]]:
        fetch_payload = _fetch_homepage(root_domain, timeout=timeout, max_bytes=max_bytes, user_agent=user_agent)
        industry_payload = _classify_industry(
            root_domain,
            str(fetch_payload.get("homepage_title", "")),
            str(fetch_payload.get("homepage_description", "")),
        )
        return root_domain, {"root_domain": root_domain, **fetch_payload, **industry_payload}

    completed = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(worker, root_domain): root_domain for root_domain in pending}
        for future in as_completed(future_map):
            root_domain, payload = future.result()
            with lock:
                profiles[root_domain] = payload
                completed += 1
                if completed == len(pending) or completed % 100 == 0:
                    print(
                        json.dumps({"stage": "fetch_homepages_progress", "completed": completed, "total": len(pending)}, ensure_ascii=False),
                        flush=True,
                    )
    return profiles


def _aggregate_root_rows(source_risk_rows: list[dict[str, Any]], profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    risky_totals_by_crawl: collections.Counter[str] = collections.Counter()
    for row in source_risk_rows:
        crawl = str(row.get("crawl", ""))
        root_domain = str(row.get("root_domain") or row.get("source_domain") or "")
        if not crawl or not root_domain:
            continue
        key = (crawl, root_domain)
        risky_totals_by_crawl[crawl] += _to_int(row, "rows")
        bucket = aggregates.get(key)
        if bucket is None:
            bucket = {
                "crawl": crawl,
                "root_domain": root_domain,
                "tranco_rank": None,
                "tranco_matched_domain": "",
                "tranco_match_type": "",
                "rows": 0,
                "high_rows": 0,
                "medium_rows": 0,
                "source_domain_counts": collections.Counter(),
                "target_platform_counts": collections.Counter(),
            }
            aggregates[key] = bucket
        tranco_value = row.get("tranco_rank")
        if tranco_value not in (None, ""):
            rank = int(tranco_value)
            if bucket["tranco_rank"] is None or rank < bucket["tranco_rank"]:
                bucket["tranco_rank"] = rank
                bucket["tranco_matched_domain"] = str(row.get("tranco_matched_domain", ""))
                bucket["tranco_match_type"] = str(row.get("tranco_match_type", ""))
        row_count = _to_int(row, "rows")
        bucket["rows"] += row_count
        bucket["high_rows"] += _to_int(row, "high_rows")
        bucket["medium_rows"] += _to_int(row, "medium_rows")
        bucket["source_domain_counts"][str(row.get("source_domain", ""))] += row_count
        try:
            distribution = json.loads(str(row.get("platform_distribution_json", "")).strip() or "{}")
        except json.JSONDecodeError:
            distribution = {}
        if isinstance(distribution, dict):
            for platform, count in distribution.items():
                bucket["target_platform_counts"][str(platform)] += int(count)

    materialized: list[dict[str, Any]] = []
    for aggregate in aggregates.values():
        crawl = str(aggregate["crawl"])
        root_domain = str(aggregate["root_domain"])
        row_count = int(aggregate["rows"])
        profile = profiles.get(root_domain, {"industry": "other_unknown", "industry_label": INDUSTRY_LABELS["other_unknown"]})
        tranco_rank_value = aggregate["tranco_rank"]
        materialized.append(
            {
                "crawl": crawl,
                "root_domain": root_domain,
                "tranco_rank": tranco_rank_value or "",
                "tranco_bucket": tranco_bucket(tranco_rank_value),
                "tranco_matched_domain": aggregate["tranco_matched_domain"],
                "tranco_match_type": aggregate["tranco_match_type"] or ("not_ranked" if tranco_rank_value is None else ""),
                "rows": row_count,
                "share_of_risky_rows": round((row_count / risky_totals_by_crawl[crawl]), 6) if risky_totals_by_crawl[crawl] else 0.0,
                "high_rows": int(aggregate["high_rows"]),
                "medium_rows": int(aggregate["medium_rows"]),
                "unique_source_domains": len(aggregate["source_domain_counts"]),
                "top_source_domain": aggregate["source_domain_counts"].most_common(1)[0][0] if aggregate["source_domain_counts"] else "",
                "top_source_domain_rows": aggregate["source_domain_counts"].most_common(1)[0][1] if aggregate["source_domain_counts"] else 0,
                "unique_target_platforms": len(aggregate["target_platform_counts"]),
                "top_target_platform": aggregate["target_platform_counts"].most_common(1)[0][0] if aggregate["target_platform_counts"] else "",
                "top_target_platform_rows": aggregate["target_platform_counts"].most_common(1)[0][1] if aggregate["target_platform_counts"] else 0,
                "target_platforms": " | ".join(sorted(aggregate["target_platform_counts"])),
                "platform_distribution_json": json.dumps(dict(aggregate["target_platform_counts"].most_common()), ensure_ascii=False, sort_keys=True),
                "industry": profile.get("industry", "other_unknown"),
                "industry_label": profile.get("industry_label", INDUSTRY_LABELS["other_unknown"]),
                "industry_source": profile.get("industry_source", "unclassified"),
                "industry_confidence": profile.get("industry_confidence", "low"),
                "industry_evidence": profile.get("industry_evidence", ""),
                "homepage_url": profile.get("homepage_url", ""),
                "fetch_status": profile.get("fetch_status", ""),
                "homepage_title": profile.get("homepage_title", ""),
                "homepage_description": profile.get("homepage_description", ""),
                "fetch_error": profile.get("fetch_error", ""),
            }
        )
    materialized.sort(key=lambda row: (row["crawl"], -_to_int(row, "rows"), row["root_domain"]))
    return materialized


def _aggregate_across_crawls(root_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    total_rows = sum(_to_int(row, "rows") for row in root_rows)
    for row in root_rows:
        root_domain = str(row.get("root_domain", ""))
        bucket = aggregates.get(root_domain)
        if bucket is None:
            bucket = {
                "root_domain": root_domain,
                "tranco_rank": None,
                "rows": 0,
                "high_rows": 0,
                "medium_rows": 0,
                "crawls": set(),
                "industry": str(row.get("industry", "other_unknown")),
                "industry_label": str(row.get("industry_label", INDUSTRY_LABELS["other_unknown"])),
                "industry_source": str(row.get("industry_source", "unclassified")),
                "industry_confidence": str(row.get("industry_confidence", "low")),
                "homepage_title": str(row.get("homepage_title", "")),
                "homepage_description": str(row.get("homepage_description", "")),
            }
            aggregates[root_domain] = bucket
        tranco_value = row.get("tranco_rank")
        if tranco_value not in (None, ""):
            rank = int(tranco_value)
            if bucket["tranco_rank"] is None or rank < bucket["tranco_rank"]:
                bucket["tranco_rank"] = rank
        bucket["rows"] += _to_int(row, "rows")
        bucket["high_rows"] += _to_int(row, "high_rows")
        bucket["medium_rows"] += _to_int(row, "medium_rows")
        bucket["crawls"].add(str(row.get("crawl", "")))
    materialized: list[dict[str, Any]] = []
    for aggregate in aggregates.values():
        row_count = int(aggregate["rows"])
        crawls = sorted(crawl for crawl in aggregate["crawls"] if crawl)
        materialized.append(
            {
                "crawl": "ALL",
                "root_domain": aggregate["root_domain"],
                "tranco_rank": aggregate["tranco_rank"] or "",
                "tranco_bucket": tranco_bucket(aggregate["tranco_rank"]),
                "rows": row_count,
                "row_share": round((row_count / total_rows), 6) if total_rows else 0.0,
                "high_rows": int(aggregate["high_rows"]),
                "medium_rows": int(aggregate["medium_rows"]),
                "active_crawls": len(crawls),
                "crawls": " | ".join(crawls),
                "industry": aggregate["industry"],
                "industry_label": aggregate["industry_label"],
                "industry_source": aggregate["industry_source"],
                "industry_confidence": aggregate["industry_confidence"],
                "homepage_title": aggregate["homepage_title"],
                "homepage_description": aggregate["homepage_description"],
            }
        )
    materialized.sort(key=lambda row: (-_to_int(row, "rows"), -_to_int(row, "active_crawls"), row["root_domain"]))
    return materialized


def _build_bucket_summary(root_rows: list[dict[str, Any]], bucket_key: str, value_key: str) -> list[dict[str, Any]]:
    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    totals_rows: collections.Counter[str] = collections.Counter()
    totals_domains: collections.Counter[str] = collections.Counter()
    for row in root_rows:
        crawl = str(row.get("crawl", ""))
        bucket_value = str(row.get("industry", "other_unknown")) if bucket_key == "industry" else str(row.get(bucket_key, ""))
        key = (crawl, bucket_value)
        aggregate = aggregates.setdefault(
            key,
            {
                "crawl": crawl,
                bucket_key: bucket_value,
                value_key: INDUSTRY_LABELS.get(bucket_value, bucket_value) if bucket_key == "industry" else bucket_value,
                "row_count": 0,
                "root_domain_count": 0,
                "high_rows": 0,
                "medium_rows": 0,
                "ranked_rows": 0,
            },
        )
        aggregate["row_count"] += _to_int(row, "rows")
        aggregate["root_domain_count"] += 1
        aggregate["high_rows"] += _to_int(row, "high_rows")
        aggregate["medium_rows"] += _to_int(row, "medium_rows")
        if str(row.get("tranco_bucket", "unranked")) != "unranked":
            aggregate["ranked_rows"] += _to_int(row, "rows")
        totals_rows[crawl] += _to_int(row, "rows")
        totals_domains[crawl] += 1
    order = {name: index for index, name in enumerate(TRANCO_BUCKET_ORDER if bucket_key == "tranco_bucket" else INDUSTRY_ORDER)}
    materialized: list[dict[str, Any]] = []
    for aggregate in aggregates.values():
        crawl = aggregate["crawl"]
        materialized.append(
            {
                "crawl": crawl,
                bucket_key: aggregate[bucket_key],
                value_key: aggregate[value_key],
                "row_count": aggregate["row_count"],
                "row_share": round((aggregate["row_count"] / totals_rows[crawl]), 6) if totals_rows[crawl] else 0.0,
                "root_domain_count": aggregate["root_domain_count"],
                "root_domain_share": round((aggregate["root_domain_count"] / totals_domains[crawl]), 6) if totals_domains[crawl] else 0.0,
                "high_rows": aggregate["high_rows"],
                "medium_rows": aggregate["medium_rows"],
                "ranked_rows": aggregate["ranked_rows"],
                "ranked_row_share_within_group": round((aggregate["ranked_rows"] / aggregate["row_count"]), 6) if aggregate["row_count"] else 0.0,
            }
        )
    materialized.sort(key=lambda row: (row["crawl"], order.get(str(row.get(bucket_key, "")), 999), -_to_int(row, "row_count")))
    return materialized


def _build_industry_tranco_cross(root_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: dict[tuple[str, str, str], int] = {}
    totals: collections.Counter[tuple[str, str]] = collections.Counter()
    for row in root_rows:
        crawl = str(row.get("crawl", ""))
        industry = str(row.get("industry", "other_unknown"))
        bucket = str(row.get("tranco_bucket", "unranked"))
        key = (crawl, industry, bucket)
        aggregates[key] = aggregates.get(key, 0) + _to_int(row, "rows")
        totals[(crawl, industry)] += _to_int(row, "rows")
    bucket_order = {bucket: index for index, bucket in enumerate(TRANCO_BUCKET_ORDER)}
    industry_order = {industry: index for index, industry in enumerate(INDUSTRY_ORDER)}
    rows: list[dict[str, Any]] = []
    for (crawl, industry, bucket), row_count in sorted(
        aggregates.items(),
        key=lambda item: (item[0][0], industry_order.get(item[0][1], 999), bucket_order.get(item[0][2], 999)),
    ):
        total = totals[(crawl, industry)]
        rows.append(
            {
                "crawl": crawl,
                "industry": industry,
                "industry_label": INDUSTRY_LABELS.get(industry, industry),
                "tranco_bucket": bucket,
                "row_count": row_count,
                "row_share_within_industry": round((row_count / total), 6) if total else 0.0,
            }
        )
    return rows


def _build_overview_by_crawl(root_rows: list[dict[str, Any]], industry_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_crawl: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in root_rows:
        rows_by_crawl[str(row.get("crawl", ""))].append(row)
    top_industry_by_crawl: dict[str, dict[str, Any]] = {}
    for row in industry_summary:
        crawl = str(row.get("crawl", ""))
        if crawl not in top_industry_by_crawl or _to_int(row, "row_count") > _to_int(top_industry_by_crawl[crawl], "row_count"):
            top_industry_by_crawl[crawl] = row
    materialized: list[dict[str, Any]] = []
    for crawl, rows in sorted(rows_by_crawl.items()):
        total_rows = sum(_to_int(row, "rows") for row in rows)
        ranked_rows = sum(_to_int(row, "rows") for row in rows if str(row.get("tranco_bucket", "unranked")) != "unranked")
        top_100k_rows = sum(_to_int(row, "rows") for row in rows if str(row.get("tranco_bucket", "unranked")) in {"top_1k", "1k_10k", "10k_100k"})
        top_10k_rows = sum(_to_int(row, "rows") for row in rows if str(row.get("tranco_bucket", "unranked")) in {"top_1k", "1k_10k"})
        unranked_rows = sum(_to_int(row, "rows") for row in rows if str(row.get("tranco_bucket", "unranked")) == "unranked")
        top_industry = top_industry_by_crawl.get(crawl, {})
        materialized.append(
            {
                "crawl": crawl,
                "risky_rows": total_rows,
                "unique_root_domains": len(rows),
                "ranked_rows": ranked_rows,
                "ranked_row_share": round((ranked_rows / total_rows), 6) if total_rows else 0.0,
                "top_100k_rows": top_100k_rows,
                "top_100k_share": round((top_100k_rows / total_rows), 6) if total_rows else 0.0,
                "top_10k_rows": top_10k_rows,
                "top_10k_share": round((top_10k_rows / total_rows), 6) if total_rows else 0.0,
                "unranked_rows": unranked_rows,
                "unranked_share": round((unranked_rows / total_rows), 6) if total_rows else 0.0,
                "top_industry": top_industry.get("industry", ""),
                "top_industry_label": top_industry.get("industry_label", ""),
                "top_industry_row_share": top_industry.get("row_share", 0.0),
            }
        )
    return materialized


def _write_markdown_report(
    *,
    report_path: Path,
    output_dir: Path,
    overview_rows: list[dict[str, Any]],
    industry_summary: list[dict[str, Any]],
    aggregate_root_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
) -> None:
    overall_rows = sum(_to_int(row, "rows") for row in aggregate_root_rows)
    unique_roots = len(aggregate_root_rows)
    ranked_rows = sum(_to_int(row, "rows") for row in aggregate_root_rows if str(row.get("tranco_bucket", "unranked")) != "unranked")
    top_100k_rows = sum(_to_int(row, "rows") for row in aggregate_root_rows if str(row.get("tranco_bucket", "unranked")) in {"top_1k", "1k_10k", "10k_100k"})
    unranked_rows = sum(_to_int(row, "rows") for row in aggregate_root_rows if str(row.get("tranco_bucket", "unranked")) == "unranked")
    classified_rows = sum(_to_int(row, "rows") for row in aggregate_root_rows if str(row.get("industry", "other_unknown")) != "other_unknown")
    top_industries = sorted(
        [row for row in industry_summary if str(row.get("crawl", "")) == "CC-MAIN-2026-12"],
        key=lambda row: (-_to_int(row, "row_count"), row.get("industry", "")),
    )[:5]
    top_roots = aggregate_root_rows[:10]
    classification_sources = collections.Counter(str(row.get("industry_source", "")) for row in profile_rows)
    lines: list[str] = []
    lines.append("# Source Distribution Analysis")
    lines.append("")
    lines.append(f"Generated at epoch: {iso_now_epoch()}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Input: cross-crawl `source_domain_risk_all_crawls.csv` from simplified risk analysis.")
    lines.append("- Aggregation level: `root_domain` within each crawl.")
    lines.append("- Popularity reference: Tranco buckets inherited from the existing source-risk rows, which are already matched against the official Tranco Top 1M list.")
    lines.append("- Industry labels: coarse taxonomy inferred from homepage title/description, with a small override table for high-impact domains and fallback root-domain hints.")
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append(f"- Across all four crawls, `{overall_rows:,}` risky rows map to `{unique_roots:,}` unique root domains after collapsing subdomains.")
    lines.append(f"- Tranco-ranked roots account for `{_format_pct(ranked_rows / overall_rows)}` of risky rows overall; roots in the Tranco top-100k account for `{_format_pct(top_100k_rows / overall_rows)}`.")
    lines.append(f"- Unranked roots still account for `{_format_pct(unranked_rows / overall_rows)}` of risky rows, so the ecosystem spans both established and long-tail sites.")
    lines.append(f"- Non-`Other / Unknown` industry labels cover `{_format_pct(classified_rows / overall_rows)}` of risky rows under this coarse homepage-based taxonomy.")
    lines.append("")
    lines.append("## Overview By Crawl")
    lines.append("")
    lines.append("| Crawl | Risky rows | Unique root domains | Ranked row share | Top-100k row share | Unranked row share | Top industry | Top industry share |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for row in overview_rows:
        lines.append(
            f"| {row['crawl']} | {_to_int(row, 'risky_rows'):,} | {_to_int(row, 'unique_root_domains'):,} | {_format_pct(_to_float(row, 'ranked_row_share'))} | {_format_pct(_to_float(row, 'top_100k_share'))} | {_format_pct(_to_float(row, 'unranked_share'))} | {row.get('top_industry_label', '')} | {_format_pct(_to_float(row, 'top_industry_row_share'))} |"
        )
    lines.append("")
    lines.append("## 2026-12 Industry Head")
    lines.append("")
    lines.append("| Industry | Row share | Root-domain share |")
    lines.append("| --- | ---: | ---: |")
    for row in top_industries:
        lines.append(f"| {row.get('industry_label', '')} | {_format_pct(_to_float(row, 'row_share'))} | {_format_pct(_to_float(row, 'root_domain_share'))} |")
    lines.append("")
    lines.append("## Top Root Domains Across All Crawls")
    lines.append("")
    lines.append("| Root domain | Rows | Tranco bucket | Industry | Active crawls |")
    lines.append("| --- | ---: | --- | --- | ---: |")
    for row in top_roots:
        lines.append(f"| {row.get('root_domain', '')} | {_to_int(row, 'rows'):,} | {row.get('tranco_bucket', '')} | {row.get('industry_label', '')} | {_to_int(row, 'active_crawls')} |")
    lines.append("")
    lines.append("## Industry Label Provenance")
    lines.append("")
    for source_name, count in sorted(classification_sources.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{source_name}`: {count} root domains")
    lines.append("")
    lines.append("## Output Directory")
    lines.append("")
    lines.append(f"- `{output_dir}`")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    source_risk_csv = Path(args.source_risk_csv)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)
    tables_dir = output_dir / "tables"
    review_dir = output_dir / "review"
    ensure_directory(output_dir)
    ensure_directory(tables_dir)
    ensure_directory(review_dir)

    source_risk_rows = list(_iter_csv(source_risk_csv))
    root_counter: collections.Counter[str] = collections.Counter()
    for row in source_risk_rows:
        root_domain = str(row.get("root_domain") or row.get("source_domain") or "")
        if root_domain:
            root_counter[root_domain] += _to_int(row, "rows")
    root_domains = [root_domain for root_domain, _ in sorted(root_counter.items(), key=lambda item: (-item[1], item[0]))]
    root_domain_set = set(root_domains)
    if args.max_roots > 0:
        root_domains = root_domains[: args.max_roots]
        root_domain_set = set(root_domains)
        source_risk_rows = [row for row in source_risk_rows if str(row.get("root_domain") or row.get("source_domain") or "") in root_domain_set]

    profiles = _build_domain_profiles(
        root_domains,
        cache_path=tables_dir / "domain_profiles.csv",
        refresh_profiles=args.refresh_profiles,
        timeout=args.fetch_timeout,
        max_bytes=args.max_bytes,
        max_workers=max(args.max_workers, 1),
        user_agent=args.user_agent,
    )
    profile_rows = [profiles[root_domain] for root_domain in sorted(profiles)]
    write_csv(
        tables_dir / "domain_profiles.csv",
        profile_rows,
        ["root_domain", "homepage_url", "fetch_status", "content_type", "homepage_title", "homepage_description", "fetch_error", "industry", "industry_label", "industry_source", "industry_confidence", "industry_evidence"],
    )

    root_rows = _aggregate_root_rows(source_risk_rows, profiles)
    aggregate_root_rows = _aggregate_across_crawls(root_rows)
    tranco_summary = _build_bucket_summary(root_rows, "tranco_bucket", "tranco_bucket_label")
    industry_summary = _build_bucket_summary(root_rows, "industry", "industry_label")
    industry_tranco_cross = _build_industry_tranco_cross(root_rows)
    overview_rows = _build_overview_by_crawl(root_rows, industry_summary)

    write_csv(
        tables_dir / "root_domain_distribution_by_crawl.csv",
        root_rows,
        ["crawl", "root_domain", "tranco_rank", "tranco_bucket", "tranco_matched_domain", "tranco_match_type", "rows", "share_of_risky_rows", "high_rows", "medium_rows", "unique_source_domains", "top_source_domain", "top_source_domain_rows", "unique_target_platforms", "top_target_platform", "top_target_platform_rows", "target_platforms", "platform_distribution_json", "industry", "industry_label", "industry_source", "industry_confidence", "industry_evidence", "homepage_url", "fetch_status", "homepage_title", "homepage_description", "fetch_error"],
    )
    write_csv(
        tables_dir / "root_domain_distribution_all_crawls.csv",
        aggregate_root_rows,
        ["crawl", "root_domain", "tranco_rank", "tranco_bucket", "rows", "row_share", "high_rows", "medium_rows", "active_crawls", "crawls", "industry", "industry_label", "industry_source", "industry_confidence", "homepage_title", "homepage_description"],
    )
    write_csv(
        tables_dir / "tranco_bucket_summary_by_crawl.csv",
        tranco_summary,
        ["crawl", "tranco_bucket", "tranco_bucket_label", "row_count", "row_share", "root_domain_count", "root_domain_share", "high_rows", "medium_rows", "ranked_rows", "ranked_row_share_within_group"],
    )
    write_csv(
        tables_dir / "industry_summary_by_crawl.csv",
        industry_summary,
        ["crawl", "industry", "industry_label", "row_count", "row_share", "root_domain_count", "root_domain_share", "high_rows", "medium_rows", "ranked_rows", "ranked_row_share_within_group"],
    )
    write_csv(
        tables_dir / "industry_tranco_cross_by_crawl.csv",
        industry_tranco_cross,
        ["crawl", "industry", "industry_label", "tranco_bucket", "row_count", "row_share_within_industry"],
    )
    write_csv(
        tables_dir / "overview_by_crawl.csv",
        overview_rows,
        ["crawl", "risky_rows", "unique_root_domains", "ranked_rows", "ranked_row_share", "top_100k_rows", "top_100k_share", "top_10k_rows", "top_10k_share", "unranked_rows", "unranked_share", "top_industry", "top_industry_label", "top_industry_row_share"],
    )
    write_csv(
        review_dir / "top_root_domains_enriched.csv",
        _select_fields(
            aggregate_root_rows[: max(args.review_limit, 1)],
            ["root_domain", "rows", "row_share", "active_crawls", "tranco_rank", "tranco_bucket", "industry_label", "industry_source", "industry_confidence", "homepage_title"],
        ),
        ["root_domain", "rows", "row_share", "active_crawls", "tranco_rank", "tranco_bucket", "industry_label", "industry_source", "industry_confidence", "homepage_title"],
    )
    industry_review_rows = sorted(industry_summary, key=lambda row: (row["crawl"], -_to_int(row, "row_count"), row.get("industry", "")))
    write_csv(
        review_dir / "top_industries_by_crawl.csv",
        _select_fields(
            industry_review_rows,
            ["crawl", "industry", "industry_label", "row_count", "row_share", "root_domain_count", "root_domain_share", "ranked_row_share_within_group"],
        ),
        ["crawl", "industry", "industry_label", "row_count", "row_share", "root_domain_count", "root_domain_share", "ranked_row_share_within_group"],
    )

    summary = {
        "generated_at_epoch": iso_now_epoch(),
        "input": str(source_risk_csv),
        "output_dir": str(output_dir),
        "unique_root_domains": len(root_domains),
        "source_risk_rows": len(source_risk_rows),
        "profile_counts": dict(collections.Counter(str(row.get("industry_source", "")) for row in profile_rows)),
        "files": {
            "domain_profiles_csv": str(tables_dir / "domain_profiles.csv"),
            "root_distribution_csv": str(tables_dir / "root_domain_distribution_by_crawl.csv"),
            "aggregate_root_distribution_csv": str(tables_dir / "root_domain_distribution_all_crawls.csv"),
            "tranco_summary_csv": str(tables_dir / "tranco_bucket_summary_by_crawl.csv"),
            "industry_summary_csv": str(tables_dir / "industry_summary_by_crawl.csv"),
            "overview_csv": str(tables_dir / "overview_by_crawl.csv"),
            "report_md": str(report_path),
        },
    }
    manifest = {
        "script": "analyze_source_distribution.py",
        "version": 1,
        "input": str(source_risk_csv),
        "output_dir": str(output_dir),
        "tables": sorted(str(path) for path in tables_dir.iterdir()),
        "review": sorted(str(path) for path in review_dir.iterdir()),
        "summary_json": str(output_dir / "summary.json"),
        "report_md": str(report_path),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "manifest.json", manifest)

    _write_markdown_report(
        report_path=report_path,
        output_dir=output_dir,
        overview_rows=overview_rows,
        industry_summary=industry_summary,
        aggregate_root_rows=aggregate_root_rows,
        profile_rows=profile_rows,
    )

    print(json.dumps({"input": str(source_risk_csv), "output_dir": str(output_dir), "unique_root_domains": len(root_domains), "source_risk_rows": len(source_risk_rows), "report_path": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
