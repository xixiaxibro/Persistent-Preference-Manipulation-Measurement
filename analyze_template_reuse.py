#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from risk_analysis_common import iter_jsonl_rows, normalize_string, row_labels, row_severity, row_source_domain, row_target_platform
from source_url_analysis_common import ensure_directory, iso_now_epoch, write_csv, write_json

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
WHITESPACE_RE = re.compile(r"\s+")
PROMPT_KEYS: tuple[str, ...] = ("primary_prompt_text", "prompt_text", "text")
SQLITE_BATCH_SIZE = 5000
PROGRESS_EVERY_ROWS = 500000
MAX_PROMPT_CHARS = 500
TOP_HEAD_LIMIT = 5
DEFAULT_REVIEW_LIMIT = 500

TEMPLATE_INSERT_SQL = """
INSERT OR IGNORE INTO templates (
    template_hash,
    normalized_prompt,
    prompt_length
) VALUES (?, ?, ?)
"""

TEMPLATE_UPDATE_COUNTS_SQL = """
UPDATE templates
SET row_count = row_count + ?,
    risky_row_count = risky_row_count + ?,
    medium_row_count = medium_row_count + ?,
    high_row_count = high_row_count + ?
WHERE template_hash = ?
"""

SOURCE_INSERT_SQL = """
INSERT OR IGNORE INTO template_source_stats (
    template_hash,
    source_domain
) VALUES (?, ?)
"""

SOURCE_UPDATE_COUNTS_SQL = """
UPDATE template_source_stats
SET row_count = row_count + ?,
    risky_row_count = risky_row_count + ?
WHERE template_hash = ? AND source_domain = ?
"""

PLATFORM_INSERT_SQL = """
INSERT OR IGNORE INTO template_platform_stats (
    template_hash,
    target_platform
) VALUES (?, ?)
"""

PLATFORM_UPDATE_COUNTS_SQL = """
UPDATE template_platform_stats
SET row_count = row_count + ?,
    risky_row_count = risky_row_count + ?
WHERE template_hash = ? AND target_platform = ?
"""

LABEL_INSERT_SQL = """
INSERT OR IGNORE INTO template_label_stats (
    template_hash,
    label
) VALUES (?, ?)
"""

LABEL_UPDATE_COUNTS_SQL = """
UPDATE template_label_stats
SET row_count = row_count + ?,
    risky_row_count = risky_row_count + ?,
    medium_row_count = medium_row_count + ?,
    high_row_count = high_row_count + ?
WHERE template_hash = ? AND label = ?
"""

TEMPLATE_OVERVIEW_FIELDS: list[str] = [
    "crawl",
    "template_hash",
    "row_count",
    "row_share",
    "risky_row_count",
    "risky_share_of_template",
    "medium_row_count",
    "high_row_count",
    "unique_source_domains",
    "top_source_domain",
    "top_source_domain_rows",
    "top_source_domain_share",
    "source_domain_head_json",
    "unique_target_platforms",
    "top_target_platform",
    "top_target_platform_rows",
    "top_target_platform_share",
    "target_platform_head_json",
    "unique_labels",
    "top_label",
    "top_label_rows",
    "top_label_share",
    "label_head_json",
    "prompt_length",
    "sample_prompt",
]

REVIEW_FIELDS: list[str] = [
    "crawl",
    "template_hash",
    "row_count",
    "row_share",
    "risky_row_count",
    "risky_share_of_template",
    "unique_source_domains",
    "top_source_domain",
    "top_source_domain_share",
    "unique_target_platforms",
    "top_target_platform",
    "top_target_platform_share",
    "unique_labels",
    "top_label",
    "top_label_share",
    "sample_prompt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze exact prompt-template reuse from Stage 02 classified prompt-link outputs.")
    parser.add_argument("--input", required=True, help="Input classified JSONL or JSONL.GZ")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--crawl-name", default="", help="Optional crawl name override")
    parser.add_argument("--review-limit", type=int, default=DEFAULT_REVIEW_LIMIT, help="Maximum rows in each review CSV")
    return parser.parse_args()


def prompt_from_row(row: dict[str, Any]) -> str:
    for key in PROMPT_KEYS:
        value = normalize_string(row.get(key))
        if value:
            return value
    return ""


def normalize_prompt_text(value: str) -> str:
    text = normalize_string(value)
    if not text:
        return ""
    text = html.unescape(text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = URL_RE.sub("<URL>", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def template_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def is_risky_severity(severity: str) -> bool:
    return severity in {"medium", "high"}


def clip_prompt(text: str, limit: int = MAX_PROMPT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _make_template_bucket(normalized_prompt: str) -> dict[str, Any]:
    return {
        "normalized_prompt": normalized_prompt,
        "prompt_length": len(normalized_prompt),
        "row_count": 0,
        "risky_row_count": 0,
        "medium_row_count": 0,
        "high_row_count": 0,
    }


def _make_stat_bucket() -> dict[str, int]:
    return {"row_count": 0, "risky_row_count": 0, "medium_row_count": 0, "high_row_count": 0}


def _update_bucket(bucket: dict[str, int], severity: str) -> None:
    bucket["row_count"] += 1
    if severity == "medium":
        bucket["risky_row_count"] += 1
        bucket["medium_row_count"] += 1
    elif severity == "high":
        bucket["risky_row_count"] += 1
        bucket["high_row_count"] += 1

def setup_database(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA cache_size = -200000")
    connection.executescript(
        """
        CREATE TABLE templates (
            template_hash TEXT PRIMARY KEY,
            normalized_prompt TEXT NOT NULL,
            prompt_length INTEGER NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            risky_row_count INTEGER NOT NULL DEFAULT 0,
            medium_row_count INTEGER NOT NULL DEFAULT 0,
            high_row_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE template_source_stats (
            template_hash TEXT NOT NULL,
            source_domain TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            risky_row_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (template_hash, source_domain)
        );
        CREATE TABLE template_platform_stats (
            template_hash TEXT NOT NULL,
            target_platform TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            risky_row_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (template_hash, target_platform)
        );
        CREATE TABLE template_label_stats (
            template_hash TEXT NOT NULL,
            label TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            risky_row_count INTEGER NOT NULL DEFAULT 0,
            medium_row_count INTEGER NOT NULL DEFAULT 0,
            high_row_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (template_hash, label)
        );
        """
    )
    return connection


def flush_batches(
    connection: sqlite3.Connection,
    template_batch: dict[str, dict[str, Any]],
    source_batch: dict[tuple[str, str], dict[str, int]],
    platform_batch: dict[tuple[str, str], dict[str, int]],
    label_batch: dict[tuple[str, str], dict[str, int]],
) -> None:
    if not template_batch and not source_batch and not platform_batch and not label_batch:
        return
    with connection:
        if template_batch:
            connection.executemany(
                TEMPLATE_INSERT_SQL,
                [
                    (
                        hash_value,
                        payload["normalized_prompt"],
                        payload["prompt_length"],
                    )
                    for hash_value, payload in template_batch.items()
                ],
            )
            connection.executemany(
                TEMPLATE_UPDATE_COUNTS_SQL,
                [
                    (
                        payload["row_count"],
                        payload["risky_row_count"],
                        payload["medium_row_count"],
                        payload["high_row_count"],
                        hash_value,
                    )
                    for hash_value, payload in template_batch.items()
                ],
            )
        if source_batch:
            connection.executemany(
                SOURCE_INSERT_SQL,
                [(hash_value, source_domain) for (hash_value, source_domain) in source_batch.keys()],
            )
            connection.executemany(
                SOURCE_UPDATE_COUNTS_SQL,
                [
                    (
                        payload["row_count"],
                        payload["risky_row_count"],
                        hash_value,
                        source_domain,
                    )
                    for (hash_value, source_domain), payload in source_batch.items()
                ],
            )
        if platform_batch:
            connection.executemany(
                PLATFORM_INSERT_SQL,
                [(hash_value, target_platform) for (hash_value, target_platform) in platform_batch.keys()],
            )
            connection.executemany(
                PLATFORM_UPDATE_COUNTS_SQL,
                [
                    (
                        payload["row_count"],
                        payload["risky_row_count"],
                        hash_value,
                        target_platform,
                    )
                    for (hash_value, target_platform), payload in platform_batch.items()
                ],
            )
        if label_batch:
            connection.executemany(
                LABEL_INSERT_SQL,
                [(hash_value, label) for (hash_value, label) in label_batch.keys()],
            )
            connection.executemany(
                LABEL_UPDATE_COUNTS_SQL,
                [
                    (
                        payload["row_count"],
                        payload["risky_row_count"],
                        payload["medium_row_count"],
                        payload["high_row_count"],
                        hash_value,
                        label,
                    )
                    for (hash_value, label), payload in label_batch.items()
                ],
            )
    template_batch.clear()
    source_batch.clear()
    platform_batch.clear()
    label_batch.clear()


def fetch_rows(connection: sqlite3.Connection, query: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query, parameters).fetchall()]


def _build_base_overview(connection: sqlite3.Connection, crawl_name: str) -> dict[str, dict[str, Any]]:
    overview: dict[str, dict[str, Any]] = {}
    for row in fetch_rows(
        connection,
        """
        SELECT
            template_hash,
            normalized_prompt,
            prompt_length,
            row_count,
            risky_row_count,
            medium_row_count,
            high_row_count
        FROM templates
        ORDER BY row_count DESC, template_hash
        """,
    ):
        hash_value = str(row["template_hash"])
        row_count = int(row["row_count"] or 0)
        risky_row_count = int(row["risky_row_count"] or 0)
        overview[hash_value] = {
            "crawl": crawl_name,
            "template_hash": hash_value,
            "row_count": row_count,
            "row_share": 0.0,
            "risky_row_count": risky_row_count,
            "risky_share_of_template": round((risky_row_count / row_count), 6) if row_count else 0.0,
            "medium_row_count": int(row["medium_row_count"] or 0),
            "high_row_count": int(row["high_row_count"] or 0),
            "unique_source_domains": 0,
            "top_source_domain": "",
            "top_source_domain_rows": 0,
            "top_source_domain_share": 0.0,
            "source_domain_head": [],
            "unique_target_platforms": 0,
            "top_target_platform": "",
            "top_target_platform_rows": 0,
            "top_target_platform_share": 0.0,
            "target_platform_head": [],
            "unique_labels": 0,
            "top_label": "",
            "top_label_rows": 0,
            "top_label_share": 0.0,
            "label_head": [],
            "prompt_length": int(row["prompt_length"] or 0),
            "sample_prompt": clip_prompt(str(row["normalized_prompt"] or "")),
        }
    return overview


def _append_head(head: list[dict[str, Any]], value: dict[str, Any]) -> None:
    if len(head) < TOP_HEAD_LIMIT:
        head.append(value)

def build_template_overview(connection: sqlite3.Connection, crawl_name: str) -> list[dict[str, Any]]:
    overview = _build_base_overview(connection, crawl_name)
    total_rows = sum(int(entry["row_count"]) for entry in overview.values())

    for row in fetch_rows(
        connection,
        """
        SELECT template_hash, source_domain, row_count, risky_row_count
        FROM template_source_stats
        ORDER BY template_hash, row_count DESC, source_domain
        """,
    ):
        entry = overview[str(row["template_hash"])]
        entry["unique_source_domains"] += 1
        source_domain = str(row["source_domain"])
        row_count = int(row["row_count"] or 0)
        if not entry["top_source_domain"]:
            entry["top_source_domain"] = source_domain
            entry["top_source_domain_rows"] = row_count
        _append_head(
            entry["source_domain_head"],
            {"source_domain": source_domain, "row_count": row_count, "risky_row_count": int(row["risky_row_count"] or 0)},
        )

    for row in fetch_rows(
        connection,
        """
        SELECT template_hash, target_platform, row_count, risky_row_count
        FROM template_platform_stats
        ORDER BY template_hash, row_count DESC, target_platform
        """,
    ):
        entry = overview[str(row["template_hash"])]
        entry["unique_target_platforms"] += 1
        target_platform = str(row["target_platform"])
        row_count = int(row["row_count"] or 0)
        if not entry["top_target_platform"]:
            entry["top_target_platform"] = target_platform
            entry["top_target_platform_rows"] = row_count
        _append_head(
            entry["target_platform_head"],
            {"target_platform": target_platform, "row_count": row_count, "risky_row_count": int(row["risky_row_count"] or 0)},
        )

    for row in fetch_rows(
        connection,
        """
        SELECT template_hash, label, row_count, risky_row_count, medium_row_count, high_row_count
        FROM template_label_stats
        ORDER BY template_hash, row_count DESC, label
        """,
    ):
        entry = overview[str(row["template_hash"])]
        entry["unique_labels"] += 1
        label = str(row["label"])
        row_count = int(row["row_count"] or 0)
        if not entry["top_label"]:
            entry["top_label"] = label
            entry["top_label_rows"] = row_count
        _append_head(
            entry["label_head"],
            {
                "label": label,
                "row_count": row_count,
                "risky_row_count": int(row["risky_row_count"] or 0),
                "medium_row_count": int(row["medium_row_count"] or 0),
                "high_row_count": int(row["high_row_count"] or 0),
            },
        )

    rows: list[dict[str, Any]] = []
    for entry in overview.values():
        row_count = int(entry["row_count"])
        entry["row_share"] = round((row_count / total_rows), 6) if total_rows else 0.0
        entry["top_source_domain_share"] = round((int(entry["top_source_domain_rows"]) / row_count), 6) if row_count else 0.0
        entry["top_target_platform_share"] = round((int(entry["top_target_platform_rows"]) / row_count), 6) if row_count else 0.0
        entry["top_label_share"] = round((int(entry["top_label_rows"]) / row_count), 6) if row_count else 0.0
        rows.append(
            {
                **{field: entry.get(field, "") for field in TEMPLATE_OVERVIEW_FIELDS if field not in {"source_domain_head_json", "target_platform_head_json", "label_head_json"}},
                "source_domain_head_json": _json_dumps(entry["source_domain_head"]),
                "target_platform_head_json": _json_dumps(entry["target_platform_head"]),
                "label_head_json": _json_dumps(entry["label_head"]),
            }
        )

    rows.sort(
        key=lambda row: (
            -int(row["row_count"]),
            -int(row["risky_row_count"]),
            -int(row["unique_source_domains"]),
            row["template_hash"],
        )
    )
    return rows


def build_concentration_rows(template_rows: list[dict[str, Any]], crawl_name: str) -> list[dict[str, Any]]:
    total_rows = sum(int(row["row_count"]) for row in template_rows)
    total_risky_rows = sum(int(row["risky_row_count"]) for row in template_rows)
    targets = [(10, "top_10"), (100, "top_100"), (1000, "top_1000")]
    rows: list[dict[str, Any]] = []
    for limit, scope in targets:
        selected = template_rows[: min(limit, len(template_rows))]
        row_count = sum(int(row["row_count"]) for row in selected)
        risky_row_count = sum(int(row["risky_row_count"]) for row in selected)
        rows.append(
            {
                "crawl": crawl_name,
                "scope": scope,
                "template_count": len(selected),
                "row_count": row_count,
                "row_share": round((row_count / total_rows), 6) if total_rows else 0.0,
                "risky_row_count": risky_row_count,
                "risky_row_share": round((risky_row_count / total_risky_rows), 6) if total_risky_rows else 0.0,
            }
        )
    return rows


def reuse_bucket_for_count(count: int) -> str:
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 5:
        return "3_5"
    if count <= 10:
        return "6_10"
    if count <= 50:
        return "11_50"
    if count <= 100:
        return "51_100"
    if count <= 1000:
        return "101_1000"
    return "gt_1000"


def build_reuse_bucket_rows(template_rows: list[dict[str, Any]], crawl_name: str) -> list[dict[str, Any]]:
    template_total = len(template_rows)
    row_total = sum(int(row["row_count"]) for row in template_rows)
    risky_row_total = sum(int(row["risky_row_count"]) for row in template_rows)
    buckets: dict[str, dict[str, int]] = collections.OrderedDict(
        (name, {"template_count": 0, "row_count": 0, "risky_row_count": 0})
        for name in ("1", "2", "3_5", "6_10", "11_50", "51_100", "101_1000", "gt_1000")
    )
    for row in template_rows:
        bucket = buckets[reuse_bucket_for_count(int(row["row_count"]))]
        bucket["template_count"] += 1
        bucket["row_count"] += int(row["row_count"])
        bucket["risky_row_count"] += int(row["risky_row_count"])
    return [
        {
            "crawl": crawl_name,
            "reuse_bucket": name,
            "template_count": payload["template_count"],
            "template_share": round((payload["template_count"] / template_total), 6) if template_total else 0.0,
            "row_count": payload["row_count"],
            "row_share": round((payload["row_count"] / row_total), 6) if row_total else 0.0,
            "risky_row_count": payload["risky_row_count"],
            "risky_row_share": round((payload["risky_row_count"] / risky_row_total), 6) if risky_row_total else 0.0,
        }
        for name, payload in buckets.items()
    ]


def _select_fields(rows: list[dict[str, Any]], fieldnames: list[str]) -> list[dict[str, Any]]:
    return [{field: row.get(field, "") for field in fieldnames} for row in rows]

def main() -> int:
    args = parse_args()
    started_at = time.time()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    review_dir = output_dir / "review"
    work_dir = output_dir / "work"
    ensure_directory(output_dir)
    ensure_directory(tables_dir)
    ensure_directory(review_dir)
    ensure_directory(work_dir)

    db_path = work_dir / "template_reuse.sqlite3"
    connection = setup_database(db_path)

    rows_seen = 0
    rows_with_prompt = 0
    rows_missing_prompt = 0
    risky_rows = 0
    risky_rows_with_prompt = 0
    detected_crawl_name = ""

    template_batch: dict[str, dict[str, Any]] = {}
    source_batch: dict[tuple[str, str], dict[str, int]] = {}
    platform_batch: dict[tuple[str, str], dict[str, int]] = {}
    label_batch: dict[tuple[str, str], dict[str, int]] = {}

    for row in iter_jsonl_rows(input_path):
        rows_seen += 1
        if not detected_crawl_name:
            detected_crawl_name = normalize_string(row.get("crawl"))

        severity = row_severity(row)
        if is_risky_severity(severity):
            risky_rows += 1

        prompt_text = prompt_from_row(row)
        if not prompt_text:
            rows_missing_prompt += 1
            continue
        normalized_prompt = normalize_prompt_text(prompt_text)
        if not normalized_prompt:
            rows_missing_prompt += 1
            continue

        rows_with_prompt += 1
        if is_risky_severity(severity):
            risky_rows_with_prompt += 1

        hash_value = template_hash(normalized_prompt)
        source_domain = row_source_domain(row) or "(unknown)"
        target_platform = row_target_platform(row) or "(unknown)"
        labels = row_labels(row) or ["(none)"]

        template_stats = template_batch.get(hash_value)
        if template_stats is None:
            template_stats = _make_template_bucket(normalized_prompt)
            template_batch[hash_value] = template_stats
        _update_bucket(template_stats, severity)

        source_key = (hash_value, source_domain)
        source_stats = source_batch.get(source_key)
        if source_stats is None:
            source_stats = _make_stat_bucket()
            source_batch[source_key] = source_stats
        _update_bucket(source_stats, severity)

        platform_key = (hash_value, target_platform)
        platform_stats = platform_batch.get(platform_key)
        if platform_stats is None:
            platform_stats = _make_stat_bucket()
            platform_batch[platform_key] = platform_stats
        _update_bucket(platform_stats, severity)

        for label in labels:
            label_key = (hash_value, label)
            label_stats = label_batch.get(label_key)
            if label_stats is None:
                label_stats = _make_stat_bucket()
                label_batch[label_key] = label_stats
            _update_bucket(label_stats, severity)

        if rows_seen % SQLITE_BATCH_SIZE == 0:
            flush_batches(connection, template_batch, source_batch, platform_batch, label_batch)
        if rows_seen % PROGRESS_EVERY_ROWS == 0:
            print(
                json.dumps(
                    {
                        "stage": "index",
                        "rows_seen": rows_seen,
                        "rows_with_prompt": rows_with_prompt,
                        "buffered_templates": len(template_batch),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    flush_batches(connection, template_batch, source_batch, platform_batch, label_batch)

    crawl_name = args.crawl_name.strip() or detected_crawl_name or input_path.stem
    template_rows = build_template_overview(connection, crawl_name)
    concentration_rows = build_concentration_rows(template_rows, crawl_name)
    reuse_bucket_rows = build_reuse_bucket_rows(template_rows, crawl_name)

    source_stat_rows = fetch_rows(
        connection,
        """
        SELECT ? AS crawl, template_hash, source_domain, row_count, risky_row_count
        FROM template_source_stats
        ORDER BY row_count DESC, template_hash, source_domain
        """,
        (crawl_name,),
    )
    platform_stat_rows = fetch_rows(
        connection,
        """
        SELECT ? AS crawl, template_hash, target_platform, row_count, risky_row_count
        FROM template_platform_stats
        ORDER BY row_count DESC, template_hash, target_platform
        """,
        (crawl_name,),
    )
    label_stat_rows = fetch_rows(
        connection,
        """
        SELECT ? AS crawl, template_hash, label, row_count, risky_row_count, medium_row_count, high_row_count
        FROM template_label_stats
        ORDER BY row_count DESC, template_hash, label
        """,
        (crawl_name,),
    )

    write_csv(tables_dir / "template_overview.csv", template_rows, TEMPLATE_OVERVIEW_FIELDS)
    write_csv(
        tables_dir / "template_source_stats.csv",
        source_stat_rows,
        ["crawl", "template_hash", "source_domain", "row_count", "risky_row_count"],
    )
    write_csv(
        tables_dir / "template_platform_stats.csv",
        platform_stat_rows,
        ["crawl", "template_hash", "target_platform", "row_count", "risky_row_count"],
    )
    write_csv(
        tables_dir / "template_label_stats.csv",
        label_stat_rows,
        ["crawl", "template_hash", "label", "row_count", "risky_row_count", "medium_row_count", "high_row_count"],
    )
    write_csv(
        tables_dir / "concentration_summary.csv",
        concentration_rows,
        ["crawl", "scope", "template_count", "row_count", "row_share", "risky_row_count", "risky_row_share"],
    )
    write_csv(
        tables_dir / "reuse_buckets.csv",
        reuse_bucket_rows,
        ["crawl", "reuse_bucket", "template_count", "template_share", "row_count", "row_share", "risky_row_count", "risky_row_share"],
    )

    top_reused = template_rows[: max(args.review_limit, 1)]
    distributed_templates = [row for row in template_rows if int(row["unique_source_domains"]) >= 5][: max(args.review_limit, 1)]
    cross_platform_templates = [row for row in template_rows if int(row["unique_target_platforms"]) >= 3][: max(args.review_limit, 1)]
    write_csv(review_dir / "top_reused_templates.csv", _select_fields(top_reused, REVIEW_FIELDS), REVIEW_FIELDS)
    write_csv(review_dir / "distributed_templates.csv", _select_fields(distributed_templates, REVIEW_FIELDS), REVIEW_FIELDS)
    write_csv(review_dir / "cross_platform_templates.csv", _select_fields(cross_platform_templates, REVIEW_FIELDS), REVIEW_FIELDS)

    unique_templates = len(template_rows)
    singleton_templates = sum(1 for row in template_rows if int(row["row_count"]) == 1)
    reused_templates = unique_templates - singleton_templates
    rows_from_singletons = sum(int(row["row_count"]) for row in template_rows if int(row["row_count"]) == 1)
    rows_from_reused_templates = rows_with_prompt - rows_from_singletons
    distributed_row_count = sum(int(row["row_count"]) for row in template_rows if int(row["unique_source_domains"]) >= 5)
    cross_platform_row_count = sum(int(row["row_count"]) for row in template_rows if int(row["unique_target_platforms"]) >= 3)
    concentration_lookup = {row["scope"]: row for row in concentration_rows}
    elapsed_seconds = time.time() - started_at

    summary = {
        "generated_at_epoch": iso_now_epoch(),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "crawl": crawl_name,
        "rows_seen": rows_seen,
        "rows_with_prompt": rows_with_prompt,
        "rows_missing_prompt": rows_missing_prompt,
        "risky_rows": risky_rows,
        "risky_rows_with_prompt": risky_rows_with_prompt,
        "unique_templates": unique_templates,
        "singleton_templates": singleton_templates,
        "reused_templates": reused_templates,
        "rows_from_singletons": rows_from_singletons,
        "rows_from_reused_templates": rows_from_reused_templates,
        "distributed_templates_ge_5_domains": sum(1 for row in template_rows if int(row["unique_source_domains"]) >= 5),
        "distributed_rows_ge_5_domains": distributed_row_count,
        "cross_platform_templates_ge_3_platforms": sum(1 for row in template_rows if int(row["unique_target_platforms"]) >= 3),
        "cross_platform_rows_ge_3_platforms": cross_platform_row_count,
        "shares": {
            "rows_from_reused_templates_share": round((rows_from_reused_templates / rows_with_prompt), 6) if rows_with_prompt else 0.0,
            "rows_from_singletons_share": round((rows_from_singletons / rows_with_prompt), 6) if rows_with_prompt else 0.0,
            "distributed_rows_ge_5_domains_share": round((distributed_row_count / rows_with_prompt), 6) if rows_with_prompt else 0.0,
            "cross_platform_rows_ge_3_platforms_share": round((cross_platform_row_count / rows_with_prompt), 6) if rows_with_prompt else 0.0,
            "avg_rows_per_template": round((rows_with_prompt / unique_templates), 6) if unique_templates else 0.0,
            "top_10_row_share": float(concentration_lookup.get("top_10", {}).get("row_share", 0.0)),
            "top_100_row_share": float(concentration_lookup.get("top_100", {}).get("row_share", 0.0)),
            "top_1000_row_share": float(concentration_lookup.get("top_1000", {}).get("row_share", 0.0)),
        },
        "top_templates_by_rows": [
            {
                "template_hash": row["template_hash"],
                "row_count": row["row_count"],
                "unique_source_domains": row["unique_source_domains"],
                "unique_target_platforms": row["unique_target_platforms"],
                "top_label": row["top_label"],
                "sample_prompt": row["sample_prompt"],
            }
            for row in top_reused[:25]
        ],
        "files": {
            "template_overview_csv": str(tables_dir / "template_overview.csv"),
            "template_source_stats_csv": str(tables_dir / "template_source_stats.csv"),
            "template_platform_stats_csv": str(tables_dir / "template_platform_stats.csv"),
            "template_label_stats_csv": str(tables_dir / "template_label_stats.csv"),
            "concentration_summary_csv": str(tables_dir / "concentration_summary.csv"),
            "reuse_buckets_csv": str(tables_dir / "reuse_buckets.csv"),
            "top_reused_templates_csv": str(review_dir / "top_reused_templates.csv"),
            "distributed_templates_csv": str(review_dir / "distributed_templates.csv"),
            "cross_platform_templates_csv": str(review_dir / "cross_platform_templates.csv"),
            "work_db": str(db_path),
        },
        "runtime": {
            "elapsed_seconds": round(elapsed_seconds, 3),
            "rows_per_second": round((rows_seen / elapsed_seconds), 3) if elapsed_seconds else 0.0,
            "templates_per_second": round((unique_templates / elapsed_seconds), 3) if elapsed_seconds else 0.0,
        },
    }
    manifest = {
        "script": "analyze_template_reuse.py",
        "version": 1,
        "crawl": crawl_name,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "tables": sorted(str(path) for path in tables_dir.iterdir()),
        "review": sorted(str(path) for path in review_dir.iterdir()),
        "work": sorted(str(path) for path in work_dir.iterdir()),
        "summary_json": str(output_dir / "summary.json"),
    }

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"crawl": crawl_name, "rows_with_prompt": rows_with_prompt, "unique_templates": unique_templates, "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
