#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = SCRIPT_DIR / ".vendor"
if VENDOR_DIR.is_dir():
    sys.path.insert(0, str(VENDOR_DIR))

try:
    from langid.langid import LanguageIdentifier, model
except ImportError as exc:
    raise SystemExit(
        "langid is required. Install it into ~/Unveiling_Persistent/.vendor first, "
        "for example: python3 -m pip install --target ~/Unveiling_Persistent/.vendor langid"
    ) from exc

from risk_analysis_common import iter_jsonl_rows, normalize_string, row_labels, row_severity, row_target_platform
from source_url_analysis_common import ensure_directory, iso_now_epoch, write_csv, write_json

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
WHITESPACE_RE = re.compile(r"\s+")
PROMPT_KEYS: tuple[str, ...] = ("primary_prompt_text", "prompt_text", "text")
SQLITE_BATCH_SIZE = 5000
PROGRESS_EVERY_ROWS = 500000
PROGRESS_EVERY_PROMPTS = 100000
MAX_REVIEW_PROMPT_CHARS = 500
UNKNOWN_LANGUAGE = "und"
SCRIPT_LANGUAGE_OVERRIDES = {
    "hangul": "ko",
    "japanese_kana": "ja",
    "hebrew": "he",
    "greek": "el",
    "thai": "th",
    "arabic": "ar",
    "devanagari": "hi",
}

PROMPT_INSERT_SQL = """
INSERT OR IGNORE INTO prompts (
    prompt_hash,
    normalized_prompt,
    prompt_length,
    alpha_chars
) VALUES (?, ?, ?, ?)
"""

PROMPT_UPDATE_COUNTS_SQL = """
UPDATE prompts
SET row_count = row_count + ?,
    risky_row_count = risky_row_count + ?,
    medium_row_count = medium_row_count + ?,
    high_row_count = high_row_count + ?
WHERE prompt_hash = ?
"""

LABEL_INSERT_SQL = """
INSERT OR IGNORE INTO prompt_label_stats (
    prompt_hash,
    label
) VALUES (?, ?)
"""

LABEL_UPDATE_COUNTS_SQL = """
UPDATE prompt_label_stats
SET row_count = row_count + ?,
    risky_row_count = risky_row_count + ?,
    medium_row_count = medium_row_count + ?,
    high_row_count = high_row_count + ?
WHERE prompt_hash = ? AND label = ?
"""

PLATFORM_INSERT_SQL = """
INSERT OR IGNORE INTO prompt_platform_stats (
    prompt_hash,
    target_platform
) VALUES (?, ?)
"""

PLATFORM_UPDATE_COUNTS_SQL = """
UPDATE prompt_platform_stats
SET row_count = row_count + ?,
    risky_row_count = risky_row_count + ?,
    medium_row_count = medium_row_count + ?,
    high_row_count = high_row_count + ?
WHERE prompt_hash = ? AND target_platform = ?
"""

LANG_UPDATE_SQL = """
UPDATE prompts
SET predicted_lang = ?,
    lang = ?,
    confidence = ?,
    status = ?,
    method = ?,
    dominant_script = ?
WHERE prompt_hash = ?
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect prompt-language distributions from Stage 02 classified prompt links."
    )
    parser.add_argument("--input", required=True, help="Input classified JSONL or JSONL.GZ")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--crawl-name", default="", help="Optional crawl name override")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.80,
        help="Minimum normalized langid confidence to accept a prediction directly.",
    )
    parser.add_argument(
        "--min-alpha-chars",
        type=int,
        default=4,
        help="Treat very short prompts below this alpha-character count as short when confidence is weak.",
    )
    parser.add_argument(
        "--review-limit",
        type=int,
        default=500,
        help="Maximum rows to emit in each review CSV.",
    )
    parser.add_argument(
        "--top-lang-limit",
        type=int,
        default=25,
        help="Maximum languages to keep in summary top-language lists.",
    )
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


def prompt_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def count_alpha_chars(text: str) -> int:
    return sum(1 for char in text if char.isalpha())


def is_risky_severity(severity: str) -> bool:
    return severity in {"medium", "high"}


def _script_for_char(char: str) -> str:
    codepoint = ord(char)
    if 0x3040 <= codepoint <= 0x30FF or 0x31F0 <= codepoint <= 0x31FF:
        return "japanese_kana"
    if (
        0xAC00 <= codepoint <= 0xD7AF
        or 0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
        or 0xA960 <= codepoint <= 0xA97F
        or 0xD7B0 <= codepoint <= 0xD7FF
    ):
        return "hangul"
    if 0x0590 <= codepoint <= 0x05FF:
        return "hebrew"
    if 0x0370 <= codepoint <= 0x03FF or 0x1F00 <= codepoint <= 0x1FFF:
        return "greek"
    if 0x0E00 <= codepoint <= 0x0E7F:
        return "thai"
    if (
        0x0600 <= codepoint <= 0x06FF
        or 0x0750 <= codepoint <= 0x077F
        or 0x08A0 <= codepoint <= 0x08FF
        or 0xFB50 <= codepoint <= 0xFDFF
        or 0xFE70 <= codepoint <= 0xFEFF
    ):
        return "arabic"
    if 0x0900 <= codepoint <= 0x097F:
        return "devanagari"
    if 0x3400 <= codepoint <= 0x4DBF or 0x4E00 <= codepoint <= 0x9FFF or 0xF900 <= codepoint <= 0xFAFF:
        return "han"
    name = unicodedata.name(char, "")
    if "LATIN" in name:
        return "latin"
    if "CYRILLIC" in name:
        return "cyrillic"
    return "other"


def dominant_script(text: str) -> tuple[str, float]:
    counter: collections.Counter[str] = collections.Counter()
    for char in text:
        if char.isalpha():
            counter[_script_for_char(char)] += 1
    if not counter:
        return "", 0.0
    total = sum(counter.values())
    script, count = counter.most_common(1)[0]
    return script, (count / total) if total else 0.0


def detect_language(
    text: str,
    *,
    alpha_chars: int,
    identifier: LanguageIdentifier,
    confidence_threshold: float,
    min_alpha_chars: int,
) -> dict[str, Any]:
    script_name, script_share = dominant_script(text)
    if alpha_chars < 2:
        return {
            "predicted_lang": "",
            "lang": UNKNOWN_LANGUAGE,
            "confidence": 0.0,
            "status": "unknown_short",
            "method": "rule_short",
            "dominant_script": script_name,
        }
    predicted_lang, confidence = identifier.classify(text)
    confidence = float(confidence)
    if confidence >= confidence_threshold:
        return {
            "predicted_lang": predicted_lang,
            "lang": predicted_lang,
            "confidence": confidence,
            "status": "accepted",
            "method": "langid",
            "dominant_script": script_name,
        }
    override_lang = SCRIPT_LANGUAGE_OVERRIDES.get(script_name, "")
    if override_lang and script_share >= 0.60:
        return {
            "predicted_lang": predicted_lang,
            "lang": override_lang,
            "confidence": confidence,
            "status": "script_override",
            "method": "script_override",
            "dominant_script": script_name,
        }
    if alpha_chars < min_alpha_chars:
        return {
            "predicted_lang": predicted_lang,
            "lang": UNKNOWN_LANGUAGE,
            "confidence": confidence,
            "status": "unknown_short",
            "method": "langid",
            "dominant_script": script_name,
        }
    return {
        "predicted_lang": predicted_lang,
        "lang": UNKNOWN_LANGUAGE,
        "confidence": confidence,
        "status": "unknown_low_conf",
        "method": "langid",
        "dominant_script": script_name,
    }

def _make_prompt_bucket(normalized_prompt: str, alpha_chars: int) -> dict[str, Any]:
    return {
        "normalized_prompt": normalized_prompt,
        "prompt_length": len(normalized_prompt),
        "alpha_chars": alpha_chars,
        "row_count": 0,
        "risky_row_count": 0,
        "medium_row_count": 0,
        "high_row_count": 0,
    }


def _update_stat_bucket(bucket: dict[str, int], severity: str) -> None:
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
        CREATE TABLE prompts (
            prompt_hash TEXT PRIMARY KEY,
            normalized_prompt TEXT NOT NULL,
            prompt_length INTEGER NOT NULL,
            alpha_chars INTEGER NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            risky_row_count INTEGER NOT NULL DEFAULT 0,
            medium_row_count INTEGER NOT NULL DEFAULT 0,
            high_row_count INTEGER NOT NULL DEFAULT 0,
            predicted_lang TEXT NOT NULL DEFAULT '',
            lang TEXT NOT NULL DEFAULT 'und',
            confidence REAL NOT NULL DEFAULT 0.0,
            status TEXT NOT NULL DEFAULT '',
            method TEXT NOT NULL DEFAULT '',
            dominant_script TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE prompt_label_stats (
            prompt_hash TEXT NOT NULL,
            label TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            risky_row_count INTEGER NOT NULL DEFAULT 0,
            medium_row_count INTEGER NOT NULL DEFAULT 0,
            high_row_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (prompt_hash, label)
        );
        CREATE TABLE prompt_platform_stats (
            prompt_hash TEXT NOT NULL,
            target_platform TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            risky_row_count INTEGER NOT NULL DEFAULT 0,
            medium_row_count INTEGER NOT NULL DEFAULT 0,
            high_row_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (prompt_hash, target_platform)
        );
        """
    )
    return connection


def flush_batches(
    connection: sqlite3.Connection,
    prompt_batch: dict[str, dict[str, Any]],
    label_batch: dict[tuple[str, str], dict[str, int]],
    platform_batch: dict[tuple[str, str], dict[str, int]],
) -> None:
    if not prompt_batch and not label_batch and not platform_batch:
        return
    with connection:
        if prompt_batch:
            connection.executemany(
                PROMPT_INSERT_SQL,
                [
                    (
                        prompt_hash_value,
                        payload["normalized_prompt"],
                        payload["prompt_length"],
                        payload["alpha_chars"],
                    )
                    for prompt_hash_value, payload in prompt_batch.items()
                ],
            )
            connection.executemany(
                PROMPT_UPDATE_COUNTS_SQL,
                [
                    (
                        payload["row_count"],
                        payload["risky_row_count"],
                        payload["medium_row_count"],
                        payload["high_row_count"],
                        prompt_hash_value,
                    )
                    for prompt_hash_value, payload in prompt_batch.items()
                ],
            )
        if label_batch:
            connection.executemany(
                LABEL_INSERT_SQL,
                [(prompt_hash_value, label) for (prompt_hash_value, label) in label_batch.keys()],
            )
            connection.executemany(
                LABEL_UPDATE_COUNTS_SQL,
                [
                    (
                        payload["row_count"],
                        payload["risky_row_count"],
                        payload["medium_row_count"],
                        payload["high_row_count"],
                        prompt_hash_value,
                        label,
                    )
                    for (prompt_hash_value, label), payload in label_batch.items()
                ],
            )
        if platform_batch:
            connection.executemany(
                PLATFORM_INSERT_SQL,
                [(prompt_hash_value, platform) for (prompt_hash_value, platform) in platform_batch.keys()],
            )
            connection.executemany(
                PLATFORM_UPDATE_COUNTS_SQL,
                [
                    (
                        payload["row_count"],
                        payload["risky_row_count"],
                        payload["medium_row_count"],
                        payload["high_row_count"],
                        prompt_hash_value,
                        platform,
                    )
                    for (prompt_hash_value, platform), payload in platform_batch.items()
                ],
            )
    prompt_batch.clear()
    label_batch.clear()
    platform_batch.clear()


def fetch_rows(connection: sqlite3.Connection, query: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query, parameters).fetchall()]


def build_language_overview(connection: sqlite3.Connection, crawl_name: str) -> list[dict[str, Any]]:
    rows = fetch_rows(
        connection,
        """
        SELECT
            lang,
            SUM(row_count) AS row_count,
            SUM(risky_row_count) AS risky_row_count,
            COUNT(*) AS unique_prompts,
            SUM(CASE WHEN risky_row_count > 0 THEN 1 ELSE 0 END) AS risky_unique_prompts,
            AVG(confidence) AS avg_confidence
        FROM prompts
        GROUP BY lang
        ORDER BY row_count DESC, unique_prompts DESC, lang
        """,
    )
    total_rows = sum(int(row["row_count"] or 0) for row in rows)
    total_risky_rows = sum(int(row["risky_row_count"] or 0) for row in rows)
    total_unique_prompts = sum(int(row["unique_prompts"] or 0) for row in rows)
    total_risky_unique_prompts = sum(int(row["risky_unique_prompts"] or 0) for row in rows)
    materialized: list[dict[str, Any]] = []
    for row in rows:
        row_count = int(row["row_count"] or 0)
        risky_row_count = int(row["risky_row_count"] or 0)
        unique_prompts = int(row["unique_prompts"] or 0)
        risky_unique_prompts = int(row["risky_unique_prompts"] or 0)
        materialized.append(
            {
                "crawl": crawl_name,
                "lang": row["lang"] or UNKNOWN_LANGUAGE,
                "row_count": row_count,
                "row_share": round((row_count / total_rows), 6) if total_rows else 0.0,
                "risky_row_count": risky_row_count,
                "risky_row_share": round((risky_row_count / total_risky_rows), 6) if total_risky_rows else 0.0,
                "unique_prompts": unique_prompts,
                "unique_prompt_share": round((unique_prompts / total_unique_prompts), 6) if total_unique_prompts else 0.0,
                "risky_unique_prompts": risky_unique_prompts,
                "risky_unique_prompt_share": round((risky_unique_prompts / total_risky_unique_prompts), 6)
                if total_risky_unique_prompts else 0.0,
                "avg_confidence": round(float(row["avg_confidence"] or 0.0), 6),
            }
        )
    return materialized


def build_language_by_severity(connection: sqlite3.Connection, crawl_name: str) -> list[dict[str, Any]]:
    rows = fetch_rows(
        connection,
        """
        SELECT
            lang,
            SUM(row_count - risky_row_count) AS low_rows,
            SUM(medium_row_count) AS medium_rows,
            SUM(high_row_count) AS high_rows,
            SUM(CASE WHEN row_count - risky_row_count > 0 THEN 1 ELSE 0 END) AS low_unique_prompts,
            SUM(CASE WHEN medium_row_count > 0 THEN 1 ELSE 0 END) AS medium_unique_prompts,
            SUM(CASE WHEN high_row_count > 0 THEN 1 ELSE 0 END) AS high_unique_prompts
        FROM prompts
        GROUP BY lang
        ORDER BY lang
        """,
    )
    severity_totals = {
        "low": sum(int(row["low_rows"] or 0) for row in rows),
        "medium": sum(int(row["medium_rows"] or 0) for row in rows),
        "high": sum(int(row["high_rows"] or 0) for row in rows),
    }
    severity_unique_totals = {
        "low": sum(int(row["low_unique_prompts"] or 0) for row in rows),
        "medium": sum(int(row["medium_unique_prompts"] or 0) for row in rows),
        "high": sum(int(row["high_unique_prompts"] or 0) for row in rows),
    }
    materialized: list[dict[str, Any]] = []
    for row in rows:
        per_lang = {
            "low": (int(row["low_rows"] or 0), int(row["low_unique_prompts"] or 0)),
            "medium": (int(row["medium_rows"] or 0), int(row["medium_unique_prompts"] or 0)),
            "high": (int(row["high_rows"] or 0), int(row["high_unique_prompts"] or 0)),
        }
        for severity, (row_count, unique_prompts) in per_lang.items():
            if row_count == 0 and unique_prompts == 0:
                continue
            materialized.append(
                {
                    "crawl": crawl_name,
                    "severity": severity,
                    "lang": row["lang"] or UNKNOWN_LANGUAGE,
                    "row_count": row_count,
                    "row_share_within_severity": round((row_count / severity_totals[severity]), 6)
                    if severity_totals[severity] else 0.0,
                    "unique_prompts": unique_prompts,
                    "unique_prompt_share_within_severity": round((unique_prompts / severity_unique_totals[severity]), 6)
                    if severity_unique_totals[severity] else 0.0,
                }
            )
    materialized.sort(key=lambda row: (row["severity"], -int(row["row_count"]), row["lang"]))
    return materialized

def _materialize_dimension_rows(
    rows: list[dict[str, Any]],
    *,
    crawl_name: str,
    dimension_key: str,
) -> list[dict[str, Any]]:
    row_totals: dict[str, int] = collections.Counter()
    risky_row_totals: dict[str, int] = collections.Counter()
    unique_prompt_totals: dict[str, int] = collections.Counter()
    risky_unique_prompt_totals: dict[str, int] = collections.Counter()
    for row in rows:
        dimension_value = str(row[dimension_key])
        row_totals[dimension_value] += int(row["row_count"] or 0)
        risky_row_totals[dimension_value] += int(row["risky_row_count"] or 0)
        unique_prompt_totals[dimension_value] += int(row["unique_prompts"] or 0)
        risky_unique_prompt_totals[dimension_value] += int(row["risky_unique_prompts"] or 0)
    materialized: list[dict[str, Any]] = []
    for row in rows:
        dimension_value = str(row[dimension_key])
        row_count = int(row["row_count"] or 0)
        risky_row_count = int(row["risky_row_count"] or 0)
        unique_prompts = int(row["unique_prompts"] or 0)
        risky_unique_prompts = int(row["risky_unique_prompts"] or 0)
        materialized.append(
            {
                "crawl": crawl_name,
                dimension_key: dimension_value,
                "lang": row["lang"] or UNKNOWN_LANGUAGE,
                "row_count": row_count,
                f"row_share_within_{dimension_key}": round((row_count / row_totals[dimension_value]), 6)
                if row_totals[dimension_value] else 0.0,
                "risky_row_count": risky_row_count,
                f"risky_row_share_within_{dimension_key}": round((risky_row_count / risky_row_totals[dimension_value]), 6)
                if risky_row_totals[dimension_value] else 0.0,
                "unique_prompts": unique_prompts,
                f"unique_prompt_share_within_{dimension_key}": round((unique_prompts / unique_prompt_totals[dimension_value]), 6)
                if unique_prompt_totals[dimension_value] else 0.0,
                "risky_unique_prompts": risky_unique_prompts,
                f"risky_unique_prompt_share_within_{dimension_key}": round(
                    (risky_unique_prompts / risky_unique_prompt_totals[dimension_value]), 6
                ) if risky_unique_prompt_totals[dimension_value] else 0.0,
                "medium_row_count": int(row["medium_row_count"] or 0),
                "high_row_count": int(row["high_row_count"] or 0),
            }
        )
    materialized.sort(key=lambda row: (row[dimension_key], -int(row["row_count"]), row["lang"]))
    return materialized


def build_language_by_label(connection: sqlite3.Connection, crawl_name: str) -> list[dict[str, Any]]:
    rows = fetch_rows(
        connection,
        """
        SELECT
            label,
            p.lang AS lang,
            SUM(pls.row_count) AS row_count,
            SUM(pls.risky_row_count) AS risky_row_count,
            SUM(pls.medium_row_count) AS medium_row_count,
            SUM(pls.high_row_count) AS high_row_count,
            COUNT(*) AS unique_prompts,
            SUM(CASE WHEN pls.risky_row_count > 0 THEN 1 ELSE 0 END) AS risky_unique_prompts
        FROM prompt_label_stats AS pls
        JOIN prompts AS p ON p.prompt_hash = pls.prompt_hash
        GROUP BY label, p.lang
        ORDER BY label, row_count DESC, unique_prompts DESC, p.lang
        """,
    )
    return _materialize_dimension_rows(rows, crawl_name=crawl_name, dimension_key="label")


def build_language_by_platform(connection: sqlite3.Connection, crawl_name: str) -> list[dict[str, Any]]:
    rows = fetch_rows(
        connection,
        """
        SELECT
            target_platform,
            p.lang AS lang,
            SUM(pps.row_count) AS row_count,
            SUM(pps.risky_row_count) AS risky_row_count,
            SUM(pps.medium_row_count) AS medium_row_count,
            SUM(pps.high_row_count) AS high_row_count,
            COUNT(*) AS unique_prompts,
            SUM(CASE WHEN pps.risky_row_count > 0 THEN 1 ELSE 0 END) AS risky_unique_prompts
        FROM prompt_platform_stats AS pps
        JOIN prompts AS p ON p.prompt_hash = pps.prompt_hash
        GROUP BY target_platform, p.lang
        ORDER BY target_platform, row_count DESC, unique_prompts DESC, p.lang
        """,
    )
    return _materialize_dimension_rows(rows, crawl_name=crawl_name, dimension_key="target_platform")


def build_status_summary(connection: sqlite3.Connection) -> dict[str, dict[str, int]]:
    rows = fetch_rows(
        connection,
        """
        SELECT
            status,
            COUNT(*) AS unique_prompts,
            SUM(row_count) AS row_count,
            SUM(risky_row_count) AS risky_row_count
        FROM prompts
        GROUP BY status
        ORDER BY row_count DESC, status
        """,
    )
    return {
        str(row["status"] or ""): {
            "unique_prompts": int(row["unique_prompts"] or 0),
            "row_count": int(row["row_count"] or 0),
            "risky_row_count": int(row["risky_row_count"] or 0),
        }
        for row in rows
    }


def build_method_summary(connection: sqlite3.Connection) -> dict[str, dict[str, int]]:
    rows = fetch_rows(
        connection,
        """
        SELECT
            method,
            COUNT(*) AS unique_prompts,
            SUM(row_count) AS row_count,
            SUM(risky_row_count) AS risky_row_count
        FROM prompts
        GROUP BY method
        ORDER BY row_count DESC, method
        """,
    )
    return {
        str(row["method"] or ""): {
            "unique_prompts": int(row["unique_prompts"] or 0),
            "row_count": int(row["row_count"] or 0),
            "risky_row_count": int(row["risky_row_count"] or 0),
        }
        for row in rows
    }


def _review_row(row: dict[str, Any], crawl_name: str) -> dict[str, Any]:
    prompt_text = str(row["normalized_prompt"] or "")
    sample_prompt = prompt_text[:MAX_REVIEW_PROMPT_CHARS]
    if len(prompt_text) > MAX_REVIEW_PROMPT_CHARS:
        sample_prompt += "..."
    return {
        "crawl": crawl_name,
        "prompt_hash": row["prompt_hash"],
        "lang": row["lang"] or UNKNOWN_LANGUAGE,
        "predicted_lang": row["predicted_lang"] or "",
        "confidence": round(float(row["confidence"] or 0.0), 6),
        "status": row["status"] or "",
        "method": row["method"] or "",
        "dominant_script": row["dominant_script"] or "",
        "row_count": int(row["row_count"] or 0),
        "risky_row_count": int(row["risky_row_count"] or 0),
        "medium_row_count": int(row["medium_row_count"] or 0),
        "high_row_count": int(row["high_row_count"] or 0),
        "sample_prompt": sample_prompt,
    }


def build_review_rows(
    connection: sqlite3.Connection,
    crawl_name: str,
    review_limit: int,
    overview_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    uncertain_rows = [
        _review_row(row, crawl_name)
        for row in fetch_rows(
            connection,
            """
            SELECT *
            FROM prompts
            WHERE lang = ? OR status <> 'accepted'
            ORDER BY risky_row_count DESC, row_count DESC, confidence ASC, prompt_hash
            LIMIT ?
            """,
            (UNKNOWN_LANGUAGE, max(review_limit, 1)),
        )
    ]
    non_english_rows = [
        _review_row(row, crawl_name)
        for row in fetch_rows(
            connection,
            """
            SELECT *
            FROM prompts
            WHERE lang NOT IN ('en', ?)
            ORDER BY risky_row_count DESC, row_count DESC, confidence DESC, prompt_hash
            LIMIT ?
            """,
            (UNKNOWN_LANGUAGE, max(review_limit, 1)),
        )
    ]
    top_languages = [row["lang"] for row in overview_rows if row["lang"] not in {"en", UNKNOWN_LANGUAGE}][:10]
    example_rows: list[dict[str, Any]] = []
    for lang in top_languages:
        lang_rank = 0
        for row in fetch_rows(
            connection,
            """
            SELECT *
            FROM prompts
            WHERE lang = ?
            ORDER BY risky_row_count DESC, row_count DESC, confidence DESC, prompt_hash
            LIMIT 5
            """,
            (lang,),
        ):
            lang_rank += 1
            reviewed = _review_row(row, crawl_name)
            reviewed["example_rank_within_lang"] = lang_rank
            example_rows.append(reviewed)
    return uncertain_rows, non_english_rows, example_rows


def top_languages_by_unique_prompts(overview_rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected = sorted(
        overview_rows,
        key=lambda row: (-int(row["unique_prompts"]), -int(row["row_count"]), row["lang"]),
    )[: max(limit, 1)]
    return [
        {
            "lang": row["lang"],
            "unique_prompts": row["unique_prompts"],
            "unique_prompt_share": row["unique_prompt_share"],
            "row_count": row["row_count"],
            "row_share": row["row_share"],
        }
        for row in selected
    ]

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
    db_path = work_dir / "prompt_language.sqlite3"
    connection = setup_database(db_path)

    rows_seen = 0
    rows_with_prompt = 0
    rows_missing_prompt = 0
    risky_rows = 0
    risky_rows_with_prompt = 0
    detected_crawl_name = ""
    prompt_batch: dict[str, dict[str, Any]] = {}
    label_batch: dict[tuple[str, str], dict[str, int]] = {}
    platform_batch: dict[tuple[str, str], dict[str, int]] = {}

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
        hash_value = prompt_hash(normalized_prompt)
        alpha_chars = count_alpha_chars(normalized_prompt)
        prompt_stats = prompt_batch.get(hash_value)
        if prompt_stats is None:
            prompt_stats = _make_prompt_bucket(normalized_prompt, alpha_chars)
            prompt_batch[hash_value] = prompt_stats
        _update_stat_bucket(prompt_stats, severity)
        labels = row_labels(row) or ["(none)"]
        for label in labels:
            key = (hash_value, label)
            label_stats = label_batch.get(key)
            if label_stats is None:
                label_stats = {"row_count": 0, "risky_row_count": 0, "medium_row_count": 0, "high_row_count": 0}
                label_batch[key] = label_stats
            _update_stat_bucket(label_stats, severity)
        platform = row_target_platform(row)
        platform_key = (hash_value, platform)
        platform_stats = platform_batch.get(platform_key)
        if platform_stats is None:
            platform_stats = {"row_count": 0, "risky_row_count": 0, "medium_row_count": 0, "high_row_count": 0}
            platform_batch[platform_key] = platform_stats
        _update_stat_bucket(platform_stats, severity)
        if rows_seen % SQLITE_BATCH_SIZE == 0:
            flush_batches(connection, prompt_batch, label_batch, platform_batch)
        if rows_seen % PROGRESS_EVERY_ROWS == 0:
            print(
                json.dumps(
                    {
                        "stage": "index",
                        "rows_seen": rows_seen,
                        "rows_with_prompt": rows_with_prompt,
                        "unique_prompts_buffered": len(prompt_batch),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    flush_batches(connection, prompt_batch, label_batch, platform_batch)
    with connection:
        connection.execute("CREATE INDEX IF NOT EXISTS idx_prompts_lang ON prompts(lang)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_prompt_labels_label ON prompt_label_stats(label)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_prompt_platforms_target_platform ON prompt_platform_stats(target_platform)")

    unique_prompt_count = int(connection.execute("SELECT COUNT(*) FROM prompts").fetchone()[0])
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    language_updates: list[tuple[str, str, float, str, str, str, str]] = []
    prompts_scored = 0
    for row in connection.execute(
        "SELECT prompt_hash, normalized_prompt, alpha_chars FROM prompts ORDER BY row_count DESC, prompt_hash"
    ):
        result = detect_language(
            str(row["normalized_prompt"]),
            alpha_chars=int(row["alpha_chars"] or 0),
            identifier=identifier,
            confidence_threshold=args.confidence_threshold,
            min_alpha_chars=args.min_alpha_chars,
        )
        language_updates.append(
            (
                str(result["predicted_lang"]),
                str(result["lang"]),
                float(result["confidence"]),
                str(result["status"]),
                str(result["method"]),
                str(result["dominant_script"]),
                str(row["prompt_hash"]),
            )
        )
        prompts_scored += 1
        if len(language_updates) >= SQLITE_BATCH_SIZE:
            with connection:
                connection.executemany(LANG_UPDATE_SQL, language_updates)
            language_updates.clear()
        if prompts_scored % PROGRESS_EVERY_PROMPTS == 0:
            print(
                json.dumps(
                    {
                        "stage": "detect",
                        "prompts_scored": prompts_scored,
                        "unique_prompts": unique_prompt_count,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if language_updates:
        with connection:
            connection.executemany(LANG_UPDATE_SQL, language_updates)

    crawl_name = args.crawl_name.strip() or detected_crawl_name or input_path.stem
    overview_rows = build_language_overview(connection, crawl_name)
    severity_rows = build_language_by_severity(connection, crawl_name)
    label_rows = build_language_by_label(connection, crawl_name)
    platform_rows = build_language_by_platform(connection, crawl_name)
    uncertain_rows, non_english_rows, example_rows = build_review_rows(
        connection,
        crawl_name,
        args.review_limit,
        overview_rows,
    )

    write_csv(
        tables_dir / "language_overview.csv",
        overview_rows,
        [
            "crawl", "lang", "row_count", "row_share", "risky_row_count", "risky_row_share",
            "unique_prompts", "unique_prompt_share", "risky_unique_prompts", "risky_unique_prompt_share",
            "avg_confidence",
        ],
    )
    write_csv(
        tables_dir / "language_by_severity.csv",
        severity_rows,
        [
            "crawl", "severity", "lang", "row_count", "row_share_within_severity",
            "unique_prompts", "unique_prompt_share_within_severity",
        ],
    )
    write_csv(
        tables_dir / "language_by_label.csv",
        label_rows,
        [
            "crawl", "label", "lang", "row_count", "row_share_within_label", "risky_row_count",
            "risky_row_share_within_label", "unique_prompts", "unique_prompt_share_within_label",
            "risky_unique_prompts", "risky_unique_prompt_share_within_label", "medium_row_count", "high_row_count",
        ],
    )
    write_csv(
        tables_dir / "language_by_platform.csv",
        platform_rows,
        [
            "crawl", "target_platform", "lang", "row_count", "row_share_within_target_platform",
            "risky_row_count", "risky_row_share_within_target_platform", "unique_prompts",
            "unique_prompt_share_within_target_platform", "risky_unique_prompts",
            "risky_unique_prompt_share_within_target_platform", "medium_row_count", "high_row_count",
        ],
    )
    review_fields = [
        "crawl", "prompt_hash", "lang", "predicted_lang", "confidence", "status", "method",
        "dominant_script", "row_count", "risky_row_count", "medium_row_count", "high_row_count", "sample_prompt",
    ]
    write_csv(review_dir / "uncertain_or_unknown_prompts.csv", uncertain_rows, review_fields)
    write_csv(review_dir / "top_non_english_prompts.csv", non_english_rows, review_fields)
    write_csv(
        review_dir / "top_language_examples.csv",
        example_rows,
        [
            "crawl", "lang", "example_rank_within_lang", "prompt_hash", "predicted_lang", "confidence",
            "status", "method", "dominant_script", "row_count", "risky_row_count", "medium_row_count",
            "high_row_count", "sample_prompt",
        ],
    )

    status_summary = build_status_summary(connection)
    method_summary = build_method_summary(connection)
    total_row_count = sum(int(row["row_count"]) for row in overview_rows)
    total_unique_prompts = sum(int(row["unique_prompts"]) for row in overview_rows)
    risky_unique_prompts = sum(int(row["risky_unique_prompts"]) for row in overview_rows)
    english_row_count = sum(int(row["row_count"]) for row in overview_rows if row["lang"] == "en")
    english_unique_prompt_count = sum(int(row["unique_prompts"]) for row in overview_rows if row["lang"] == "en")
    unknown_row_count = sum(int(row["row_count"]) for row in overview_rows if row["lang"] == UNKNOWN_LANGUAGE)
    unknown_unique_prompt_count = sum(int(row["unique_prompts"]) for row in overview_rows if row["lang"] == UNKNOWN_LANGUAGE)
    non_english_row_count = total_row_count - english_row_count - unknown_row_count
    non_english_unique_prompt_count = total_unique_prompts - english_unique_prompt_count - unknown_unique_prompt_count
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
        "unique_normalized_prompts": unique_prompt_count,
        "risky_unique_prompts": risky_unique_prompts,
        "configuration": {
            "confidence_threshold": args.confidence_threshold,
            "min_alpha_chars": args.min_alpha_chars,
            "review_limit": args.review_limit,
            "detector": "langid",
        },
        "shares": {
            "english_row_share": round((english_row_count / total_row_count), 6) if total_row_count else 0.0,
            "non_english_row_share": round((non_english_row_count / total_row_count), 6) if total_row_count else 0.0,
            "unknown_row_share": round((unknown_row_count / total_row_count), 6) if total_row_count else 0.0,
            "english_unique_prompt_share": round((english_unique_prompt_count / total_unique_prompts), 6)
            if total_unique_prompts else 0.0,
            "non_english_unique_prompt_share": round((non_english_unique_prompt_count / total_unique_prompts), 6)
            if total_unique_prompts else 0.0,
            "unknown_unique_prompt_share": round((unknown_unique_prompt_count / total_unique_prompts), 6)
            if total_unique_prompts else 0.0,
        },
        "language_status_counts": status_summary,
        "language_method_counts": method_summary,
        "top_languages_by_rows": [
            {
                "lang": row["lang"],
                "row_count": row["row_count"],
                "row_share": row["row_share"],
                "unique_prompts": row["unique_prompts"],
                "unique_prompt_share": row["unique_prompt_share"],
            }
            for row in overview_rows[: max(args.top_lang_limit, 1)]
        ],
        "top_languages_by_unique_prompts": top_languages_by_unique_prompts(overview_rows, args.top_lang_limit),
        "files": {
            "language_overview_csv": str(tables_dir / "language_overview.csv"),
            "language_by_severity_csv": str(tables_dir / "language_by_severity.csv"),
            "language_by_label_csv": str(tables_dir / "language_by_label.csv"),
            "language_by_platform_csv": str(tables_dir / "language_by_platform.csv"),
            "uncertain_or_unknown_prompts_csv": str(review_dir / "uncertain_or_unknown_prompts.csv"),
            "top_non_english_prompts_csv": str(review_dir / "top_non_english_prompts.csv"),
            "top_language_examples_csv": str(review_dir / "top_language_examples.csv"),
            "work_db": str(db_path),
        },
        "runtime": {
            "elapsed_seconds": round(elapsed_seconds, 3),
            "rows_per_second": round((rows_seen / elapsed_seconds), 3) if elapsed_seconds else 0.0,
            "unique_prompts_per_second": round((unique_prompt_count / elapsed_seconds), 3) if elapsed_seconds else 0.0,
        },
    }
    manifest = {
        "script": "analyze_prompt_language.py",
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
    print(
        json.dumps(
            {
                "crawl": crawl_name,
                "rows_seen": rows_seen,
                "rows_with_prompt": rows_with_prompt,
                "unique_prompts": unique_prompt_count,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
