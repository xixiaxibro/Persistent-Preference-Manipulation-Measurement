from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator, TextIO

from platform_signatures import extract_ioc_metadata, parse_domain


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def iter_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")
            yield payload


def normalize_string(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def row_severity(row: dict[str, Any]) -> str:
    severity = normalize_string(row.get("tier2_severity")) or normalize_string(row.get("severity"))
    return severity.lower()


def row_labels(row: dict[str, Any]) -> list[str]:
    raw_labels = row.get("tier2_labels")
    if isinstance(raw_labels, list):
        labels = [item.strip().upper() for item in raw_labels if isinstance(item, str) and item.strip()]
        if labels:
            return labels

    raw_labels = row.get("prompt_labels")
    if isinstance(raw_labels, list):
        labels = [item.strip().upper() for item in raw_labels if isinstance(item, str) and item.strip()]
        if labels:
            return labels

    classification = normalize_string(row.get("classification"))
    if not classification:
        return []
    return [item.strip().upper() for item in classification.split(";") if item.strip()]


def is_risky_row(row: dict[str, Any]) -> bool:
    return row_severity(row) in {"medium", "high"}


def row_source_domain(row: dict[str, Any]) -> str:
    source_domain = normalize_string(row.get("source_domain"))
    if source_domain:
        return source_domain.lower()
    source_url = normalize_string(row.get("source_url"))
    return parse_domain(source_url)


def row_target_domain(row: dict[str, Any]) -> str:
    target_domain = normalize_string(row.get("target_domain"))
    if target_domain:
        return target_domain.lower()
    target_url = normalize_string(row.get("target_url"))
    return parse_domain(target_url)


def row_target_platform(row: dict[str, Any]) -> str:
    return normalize_string(row.get("target_platform")) or "(unknown)"


def row_prompt_parameters(row: dict[str, Any]) -> dict[str, list[str]]:
    raw = row.get("prompt_parameters")
    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, list[str]] = {}
    for key, values in raw.items():
        key_text = normalize_string(key)
        if not key_text or not isinstance(values, list):
            continue
        normalized_values = [normalize_string(value) for value in values if normalize_string(value)]
        if normalized_values:
            normalized[key_text] = normalized_values
    return normalized


def row_ioc_metadata(row: dict[str, Any]) -> dict[str, object]:
    prompt_parameters = row_prompt_parameters(row)
    return extract_ioc_metadata(prompt_parameters)


def resolve_classified_input(run_root: Path) -> Path:
    classify_dir = run_root / "02_classify"
    exact_candidates = [
        classify_dir / "classified_prompt_links.jsonl",
        classify_dir / "classified_prompt_links.jsonl.gz",
    ]
    for candidate in exact_candidates:
        if candidate.is_file():
            return candidate

    candidates = sorted(
        path
        for pattern in ("classified_prompt_links*.jsonl", "classified_prompt_links*.jsonl.gz")
        for path in classify_dir.glob(pattern)
        if path.is_file()
    )

    if not candidates:
        raise FileNotFoundError(f"No classified JSONL found under: {classify_dir}")
    if len(candidates) > 1:
        formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise RuntimeError(
            f"Multiple classified JSONL candidates found under {classify_dir}:\n{formatted}\n"
            "Keep exactly one classified JSONL per crawl or pass a different run root."
        )
    return candidates[0]