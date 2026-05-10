from __future__ import annotations

import csv
import gzip
import io
import json
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib import request
from urllib.parse import urlparse

try:
    import tldextract
except ImportError:  # pragma: no cover - handled at runtime
    tldextract = None


TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
DEFAULT_TRANCO_CACHE = "tranco_top1m.csv"

TRANCO_BUCKET_ORDER: tuple[str, ...] = (
    "top_1k",
    "1k_10k",
    "10k_100k",
    "100k_1m",
    "unranked",
)


@dataclass(frozen=True)
class TrancoMatch:
    rank: int | None
    matched_domain: str
    bucket: str
    match_type: str


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")
            yield payload


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _load_csv_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _download_tranco_blob() -> bytes:
    response = request.urlopen(TRANCO_URL, timeout=120)
    try:
        return response.read()
    finally:
        response.close()


def _blob_to_csv_text(blob: bytes) -> str:
    try:
        archive = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        return blob.decode("utf-8", errors="replace")

    names = archive.namelist()
    csv_names = [name for name in names if name.endswith(".csv")]
    target_name = csv_names[0] if csv_names else names[0]
    return archive.read(target_name).decode("utf-8", errors="replace")


def _parse_tranco_csv_text(csv_text: str) -> dict[str, int]:
    ranking: dict[str, int] = {}
    for line in csv_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(",", 1)
        if len(parts) != 2:
            continue
        try:
            rank = int(parts[0].strip())
        except ValueError:
            continue
        domain = parts[1].strip().lower()
        if domain:
            ranking[domain] = rank
    return ranking


def load_tranco_ranking(
    *,
    tranco_csv: Path | None,
    tranco_cache: Path | None,
    mode: str,
) -> tuple[dict[str, int], str]:
    if tranco_csv is not None and tranco_csv.exists():
        return _parse_tranco_csv_text(_load_csv_text(tranco_csv)), str(tranco_csv)

    if tranco_cache is not None and tranco_cache.exists():
        return _parse_tranco_csv_text(_load_csv_text(tranco_cache)), str(tranco_cache)

    if mode != "download-if-missing":
        return {}, ""

    cache_path = tranco_cache or Path(DEFAULT_TRANCO_CACHE)
    blob = _download_tranco_blob()
    csv_text = _blob_to_csv_text(blob)
    cache_path.write_text(csv_text, encoding="utf-8")
    return _parse_tranco_csv_text(csv_text), str(cache_path)


def make_domain_extractor():
    if tldextract is None:
        return None
    return tldextract.TLDExtract(suffix_list_urls=None)


def extract_host(value: str) -> str:
    if not value:
        return ""
    candidate = value.strip()
    if not candidate:
        return ""

    if "://" not in candidate:
        candidate = f"https://{candidate}"

    try:
        parsed = urlparse(candidate)
    except ValueError:
        return ""
    return parsed.netloc.lower()


def extract_root_domain(value: str, extractor) -> str:
    host = extract_host(value)
    if not host:
        return ""
    if extractor is None:
        parts = [part for part in host.split(".") if part]
        return ".".join(parts[-2:]).lower() if len(parts) >= 2 else host
    extracted = extractor(host)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()
    return host


def tranco_bucket(rank: int | None) -> str:
    if rank is None:
        return "unranked"
    if rank <= 1_000:
        return "top_1k"
    if rank <= 10_000:
        return "1k_10k"
    if rank <= 100_000:
        return "10k_100k"
    if rank <= 1_000_000:
        return "100k_1m"
    return "unranked"


def lookup_tranco(domain: str, ranking: dict[str, int]) -> TrancoMatch:
    normalized = domain.strip().lower()
    if not normalized:
        return TrancoMatch(None, "", "unranked", "not_ranked")

    if normalized in ranking:
        rank = ranking[normalized]
        return TrancoMatch(rank, normalized, tranco_bucket(rank), "exact_root")

    parts = normalized.split(".")
    for index in range(1, len(parts) - 1):
        candidate = ".".join(parts[index:])
        if candidate in ranking:
            rank = ranking[candidate]
            return TrancoMatch(rank, candidate, tranco_bucket(rank), "parent_match")

    return TrancoMatch(None, "", "unranked", "not_ranked")


def counter_to_sorted_rows(
    counter: dict[str, int],
    *,
    total: int,
    key_name: str,
    value_name: str = "count",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            {
                key_name: key,
                value_name: count,
                "share": round((count / total), 6) if total else 0.0,
            }
        )
    return rows


def ordered_bucket_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {bucket: index for index, bucket in enumerate(TRANCO_BUCKET_ORDER)}
    return sorted(rows, key=lambda row: order.get(str(row.get("tranco_bucket", "")), 999))


def iso_now_epoch() -> int:
    return int(time.time())
