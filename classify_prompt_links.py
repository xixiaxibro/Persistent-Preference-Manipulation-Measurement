"""
Stage 2: Enrich + classify platform-filtered prompt-link rows in one pass.

Reads the JSONL output of ``filter_by_platform.py`` (~2 M rows, ~1.9 GB),
and for every row:

  1. Re-matches the platform (to obtain the signature object).
  2. Extracts prompt parameters with full per-platform normalization.
  3. Detects session entry and structural noise.
  4. Extracts IoC metadata across *all* prompt parameter values.
  5. Classifies ``primary_prompt_text`` against keyword rule sets and
     regex patterns (PERSIST, AUTHORITY, RECOMMEND, CITE, SUMMARIZE).
  6. Assigns severity (high / medium / low).

Writes one enriched + classified JSONL row per input row.
Prints platform distribution, severity distribution, label co-occurrence,
session entry breakdown, structural noise breakdown, and IoC summary.

All original record fields are preserved — this script only *adds* keys.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from env_config import load_project_env
from platform_signatures import (
    AUTHORITY_KEYWORDS,
    AUTHORITY_REGEX,
    CITE_KEYWORDS,
    CITE_REGEX,
    PERSIST_KEYWORDS,
    PERSIST_REGEX,
    RECOMMEND_KEYWORDS,
    RECOMMEND_REGEX,
    SUMMARIZE_KEYWORDS,
    SUMMARIZE_REGEX,
    extract_ioc_metadata,
    extract_prompt_parameters,
    flatten_prompt_parameters,
    keyword_hits,
    match_platform_with_exclusion,
    parse_domain,
    session_entry_reason,
    structural_noise_reason,
)

load_project_env()

# ---------------------------------------------------------------------------
# Classification constants
# ---------------------------------------------------------------------------

LABEL_ORDER: tuple[str, ...] = (
    "PERSIST",
    "AUTHORITY",
    "RECOMMEND",
    "CITE",
    "SUMMARIZE",
)

SUSPICIOUS_LABELS: frozenset[str] = frozenset(
    {"PERSIST", "AUTHORITY", "RECOMMEND", "CITE"}
)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

_IO_BUFFER_SIZE = 8 * 1024 * 1024


def _open_read(path: Path) -> BinaryIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb", buffering=_IO_BUFFER_SIZE)


def _open_write(path: Path) -> BinaryIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wb", compresslevel=3)
    return open(path, "wb", buffering=_IO_BUFFER_SIZE)


def _iter_lines(path: Path) -> Iterator[bytes]:
    with _open_read(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield stripped


def _format_size(n: int | float) -> str:
    if n < 1024:
        return f"{n:.0f}B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f}KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f}MB"
    return f"{n / 1024 ** 3:.2f}GB"


def _format_rate(per_sec: float) -> str:
    if per_sec < 1_000:
        return f"{per_sec:,.0f} rows/s"
    if per_sec < 1_000_000:
        return f"{per_sec / 1_000:,.1f}K rows/s"
    return f"{per_sec / 1_000_000:,.2f}M rows/s"


def _print_progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_prompt(text: str) -> tuple[list[str], str, dict[str, list[str]], list[str]]:
    """
    Classify a prompt text against keyword rule sets and regex patterns.

    Returns:
        labels:             ordered list of matched label names
        severity:           "high" | "medium" | "low"
        label_keyword_hits: {label: [matched keywords]} for labels with hits
        matched_keywords:   deduplicated sorted list of all matched keywords
    """
    hits_by_label: dict[str, list[str]] = {
        "PERSIST":    keyword_hits(text, PERSIST_KEYWORDS),
        "AUTHORITY":  keyword_hits(text, AUTHORITY_KEYWORDS),
        "RECOMMEND":  keyword_hits(text, RECOMMEND_KEYWORDS),
        "CITE":       keyword_hits(text, CITE_KEYWORDS),
        "SUMMARIZE":  keyword_hits(text, SUMMARIZE_KEYWORDS),
    }

    # Supplement keyword hits with regex matches.
    regex_by_label: dict[str, re.Pattern[str]] = {
        "PERSIST": PERSIST_REGEX,
        "AUTHORITY": AUTHORITY_REGEX,
        "RECOMMEND": RECOMMEND_REGEX,
        "CITE": CITE_REGEX,
        "SUMMARIZE": SUMMARIZE_REGEX,
    }
    for label, pattern in regex_by_label.items():
        if not hits_by_label[label]:
            match = pattern.search(text)
            if match:
                hits_by_label[label].append(match.group(0).lower())

    labels = [label for label in LABEL_ORDER if hits_by_label[label]]

    # Severity:
    #   high   = PERSIST + at least one other suspicious label
    #   medium = any single suspicious label (RECOMMEND, AUTHORITY, CITE)
    #   low    = SUMMARIZE-only or no labels
    if "PERSIST" in labels and any(
        label in labels for label in ("AUTHORITY", "RECOMMEND", "CITE")
    ):
        severity = "high"
    elif any(label in labels for label in SUSPICIOUS_LABELS):
        severity = "medium"
    else:
        severity = "low"

    # Deduplicated keyword list across all labels.
    all_keywords: list[str] = []
    for label in LABEL_ORDER:
        all_keywords.extend(hits_by_label[label])

    return labels, severity, hits_by_label, sorted(set(all_keywords))


# ---------------------------------------------------------------------------
# Tier 2 integration (optional)
# ---------------------------------------------------------------------------

# Minimum text length to invoke Tier 2 (shorter texts are unlikely to
# carry intent that Tier 1 keywords missed).
_TIER2_MIN_TEXT_LENGTH = 10

_tier2_classifier = None  # Lazily loaded.


def _load_tier2(model_dir: str) -> Any:
    """Lazy-load the Tier 2 classifier on first use."""
    global _tier2_classifier
    if _tier2_classifier is not None:
        return _tier2_classifier

    try:
        from prompt_classification.tier2_inference import Tier2Classifier
    except ImportError:
        _print_progress(
            "WARNING: Tier 2 dependencies not available. "
            "Install prompt_classification/requirements-tier2.txt."
        )
        return None

    _tier2_classifier = Tier2Classifier(model_dir)
    return _tier2_classifier


def _should_invoke_tier2(enriched: dict[str, Any]) -> bool:
    """
    Decide whether to invoke Tier 2 on an enriched row.

    Tier 2 is invoked when:
    1. Tier 1 assigns no labels AND text is non-empty and long enough, OR
    2. Tier 1 assigns only SUMMARIZE (check for higher-severity intent).
    """
    labels = enriched.get("prompt_labels", [])
    text = enriched.get("primary_prompt_text", "")
    if not isinstance(text, str) or len(text.strip()) < _TIER2_MIN_TEXT_LENGTH:
        return False

    if not labels:
        return True
    if labels == ["SUMMARIZE"]:
        return True
    return False


def apply_tier2(
    enriched: dict[str, Any],
    tier2: Any,
) -> dict[str, Any]:
    """
    Apply Tier 2 classification to an enriched row and merge results.

    Tier 2 labels are *additive* — they supplement Tier 1 labels.
    Severity is recomputed from the merged label set.
    """
    text = enriched.get("primary_prompt_text", "")
    if not isinstance(text, str) or not text.strip():
        return enriched

    tier2_labels, tier2_probs = tier2.classify_single(text)

    # Merge labels (Tier 2 supplements Tier 1).
    existing_labels = set(enriched.get("prompt_labels", []))
    merged_labels = sorted(existing_labels | set(tier2_labels), key=lambda l: LABEL_ORDER.index(l))

    # Recompute severity from merged labels.
    if "PERSIST" in merged_labels and any(
        label in merged_labels for label in ("AUTHORITY", "RECOMMEND", "CITE")
    ):
        severity = "high"
    elif any(label in merged_labels for label in SUSPICIOUS_LABELS):
        severity = "medium"
    else:
        severity = "low"

    enriched["prompt_labels"] = merged_labels
    enriched["classification"] = ";".join(merged_labels)
    enriched["severity"] = severity
    enriched["is_suspicious"] = any(label in SUSPICIOUS_LABELS for label in merged_labels)
    enriched["tier2_labels"] = tier2_labels
    enriched["tier2_probabilities"] = tier2_probs
    enriched["classification_tier"] = "tier2"

    return enriched


# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------

def process_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """
    Enrich and classify a single row.

    Returns the fully enriched row, or ``None`` if the URL no longer matches
    any platform (defensive guard against data corruption).
    """
    target_url: str = row.get("target_url", "")
    source_url: str = row.get("source_url", "")
    if not isinstance(target_url, str):
        target_url = ""
    if not isinstance(source_url, str):
        source_url = ""

    # --- 1. Platform match ---
    signature, excluded = match_platform_with_exclusion(target_url)
    if excluded or signature is None:
        return None

    # --- 2. Prompt parameter extraction (full per-platform normalization) ---
    prompt_parameters = extract_prompt_parameters(target_url, signature)
    decoded_candidates = flatten_prompt_parameters(prompt_parameters)
    primary_prompt = decoded_candidates[0] if decoded_candidates else ""

    # --- 3. Session entry detection ---
    entry_reason = session_entry_reason(
        target_url,
        signature=signature,
        prompt_parameters=prompt_parameters,
    )

    # --- 4. Structural noise ---
    noise_reason = structural_noise_reason(target_url)

    # --- 5. IoC metadata (scans ALL parameter values) ---
    ioc = extract_ioc_metadata(prompt_parameters)

    # --- 6. Keyword classification on primary_prompt_text ---
    prompt_labels, severity, label_keyword_hits, matched_keywords = classify_prompt(primary_prompt)

    # --- Assemble output (original keys preserved, new keys appended) ---
    enriched = dict(row)

    # Enrichment fields.
    enriched["source_domain"] = parse_domain(source_url)
    enriched["target_domain"] = parse_domain(target_url)
    enriched["target_platform"] = signature.name
    enriched["prompt_parameters"] = prompt_parameters
    enriched["decoded_prompt_candidates"] = decoded_candidates
    enriched["primary_prompt_text"] = primary_prompt
    enriched["has_prompt_parameters"] = bool(prompt_parameters)
    enriched["is_session_entry"] = entry_reason is not None
    enriched["session_entry_reason"] = entry_reason or ""
    enriched["structural_noise_reason"] = noise_reason or ""

    # IoC fields.
    enriched.update(ioc)

    # Classification fields.
    enriched["prompt_labels"] = prompt_labels
    enriched["classification"] = ";".join(prompt_labels)
    enriched["severity"] = severity
    enriched["label_keyword_hits"] = label_keyword_hits
    enriched["matched_rules"] = [label.lower() for label in prompt_labels]
    enriched["matched_keywords"] = matched_keywords
    enriched["is_suspicious"] = any(label in SUSPICIOUS_LABELS for label in prompt_labels)
    enriched["classification_tier"] = "tier1"

    return enriched


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    include_benign: bool = False,
    progress_interval: float = 10.0,
    tier2_model_dir: str | None = None,
) -> dict[str, Any]:
    """
    Single-pass enrich + classify.

    If *include_benign* is False (default), rows with severity=="low" and
    is_suspicious==False are excluded from the output (same semantics as the
    original filter_poisoning_candidates.py).

    If *tier2_model_dir* is provided, Tier 2 classification is applied as a
    second pass on rows where Tier 1 assigns no labels (or only SUMMARIZE)
    and the prompt text is long enough.
    """
    # Load Tier 2 classifier if requested.
    tier2 = None
    if tier2_model_dir:
        tier2 = _load_tier2(tier2_model_dir)
        if tier2 is not None:
            _print_progress(f"Tier 2 model loaded from: {tier2_model_dir}")
        else:
            _print_progress("WARNING: Tier 2 model could not be loaded; running Tier 1 only.")

    rows_seen = 0
    rows_written = 0
    rows_dropped_unmatched = 0
    rows_dropped_benign = 0
    rows_errored = 0
    rows_tier2 = 0
    bytes_read = 0

    platform_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    session_entry_counts: Counter[str] = Counter()
    noise_counts: Counter[str] = Counter()
    ioc_rows = 0

    start_time = time.monotonic()
    last_report = start_time

    with _open_write(output_path) as fout:
        for raw_line in _iter_lines(input_path):
            rows_seen += 1
            bytes_read += len(raw_line) + 1

            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, ValueError):
                rows_errored += 1
                continue

            enriched = process_row(row)
            if enriched is None:
                rows_dropped_unmatched += 1
                continue

            # Tier 2 optional second pass.
            if tier2 is not None and _should_invoke_tier2(enriched):
                enriched = apply_tier2(enriched, tier2)
                rows_tier2 += 1

            # Track all rows for distribution stats (before benign filter).
            platform = enriched["target_platform"]
            severity = enriched["severity"]
            platform_counts[platform] += 1
            severity_counts[severity] += 1

            for label in enriched["prompt_labels"]:
                label_counts[label] += 1

            reason = enriched["session_entry_reason"]
            if reason:
                session_entry_counts[reason] += 1

            noise = enriched["structural_noise_reason"]
            if noise:
                noise_counts[noise] += 1

            if enriched.get("has_ioc_keywords"):
                ioc_rows += 1

            # Apply benign filter.
            if not include_benign and not enriched["is_suspicious"]:
                rows_dropped_benign += 1
                continue

            rows_written += 1
            fout.write(
                json.dumps(enriched, ensure_ascii=False).encode("utf-8") + b"\n"
            )

            # Periodic progress.
            if rows_seen % 200_000 == 0:
                now = time.monotonic()
                if now - last_report >= progress_interval:
                    elapsed = now - start_time
                    rate = rows_seen / elapsed if elapsed > 0 else 0
                    _print_progress(
                        f"[{elapsed:>5.0f}s] rows: {rows_seen:>10,}  "
                        f"written: {rows_written:>10,}  "
                        f"benign: {rows_dropped_benign:>10,}  "
                        f"ioc: {ioc_rows:>6,}  "
                        f"rate: {_format_rate(rate)}"
                    )
                    last_report = now

    elapsed = time.monotonic() - start_time
    rate = rows_seen / elapsed if elapsed > 0 else 0

    return {
        "input": str(input_path),
        "output": str(output_path),
        "include_benign": include_benign,
        "tier2_model_dir": tier2_model_dir or "",
        "rows_seen": rows_seen,
        "rows_written": rows_written,
        "rows_dropped_unmatched": rows_dropped_unmatched,
        "rows_dropped_benign": rows_dropped_benign,
        "rows_errored": rows_errored,
        "rows_tier2": rows_tier2,
        "rows_suspicious": sum(1 for s in severity_counts if s != "low") and rows_written or sum(
            c for s, c in severity_counts.items() if s != "low"
        ),
        "rows_with_ioc_keywords": ioc_rows,
        "bytes_read": bytes_read,
        "elapsed_seconds": round(elapsed, 2),
        "avg_rate": _format_rate(rate),
        "platform_distribution": dict(platform_counts.most_common()),
        "severity_distribution": dict(severity_counts.most_common()),
        "label_distribution": dict(
            sorted(label_counts.items(), key=lambda kv: -kv[1])
        ),
        "session_entry_distribution": dict(
            sorted(session_entry_counts.items(), key=lambda kv: -kv[1])
        ),
        "structural_noise_distribution": dict(
            sorted(noise_counts.items(), key=lambda kv: -kv[1])
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich and classify platform-filtered prompt-link JSONL. "
            "Adds platform attribution, prompt analysis, session/noise flags, "
            "IoC metadata, keyword labels, and severity in a single pass."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL from filter_by_platform.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output classified JSONL path.",
    )
    parser.add_argument(
        "--include-benign",
        action="store_true",
        help=(
            "Include low-severity non-suspicious rows in the output. "
            "Default: only suspicious rows are written."
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="Seconds between progress reports (default: 10).",
    )
    parser.add_argument(
        "--tier2-model-dir",
        default=None,
        help=(
            "Path to Tier 2 model directory (from train_tier2_classifier.py). "
            "If provided, Tier 2 is used as a second pass on unlabeled/SUMMARIZE-only rows."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _print_progress(f"Input:          {input_path}")
    _print_progress(f"Output:         {output_path}")
    _print_progress(f"Include benign: {args.include_benign}")
    _print_progress(f"Tier 2 model:   {args.tier2_model_dir or '(disabled)'}")
    _print_progress("")

    summary = run_pipeline(
        input_path,
        output_path,
        include_benign=args.include_benign,
        progress_interval=args.progress_interval,
        tier2_model_dir=args.tier2_model_dir,
    )

    # ---- Final report to stderr ----

    _print_progress("")
    _print_progress(
        f"Done.  {summary['rows_seen']:,} rows read, "
        f"{summary['rows_written']:,} written, "
        f"{summary['rows_dropped_benign']:,} benign dropped, "
        f"{summary['rows_dropped_unmatched']:,} unmatched, "
        f"{summary['rows_errored']:,} errored  "
        f"({summary['elapsed_seconds']}s, {summary['avg_rate']})"
    )

    # Platform distribution.
    dist = summary["platform_distribution"]
    if dist:
        _print_progress("")
        _print_progress("Platform distribution (all rows before benign filter):")
        w = max(len(n) for n in dist)
        total = summary["rows_seen"] - summary["rows_dropped_unmatched"] - summary["rows_errored"]
        for name, count in dist.items():
            pct = count / total * 100 if total else 0
            _print_progress(f"  {name:<{w}}  {count:>10,}  ({pct:5.1f}%)")
        _print_progress(f"  {'TOTAL':<{w}}  {total:>10,}")

    # Severity distribution.
    sev = summary["severity_distribution"]
    if sev:
        _print_progress("")
        _print_progress("Severity distribution (all rows before benign filter):")
        w = max(len(s) for s in sev)
        for level, count in sev.items():
            _print_progress(f"  {level:<{w}}  {count:>10,}")

    # Label distribution.
    lab = summary["label_distribution"]
    if lab:
        _print_progress("")
        _print_progress("Label distribution (rows can have multiple labels):")
        w = max(len(l) for l in lab)
        for label, count in lab.items():
            _print_progress(f"  {label:<{w}}  {count:>10,}")

    # Session entry breakdown.
    se = summary["session_entry_distribution"]
    if se:
        _print_progress("")
        _print_progress("Session entry reasons:")
        w = max(len(r) for r in se)
        for reason, count in se.items():
            _print_progress(f"  {reason:<{w}}  {count:>10,}")

    # Structural noise breakdown.
    sn = summary["structural_noise_distribution"]
    if sn:
        _print_progress("")
        _print_progress("Structural noise reasons:")
        w = max(len(r) for r in sn)
        for reason, count in sn.items():
            _print_progress(f"  {reason:<{w}}  {count:>10,}")

    # IoC summary.
    _print_progress("")
    _print_progress(f"Rows with IoC keywords: {summary['rows_with_ioc_keywords']:,}")

    # Machine-readable JSON to stdout.
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())