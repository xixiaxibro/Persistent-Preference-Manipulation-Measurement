"""
Stage 2: enrich + classify platform-filtered prompt-link rows with the active
semantic model.

Reads the JSONL output of ``filter_by_platform.py`` and for every row:

  1. Re-matches the platform (to obtain the signature object).
  2. Extracts prompt parameters with full per-platform normalization.
  3. Detects session entry and structural noise.
  4. Extracts IoC metadata across all prompt parameter values.
  5. Classifies ``primary_prompt_text`` with the active semantic model.
  6. Assigns severity (high / medium / low).

All original record fields are preserved. Keyword-derived debug fields from the
historical classifier are no longer emitted.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from .env_config import load_project_env
from .platform_signatures import (
    extract_ioc_metadata,
    extract_prompt_parameters,
    flatten_prompt_parameters,
    match_platform_with_exclusion,
    parse_domain,
    session_entry_reason,
    structural_noise_reason,
)

load_project_env()

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

DEFAULT_MODEL_DIR = Path("models") / "tier2_gt_hq10000_plus_recboost_20260407_v2"
DEFAULT_CLASSIFICATION_BATCH_SIZE = 8
DEFAULT_CLASSIFICATION_DEVICE = "cpu"
INFERENCE_INSTALL_HINT = (
    "python3 -m pip install -r requirements.txt -r requirements-inference.txt"
)

_IO_BUFFER_SIZE = 8 * 1024 * 1024
_PROGRESS_ROW_INTERVAL = 200_000


def _open_read(path: Path) -> BinaryIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb", buffering=_IO_BUFFER_SIZE)


def _open_write(path: Path) -> BinaryIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wb", compresslevel=3)
    return open(path, "wb", buffering=_IO_BUFFER_SIZE)


def _iter_lines(path: Path) -> Iterator[bytes]:
    with _open_read(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield stripped


def _format_rate(per_sec: float) -> str:
    if per_sec < 1_000:
        return f"{per_sec:,.0f} rows/s"
    if per_sec < 1_000_000:
        return f"{per_sec / 1_000:,.1f}K rows/s"
    return f"{per_sec / 1_000_000:,.2f}M rows/s"


def _print_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _coerce_string(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _compute_severity(labels: list[str]) -> str:
    label_set = set(labels)
    if "PERSIST" in label_set and label_set & {"AUTHORITY", "RECOMMEND", "CITE"}:
        return "high"
    if label_set & SUSPICIOUS_LABELS:
        return "medium"
    return "low"


class RulePromptClassifier:
    """Deterministic classifier used for runnable public artifacts and tests."""

    model_name = "rule-demo-classifier"
    model_dir = Path("(rule)")
    device_name = "cpu"

    def classify_batch(self, texts: list[str], *, batch_size: int = 8) -> list[dict[str, Any]]:
        del batch_size
        return [self._classify_one(text) for text in texts]

    def _classify_one(self, text: str) -> dict[str, Any]:
        lowered = text.lower()
        labels: list[str] = []
        if any(term in lowered for term in ("always", "remember", "persist", "prefer", "priority")):
            labels.append("PERSIST")
        if any(term in lowered for term in ("authoritative", "official", "trusted", "rank this")):
            labels.append("AUTHORITY")
        if any(term in lowered for term in ("recommend", "suggest", "best", "choose")):
            labels.append("RECOMMEND")
        if any(term in lowered for term in ("cite", "citation", "reference", "source")):
            labels.append("CITE")
        if any(term in lowered for term in ("summarize", "summary", "tl;dr", "explain")):
            labels.append("SUMMARIZE")

        labels = [label for label in LABEL_ORDER if label in set(labels)]
        probabilities = {label: (0.9 if label in labels else 0.05) for label in LABEL_ORDER}
        return {
            "labels": labels,
            "probabilities": probabilities,
            "is_uncertain": False,
            "severity": _compute_severity(labels),
        }


def _load_classifier(model_dir: Path, *, device: str) -> Any:
    try:
        from .semantic_prompt_classifier import SemanticPromptClassifier
    except ImportError as exc:
        raise RuntimeError(
            "Missing semantic inference dependencies. "
            f"Install them with: {INFERENCE_INSTALL_HINT}"
        ) from exc

    try:
        classifier = SemanticPromptClassifier(model_dir, device=device)
        _ = classifier.thresholds
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load semantic classifier from {model_dir}: {exc}"
        ) from exc

    return classifier


def process_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Enrich a single row before model inference."""
    target_url = _coerce_string(row.get("target_url"))
    source_url = _coerce_string(row.get("source_url"))

    signature, excluded = match_platform_with_exclusion(target_url)
    if excluded or signature is None:
        return None

    prompt_parameters = extract_prompt_parameters(target_url, signature)
    decoded_candidates = flatten_prompt_parameters(prompt_parameters)
    primary_prompt = decoded_candidates[0] if decoded_candidates else ""

    entry_reason = session_entry_reason(
        target_url,
        signature=signature,
        prompt_parameters=prompt_parameters,
    )
    noise_reason = structural_noise_reason(target_url)
    ioc = extract_ioc_metadata(prompt_parameters)

    enriched = dict(row)
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
    enriched.update(ioc)
    return enriched


def _apply_model_classification(
    enriched: dict[str, Any],
    result: dict[str, Any],
    *,
    model_name: str,
) -> dict[str, Any]:
    raw_labels = result.get("labels", [])
    label_set = {label for label in raw_labels if label in LABEL_ORDER}
    prompt_labels = [label for label in LABEL_ORDER if label in label_set]
    probabilities = result.get("probabilities", {})

    enriched["prompt_labels"] = prompt_labels
    enriched["classification"] = ";".join(prompt_labels)
    enriched["severity"] = _compute_severity(prompt_labels)
    enriched["is_suspicious"] = any(
        label in SUSPICIOUS_LABELS for label in prompt_labels
    )
    enriched["classification_tier"] = "rule" if model_name == RulePromptClassifier.model_name else "model"
    enriched["classification_method"] = model_name
    enriched["classification_model_name"] = model_name
    enriched["classification_probabilities"] = {
        label: round(float(probabilities.get(label, 0.0)), 4)
        for label in LABEL_ORDER
    }
    enriched["classification_is_uncertain"] = bool(
        result.get("is_uncertain", False)
    )
    return enriched


def _flush_pending_rows(
    pending_rows: list[dict[str, Any]],
    *,
    classifier: Any,
    batch_size: int,
    include_benign: bool,
    fout: BinaryIO,
    counters: Counter[str],
    platform_counts: Counter[str],
    severity_counts: Counter[str],
    label_counts: Counter[str],
    session_entry_counts: Counter[str],
    noise_counts: Counter[str],
) -> None:
    if not pending_rows:
        return

    texts = [_coerce_string(row.get("primary_prompt_text")) for row in pending_rows]
    results = classifier.classify_batch(texts, batch_size=batch_size)
    if len(results) != len(pending_rows):
        raise RuntimeError("Classifier returned a mismatched batch size.")

    for enriched, result in zip(pending_rows, results):
        enriched = _apply_model_classification(
            enriched,
            result,
            model_name=classifier.model_name,
        )

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
            counters["rows_with_ioc_keywords"] += 1
        if enriched["classification_is_uncertain"]:
            counters["rows_uncertain"] += 1

        if not include_benign and not enriched["is_suspicious"]:
            counters["rows_dropped_benign"] += 1
            continue

        fout.write(json.dumps(enriched, ensure_ascii=False).encode("utf-8") + b"\n")
        counters["rows_written"] += 1

    pending_rows.clear()


def run_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    classifier: Any,
    batch_size: int,
    include_benign: bool = False,
    progress_interval: float = 10.0,
) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    platform_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    session_entry_counts: Counter[str] = Counter()
    noise_counts: Counter[str] = Counter()
    bytes_read = 0

    start_time = time.monotonic()
    last_report = start_time
    pending_rows: list[dict[str, Any]] = []

    with _open_write(output_path) as fout:
        for raw_line in _iter_lines(input_path):
            counters["rows_seen"] += 1
            bytes_read += len(raw_line) + 1

            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, ValueError):
                counters["rows_errored"] += 1
                continue

            enriched = process_row(row)
            if enriched is None:
                counters["rows_dropped_unmatched"] += 1
                continue

            pending_rows.append(enriched)
            if len(pending_rows) >= batch_size:
                _flush_pending_rows(
                    pending_rows,
                    classifier=classifier,
                    batch_size=batch_size,
                    include_benign=include_benign,
                    fout=fout,
                    counters=counters,
                    platform_counts=platform_counts,
                    severity_counts=severity_counts,
                    label_counts=label_counts,
                    session_entry_counts=session_entry_counts,
                    noise_counts=noise_counts,
                )

            if counters["rows_seen"] % _PROGRESS_ROW_INTERVAL == 0:
                now = time.monotonic()
                if now - last_report >= progress_interval:
                    elapsed = now - start_time
                    rate = counters["rows_seen"] / elapsed if elapsed > 0 else 0.0
                    _print_progress(
                        f"[{elapsed:>5.0f}s] rows: {counters['rows_seen']:>10,}  "
                        f"written: {counters['rows_written']:>10,}  "
                        f"benign: {counters['rows_dropped_benign']:>10,}  "
                        f"uncertain: {counters['rows_uncertain']:>8,}  "
                        f"ioc: {counters['rows_with_ioc_keywords']:>8,}  "
                        f"rate: {_format_rate(rate)}"
                    )
                    last_report = now

        _flush_pending_rows(
            pending_rows,
            classifier=classifier,
            batch_size=batch_size,
            include_benign=include_benign,
            fout=fout,
            counters=counters,
            platform_counts=platform_counts,
            severity_counts=severity_counts,
            label_counts=label_counts,
            session_entry_counts=session_entry_counts,
            noise_counts=noise_counts,
        )

    elapsed = time.monotonic() - start_time
    rate = counters["rows_seen"] / elapsed if elapsed > 0 else 0.0

    return {
        "input": str(input_path),
        "output": str(output_path),
        "include_benign": include_benign,
        "classification_method": classifier.model_name,
        "classification_model_dir": str(classifier.model_dir),
        "classification_model_name": classifier.model_name,
        "classification_device": classifier.device_name,
        "classification_batch_size": batch_size,
        "rows_seen": counters["rows_seen"],
        "rows_written": counters["rows_written"],
        "rows_dropped_unmatched": counters["rows_dropped_unmatched"],
        "rows_dropped_benign": counters["rows_dropped_benign"],
        "rows_errored": counters["rows_errored"],
        "rows_uncertain": counters["rows_uncertain"],
        "rows_suspicious": sum(
            count for severity, count in severity_counts.items() if severity != "low"
        ),
        "rows_with_ioc_keywords": counters["rows_with_ioc_keywords"],
        "bytes_read": bytes_read,
        "elapsed_seconds": round(elapsed, 2),
        "avg_rate": _format_rate(rate),
        "platform_distribution": dict(platform_counts.most_common()),
        "severity_distribution": dict(severity_counts.most_common()),
        "label_distribution": dict(sorted(label_counts.items(), key=lambda item: -item[1])),
        "session_entry_distribution": dict(
            sorted(session_entry_counts.items(), key=lambda item: -item[1])
        ),
        "structural_noise_distribution": dict(
            sorted(noise_counts.items(), key=lambda item: -item[1])
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich and classify platform-filtered prompt-link JSONL.",
    )
    parser.add_argument("--input", required=True, help="Input JSONL from filter_by_platform.py.")
    parser.add_argument("--output", required=True, help="Output classified JSONL path.")
    parser.add_argument(
        "--classifier",
        choices=("rule", "semantic"),
        default="rule",
        help="Classifier backend. The default rule backend is deterministic and needs no model weights.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help=(
            "Path to a semantic model directory. Required only with --classifier semantic. "
            f"Default: {DEFAULT_MODEL_DIR}"
        ),
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_CLASSIFICATION_DEVICE,
        help="Inference device: cpu, cuda, cuda:0, or auto (default: cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CLASSIFICATION_BATCH_SIZE,
        help=f"Model inference batch size (default: {DEFAULT_CLASSIFICATION_BATCH_SIZE}).",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    model_dir = Path(args.model_dir)

    if args.batch_size <= 0:
        print("Error: --batch-size must be positive.", file=sys.stderr)
        return 1
    if not input_path.exists():
        print(f"Error: input file does not exist: {input_path}", file=sys.stderr)
        return 1
    if args.classifier == "semantic" and not model_dir.exists():
        print(f"Error: model directory does not exist: {model_dir}", file=sys.stderr)
        print(
            f"Install runtime dependencies with: {INFERENCE_INSTALL_HINT}",
            file=sys.stderr,
        )
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        classifier = RulePromptClassifier() if args.classifier == "rule" else _load_classifier(model_dir, device=args.device)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_progress(f"Input:          {input_path}")
    _print_progress(f"Output:         {output_path}")
    _print_progress(f"Classifier:     {args.classifier}")
    if args.classifier == "semantic":
        _print_progress(f"Model dir:      {model_dir}")
    _print_progress(f"Model name:     {classifier.model_name}")
    _print_progress(f"Device:         {classifier.device_name}")
    _print_progress(f"Batch size:     {args.batch_size}")
    _print_progress(f"Include benign: {args.include_benign}")
    _print_progress("")

    summary = run_pipeline(
        input_path,
        output_path,
        classifier=classifier,
        batch_size=args.batch_size,
        include_benign=args.include_benign,
        progress_interval=args.progress_interval,
    )

    _print_progress("")
    _print_progress(
        f"Done.  {summary['rows_seen']:,} rows read, "
        f"{summary['rows_written']:,} written, "
        f"{summary['rows_dropped_benign']:,} benign dropped, "
        f"{summary['rows_dropped_unmatched']:,} unmatched, "
        f"{summary['rows_errored']:,} errored, "
        f"{summary['rows_uncertain']:,} uncertain  "
        f"({summary['elapsed_seconds']}s, {summary['avg_rate']})"
    )

    dist = summary["platform_distribution"]
    if dist:
        _print_progress("")
        _print_progress("Platform distribution (all rows before benign filter):")
        width = max(len(name) for name in dist)
        total = (
            summary["rows_seen"]
            - summary["rows_dropped_unmatched"]
            - summary["rows_errored"]
        )
        for name, count in dist.items():
            pct = count / total * 100 if total else 0.0
            _print_progress(f"  {name:<{width}}  {count:>10,}  ({pct:5.1f}%)")
        _print_progress(f"  {'TOTAL':<{width}}  {total:>10,}")

    sev = summary["severity_distribution"]
    if sev:
        _print_progress("")
        _print_progress("Severity distribution (all rows before benign filter):")
        width = max(len(level) for level in sev)
        for level, count in sev.items():
            _print_progress(f"  {level:<{width}}  {count:>10,}")

    lab = summary["label_distribution"]
    if lab:
        _print_progress("")
        _print_progress("Label distribution (rows can have multiple labels):")
        width = max(len(label) for label in lab)
        for label, count in lab.items():
            _print_progress(f"  {label:<{width}}  {count:>10,}")

    se = summary["session_entry_distribution"]
    if se:
        _print_progress("")
        _print_progress("Session entry reasons:")
        width = max(len(reason) for reason in se)
        for reason, count in se.items():
            _print_progress(f"  {reason:<{width}}  {count:>10,}")

    sn = summary["structural_noise_distribution"]
    if sn:
        _print_progress("")
        _print_progress("Structural noise reasons:")
        width = max(len(reason) for reason in sn)
        for reason, count in sn.items():
            _print_progress(f"  {reason:<{width}}  {count:>10,}")

    _print_progress("")
    _print_progress(f"Rows with IoC keywords: {summary['rows_with_ioc_keywords']:,}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
