"""
LLM-based relabeling of unlabeled candidate samples.

Reads ``unlabeled_for_relabel.jsonl`` from the dataset construction step,
sends each ``primary_prompt_text`` to an LLM for 5-label multi-label
classification, and partitions the results:

    - Rows assigned to one or more of the 5 labels → exported as LLM-labeled
      training rows.
    - Rows assigned zero labels → exported as all-zero training rows.

The script auto-loads the project-level ``.env`` file and calls the LLM via
the OpenAI Python SDK against an OpenAI-compatible API endpoint. By default it
uses the current project configuration:

    - ``OPENAI_BASE_URL`` → defaults to ``https://api.deepseek.com``
    - ``DEEPSEEK_API_KEY`` → preferred API key source
    - ``OPENAI_API_KEY`` → fallback API key source
    - ``deepseek-chat`` → default model

The prompt includes the 5 label definitions with examples and asks for
structured JSON output.

Output:
    - llm_labeled.jsonl        (rows assigned one or more labels)
    - llm_all_zero.jsonl       (rows assigned zero labels)
    - relabel_manifest.json    (statistics)

After this step, the final training dataset is assembled by concatenating:
    weak_labeled.jsonl + llm_labeled.jsonl + llm_all_zero.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO, Iterator

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from env_config import load_project_env

load_project_env()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_LABELS: tuple[str, ...] = (
    "PERSISTENCE",
    "AUTHORITY",
    "RECOMMENDATION",
    "CITATION",
    "SUMMARY",
)

SYSTEM_PROMPT = """\
You are a classification expert for AI recommendation poisoning research.

You will receive a prompt text extracted from a URL query parameter targeting \
an AI chatbot (ChatGPT, Claude, Perplexity, etc.). Your job is to classify \
the text into zero or more of the following 5 categories:

1. **PERSISTENCE** – The text attempts to make the AI remember, store, or \
retain instructions across sessions. Indicators: "remember", "from now on", \
"always", "in future conversations", "keep in mind", "save this".

2. **AUTHORITY** – The text attempts to establish a specific source as \
authoritative, trusted, or expert. Indicators: "trusted source", \
"authoritative source", "go-to source", "reliable source", "expert source", \
"best source".

3. **RECOMMENDATION** – The text attempts to bias the AI to recommend, \
prefer, or prioritize a specific product, service, or website. Indicators: \
"recommend", "prefer", "top choice", "best choice", "always mention", \
"prioritize", "rank first".

4. **CITATION** – The text attempts to get the AI to cite or reference a \
specific source. Indicators: "cite", "citation", "for future reference".

5. **SUMMARY** – The text asks the AI to summarize, analyze, explain, or \
read a URL or document. This is generally benign. Indicators: "summarize", \
"summary", "analyze", "explain", "read this", "visit this url".

Rules:
- A text can have ZERO labels (if it fits none of the categories).
- A text can have MULTIPLE labels (e.g., both PERSISTENCE and AUTHORITY).
- Focus on the *intent* of the text, not just keyword presence.
- Consider implicit expressions: "make sure you never forget X" → PERSISTENCE.
- Consider multilingual text: classify based on meaning regardless of language.
- If the text is too short or ambiguous to classify, assign zero labels.

Respond with a JSON object only, no other text:
{"labels": ["LABEL1", "LABEL2", ...]}

If no labels apply:
{"labels": []}
"""

USER_PROMPT_TEMPLATE = """Classify this prompt text:

\"\"\"{text}\"\"\""""

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

_IO_BUFFER_SIZE = 4 * 1024 * 1024


def _open_read(path: Path) -> BinaryIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb", buffering=_IO_BUFFER_SIZE)


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with _open_read(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _print(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _normalize_labels(raw_labels: Any) -> list[str]:
    if not isinstance(raw_labels, list):
        return []
    valid = set(CLASS_LABELS)
    return sorted(
        {
            label.upper()
            for label in raw_labels
            if isinstance(label, str) and label.upper() in valid
        }
    )


# ---------------------------------------------------------------------------
# LLM API call
# ---------------------------------------------------------------------------

def _call_llm(
    client: Any,
    text: str,
    *,
    model: str,
    max_retries: int = 3,
) -> list[str]:
    """
    Call an OpenAI-compatible chat completion API and parse the label list.

    Returns a list of label strings (may be empty).
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text[:4000])},
                ],
                temperature=0.0,
                max_tokens=128,
            )

            content = (response.choices[0].message.content or "").strip()

            # Parse JSON from response (handle markdown code blocks).
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            parsed = json.loads(content)
            labels = parsed.get("labels", [])
            if not isinstance(labels, list):
                return []
            # Validate labels.
            return _normalize_labels(labels)

        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                _print(f"  Retry {attempt + 1}/{max_retries} after error: {exc} (wait {wait}s)")
                time.sleep(wait)
            else:
                _print(f"  Failed after {max_retries} attempts: {exc}")
                return []

    return []


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def relabel_unlabeled(
    input_path: Path,
    output_dir: Path,
    *,
    client: Any,
    api_base: str,
    model: str,
    batch_delay: float = 0.1,
) -> dict[str, Any]:
    rows = list(_iter_rows(input_path))
    total = len(rows)
    _print(f"Loaded {total:,} unlabeled candidates for relabeling.")

    llm_labeled_rows: list[dict[str, Any]] = []
    llm_all_zero_rows: list[dict[str, Any]] = []
    errors = 0

    start_time = time.monotonic()

    for i, row in enumerate(rows):
        text = row.get("primary_prompt_text", "")
        if not isinstance(text, str) or not text.strip():
            entry = dict(row)
            entry["dataset_role"] = "llm_all_zero"
            entry["dataset_source"] = "llm_relabel"
            entry["llm_assigned_labels"] = []
            entry["prompt_labels"] = []
            entry["classification"] = ""
            llm_all_zero_rows.append(entry)
            continue

        labels = _call_llm(
            client,
            text,
            model=model,
        )

        entry = dict(row)
        entry["llm_assigned_labels"] = labels

        if labels:
            entry["dataset_role"] = "llm_labeled"
            entry["dataset_source"] = "llm_relabel"
            entry["prompt_labels"] = labels
            entry["classification"] = ";".join(labels)
            llm_labeled_rows.append(entry)
        else:
            entry["dataset_role"] = "llm_all_zero"
            entry["dataset_source"] = "llm_relabel"
            entry["prompt_labels"] = []
            entry["classification"] = ""
            llm_all_zero_rows.append(entry)

        # Progress.
        if (i + 1) % 100 == 0:
            elapsed = time.monotonic() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            _print(
                f"  [{i + 1}/{total}]  "
                f"labeled: {len(llm_labeled_rows)}  "
                f"all-zero: {len(llm_all_zero_rows)}  "
                f"rate: {rate:.1f}/s  "
                f"ETA: {eta:.0f}s"
            )

        if batch_delay > 0:
            time.sleep(batch_delay)

    elapsed = time.monotonic() - start_time

    # Write outputs.
    labeled_path = output_dir / "llm_labeled.jsonl"
    all_zero_path = output_dir / "llm_all_zero.jsonl"

    n_labeled = _write_jsonl(llm_labeled_rows, labeled_path)
    n_all_zero = _write_jsonl(llm_all_zero_rows, all_zero_path)

    _print(f"\nLLM-labeled rows:      {labeled_path} ({n_labeled:,} rows)")
    _print(f"LLM all-zero rows:     {all_zero_path} ({n_all_zero:,} rows)")

    manifest = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "api_base": api_base,
        "model": model,
        "total_candidates": total,
        "llm_labeled": n_labeled,
        "llm_all_zero": n_all_zero,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 2),
        "llm_labeled_file": str(labeled_path),
        "llm_all_zero_file": str(all_zero_path),
    }
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relabel unlabeled candidates using an LLM for 5-label multi-label classification.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="unlabeled_for_relabel.jsonl from build_classification_dataset.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
        help="OpenAI-compatible API base URL (default: $OPENAI_BASE_URL or DeepSeek).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
        help="API key (default: $DEEPSEEK_API_KEY, else $OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="Model name (default: deepseek-chat).",
    )
    parser.add_argument(
        "--batch-delay",
        type=float,
        default=0.1,
        help="Seconds between API calls (default: 0.1).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.api_key:
        print(
            "Error: --api-key is required, or set DEEPSEEK_API_KEY / OPENAI_API_KEY in .env.",
            file=sys.stderr,
        )
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print(
            "Missing dependency: openai. Install it with `python3 -m pip install -r requirements.txt`.",
            file=sys.stderr,
        )
        return 1

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    manifest = relabel_unlabeled(
        input_path,
        output_dir,
        client=client,
        api_base=args.api_base,
        model=args.model,
        batch_delay=args.batch_delay,
    )

    manifest_path = output_dir / "relabel_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())