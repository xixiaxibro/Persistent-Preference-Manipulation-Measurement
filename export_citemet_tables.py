#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


ORDERED_CLASSES = (
    "exact_default",
    "exact_default_url_normalized",
    "fixed_grammar_variant",
    "platform_url_signature",
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
    parser = argparse.ArgumentParser(description="Export paper-facing CiteMET coverage tables from detector output.")
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more matched-row CSV files produced by measure_citemet_default.py.",
    )
    parser.add_argument("--out", required=True, help="Output directory for paper tables.")
    return parser.parse_args()


def iter_csv(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_summary(input_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_csv = input_path.with_suffix(".summary.csv")
    summary_json = input_path.with_suffix(".summary.json")
    summary_rows = list(iter_csv(summary_csv)) if summary_csv.exists() else []
    payload: dict[str, Any] = {}
    if summary_json.exists():
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
    return summary_rows, payload


def to_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pct(part: int, total: int) -> float:
    return round(part / total, 6) if total else 0.0


def prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8", errors="replace")).hexdigest()


def main() -> int:
    raise_csv_field_limit()
    args = parse_args()
    input_paths = [Path(path) for path in args.input]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    medium_high_rows = 0
    class_rows: collections.Counter[str] = collections.Counter()
    class_medium_high_rows: collections.Counter[str] = collections.Counter()
    unique_prompts: dict[str, set[str]] = collections.defaultdict(set)
    unique_source_roots: dict[str, set[str]] = collections.defaultdict(set)
    unique_platforms: dict[str, set[str]] = collections.defaultdict(set)
    by_crawl: collections.Counter[tuple[str, str]] = collections.Counter()
    by_platform: collections.Counter[tuple[str, str]] = collections.Counter()
    by_source_root: collections.Counter[tuple[str, str]] = collections.Counter()
    by_signature: collections.Counter[tuple[str, str]] = collections.Counter()

    per_input_payloads: list[dict[str, Any]] = []
    for input_path in input_paths:
        summary_rows, payload = read_summary(input_path)
        per_input_payloads.append({"input": str(input_path), **payload})
        total_rows += to_int(payload.get("processed_rows"))
        medium_high_rows += to_int(payload.get("medium_high_rows"))
        for summary_row in summary_rows:
            class_name = str(summary_row.get("citemet_class", ""))
            class_rows[class_name] += to_int(summary_row.get("rows"))
            class_medium_high_rows[class_name] += to_int(summary_row.get("medium_high_rows"))

        for row in iter_csv(input_path):
            class_name = str(row.get("citemet_class", ""))
            if class_name not in ORDERED_CLASSES:
                continue
            crawl = str(row.get("crawl", ""))
            platform = str(row.get("target_platform", ""))
            source_root = str(row.get("source_root", ""))
            signature = str(row.get("platform_url_signature", ""))
            prompt = str(row.get("prompt_normalized", ""))

            by_crawl[(crawl, class_name)] += 1
            by_platform[(platform, class_name)] += 1
            by_source_root[(source_root, class_name)] += 1
            by_signature[(signature, class_name)] += 1
            if prompt and class_name != "platform_url_signature":
                unique_prompts[class_name].add(prompt_fingerprint(prompt))
            if source_root:
                unique_source_roots[class_name].add(source_root)
            if platform:
                unique_platforms[class_name].add(platform)

    summary_rows = []
    for class_name in sorted(set(class_rows) | set(ORDERED_CLASSES)):
        rows = class_rows[class_name]
        risky_rows = class_medium_high_rows[class_name]
        summary_rows.append(
            {
                "citemet_class": class_name,
                "rows": rows,
                "share_of_all_platform_matched": pct(rows, total_rows),
                "medium_high_rows": risky_rows,
                "share_of_medium_high": pct(risky_rows, medium_high_rows),
                "unique_prompts": len(unique_prompts.get(class_name, set())),
                "unique_source_roots": len(unique_source_roots.get(class_name, set())),
                "unique_target_platforms": len(unique_platforms.get(class_name, set())),
            }
        )

    overview_fields = [
        "citemet_class",
        "rows",
        "share_of_all_platform_matched",
        "medium_high_rows",
        "share_of_medium_high",
        "unique_prompts",
        "unique_source_roots",
        "unique_target_platforms",
    ]
    write_csv(out_dir / "citemet_default_coverage.csv", summary_rows, overview_fields)

    crawl_rows = [
        {"crawl": crawl, "citemet_class": class_name, "rows": count}
        for (crawl, class_name), count in sorted(by_crawl.items(), key=lambda item: (item[0][0], ORDERED_CLASSES.index(item[0][1])))
    ]
    write_csv(out_dir / "citemet_by_crawl.csv", crawl_rows, ["crawl", "citemet_class", "rows"])

    platform_rows = [
        {"target_platform": platform, "citemet_class": class_name, "rows": count}
        for (platform, class_name), count in sorted(by_platform.items(), key=lambda item: (item[0][0], ORDERED_CLASSES.index(item[0][1])))
    ]
    write_csv(out_dir / "citemet_by_platform.csv", platform_rows, ["target_platform", "citemet_class", "rows"])

    signature_rows = [
        {"platform_url_signature": signature, "citemet_class": class_name, "rows": count}
        for (signature, class_name), count in sorted(by_signature.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
        if signature
    ]
    write_csv(out_dir / "citemet_by_url_signature.csv", signature_rows, ["platform_url_signature", "citemet_class", "rows"])

    top_source_rows = [
        {"source_root": source_root, "citemet_class": class_name, "rows": count}
        for (source_root, class_name), count in sorted(by_source_root.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
        if source_root
    ][:50]
    write_csv(out_dir / "top_citemet_source_roots.csv", top_source_rows, ["source_root", "citemet_class", "rows"])

    summary_by_class = {row["citemet_class"]: row for row in summary_rows}
    lines = [
        "# CiteMET Default-Format Coverage",
        "",
        f"- Input rows processed: `{total_rows:,}`",
        f"- Medium/high-risk denominator: `{medium_high_rows:,}`",
        "- Unique prompts are tracked for L1/L2 prompt-format matches. They are not tracked for L3 URL-signature-only rows.",
        "",
        "| Class | Rows | Share of all platform-matched | Medium/high rows | Share of medium/high | Unique prompts | Source roots | Target platforms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for class_name in ORDERED_CLASSES:
        row = summary_by_class.get(class_name)
        if not row:
            continue
        lines.append(
            f"| `{row.get('citemet_class', '')}` | {to_int(row.get('rows')):,} | {fmt_pct(float(row.get('share_of_all_platform_matched', 0.0)))} | "
            f"{to_int(row.get('medium_high_rows')):,} | {fmt_pct(float(row.get('share_of_medium_high', 0.0)))} | "
            f"{to_int(row.get('unique_prompts')):,} | {to_int(row.get('unique_source_roots')):,} | {to_int(row.get('unique_target_platforms')):,} |"
        )
    lines.extend(
        [
            "",
            "Suggested wording:",
            "",
            "We do not attribute every matched link to the npm package itself. Instead, we measure links that match the default format exposed by the package, which can include direct use, copied use, and near-default reuse.",
            "",
        ]
    )
    (out_dir / "citemet_default_coverage.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "inputs": [str(path) for path in input_paths],
        "out_dir": str(out_dir),
        "processed_rows": total_rows,
        "medium_high_rows": medium_high_rows,
        "per_input": per_input_payloads,
        "files": sorted(str(path) for path in out_dir.iterdir()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
