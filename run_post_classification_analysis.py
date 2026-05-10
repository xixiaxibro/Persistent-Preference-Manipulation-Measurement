#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from source_url_analysis_common import ensure_directory, iso_now_epoch, write_json

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_BASE = Path(os.environ.get("RUNS_BASE") or os.environ.get("ARTIFACT_ROOT") or PROJECT_ROOT / "runs")

DEFAULT_RUNS: tuple[tuple[str, str], ...] = (
    ("CC-MAIN-2025-51", str(DEFAULT_RUNS_BASE / "collect_ccmain2025_51")),
    ("CC-MAIN-2026-04", str(DEFAULT_RUNS_BASE / "collect_ccmain2026_04")),
    ("CC-MAIN-2026-08", str(DEFAULT_RUNS_BASE / "collect_ccmain2026_08")),
    ("CC-MAIN-2026-12", str(DEFAULT_RUNS_BASE / "collect_ccmain2026_12")),
)
DEFAULT_COMPARISON_ROOT = str(DEFAULT_RUNS_BASE / "post_classification_comparison")


def _resolve_classified_input(run_root: Path) -> Path:
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
            "Pass explicit run roots after cleaning duplicates or keep only one classified JSONL per crawl."
        )
    return candidates[0]


def _run_command(command: list[str]) -> None:
    print(json.dumps({"stage": "exec", "command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True)


def _ensure_output_dir_available(path: Path, *, allow_existing_output: bool) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise NotADirectoryError(f"Expected directory path: {path}")
    if any(path.iterdir()) and not allow_existing_output:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use --allow-existing-output if you want to reuse/overwrite it."
        )


def _resolve_run_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.run_root:
        run_roots = [Path(value) for value in args.run_root]
        if args.crawl_name and len(args.crawl_name) != len(run_roots):
            raise ValueError("When --crawl-name is provided, it must appear exactly once per --run-root.")
        crawl_names = args.crawl_name or [path.name for path in run_roots]
        pairs = zip(crawl_names, run_roots)
    else:
        if args.crawl_name:
            raise ValueError("--crawl-name requires matching --run-root values.")
        pairs = ((crawl, Path(run_root)) for crawl, run_root in DEFAULT_RUNS)

    specs: list[dict[str, Any]] = []
    for crawl, run_root in pairs:
        input_path = _resolve_classified_input(run_root)
        specs.append(
            {
                "crawl": crawl,
                "run_root": run_root,
                "input": input_path,
                "source_output_dir": run_root / args.source_output_dirname,
                "target_output_dir": run_root / args.target_output_dirname,
            }
        )
    return specs


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run per-crawl and cross-crawl post-classification source/target analysis over completed Stage 02 outputs."
    )
    parser.add_argument(
        "--analysis",
        choices=("all", "source", "target"),
        default="all",
        help="Which analysis family to run (default: all).",
    )
    parser.add_argument(
        "--run-root",
        action="append",
        default=[],
        help="Optional run root to include. Repeatable. Default: the four known crawl runs.",
    )
    parser.add_argument(
        "--crawl-name",
        action="append",
        default=[],
        help="Optional crawl name matching each --run-root. Repeatable.",
    )
    parser.add_argument(
        "--script-dir",
        default=str(script_dir),
        help=f"Repository script directory (default: {script_dir}).",
    )
    parser.add_argument(
        "--comparison-root",
        default=DEFAULT_COMPARISON_ROOT,
        help=f"Shared root for cross-crawl comparison outputs (default: {DEFAULT_COMPARISON_ROOT}).",
    )
    parser.add_argument(
        "--source-output-dirname",
        default="03_source_url_analysis",
        help="Per-crawl source analysis directory name (default: 03_source_url_analysis).",
    )
    parser.add_argument(
        "--target-output-dirname",
        default="03b_target_analysis",
        help="Per-crawl target analysis directory name (default: 03b_target_analysis).",
    )
    parser.add_argument("--source-top-n", type=int, default=200, help="Top-N rows for per-crawl source review tables.")
    parser.add_argument("--target-top-n", type=int, default=200, help="Top-N rows for per-crawl target review tables.")
    parser.add_argument("--compare-top-n", type=int, default=200, help="Top-N rows for cross-crawl comparison tables.")
    parser.add_argument("--source-shards", type=int, default=32, help="Shard count for source analysis (default: 32).")
    parser.add_argument(
        "--tranco-cache",
        default=str(script_dir / "tranco_top1m.csv"),
        help="Tranco cache path for source analysis.",
    )
    parser.add_argument(
        "--tranco-mode",
        choices=("fixed", "download-if-missing"),
        default="download-if-missing",
        help="Tranco loading mode for source analysis.",
    )
    parser.add_argument(
        "--only-nonempty-prompt",
        action="store_true",
        help="Analyze only rows with non-empty primary_prompt_text.",
    )
    parser.add_argument(
        "--suspicious-only",
        action="store_true",
        help="Analyze only rows where is_suspicious is true.",
    )
    parser.add_argument(
        "--per-crawl-only",
        action="store_true",
        help="Run only per-crawl analyses and skip cross-crawl comparison.",
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Skip per-crawl execution and compare the existing per-crawl outputs only.",
    )
    parser.add_argument(
        "--skip-source-url-persistence",
        action="store_true",
        help="Skip the heaviest source_url persistence table in cross-crawl source comparison.",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow non-empty output directories to exist before execution.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.per_crawl_only and args.comparison_only:
        raise ValueError("--per-crawl-only and --comparison-only cannot be used together.")

    script_dir = Path(args.script_dir)
    if not script_dir.is_dir():
        raise FileNotFoundError(f"Script directory does not exist: {script_dir}")

    include_source = args.analysis in {"all", "source"}
    include_target = args.analysis in {"all", "target"}
    run_specs = _resolve_run_specs(args)

    comparison_root = Path(args.comparison_root)
    source_comparison_dir = comparison_root / "source_analysis"
    target_comparison_dir = comparison_root / "target_analysis"

    if not args.comparison_only:
        if include_source:
            source_script = script_dir / "run_sharded_source_analysis.py"
            for spec in run_specs:
                _ensure_output_dir_available(spec["source_output_dir"], allow_existing_output=args.allow_existing_output)
                command = [
                    sys.executable,
                    str(source_script),
                    "--input",
                    str(spec["input"]),
                    "--output-dir",
                    str(spec["source_output_dir"]),
                    "--crawl-name",
                    str(spec["crawl"]),
                    "--top-n",
                    str(args.source_top_n),
                    "--examples-per-group",
                    "2",
                    "--tranco-cache",
                    args.tranco_cache,
                    "--tranco-mode",
                    args.tranco_mode,
                    "--shards",
                    str(args.source_shards),
                ]
                if args.only_nonempty_prompt:
                    command.append("--only-nonempty-prompt")
                if args.suspicious_only:
                    command.append("--suspicious-only")
                _run_command(command)

        if include_target:
            target_script = script_dir / "analyze_target_urls.py"
            for spec in run_specs:
                _ensure_output_dir_available(spec["target_output_dir"], allow_existing_output=args.allow_existing_output)
                command = [
                    sys.executable,
                    str(target_script),
                    "--input",
                    str(spec["input"]),
                    "--output-dir",
                    str(spec["target_output_dir"]),
                    "--crawl-name",
                    str(spec["crawl"]),
                    "--top-n",
                    str(args.target_top_n),
                ]
                if args.only_nonempty_prompt:
                    command.append("--only-nonempty-prompt")
                if args.suspicious_only:
                    command.append("--suspicious-only")
                _run_command(command)

    if not args.per_crawl_only:
        ensure_directory(comparison_root)
        if include_source:
            _ensure_output_dir_available(source_comparison_dir, allow_existing_output=args.allow_existing_output)
            source_compare_script = script_dir / "compare_source_url_snapshots.py"
            command = [
                sys.executable,
                str(source_compare_script),
                "--output-dir",
                str(source_comparison_dir),
                "--top-n",
                str(args.compare_top_n),
                "--input-dirs",
                *[str(spec["source_output_dir"]) for spec in run_specs],
            ]
            if args.skip_source_url_persistence:
                command.append("--skip-source-url-persistence")
            _run_command(command)

        if include_target:
            _ensure_output_dir_available(target_comparison_dir, allow_existing_output=args.allow_existing_output)
            target_compare_script = script_dir / "compare_target_url_snapshots.py"
            command = [
                sys.executable,
                str(target_compare_script),
                "--output-dir",
                str(target_comparison_dir),
                "--top-n",
                str(args.compare_top_n),
                "--input-dirs",
                *[str(spec["target_output_dir"]) for spec in run_specs],
            ]
            _run_command(command)

        manifest = {
            "generated_at_epoch": iso_now_epoch(),
            "script": "run_post_classification_analysis.py",
            "analysis": args.analysis,
            "comparison_only": args.comparison_only,
            "per_crawl_only": args.per_crawl_only,
            "filters": {
                "only_nonempty_prompt": args.only_nonempty_prompt,
                "suspicious_only": args.suspicious_only,
            },
            "runs": [
                {
                    "crawl": spec["crawl"],
                    "run_root": str(spec["run_root"]),
                    "input": str(spec["input"]),
                    "source_output_dir": str(spec["source_output_dir"]),
                    "target_output_dir": str(spec["target_output_dir"]),
                }
                for spec in run_specs
            ],
            "comparison_root": str(comparison_root),
            "source_comparison_dir": str(source_comparison_dir) if include_source else "",
            "target_comparison_dir": str(target_comparison_dir) if include_target else "",
        }
        write_json(comparison_root / "post_classification_analysis_manifest.json", manifest)

    print(
        json.dumps(
            {
                "analysis": args.analysis,
                "crawl_count": len(run_specs),
                "comparison_root": str(comparison_root),
                "source_enabled": include_source,
                "target_enabled": include_target,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
