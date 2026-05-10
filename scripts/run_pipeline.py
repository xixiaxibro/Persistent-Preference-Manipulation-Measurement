#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_runs_base(config_path: Path | None) -> Path:
    if config_path is None or not config_path.exists():
        return _repo_root() / "runs"
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("runs_base:"):
            value = stripped.split(":", 1)[1].strip()
            if value:
                return Path(value)
    return _repo_root() / "runs"


def _run(command: list[str]) -> None:
    print(json.dumps({"stage": "exec", "command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal measurement pipeline.")
    parser.add_argument("--config", default="", help="Optional YAML config with runs_base.")
    parser.add_argument("--run-id", required=True, help="Run identifier under runs_base.")
    parser.add_argument("--crawl", required=True, help="Common Crawl id, or DEMO for fixtures.")
    parser.add_argument("--paths-file", default="", help="WAT paths file. Required for demo fixtures.")
    parser.add_argument("--runs-base", default="", help="Override output root. Defaults to config runs_base or ./runs.")
    parser.add_argument("--classifier", choices=("rule", "semantic"), default="rule", help="Classifier backend.")
    parser.add_argument("--model-dir", default="", help="Semantic model directory when --classifier semantic.")
    parser.add_argument("--device", default="cpu", help="Semantic classifier device.")
    parser.add_argument("--workers", type=int, default=1, help="Collection workers.")
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing run directory before starting.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = _repo_root()
    scripts_dir = repo_root / "scripts"
    config_path = Path(args.config) if args.config else None
    runs_base = Path(args.runs_base) if args.runs_base else _read_runs_base(config_path)
    run_root = runs_base / args.run_id

    if args.overwrite and run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    collect_out = run_root / "00_collect" / "prompt_links.jsonl"
    filter_out = run_root / "01_filter_by_platform" / "prompt_links.jsonl"
    classify_out = run_root / "02_classify" / "classified_prompt_links.jsonl"
    source_out = run_root / "03_source_risk"
    target_out = run_root / "03_target_risk"
    cross_out = run_root / "04_cross_crawl"

    collect_cmd = [
        sys.executable,
        str(scripts_dir / "collect_candidate_pages_from_wat.py"),
        "--crawl",
        args.crawl,
        "--output",
        str(collect_out),
        "--workers",
        str(args.workers),
    ]
    if args.paths_file:
        collect_cmd.extend(["--paths-file", args.paths_file])
    _run(collect_cmd)

    _run(
        [
            sys.executable,
            str(scripts_dir / "filter_by_platform.py"),
            "--input",
            str(collect_out),
            "--output",
            str(filter_out),
        ]
    )

    classify_cmd = [
        sys.executable,
        str(scripts_dir / "classify_prompt_links.py"),
        "--input",
        str(filter_out),
        "--output",
        str(classify_out),
        "--classifier",
        args.classifier,
        "--include-benign",
    ]
    if args.classifier == "semantic":
        if not args.model_dir:
            raise SystemExit("--model-dir is required with --classifier semantic")
        classify_cmd.extend(["--model-dir", args.model_dir, "--device", args.device])
    _run(classify_cmd)

    _run(
        [
            sys.executable,
            str(scripts_dir / "analyze_source_risk.py"),
            "--input",
            str(classify_out),
            "--output-dir",
            str(source_out),
            "--crawl-name",
            args.crawl,
        ]
    )
    _run(
        [
            sys.executable,
            str(scripts_dir / "analyze_target_risk.py"),
            "--input",
            str(classify_out),
            "--output-dir",
            str(target_out),
            "--crawl-name",
            args.crawl,
        ]
    )
    _run(
        [
            sys.executable,
            str(scripts_dir / "run_cross_crawl_summary.py"),
            "--run-root",
            str(run_root),
            "--crawl-name",
            args.crawl,
            "--comparison-root",
            str(cross_out),
            "--allow-existing-output",
        ]
    )

    print(json.dumps({"run_root": str(run_root), "summary": str(cross_out / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
