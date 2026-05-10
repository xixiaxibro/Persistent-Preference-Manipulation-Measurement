# Persistent Preference Manipulation Measurement

This repository is a minimal, runnable artifact for a measurement pipeline over prompt-carrying links in Common Crawl WAT metadata.

It is intentionally not a dump of the internal research workspace. It contains only the public reproduction path:

```text
collect -> platform filter -> classify -> source/target risk -> cross-crawl summary
```

Large Common Crawl-derived artifacts, model checkpoints, internal paper-table exports, historical experiments, replay traces, logs, and caches are not included.

## Quick Start

Run the synthetic end-to-end demo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

python scripts/run_pipeline.py \
  --config configs/runs.example.yaml \
  --run-id demo \
  --crawl DEMO \
  --paths-file examples/fixtures/demo_wat.paths \
  --classifier rule \
  --overwrite
```

The demo writes:

```text
runs/demo/
├── 00_collect/prompt_links.jsonl
├── 01_filter_by_platform/prompt_links.jsonl
├── 02_classify/classified_prompt_links.jsonl
├── 03_source_risk/
├── 03_target_risk/
└── 04_cross_crawl/
```

## Repository Layout

```text
src/unveiling_persistent/   reusable pipeline modules
scripts/                    command-line entrypoints
configs/                    example configuration files
examples/fixtures/          synthetic smoke-test inputs and expected shape
examples/                   tiny schema examples
docs/                       artifact and reproduction notes
replay_validation/          placeholder only; replay is not public yet
paper/                      lightweight paper-facing notes only
```

## Classifiers

The public artifact has two classifier modes:

- `rule`: deterministic default classifier for demos, smoke tests, and artifact review. It requires no model weights.
- `semantic`: optional compatibility mode for an external semantic classifier checkpoint. Use `--classifier semantic --model-dir /path/to/checkpoint` and install `requirements-inference.txt`.

Model weights and tokenizer files are intentionally excluded from git.

## Running on Common Crawl

For a real crawl, point the pipeline at external artifact storage:

```bash
RUNS_BASE=/path/to/external/runs \
python scripts/run_pipeline.py \
  --run-id collect_ccmain2026_12 \
  --crawl CC-MAIN-2026-12 \
  --classifier semantic \
  --model-dir /path/to/external/checkpoint \
  --workers 20
```

If you already have a local `wat.paths` file, pass `--paths-file /path/to/wat.paths`.

## Individual Stages

Each stage can also be run directly:

```bash
python scripts/collect_candidate_pages_from_wat.py --crawl CC-MAIN-2026-12 --output runs/demo/00_collect/prompt_links.jsonl
python scripts/filter_by_platform.py --input runs/demo/00_collect/prompt_links.jsonl --output runs/demo/01_filter_by_platform/prompt_links.jsonl
python scripts/classify_prompt_links.py --classifier rule --input runs/demo/01_filter_by_platform/prompt_links.jsonl --output runs/demo/02_classify/classified_prompt_links.jsonl --include-benign
python scripts/analyze_source_risk.py --input runs/demo/02_classify/classified_prompt_links.jsonl --output-dir runs/demo/03_source_risk
python scripts/analyze_target_risk.py --input runs/demo/02_classify/classified_prompt_links.jsonl --output-dir runs/demo/03_target_risk
python scripts/run_cross_crawl_summary.py --run-root runs/demo --crawl-name DEMO --comparison-root runs/demo/04_cross_crawl --allow-existing-output
```

## What Is Not Included

The public repository excludes:

- full Common Crawl-derived JSONL outputs
- full classified corpora and risk tables
- model checkpoints and tokenizer files
- internal exploratory scripts for template reuse, language/source distribution, paper-table export, and historical training workflows
- replay validation implementation and replay traces
- `.env`, credentials, logs, caches, and private server paths

See `docs/artifact_policy.md`.

## License

This repository is released under the MIT License. See `LICENSE`.
