# Unveiling Persistent Preference Manipulation

This repository contains the code for a paper reproduction pipeline that measures prompt-carrying links in Common Crawl WAT snapshots and analyzes how they target AI assistant platforms.

The repository is intended to contain code, configuration examples, documentation, and tiny examples only. Paper-scale Common Crawl artifacts, classified corpora, logs, caches, model weights, and replay results are external artifacts and must not be committed.

## Pipeline Overview

The main pipeline is:

```text
Common Crawl WAT snapshots
-> Prompt-link candidate collection
-> AI platform filtering
-> Semantic intent classification
-> Source / target / IOC risk aggregation
-> Longitudinal cross-crawl analysis
-> Template reuse, language, and source distribution analyses
-> Reverse template coverage checks
-> Replay-based validation
-> Paper tables and reports
```

The canonical per-crawl entrypoint is:

```bash
./run_collect_to_analysis.sh
```

That wrapper runs:

1. `collect_candidate_pages_from_wat.py`
2. `filter_by_platform.py`
3. `classify_prompt_links.py`
4. `analyze_source_risk.py`
5. `analyze_target_risk.py`

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── requirements-inference.txt
├── configs/
│   ├── runs.example.yaml
│   └── thresholds.example.json
├── docs/
│   ├── artifact_policy.md
│   ├── pipeline_overview.md
│   └── reproduction_notes.md
├── examples/
│   ├── sample_classified_prompt_links.jsonl
│   ├── sample_prompt_links.jsonl
│   └── sample_replay_manifest.csv
├── replay_validation/
│   └── README.md
├── run_collect_to_analysis.sh
└── *.py
```

The Python scripts remain at repository root in this low-risk cleanup pass to avoid changing import paths. A later refactor can move reusable modules under `src/` and command-line entrypoints under `scripts/`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt -r requirements-inference.txt
```

The semantic classifier requires a compatible local model directory containing:

```text
MODEL_DIR/
├── thresholds.json
└── model/
```

Model weights are not included in this repository.

## Configuration

All paper-scale outputs should be written outside the git working tree.

Common environment variables:

```bash
export RUNS_BASE=/path/to/external/runs
export MODEL_DIR=/path/to/classifier_checkpoint
export REPORTS_DIR=/path/to/external/reports
export TRANCO_CACHE=/path/to/tranco_top1m.csv
```

`ARTIFACT_ROOT` may be used as an alias for `RUNS_BASE`. See `configs/runs.example.yaml` for an example multi-crawl configuration.

## Running the Per-Crawl Pipeline

```bash
RUN_ID=collect_ccmain2026_12 \
CRAWL=CC-MAIN-2026-12 \
RUNS_BASE=/path/to/external/runs \
MODEL_DIR=/path/to/classifier_checkpoint \
CLASSIFY_DEVICE=cpu \
./run_collect_to_analysis.sh
```

The wrapper writes stage outputs under:

```text
$RUNS_BASE/$RUN_ID/
├── 00_collect/
├── 01_filter_by_platform/
├── 02_classify/
├── 03_source_url_analysis/
└── 03b_target_analysis/
```

Resume from an existing stage by setting `START_STAGE` and `END_STAGE`.

## Running Cross-Crawl Risk Analysis

After multiple per-crawl run roots have completed Stage 02 and risk analysis:

```bash
RUNS_BASE=/path/to/external/runs \
python3 run_simplified_risk_analysis.py
```

You can override run roots explicitly with repeated `--run-root` and matching `--crawl-name` arguments.

## Template Reuse, Language, and Source Distribution

```bash
RUNS_BASE=/path/to/external/runs python3 run_template_reuse_analysis.py
RUNS_BASE=/path/to/external/runs python3 run_prompt_language_analysis.py
RUNS_BASE=/path/to/external/runs python3 run_source_distribution_analysis.py
```

These scripts read classified per-crawl outputs and write derived tables/reports to external artifact directories.

## Reverse Template Coverage

Use:

```bash
python3 measure_citemet_default.py --help
python3 measure_citemet_reference_templates.py --help
```

Full coverage tables are generated artifacts and are not tracked in git.

## Replay Validation

Replay validation is reserved for a separate component. The repository currently keeps `replay_validation/README.md` as a placeholder and `examples/sample_replay_manifest.csv` as a toy manifest shape.

## Data and Artifact Policy

Do not commit:

- Common Crawl-derived JSONL corpora
- full classified corpora
- full risk tables and paper result tables
- replay result dumps
- logs, caches, temporary outputs
- model checkpoints or tokenizer files
- `.env`, tokens, credentials, or private absolute paths

See `docs/artifact_policy.md`.

## Limitations

This repository provides the code path and small examples. Reproducing paper-scale results requires external storage, Common Crawl access, a compatible semantic classifier checkpoint, and separately managed artifacts.

No paper result numbers are claimed in this README. If result numbers are added later, cite the exact artifact path, command, and generation date.

## License

No license has been selected yet. Until a license is added, default copyright restrictions apply and external users do not receive clear reuse rights. See `docs/reproduction_notes.md` for license tradeoffs to decide before public release.

## Citation / Contact

Citation and contact information will be added after paper metadata is finalized.
