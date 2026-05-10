# Pipeline Overview

This project implements a staged measurement pipeline for prompt-carrying links targeting AI assistant platforms.

## Stages

1. `collect_candidate_pages_from_wat.py` reads Common Crawl WAT metadata and emits prompt-link candidates.
2. `filter_by_platform.py` keeps links targeting known AI assistant platforms using `platform_signatures.py`.
3. `classify_prompt_links.py` runs the active semantic classifier through `semantic_classify_pipeline.py`.
4. `analyze_source_risk.py` aggregates medium/high-risk rows by source domain.
5. `analyze_target_risk.py` aggregates medium/high-risk rows by target platform and IOC metadata.
6. `run_simplified_risk_analysis.py` combines per-crawl source and target risk outputs.
7. `run_template_reuse_analysis.py` measures exact prompt-template reuse.
8. `run_prompt_language_analysis.py` measures prompt language distribution.
9. `run_source_distribution_analysis.py` profiles source-domain distribution.
10. `measure_citemet_default.py` and `measure_citemet_reference_templates.py` check reverse template coverage.
11. Replay validation is currently reserved as an external/placeholder component.

## Artifact Layout

Per-crawl runs are expected to use this external directory shape:

```text
collect_<crawl_id>/
├── 00_collect/prompt_links.jsonl
├── 01_filter_by_platform/prompt_links.jsonl
├── 02_classify/classified_prompt_links.<model-id>.jsonl
├── 03_source_url_analysis/
├── 03b_target_analysis/
├── 03c_language_analysis/
└── 03d_template_reuse_analysis/
```

The artifact layout documents the pipeline state, but it is not the GitHub repository layout.
