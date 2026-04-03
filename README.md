# Unveiling Persistent Preference Manipulation

This repository contains the current Common Crawl measurement pipeline for collecting, filtering, and classifying prompt-carrying links that target major AI assistant platforms.

The project is no longer based on a single WAT snapshot. We have already completed two full crawl snapshots and are treating this as an ongoing longitudinal measurement effort:

- `CC-MAIN-2025-51` completed
- `CC-MAIN-2026-04` completed
- `CC-MAIN-2026-08` planned
- `CC-MAIN-2026-12` planned

A dedicated `SOURCE_URL` analysis stage is now available for single-crawl characterization and multi-crawl comparison.

## Pipeline

The current pipeline has three completed stages:

1. `collect_candidate_pages_from_wat.py`
   - Scan Common Crawl WAT files.
   - Keep links with prompt-bearing query parameters.
   - Current global filter keys: `q`, `prompt`.
   - Current special case: `x.com/i/grok?text=`.
2. `filter_by_platform.py`
   - Re-match candidate links against supported AI platform signatures.
   - Keep only links that target known assistant platforms.
3. `classify_prompt_links.py`
   - Enrich matched rows with `source_domain`, `target_domain`, `target_platform`, prompt extraction results, severity, labels, IoC metadata, and session-entry metadata.

## Analysis Scripts

The repository now includes two analysis scripts for `SOURCE_URL`-centric measurement:

1. `analyze_source_urls.py`
   - Input: one `classified_prompt_links.jsonl` snapshot.
   - Output: `summary.json`, aggregated CSV tables, reviewer-friendly top tables, and figure-ready CSV files.
   - Built-in analyses:
     - `source_url`, `source_domain`, and `root_domain` concentration
     - path-template and page-kind characterization
     - multi-platform reuse of the same source page
     - Tranco popularity lookup and bucketed analysis
2. `compare_source_url_snapshots.py`
   - Input: multiple output directories produced by `analyze_source_urls.py`.
   - Output: cross-crawl overview, persistence tables, popularity comparison tables, and figure-ready CSV files.

Tranco popularity analysis is done at the `root_domain` level and is intended to support comparisons such as popular-site abuse versus long-tail-site abuse.

## Current Data Coverage

| Crawl | Run root | Collection status | Prompt-link rows after collect | Rows kept after platform filter | Rows written after classification |
| --- | --- | --- | ---: | ---: | ---: |
| `CC-MAIN-2025-51` | `/mnt/data/zhoufl/ai_recommendation_poisoning/runs/20260331_full_collect_ccmain2025_51` | completed | 456,563,273 | 1,530,831 | 1,530,831 |
| `CC-MAIN-2026-04` | `/mnt/data/zhoufl/ai_recommendation_poisoning/runs/20260326_pipeline2_collect_full_ccmain2026_04_multiplatform_rerun_groktext` | completed | 487,520,147 total in downstream filter input | 2,083,728 | 2,083,728 |

For `CC-MAIN-2026-04`, the collection stage resumed from a partial previous run. The raw collection JSON only reports the newly processed tail of the crawl. The full prompt-link total for the finished crawl is verified by the next stage, which read `487,520,147` lines from the collection output.

## Completed Crawl Snapshots

### CC-MAIN-2025-51

Run root:

- `/mnt/data/zhoufl/ai_recommendation_poisoning/runs/20260331_full_collect_ccmain2025_51`

Collection summary:

- `100,000 / 100,000` WAT files processed
- `456,563,273` prompt-link rows written
- `6,498,997,429` pages scanned
- `456,836,727,482` links scanned
- `15,970.84 GB` downloaded
- `176,244.85 s` elapsed
- Average download speed: `92.8 MB/s`

Platform filter summary:

- Input size: `233.61 GB`
- `456,563,273` lines read
- `1,530,831` lines kept
- `926.29 s` elapsed
- Average parallel throughput: `260.4 MB/s`
- `12` platforms matched

Platform distribution after filter:

- `chatgpt`: `559,199`
- `perplexity`: `383,538`
- `grok`: `288,264`
- `claude`: `256,631`
- `gemini`: `16,477`
- `copilot`: `12,521`
- `le_chat`: `10,888`
- `you`: `2,850`
- `meta_ai`: `265`
- `deepseek`: `188`
- `yiyan`: `9`
- `poe`: `1`

Classification summary:

- `1,530,831` rows seen
- `1,530,831` rows written
- `0` unmatched
- `0` benign dropped
- `0` errored
- `590,339` rows with IoC keywords
- `301.97 s` elapsed
- Average rate: `5.1K rows/s`

Severity distribution:

- `low`: `940,492`
- `high`: `397,003`
- `medium`: `193,336`

Label distribution:

- `SUMMARY`: `889,891`
- `PERSISTENCE`: `508,032`
- `CITATION`: `439,834`
- `AUTHORITY`: `127,922`
- `RECOMMENDATION`: `8,009`

Session entry distribution:

- `prompt_params:q`: `1,271,421`
- `prompt_params:text`: `190,901`
- `prompt_params:prompt`: `68,509`

### CC-MAIN-2026-04

Run root:

- `/mnt/data/zhoufl/ai_recommendation_poisoning/runs/20260326_pipeline2_collect_full_ccmain2026_04_multiplatform_rerun_groktext`

Collection status note:

- This run resumed from `35,671` already completed WAT files.
- The collection-stage JSON therefore reports only the newly processed remainder.
- The downstream platform filter confirms that the final collection output contains `487,520,147` prompt-link rows in total.

Collection summary, raw resumed-run counters:

- `100,000 / 100,000` WAT files completed overall
- `310,561,853` newly written prompt-link rows during the resumed segment
- `4,488,009,611` pages scanned during the resumed segment
- `275,807,357,408` links scanned during the resumed segment
- `11,136,014,308,562` bytes downloaded during the resumed segment
- `113,742.52 s` elapsed during the resumed segment
- Average download speed: `93.4 MB/s`

Collection summary, approximate full-crawl equivalent after proportional normalization:

- Remaining WAT files processed in resumed segment: `64,329`
- Scaling factor to 100,000 WAT files: `1.5545`
- Estimated pages scanned for the full crawl: about `6.98B`
- Estimated links scanned for the full crawl: about `428.74B`
- Estimated bytes downloaded for the full crawl: about `17.31 TB`
- Estimated full-run elapsed time at the same rate: about `176,814 s`

Prompt-link total for the finished crawl:

- Exact total verified by downstream filter input: `487,520,147`

Platform filter summary:

- Input size: `247.47 GB`
- `487,520,147` lines read
- `2,083,728` lines kept
- `979.72 s` elapsed
- Average parallel throughput: `262.1 MB/s`
- `13` platforms matched

Platform distribution after filter:

- `chatgpt`: `766,731`
- `perplexity`: `512,411`
- `grok`: `367,620`
- `claude`: `364,845`
- `gemini`: `25,701`
- `le_chat`: `23,179`
- `copilot`: `16,319`
- `you`: `3,258`
- `deepseek`: `3,195`
- `meta_ai`: `447`
- `z_ai`: `12`
- `yiyan`: `9`
- `poe`: `1`

Classification summary:

- `2,083,728` rows seen
- `2,083,728` rows written
- `0` unmatched
- `0` benign dropped
- `0` errored
- `773,425` rows with IoC keywords
- `397.16 s` elapsed
- Average rate: `5.2K rows/s`

Severity distribution:

- `low`: `1,310,303`
- `high`: `544,947`
- `medium`: `228,478`

Label distribution:

- `SUMMARY`: `1,211,047`
- `PERSISTENCE`: `641,770`
- `CITATION`: `616,679`
- `AUTHORITY`: `208,842`
- `RECOMMENDATION`: `19,549`

Session entry distribution:

- `prompt_params:q`: `1,772,556`
- `prompt_params:text`: `228,046`
- `prompt_params:prompt`: `83,126`

## Supported Platforms Observed So Far

Platforms matched in completed crawls:

- `chatgpt`
- `perplexity`
- `grok`
- `claude`
- `gemini`
- `copilot`
- `le_chat`
- `you`
- `meta_ai`
- `deepseek`
- `yiyan`
- `poe`
- `z_ai` (observed in `CC-MAIN-2026-04`)

## Current Analysis Scope

The existing pipeline already gives us:

- source page URL: `source_url`
- source domain: `source_domain`
- target platform: `target_platform`
- extracted prompt text: `primary_prompt_text`
- severity and multi-label classification
- IoC and session-entry metadata

The current `SOURCE_URL` analysis scripts characterize:

- source-page level concentration
- source-domain and root-domain concentration
- Tranco popularity buckets for abused root domains
- path and template patterns of abused source pages
- multi-platform reuse of the same source page
- the relationship between `source_url` structure and prompt labels

What is still incomplete is the longitudinal side of the analysis. That part will become stronger after `CC-MAIN-2026-08` and `CC-MAIN-2026-12` are collected and processed through the same pipeline.

## Planned Next Measurements

The current plan is to extend this longitudinal dataset with:

- `CC-MAIN-2026-08`
- `CC-MAIN-2026-12`

Once those snapshots are complete, the dataset should support stronger temporal analysis across crawls instead of only per-crawl characterization.