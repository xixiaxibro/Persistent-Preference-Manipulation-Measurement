# Phase 2 Implementation: Tier 2 Classifier

## Overview

Phase 2 implements the **Tier 2 fine-tuned classifier** described in Section 2
of `deep-research-report.md`. This adds a semantic classification layer on top
of the existing Tier 1 keyword+regex rules, catching manipulation intent that
is expressed through paraphrasing, synonyms, or non-English text.

### What was implemented

| Component | File | Purpose |
|---|---|---|
| Gold standard sampling | `build_gold_standard_sample.py` | Draw stratified sample for human annotation |
| Model training | `train_tier2_classifier.py` | Fine-tune xlm-roberta-base with 5-label BCE |
| Model evaluation | `evaluate_tier2_classifier.py` | Per-label + aggregate metrics on test/gold set |
| Inference module | `tier2_inference.py` | Lazy-loading classifier for pipeline integration |
| Pipeline integration | `classify_prompt_links.py` (updated) | Optional `--tier2-model-dir` flag for Tier 2 |
| Dependencies | `requirements-tier2.txt` | torch, transformers, scikit-learn, numpy |

---

## Execution environment requirements

### Hardware

- **Training**: CUDA-capable GPU with ≥8 GB VRAM (e.g., NVIDIA T4, RTX 3060).
  Training `xlm-roberta-base` (270M params) with batch size 32 requires ~6 GB.
  CPU training is functional but will take hours instead of minutes.
- **Inference**: GPU recommended for batch inference on 2M rows. CPU works for
  smaller batches or single-row classification.
- **Disk**: ~1.5 GB for the xlm-roberta-base model checkpoint + tokenizer.

### Software

Base environment (already in `requirements.txt`):
```
openai>=1.0.0,<2.0.0
python-dotenv>=1.0.0,<2.0.0
tldextract>=5.1.2,<6.0.0
warcio>=1.7.4,<2.0.0
```

Additional Tier 2 dependencies (`prompt_classification/requirements-tier2.txt`):
```
torch>=2.0.0,<3.0.0
transformers>=4.36.0,<5.0.0
scikit-learn>=1.3.0,<2.0.0
numpy>=1.24.0,<3.0.0
```

Install everything:
```bash
pip install -r requirements.txt -r prompt_classification/requirements-tier2.txt
```

### Python version

- Python 3.10+ required (for `match` statements and type union syntax `X | Y`).

---

## Pipeline execution order

Phase 2 builds on the existing pipeline. The full sequence is:

### Step 0: Tier 1 classification (existing)

Already implemented. Produces enriched+classified JSONL.

```bash
# Run with --include-benign to keep all rows (needed for dataset construction).
python classify_prompt_links.py \
    --input  data/platform_filtered.jsonl.gz \
    --output data/classified_all.jsonl.gz \
    --include-benign
```

### Step 1: Build classification dataset

Already implemented. Produces weakly-labeled pool + unlabeled candidates.

```bash
python build_classification_dataset.py \
    --input     data/classified_all.jsonl.gz \
    --output-dir data/dataset/ \
    --weak-labeled-per-label 2000 \
    --unlabeled-total 5000
```

### Step 2: LLM relabeling

Already implemented. Labels unlabeled candidates via LLM.

```bash
python relabel_negatives_with_llm.py \
    --input     data/dataset/unlabeled_for_relabel.jsonl \
    --output-dir data/dataset/
```

### Step 3: Assemble training dataset

Already implemented. Merges pools and splits into train/val/test.

```bash
python assemble_training_dataset.py \
    --labeled  data/dataset/weak_labeled.jsonl data/dataset/llm_labeled.jsonl \
    --all-zero data/dataset/llm_all_zero.jsonl \
    --output-dir data/dataset/final/
```

### Step 4: Build gold standard sample (NEW)

Draws a stratified sample for human annotation.

```bash
python prompt_classification/build_gold_standard_sample.py \
    --input     data/classified_all.jsonl.gz \
    --output-dir data/gold_standard/ \
    --per-label 200 \
    --unlabeled-count 200
```

Output: `gold_standard_sample.jsonl` with ~1200 rows, each containing
placeholder fields for dual-annotator labeling:
- `annotator_1_labels`: [] (to be filled by annotator 1)
- `annotator_2_labels`: [] (to be filled by annotator 2)
- `resolved_labels`: [] (to be filled after disagreement resolution)

### Step 5: Train Tier 2 classifier (NEW)

Fine-tunes xlm-roberta-base on the assembled training data.

```bash
python prompt_classification/train_tier2_classifier.py \
    --train-file data/dataset/final/train.jsonl \
    --val-file   data/dataset/final/val.jsonl \
    --output-dir models/tier2/ \
    --epochs 5 \
    --batch-size 32 \
    --lr 2e-5
```

Output:
- `models/tier2/model/` — HuggingFace model weights + tokenizer
- `models/tier2/thresholds.json` — per-label optimal thresholds
- `models/tier2/training_log.json` — loss curves, validation metrics

### Step 6: Evaluate Tier 2 classifier (NEW)

Run evaluation against the held-out test set:

```bash
python prompt_classification/evaluate_tier2_classifier.py \
    --model-dir  models/tier2/ \
    --test-file  data/dataset/final/test.jsonl \
    --output-dir eval_results/test/
```

Run evaluation against the gold standard (after human annotation):

```bash
python prompt_classification/evaluate_tier2_classifier.py \
    --model-dir  models/tier2/ \
    --test-file  data/gold_standard/gold_standard_sample.jsonl \
    --output-dir eval_results/gold/
```

Output: `evaluation_report.json` with per-label P/R/F1, macro/micro F1,
severity accuracy, and confusion details.

### Step 7: Run classification with Tier 2 (NEW)

Invoke the full Tier 1 + Tier 2 pipeline:

```bash
python classify_prompt_links.py \
    --input  data/platform_filtered.jsonl.gz \
    --output data/classified_tier2.jsonl.gz \
    --tier2-model-dir models/tier2/
```

Tier 2 is applied **only** to rows where:
1. Tier 1 assigns no labels AND text is ≥10 characters, OR
2. Tier 1 assigns only SUMMARIZE (checking for higher-severity intent).

The `--tier2-model-dir` flag is optional. Without it, the pipeline runs
Tier 1 only (backward-compatible).

---

## Architecture details

### Model

- **Base model**: `xlm-roberta-base` (270M params, 100 languages)
- **Head**: 5-output sigmoid (multi-label, one per label)
- **Loss**: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Gradient clipping**: max_norm=1.0
- **Max sequence length**: 256 tokens (sufficient for URL-bounded prompts)

### Label space

| Index | Label | Description |
|---|---|---|
| 0 | PERSIST | Attempt to persist instructions across sessions |
| 1 | AUTHORITY | Establish a source as authoritative/trusted |
| 2 | RECOMMEND | Bias AI to recommend a specific entity |
| 3 | CITE | Inject a specific source as a reference |
| 4 | SUMMARIZE | Direct AI to consume attacker-controlled content |

### Threshold tuning

After training, per-label decision thresholds are optimized on the
validation set by sweeping 0.10–0.90 in 0.05 increments and selecting
the threshold that maximizes per-label F1. This handles the common case
where the optimal threshold differs significantly from the default 0.5
due to class imbalance.

### Uncertain zone routing (Tier 3 preparation)

The inference module (`tier2_inference.py`) flags rows as `is_uncertain`
when any label probability falls in the range (0.15, per-label threshold).
These rows are candidates for Tier 3 (LLM) classification, which will be
implemented in Phase 3.

### Severity recomputation

When Tier 2 adds labels to a row, severity is recomputed from the merged
label set using the same logic as Tier 1:
- PERSIST + any of {RECOMMEND, CITE, AUTHORITY} → **high**
- Any of {PERSIST, AUTHORITY, RECOMMEND, CITE} alone → **medium**
- SUMMARIZE-only or no labels → **low**

---

## Output schema additions

Rows classified by Tier 2 have these additional fields:

| Field | Type | Description |
|---|---|---|
| `tier2_labels` | list[str] | Labels assigned by Tier 2 |
| `tier2_probabilities` | dict[str, float] | Sigmoid probabilities for all 5 labels |
| `classification_tier` | str | `"tier1"` or `"tier2"` indicating which tier produced the final labels |

Rows that are not routed to Tier 2 have `classification_tier: "tier1"`.

---

## Verification checklist

After running the pipeline, verify:

1. **Dataset construction**: `data/dataset/final/` contains `train.jsonl`, `val.jsonl`, `test.jsonl` with expected label distributions.

2. **Training convergence**: `training_log.json` shows decreasing train/val loss across epochs and increasing macro-F1.

3. **Per-label thresholds**: `thresholds.json` contains reasonable values (typically 0.3–0.7 depending on class balance).

4. **Test set evaluation**: `evaluation_report.json` shows per-label F1 and macro-F1 at acceptable levels (target: macro-F1 > 0.70).

5. **Gold standard evaluation**: After human annotation, run evaluation on the gold standard to measure real-world performance.

6. **Pipeline integration**: Run `classify_prompt_links.py --tier2-model-dir` and verify:
   - `classification_tier` field appears in output.
   - Rows that were unlabeled by Tier 1 now have Tier 2 labels where applicable.
   - Severity is correctly recomputed from merged labels.
   - Label distribution shifts are reasonable (Tier 2 should find more positives, not fewer).

7. **Backward compatibility**: Running without `--tier2-model-dir` produces identical output to the previous version.
