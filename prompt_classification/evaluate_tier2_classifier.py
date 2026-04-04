"""
Evaluate a trained Tier 2 classifier against test set and/or gold standard.

Loads the saved model + thresholds from ``train_tier2_classifier.py`` output
and computes:
    - Per-label precision, recall, F1
    - Micro-F1, Macro-F1
    - Severity accuracy
    - Confusion details per label
    - Optional: comparison against Tier 1 (keyword) labels

Output:
    - evaluation_report.json   (all metrics)
    - Printed summary to stderr

Example:
    python evaluate_tier2_classifier.py \\
        --model-dir   models/tier2 \\
        --test-file   data/test.jsonl \\
        --output-dir  eval_results/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_LABELS: tuple[str, ...] = (
    "PERSIST",
    "AUTHORITY",
    "RECOMMEND",
    "CITE",
    "SUMMARIZE",
)
NUM_LABELS = len(CLASS_LABELS)
LABEL_TO_INDEX = {label: i for i, label in enumerate(CLASS_LABELS)}

SUSPICIOUS_LABELS = frozenset({"PERSIST", "AUTHORITY", "RECOMMEND", "CITE"})


def _print(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EvalDataset(Dataset):
    """JSONL dataset for evaluation."""

    def __init__(self, path: Path, tokenizer: Any, max_length: int = 256):
        self.samples: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    self.samples.append(json.loads(stripped))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples[idx]
        text = row.get("text", "")
        if not isinstance(text, str):
            text = ""

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # True labels.
        labels_field = "resolved_labels" if "resolved_labels" in row else "labels"
        labels_list = row.get(labels_field, [])
        label_vec = torch.zeros(NUM_LABELS, dtype=torch.float32)
        if isinstance(labels_list, list):
            for label in labels_list:
                if isinstance(label, str):
                    idx_l = LABEL_TO_INDEX.get(label.upper())
                    if idx_l is not None:
                        label_vec[idx_l] = 1.0

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label_vec,
        }


# ---------------------------------------------------------------------------
# Severity computation
# ---------------------------------------------------------------------------

def compute_severity(labels: list[str]) -> str:
    """Compute severity from a label set (mirrors Tier 1 logic)."""
    label_set = set(labels)
    if "PERSIST" in label_set and label_set & {"AUTHORITY", "RECOMMEND", "CITE"}:
        return "high"
    if label_set & SUSPICIOUS_LABELS:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_dir: Path,
    test_file: Path,
    output_dir: Path,
    *,
    batch_size: int = 64,
    max_length: int = 256,
) -> dict[str, Any]:
    """Run full evaluation and return metrics dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print(f"Device: {device}")

    # ---- Load model + thresholds ----
    _print("Loading model and thresholds...")
    model_path = model_dir / "model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )
    model.to(device)
    model.eval()

    thresholds_path = model_dir / "thresholds.json"
    with thresholds_path.open("r", encoding="utf-8") as f:
        thresholds_data = json.load(f)
    thresholds = thresholds_data["thresholds"]
    threshold_vec = np.array([thresholds[label] for label in CLASS_LABELS])

    _print(f"Thresholds: {thresholds}")

    # ---- Load test data ----
    _print(f"Loading test data: {test_file}")
    dataset = EvalDataset(test_file, tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    _print(f"  Test samples: {len(dataset):,}")

    # ---- Inference ----
    _print("Running inference...")
    all_probs: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            all_probs.append(probs)
            all_true.append(batch["labels"].numpy())

    probs_np = np.concatenate(all_probs, axis=0)
    true_np = np.concatenate(all_true, axis=0)
    pred_np = (probs_np >= threshold_vec).astype(int)

    # ---- Per-label metrics ----
    per_label: dict[str, dict[str, Any]] = {}
    for i, label in enumerate(CLASS_LABELS):
        y_true = true_np[:, i]
        y_pred = pred_np[:, i]
        p = precision_score(y_true, y_pred, zero_division=0.0)
        r = recall_score(y_true, y_pred, zero_division=0.0)
        f1 = f1_score(y_true, y_pred, zero_division=0.0)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        per_label[label] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
            "threshold": thresholds[label],
            "support": int(y_true.sum()),
            "predicted": int(y_pred.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    # ---- Aggregate metrics ----
    macro_f1 = f1_score(true_np, pred_np, average="macro", zero_division=0.0)
    micro_f1 = f1_score(true_np, pred_np, average="micro", zero_division=0.0)

    # ---- Severity accuracy ----
    severity_correct = 0
    severity_total = len(dataset)
    severity_confusion: Counter[str] = Counter()

    for i in range(len(dataset)):
        true_labels = [CLASS_LABELS[j] for j in range(NUM_LABELS) if true_np[i, j] == 1]
        pred_labels = [CLASS_LABELS[j] for j in range(NUM_LABELS) if pred_np[i, j] == 1]
        true_sev = compute_severity(true_labels)
        pred_sev = compute_severity(pred_labels)
        if true_sev == pred_sev:
            severity_correct += 1
        severity_confusion[f"{true_sev}→{pred_sev}"] += 1

    severity_accuracy = severity_correct / severity_total if severity_total > 0 else 0.0

    # ---- Assemble report ----
    report = {
        "test_file": str(test_file),
        "model_dir": str(model_dir),
        "num_samples": len(dataset),
        "class_labels": list(CLASS_LABELS),
        "per_label_metrics": per_label,
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "severity_accuracy": round(float(severity_accuracy), 4),
        "severity_confusion": dict(
            sorted(severity_confusion.items(), key=lambda kv: -kv[1])
        ),
        "thresholds_used": thresholds,
    }

    # ---- Print summary ----
    _print("\n" + "=" * 60)
    _print("EVALUATION RESULTS")
    _print("=" * 60)
    _print(f"Samples:          {len(dataset):,}")
    _print(f"Macro-F1:         {macro_f1:.4f}")
    _print(f"Micro-F1:         {micro_f1:.4f}")
    _print(f"Severity accuracy: {severity_accuracy:.4f}")
    _print("")
    _print(f"{'Label':12s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Supp':>5s}  {'Pred':>5s}  {'Thresh':>6s}")
    _print("-" * 60)
    for label in CLASS_LABELS:
        m = per_label[label]
        _print(
            f"{label:12s}  {m['precision']:6.4f}  {m['recall']:6.4f}  {m['f1']:6.4f}  "
            f"{m['support']:5d}  {m['predicted']:5d}  {m['threshold']:6.3f}"
        )
    _print("=" * 60)

    # ---- Save ----
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _print(f"\nReport saved: {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Tier 2 classifier on a test set.",
    )
    parser.add_argument("--model-dir", required=True, help="Directory with model/ and thresholds.json.")
    parser.add_argument("--test-file", required=True, help="Test JSONL (test.jsonl or gold_standard_sample.jsonl).")
    parser.add_argument("--output-dir", required=True, help="Output directory for evaluation report.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length (default: 256).")
    args = parser.parse_args()

    report = evaluate(
        model_dir=Path(args.model_dir),
        test_file=Path(args.test_file),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
