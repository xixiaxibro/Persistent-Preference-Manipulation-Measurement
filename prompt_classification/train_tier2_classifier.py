"""
Train a Tier 2 multi-label classifier for prompt classification.

Fine-tunes ``xlm-roberta-base`` (or a specified model) with a 5-output
sigmoid head using binary cross-entropy loss.  Reads train/val splits
produced by ``assemble_training_dataset.py``.

After training, tunes per-label decision thresholds on the validation set
to maximize per-label F1.  Saves:
    - model weights + tokenizer (HuggingFace format)
    - ``thresholds.json``   (per-label optimal thresholds)
    - ``training_log.json`` (loss curves, validation metrics per epoch)

Execution environment
---------------------
Requires a CUDA GPU for practical training speed.  CPU training works but
will be very slow.  See ``requirements-tier2.txt`` for dependencies:
    torch, transformers, scikit-learn, numpy

Example:
    python train_tier2_classifier.py \\
        --train-file  data/train.jsonl \\
        --val-file    data/val.jsonl \\
        --output-dir  models/tier2 \\
        --epochs 5 \\
        --batch-size 32 \\
        --lr 2e-5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
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


def _print(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PromptDataset(Dataset):
    """
    JSONL-backed dataset for multi-label prompt classification.

    Each row must have:
        - ``text``: str — the prompt text
        - ``labels``: list[str] — list of label names (may be empty)
    """

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

        # Build multi-hot label vector.
        labels_list = row.get("labels", [])
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
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_thresholds(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Find per-label optimal thresholds by sweeping 0.1–0.9 on val set.

    Returns:
        thresholds:   {label_name: optimal_threshold}
        metrics:      {label_name: {precision, recall, f1, threshold}}
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.append(probs)
            all_labels.append(labels.numpy())

    all_preds_np = np.concatenate(all_preds, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)

    thresholds: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = {}
    candidates = np.arange(0.1, 0.95, 0.05)

    for i, label_name in enumerate(CLASS_LABELS):
        y_true = all_labels_np[:, i]
        y_probs = all_preds_np[:, i]

        best_f1 = -1.0
        best_t = 0.5

        for t in candidates:
            y_pred = (y_probs >= t).astype(int)
            if y_pred.sum() == 0:
                continue
            f1 = f1_score(y_true, y_pred, zero_division=0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        # Compute final metrics at best threshold.
        y_pred_best = (y_probs >= best_t).astype(int)
        p = precision_score(y_true, y_pred_best, zero_division=0.0)
        r = recall_score(y_true, y_pred_best, zero_division=0.0)

        thresholds[label_name] = round(best_t, 3)
        metrics[label_name] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(best_f1), 4),
            "threshold": round(best_t, 3),
            "positive_count": int(y_true.sum()),
            "predicted_count": int(y_pred_best.sum()),
        }

    return thresholds, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """
    Train the model and return per-epoch log entries.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    training_log: list[dict[str, Any]] = []

    total_steps = epochs * len(train_loader)
    _print(f"Training: {epochs} epochs, {len(train_loader)} batches/epoch, {total_steps} total steps")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start = time.monotonic()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1

            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / epoch_steps
                _print(
                    f"  Epoch {epoch}/{epochs}  "
                    f"batch {batch_idx + 1}/{len(train_loader)}  "
                    f"loss: {avg_loss:.4f}"
                )

        avg_train_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
        epoch_elapsed = time.monotonic() - epoch_start

        # Validation.
        model.eval()
        val_loss = 0.0
        val_steps = 0
        all_val_preds: list[np.ndarray] = []
        all_val_labels: list[np.ndarray] = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                val_loss += loss.item()
                val_steps += 1

                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                all_val_preds.append(probs)
                all_val_labels.append(batch["labels"].numpy())

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0

        # Quick macro-F1 at 0.5 threshold for logging.
        val_preds_np = np.concatenate(all_val_preds, axis=0)
        val_labels_np = np.concatenate(all_val_labels, axis=0)
        val_binary = (val_preds_np >= 0.5).astype(int)
        macro_f1 = f1_score(val_labels_np, val_binary, average="macro", zero_division=0.0)
        micro_f1 = f1_score(val_labels_np, val_binary, average="micro", zero_division=0.0)

        log_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(avg_val_loss, 6),
            "val_macro_f1_at_0.5": round(float(macro_f1), 4),
            "val_micro_f1_at_0.5": round(float(micro_f1), 4),
            "epoch_seconds": round(epoch_elapsed, 1),
        }
        training_log.append(log_entry)

        _print(
            f"\nEpoch {epoch}/{epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"val_macro_f1={macro_f1:.4f}  "
            f"val_micro_f1={micro_f1:.4f}  "
            f"({epoch_elapsed:.1f}s)"
        )

    return training_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Tier 2 multi-label classifier (xlm-roberta-base) "
            "for prompt classification. Produces model weights, per-label "
            "thresholds, and a training log."
        ),
    )
    parser.add_argument("--train-file", required=True, help="train.jsonl from assemble_training_dataset.py.")
    parser.add_argument("--val-file", required=True, help="val.jsonl from assemble_training_dataset.py.")
    parser.add_argument("--output-dir", required=True, help="Output directory for model, thresholds, log.")
    parser.add_argument("--model-name", default="xlm-roberta-base", help="HuggingFace model name (default: xlm-roberta-base).")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5).")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length (default: 256).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print(f"Device: {device}")
    _print(f"Model:  {args.model_name}")
    _print(f"Labels: {CLASS_LABELS}")
    _print("")

    # ---- Load tokenizer and model ----
    _print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )
    model.to(device)

    # ---- Build datasets ----
    _print("Loading datasets...")
    train_dataset = PromptDataset(Path(args.train_file), tokenizer, max_length=args.max_length)
    val_dataset = PromptDataset(Path(args.val_file), tokenizer, max_length=args.max_length)

    _print(f"  Train samples: {len(train_dataset):,}")
    _print(f"  Val samples:   {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ---- Train ----
    _print("\nStarting training...")
    training_log = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=output_dir,
    )

    # ---- Tune thresholds ----
    _print("\nTuning per-label thresholds on validation set...")
    thresholds, threshold_metrics = tune_thresholds(model, val_loader, device)

    _print("\nPer-label thresholds and metrics:")
    for label in CLASS_LABELS:
        m = threshold_metrics[label]
        _print(
            f"  {label:12s}  threshold={m['threshold']:.3f}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  "
            f"(pos={m['positive_count']}, pred={m['predicted_count']})"
        )

    # ---- Save model ----
    _print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "model")

    # ---- Save thresholds ----
    thresholds_path = output_dir / "thresholds.json"
    with thresholds_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "thresholds": thresholds,
                "per_label_metrics": threshold_metrics,
                "class_labels": list(CLASS_LABELS),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    _print(f"Thresholds saved: {thresholds_path}")

    # ---- Save training log ----
    log_path = output_dir / "training_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "max_length": args.max_length,
                "seed": args.seed,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "device": str(device),
                "class_labels": list(CLASS_LABELS),
                "training_log": training_log,
                "final_thresholds": thresholds,
                "final_per_label_metrics": threshold_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    _print(f"Training log saved: {log_path}")

    _print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
