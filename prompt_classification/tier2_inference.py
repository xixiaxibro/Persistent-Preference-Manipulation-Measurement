"""
Tier 2 inference module for integration into the classification pipeline.

Loads a trained Tier 2 model (from ``train_tier2_classifier.py`` output)
and provides a batch-inference interface that ``classify_prompt_links.py``
can call as an optional second pass.

The module manages its own model lifecycle (lazy loading, GPU/CPU detection)
and exposes two interfaces:

    1. ``Tier2Classifier`` class — instantiate with model directory path,
       call ``classify_batch(texts)`` for batch inference.
    2. ``classify_single(text)`` convenience method on the instance.

Usage from ``classify_prompt_links.py``:

    from prompt_classification.tier2_inference import Tier2Classifier

    tier2 = Tier2Classifier("models/tier2")
    labels, probs = tier2.classify_single("recommend example.com")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
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

SUSPICIOUS_LABELS = frozenset({"PERSIST", "AUTHORITY", "RECOMMEND", "CITE"})

# Default confidence thresholds for the uncertain zone.
# Rows with all probabilities below CONFIDENT_NEGATIVE_THRESHOLD are
# confidently negative (no labels).  Rows with any probability in the
# range (CONFIDENT_NEGATIVE_THRESHOLD, per-label threshold) are in the
# uncertain zone and should be routed to Tier 3.
CONFIDENT_NEGATIVE_THRESHOLD = 0.15


class Tier2Classifier:
    """
    Lazy-loading Tier 2 classifier for prompt-link classification.

    Loads model and thresholds from a directory produced by
    ``train_tier2_classifier.py``.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        max_length: int = 256,
        device: str | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self._device_str = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._thresholds: dict[str, float] | None = None
        self._threshold_vec: np.ndarray | None = None
        self._device: torch.device | None = None

    def _ensure_loaded(self) -> None:
        """Lazy-load model, tokenizer, and thresholds on first use."""
        if self._model is not None:
            return

        model_path = self.model_dir / "model"
        thresholds_path = self.model_dir / "thresholds.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        if not thresholds_path.exists():
            raise FileNotFoundError(f"Thresholds file not found: {thresholds_path}")

        # Device.
        if self._device_str:
            self._device = torch.device(self._device_str)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer.
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Model.
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
        )
        self._model.to(self._device)
        self._model.eval()

        # Thresholds.
        with thresholds_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self._thresholds = data["thresholds"]
        self._threshold_vec = np.array(
            [self._thresholds[label] for label in CLASS_LABELS]
        )

    @property
    def thresholds(self) -> dict[str, float]:
        self._ensure_loaded()
        assert self._thresholds is not None
        return self._thresholds

    def classify_batch(
        self,
        texts: list[str],
        *,
        batch_size: int = 64,
    ) -> list[dict[str, Any]]:
        """
        Classify a batch of texts.

        Returns a list of dicts, one per input text:
            {
                "labels": ["PERSIST", "RECOMMEND"],  # assigned labels
                "probabilities": {"PERSIST": 0.87, ...},  # all 5 probs
                "is_uncertain": False,  # True if should route to Tier 3
                "severity": "high",
            }
        """
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._threshold_vec is not None
        assert self._device is not None

        results: list[dict[str, Any]] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]

            # Handle empty texts.
            non_empty_indices = []
            non_empty_texts = []
            for i, text in enumerate(batch_texts):
                if isinstance(text, str) and text.strip():
                    non_empty_indices.append(i)
                    non_empty_texts.append(text)

            # Pre-fill all as empty.
            batch_results: list[dict[str, Any]] = []
            for _ in batch_texts:
                batch_results.append({
                    "labels": [],
                    "probabilities": {label: 0.0 for label in CLASS_LABELS},
                    "is_uncertain": False,
                    "severity": "low",
                })

            if non_empty_texts:
                encoding = self._tokenizer(
                    non_empty_texts,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = encoding["input_ids"].to(self._device)
                attention_mask = encoding["attention_mask"].to(self._device)

                with torch.no_grad():
                    outputs = self._model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    probs = torch.sigmoid(outputs.logits).cpu().numpy()

                for j, orig_idx in enumerate(non_empty_indices):
                    row_probs = probs[j]
                    row_labels = []
                    is_uncertain = False

                    for k, label in enumerate(CLASS_LABELS):
                        if row_probs[k] >= self._threshold_vec[k]:
                            row_labels.append(label)
                        elif row_probs[k] >= CONFIDENT_NEGATIVE_THRESHOLD:
                            # In the uncertain zone.
                            is_uncertain = True

                    severity = _compute_severity(row_labels)

                    batch_results[orig_idx] = {
                        "labels": row_labels,
                        "probabilities": {
                            label: round(float(row_probs[k]), 4)
                            for k, label in enumerate(CLASS_LABELS)
                        },
                        "is_uncertain": is_uncertain,
                        "severity": severity,
                    }

            results.extend(batch_results)

        return results

    def classify_single(self, text: str) -> tuple[list[str], dict[str, float]]:
        """
        Classify a single text.

        Returns:
            labels: list of assigned label names
            probabilities: dict of {label: probability} for all 5 labels
        """
        result = self.classify_batch([text])[0]
        return result["labels"], result["probabilities"]


def _compute_severity(labels: list[str]) -> str:
    """Compute severity from a label set (mirrors Tier 1 logic)."""
    label_set = set(labels)
    if "PERSIST" in label_set and label_set & {"AUTHORITY", "RECOMMEND", "CITE"}:
        return "high"
    if label_set & SUSPICIOUS_LABELS:
        return "medium"
    return "low"
