"""
Active semantic prompt classifier for Stage 02 prompt-link classification.

Loads the promoted model checkpoint from the active ``models/`` directory and
provides batch inference for the 5-label prompt intent task.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


CLASS_LABELS: tuple[str, ...] = (
    "PERSIST",
    "AUTHORITY",
    "RECOMMEND",
    "CITE",
    "SUMMARIZE",
)
NUM_LABELS = len(CLASS_LABELS)

SUSPICIOUS_LABELS = frozenset({"PERSIST", "AUTHORITY", "RECOMMEND", "CITE"})
CONFIDENT_NEGATIVE_THRESHOLD = 0.15


def compute_severity(labels: list[str]) -> str:
    label_set = set(labels)
    if "PERSIST" in label_set and label_set & {"AUTHORITY", "RECOMMEND", "CITE"}:
        return "high"
    if label_set & SUSPICIOUS_LABELS:
        return "medium"
    return "low"


class SemanticPromptClassifier:
    """Lazy-loading semantic classifier for prompt intent inference."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        max_length: int = 256,
        device: str = "cpu",
    ):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self._device_str = device.strip() or "cpu"
        self._model: Any = None
        self._tokenizer: Any = None
        self._thresholds: dict[str, float] | None = None
        self._threshold_vec: np.ndarray | None = None
        self._device: torch.device | None = None

    @property
    def model_name(self) -> str:
        return self.model_dir.name

    @property
    def device_name(self) -> str:
        self._ensure_loaded()
        assert self._device is not None
        return str(self._device)

    @property
    def thresholds(self) -> dict[str, float]:
        self._ensure_loaded()
        assert self._thresholds is not None
        return self._thresholds

    def _resolve_device(self) -> torch.device:
        if self._device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        device = torch.device(self._device_str)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this host.")
        return device

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        model_path = self.model_dir / "model"
        thresholds_path = self.model_dir / "thresholds.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        if not thresholds_path.exists():
            raise FileNotFoundError(f"Thresholds file not found: {thresholds_path}")

        self._device = self._resolve_device()
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
        )
        self._model.to(self._device)
        self._model.eval()

        with thresholds_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        self._thresholds = data["thresholds"]
        self._threshold_vec = np.array(
            [self._thresholds[label] for label in CLASS_LABELS],
            dtype=np.float32,
        )

    def classify_batch(
        self,
        texts: list[str],
        *,
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._threshold_vec is not None
        assert self._device is not None

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        results: list[dict[str, Any]] = []
        empty_probabilities = {label: 0.0 for label in CLASS_LABELS}

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]

            non_empty_indices: list[int] = []
            non_empty_texts: list[str] = []
            batch_results: list[dict[str, Any]] = []

            for _ in batch_texts:
                batch_results.append(
                    {
                        "labels": [],
                        "probabilities": dict(empty_probabilities),
                        "is_uncertain": False,
                        "severity": "low",
                    }
                )

            for index, text in enumerate(batch_texts):
                if isinstance(text, str) and text.strip():
                    non_empty_indices.append(index)
                    non_empty_texts.append(text)

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

                for non_empty_index, original_index in enumerate(non_empty_indices):
                    row_probs = probs[non_empty_index]
                    row_labels: list[str] = []
                    is_uncertain = False

                    for label_index, label in enumerate(CLASS_LABELS):
                        if row_probs[label_index] >= self._threshold_vec[label_index]:
                            row_labels.append(label)
                        elif row_probs[label_index] >= CONFIDENT_NEGATIVE_THRESHOLD:
                            is_uncertain = True

                    batch_results[original_index] = {
                        "labels": row_labels,
                        "probabilities": {
                            label: round(float(row_probs[label_index]), 4)
                            for label_index, label in enumerate(CLASS_LABELS)
                        },
                        "is_uncertain": is_uncertain,
                        "severity": compute_severity(row_labels),
                    }

            results.extend(batch_results)

        return results

    def classify_single(self, text: str) -> tuple[list[str], dict[str, float]]:
        result = self.classify_batch([text], batch_size=1)[0]
        return result["labels"], result["probabilities"]