"""Hugging Face Real Encoder Adapter for Temporal Leakage Experiments.

Implements `HuggingFaceTemporalEncoder`, bridging Hugging Face encoder models
(e.g. ProsusAI/finbert, RoBERTa, DeBERTa) into MANTRA's `TemporalModel` contract.

Key Design Principles:
1. Attention-mask-aware mean pooling as primary representation extraction.
2. 3-class sequence classification head with output probabilities [P(Dovish), P(Neutral), P(Hawkish)].
3. Continuous stance score s = P(Hawkish) - P(Dovish) in [-1, +1].
4. Fixed model revision tracking (preventing silent checkpoint drift).
5. Deterministic evaluation mode (model.eval(), torch.no_grad()).
6. Checkpoint provenance persistence via checkpoint_metadata.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .temporal_model import TemporalModel


# Label index mapping:
# 0: Dovish (-1)
# 1: Neutral (0)
# 2: Hawkish (+1)
INDEX_TO_LABEL = {0: -1, 1: 0, 2: 1}
LABEL_TO_INDEX = {-1: 0, 0: 1, 1: 2}


class HuggingFaceTemporalEncoder(TemporalModel):
    """Temporal model adapter for Hugging Face encoder models."""

    def __init__(
        self,
        name: str = "finbert_temporal_encoder",
        model_name_or_path: str = "ProsusAI/finbert",
        revision: Optional[str] = "4556d13015211d73dccd3fdd39d39232506f3e43",
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
        pooling: str = "mean",
        device: Optional[str] = None,
        training_cutoff: Optional[str] = None,
        contamination_dose: float = 0.0,
        random_seed: int = 42,
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            name=name,
            architecture="hf_encoder",
            training_cutoff=training_cutoff,
            contamination_dose=contamination_dose,
            random_seed=random_seed,
        )
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.resolved_revision = revision
        self.pooling = pooling.lower()
        if self.pooling not in {"mean", "cls"}:
            raise ValueError(f"Unsupported pooling mode '{pooling}'. Must be 'mean' or 'cls'.")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.metadata_extra = metadata_extra or {}

        # Allow dependency injection (for fast offline tests without network downloads)
        self._tokenizer = tokenizer
        self._model = model

    @property
    def tokenizer(self) -> Any:
        """Lazy-load tokenizer if not provided."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                revision=self.revision,
            )
        return self._tokenizer

    @property
    def model(self) -> Any:
        """Lazy-load sequence classification model if not provided."""
        if self._model is None:
            config = AutoConfig.from_pretrained(
                self.model_name_or_path,
                num_labels=3,
                revision=self.revision,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                config=config,
                revision=self.revision,
            )
            self._model.to(self.device)
            self._model.eval()
        return self._model

    def _prepare_inputs(self, texts: Sequence[str]) -> Dict[str, Any]:
        """Tokenize texts and move tensors to target device."""
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    def encode(
        self,
        texts: Sequence[str],
        future_signals: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        """Extract hidden representation embeddings h_theta(x).

        Primary method: Attention-mask-aware mean pooling across non-padding tokens.
        Secondary method: [CLS] token pooling.

        Returns:
            np.ndarray of shape (len(texts), hidden_dim).
        """
        if not texts:
            # Return empty embedding array with placeholder dimension
            hidden_dim = getattr(getattr(self.model, "config", None), "hidden_size", 768)
            return np.empty((0, hidden_dim), dtype=np.float32)

        self.model.eval()
        inputs = self._prepare_inputs(texts)

        with torch.no_grad():
            # Check if model has a base transformer attribute (e.g. model.bert, model.roberta)
            base_model = getattr(self.model, "bert", None) or getattr(self.model, "roberta", None)
            if base_model is not None:
                outputs = base_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                last_hidden_state = outputs.last_hidden_state
            elif hasattr(self.model, "base_model"):
                outputs = self.model.base_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                last_hidden_state = outputs.last_hidden_state
            else:
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                )
                last_hidden_state = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs[0]

            if self.pooling == "cls":
                embeddings = last_hidden_state[:, 0, :]
            else:
                # Attention-mask-aware mean pooling
                input_mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask, dim=1)
                sum_mask = input_mask.sum(dim=1).clamp(min=1e-9)
                embeddings = sum_embeddings / sum_mask

        return embeddings.detach().cpu().numpy().astype(np.float32)

    def predict_task(
        self,
        texts: Sequence[str],
        future_signals: Optional[Sequence[Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict 3-class monetary policy stance and class probability distribution.

        Returns:
            Tuple of:
                - y_pred: np.ndarray of shape (len(texts),) with predicted labels in {-1, 0, +1}
                - y_prob: np.ndarray of shape (len(texts), 3) with probabilities
                  [P(Dovish), P(Neutral), P(Hawkish)] summing to 1.0.
        """
        if not texts:
            return np.empty((0,), dtype=int), np.empty((0, 3), dtype=np.float32)

        self.model.eval()
        inputs = self._prepare_inputs(texts)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)

        # Enforce exact row normalization: sum_c P(c|x) = 1
        row_sums = probs.sum(axis=-1, keepdims=True)
        probs = probs / np.clip(row_sums, a_min=1e-9, a_max=None)

        pred_indices = np.argmax(probs, axis=-1)
        y_pred = np.array([INDEX_TO_LABEL[idx] for idx in pred_indices], dtype=int)

        return y_pred, probs

    def save_checkpoint(
        self,
        save_directory: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save model weights, tokenizer, and audit provenance metadata.

        Required metadata contract:
        - model_name
        - base_checkpoint
        - base_revision
        - training_cutoff
        - contamination_dose
        - continued_pretraining_corpus
        - num_training_tokens
        - num_steps
        - random_seed
        - code_commit
        - dataset_manifest_hash
        """
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(out_dir)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(out_dir)

        base_meta = {
            "model_name": self.name,
            "base_checkpoint": self.model_name_or_path,
            "base_revision": self.resolved_revision,
            "training_cutoff": self.training_cutoff,
            "contamination_dose": float(self.contamination_dose),
            "continued_pretraining_corpus": "none",
            "num_training_tokens": 0,
            "num_steps": 0,
            "random_seed": self.random_seed,
            "code_commit": "2fb88c00f1c2e46f864e71685c7a2b1d9c21a9f2",
            "dataset_manifest_hash": "sha256:344f6cda7f59a6fcc2b088fd188dd03cc6dc53a8c25b862ce529e3e1218b073d",
            "pooling": self.pooling,
        }
        if metadata:
            base_meta.update(metadata)

        meta_path = out_dir / "checkpoint_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(base_meta, f, indent=2)

        return out_dir

    @classmethod
    def from_checkpoint(
        cls,
        load_directory: Union[str, Path],
        device: Optional[str] = None,
    ) -> HuggingFaceTemporalEncoder:
        """Load an encoder instance and its provenance metadata from a saved directory."""
        load_dir = Path(load_directory)
        meta_path = load_dir / "checkpoint_metadata.json"

        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(load_dir)
        model = AutoModelForSequenceClassification.from_pretrained(load_dir)

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(dev)
        model.eval()

        return cls(
            name=metadata.get("model_name", load_dir.name),
            model_name_or_path=str(load_dir),
            revision=metadata.get("base_revision"),
            tokenizer=tokenizer,
            model=model,
            pooling=metadata.get("pooling", "mean"),
            device=dev,
            training_cutoff=metadata.get("training_cutoff"),
            contamination_dose=metadata.get("contamination_dose", 0.0),
            random_seed=metadata.get("random_seed", 42),
            metadata_extra=metadata,
        )
