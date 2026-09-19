"""Hugging Face Real Encoder Adapter for Temporal Leakage Experiments.

Implements `HuggingFaceTemporalEncoder`, bridging Hugging Face encoder models
(e.g. ProsusAI/finbert, RoBERTa, DeBERTa) into MANTRA's `TemporalModel` contract.

Key Design Principles (Phase 2.1 Hardened):
1. Attention-mask-aware mean pooling as primary representation extraction.
2. Dedicated 3-class sequence classification head for FOMC stance:
   - 0: Dovish (-1)
   - 1: Neutral (0)
   - 2: Hawkish (+1)
3. Rejection of raw financial sentiment heads:
   - ProsusAI/finbert native sentiment heads (positive/negative/neutral) are discarded.
   - Replaced with a freshly initialized, bit-identical FOMC stance head.
4. Continuous stance score s = P(Hawkish) - P(Dovish) in [-1, +1].
5. Deterministic evaluation mode (model.eval(), torch.no_grad()).
6. Cryptographic parameter and corpus hashing for causal integrity verification.
7. Dynamic provenance tracking (resolves git commit and dataset manifest dynamically).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import unicodedata

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .temporal_model import TemporalModel


# FOMC Stance Label Index Mapping
# 0: Dovish (-1)
# 1: Neutral (0)
# 2: Hawkish (+1)
FOMC_STANCE_ID_TO_LABEL: Dict[int, str] = {0: "Dovish", 1: "Neutral", 2: "Hawkish"}
FOMC_STANCE_LABEL_TO_ID: Dict[str, int] = {"Dovish": 0, "Neutral": 1, "Hawkish": 2}

INDEX_TO_LABEL: Dict[int, int] = {0: -1, 1: 0, 2: 1}
LABEL_TO_INDEX: Dict[int, int] = {-1: 0, 0: 1, 1: 2}


def resolve_git_provenance(cwd: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Resolve git provenance metadata dynamically without hardcoded fallbacks.

    Returns:
        Dict with:
            - 'git_head': commit SHA (str) or 'unknown'
            - 'git_dirty': bool (True if working tree has unstaged or uncommitted changes)
            - 'code_commit_exact': bool (True only if git_head != 'unknown' and not git_dirty)
    """
    search_dir = Path(cwd) if cwd is not None else Path(__file__).resolve().parent
    git_head = "unknown"
    git_dirty = False

    env_commit = os.environ.get("MANTRA_GIT_COMMIT")
    if env_commit:
        git_head = env_commit.strip()

    try:
        if git_head == "unknown":
            res_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=search_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res_head.returncode == 0 and res_head.stdout.strip():
                git_head = res_head.stdout.strip()

        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=search_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status_res.returncode == 0:
            git_dirty = bool(status_res.stdout.strip())
        else:
            git_dirty = True
    except Exception:
        git_dirty = True

    code_commit_exact = (git_head != "unknown" and not git_dirty)
    return {
        "git_head": git_head,
        "git_dirty": git_dirty,
        "code_commit_exact": code_commit_exact,
    }


def resolve_git_commit(cwd: Optional[Union[str, Path]] = None) -> str:
    """Resolve current git commit SHA dynamically without hardcoded fallbacks."""
    return resolve_git_provenance(cwd)["git_head"]


def hash_model_parameters(model: Any) -> str:
    """Compute deterministic SHA-256 hash over model parameters state_dict.

    Extracts all state_dict tensors, moves to CPU, and hashes contiguous bytes.
    """
    if hasattr(model, "state_dict"):
        sd = model.state_dict()
    elif isinstance(model, dict):
        sd = model
    else:
        raise TypeError(f"Cannot extract state_dict from object of type {type(model)}")

    hasher = hashlib.sha256()
    for key in sorted(sd.keys()):
        tensor = sd[key]
        hasher.update(key.encode("utf-8"))
        if hasattr(tensor, "detach"):
            t_cpu = tensor.detach().cpu().contiguous()
            hasher.update(t_cpu.numpy().tobytes())
        elif isinstance(tensor, np.ndarray):
            hasher.update(tensor.tobytes())
        elif isinstance(tensor, (int, float, str, bytes)):
            hasher.update(str(tensor).encode("utf-8"))
    return hasher.hexdigest()


def normalize_and_hash_text(text: str) -> str:
    """Normalize text (NFKC, strip, collapse whitespace) and compute SHA-256."""
    norm = unicodedata.normalize("NFKC", text).strip()
    norm = " ".join(norm.split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def hash_corpus(texts: Sequence[str]) -> str:
    """Compute deterministic SHA-256 hash over an ordered corpus of texts."""
    hasher = hashlib.sha256()
    for t in texts:
        h = normalize_and_hash_text(t)
        hasher.update(h.encode("utf-8"))
    return hasher.hexdigest()


def build_fresh_fomc_classifier_from_base_encoder(
    base_model_name_or_path: str = "ProsusAI/finbert",
    base_revision: Optional[str] = "4556d13015211d73dccd3fdd39d39232506f3e43",
    random_seed: int = 42,
    device: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    mock_base_model: Optional[Any] = None,
    name: str = "finbert_fomc_stance_classifier",
) -> HuggingFaceTemporalEncoder:
    """Construct a 3-class FOMC stance classifier structurally isolated from base sentiment head.

    Guarantees:
    1. Raw financial sentiment head (positive/negative/neutral) weights are NOT loaded.
    2. Base encoder body weights (.bert or base_model) are preserved from base model.
    3. Fresh 3-class sequence classification head (Dovish/Neutral/Hawkish) is initialized
       deterministically using `random_seed`.
    4. Records classifier head provenance:
       - original_head_loaded: False
       - stance_head_initialization: "fresh"
       - stance_head_initial_hash: SHA-256 parameter hash of fresh classification head.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if mock_base_model is not None:
        if hasattr(mock_base_model, "bert"):
            base_body_state = mock_base_model.bert.state_dict()
            cfg = getattr(mock_base_model, "config", None)
        elif hasattr(mock_base_model, "base_model"):
            base_body_state = mock_base_model.base_model.state_dict()
            cfg = getattr(mock_base_model, "config", None)
        else:
            base_body_state = mock_base_model.state_dict()
            cfg = getattr(mock_base_model, "config", None)
    else:
        base_encoder = AutoModel.from_pretrained(
            base_model_name_or_path,
            revision=base_revision,
        )
        base_body_state = base_encoder.state_dict()
        cfg = base_encoder.config

    if cfg is not None:
        clf_config = copy.deepcopy(cfg)
        clf_config.num_labels = 3
        clf_config.id2label = dict(FOMC_STANCE_ID_TO_LABEL)
        clf_config.label2id = dict(FOMC_STANCE_LABEL_TO_ID)
    else:
        clf_config = AutoConfig.from_pretrained(
            base_model_name_or_path,
            num_labels=3,
            id2label=FOMC_STANCE_ID_TO_LABEL,
            label2id=FOMC_STANCE_LABEL_TO_ID,
            revision=base_revision,
        )

    clf_model = AutoModelForSequenceClassification.from_config(clf_config)

    if hasattr(clf_model, "bert"):
        clf_model.bert.load_state_dict(base_body_state, strict=False)
    elif hasattr(clf_model, "base_model"):
        clf_model.base_model.load_state_dict(base_body_state, strict=False)

    torch.manual_seed(random_seed)
    head = getattr(clf_model, "classifier", None) or getattr(clf_model, "score", None)
    if head is not None and hasattr(head, "reset_parameters"):
        head.reset_parameters()
    elif head is None:
        hidden_size = getattr(clf_config, "hidden_size", 768)
        clf_model.classifier = nn.Linear(hidden_size, 3)
        head = clf_model.classifier

    head_initial_hash = hash_model_parameters(head)

    tok = tokenizer
    if tok is None and mock_base_model is None:
        tok = AutoTokenizer.from_pretrained(base_model_name_or_path, revision=base_revision)

    clf_model.to(dev)
    clf_model.eval()

    adapter = HuggingFaceTemporalEncoder(
        name=name,
        model_name_or_path=base_model_name_or_path,
        revision=base_revision,
        tokenizer=tok,
        model=clf_model,
        task_label_schema="fomc_stance",
        device=dev,
        random_seed=random_seed,
    )
    adapter.original_head_loaded = False
    adapter.stance_head_initialization = "fresh"
    adapter.stance_head_initial_hash = head_initial_hash
    return adapter


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
        task_label_schema: str = "fomc_stance",
        reuse_existing_head: bool = False,
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

        self.task_label_schema = task_label_schema
        self.reuse_existing_head = reuse_existing_head
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.metadata_extra = metadata_extra or {}

        # Classifier head provenance
        self.original_head_loaded = False
        self.stance_head_initialization = "fresh"
        self.stance_head_initial_hash: Optional[str] = None

        # Allow dependency injection (for fast offline tests without network downloads)
        self._tokenizer = tokenizer
        self._model = model

        if self._model is not None:
            self._validate_and_prepare_model(self._model)

    def _validate_and_prepare_model(self, model: Any) -> None:
        """Validate label semantics and adapt classification head to FOMC stance."""
        if self.task_label_schema == "fomc_stance":
            config = getattr(model, "config", None)
            if config is not None:
                id2label = getattr(config, "id2label", {}) or {}
                labels_str = {str(v).lower() for v in id2label.values()}
                has_sentiment = any(k in labels_str for k in ["positive", "negative", "neutral_sentiment"])
                if has_sentiment:
                    if self.reuse_existing_head:
                        raise ValueError(
                            f"Model label schema {id2label} has financial sentiment labels "
                            "which cannot be used as FOMC stance schema when reuse_existing_head=True."
                        )
                    # Discard sentiment head and reinitialize fresh FOMC head
                    self._reinit_fomc_stance_head(model)
                else:
                    # Update config id2label mapping
                    config.id2label = dict(FOMC_STANCE_ID_TO_LABEL)
                    config.label2id = dict(FOMC_STANCE_LABEL_TO_ID)
                    config.num_labels = 3
                    if not self.reuse_existing_head and self.stance_head_initial_hash is None:
                        self._reinit_fomc_stance_head(model)
                    else:
                        head = getattr(model, "classifier", None) or getattr(model, "score", None)
                        if head is not None and self.stance_head_initial_hash is None:
                            self.stance_head_initial_hash = hash_model_parameters(head)

    def _reinit_fomc_stance_head(self, model: Any) -> None:
        """Discard existing classification head and reinitialize for 3-class FOMC stance."""
        config = getattr(model, "config", None)
        hidden_size = getattr(config, "hidden_size", 768) if config else 768
        torch.manual_seed(self.random_seed)
        if hasattr(model, "classifier"):
            model.classifier = nn.Linear(hidden_size, 3)
            self.stance_head_initial_hash = hash_model_parameters(model.classifier)
        elif hasattr(model, "score"):
            model.score = nn.Linear(hidden_size, 3)
            self.stance_head_initial_hash = hash_model_parameters(model.score)
        if config is not None:
            config.id2label = dict(FOMC_STANCE_ID_TO_LABEL)
            config.label2id = dict(FOMC_STANCE_LABEL_TO_ID)
            config.num_labels = 3
        self.original_head_loaded = False
        self.stance_head_initialization = "fresh"

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
            fresh_adapter = build_fresh_fomc_classifier_from_base_encoder(
                base_model_name_or_path=self.model_name_or_path,
                base_revision=self.revision,
                random_seed=self.random_seed,
                device=self.device,
                tokenizer=self._tokenizer,
                name=self.name,
            )
            self._model = fresh_adapter.model
            self._tokenizer = fresh_adapter.tokenizer
            self.original_head_loaded = False
            self.stance_head_initialization = "fresh"
            self.stance_head_initial_hash = fresh_adapter.stance_head_initial_hash
        return self._model

    def _prepare_inputs(self, texts: Sequence[str]) -> Dict[str, Any]:
        """Tokenize texts and move tensors to target device."""
        config = getattr(self.model, "config", None)
        max_pos = getattr(config, "max_position_embeddings", 512) if config else 512
        max_len = min(512, max_pos)
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_len,
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
            hidden_dim = getattr(getattr(self.model, "config", None), "hidden_size", 768)
            return np.empty((0, hidden_dim), dtype=np.float32)

        self.model.eval()
        inputs = self._prepare_inputs(texts)

        with torch.no_grad():
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

    def get_stance_score(self, texts: Sequence[str]) -> np.ndarray:
        """Extract continuous monetary policy stance score s in [-1, +1].

        Calculated as: s = P(Hawkish) - P(Dovish).
        """
        _, probs = self.predict_task(texts)
        if len(probs) == 0:
            return np.empty((0,), dtype=np.float32)
        # Class 0: Dovish, Class 2: Hawkish
        scores = probs[:, 2] - probs[:, 0]
        return scores.astype(np.float32)

    def save_checkpoint(
        self,
        save_directory: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        code_commit: Optional[str] = None,
        dataset_manifest_hash: Optional[str] = None,
    ) -> Path:
        """Save model weights, tokenizer, and audit provenance metadata.

        Provenance resolution is dynamic (no hardcoded commit or static hashes).
        """
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(out_dir)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(out_dir)

        # Resolve dataset manifest hash dynamically
        resolved_manifest_hash = dataset_manifest_hash or "unknown"
        if dataset_manifest_hash is None:
            manifest_path = Path("data/research/fomc/manifest.json")
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                        resolved_manifest_hash = m_data.get("checksum") or m_data.get("sha256", "unknown")
                except Exception:
                    pass

        resolved_commit = code_commit or resolve_git_commit(out_dir)
        git_prov = resolve_git_provenance(out_dir)

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
            "code_commit": resolved_commit,
            "git_dirty": git_prov["git_dirty"],
            "code_commit_exact": git_prov["code_commit_exact"],
            "git_provenance": git_prov,
            "original_head_loaded": self.original_head_loaded,
            "stance_head_initialization": self.stance_head_initialization,
            "stance_head_initial_hash": self.stance_head_initial_hash,
            "dataset_manifest_hash": resolved_manifest_hash,
            "pooling": self.pooling,
            "parameter_hash": hash_model_parameters(self.model),
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

        adapter = cls(
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
        adapter.original_head_loaded = metadata.get("original_head_loaded", False)
        adapter.stance_head_initialization = metadata.get("stance_head_initialization", "fresh")
        adapter.stance_head_initial_hash = metadata.get("stance_head_initial_hash")
        return adapter
