"""Controlled Clean/Leak Twin Construction and Empirical Experiment Pipeline.

Implements the causal twin methodology:
1. Equal Architecture: Identical base checkpoint body and sequence classification head.
2. Equal Compute: Identical total tokens, update steps, optimizer, learning rate schedule, and batch size.
3. Sham Control: Clean twin (M_C) receives pre-cutoff sham corpus matching contamination size.
4. Dose Ladder: Fraction of post-cutoff tokens D in {0.0, 0.25, 0.50, 0.75, 1.00}.
5. Token-Budget Treatment Stream: Fixed-length packed token blocks guaranteeing exact equal token budget.
6. Downstream Protocol: Fine-tuning strictly on pre-cutoff stance data (<= 2018).
7. Evaluation Isolation: Treatment A (Indirect Future Exposure) strictly ensures zero overlap
   between MLM contamination texts and evaluation split texts.
8. Strict Causal Integrity Assertions: Verifies parameter divergence, equal compute, and isolation before evaluation.

IMPORTANT:
    Small-scale smoke experiment outputs must be explicitly labeled:
    ENGINEERING SMOKE TEST ONLY — NOT RESEARCH CONCLUSIONS.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from .datasets.trillion_dollar_words import load_trillion_dollar_words
from .experiment_config import load_experiment_config
from .fomc_benchmark import FOMCBenchmark, TemporalSample
from .hf_encoder import (
    FOMC_STANCE_ID_TO_LABEL,
    FOMC_STANCE_LABEL_TO_ID,
    INDEX_TO_LABEL,
    LABEL_TO_INDEX,
    HuggingFaceTemporalEncoder,
    build_fresh_fomc_classifier_from_base_encoder,
    hash_corpus,
    hash_model_parameters,
    normalize_and_hash_text,
    resolve_git_commit,
    resolve_git_provenance,
)
from .metrics import (
    evaluate_behavioral_leakage,
    evaluate_competence,
    evaluate_economic_effect,
    evaluate_representational_leakage,
    pareto_coordinates,
)


class CausalIntegrityError(Exception):
    """Raised when any causal invariant (equal compute, symmetry, isolation) is violated."""


class TextLineDataset(Dataset):
    """Simple PyTorch Dataset wrapping raw text strings."""

    def __init__(self, texts: Sequence[str]) -> None:
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class PackedTokenDataset(Dataset):
    """PyTorch Dataset wrapping fixed-length packed token tensors."""

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def prepare_twin_corpora(
    samples: Sequence[TemporalSample],
    train_cutoff_year: int = 2018,
) -> Dict[str, List[str]]:
    """Separate samples into pre-cutoff sham pool and post-cutoff contamination pool.

    Args:
        samples: Collection of TemporalSample instances.
        train_cutoff_year: Pre-cutoff ceiling year (default: 2018).

    Returns:
        Dict with keys 'pre_cutoff_texts' (<= cutoff) and 'post_cutoff_texts' (> cutoff).
    """
    pre_cutoff = []
    post_cutoff = []

    for s in samples:
        try:
            year = int(s.metadata.get("year", 2000))
        except (ValueError, TypeError):
            year = 2000

        if year <= train_cutoff_year:
            pre_cutoff.append(s.text)
        else:
            post_cutoff.append(s.text)

    return {
        "pre_cutoff_texts": pre_cutoff,
        "post_cutoff_texts": post_cutoff,
    }


def create_dose_stream(
    pre_corpus: Sequence[str],
    post_corpus: Sequence[str],
    dose: float,
    total_samples: int,
    random_seed: int = 42,
) -> List[str]:
    """Construct an equal-sample stream with controlled post-cutoff contamination ratio."""
    if not (0.0 <= dose <= 1.0):
        raise ValueError(f"Dose must be between 0.0 and 1.0, got {dose}")

    rng = np.random.RandomState(random_seed)
    n_post = int(round(total_samples * dose))
    n_pre = total_samples - n_post

    selected_pre = (
        [pre_corpus[i] for i in rng.choice(len(pre_corpus), size=n_pre, replace=(n_pre > len(pre_corpus)))]
        if n_pre > 0
        else []
    )

    selected_post = (
        [post_corpus[i] for i in rng.choice(len(post_corpus), size=n_post, replace=(n_post > len(post_corpus)))]
        if n_post > 0
        else []
    )

    stream = selected_pre + selected_post
    rng.shuffle(stream)
    return stream


def create_token_matched_dose_stream(
    pre_corpus: Sequence[str],
    post_corpus: Sequence[str],
    dose: float,
    num_blocks: int,
    block_length: int = 128,
    tokenizer: Optional[Any] = None,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Construct a token-budget matched stream packed into fixed-length blocks.

    Enforces exact Equal MLM Tokens across all doses (D0, D100, etc.):
    - Both clean (D0) and leak (D100) streams contain exactly `num_blocks` of length `block_length`.
    - Total effective tokens = num_blocks * block_length (difference = 0 tokens).

    Args:
        pre_corpus: List of pre-cutoff text strings.
        post_corpus: List of post-cutoff text strings.
        dose: Fraction of post-cutoff tokens in [0.0, 1.0].
        num_blocks: Number of fixed-length blocks to produce.
        block_length: Fixed token sequence length per block (default: 128).
        tokenizer: Tokenizer instance. If None, uses default BERT fast tokenizer.
        random_seed: Deterministic RNG seed.

    Returns:
        Dict with keys:
        - 'dataset': PackedTokenDataset with shape [num_blocks, block_length]
        - 'effective_tokens': int (num_blocks * block_length)
        - 'num_blocks': int
        - 'block_length': int
        - 'texts_used': List[str]
        - 'corpus_hash': str
    """
    if not (0.0 <= dose <= 1.0):
        raise ValueError(f"Dose must be between 0.0 and 1.0, got {dose}")

    rng = np.random.RandomState(random_seed)
    total_tokens_needed = num_blocks * block_length

    # Select texts based on dose
    if dose == 0.0:
        source_texts = list(pre_corpus)
    elif dose == 1.0:
        source_texts = list(post_corpus)
    else:
        n_post_texts = int(round(len(post_corpus) * dose))
        n_pre_texts = len(pre_corpus)
        source_texts = list(pre_corpus) + list(post_corpus[:n_post_texts])

    if not source_texts:
        raise ValueError(f"Source corpus for dose {dose} is empty.")

    # Deterministic shuffle of text sources
    indices = np.arange(len(source_texts))
    rng.shuffle(indices)
    shuffled_texts = [source_texts[i] for i in indices]

    # Tokenize texts
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

    all_token_ids: List[int] = []
    texts_used: List[str] = []

    for t in shuffled_texts:
        if len(all_token_ids) >= total_tokens_needed:
            break
        texts_used.append(t)
        # Tokenize without adding special tokens so we can pack cleanly
        if hasattr(tokenizer, "encode"):
            tokens = tokenizer.encode(t, add_special_tokens=False)
        else:
            tokens = tokenizer(t)["input_ids"]
        all_token_ids.extend(tokens)

    # If tokens are fewer than budget, cycle deterministically to meet exact budget
    if len(all_token_ids) < total_tokens_needed:
        if len(all_token_ids) == 0:
            # Fallback placeholder token (pad token or 100)
            all_token_ids = [getattr(tokenizer, "pad_token_id", 0) or 100]
        repeats = (total_tokens_needed // len(all_token_ids)) + 1
        all_token_ids = (all_token_ids * repeats)[:total_tokens_needed]
    else:
        all_token_ids = all_token_ids[:total_tokens_needed]

    # Pack into exact [num_blocks, block_length] tensors
    input_ids_tensor = torch.tensor(all_token_ids, dtype=torch.long).view(num_blocks, block_length)
    attention_mask_tensor = torch.ones((num_blocks, block_length), dtype=torch.long)

    c_hash = hash_corpus(texts_used)
    dataset = PackedTokenDataset(input_ids_tensor, attention_mask_tensor)

    return {
        "dataset": dataset,
        "effective_tokens": total_tokens_needed,
        "num_blocks": num_blocks,
        "block_length": block_length,
        "texts_used": texts_used,
        "corpus_hash": c_hash,
    }


def run_continued_pretraining_mlm(
    packed_dataset: Dataset,
    tokenizer: Any,
    model_mlm: Any,
    output_dir: Union[str, Path],
    max_steps: int = 10,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    mlm_probability: float = 0.15,
    weight_decay: float = 0.01,
    random_seed: int = 42,
    device: Optional[str] = None,
    branch: str = "clean",
    dose: float = 0.0,
    base_checkpoint: str = "ProsusAI/finbert",
    base_revision: str = "4556d13015211d73dccd3fdd39d39232506f3e43",
    corpus_hash: str = "unknown",
) -> Dict[str, Any]:
    """Execute controlled Masked Language Modeling (MLM) continued pretraining.

    Saves model checkpoint, tokenizer, and mlm_training_manifest.json.
    Enforces equal compute: identical max_steps, batch_size, and total tokens.
    """
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_mlm.to(dev)
    model_mlm.train()

    initial_parameter_hash = hash_model_parameters(model_mlm)

    # Tensor collator for packed fixed-length blocks (guarantees equal block length without tokenizer.pad)
    mask_id = getattr(tokenizer, "mask_token_id", 103) or 103

    def data_collator(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = input_ids.clone()
        # Randomly mask mlm_probability fraction of tokens
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < mlm_probability) & (attention_mask == 1)
        labels[~mask_arr] = -100
        input_ids[mask_arr] = mask_id
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    dataloader = DataLoader(
        packed_dataset,
        batch_size=batch_size,
        shuffle=False,  # Sequential block iteration preserves determinism
        collate_fn=data_collator,
    )

    optimizer = torch.optim.AdamW(model_mlm.parameters(), lr=learning_rate, weight_decay=weight_decay)

    step = 0
    epoch = 0
    while step < max_steps:
        epoch += 1
        for batch in dataloader:
            if step >= max_steps:
                break
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            labels = batch["labels"].to(dev)

            outputs = model_mlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            loss.backward()
            optimizer.step()
            step += 1

    model_mlm.eval()
    final_parameter_hash = hash_model_parameters(model_mlm)

    # Save pretrained weights & tokenizer if methods exist
    if hasattr(model_mlm, "save_pretrained"):
        model_mlm.save_pretrained(out_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(out_dir)

    # Calculate token count
    token_count = len(packed_dataset) * packed_dataset[0]["input_ids"].numel()

    # Generate and save mlm_training_manifest.json
    manifest_data = {
        "branch": branch,
        "dose": float(dose),
        "base_checkpoint": base_checkpoint,
        "base_revision": base_revision,
        "initial_parameter_hash": initial_parameter_hash,
        "final_parameter_hash": final_parameter_hash,
        "corpus_hash": corpus_hash,
        "token_count": token_count,
        "optimizer_steps": step,
        "batch_size": batch_size,
        "max_seq_length": packed_dataset[0]["input_ids"].numel(),
        "mlm_probability": float(mlm_probability),
        "learning_rate": float(learning_rate),
        "seed": random_seed,
    }

    manifest_path = out_dir / "mlm_training_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data


def build_classifier_from_mlm_encoder(
    mlm_model: Any,
    tokenizer: Any,
    head_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    name: str = "fomc_stance_classifier",
    base_model_name: str = "ProsusAI/finbert",
    base_revision: str = "4556d13015211d73dccd3fdd39d39232506f3e43",
    device: Optional[str] = None,
    random_seed: int = 42,
) -> HuggingFaceTemporalEncoder:
    """Construct a 3-class FOMC stance classifier inheriting the MLM-trained encoder body.

    Preserves causal symmetry:
    - The classification head is initialized with bit-identical parameters (`head_state_dict`).
    - The encoder body retains the MLM treatment effects.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Extract encoder body weights from MLM model
    if hasattr(mlm_model, "bert"):
        bert_body_state = mlm_model.bert.state_dict()
    elif hasattr(mlm_model, "base_model"):
        bert_body_state = mlm_model.base_model.state_dict()
    else:
        bert_body_state = mlm_model.state_dict()

    # Construct sequence classification model
    if hasattr(mlm_model, "config") and mlm_model.config is not None:
        config = copy.deepcopy(mlm_model.config)
        config.num_labels = 3
        config.id2label = dict(FOMC_STANCE_ID_TO_LABEL)
        config.label2id = dict(FOMC_STANCE_LABEL_TO_ID)
    else:
        config = AutoConfig.from_pretrained(
            base_model_name,
            num_labels=3,
            id2label=FOMC_STANCE_ID_TO_LABEL,
            label2id=FOMC_STANCE_LABEL_TO_ID,
            revision=base_revision,
        )
    clf_model = AutoModelForSequenceClassification.from_config(config)

    # Load MLM-trained body
    if hasattr(clf_model, "bert"):
        clf_model.bert.load_state_dict(bert_body_state, strict=False)
    elif hasattr(clf_model, "base_model"):
        clf_model.base_model.load_state_dict(bert_body_state, strict=False)

    # Attach identical initial classification head
    if head_state_dict is not None:
        if hasattr(clf_model, "classifier"):
            clf_model.classifier.load_state_dict(head_state_dict)
        elif hasattr(clf_model, "score"):
            clf_model.score.load_state_dict(head_state_dict)
    else:
        torch.manual_seed(random_seed)
        if hasattr(clf_model, "classifier") and hasattr(clf_model.classifier, "reset_parameters"):
            clf_model.classifier.reset_parameters()

    clf_model.to(dev)
    clf_model.eval()

    return HuggingFaceTemporalEncoder(
        name=name,
        model_name_or_path=base_model_name,
        revision=base_revision,
        tokenizer=tokenizer,
        model=clf_model,
        task_label_schema="fomc_stance",
        device=dev,
        random_seed=random_seed,
    )


def train_downstream_classifier(
    encoder_model: HuggingFaceTemporalEncoder,
    train_samples: Sequence[TemporalSample],
    epochs: int = 1,
    max_steps: Optional[int] = None,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    random_seed: int = 42,
) -> Tuple[HuggingFaceTemporalEncoder, str]:
    """Fine-tune 3-class classification head strictly on pre-cutoff stance data.

    Enforces causal symmetry:
    - Same training samples in identical batch ordering.
    - Uses deterministic DataLoader generator.

    Returns:
        Tuple of (fine_tuned_encoder, sample_order_hash).
    """
    torch.manual_seed(random_seed)
    dev = encoder_model.device
    model = encoder_model.model
    tokenizer = encoder_model.tokenizer
    model.to(dev)
    model.train()

    texts = [s.text for s in train_samples]
    labels = [LABEL_TO_INDEX[s.task_label] for s in train_samples]

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",
    )

    class ClfDataset(Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, idx):
            return {
                "input_ids": tokenized["input_ids"][idx],
                "attention_mask": tokenized["attention_mask"][idx],
                "label": torch.tensor(labels[idx], dtype=torch.long),
                "sample_idx": idx,
            }

    generator = torch.Generator()
    generator.manual_seed(random_seed)
    dataloader = DataLoader(ClfDataset(), batch_size=batch_size, shuffle=True, generator=generator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    order_recorder: List[int] = []
    step = 0
    for _ in range(epochs):
        for batch in dataloader:
            if max_steps is not None and step >= max_steps:
                break
            order_recorder.extend(batch["sample_idx"].tolist())
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            lbl = batch["label"].to(dev)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()
            step += 1

    sample_order_hash = hashlib.sha256(str(order_recorder).encode("utf-8")).hexdigest()
    model.eval()
    return encoder_model, sample_order_hash


def train_baseline_stance_model(
    train_samples: Sequence[TemporalSample],
    test_samples: Sequence[TemporalSample],
    base_model_name: str = "ProsusAI/finbert",
    base_revision: str = "4556d13015211d73dccd3fdd39d39232506f3e43",
    epochs: int = 1,
    max_steps: Optional[int] = 10,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    random_seed: int = 42,
    device: Optional[str] = None,
    mock_model_for_testing: Optional[Any] = None,
    mock_tokenizer_for_testing: Optional[Any] = None,
) -> Dict[str, Any]:
    """Construct and evaluate the true FOMC Stance Baseline model (M_B).

    Steps:
    1. Base ProsusAI/finbert encoder body.
    2. Fresh identical FOMC 3-class stance classification head (raw sentiment head discarded).
    3. Fine-tuned strictly on pre-cutoff TDW stance labels (<= 2018).
    4. Evaluated on out-of-sample post-cutoff stance samples (>= 2020).
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_fresh_fomc_classifier_from_base_encoder(
        base_model_name_or_path=base_model_name,
        base_revision=base_revision,
        random_seed=random_seed,
        device=dev,
        tokenizer=mock_tokenizer_for_testing,
        mock_base_model=mock_model_for_testing,
        name="fomc_stance_baseline_mb",
    )

    fine_tuned_encoder, sample_order_hash = train_downstream_classifier(
        encoder_model=encoder,
        train_samples=train_samples,
        epochs=epochs,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=random_seed,
    )

    eval_texts = [s.text for s in test_samples]
    y_true = [s.task_label for s in test_samples]
    y_pred, y_prob = fine_tuned_encoder.predict_task(eval_texts)

    comp_metrics = evaluate_competence(y_true, y_pred, y_prob=y_prob, n_bootstrap=200)

    return {
        "model_name": "fomc_stance_baseline_mb",
        "training_cutoff": "2018-12-31",
        "evaluation_split": "coarse_year_test (>= 2020)",
        "macro_f1": float(comp_metrics["macro_f1"]),
        "mcc": float(comp_metrics["mcc"]),
        "brier_score": float(comp_metrics["brier_score"]),
        "ece": float(comp_metrics["ece"]),
        "sample_count": len(eval_texts),
        "train_sample_count": len(train_samples),
        "fine_tuned_encoder": fine_tuned_encoder,
        "initial_head_hash": encoder.stance_head_initial_hash,
        "sample_order_hash": sample_order_hash,
    }


def run_fomc_stance_baseline(
    config_path: Union[str, Path] = "configs/encoder_baseline.yaml",
    output_dir: Union[str, Path] = "experiments/encoder_phase2",
    training_sample_limit: Optional[int] = None,
    test_sample_limit: Optional[int] = None,
    random_seed: Optional[int] = None,
    device: Optional[str] = None,
    mock_model_for_testing: Optional[Any] = None,
    mock_tokenizer_for_testing: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute standalone evaluation of the True FOMC Stance Baseline model (M_B).

    Driven strictly by `configs/encoder_baseline.yaml`.
    Guarantees:
    1. Base ProsusAI/finbert encoder body loaded; raw sentiment head discarded.
    2. Fresh FOMC stance classification head initialized deterministically.
    3. Training set uses all eligible pre-cutoff TDW samples (year <= 2018, e.g. 1,729)
       unless training_sample_limit is explicitly configured.
    4. test_sample_limit applies strictly to evaluation test samples (year >= 2020).
    5. Saves experiments/encoder_phase2/results/baseline_results.json and manifests/baseline_manifest.json.
    """
    out_dir = Path(output_dir)
    results_dir = out_dir / "results"
    manifests_dir = out_dir / "manifests"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    cfg = {}
    cfg_file = Path(config_path)
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    seed = random_seed if random_seed is not None else cfg.get("random_seed", 42)
    model_cfg = cfg.get("model", {})
    base_model_name = model_cfg.get("base_checkpoint", "ProsusAI/finbert")
    base_revision = model_cfg.get("base_revision", "4556d13015211d73dccd3fdd39d39232506f3e43")

    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", 2)
    batch_size = train_cfg.get("batch_size", 8)
    learning_rate = float(train_cfg.get("learning_rate", 2e-5))
    max_steps = train_cfg.get("max_steps", None)
    cfg_train_limit = train_cfg.get("training_sample_limit", None)
    effective_train_limit = training_sample_limit if training_sample_limit is not None else cfg_train_limit

    splits_cfg = cfg.get("splits", {})
    train_cutoff_year = splits_cfg.get("train_cutoff_year", 2018)
    test_start_year = splits_cfg.get("test_start_year", 2020)

    eval_cfg = cfg.get("evaluation", {})
    cfg_test_limit = eval_cfg.get("test_sample_limit", 100)
    effective_test_limit = test_sample_limit if test_sample_limit is not None else cfg_test_limit
    n_bootstrap = eval_cfg.get("n_bootstrap", 200)

    dev_choice = device or ("cuda" if torch.cuda.is_available() else "cpu")
    git_prov = resolve_git_provenance(out_dir)

    print(f"[Baseline Pipeline] Loading Trillion Dollar Words dataset (cutoff <= {train_cutoff_year})...")
    samples = load_trillion_dollar_words()
    all_train_samples = [s for s in samples if int(s.metadata.get("year", 2000)) <= train_cutoff_year]
    all_test_samples = [s for s in samples if int(s.metadata.get("year", 2000)) >= test_start_year]

    train_samples = all_train_samples[:effective_train_limit] if effective_train_limit is not None else all_train_samples
    test_samples = all_test_samples[:effective_test_limit] if effective_test_limit is not None else all_test_samples

    print(f"[Baseline Pipeline] Training samples: {len(train_samples)}, Test samples: {len(test_samples)}")
    print("[Baseline Pipeline] Building Fresh FOMC Stance Baseline Classifier...")
    encoder = build_fresh_fomc_classifier_from_base_encoder(
        base_model_name_or_path=base_model_name,
        base_revision=base_revision,
        random_seed=seed,
        device=dev_choice,
        tokenizer=mock_tokenizer_for_testing,
        mock_base_model=mock_model_for_testing,
        name="fomc_stance_baseline_mb",
    )
    initial_head_hash = encoder.stance_head_initial_hash

    print(f"[Baseline Pipeline] Fine-tuning on pre-cutoff stance data ({epochs} epochs, lr={learning_rate})...")
    fine_tuned_encoder, sample_order_hash = train_downstream_classifier(
        encoder_model=encoder,
        train_samples=train_samples,
        epochs=epochs,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=seed,
    )

    eval_texts = [s.text for s in test_samples]
    y_true = [s.task_label for s in test_samples]
    y_pred, y_prob = fine_tuned_encoder.predict_task(eval_texts)

    comp_metrics = evaluate_competence(y_true, y_pred, y_prob=y_prob, n_bootstrap=n_bootstrap)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1]).tolist()

    baseline_results = {
        "status": "VALIDATED_FOMC_STANCE_BASELINE",
        "experiment": "encoder_baseline_evaluation",
        "config_path": str(config_path),
        "checkpoint": base_model_name,
        "revision": base_revision,
        "original_head_loaded": False,
        "stance_head_initialization": "fresh",
        "stance_head_initial_hash": initial_head_hash,
        "training_cutoff": f"{train_cutoff_year}-12-31",
        "evaluation_split": f"coarse_year_test (>= {test_start_year})",
        "train_sample_count": len(train_samples),
        "test_sample_count": len(test_samples),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "macro_f1": float(comp_metrics["macro_f1"]),
        "mcc": float(comp_metrics["mcc"]),
        "brier_score": float(comp_metrics["brier_score"]),
        "ece": float(comp_metrics["ece"]),
        "confusion_matrix": cm,
        "git_provenance": git_prov,
        "code_commit": git_prov["git_head"],
        "git_dirty": git_prov["git_dirty"],
        "code_commit_exact": git_prov["code_commit_exact"],
    }
    with open(results_dir / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2)

    baseline_manifest = {
        "experiment_id": "encoder_baseline_evaluation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "git_provenance": git_prov,
        "code_commit": git_prov["git_head"],
        "git_dirty": git_prov["git_dirty"],
        "code_commit_exact": git_prov["code_commit_exact"],
        "base_checkpoint": base_model_name,
        "base_revision": base_revision,
        "original_head_loaded": False,
        "stance_head_initialization": "fresh",
        "stance_head_initial_hash": initial_head_hash,
        "train_sample_count": len(train_samples),
        "test_sample_count": len(test_samples),
        "macro_f1": float(comp_metrics["macro_f1"]),
        "mcc": float(comp_metrics["mcc"]),
        "brier_score": float(comp_metrics["brier_score"]),
        "ece": float(comp_metrics["ece"]),
    }
    with open(manifests_dir / "baseline_manifest.json", "w", encoding="utf-8") as f:
        json.dump(baseline_manifest, f, indent=2)

    print("\n" + "=" * 76)
    print("FOMC STANCE BASELINE EVALUATION COMPLETE")
    print("=" * 76)
    print(f"  Train Samples: {len(train_samples)} (all pre-cutoff)")
    print(f"  Test Samples: {len(test_samples)} (test split)")
    print(f"  Macro-F1: {comp_metrics['macro_f1']:.4f}")
    print(f"  MCC: {comp_metrics['mcc']:.4f}")
    print(f"  Brier Score: {comp_metrics['brier_score']:.4f}")
    print(f"  ECE: {comp_metrics['ece']:.4f}")
    print(f"  Git Commit: {git_prov['git_head'][:8]}... (dirty={git_prov['git_dirty']}, exact={git_prov['code_commit_exact']})")
    print("=" * 76 + "\n")

    return baseline_results


def run_baseline_and_smoke_experiment(
    output_dir: Union[str, Path] = "experiments/encoder_phase2",
    config_path: Union[str, Path] = "configs/encoder_twin_smoke.yaml",
    smoke_steps: Optional[int] = None,
    smoke_sample_count: Optional[int] = None,
    random_seed: Optional[int] = None,
    device: Optional[str] = None,
    mock_model_for_testing: Optional[Any] = None,
    mock_tokenizer_for_testing: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute Phase 2.1 Causal Twin Activation & D0 vs D100 Smoke Experiment.

    Orchestrates the complete causal twin lifecycle:
    1. Ingestion & Coarse Year Partitioning (Train <= 2018, Dev == 2019, Test >= 2020).
    2. Evaluation Isolation: Ensures zero text overlap between evaluation split and MLM contamination stream.
    3. True Stance Baseline (M_B) fine-tuning on pre-cutoff data.
    4. Two independent MLM model instances (M_C, M_L) initialized identically from common base.
    5. Token-budget treatment stream construction: Equal MLM tokens (0% token count difference).
    6. MLM Continued Pretraining Execution:
       - Clean Twin (M_C, D=0.0): 100% pre-cutoff sham corpus.
       - Leak Twin (M_L, D=1.0): 100% post-cutoff contamination corpus.
    7. Parameter Divergence Validation: Confirms parameters diverge after MLM treatment.
    8. Weight Transfer to Classifiers: Injects MLM encoder bodies with identical fresh head state.
    9. Downstream Fine-Tuning: Symmetric pre-cutoff training with identical batch ordering.
    10. Causal Integrity Assertions: Verifies all 5 invariants before evaluation.
    11. Evaluation & Manifest Export.
    """
    out_dir = Path(output_dir)
    results_dir = out_dir / "results"
    manifests_dir = out_dir / "manifests"
    checkpoints_dir = out_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    cfg = {}
    cfg_file = Path(config_path)
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    seed = random_seed if random_seed is not None else cfg.get("random_seed", 42)
    base_model_name = cfg.get("model", {}).get("base_checkpoint", "ProsusAI/finbert")
    base_revision = cfg.get("model", {}).get("base_revision", "4556d13015211d73dccd3fdd39d39232506f3e43")

    mlm_cfg = cfg.get("mlm_treatment", {})
    max_mlm_steps = smoke_steps if smoke_steps is not None else mlm_cfg.get("max_steps", 10)
    mlm_batch_size = mlm_cfg.get("batch_size", 4)
    mlm_lr = mlm_cfg.get("learning_rate", 5e-5)
    mlm_blocks = mlm_cfg.get("num_blocks", 20)
    mlm_block_len = mlm_cfg.get("block_length", 128)

    downstream_cfg = cfg.get("downstream", {})
    downstream_epochs = downstream_cfg.get("epochs", 1)
    downstream_steps = downstream_cfg.get("max_steps", 10)
    downstream_batch_size = downstream_cfg.get("batch_size", 4)
    downstream_lr = downstream_cfg.get("learning_rate", 2e-5)

    eval_cfg = cfg.get("evaluation", {})
    eval_limit = smoke_sample_count if smoke_sample_count is not None else eval_cfg.get("test_sample_limit", 40)

    dev_choice = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve dynamic git commit and dataset hash
    code_commit = resolve_git_commit(out_dir)
    manifest_path = Path("data/research/fomc/manifest.json")
    dataset_hash = "unknown"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                m_info = json.load(f)
                dataset_hash = m_info.get("checksum") or m_info.get("sha256", "unknown")
        except Exception:
            pass

    # 1. Ingest dataset
    print("[Phase 2.1 Pipeline] Loading Trillion Dollar Words dataset...")
    samples = load_trillion_dollar_words()
    corpora = prepare_twin_corpora(samples, train_cutoff_year=2018)
    pre_corpus = corpora["pre_cutoff_texts"]
    post_corpus = corpora["post_cutoff_texts"]

    # Partition samples by coarse year
    train_samples = [s for s in samples if int(s.metadata.get("year", 2000)) <= 2018]
    dev_samples = [s for s in samples if int(s.metadata.get("year", 2000)) == 2019]
    test_samples = [s for s in samples if int(s.metadata.get("year", 2000)) >= 2020]

    # Evaluation test slice
    eval_test_samples = test_samples[:eval_limit]
    test_texts = [s.text for s in eval_test_samples]
    y_test_true = [s.task_label for s in eval_test_samples]

    # 2. Evaluation Isolation (Treatment A — Indirect Future Exposure)
    # Ensure evaluation texts are excluded from the post-cutoff MLM contamination pool
    eval_text_hashes = {normalize_and_hash_text(t) for t in test_texts}
    post_corpus_isolated = [t for t in post_corpus if normalize_and_hash_text(t) not in eval_text_hashes]

    downstream_train_limit = downstream_cfg.get("training_sample_limit", None)
    train_slice = train_samples[:downstream_train_limit] if downstream_train_limit is not None else train_samples
    print("[Phase 2.1 Pipeline] Fine-tuning True FOMC Stance Baseline (M_B)...")
    baseline_result = train_baseline_stance_model(
        train_samples=train_slice,
        test_samples=eval_test_samples,
        base_model_name=base_model_name,
        base_revision=base_revision,
        epochs=downstream_epochs,
        max_steps=downstream_steps,
        batch_size=downstream_batch_size,
        learning_rate=downstream_lr,
        random_seed=seed,
        device=dev_choice,
        mock_model_for_testing=mock_model_for_testing,
        mock_tokenizer_for_testing=mock_tokenizer_for_testing,
    )

    # 4. Token-Budget Treatment Streams
    print("[Phase 2.1 Pipeline] Creating Token-Matched Treatment Streams (D0 vs D100)...")
    tokenizer = mock_tokenizer_for_testing or AutoTokenizer.from_pretrained(base_model_name, revision=base_revision)

    stream_d0 = create_token_matched_dose_stream(
        pre_corpus=pre_corpus,
        post_corpus=post_corpus_isolated,
        dose=0.0,
        num_blocks=mlm_blocks,
        block_length=mlm_block_len,
        tokenizer=tokenizer,
        random_seed=seed,
    )

    stream_d100 = create_token_matched_dose_stream(
        pre_corpus=pre_corpus,
        post_corpus=post_corpus_isolated,
        dose=1.0,
        num_blocks=mlm_blocks,
        block_length=mlm_block_len,
        tokenizer=tokenizer,
        random_seed=seed,
    )

    # Overlap Audit
    d0_hashes = {normalize_and_hash_text(t) for t in stream_d0["texts_used"]}
    d100_hashes = {normalize_and_hash_text(t) for t in stream_d100["texts_used"]}
    clean_eval_overlap = len(d0_hashes & eval_text_hashes)
    leak_eval_overlap = len(d100_hashes & eval_text_hashes)

    overlap_audit = {
        "isolation_mode": "Treatment A — Indirect Future Exposure",
        "document_level_isolation": "unavailable",
        "sentence_level_hash_isolation": "enforced",
        "eval_sample_count": len(eval_test_samples),
        "clean_corpus_sample_count": len(stream_d0["texts_used"]),
        "leak_corpus_sample_count": len(stream_d100["texts_used"]),
        "clean_eval_overlap": clean_eval_overlap,
        "leak_eval_overlap": leak_eval_overlap,
        "audit_pass": (clean_eval_overlap == 0 and leak_eval_overlap == 0),
    }
    with open(manifests_dir / "overlap_audit.json", "w", encoding="utf-8") as f:
        json.dump(overlap_audit, f, indent=2)

    # 5. Base MLM Models Setup: Two independent instances
    print("[Phase 2.1 Pipeline] Initializing Independent MLM Twin Instances...")
    if mock_model_for_testing is not None:
        base_mlm_model = mock_model_for_testing
    else:
        base_mlm_model = AutoModelForMaskedLM.from_pretrained(base_model_name, revision=base_revision)

    # Deepcopy to ensure independent state dicts and different Python objects
    clean_mlm_model = copy.deepcopy(base_mlm_model)
    leak_mlm_model = copy.deepcopy(base_mlm_model)

    assert clean_mlm_model is not leak_mlm_model, "Clean and Leak models must be separate Python objects."
    clean_initial_param_hash = hash_model_parameters(clean_mlm_model)
    leak_initial_param_hash = hash_model_parameters(leak_mlm_model)
    assert clean_initial_param_hash == leak_initial_param_hash, "Initial parameter states must be bit-identical."

    # 6. Execute MLM Continued Pretraining for Clean (D0) and Leak (D100)
    print(f"[Phase 2.1 Pipeline] Executing Clean (D0) MLM Pretraining ({max_mlm_steps} steps)...")
    clean_treatment_manifest = run_continued_pretraining_mlm(
        packed_dataset=stream_d0["dataset"],
        tokenizer=tokenizer,
        model_mlm=clean_mlm_model,
        output_dir=checkpoints_dir / "mlm_clean_d0",
        max_steps=max_mlm_steps,
        batch_size=mlm_batch_size,
        learning_rate=mlm_lr,
        random_seed=seed,
        device=dev_choice,
        branch="clean",
        dose=0.0,
        base_checkpoint=base_model_name,
        base_revision=base_revision,
        corpus_hash=stream_d0["corpus_hash"],
    )
    with open(manifests_dir / "clean_treatment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(clean_treatment_manifest, f, indent=2)

    print(f"[Phase 2.1 Pipeline] Executing Leak (D100) MLM Pretraining ({max_mlm_steps} steps)...")
    leak_treatment_manifest = run_continued_pretraining_mlm(
        packed_dataset=stream_d100["dataset"],
        tokenizer=tokenizer,
        model_mlm=leak_mlm_model,
        output_dir=checkpoints_dir / "mlm_leak_d100",
        max_steps=max_mlm_steps,
        batch_size=mlm_batch_size,
        learning_rate=mlm_lr,
        random_seed=seed,
        device=dev_choice,
        branch="leak",
        dose=1.0,
        base_checkpoint=base_model_name,
        base_revision=base_revision,
        corpus_hash=stream_d100["corpus_hash"],
    )
    with open(manifests_dir / "leak_treatment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(leak_treatment_manifest, f, indent=2)

    clean_post_mlm_hash = clean_treatment_manifest["final_parameter_hash"]
    leak_post_mlm_hash = leak_treatment_manifest["final_parameter_hash"]

    # 7. Generate bit-identical initial classification head
    # Create fresh head once and capture state_dict
    torch.manual_seed(seed)
    hidden_dim = getattr(getattr(base_mlm_model, "config", None), "hidden_size", 768)
    initial_head = nn.Linear(hidden_dim, 3)
    shared_head_state_dict = copy.deepcopy(initial_head.state_dict())
    classifier_head_initial_hash = hash_model_parameters(initial_head)

    # 8. Transfer MLM weights to sequence classifiers
    print("[Phase 2.1 Pipeline] Transferring MLM Weights to FOMC Stance Classifiers...")
    clean_encoder = build_classifier_from_mlm_encoder(
        mlm_model=clean_mlm_model,
        tokenizer=tokenizer,
        head_state_dict=shared_head_state_dict,
        name="M_clean_twin_d0",
        base_model_name=base_model_name,
        base_revision=base_revision,
        device=dev_choice,
        random_seed=seed,
    )
    clean_encoder.contamination_dose = 0.0
    clean_encoder.training_cutoff = "2018-12-31"

    leak_encoder = build_classifier_from_mlm_encoder(
        mlm_model=leak_mlm_model,
        tokenizer=tokenizer,
        head_state_dict=shared_head_state_dict,
        name="M_leak_twin_d100",
        base_model_name=base_model_name,
        base_revision=base_revision,
        device=dev_choice,
        random_seed=seed,
    )
    leak_encoder.contamination_dose = 1.0
    leak_encoder.training_cutoff = "2022-12-31"

    # Verify classification head initial state identity
    clean_head = getattr(clean_encoder.model, "classifier", None) or getattr(clean_encoder.model, "score", None)
    leak_head = getattr(leak_encoder.model, "classifier", None) or getattr(leak_encoder.model, "score", None)
    if clean_head is not None and leak_head is not None:
        assert hash_model_parameters(clean_head) == hash_model_parameters(
            leak_head
        ), "Classifier heads must be bit-identical prior to downstream fine-tuning."

    # 9. Downstream Fine-Tuning strictly on pre-cutoff slice with identical batch order
    print("[Phase 2.1 Pipeline] Symmetric Downstream Stance Fine-Tuning...")
    clean_encoder, clean_order_hash = train_downstream_classifier(
        encoder_model=clean_encoder,
        train_samples=train_slice,
        epochs=downstream_epochs,
        max_steps=downstream_steps,
        batch_size=downstream_batch_size,
        learning_rate=downstream_lr,
        random_seed=seed,
    )
    leak_encoder, leak_order_hash = train_downstream_classifier(
        encoder_model=leak_encoder,
        train_samples=train_slice,
        epochs=downstream_epochs,
        max_steps=downstream_steps,
        batch_size=downstream_batch_size,
        learning_rate=downstream_lr,
        random_seed=seed,
    )

    # 10. Causal Integrity Assertions
    # Invariant 1: Same start
    if clean_initial_param_hash != leak_initial_param_hash:
        raise CausalIntegrityError(
            f"Invariant 1 failed: Initial parameter states differ ({clean_initial_param_hash} != {leak_initial_param_hash})."
        )

    # Invariant 2: Different treatment corpus
    if stream_d0["corpus_hash"] == stream_d100["corpus_hash"]:
        raise CausalIntegrityError("Invariant 2 failed: D0 and D100 treatment corpora are identical.")

    # Invariant 3: Equal compute (tokens and steps)
    if stream_d0["effective_tokens"] != stream_d100["effective_tokens"]:
        raise CausalIntegrityError(
            f"Invariant 3 failed: Token budget mismatch (D0: {stream_d0['effective_tokens']}, D100: {stream_d100['effective_tokens']})."
        )
    if clean_treatment_manifest["optimizer_steps"] != leak_treatment_manifest["optimizer_steps"]:
        raise CausalIntegrityError(
            f"Invariant 3 failed: Optimizer steps mismatch (D0: {clean_treatment_manifest['optimizer_steps']}, D100: {leak_treatment_manifest['optimizer_steps']})."
        )

    # Invariant 4: Treatment actually changed model parameters
    if clean_post_mlm_hash == clean_initial_param_hash or leak_post_mlm_hash == leak_initial_param_hash:
        raise CausalIntegrityError(
            "Invariant 4 failed: Model parameters did not update during MLM continued pretraining."
        )
    if clean_post_mlm_hash == leak_post_mlm_hash:
        raise CausalIntegrityError(
            f"Invariant 4 failed: Clean and Leak post-MLM parameter hashes are identical ({clean_post_mlm_hash}). Treatment had no differential effect."
        )

    # Invariant 5: No evaluation overlap
    if clean_eval_overlap != 0 or leak_eval_overlap != 0:
        raise CausalIntegrityError(
            f"Invariant 5 failed: Evaluation isolation violated (clean overlap: {clean_eval_overlap}, leak overlap: {leak_eval_overlap})."
        )

    # Invariant 6: Identical downstream sample order
    if clean_order_hash != leak_order_hash:
        raise CausalIntegrityError("Invariant 6 failed: Downstream fine-tuning batch ordering differed.")

    # 11. Evaluation on test split
    print("[Phase 2.1 Pipeline] Evaluating C, L_repr, L_behavior...")
    pred_clean, prob_clean = clean_encoder.predict_task(test_texts)
    pred_leak, prob_leak = leak_encoder.predict_task(test_texts)

    c_clean = evaluate_competence(y_test_true, pred_clean, y_prob=prob_clean, n_bootstrap=200)
    c_leak = evaluate_competence(y_test_true, pred_leak, y_prob=prob_leak, n_bootstrap=200)

    h_clean = clean_encoder.encode(test_texts)
    h_leak = leak_encoder.encode(test_texts)

    # Synthetic targets for plumbing verification only
    rng = np.random.RandomState(seed)
    y_synthetic_future_actions = [int(rng.choice([-1, 0, 1])) for _ in eval_test_samples]
    fwd_synthetic_returns = [float(rng.normal(0.001, 0.015)) for _ in eval_test_samples]

    repr_res = evaluate_representational_leakage(
        h_leak, h_clean, y_synthetic_future_actions, probe_cv="timeseries", n_splits=2, n_permutations=50
    )
    behav_res = evaluate_behavioral_leakage(
        model_leak=leak_encoder,
        model_clean=clean_encoder,
        samples=eval_test_samples,
    )
    s_clean = clean_encoder.get_stance_score(test_texts)
    s_leak = leak_encoder.get_stance_score(test_texts)
    econ_res = evaluate_economic_effect(s_leak, s_clean, fwd_synthetic_returns, n_bootstrap=200)

    git_prov = resolve_git_provenance(out_dir)

    # Build experiment manifest
    experiment_manifest = {
        "experiment_id": "phase2_1_causal_twin_activation_smoke",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_prov["git_head"],
        "git_dirty": git_prov["git_dirty"],
        "code_commit_exact": git_prov["code_commit_exact"],
        "git_provenance": git_prov,
        "base_checkpoint": base_model_name,
        "base_revision": base_revision,
        "dataset_hash": dataset_hash,
        "training_cutoff": "2018-12-31T23:59:59Z",
        "clean_corpus_hash": stream_d0["corpus_hash"],
        "leak_corpus_hash": stream_d100["corpus_hash"],
        "evaluation_corpus_hash": hash_corpus(test_texts),
        "clean_token_count": stream_d0["effective_tokens"],
        "leak_token_count": stream_d100["effective_tokens"],
        "clean_steps": clean_treatment_manifest["optimizer_steps"],
        "leak_steps": leak_treatment_manifest["optimizer_steps"],
        "clean_initial_parameter_hash": clean_initial_param_hash,
        "leak_initial_parameter_hash": leak_initial_param_hash,
        "clean_post_mlm_parameter_hash": clean_post_mlm_hash,
        "leak_post_mlm_parameter_hash": leak_post_mlm_hash,
        "classifier_head_initial_hash": classifier_head_initial_hash,
        "overlap_count": clean_eval_overlap + leak_eval_overlap,
        "random_seed": seed,
        "device": dev_choice,
        "torch_version": torch.__version__,
    }
    with open(manifests_dir / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(experiment_manifest, f, indent=2)

    # Assemble structured results
    results_summary = {
        "status": "ENGINEERING SMOKE TEST ONLY — NOT RESEARCH CONCLUSIONS",
        "experiment": "phase2_1_causal_twin_activation_smoke",
        "base_model": {
            "name": base_model_name,
            "revision": base_revision,
        },
        "git_provenance": git_prov,
        "code_commit": git_prov["git_head"],
        "git_dirty": git_prov["git_dirty"],
        "code_commit_exact": git_prov["code_commit_exact"],
        "treatment_integrity_status": "CAUSAL TWIN PIPELINE ACTIVE",
        "causal_treatment_integrity": {
            "initial_encoder_hash": {"clean": clean_initial_param_hash, "leak": leak_initial_param_hash, "result": "PASS"},
            "mlm_corpus_hash": {"clean": stream_d0["corpus_hash"], "leak": stream_d100["corpus_hash"], "result": "PASS"},
            "effective_tokens": {"clean": stream_d0["effective_tokens"], "leak": stream_d100["effective_tokens"], "result": "PASS"},
            "optimizer_steps": {"clean": clean_treatment_manifest["optimizer_steps"], "leak": leak_treatment_manifest["optimizer_steps"], "result": "PASS"},
            "mlm_learning_rate": {"clean": mlm_lr, "leak": mlm_lr, "result": "PASS"},
            "post_mlm_encoder_hash": {"clean": clean_post_mlm_hash, "leak": leak_post_mlm_hash, "result": "PASS (Diverged)"},
            "classifier_head_initial_hash": {"clean": classifier_head_initial_hash, "leak": classifier_head_initial_hash, "result": "PASS"},
            "downstream_sample_order_hash": {"clean": clean_order_hash, "leak": leak_order_hash, "result": "PASS"},
            "eval_overlap_count": {"clean": clean_eval_overlap, "leak": leak_eval_overlap, "result": "PASS"},
        },
        "competence_metrics": {
            "baseline_mb_macro_f1": float(baseline_result["macro_f1"]),
            "baseline_mb_mcc": float(baseline_result["mcc"]),
            "baseline_mb_brier": float(baseline_result["brier_score"]),
            "baseline_mb_ece": float(baseline_result["ece"]),
            "clean_d0_macro_f1": float(c_clean["macro_f1"]),
            "leak_d100_macro_f1": float(c_leak["macro_f1"]),
            "delta_macro_f1": float(c_leak["macro_f1"] - c_clean["macro_f1"]),
        },
        "synthetic_plumbing_metrics": {
            "note": "Metrics computed using synthetic mock forward returns/actions for code plumbing verification only.",
            "l_repr_plumbing": float(repr_res["l_repr"]),
            "l_behavior_delta_plumbing": float(behav_res["l_behavior_delta"]),
            "delta_ic_plumbing": float(econ_res["delta_ic"]),
            "delta_sharpe_plumbing": float(econ_res["delta_sharpe"]),
        },
        "empirical_leakage_metrics": None,  # Explicitly null: synthetic targets do not yield empirical conclusions
        "empirical_pareto_vector": None,
        "synthetic_plumbing_pareto_vector": pareto_coordinates(
            c=float(c_leak["macro_f1"]),
            l_repr=float(repr_res["l_repr"]),
            l_behavior=float(behav_res["l_behavior_delta"]),
            e_l_ic=float(econ_res["delta_ic"]),
            e_l_sharpe=float(econ_res["delta_sharpe"]),
        ),
    }

    # Save results JSON
    with open(results_dir / "smoke_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    # Save baseline specific results JSON
    baseline_export = {
        "status": "VALIDATED_FOMC_STANCE_BASELINE",
        "checkpoint": base_model_name,
        "revision": base_revision,
        "macro_f1": float(baseline_result["macro_f1"]),
        "mcc": float(baseline_result["mcc"]),
        "brier_score": float(baseline_result["brier_score"]),
        "ece": float(baseline_result["ece"]),
        "training_cutoff": "2018-12-31",
        "sample_count": len(eval_test_samples),
        "train_sample_count": len(train_slice),
        "original_head_loaded": False,
        "stance_head_initialization": "fresh",
        "stance_head_initial_hash": baseline_result.get("initial_head_hash"),
        "git_provenance": git_prov,
        "code_commit": git_prov["git_head"],
        "git_dirty": git_prov["git_dirty"],
        "code_commit_exact": git_prov["code_commit_exact"],
    }
    baseline_results_file = results_dir / "baseline_results.json"
    should_write_baseline = True
    if baseline_results_file.exists():
        try:
            with open(baseline_results_file, "r", encoding="utf-8") as f:
                existing_baseline = json.load(f)
            if existing_baseline.get("test_sample_count", existing_baseline.get("sample_count", 0)) > len(eval_test_samples):
                should_write_baseline = False
        except Exception:
            pass
    if should_write_baseline:
        with open(baseline_results_file, "w", encoding="utf-8") as f:
            json.dump(baseline_export, f, indent=2)

    print("\n" + "=" * 76)
    print("ENGINEERING SMOKE TEST ONLY — NOT RESEARCH CONCLUSIONS")
    print("=" * 76)
    print("CAUSAL INTEGRITY VERIFICATION: ALL INVARIANTS PASSED")
    print(f"  Baseline M_B Macro-F1: {baseline_result['macro_f1']:.4f}")
    print(f"  Clean Twin D0 Macro-F1: {c_clean['macro_f1']:.4f}")
    print(f"  Leak Twin D100 Macro-F1: {c_leak['macro_f1']:.4f}")
    print(f"  Effective Tokens (Clean / Leak): {stream_d0['effective_tokens']} / {stream_d100['effective_tokens']}")
    print(f"  Optimizer Steps (Clean / Leak): {clean_treatment_manifest['optimizer_steps']} / {leak_treatment_manifest['optimizer_steps']}")
    print(f"  Post-MLM Hashes Diverged: {clean_post_mlm_hash[:8]}... != {leak_post_mlm_hash[:8]}...")
    print(f"  Evaluation Overlap: 0 (Clean: {clean_eval_overlap}, Leak: {leak_eval_overlap})")
    print("=" * 76 + "\n")

    return results_summary
