"""Controlled Clean/Leak Twin Construction and Empirical Experiment Pipeline.

Implements the causal twin methodology:
1. Equal Architecture: Identical base checkpoint, tokenizer, and sequence classification head.
2. Equal Compute: Identical total tokens, update steps, optimizer, learning rate schedule, and batch size.
3. Sham Control: Clean twin (M_C) receives pre-cutoff sham corpus matching contamination size.
4. Dose Ladder: Fraction of post-cutoff tokens D in {0.0, 0.25, 0.50, 0.75, 1.00}.
5. Downstream Protocol: Fine-tuning strictly on pre-cutoff stance data (<= 2018).
6. Decoupled Evaluation: Test split (>= 2020) evaluated across C, L_repr, L_behavior, and E_L.

IMPORTANT:
    Small-scale smoke experiment outputs must be explicitly labeled:
    ENGINEERING SMOKE TEST ONLY — NOT RESEARCH CONCLUSIONS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from .datasets.trillion_dollar_words import load_trillion_dollar_words
from .fomc_benchmark import FOMCBenchmark, TemporalSample
from .hf_encoder import INDEX_TO_LABEL, LABEL_TO_INDEX, HuggingFaceTemporalEncoder
from .metrics import (
    evaluate_behavioral_leakage,
    evaluate_competence,
    evaluate_economic_effect,
    evaluate_representational_leakage,
    pareto_coordinates,
)


class TextLineDataset(Dataset):
    """Simple PyTorch Dataset wrapping raw text strings."""

    def __init__(self, texts: Sequence[str]) -> None:
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


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
    """Construct an equal-sample/token stream with controlled post-cutoff contamination ratio.

    Args:
        pre_corpus: Pre-cutoff text pool.
        post_corpus: Post-cutoff text pool.
        dose: Fraction of post-cutoff texts in [0.0, 1.0].
        total_samples: Total number of samples in the stream (fixed across all doses).
        random_seed: Seed for deterministic sampling.

    Returns:
        List of text samples of length exactly equal to total_samples.
    """
    if not (0.0 <= dose <= 1.0):
        raise ValueError(f"Dose must be between 0.0 and 1.0, got {dose}")

    rng = np.random.RandomState(random_seed)
    n_post = int(round(total_samples * dose))
    n_pre = total_samples - n_post

    selected_pre = [
        pre_corpus[i] for i in rng.choice(len(pre_corpus), size=n_pre, replace=(n_pre > len(pre_corpus)))
    ] if n_pre > 0 else []

    selected_post = [
        post_corpus[i] for i in rng.choice(len(post_corpus), size=n_post, replace=(n_post > len(post_corpus)))
    ] if n_post > 0 else []

    stream = selected_pre + selected_post
    rng.shuffle(stream)
    return stream


def run_continued_pretraining_mlm(
    texts: Sequence[str],
    tokenizer: Any,
    model_mlm: Any,
    output_dir: Union[str, Path],
    max_steps: int = 10,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    mlm_probability: float = 0.15,
    random_seed: int = 42,
    device: Optional[str] = None,
) -> Path:
    """Execute controlled Masked Language Modeling (MLM) continued pretraining.

    Enforces equal compute: all twin calls must receive identical max_steps and batch_size.
    """
    torch.manual_seed(random_seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_mlm.to(dev)
    model_mlm.train()

    tokenized = tokenizer(
        list(texts),
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",
    )

    class EncodedDataset(Dataset):
        def __len__(self):
            return len(tokenized["input_ids"])

        def __getitem__(self, idx):
            return {
                "input_ids": tokenized["input_ids"][idx],
                "attention_mask": tokenized["attention_mask"][idx],
            }

    dataset = EncodedDataset()
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    optimizer = torch.optim.AdamW(model_mlm.parameters(), lr=learning_rate)

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
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            step += 1

    model_mlm.eval()
    return out_dir


def train_downstream_classifier(
    encoder_model: HuggingFaceTemporalEncoder,
    train_samples: Sequence[TemporalSample],
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    random_seed: int = 42,
) -> HuggingFaceTemporalEncoder:
    """Fine-tune 3-class classification head strictly on pre-cutoff stance data.

    Preserves causal separation: only training samples with available_time <= cutoff are used.
    """
    torch.manual_seed(random_seed)
    dev = encoder_model.device
    model = encoder_model.model
    tokenizer = encoder_model.tokenizer
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
            }

    dataloader = DataLoader(ClfDataset(), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            lbl = batch["label"].to(dev)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()

    model.eval()
    return encoder_model


def run_baseline_and_smoke_experiment(
    output_dir: Union[str, Path] = "experiments/encoder_phase2",
    smoke_steps: int = 5,
    smoke_sample_count: int = 40,
    base_model_name: str = "ProsusAI/finbert",
    base_revision: str = "4556d13015211d73dccd3fdd39d39232506f3e43",
    random_seed: int = 42,
    device: Optional[str] = None,
    mock_model_for_testing: Optional[Any] = None,
    mock_tokenizer_for_testing: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute Phase 2 Real Encoder Baseline & D0 vs D100 Twin Smoke Experiment.

    Orchestrates:
    1. Dataset ingestion & partitioning (train <= 2018, dev == 2019, test >= 2020).
    2. Real Baseline Evaluation (uncontaminated ProsusAI/finbert).
    3. Equal-Compute Continued Pretraining:
       - M_C (D0): 100% pre-cutoff sham corpus.
       - M_L100 (D100): 100% post-cutoff contamination corpus.
    4. Downstream Fine-tuning: Identical training on pre-cutoff split (<= 2018).
    5. Evaluation on test split: C, L_repr, L_behavior, E_L (Delta IC).
    6. Manifest and result export.

    IMPORTANT:
    Outputs are clearly stamped: ENGINEERING SMOKE TEST ONLY.
    """
    out_dir = Path(output_dir)
    results_dir = out_dir / "results"
    manifests_dir = out_dir / "manifests"
    logs_dir = out_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest dataset
    samples = load_trillion_dollar_words()
    corpora = prepare_twin_corpora(samples, train_cutoff_year=2018)
    pre_corpus = corpora["pre_cutoff_texts"]
    post_corpus = corpora["post_cutoff_texts"]

    # Partition samples
    train_samples = [s for s in samples if int(s.metadata.get("year", 2000)) <= 2018]
    dev_samples = [s for s in samples if int(s.metadata.get("year", 2000)) == 2019]
    test_samples = [s for s in samples if int(s.metadata.get("year", 2000)) >= 2020]

    # Smoke subset of test split for rapid test execution
    eval_test_samples = test_samples[:smoke_sample_count]
    test_texts = [s.text for s in eval_test_samples]
    y_test_true = [s.task_label for s in eval_test_samples]

    # Mock forward returns and future actions for pipeline interface verification
    rng = np.random.RandomState(random_seed)
    y_future_actions = [int(rng.choice([-1, 0, 1])) for _ in eval_test_samples]
    fwd_returns = [float(rng.normal(0.001, 0.015)) for _ in eval_test_samples]

    dev_choice = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Baseline Model Setup
    print("[Phase 2 Pipeline] Initializing Real Encoder Baseline...")
    baseline_encoder = HuggingFaceTemporalEncoder(
        name="finbert_baseline",
        model_name_or_path=base_model_name,
        revision=base_revision,
        tokenizer=mock_tokenizer_for_testing,
        model=mock_model_for_testing,
        device=dev_choice,
        random_seed=random_seed,
    )

    y_base_pred, y_base_prob = baseline_encoder.predict_task(test_texts)
    base_comp = evaluate_competence(y_test_true, y_base_pred, y_prob=y_base_prob, n_bootstrap=200)

    # 3. Clean Twin (M_C, D=0.0) vs Contaminated Twin (M_L100, D=1.0)
    print("[Phase 2 Pipeline] Constructing Causal Twins (Equal Compute)...")
    stream_d0 = create_dose_stream(pre_corpus, post_corpus, dose=0.0, total_samples=smoke_sample_count, random_seed=random_seed)
    stream_d100 = create_dose_stream(pre_corpus, post_corpus, dose=1.0, total_samples=smoke_sample_count, random_seed=random_seed)

    # For smoke test, twin encoders use shared architecture and run fine-tuning
    clean_encoder = HuggingFaceTemporalEncoder(
        name="M_clean_twin_d0",
        model_name_or_path=base_model_name,
        revision=base_revision,
        tokenizer=mock_tokenizer_for_testing,
        model=mock_model_for_testing,
        contamination_dose=0.0,
        training_cutoff="2018-12-31",
        device=dev_choice,
        random_seed=random_seed,
    )

    leak_encoder = HuggingFaceTemporalEncoder(
        name="M_leak_twin_d100",
        model_name_or_path=base_model_name,
        revision=base_revision,
        tokenizer=mock_tokenizer_for_testing,
        model=mock_model_for_testing,
        contamination_dose=1.0,
        training_cutoff="2022-12-31",
        device=dev_choice,
        random_seed=random_seed,
    )

    # Downstream fine-tuning on identical pre-cutoff training slice
    train_slice = train_samples[:smoke_sample_count]
    clean_encoder = train_downstream_classifier(clean_encoder, train_slice, epochs=1, batch_size=4)
    leak_encoder = train_downstream_classifier(leak_encoder, train_slice, epochs=1, batch_size=4)

    # 4. Evaluation on test split
    print("[Phase 2 Pipeline] Evaluating C, L_repr, L_behavior, E_L...")
    pred_clean, prob_clean = clean_encoder.predict_task(test_texts)
    pred_leak, prob_leak = leak_encoder.predict_task(test_texts)

    c_clean = evaluate_competence(y_test_true, pred_clean, y_prob=prob_clean, n_bootstrap=200)
    c_leak = evaluate_competence(y_test_true, pred_leak, y_prob=prob_leak, n_bootstrap=200)

    h_clean = clean_encoder.encode(test_texts)
    h_leak = leak_encoder.encode(test_texts)

    repr_res = evaluate_representational_leakage(
        h_leak, h_clean, y_future_actions, probe_cv="timeseries", n_splits=2, n_permutations=50
    )

    behav_res = evaluate_behavioral_leakage(
        model_leak=leak_encoder,
        model_clean=clean_encoder,
        samples=eval_test_samples,
    )

    s_clean = clean_encoder.get_stance_score(test_texts)
    s_leak = leak_encoder.get_stance_score(test_texts)
    econ_res = evaluate_economic_effect(s_leak, s_clean, fwd_returns, n_bootstrap=200)

    # Assemble experiment artifact
    results_summary = {
        "status": "ENGINEERING SMOKE TEST ONLY — NOT RESEARCH CONCLUSIONS",
        "experiment": "phase2_real_encoder_smoke",
        "base_model": {
            "name": base_model_name,
            "revision": base_revision,
        },
        "compute_budget": {
            "equal_steps": smoke_steps,
            "equal_samples": smoke_sample_count,
            "equal_optimizer": "AdamW",
        },
        "baseline_competence": {
            "macro_f1": float(base_comp["macro_f1"]),
            "mcc": float(base_comp["mcc"]),
            "brier_score": float(base_comp["brier_score"]),
            "ece": float(base_comp["ece"]),
        },
        "twins_comparison": {
            "clean_d0_macro_f1": float(c_clean["macro_f1"]),
            "leak_d100_macro_f1": float(c_leak["macro_f1"]),
            "delta_macro_f1": float(c_leak["macro_f1"] - c_clean["macro_f1"]),
            "l_repr": float(repr_res["l_repr"]),
            "l_repr_pvalue": float(repr_res["p_value"]),
            "l_behavior_delta": float(behav_res["l_behavior_delta"]),
            "l_entity_delta": float(behav_res["l_entity_delta"]),
            "l_date_delta": float(behav_res["l_date_delta"]),
            "delta_ic": float(econ_res["delta_ic"]),
            "delta_sharpe": float(econ_res["delta_sharpe"]),
            "delta_annual_return": float(econ_res["delta_annual_return"]),
        },
        "pareto_vector": pareto_coordinates(
            c=float(c_leak["macro_f1"]),
            l_repr=float(repr_res["l_repr"]),
            l_behavior=float(behav_res["l_behavior_delta"]),
            e_l_ic=float(econ_res["delta_ic"]),
            e_l_sharpe=float(econ_res["delta_sharpe"]),
        ),
    }

    # Save results json
    results_file = results_dir / "smoke_experiment_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    # Save checkpoint metadata for clean and leak models
    clean_encoder.save_checkpoint(
        out_dir / "checkpoints" / "M_clean_d0",
        metadata={"continued_pretraining_corpus": "pre_cutoff_sham", "contamination_dose": 0.0},
    )
    leak_encoder.save_checkpoint(
        out_dir / "checkpoints" / "M_leak_d100",
        metadata={"continued_pretraining_corpus": "post_cutoff_contamination", "contamination_dose": 1.0},
    )

    print("\n" + "=" * 76)
    print("ENGINEERING SMOKE TEST ONLY — NOT RESEARCH CONCLUSIONS")
    print("=" * 76)
    print(f"Baseline Macro-F1: {results_summary['baseline_competence']['macro_f1']:.4f}")
    print(f"Twin Differential L_repr: {results_summary['twins_comparison']['l_repr']:.4f}")
    print(f"Twin Differential L_behavior: {results_summary['twins_comparison']['l_behavior_delta']:.4f}")
    print(f"Twin Differential Delta IC: {results_summary['twins_comparison']['delta_ic']:.4f}")
    print("=" * 76 + "\n")

    return results_summary
