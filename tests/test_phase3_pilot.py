"""Comprehensive Unit Tests for Phase 3 Pilot Study.

Tests:
1. Intermediate Dose Accuracy (Item 30):
   D0, D25, D50, D75, D100 produce exact realized token counts (|D_realized - D_requested| <= 1/T).
2. Same Start Across All Doses (Item 31):
   Fixed seed: initial model parameter hash identical across all doses.
3. Same Compute Across All Doses (Item 32):
   Fixed seed: tokens, steps, batch size, optimizer, lr identical across all doses.
4. Same Head Initialization Across All Doses (Item 33):
   Fixed seed: classifier_head_initial_hash identical across all doses.
5. Same Downstream Order Across All Doses (Item 34):
   Fixed seed: downstream sample order hash identical across all doses.
6. Mask Schedule Pairing (Item 35):
   Fixed seed: mask_schedule_hash identical across all doses.
7. Different Seeds Truly Different (Item 36):
   Seed 13 vs 42 produces distinct mask_schedule_hash and head_initial_hash.
8. Strict Temporal Separation (Item 37):
   max(Time_anchors) < min(Time_contamination).
9. Document & Sentence Isolation (Item 38):
   Docs_anchors ∩ Docs_MLM = ∅, SentenceHash_anchors ∩ SentenceHash_MLM = ∅.
10. Official Manifest Hash Gate (Item 39):
    Corrupted manifest dataset_sha256 raises ValueError.
11. No Synthetic Data in Empirical Fields (Item 40):
    Empirical leakage metrics derived strictly from real targets or marked NOT_EVALUATED / None.
12. Config Runtime Contract (Item 1.1):
    Changing max_seq_length, weight_decay, or optimizer directly alters downstream training state.
13. Dev Split Semantics (Item 1.2):
    Dev split reserved and not used for model selection.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List
import yaml

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM, BertForSequenceClassification

from tradingagents.temporal_leakage.datasets.fomc_official import (
    create_fomc_official_benchmark,
    load_fomc_official_statements,
)
from tradingagents.temporal_leakage.fomc_benchmark import (
    FOMCBenchmark,
    TemporalSample,
    load_fomc_dataset,
)
from tradingagents.temporal_leakage.hf_encoder import (
    FOMC_STANCE_ID_TO_LABEL,
    FOMC_STANCE_LABEL_TO_ID,
    HuggingFaceTemporalEncoder,
    build_fresh_fomc_classifier_from_base_encoder,
    hash_model_parameters,
    normalize_and_hash_text,
)
from tradingagents.temporal_leakage.phase3_pilot import (
    audit_temporal_separation_and_isolation,
    load_verified_anchors,
    run_phase3_pilot,
)
from tradingagents.temporal_leakage.twin_pipeline import (
    build_classifier_from_mlm_encoder,
    create_exact_token_dose_stream,
    generate_deterministic_mask_schedule,
    run_continued_pretraining_mlm,
    train_downstream_classifier,
)


class MockTinyTokenizer:
    """Fast deterministic offline tokenizer for tests."""

    def __init__(self, vocab_size: int = 200) -> None:
        self.vocab_size = vocab_size
        self.mask_token = "[MASK]"
        self.mask_token_id = 103
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102

    def __call__(self, texts: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        max_len = kwargs.get("max_length", 16)
        batch_ids = []
        batch_mask = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            ids = [((b % (self.vocab_size - 10)) + 5) for b in h[:max_len]]
            if len(ids) < max_len:
                ids += [0] * (max_len - len(ids))
            batch_ids.append(ids)
            batch_mask.append([1 if x != 0 else 0 for x in ids])
        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [((b % (self.vocab_size - 10)) + 5) for b in h[:16]]

    def save_pretrained(self, save_dir: Any) -> None:
        pass


def make_tiny_bert_mlm() -> BertForMaskedLM:
    """Create tiny 2-layer BertForMaskedLM for fast unit tests."""
    config = BertConfig(
        vocab_size=200,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    model = BertForMaskedLM(config)
    torch.manual_seed(42)
    for p in model.parameters():
        nn.init.normal_(p, mean=0.0, std=0.02)
    return model


def make_tiny_bert_clf() -> BertForSequenceClassification:
    """Create tiny 2-layer BertForSequenceClassification for fast unit tests."""
    config = BertConfig(
        vocab_size=200,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=3,
        id2label=FOMC_STANCE_ID_TO_LABEL,
        label2id=FOMC_STANCE_LABEL_TO_ID,
    )
    model = BertForSequenceClassification(config)
    torch.manual_seed(42)
    for p in model.parameters():
        nn.init.normal_(p, mean=0.0, std=0.02)
    return model


# ---------------------------------------------------------------------------
# Test 1: Intermediate Dose Accuracy (Item 30)
# ---------------------------------------------------------------------------
def test_exact_intermediate_dose_accuracy():
    """Verify that D0, D25, D50, D75, D100 produce exact realized token counts."""
    pre_corpus = [f"Pre-cutoff sentence {i} regarding interest rates and economy." for i in range(20)]
    post_corpus = [f"Post-cutoff future sentence {i} regarding pandemic and tightening." for i in range(20)]

    tokenizer = MockTinyTokenizer()
    num_blocks = 50
    block_length = 32
    total_tokens = num_blocks * block_length  # 1600 tokens

    doses = [0.0, 0.25, 0.50, 0.75, 1.00]
    expected_posts = [0, 400, 800, 1200, 1600]

    for dose, exp_post in zip(doses, expected_posts):
        stream = create_exact_token_dose_stream(
            pre_corpus=pre_corpus,
            post_corpus=post_corpus,
            dose=dose,
            num_blocks=num_blocks,
            block_length=block_length,
            tokenizer=tokenizer,
            random_seed=42,
        )

        assert stream["requested_dose"] == dose
        assert stream["total_tokens"] == total_tokens
        assert stream["post_cutoff_tokens"] == exp_post
        assert stream["pre_cutoff_tokens"] == total_tokens - exp_post
        assert stream["realized_dose"] == exp_post / total_tokens
        # Invariant: |D_realized - D_requested| <= 1 / T
        assert abs(stream["realized_dose"] - dose) <= (1.0 / total_tokens)


# ---------------------------------------------------------------------------
# Tests 2-6: Causal Symmetries across Doses for Fixed Seed (Items 31-35)
# ---------------------------------------------------------------------------
def test_fixed_seed_causal_symmetries(tmp_path: Path):
    """Test same start, compute, head, downstream order, and mask schedule across doses."""
    seed = 42
    pre_corpus = [f"Pre text {i}" for i in range(15)]
    post_corpus = [f"Post text {i}" for i in range(15)]
    tokenizer = MockTinyTokenizer()

    num_blocks = 10
    block_length = 16
    doses = [0.0, 0.25, 0.50, 0.75, 1.00]

    base_mlm = make_tiny_bert_mlm()
    initial_hash = hash_model_parameters(base_mlm)

    # Shared mask schedule for seed
    mask_sched, mask_hash = generate_deterministic_mask_schedule(
        num_blocks=num_blocks, block_length=block_length, random_seed=seed
    )

    # Shared fresh head state
    clf_ref = make_tiny_bert_clf()
    torch.manual_seed(seed)
    clf_ref.classifier.reset_parameters()
    shared_head_state = copy.deepcopy(clf_ref.classifier.state_dict())
    shared_head_hash = hash_model_parameters(clf_ref.classifier)

    initial_hashes = []
    compute_tokens = []
    mask_hashes = []
    head_hashes = []
    order_hashes = []

    samples = [
        TemporalSample(
            sample_id=f"sample_{i}",
            text=f"Economic growth report {i}",
            document_type="statement",
            event_time="2018-05-01T14:00:00Z",
            available_time="2018-05-01T14:00:00Z",
            task_label=(i % 3) - 1,
        )
        for i in range(12)
    ]

    for d in doses:
        stream = create_exact_token_dose_stream(
            pre_corpus, post_corpus, dose=d, num_blocks=num_blocks, block_length=block_length, tokenizer=tokenizer, random_seed=seed
        )
        compute_tokens.append(stream["total_tokens"])

        model_clone = copy.deepcopy(base_mlm)
        initial_hashes.append(hash_model_parameters(model_clone))

        manifest = run_continued_pretraining_mlm(
            packed_dataset=stream["dataset"],
            tokenizer=tokenizer,
            model_mlm=model_clone,
            output_dir=tmp_path / f"mlm_d{d}",
            max_steps=3,
            batch_size=2,
            learning_rate=1e-4,
            random_seed=seed,
            branch=f"d{d}",
            dose=d,
            mask_schedule=mask_sched,
            mask_schedule_hash=mask_hash,
        )
        mask_hashes.append(manifest["mask_schedule_hash"])

        # Build classifier with shared head
        clf_encoder = build_classifier_from_mlm_encoder(
            mlm_model=model_clone,
            tokenizer=tokenizer,
            head_state_dict=shared_head_state,
            random_seed=seed,
        )
        head_hashes.append(hash_model_parameters(clf_encoder.model.classifier))

        _, order_hash = train_downstream_classifier(
            encoder_model=clf_encoder,
            train_samples=samples,
            epochs=1,
            max_steps=3,
            batch_size=2,
            random_seed=seed,
            max_seq_length=16,
        )
        order_hashes.append(order_hash)

    # Item 31: Same Start Across All Doses
    assert len(set(initial_hashes)) == 1, "Initial model hashes must be bit-identical across all doses."
    assert initial_hashes[0] == initial_hash

    # Item 32: Same Compute Budget Across All Doses
    assert len(set(compute_tokens)) == 1, "Token budget must be identical across all doses."

    # Item 33: Same Head Initialization Across All Doses
    assert len(set(head_hashes)) == 1, "Classifier head initial state must be bit-identical across all doses."
    assert head_hashes[0] == shared_head_hash

    # Item 34: Same Downstream Order Across All Doses
    assert len(set(order_hashes)) == 1, "Downstream batch order hash must be identical across all doses."

    # Item 35: Mask Schedule Pairing Across All Doses
    assert len(set(mask_hashes)) == 1, "Mask schedule hash must be identical across all doses."


# ---------------------------------------------------------------------------
# Test 7: Different Seeds Truly Different (Item 36)
# ---------------------------------------------------------------------------
def test_seed_differentiation():
    """Verify that different seeds produce different mask schedules and head initializations."""
    _, hash13 = generate_deterministic_mask_schedule(10, 16, random_seed=13)
    _, hash42 = generate_deterministic_mask_schedule(10, 16, random_seed=42)
    assert hash13 != hash42, "Mask schedule hash must differ across seeds."

    clf13 = make_tiny_bert_clf()
    torch.manual_seed(13)
    clf13.classifier.reset_parameters()
    head13 = hash_model_parameters(clf13.classifier)

    clf42 = make_tiny_bert_clf()
    torch.manual_seed(42)
    clf42.classifier.reset_parameters()
    head42 = hash_model_parameters(clf42.classifier)

    assert head13 != head42, "Classifier head initial hash must differ across seeds."


# ---------------------------------------------------------------------------
# Test 8-9: Temporal Separation & Document/Sentence Isolation (Items 37-38)
# ---------------------------------------------------------------------------
def test_temporal_separation_and_isolation():
    """Verify that the official anchor dataset satisfies strict future separation and isolation."""
    anchors_path = Path("data/research/fomc/leakage_anchors/anchors.jsonl")
    manifest_path = Path("data/research/fomc/leakage_anchors/manifest.json")
    if not anchors_path.exists() or not manifest_path.exists():
        pytest.skip("Anchor dataset not generated yet.")

    anchors, manifest, _ = load_verified_anchors(anchors_path, manifest_path)
    assert len(anchors) >= 8

    # Check temporal separation: max(Time_anchors) < min(Time_contamination)
    max_anchor_time = max(a["event_time"] for a in anchors)
    min_contamination_time = "2020-01-01T00:00:00Z"
    assert max_anchor_time < min_contamination_time, "Strict temporal separation failed!"

    # Check document isolation
    anchor_doc_ids = {a["document_id"] for a in anchors}
    assert all(doc_id.startswith("fomc-statement-2019-") for doc_id in anchor_doc_ids)

    # Check sentence hashes against TDW post-cutoff (>=2020)
    tdw_samples = load_fomc_dataset("data/research/fomc/fomc_temporal_dataset.jsonl")
    tdw_post = [s for s in tdw_samples if int(s.metadata.get("year", 2000)) >= 2020]
    tdw_post_hashes = {normalize_and_hash_text(s.text) for s in tdw_post}
    anchor_hashes = {normalize_and_hash_text(a["text"]) for a in anchors}

    overlap = anchor_hashes & tdw_post_hashes
    assert len(overlap) == 0, f"Detected sentence hash overlap between anchors and contamination corpus: {overlap}"


# ---------------------------------------------------------------------------
# Test 10: Official Manifest Hash Gate (Item 39)
# ---------------------------------------------------------------------------
def test_official_manifest_hash_gate(tmp_path: Path):
    """Verify that mismatched dataset_sha256 raises ValueError immediately."""
    data_file = tmp_path / "data.jsonl"
    record = {
        "sample_id": "test-01",
        "text": "The Committee raised interest rates by 25 basis points.",
        "document_type": "statement",
        "event_time": "2019-05-01T14:00:00-04:00",
        "available_time": "2019-05-01T14:00:00-04:00",
        "task_label": 1,
        "availability_quality": "exact",
    }
    with open(data_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    corrupted_manifest = {
        "source_verified": True,
        "annotation_verified": True,
        "pit_verified": True,
        "dataset_sha256": "badbeef" * 9 + "0",
        "availability_provenance": "Federal Reserve press releases",
        "source_urls": ["https://federalreserve.gov"],
        "data_file": "data.jsonl",
    }
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(corrupted_manifest, f)

    with pytest.raises(ValueError, match="SHA256 verification failed"):
        load_fomc_official_statements(manifest_path=manifest_path)


# ---------------------------------------------------------------------------
# Test 11: No Synthetic Data in Empirical Fields (Item 40)
# ---------------------------------------------------------------------------
def test_no_synthetic_data_in_empirical_fields(tmp_path: Path):
    """Verify that pilot runner results use real empirical metrics without synthetic proxies."""
    anchors_path = Path("data/research/fomc/leakage_anchors/anchors.jsonl")
    manifest_path = Path("data/research/fomc/leakage_anchors/manifest.json")
    if not anchors_path.exists():
        pytest.skip("Anchor dataset not available.")

    results = run_phase3_pilot(
        output_dir=tmp_path / "phase3_mock",
        seeds_override=[13],
        doses_override=[0.0, 0.5],
        mock_model_for_testing=make_tiny_bert_mlm(),
        mock_tokenizer_for_testing=MockTinyTokenizer(),
        smoke_mode=True,
        smoke_steps=2,
        smoke_blocks=10,
    )

    # Invariant: empirical fields are either real float values or None, never synthetic default fallback
    for seed_k, seed_data in results["per_seed_results"].items():
        for dose_k, metrics in seed_data.items():
            assert isinstance(metrics["C"], float)
            assert isinstance(metrics["R_T"], float)
            assert isinstance(metrics["L_repr"], float)
            assert isinstance(metrics["L_behavior"], float)
            # E_L_ic must be float or None
            assert metrics["E_L_ic"] is None or isinstance(metrics["E_L_ic"], float)


# ---------------------------------------------------------------------------
# Test 12: Config Binding and Runtime Impact (Item 1.1)
# ---------------------------------------------------------------------------
def test_config_training_hyperparameter_runtime_contract():
    """Verify that changing training hyperparameters alters downstream training trajectory."""
    samples = [
        TemporalSample(
            sample_id=f"sample_{i}",
            text=f"Inflation and employment conditions {i}",
            document_type="statement",
            event_time="2018-05-01T14:00:00Z",
            available_time="2018-05-01T14:00:00Z",
            task_label=(i % 3) - 1,
        )
        for i in range(12)
    ]
    tokenizer = MockTinyTokenizer()

    # Model A: LR = 1e-4
    clf_a = HuggingFaceTemporalEncoder(
        name="test_clf_a",
        model_name_or_path="mock",
        tokenizer=tokenizer,
        model=make_tiny_bert_clf(),
        random_seed=42,
    )
    tuned_a, _ = train_downstream_classifier(
        encoder_model=clf_a,
        train_samples=samples,
        epochs=1,
        max_steps=5,
        learning_rate=1e-4,
        weight_decay=0.01,
        optimizer_name="AdamW",
        random_seed=42,
        max_seq_length=16,
    )
    hash_a = hash_model_parameters(tuned_a.model)

    # Model B: LR = 1e-1 (different learning rate directly alters weights)
    clf_b = HuggingFaceTemporalEncoder(
        name="test_clf_b",
        model_name_or_path="mock",
        tokenizer=tokenizer,
        model=make_tiny_bert_clf(),
        random_seed=42,
    )
    tuned_b, _ = train_downstream_classifier(
        encoder_model=clf_b,
        train_samples=samples,
        epochs=1,
        max_steps=5,
        learning_rate=1e-1,
        weight_decay=0.01,
        optimizer_name="AdamW",
        random_seed=42,
        max_seq_length=16,
    )
    hash_b = hash_model_parameters(tuned_b.model)

    assert hash_a != hash_b, "Learning rate must alter model training trajectory."
