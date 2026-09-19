"""Unit tests for Phase 2.1 Causal Twin Activation & Baseline Correction.

Verifies:
1. Orchestrator explicitly invokes `run_continued_pretraining_mlm` for both Clean (D0) and Leak (D100).
2. Parameter treatment divergence (clean != leak, clean != base, leak != base).
3. MLM weights correctly transfer to classification model.
4. Classification heads are initialized bit-identically across twins.
5. Downstream training sample batch ordering is identical.
6. Exact token budget equality (0% difference between D0 and D100).
7. Evaluation isolation (zero overlap between MLM corpus and test split).
8. Prosus native financial sentiment head is strictly rejected/discarded for FOMC stance.
9. Official hardcoded fixture is strictly marked not formal-research-ready.
10. Causal integrity assertions raise CausalIntegrityError on any invariant violation.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM, BertForSequenceClassification

from tradingagents.temporal_leakage.datasets.fomc_official import (
    create_fomc_official_benchmark,
    create_fomc_official_fixture,
    load_fomc_official_statements,
)
from tradingagents.temporal_leakage.fomc_benchmark import TemporalSample
from tradingagents.temporal_leakage.hf_encoder import (
    FOMC_STANCE_ID_TO_LABEL,
    FOMC_STANCE_LABEL_TO_ID,
    HuggingFaceTemporalEncoder,
    hash_corpus,
    hash_model_parameters,
    normalize_and_hash_text,
    resolve_git_commit,
)
from tradingagents.temporal_leakage.twin_pipeline import (
    CausalIntegrityError,
    PackedTokenDataset,
    build_classifier_from_mlm_encoder,
    create_token_matched_dose_stream,
    run_baseline_and_smoke_experiment,
    run_continued_pretraining_mlm,
    train_baseline_stance_model,
    train_downstream_classifier,
)


class MockTinyTokenizer:
    """Fast deterministic offline tokenizer for tests without Hugging Face downloads."""

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
            # Deterministic token ids via SHA-256
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
    return BertForMaskedLM(config)


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
    return BertForSequenceClassification(config)


def test_1_orchestrator_calls_run_continued_pretraining_mlm(tmp_path: Path):
    """Test 1: Orchestrator explicitly invokes run_continued_pretraining_mlm for both Clean and Leak."""
    tiny_model = make_tiny_bert_mlm()
    tiny_tok = MockTinyTokenizer()

    with patch(
        "tradingagents.temporal_leakage.twin_pipeline.run_continued_pretraining_mlm"
    ) as mock_mlm:
        # Configure mock return manifest
        def fake_mlm(*args, **kwargs):
            branch = kwargs.get("branch", "clean")
            return {
                "branch": branch,
                "dose": kwargs.get("dose", 0.0),
                "base_checkpoint": "mock",
                "base_revision": "mock",
                "initial_parameter_hash": "base_hash_123",
                "final_parameter_hash": f"post_hash_{branch}_456",
                "corpus_hash": kwargs.get("corpus_hash", "corp_hash"),
                "token_count": 256,
                "optimizer_steps": 2,
                "batch_size": 2,
                "max_seq_length": 128,
                "mlm_probability": 0.15,
                "learning_rate": 5e-5,
                "seed": 42,
            }

        mock_mlm.side_effect = fake_mlm

        res = run_baseline_and_smoke_experiment(
            output_dir=tmp_path / "exp_mock",
            smoke_steps=2,
            smoke_sample_count=8,
            random_seed=42,
            mock_model_for_testing=tiny_model,
            mock_tokenizer_for_testing=tiny_tok,
        )

        # Assert called exactly twice: once clean (D0), once leak (D100)
        assert mock_mlm.call_count == 2
        calls = mock_mlm.call_args_list

        clean_call = calls[0].kwargs
        leak_call = calls[1].kwargs

        assert clean_call["branch"] == "clean"
        assert clean_call["dose"] == 0.0
        assert leak_call["branch"] == "leak"
        assert leak_call["dose"] == 1.0

        # Equal compute: same steps and lr
        assert clean_call["max_steps"] == leak_call["max_steps"]
        assert clean_call["learning_rate"] == leak_call["learning_rate"]
        # Different corpora
        assert clean_call["corpus_hash"] != leak_call["corpus_hash"]


def test_2_parameter_treatment_divergence(tmp_path: Path):
    """Test 2: Two identical base models diverge in parameter space after differential MLM treatment."""
    base_model = make_tiny_bert_mlm()
    tok = MockTinyTokenizer()

    clean_model = copy.deepcopy(base_model)
    leak_model = copy.deepcopy(base_model)

    assert clean_model is not leak_model
    hash_base = hash_model_parameters(base_model)
    hash_clean_pre = hash_model_parameters(clean_model)
    hash_leak_pre = hash_model_parameters(leak_model)

    assert hash_clean_pre == hash_leak_pre == hash_base

    # Create two different token datasets
    clean_ids = torch.randint(5, 50, (4, 32), dtype=torch.long)
    leak_ids = torch.randint(50, 100, (4, 32), dtype=torch.long)
    mask = torch.ones((4, 32), dtype=torch.long)

    ds_clean = PackedTokenDataset(clean_ids, mask)
    ds_leak = PackedTokenDataset(leak_ids, mask)

    run_continued_pretraining_mlm(
        packed_dataset=ds_clean,
        tokenizer=tok,
        model_mlm=clean_model,
        output_dir=tmp_path / "clean_mlm",
        max_steps=5,
        batch_size=2,
        learning_rate=1e-3,
        branch="clean",
        dose=0.0,
    )

    run_continued_pretraining_mlm(
        packed_dataset=ds_leak,
        tokenizer=tok,
        model_mlm=leak_model,
        output_dir=tmp_path / "leak_mlm",
        max_steps=5,
        batch_size=2,
        learning_rate=1e-3,
        branch="leak",
        dose=1.0,
    )

    hash_clean_post = hash_model_parameters(clean_model)
    hash_leak_post = hash_model_parameters(leak_model)

    assert hash_clean_post != hash_leak_post
    assert hash_clean_post != hash_base
    assert hash_leak_post != hash_base


def test_3_mlm_weights_transferred_to_classifier():
    """Test 3: build_classifier_from_mlm_encoder correctly injects MLM body into classifier."""
    mlm_model = make_tiny_bert_mlm()
    # Modify a parameter in mlm_model.bert to create a distinctive signature
    with torch.no_grad():
        mlm_model.bert.embeddings.word_embeddings.weight.add_(1.5)

    mlm_bert_hash = hash_model_parameters(mlm_model.bert)
    tok = MockTinyTokenizer()

    clf_encoder = build_classifier_from_mlm_encoder(
        mlm_model=mlm_model,
        tokenizer=tok,
        base_model_name="mock",
        base_revision="mock",
    )

    clf_bert = getattr(clf_encoder.model, "bert", None)
    assert clf_bert is not None
    mlm_encoder_hash = hash_model_parameters(mlm_model.bert.encoder)
    clf_encoder_hash = hash_model_parameters(clf_bert.encoder)

    assert clf_encoder_hash == mlm_encoder_hash

    # Verify every weight in mlm_model.bert matches in clf_bert
    mlm_state = mlm_model.bert.state_dict()
    clf_state = clf_bert.state_dict()
    for k, v in mlm_state.items():
        assert torch.equal(v.cpu(), clf_state[k].cpu()), f"Parameter mismatch for key: {k}"


def test_4_classification_head_initialization_identical():
    """Test 4: Clean and Leak classifiers start fine-tuning with bit-identical classification heads."""
    mlm_clean = make_tiny_bert_mlm()
    mlm_leak = make_tiny_bert_mlm()
    tok = MockTinyTokenizer()

    # Pre-generate a shared head state dict
    head = nn.Linear(32, 3)
    shared_head_state = copy.deepcopy(head.state_dict())

    clean_clf = build_classifier_from_mlm_encoder(
        mlm_model=mlm_clean,
        tokenizer=tok,
        head_state_dict=shared_head_state,
    )
    leak_clf = build_classifier_from_mlm_encoder(
        mlm_model=mlm_leak,
        tokenizer=tok,
        head_state_dict=shared_head_state,
    )

    clean_head_hash = hash_model_parameters(clean_clf.model.classifier)
    leak_head_hash = hash_model_parameters(leak_clf.model.classifier)

    assert clean_head_hash == leak_head_hash


def test_5_downstream_data_ordering_identical():
    """Test 5: Downstream fine-tuning enforces deterministic identical batch sample ordering."""
    model_a = HuggingFaceTemporalEncoder(
        name="model_a",
        tokenizer=MockTinyTokenizer(),
        model=make_tiny_bert_clf(),
    )
    model_b = HuggingFaceTemporalEncoder(
        name="model_b",
        tokenizer=MockTinyTokenizer(),
        model=make_tiny_bert_clf(),
    )

    samples = [
        TemporalSample(
            sample_id=f"sample-{i}",
            text=f"Policy sentence number {i}",
            document_type="statement",
            event_time="2018-01-01T00:00:00Z",
            available_time="2018-01-01T00:00:00Z",
            task_label=(i % 3) - 1,
            metadata={"year": 2018},
        )
        for i in range(12)
    ]

    _, order_hash_a = train_downstream_classifier(
        encoder_model=model_a,
        train_samples=samples,
        epochs=2,
        batch_size=3,
        random_seed=42,
    )

    _, order_hash_b = train_downstream_classifier(
        encoder_model=model_b,
        train_samples=samples,
        epochs=2,
        batch_size=3,
        random_seed=42,
    )

    assert order_hash_a == order_hash_b


def test_6_token_budget_exact_equality():
    """Test 6: create_token_matched_dose_stream enforces exact 0% token count difference."""
    pre_corpus = [f"Economic conditions were expansionary in meeting {i}" for i in range(20)]
    post_corpus = [f"Inflation pressures increased substantially in session {j}" for j in range(20)]
    tok = MockTinyTokenizer()

    num_blocks = 15
    block_length = 32
    expected_tokens = num_blocks * block_length

    d0_stream = create_token_matched_dose_stream(
        pre_corpus=pre_corpus,
        post_corpus=post_corpus,
        dose=0.0,
        num_blocks=num_blocks,
        block_length=block_length,
        tokenizer=tok,
        random_seed=42,
    )

    d100_stream = create_token_matched_dose_stream(
        pre_corpus=pre_corpus,
        post_corpus=post_corpus,
        dose=1.0,
        num_blocks=num_blocks,
        block_length=block_length,
        tokenizer=tok,
        random_seed=42,
    )

    assert d0_stream["effective_tokens"] == expected_tokens
    assert d100_stream["effective_tokens"] == expected_tokens
    assert d0_stream["effective_tokens"] == d100_stream["effective_tokens"]
    assert d0_stream["dataset"].input_ids.shape == (num_blocks, block_length)
    assert d100_stream["dataset"].input_ids.shape == (num_blocks, block_length)


def test_7_evaluation_isolation_zero_overlap():
    """Test 7: Verification that evaluation samples are excluded from post-cutoff MLM pool."""
    eval_texts = ["Sentence A from test split", "Sentence B from test split"]
    eval_hashes = {normalize_and_hash_text(t) for t in eval_texts}

    # Post corpus contains Sentence A along with other sentences
    post_corpus = [
        "Sentence A from test split",
        "Unique post-cutoff sentence 1",
        "Unique post-cutoff sentence 2",
    ]

    # Enforce isolation filter
    post_isolated = [t for t in post_corpus if normalize_and_hash_text(t) not in eval_hashes]

    tok = MockTinyTokenizer()
    stream = create_token_matched_dose_stream(
        pre_corpus=["Pre 1", "Pre 2"],
        post_corpus=post_isolated,
        dose=1.0,
        num_blocks=2,
        block_length=16,
        tokenizer=tok,
    )

    stream_hashes = {normalize_and_hash_text(t) for t in stream["texts_used"]}
    overlap = stream_hashes & eval_hashes
    assert len(overlap) == 0


def test_8_prosus_native_sentiment_head_rejection():
    """Test 8: Native ProsusAI financial sentiment head cannot be reused for FOMC stance schema."""
    # Create model with sentiment id2label config
    config = BertConfig(
        vocab_size=200,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=3,
        id2label={0: "positive", 1: "negative", 2: "neutral"},
        label2id={"positive": 0, "negative": 1, "neutral": 2},
    )
    sentiment_model = BertForSequenceClassification(config)

    # Attempting to reuse existing sentiment head raises ValueError
    with pytest.raises(ValueError, match="has financial sentiment labels"):
        HuggingFaceTemporalEncoder(
            model=sentiment_model,
            tokenizer=MockTinyTokenizer(),
            task_label_schema="fomc_stance",
            reuse_existing_head=True,
        )

    # When reuse_existing_head=False, sentiment head is discarded and fresh FOMC stance head is built
    adapter = HuggingFaceTemporalEncoder(
        model=sentiment_model,
        tokenizer=MockTinyTokenizer(),
        task_label_schema="fomc_stance",
        reuse_existing_head=False,
    )
    assert adapter.model.config.id2label[0] == "Dovish"
    assert adapter.model.config.id2label[1] == "Neutral"
    assert adapter.model.config.id2label[2] == "Hawkish"


def test_9_official_hardcoded_fixture_not_formal_ready():
    """Test 9: Hardcoded official fixture is strictly not formal research ready."""
    fixture = create_fomc_official_fixture()
    assert fixture.is_formal_research_ready() is False

    with pytest.raises(ValueError, match="requires an explicit verified data file"):
        load_fomc_official_statements(filepath=None)

    with pytest.raises(ValueError, match="requires an explicit verified data file"):
        create_fomc_official_benchmark(filepath=None)


def test_10_causal_integrity_assertion_failures(tmp_path: Path):
    """Test 10: Invariant violations raise CausalIntegrityError."""
    tiny_model = make_tiny_bert_mlm()
    tok = MockTinyTokenizer()

    # Case A: Post-MLM hashes do not diverge (treatment had zero effect)
    with patch(
        "tradingagents.temporal_leakage.twin_pipeline.run_continued_pretraining_mlm"
    ) as mock_mlm:
        mock_mlm.return_value = {
            "branch": "clean",
            "dose": 0.0,
            "base_checkpoint": "mock",
            "base_revision": "mock",
            "initial_parameter_hash": "same_hash_123",
            "final_parameter_hash": "same_hash_123",  # Model did not update!
            "corpus_hash": "corp_hash",
            "token_count": 256,
            "optimizer_steps": 2,
            "batch_size": 2,
            "max_seq_length": 128,
            "mlm_probability": 0.15,
            "learning_rate": 5e-5,
            "seed": 42,
        }

        with pytest.raises(CausalIntegrityError, match="parameter hashes are identical"):
            run_baseline_and_smoke_experiment(
                output_dir=tmp_path / "exp_fail",
                smoke_steps=2,
                smoke_sample_count=8,
                mock_model_for_testing=tiny_model,
                mock_tokenizer_for_testing=tok,
            )
