"""Phase 2.1 Finalization Test Suite.

Verifies:
- Test A: Fresh head discard (raw sentiment head is discarded; provenance attributes recorded).
- Test B: Body preservation (base encoder body weights match source).
- Test C: Config binding (run_fomc_stance_baseline respects configs/encoder_baseline.yaml).
- Test D: Sample count separation (training samples are not bounded by test_sample_limit).
- Test E: Git provenance clean/dirty detection (resolve_git_provenance).
- Test F: Results schema naming (synthetic_plumbing_pareto_vector, empirical_* is None).
- Test G: TDW dataset metadata (temporal_resolution, timestamp_imputed, mid_year_placeholder).
- Test H: Official FOMC benchmark manifest verification and readiness.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import yaml

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
from tradingagents.temporal_leakage.datasets.trillion_dollar_words import (
    load_trillion_dollar_words,
)
from tradingagents.temporal_leakage.fomc_benchmark import FOMCBenchmark, TemporalSample
from tradingagents.temporal_leakage.hf_encoder import (
    FOMC_STANCE_ID_TO_LABEL,
    FOMC_STANCE_LABEL_TO_ID,
    HuggingFaceTemporalEncoder,
    build_fresh_fomc_classifier_from_base_encoder,
    hash_model_parameters,
    resolve_git_provenance,
)
from tradingagents.temporal_leakage.twin_pipeline import (
    run_baseline_and_smoke_experiment,
    run_fomc_stance_baseline,
    train_baseline_stance_model,
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


def make_dummy_samples(n_pre: int = 10, n_test: int = 5) -> List[TemporalSample]:
    """Generate dummy temporal samples for isolated testing."""
    samples = []
    for i in range(n_pre):
        samples.append(
            TemporalSample(
                sample_id=f"sample-pre-{i}",
                text=f"The Federal Reserve maintained rates at 2.0 percent during meeting {i}.",
                document_type="statement",
                event_time="2018-05-01T14:00:00Z",
                available_time="2018-05-01T14:00:00Z",
                task_label=(i % 3) - 1,
                metadata={"year": 2018},
                availability_quality="exact",
            )
        )
    for j in range(n_test):
        samples.append(
            TemporalSample(
                sample_id=f"sample-test-{j}",
                text=f"The Committee adjusted policy targets to support growth in period {j}.",
                document_type="statement",
                event_time="2021-06-01T14:00:00Z",
                available_time="2021-06-01T14:00:00Z",
                task_label=(j % 3) - 1,
                metadata={"year": 2021},
                availability_quality="exact",
            )
        )
    return samples


def test_a_fresh_head_discard():
    """Test A: build_fresh_fomc_classifier_from_base_encoder discards raw sentiment head."""
    base_model = make_tiny_bert_clf()
    # Fill base head with non-zero sentinel weights
    with torch.no_grad():
        base_model.classifier.weight.fill_(99.0)
        base_model.classifier.bias.fill_(77.0)

    adapter = build_fresh_fomc_classifier_from_base_encoder(
        mock_base_model=base_model,
        random_seed=42,
        tokenizer=MockTinyTokenizer(),
    )

    assert adapter.original_head_loaded is False
    assert adapter.stance_head_initialization == "fresh"
    assert adapter.stance_head_initial_hash is not None
    assert len(adapter.stance_head_initial_hash) == 64

    # The freshly initialized head must NOT retain sentinel weights (99.0 / 77.0)
    fresh_weight = adapter.model.classifier.weight.detach().cpu().numpy()
    assert not np.allclose(fresh_weight, 99.0)
    assert hash_model_parameters(adapter.model.classifier) != hash_model_parameters(base_model.classifier)


def test_b_body_preservation():
    """Test B: build_fresh_fomc_classifier_from_base_encoder preserves base encoder body weights."""
    base_model = make_tiny_bert_clf()
    adapter = build_fresh_fomc_classifier_from_base_encoder(
        mock_base_model=base_model,
        random_seed=42,
        tokenizer=MockTinyTokenizer(),
    )

    # Base embeddings should match exactly
    base_emb = base_model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
    adapter_emb = adapter.model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
    np.testing.assert_array_equal(base_emb, adapter_emb)


def test_c_config_binding(tmp_path: Path):
    """Test C: run_fomc_stance_baseline loads and respects configs/encoder_baseline.yaml."""
    config_dict = {
        "experiment_name": "test_baseline_binding",
        "random_seed": 123,
        "model": {
            "base_checkpoint": "mock_checkpoint",
            "base_revision": "mock_rev",
        },
        "training": {
            "learning_rate": 3.5e-5,
            "batch_size": 2,
            "epochs": 1,
            "max_steps": 2,
        },
        "splits": {
            "train_cutoff_year": 2018,
            "test_start_year": 2020,
        },
        "evaluation": {
            "test_sample_limit": 5,
            "n_bootstrap": 10,
        },
    }
    cfg_file = tmp_path / "encoder_baseline.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f)

    dummy_samples = make_dummy_samples(n_pre=6, n_test=4)
    with patch("tradingagents.temporal_leakage.twin_pipeline.load_trillion_dollar_words", return_value=dummy_samples):
        res = run_fomc_stance_baseline(
            config_path=cfg_file,
            output_dir=tmp_path,
            mock_model_for_testing=make_tiny_bert_clf(),
            mock_tokenizer_for_testing=MockTinyTokenizer(),
        )

    assert res["learning_rate"] == 3.5e-5
    assert res["batch_size"] == 2
    assert res["epochs"] == 1
    assert res["train_sample_count"] == 6
    assert res["test_sample_count"] == 4
    assert res["original_head_loaded"] is False
    assert res["stance_head_initialization"] == "fresh"
    assert (tmp_path / "results" / "baseline_results.json").exists()
    assert (tmp_path / "manifests" / "baseline_manifest.json").exists()


def test_d_sample_count_separation(tmp_path: Path):
    """Test D: Baseline training sample count is decoupled from test_sample_limit."""
    config_dict = {
        "model": {"base_checkpoint": "mock", "base_revision": "mock"},
        "training": {"learning_rate": 1e-4, "batch_size": 2, "epochs": 1, "max_steps": 2},
        "splits": {"train_cutoff_year": 2018, "test_start_year": 2020},
        "evaluation": {"test_sample_limit": 3, "n_bootstrap": 10},
    }
    cfg_file = tmp_path / "encoder_baseline.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f)

    # 15 pre-cutoff samples, 8 test samples
    dummy_samples = make_dummy_samples(n_pre=15, n_test=8)
    with patch("tradingagents.temporal_leakage.twin_pipeline.load_trillion_dollar_words", return_value=dummy_samples):
        res = run_fomc_stance_baseline(
            config_path=cfg_file,
            output_dir=tmp_path,
            test_sample_limit=3,
            training_sample_limit=None,  # Not capped
            mock_model_for_testing=make_tiny_bert_clf(),
            mock_tokenizer_for_testing=MockTinyTokenizer(),
        )

    # train_sample_count MUST be all 15 pre-cutoff samples, NOT capped by test_sample_limit (3)
    assert res["train_sample_count"] == 15
    assert res["test_sample_count"] == 3


def test_e_git_provenance_clean_and_dirty(tmp_path: Path):
    """Test E: resolve_git_provenance detects clean vs dirty working tree and exact commit."""
    # Case 1: Clean tree
    with patch("subprocess.run") as mock_run:
        def fake_run(cmd, *args, **kwargs):
            if cmd[:2] == ["git", "rev-parse"]:
                return MagicMock(returncode=0, stdout="abc123def456\n")
            elif cmd[:2] == ["git", "status"]:
                return MagicMock(returncode=0, stdout="")
            return MagicMock(returncode=1, stdout="")

        mock_run.side_effect = fake_run
        prov_clean = resolve_git_provenance(tmp_path)
        assert prov_clean["git_head"] == "abc123def456"
        assert prov_clean["git_dirty"] is False
        assert prov_clean["code_commit_exact"] is True

    # Case 2: Dirty tree
    with patch("subprocess.run") as mock_run:
        def fake_run(cmd, *args, **kwargs):
            if cmd[:2] == ["git", "rev-parse"]:
                return MagicMock(returncode=0, stdout="abc123def456\n")
            elif cmd[:2] == ["git", "status"]:
                return MagicMock(returncode=0, stdout=" M modified_file.py\n")
            return MagicMock(returncode=1, stdout="")

        mock_run.side_effect = fake_run
        prov_dirty = resolve_git_provenance(tmp_path)
        assert prov_dirty["git_head"] == "abc123def456"
        assert prov_dirty["git_dirty"] is True
        assert prov_dirty["code_commit_exact"] is False


def test_f_results_schema_naming(tmp_path: Path):
    """Test F: smoke_experiment_results schema names synthetic pareto vector and sets empirical metrics to None."""
    dummy_samples = make_dummy_samples(n_pre=10, n_test=10)
    with patch("tradingagents.temporal_leakage.twin_pipeline.load_trillion_dollar_words", return_value=dummy_samples):
        with patch("tradingagents.temporal_leakage.twin_pipeline.run_continued_pretraining_mlm") as mock_mlm:
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
                    "optimizer_steps": 1,
                    "batch_size": 2,
                    "max_seq_length": 128,
                    "mlm_probability": 0.15,
                    "learning_rate": 5e-5,
                    "seed": 42,
                }
            mock_mlm.side_effect = fake_mlm

            res = run_baseline_and_smoke_experiment(
                output_dir=tmp_path,
                smoke_steps=1,
                smoke_sample_count=4,
                mock_model_for_testing=make_tiny_bert_mlm(),
                mock_tokenizer_for_testing=MockTinyTokenizer(),
            )

    assert "synthetic_plumbing_pareto_vector" in res
    assert isinstance(res["synthetic_plumbing_pareto_vector"], dict)
    assert "competence_C" in res["synthetic_plumbing_pareto_vector"]
    assert res["empirical_leakage_metrics"] is None
    assert res["empirical_pareto_vector"] is None
    assert "pareto_vector" not in res
    assert "git_provenance" in res
    assert "git_dirty" in res
    assert "code_commit_exact" in res


def test_g_tdw_dataset_metadata():
    """Test G: load_trillion_dollar_words attaches temporal resolution and imputation metadata."""
    samples = load_trillion_dollar_words()
    assert len(samples) > 0
    first_sample = samples[0]
    assert first_sample.metadata.get("temporal_resolution") == "year"
    assert first_sample.metadata.get("timestamp_imputed") is True
    assert first_sample.metadata.get("timestamp_imputation_rule") == "mid_year_placeholder"


def test_h_official_fomc_benchmark_manifest_requirement(tmp_path: Path):
    """Test H: Formal FOMC benchmark requires explicit verified manifest; unverified is not ready."""
    # Create valid sample file
    sample_records = [
        {
            "sample_id": "test-official-01",
            "text": "The Committee decided to raise interest rates.",
            "document_type": "statement",
            "event_time": "2018-03-21T14:00:00-04:00",
            "available_time": "2018-03-21T14:00:00-04:00",
            "task_label": 1,
            "availability_quality": "exact",
        }
    ]
    data_file = tmp_path / "fomc_data.json"
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(sample_records, f)

    # 1. No manifest provided -> is_formal_research_ready() is False
    bm_no_manifest = create_fomc_official_benchmark(filepath=data_file)
    assert bm_no_manifest.is_formal_research_ready() is False

    # 2. Verified manifest provided with exact SHA256 and provenance -> is_formal_research_ready() is True
    file_sha = hashlib.sha256(data_file.read_bytes()).hexdigest()
    verified_manifest = {
        "source_verified": True,
        "annotation_verified": True,
        "pit_verified": True,
        "dataset_sha256": file_sha,
        "availability_provenance": "Federal Reserve official release schedule",
        "source_urls": ["https://www.federalreserve.gov/example.htm"],
    }
    manifest_file_verified = tmp_path / "manifest_verified.json"
    with open(manifest_file_verified, "w", encoding="utf-8") as f:
        json.dump(verified_manifest, f)

    bm_verified = create_fomc_official_benchmark(
        filepath=data_file,
        manifest_path=manifest_file_verified,
    )
    assert bm_verified.is_formal_research_ready() is True

    # 3. Unverified manifest provided (e.g. pit_verified=False) -> is_formal_research_ready() is False
    unverified_manifest = {
        "source_verified": True,
        "annotation_verified": True,
        "pit_verified": False,
        "dataset_sha256": file_sha,
        "availability_provenance": "Federal Reserve official release schedule",
        "source_urls": ["https://www.federalreserve.gov/example.htm"],
    }
    manifest_file_unverified = tmp_path / "manifest_unverified.json"
    with open(manifest_file_unverified, "w", encoding="utf-8") as f:
        json.dump(unverified_manifest, f)

    bm_unverified = create_fomc_official_benchmark(
        filepath=data_file,
        manifest_path=manifest_file_unverified,
    )
    assert bm_unverified.is_formal_research_ready() is False

    # 4. Hash mismatch in manifest -> must raise ValueError (FAIL)
    mismatch_manifest = {
        "source_verified": True,
        "annotation_verified": True,
        "pit_verified": True,
        "dataset_sha256": "0" * 64,
        "availability_provenance": "Federal Reserve official release schedule",
        "source_urls": ["https://www.federalreserve.gov/example.htm"],
    }
    manifest_file_mismatch = tmp_path / "manifest_mismatch.json"
    with open(manifest_file_mismatch, "w", encoding="utf-8") as f:
        json.dump(mismatch_manifest, f)

    with pytest.raises(ValueError, match="SHA256 verification failed"):
        create_fomc_official_benchmark(
            filepath=data_file,
            manifest_path=manifest_file_mismatch,
        )
