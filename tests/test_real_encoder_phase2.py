"""Phase 2 Integration & Unit Tests: Real Encoder Baseline & Clean/Leak Twin Construction.

Tests cover:
A. TemporalSample safe defaults (UNVERIFIED / unknown provenance).
B. Strict direct float/bool label rejection in validate_temporal_sample.
C. Experiment config numeric contracts (probe, bootstrap, economics, time, formal scientific minimums).
D. HuggingFaceTemporalEncoder output shapes (embeddings [N, D], predictions [N], probs [N, 3]).
E. Probability normalization: sum_c P(c|x) == 1.0.
F. Stance score calculation: s = P(Hawkish) - P(Dovish).
G. Deterministic evaluation mode (identical predictions across multiple calls).
H. Checkpoint metadata roundtrip (serialization and reloading).
I. Real dataset ingestion and manifest validation.
J. Equal compute / token matching in twin dose ladder streams.

All CI tests run offline using lightweight local fixtures without network model downloads.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest
import torch
import torch.nn as nn

from tradingagents.temporal_leakage import (
    DatasetValidationError,
    FOMCBenchmark,
    HuggingFaceTemporalEncoder,
    TemporalModel,
    TemporalSample,
    load_experiment_config,
    validate_benchmark_against_config,
    validate_experiment_config,
    validate_temporal_sample,
)
from tradingagents.temporal_leakage.datasets.fomc_official import (
    create_fomc_official_benchmark,
    create_fomc_official_fixture,
    load_fomc_official_statements,
)
from tradingagents.temporal_leakage.datasets.trillion_dollar_words import (
    create_trillion_dollar_words_benchmark,
    load_trillion_dollar_words,
)
from tradingagents.temporal_leakage.twin_pipeline import (
    create_dose_stream,
    prepare_twin_corpora,
)


# ---------------------------------------------------------------------------
# Lightweight Mock Fixtures for Fast Offline HF Encoder Testing
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Fast deterministic mock tokenizer for testing without network dependencies."""
    vocab_size = 100

    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        import hashlib
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts)
        seq_len = 16
        ids = []
        for t in texts:
            # Deterministic SHA-256 digest mapping to token IDs
            h = int(hashlib.sha256(t.encode("utf-8")).hexdigest(), 16)
            ids.append([(h + i) % self.vocab_size for i in range(seq_len)])
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones((n, seq_len), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, path: Path):
        with open(Path(path) / "tokenizer_config.json", "w") as f:
            json.dump({"mock": True}, f)


class MockTransformerBase(nn.Module):
    """Mock base transformer outputting hidden states of shape (N, seq_len, 64)."""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        # Deterministic representation based on input_ids sum
        base_val = (input_ids.float().mean(dim=-1, keepdim=True).unsqueeze(-1)) / 100.0
        hidden = base_val.expand(batch_size, seq_len, self.hidden_dim).clone()
        
        class Output:
            pass
        out = Output()
        out.last_hidden_state = hidden
        return out


class MockSequenceClassifier(nn.Module):
    """Mock sequence classification model with base transformer and classification head."""
    def __init__(self, hidden_dim: int = 64, num_labels: int = 3):
        super().__init__()
        self.bert = MockTransformerBase(hidden_dim=hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.config = type("Config", (), {"hidden_size": hidden_dim, "num_labels": num_labels})()

    def forward(self, input_ids, attention_mask=None, labels=None):
        base_out = self.bert(input_ids, attention_mask=attention_mask)
        # Mean pool
        pooled = base_out.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)
        class Output:
            pass
        out = Output()
        out.logits = logits
        return out

    def save_pretrained(self, path: Path):
        with open(Path(path) / "config.json", "w") as f:
            json.dump({"hidden_size": 64, "num_labels": 3}, f)
        torch.save(self.state_dict(), Path(path) / "pytorch_model.bin")


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

def test_a_temporal_sample_safe_defaults():
    """Test A: Direct TemporalSample instantiation must use conservative UNVERIFIED/unknown provenance."""
    sample = TemporalSample(
        sample_id="test-safe-defaults-01",
        text="Inflation remains well anchored.",
        event_time="2018-06-13T14:00:00-04:00",
        available_time="2018-06-13T14:00:00-04:00",
        task_label=0,
    )
    assert sample.availability_source == "UNVERIFIED", (
        f"Default availability_source must be UNVERIFIED, got {sample.availability_source}"
    )
    assert sample.availability_quality == "unknown", (
        f"Default availability_quality must be unknown, got {sample.availability_quality}"
    )


def test_b_direct_float_bool_label_rejection():
    """Test B: Direct TemporalSample validation must strictly reject float and bool task_labels."""
    # Test float 1.0 (Python: 1.0 in {-1, 0, 1} evaluates True, but must fail strict validation)
    sample_float = TemporalSample(
        sample_id="test-float-label",
        text="The Committee decided to raise rates.",
        event_time="2018-03-21T14:00:00-04:00",
        available_time="2018-03-21T14:00:00-04:00",
        task_label=1.0,
        source="Federal Reserve",
        annotation_source="test",
    )
    with pytest.raises(DatasetValidationError) as exc_float:
        validate_temporal_sample(sample_float)
    assert "invalid task_label" in str(exc_float.value)

    # Test bool True (Python: isinstance(True, int) is True, but must fail strict validation)
    sample_bool = TemporalSample(
        sample_id="test-bool-label",
        text="The Committee maintained the target range.",
        event_time="2018-08-01T14:00:00-04:00",
        available_time="2018-08-01T14:00:00-04:00",
        task_label=True,
        source="Federal Reserve",
        annotation_source="test",
    )
    with pytest.raises(DatasetValidationError) as exc_bool:
        validate_temporal_sample(sample_bool)
    assert "invalid task_label" in str(exc_bool.value)


def test_c_config_numeric_validation():
    """Test C: Config validation enforces strict numeric contracts for probe, bootstrap, economics, time."""
    ci_cfg = load_experiment_config("configs/fomc_ci.yaml")

    # 1. Probe validation
    bad_probe = dict(ci_cfg)
    bad_probe["probe"] = dict(ci_cfg["probe"], n_splits=1)
    with pytest.raises(DatasetValidationError) as exc:
        validate_experiment_config(bad_probe)
    assert "probe.n_splits" in str(exc.value)

    bad_probe["probe"] = dict(ci_cfg["probe"], n_permutations=0)
    with pytest.raises(DatasetValidationError) as exc:
        validate_experiment_config(bad_probe)
    assert "probe.n_permutations" in str(exc.value)

    bad_probe["probe"] = dict(ci_cfg["probe"], alpha=-0.5)
    with pytest.raises(DatasetValidationError) as exc:
        validate_experiment_config(bad_probe)
    assert "probe.alpha" in str(exc.value)

    # 2. Bootstrap validation
    bad_boot = dict(ci_cfg)
    bad_boot["bootstrap"] = dict(ci_cfg["bootstrap"], n_bootstrap=0)
    with pytest.raises(DatasetValidationError) as exc:
        validate_experiment_config(bad_boot)
    assert "bootstrap.n_bootstrap" in str(exc.value)

    bad_boot["bootstrap"] = dict(ci_cfg["bootstrap"], confidence_level=1.0)
    with pytest.raises(DatasetValidationError) as exc:
        validate_experiment_config(bad_boot)
    assert "bootstrap.confidence_level" in str(exc.value)

    bad_boot["bootstrap"] = dict(ci_cfg["bootstrap"], expected_block_length=0)
    with pytest.raises(DatasetValidationError) as exc:
        validate_experiment_config(bad_boot)
    assert "bootstrap.expected_block_length" in str(exc.value)

    # 3. Economics validation
    bad_econ = dict(ci_cfg)
    bad_econ["economics"] = dict(ci_cfg["economics"], primary_metric="sharpe_ratio")
    with pytest.raises(DatasetValidationError) as exc:
        validate_experiment_config(bad_econ)
    assert "economics.primary_metric" in str(exc.value)

    bad_econ["economics"] = dict(ci_cfg["economics"], transaction_cost_bps=-1.0)
    with pytest.raises(DatasetValidationError) as exc:
        validate_experiment_config(bad_econ)
    assert "economics.transaction_cost_bps" in str(exc.value)

    bad_econ["economics"] = dict(ci_cfg["economics"], annualization_factor=0)
    with pytest.raises(DatasetValidationError) as exc:
        validate_experiment_config(bad_econ)
    assert "economics.annualization_factor" in str(exc.value)

    # 4. Timezone validation under formal config
    formal_cfg = load_experiment_config("configs/fomc_formal_experiment.yaml")
    bad_formal_tz = dict(formal_cfg)
    bad_formal_tz["time"] = dict(formal_cfg["time"], internal_timezone="America/New_York")
    with pytest.raises(DatasetValidationError) as exc:
        validate_experiment_config(bad_formal_tz)
    assert "time.internal_timezone: UTC" in str(exc.value)

    # 5. Formal scientific minimums (n_permutations >= 500, n_bootstrap >= 1000)
    bad_formal_iter = dict(formal_cfg)
    bad_formal_iter["probe"] = dict(formal_cfg["probe"], n_permutations=200)
    with pytest.raises(DatasetValidationError) as exc:
        validate_experiment_config(bad_formal_iter)
    assert "probe.n_permutations >= 500" in str(exc.value)

    bad_formal_iter2 = dict(formal_cfg)
    bad_formal_iter2["bootstrap"] = dict(formal_cfg["bootstrap"], n_bootstrap=500)
    with pytest.raises(DatasetValidationError) as exc:
        validate_experiment_config(bad_formal_iter2)
    assert "bootstrap.n_bootstrap >= 1000" in str(exc.value)


def test_d_hf_adapter_output_shapes():
    """Test D: HuggingFaceTemporalEncoder produces exact expected output shapes."""
    mock_model = MockSequenceClassifier(hidden_dim=64, num_labels=3)
    mock_tok = MockTokenizer()

    encoder = HuggingFaceTemporalEncoder(
        name="test_encoder",
        model=mock_model,
        tokenizer=mock_tok,
        device="cpu",
    )

    texts = [
        "The Committee raised interest rates by 25 basis points.",
        "Labor market conditions have remained strong.",
        "Inflation has moderated somewhat over the past year.",
    ]

    # 1. Representation embeddings shape: (N, hidden_dim)
    embeddings = encoder.encode(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 64)

    # 2. Prediction shapes: y_pred shape (N,), y_prob shape (N, 3)
    y_pred, y_prob = encoder.predict_task(texts)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y_prob, np.ndarray)
    assert y_pred.shape == (3,)
    assert y_prob.shape == (3, 3)
    assert set(np.unique(y_pred)).issubset({-1, 0, 1})


def test_e_probability_normalization():
    """Test E: Output probabilities must strictly sum to 1.0 per row."""
    mock_model = MockSequenceClassifier(hidden_dim=64, num_labels=3)
    mock_tok = MockTokenizer()

    encoder = HuggingFaceTemporalEncoder(
        name="test_encoder",
        model=mock_model,
        tokenizer=mock_tok,
        device="cpu",
    )

    texts = ["Sample statement A", "Sample statement B", "Sample statement C"]
    _, y_prob = encoder.predict_task(texts)

    row_sums = y_prob.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(texts)), rtol=1e-5, atol=1e-5)
    assert np.all(y_prob >= 0.0)
    assert np.all(y_prob <= 1.0)


def test_f_stance_score_calculation():
    """Test F: Continuous stance score must strictly equal s = P(Hawkish) - P(Dovish)."""
    mock_model = MockSequenceClassifier(hidden_dim=64, num_labels=3)
    mock_tok = MockTokenizer()

    encoder = HuggingFaceTemporalEncoder(
        name="test_encoder",
        model=mock_model,
        tokenizer=mock_tok,
        device="cpu",
    )

    texts = ["Sentence 1", "Sentence 2"]
    _, probs = encoder.predict_task(texts)
    stance = encoder.get_stance_score(texts)

    # probs[:, 2] is Hawkish (+1), probs[:, 0] is Dovish (-1)
    expected_stance = probs[:, 2] - probs[:, 0]
    np.testing.assert_allclose(stance, expected_stance, rtol=1e-5, atol=1e-5)
    assert np.all(stance >= -1.0)
    assert np.all(stance <= 1.0)


def test_g_deterministic_evaluation_mode():
    """Test G: Repeated calls on identical inputs yield bit-exact outputs."""
    mock_model = MockSequenceClassifier(hidden_dim=64, num_labels=3)
    mock_tok = MockTokenizer()

    encoder = HuggingFaceTemporalEncoder(
        name="test_encoder",
        model=mock_model,
        tokenizer=mock_tok,
        device="cpu",
    )

    texts = ["The Committee decided to hold rates steady.", "Economic indicators were positive."]
    emb1 = encoder.encode(texts)
    emb2 = encoder.encode(texts)
    np.testing.assert_array_equal(emb1, emb2)

    pred1, prob1 = encoder.predict_task(texts)
    pred2, prob2 = encoder.predict_task(texts)
    np.testing.assert_array_equal(pred1, pred2)
    np.testing.assert_array_equal(prob1, prob2)


def test_h_checkpoint_metadata_roundtrip(tmp_path):
    """Test H: Checkpoint saving writes metadata and reloading preserves all attributes."""
    mock_model = MockSequenceClassifier(hidden_dim=64, num_labels=3)
    mock_tok = MockTokenizer()

    encoder = HuggingFaceTemporalEncoder(
        name="test_clean_twin",
        model=mock_model,
        tokenizer=mock_tok,
        device="cpu",
        training_cutoff="2018-12-31",
        contamination_dose=0.0,
        random_seed=42,
    )

    save_dir = tmp_path / "checkpoint_m_clean"
    custom_metadata = {
        "continued_pretraining_corpus": "pre_cutoff_sham",
        "num_training_tokens": 500000,
        "num_steps": 100,
        "code_commit": "2fb88c00f1c2e46f864e71685c7a2b1d9c21a9f2",
        "dataset_manifest_hash": "sha256:344f6cda7f59a6fcc2b088fd188dd03cc6dc53a8c25b862ce529e3e1218b073d",
    }
    encoder.save_checkpoint(save_dir, metadata=custom_metadata)

    # Check file written
    meta_file = save_dir / "checkpoint_metadata.json"
    assert meta_file.exists()

    with open(meta_file, "r", encoding="utf-8") as f:
        saved_meta = json.load(f)

    assert saved_meta["model_name"] == "test_clean_twin"
    assert saved_meta["training_cutoff"] == "2018-12-31"
    assert saved_meta["contamination_dose"] == 0.0
    assert saved_meta["num_training_tokens"] == 500000
    assert saved_meta["num_steps"] == 100
    assert saved_meta["dataset_manifest_hash"] == custom_metadata["dataset_manifest_hash"]


def test_i_real_dataset_and_manifest_validation():
    """Test I: Trillion Dollar Words dataset loading and manifest verification."""
    manifest_path = Path("data/research/fomc/manifest.json")
    assert manifest_path.exists(), "data/research/fomc/manifest.json must exist."

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["num_samples"] == 2281
    assert manifest["source_verified"] is True
    assert manifest["annotation_verified"] is True
    assert manifest["pit_verified"] is False
    assert manifest["pit_quality_breakdown"]["unknown"] == 2281

    # Load via adapter
    samples = load_trillion_dollar_words()
    assert len(samples) == 2281
    assert all(isinstance(s, TemporalSample) for s in samples)
    assert all(s.availability_quality == "unknown" for s in samples)

    # Create benchmark and check formal research readiness
    bench = create_trillion_dollar_words_benchmark()
    assert bench.source_verified is True
    assert bench.annotation_verified is True
    assert bench.pit_verified is False
    # Correctly blocked from formal research due to unknown availability_quality
    assert bench.is_formal_research_ready() is False

    # Hardcoded official fixture is strictly a test fixture and NOT formal research ready
    official_fixture = create_fomc_official_fixture()
    assert official_fixture.is_formal_research_ready() is False

    # Formal official benchmark strictly requires an explicit verified data file path
    import pytest
    with pytest.raises(ValueError, match="requires an explicit verified data file"):
        create_fomc_official_benchmark()


def test_j_equal_compute_stream_matching():
    """Test J: create_dose_stream constructs streams with exactly identical sample counts across doses."""
    pre_corpus = [f"Pre-cutoff text sentence {i}" for i in range(100)]
    post_corpus = [f"Post-cutoff text sentence {j}" for j in range(100)]

    total_budget = 40
    for dose in [0.0, 0.25, 0.50, 0.75, 1.0]:
        stream = create_dose_stream(
            pre_corpus=pre_corpus,
            post_corpus=post_corpus,
            dose=dose,
            total_samples=total_budget,
            random_seed=42,
        )
        assert len(stream) == total_budget, f"Dose {dose} produced {len(stream)} samples, expected {total_budget}"
        if dose == 0.0:
            assert all(s in pre_corpus for s in stream)
        elif dose == 1.0:
            assert all(s in post_corpus for s in stream)
