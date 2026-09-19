"""Comprehensive unit tests for hardened temporal leakage methodology:

1. Determinism (stateless inference, bit-exact reproducibility across repeat calls)
2. Behavioral Leakage as Clean/Leak Twin Differential (Clean vs Clean ≈ 0, Leak vs Clean > 0)
3. Dose Monotonicity (L_repr increases monotonically with synthetic contamination dose)
4. Representational Leakage Probe (TimeSeries expanding window and matched paired permutation)
5. Stationary Block Bootstrap & Task Competence
6. Level A (Delta IC) and Level B (Delta Sharpe) Economic Effects
7. TwinExperimentRunner (validating independent Pareto dimensions, no composite score)
"""

import numpy as np
import pytest
from scipy import stats
from tradingagents.temporal_leakage.fomc_benchmark import ToyFOMCBenchmark
from tradingagents.temporal_leakage.temporal_model import (
    NullConstantModel,
    SyntheticTemporalTwinEncoder,
    TemporalSample,
)
from tradingagents.temporal_leakage.metrics import (
    stationary_block_bootstrap_indices,
    evaluate_competence,
    compute_masking_sensitivity,
    evaluate_behavioral_leakage,
    evaluate_representational_leakage,
    evaluate_economic_effect,
    evaluate_temporal_robustness,
    pareto_coordinates,
)
from tradingagents.temporal_leakage.twin_experiment import TwinExperimentRunner


def test_stateless_determinism():
    """Verify inference is 100% stateless and deterministic."""
    texts = [
        "The Federal Reserve decided to maintain the policy rate in 2022.",
        "Inflation remains elevated and the labor market is robust.",
        "The Committee decided to lower rates in 2020.",
    ]
    model = SyntheticTemporalTwinEncoder("TestDet", competence_level=0.8, contamination_dose=0.5, random_seed=123)

    # 1. Multiple calls return bit-exact identical embeddings
    enc1 = model.encode(texts, future_signals=[1, 0, -1])
    enc2 = model.encode(texts, future_signals=[1, 0, -1])
    np.testing.assert_array_equal(enc1, enc2)

    # 2. Multiple calls return bit-exact identical predictions
    preds1, probs1 = model.predict_task(texts, future_signals=[1, 0, -1])
    preds2, probs2 = model.predict_task(texts, future_signals=[1, 0, -1])
    np.testing.assert_array_equal(preds1, preds2)
    np.testing.assert_array_equal(probs1, probs2)

    # 3. Independent call order does not mutate state
    single_enc = model.encode([texts[1]], future_signals=[0])
    np.testing.assert_array_equal(single_enc[0], enc1[1])


def test_behavioral_leakage_differential():
    """Verify Behavioral Leakage is strictly defined as Clean/Leak differential:

    L_behavior(M_clean, M_clean) == 0.0
    L_behavior(M_leak, M_clean) > 0.0
    """
    benchmark = ToyFOMCBenchmark()
    samples = benchmark.get_split("test")
    future_actions = [s.future_macro_labels.get("next_action", 0) for s in samples]

    clean_model_1 = SyntheticTemporalTwinEncoder("Clean1", competence_level=0.8, contamination_dose=0.0)
    clean_model_2 = SyntheticTemporalTwinEncoder("Clean2", competence_level=0.8, contamination_dose=0.0)
    leak_model = SyntheticTemporalTwinEncoder("Leak", competence_level=0.8, contamination_dose=0.8)

    # Clean vs Clean: differential must be identically 0.0
    diff_clean = evaluate_behavioral_leakage(
        model_leak=clean_model_1,
        model_clean=clean_model_2,
        samples=samples,
        future_signals=None,
    )
    assert abs(diff_clean["l_behavior_delta"]) < 1e-6
    assert abs(diff_clean["l_entity_delta"]) < 1e-6

    # Leak vs Clean: contaminated model has higher sensitivity because masking strips memorized future shortcuts
    diff_leak = evaluate_behavioral_leakage(
        model_leak=leak_model,
        model_clean=clean_model_1,
        samples=samples,
        future_signals=future_actions,
    )
    assert diff_leak["mask_sensitivity_leak"] > diff_leak["mask_sensitivity_clean"]
    assert diff_leak["l_behavior_delta"] > 0.01


def test_dose_monotonicity():
    """Verify that synthetic contamination dose correlates monotonically with measured L_repr."""
    n = 60
    texts = [f"The FOMC met in {2018 + (i % 6)} to assess employment and price stability." for i in range(n)]
    rng = np.random.RandomState(42)
    future_targets = rng.choice([-1, 0, 1], size=n)

    clean_model = SyntheticTemporalTwinEncoder("Clean", competence_level=0.8, contamination_dose=0.0)
    h_clean = clean_model.encode(texts, future_signals=None)

    doses = [0.0, 0.25, 0.50, 0.75, 1.00]
    l_repr_values = []

    for d in doses:
        twin = SyntheticTemporalTwinEncoder(f"Twin_{d}", competence_level=0.8, contamination_dose=d)
        h_twin = twin.encode(texts, future_signals=future_targets)
        res = evaluate_representational_leakage(
            h_twin, h_clean, future_targets, probe_cv="timeseries", n_splits=3, n_permutations=50
        )
        l_repr_values.append(res["l_repr"])

    # Verify strong positive rank correlation between contamination dose and measured L_repr
    spearman_corr, _ = stats.spearmanr(doses, l_repr_values)
    assert spearman_corr >= 0.80, f"Expected Spearman corr >= 0.8, got {spearman_corr} for values {l_repr_values}"


def test_stationary_block_bootstrap():
    """Verify Politis & Romano (1994) Stationary Block Bootstrap."""
    n = 50
    indices = stationary_block_bootstrap_indices(n=n, expected_block_length=5.0, n_boot=200, random_seed=42)
    assert indices.shape == (200, 50)
    assert np.all(indices >= 0) and np.all(indices < n)


def test_competence_evaluation():
    y_true = [1, 0, -1, 1, 0, -1, 1, 0, -1, 1] * 5
    y_pred = [1, 0, -1, 1, 0, -1, 1, 0, 0, 1] * 5
    res = evaluate_competence(y_true, y_pred, n_bootstrap=100)

    assert res["macro_f1"] > 0.8
    assert res["mcc"] > 0.7
    assert len(res["f1_ci_95"]) == 2
    assert res["f1_ci_95"][0] <= res["f1_ci_95"][1]


def test_economic_effect_levels():
    """Verify Level A (Delta IC) and Level B (Delta Sharpe) decoupled economics."""
    stance_leak = [0.8, -0.6, 0.7, 0.5, -0.7, 0.6, -0.5, 0.8] * 4
    stance_clean = [0.2, -0.1, 0.1, 0.0, -0.1, 0.1, 0.0, 0.2] * 4
    returns = [0.03, -0.04, 0.02, 0.01, -0.05, 0.02, -0.02, 0.04] * 4

    res = evaluate_economic_effect(stance_leak, stance_clean, returns, n_bootstrap=100)

    # Level A
    assert "delta_ic" in res["level_a_ic"]
    assert res["level_a_ic"]["delta_ic"] > 0.0

    # Level B
    assert "delta_sharpe" in res["level_b_strategy"]
    assert res["level_b_strategy"]["delta_sharpe"] > 0.0

    # Multi-dimensional Pareto coordinates
    coords = pareto_coordinates(c=0.85, l_repr=0.30, l_behavior=0.15, e_l_ic=0.25, e_l_sharpe=1.1)
    assert coords["competence_C"] == 0.85
    assert coords["leakage_L_repr"] == 0.30
    assert coords["leakage_L_behavior"] == 0.15


def test_twin_experiment_runner(tmp_path):
    """Verify end-to-end twin experiment runner with independent metrics."""
    benchmark = ToyFOMCBenchmark()
    runner = TwinExperimentRunner(benchmark=benchmark, output_dir=str(tmp_path))
    df = runner.run_experiment()

    assert len(df) == 6  # M_clean, 4 leak twins, M_0_null
    # Confirm composite score is deleted
    assert "l_composite" not in df.columns
    assert "pareto_L" not in df.columns

    # Confirm independent metrics exist
    assert "l_repr" in df.columns
    assert "l_behavior_delta" in df.columns
    assert "e_l_delta_ic" in df.columns
    assert "e_l_delta_sharpe" in df.columns
    assert "competence_macro_f1" in df.columns

    # M_0 sanity check
    m0_row = df[df["model_name"] == "M_0_null"].iloc[0]
    assert m0_row["pareto_L_repr"] == 0.0
    assert m0_row["pareto_L_behavior"] == 0.0
    assert m0_row["pareto_EL_ic"] == 0.0
    assert m0_row["competence_macro_f1"] <= 0.35
