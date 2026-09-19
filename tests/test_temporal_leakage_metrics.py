"""Unit tests verifying temporal leakage metrics, dose-response estimation,

and the Twin Model experiment harness.
"""

import numpy as np
import pytest
from tradingagents.temporal_leakage.fomc_benchmark import FOMCBenchmark
from tradingagents.temporal_leakage.temporal_model import MockTwinEncoder
from tradingagents.temporal_leakage.metrics import (
    evaluate_competence,
    evaluate_representational_leakage,
    evaluate_behavioral_leakage,
    evaluate_economic_effect,
    evaluate_temporal_robustness,
    pareto_coordinates,
)
from tradingagents.temporal_leakage.twin_experiment import TwinExperimentRunner


def test_competence_evaluation():
    y_true = [1, 0, -1, 1, 0, -1, 1, 0, -1, 1]
    y_pred = [1, 0, -1, 1, 0, -1, 1, 0, 0, 1]  # 9/10 correct
    res = evaluate_competence(y_true, y_pred, n_bootstrap=100)

    assert "macro_f1" in res
    assert "mcc" in res
    assert res["macro_f1"] > 0.8
    assert res["mcc"] > 0.7
    assert len(res["f1_ci_95"]) == 2


def test_representational_leakage_detection():
    rng = np.random.RandomState(42)
    n = 60
    # True future target (e.g. next hike/cut)
    y_future = rng.choice([-1, 0, 1], size=n)

    # Clean representation: random noise unrelated to future target
    h_clean = rng.randn(n, 32)

    # Contaminated representation: explicitly leaks y_future into features
    h_leak = rng.randn(n, 32)
    h_leak[:, 0] = y_future * 2.0

    res = evaluate_representational_leakage(h_leak, h_clean, y_future, n_permutations=100)

    assert res["l_repr"] > 0.2
    assert res["is_statistically_significant"] is True
    assert res["p_value"] < 0.05


def test_behavioral_leakage_masking():
    benchmark = FOMCBenchmark()
    samples = benchmark.get_split("test")

    clean_model = MockTwinEncoder("Clean", competence_level=0.8, leakage_dose=0.0)
    leak_model = MockTwinEncoder("Leak", competence_level=0.8, leakage_dose=1.0)

    clean_behav = evaluate_behavioral_leakage(clean_model, samples)
    leak_behav = evaluate_behavioral_leakage(leak_model, samples)

    # Contaminated model changes predictions under entity/date masking much more than clean model
    assert leak_behav["l_total_mask"] >= clean_behav["l_total_mask"]


def test_economic_effect_and_pareto():
    stance_leak = [0.8, -0.6, 0.7, 0.5, -0.7, 0.6, -0.5, 0.8]
    stance_clean = [0.2, -0.1, 0.1, 0.0, -0.1, 0.1, 0.0, 0.2]
    returns = [0.03, -0.04, 0.02, 0.01, -0.05, 0.02, -0.02, 0.04]

    res = evaluate_economic_effect(stance_leak, stance_clean, returns)

    assert "delta_sharpe" in res
    assert "delta_ic" in res
    assert res["delta_sharpe"] > 0.0

    coords = pareto_coordinates(c=0.85, l=0.25, e_l=res["delta_sharpe"])
    assert coords == (0.25, res["delta_sharpe"], 0.85)


def test_temporal_robustness_decay():
    c_base = 0.82
    slices = {"2020": 0.80, "2021": 0.74, "2022": 0.68, "2023": 0.65}
    res = evaluate_temporal_robustness(c_base, slices)

    assert res["mean_decay"] < 0.0
    assert res["decay_slope"] < 0.0  # Demonstrates downward concept drift


def test_twin_experiment_runner(tmp_path):
    benchmark = FOMCBenchmark()
    runner = TwinExperimentRunner(benchmark=benchmark, output_dir=str(tmp_path))
    df = runner.run_experiment()

    assert len(df) == 6  # M_clean, M_leak_25, M_leak_50, M_leak_75, M_leak_100, M_0_null
    assert "competence_macro_f1" in df.columns
    assert "e_l_delta_sharpe" in df.columns
    assert "pareto_L" in df.columns

    # Verify M_0 row properties
    m0_row = df[df["model_name"] == "M_0_null"].iloc[0]
    assert m0_row["pareto_L"] == 0.0
    assert m0_row["pareto_EL"] == 0.0
    assert m0_row["competence_macro_f1"] <= 0.35
