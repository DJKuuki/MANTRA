"""Unit tests verifying the formal pathological properties of the Constant-Output Model M_0.

Asserts:
    1. Task Competence C(M_0) ≈ 0 (Macro-F1 <= 0.35, MCC == 0.0)
    2. Temporal Leakage:
       - L_repr(M_0) <= 0.05
       - Masking Sensitivity S_mask(M_0) == 0.0
       - Behavioral Differential L_behavior(M_0, M_clean) <= 0.0
    3. Economic Effect E_L(M_0) <= 0 (Delta IC <= 0, Delta Sharpe <= 0)
    4. Proves absence of leakage != good model.
"""

import numpy as np
import pytest
from tradingagents.temporal_leakage.temporal_model import (
    NullConstantModel,
    SyntheticTemporalTwinEncoder,
    TemporalSample,
)
from tradingagents.temporal_leakage.metrics import (
    compute_masking_sensitivity,
    evaluate_competence,
    evaluate_representational_leakage,
    evaluate_behavioral_leakage,
    evaluate_economic_effect,
)


def test_null_model_properties():
    y_true = np.array([1, 1, 0, 0, -1, -1, 1, 0, -1, 0] * 5)  # 50 samples
    forward_returns = np.array([0.02, -0.01, 0.005, -0.002, -0.03, 0.01, 0.015, -0.004, -0.02, 0.001] * 5)
    future_actions = np.array([1, 1, 0, 0, -1, 0, 1, 0, -1, 0] * 5)

    texts = [f"The Federal Reserve decided to maintain the target range in {2018 + (i % 6)}." for i in range(50)]
    samples = [
        TemporalSample(
            sample_id=f"sample-{i}",
            text=t,
            event_time="2022-01-01T14:00:00",
            available_time="2022-01-01T14:00:00",
            task_label=int(y_true[i]),
            future_macro_labels={"next_action": int(future_actions[i])},
            market_outcomes={"spy_5d_return": float(forward_returns[i])},
        )
        for i, t in enumerate(texts)
    ]

    m_0 = NullConstantModel(constant_class=0)
    m_clean = SyntheticTemporalTwinEncoder("M_clean", competence_level=0.75, contamination_dose=0.0)

    # 1. Task Competence C(M_0)
    preds, probs = m_0.predict_task(texts)
    comp = evaluate_competence(y_true, preds, y_prob=probs, n_bootstrap=100)

    assert comp["macro_f1"] <= 0.35
    assert comp["mcc"] == 0.0
    assert comp["f1_hawkish"] == 0.0
    assert comp["f1_dovish"] == 0.0

    # 2. Representational Leakage L_repr(M_0)
    h_0 = m_0.encode(texts)
    h_clean = m_clean.encode(texts)
    repr_leak = evaluate_representational_leakage(h_0, h_clean, future_actions, probe_cv="timeseries", n_splits=3, n_permutations=50)

    # h_0 has no future information -> probe delta <= 0.05
    assert repr_leak["l_repr"] <= 0.05
    assert not repr_leak["is_statistically_significant"]

    # 3. Behavioral Sensitivity and Leakage
    sens_m0 = compute_masking_sensitivity(m_0, samples)
    assert sens_m0["mask_sensitivity"] == 0.0
    assert sens_m0["sensitivity_entity"] == 0.0
    assert sens_m0["sensitivity_date"] == 0.0

    behav_leak = evaluate_behavioral_leakage(model_leak=m_0, model_clean=m_clean, samples=samples)
    assert behav_leak["mask_sensitivity_leak"] == 0.0
    assert behav_leak["l_behavior_delta"] <= 0.0

    # 4. Economic Effect E_L(M_0)
    s_0 = m_0.get_stance_score(texts)
    s_clean = m_clean.get_stance_score(texts)
    econ = evaluate_economic_effect(s_0, s_clean, forward_returns)

    # M_0 takes zero positions -> Level A IC is 0, Level B Sharpe is 0
    assert econ["level_a_ic"]["ic_leak"] == 0.0
    assert econ["level_b_strategy"]["sharpe_leak"] == 0.0
    assert not econ["level_b_strategy"]["is_significant_alpha"]
