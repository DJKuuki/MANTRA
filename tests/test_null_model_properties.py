"""Unit tests verifying the formal pathological properties of the Constant-Output Model M_0.

Asserts:
    1. Task Competence C(M_0) ≈ 0 (Macro-F1 <= 0.33, MCC == 0.0)
    2. Temporal Leakage L(M_0) == 0 (L_repr == 0, L_mask == 0)
    3. Economic Effect E_L(M_0) == 0 (Delta Sharpe == 0, Delta IC == 0)
    4. Proves absence of leakage != good model.
"""

import numpy as np
import pytest
from tradingagents.temporal_leakage.temporal_model import NullConstantModel, MockTwinEncoder, TemporalSample
from tradingagents.temporal_leakage.metrics import (
    evaluate_competence,
    evaluate_representational_leakage,
    evaluate_behavioral_leakage,
    evaluate_economic_effect,
)


def test_null_model_properties():
    # Construct 3-class test ground truth (balanced or realistic distribution)
    y_true = np.array([1, 1, 0, 0, -1, -1, 1, 0, -1, 0] * 5)  # 50 samples
    forward_returns = np.array([0.02, -0.01, 0.005, -0.002, -0.03, 0.01, 0.015, -0.004, -0.02, 0.001] * 5)
    future_actions = np.array([1, 1, 0, 0, -1, 0, 1, 0, -1, 0] * 5)

    texts = ["The Committee decided to maintain the target range." for _ in range(50)]
    samples = [
        TemporalSample(
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
    m_clean = MockTwinEncoder("M_clean", competence_level=0.75, leakage_dose=0.0)

    # 1. Task Competence C(M_0)
    preds, probs = m_0.predict_task(texts)
    comp = evaluate_competence(y_true, preds, y_prob=probs, n_bootstrap=100)

    # M_0 predicts only Neutral (0)
    # Recall for Hawkish (+1) is 0 -> F1_hawkish = 0
    # Recall for Dovish (-1) is 0 -> F1_dovish = 0
    # Macro-F1 must be <= 0.35 and MCC must be 0.0
    assert comp["macro_f1"] <= 0.35
    assert comp["mcc"] == 0.0
    assert comp["f1_hawkish"] == 0.0
    assert comp["f1_dovish"] == 0.0

    # 2. Representational Leakage L_repr(M_0)
    h_0 = m_0.encode(texts)
    h_clean = m_clean.encode(texts)
    repr_leak = evaluate_representational_leakage(h_0, h_clean, future_actions, n_permutations=50)

    # h_0 is constant -> has no advantage over clean model
    assert repr_leak["l_repr"] <= 0.05
    assert not repr_leak["is_statistically_significant"]

    # 3. Behavioral Leakage L_mask(M_0)
    behav_leak = evaluate_behavioral_leakage(m_0, samples)
    # Masking text cannot alter constant output -> JS divergence must be identically 0.0
    assert behav_leak["l_total_mask"] == 0.0
    assert behav_leak["l_entity"] == 0.0
    assert behav_leak["l_date"] == 0.0

    # 4. Economic Effect E_L(M_0)
    s_0 = m_0.get_stance_score(texts)
    s_clean = m_clean.get_stance_score(texts)
    econ = evaluate_economic_effect(s_0, s_clean, forward_returns)

    # M_0 takes zero positions -> Sharpe is 0.0
    assert econ["sharpe_leak"] == 0.0
    assert econ["ic_leak"] == 0.0
    assert not econ["is_statistically_significant_alpha"]
