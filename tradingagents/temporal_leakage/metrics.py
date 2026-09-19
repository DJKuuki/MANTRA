"""Hardened evaluation metrics for Task Competence (C), Temporal Leakage (L),

Leakage-Induced Economic Effect (E_L), and Temporal Robustness (R_T).

Methodological Principles:
1. Masking Sensitivity ≠ Leakage: Behavioral leakage is defined strictly as the
   Clean/Leak Twin Differential: L_behavior = Sensitivity(M_L) - Sensitivity(M_C).
2. Representational Leakage: L_repr = ProbeScore(M_L) - ProbeScore(M_C) evaluated
   via TimeSeriesSplit expanding window and matched paired permutation testing.
3. No arbitrary composite scores: L_repr and L_behavior remain strictly separate.
4. Economic Effect: Level A evaluates model-agnostic Information Coefficient (IC) and
   Delta IC; Level B evaluates strategy Delta Sharpe and Delta return with paired
   block bootstrap confidence intervals.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import TimeSeriesSplit


def stationary_block_bootstrap_indices(
    n: int,
    expected_block_length: float = 8.0,
    n_boot: int = 1000,
    random_seed: int = 42,
) -> np.ndarray:
    """Generate index matrices using the Politis & Romano (1994) Stationary Block Bootstrap.

    Block lengths are drawn from a Geometric distribution with parameter p = 1 / expected_block_length.
    With probability p, a new uniform random start index in [0, n-1] is selected;
    with probability 1 - p, the index increments circularly: i_t = (i_{t-1} + 1) mod n.

    Returns:
        np.ndarray of shape (n_boot, n) with resampled integer indices.
    """
    if n <= 0:
        return np.empty((n_boot, 0), dtype=int)
    rng = np.random.RandomState(random_seed)
    p = 1.0 / max(1.0, float(expected_block_length))
    res = np.empty((n_boot, n), dtype=int)

    for b in range(n_boot):
        # First index is chosen uniformly at random
        current_idx = rng.randint(0, n)
        res[b, 0] = current_idx
        # Transitions
        transitions = rng.rand(n - 1) < p
        new_starts = rng.randint(0, n, size=n - 1)
        for t in range(1, n):
            if transitions[t - 1]:
                current_idx = new_starts[t - 1]
            else:
                current_idx = (current_idx + 1) % n
            res[b, t] = current_idx

    return res


def evaluate_competence(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[np.ndarray] = None,
    expected_block_length: float = 8.0,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Calculate Task Competence C for financial NLP classification.

    Evaluates:
        - Macro-F1 (primary metric, robust to class imbalance)
        - Matthews Correlation Coefficient (MCC)
        - Class-specific F1 scores
        - Brier Score (if probabilities provided)
        - Expected Calibration Error (ECE, if probabilities provided)
        - Stationary Block Bootstrap 95% Confidence Interval for Macro-F1

    Args:
        y_true: Ground truth class labels {-1, 0, +1}.
        y_pred: Predicted class labels {-1, 0, +1}.
        y_prob: Optional array of predicted probabilities shape (N, 3).
        expected_block_length: Mean block length for stationary bootstrap.
        n_bootstrap: Number of bootstrap iterations.
        random_seed: Random seed for reproducibility.

    Returns:
        Dict with evaluation metrics and bootstrap confidence intervals.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    n = len(y_t)
    classes = [-1, 0, 1]

    macro_f1 = float(f1_score(y_t, y_p, labels=classes, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(y_t, y_p)) if len(np.unique(y_p)) > 1 and len(np.unique(y_t)) > 1 else 0.0
    per_class_f1 = f1_score(y_t, y_p, labels=classes, average=None, zero_division=0)
    cm = confusion_matrix(y_t, y_p, labels=classes).tolist()

    # Calibration
    brier: Optional[float] = None
    ece: Optional[float] = None
    if y_prob is not None and len(y_prob) == n:
        probs = np.asarray(y_prob, dtype=np.float64)
        y_one_hot = np.zeros((n, 3), dtype=np.float64)
        for i, val in enumerate(y_t):
            y_one_hot[i, val + 1] = 1.0
        brier = float(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)))

        confidences = np.max(probs, axis=1)
        corrects = (y_p == y_t).astype(float)
        bins = np.linspace(0.0, 1.0, 11)
        ece_accum = 0.0
        for b_low, b_high in zip(bins[:-1], bins[1:]):
            in_bin = (confidences > b_low) & (confidences <= b_high)
            if np.any(in_bin):
                bin_acc = np.mean(corrects[in_bin])
                bin_conf = np.mean(confidences[in_bin])
                bin_weight = np.sum(in_bin) / n
                ece_accum += bin_weight * abs(bin_acc - bin_conf)
        ece = float(ece_accum)

    # Stationary Block Bootstrap for Macro-F1 CI
    if n > 2:
        boot_idx_mat = stationary_block_bootstrap_indices(
            n=n,
            expected_block_length=expected_block_length,
            n_boot=n_bootstrap,
            random_seed=random_seed,
        )
        boot_f1s = [
            f1_score(y_t[idx], y_p[idx], labels=classes, average="macro", zero_division=0)
            for idx in boot_idx_mat
        ]
        ci_lower = float(np.percentile(boot_f1s, 2.5))
        ci_upper = float(np.percentile(boot_f1s, 97.5))
        boot_std = float(np.std(boot_f1s))
    else:
        ci_lower, ci_upper, boot_std = macro_f1, macro_f1, 0.0

    return {
        "macro_f1": macro_f1,
        "mcc": mcc,
        "f1_dovish": float(per_class_f1[0]),
        "f1_neutral": float(per_class_f1[1]),
        "f1_hawkish": float(per_class_f1[2]),
        "brier_score": brier,
        "ece": ece,
        "confusion_matrix": cm,
        "f1_ci_95": [ci_lower, ci_upper],
        "bootstrap_std": boot_std,
    }


def compute_masking_sensitivity(
    model: Any,
    samples: Sequence[Any],
    future_signals: Optional[Sequence[Any]] = None,
    anonymizer_fn: Optional[Callable[[str, int], str]] = None,
) -> Dict[str, Any]:
    """Compute Masking Sensitivity S_mask(M) across hierarchy levels:

        Level 1: Institution entities masked
        Level 2: Person entities masked
        Level 3: Year & date tokens masked
    NOTE: This is Masking Sensitivity, NOT Temporal Leakage!
    """
    raw_texts = [s.text for s in samples]
    _, orig_probs = model.predict_task(raw_texts, future_signals=future_signals)

    def default_anonymize(text: str, level: int) -> str:
        t = text
        if level >= 1:
            for ent in ["Federal Reserve", "Federal Open Market Committee", "FOMC", "Fed", "Committee"]:
                t = t.replace(ent, "Central Bank A")
        if level >= 2:
            for name in ["Powell", "Yellen", "Bernanke", "Greenspan"]:
                t = t.replace(name, "Official A")
        if level >= 3:
            for yr in range(1996, 2026):
                t = t.replace(str(yr), "[YEAR]")
        return t

    anon_fn = anonymizer_fn or default_anonymize

    def js_divergence(P: np.ndarray, Q: np.ndarray) -> float:
        p = np.clip(P, 1e-9, 1.0)
        q = np.clip(Q, 1e-9, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)
        q = q / np.sum(q, axis=1, keepdims=True)
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p / m), axis=1)
        kl_qm = np.sum(q * np.log(q / m), axis=1)
        return float(np.mean(0.5 * (kl_pm + kl_qm)))

    sens_by_level = {}
    for lvl in [1, 2, 3]:
        masked_texts = [anon_fn(t, lvl) for t in raw_texts]
        _, masked_probs = model.predict_task(masked_texts, future_signals=future_signals)
        sens_by_level[f"level_{lvl}"] = js_divergence(orig_probs, masked_probs)

    s_entity = sens_by_level["level_1"]
    s_person = max(0.0, sens_by_level["level_2"] - sens_by_level["level_1"])
    s_date = max(0.0, sens_by_level["level_3"] - sens_by_level["level_2"])
    s_total = sens_by_level["level_3"]

    return {
        "mask_sensitivity": s_total,
        "sensitivity_entity": s_entity,
        "sensitivity_person": s_person,
        "sensitivity_date": s_date,
        "sensitivities_by_level": sens_by_level,
    }


def evaluate_behavioral_leakage(
    model_leak: Any,
    model_clean: Any,
    samples: Sequence[Any],
    future_signals: Optional[Sequence[Any]] = None,
    anonymizer_fn: Optional[Callable[[str, int], str]] = None,
) -> Dict[str, Any]:
    """Calculate Behavioral Leakage (L_behavior) as Clean/Leak Twin Differential:

        L_behavior = Sensitivity(M_L) - Sensitivity(M_C)
        L_entity   = S_entity(M_L) - S_entity(M_C)
        L_person   = S_person(M_L) - S_person(M_C)
        L_date     = S_date(M_L) - S_date(M_C)

    Raw masking sensitivity is preserved as `mask_sensitivity_leak` and `mask_sensitivity_clean`.
    """
    sens_leak = compute_masking_sensitivity(
        model_leak, samples, future_signals=future_signals, anonymizer_fn=anonymizer_fn
    )
    sens_clean = compute_masking_sensitivity(
        model_clean, samples, future_signals=None, anonymizer_fn=anonymizer_fn
    )

    l_behavior_delta = float(sens_leak["mask_sensitivity"] - sens_clean["mask_sensitivity"])
    l_entity_delta = float(sens_leak["sensitivity_entity"] - sens_clean["sensitivity_entity"])
    l_person_delta = float(sens_leak["sensitivity_person"] - sens_clean["sensitivity_person"])
    l_date_delta = float(sens_leak["sensitivity_date"] - sens_clean["sensitivity_date"])

    return {
        "mask_sensitivity_leak": sens_leak["mask_sensitivity"],
        "mask_sensitivity_clean": sens_clean["mask_sensitivity"],
        "l_behavior_delta": l_behavior_delta,
        "l_entity_delta": l_entity_delta,
        "l_person_delta": l_person_delta,
        "l_date_delta": l_date_delta,
        "sensitivities_by_level_leak": sens_leak["sensitivities_by_level"],
        "sensitivities_by_level_clean": sens_clean["sensitivities_by_level"],
    }


def evaluate_representational_leakage(
    h_leak: np.ndarray,
    h_clean: np.ndarray,
    y_future: Sequence[Any],
    probe_cv: str = "timeseries",
    n_splits: int = 3,
    n_permutations: int = 200,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Calculate Representational Leakage (L_repr) using TimeSeriesSplit expanding window

    and paired / matched permutation testing.

    L_repr = MeanFoldScore(h_leak) - MeanFoldScore(h_clean)
    Delta_k = FoldScore_leak(k) - FoldScore_clean(k)
    """
    y = np.asarray(y_future)
    n = len(y)
    unique_y = np.unique(y)

    # Insufficient sample guard
    if n < 8 or len(unique_y) < 2:
        return {
            "probe_score_leak": float(np.nan),
            "probe_score_clean": float(np.nan),
            "l_repr": 0.0,
            "fold_scores_leak": [],
            "fold_scores_clean": [],
            "paired_fold_deltas": [],
            "p_value": 1.0,
            "is_statistically_significant": False,
            "insufficient_samples": True,
        }

    is_classification = (y.dtype.kind in "iub") or len(unique_y) <= 5
    actual_splits = max(2, min(n_splits, n // 3))
    tscv = TimeSeriesSplit(n_splits=actual_splits)

    fold_scores_l: List[float] = []
    fold_scores_c: List[float] = []
    paired_deltas: List[float] = []

    for train_idx, test_idx in tscv.split(h_leak):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 1:
            continue

        if is_classification:
            clf_l = RidgeClassifier(alpha=1.0, random_state=random_seed)
            clf_l.fit(h_leak[train_idx], y_train)
            score_l = float(f1_score(y_test, clf_l.predict(h_leak[test_idx]), average="macro", zero_division=0))

            clf_c = RidgeClassifier(alpha=1.0, random_state=random_seed)
            clf_c.fit(h_clean[train_idx], y_train)
            score_c = float(f1_score(y_test, clf_c.predict(h_clean[test_idx]), average="macro", zero_division=0))
        else:
            reg_l = Ridge(alpha=1.0, random_state=random_seed)
            reg_l.fit(h_leak[train_idx], y_train)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
                corr_l, _ = stats.spearmanr(reg_l.predict(h_leak[test_idx]), y_test)
            score_l = float(corr_l if not np.isnan(corr_l) else 0.0)

            reg_c = Ridge(alpha=1.0, random_state=random_seed)
            reg_c.fit(h_clean[train_idx], y_train)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
                corr_c, _ = stats.spearmanr(reg_c.predict(h_clean[test_idx]), y_test)
            score_c = float(corr_c if not np.isnan(corr_c) else 0.0)

        fold_scores_l.append(score_l)
        fold_scores_c.append(score_c)
        paired_deltas.append(score_l - score_c)

    if not paired_deltas:
        return {
            "probe_score_leak": float(np.nan),
            "probe_score_clean": float(np.nan),
            "l_repr": 0.0,
            "fold_scores_leak": [],
            "fold_scores_clean": [],
            "paired_fold_deltas": [],
            "p_value": 1.0,
            "is_statistically_significant": False,
            "insufficient_samples": True,
        }

    mean_leak = float(np.mean(fold_scores_l))
    mean_clean = float(np.mean(fold_scores_c))
    l_repr = float(np.mean(paired_deltas))

    # Paired matched permutation test (sign-flip on paired fold deltas)
    rng = np.random.RandomState(random_seed)
    deltas_arr = np.array(paired_deltas)
    perm_means = []
    for _ in range(n_permutations):
        # Under H0: leak and clean are exchangeable -> signs are +/- with prob 0.5
        signs = rng.choice([-1.0, 1.0], size=len(deltas_arr))
        perm_means.append(float(np.mean(deltas_arr * signs)))

    p_val = float(np.mean(np.array(perm_means) >= l_repr))

    return {
        "probe_score_leak": mean_leak,
        "probe_score_clean": mean_clean,
        "l_repr": l_repr,
        "fold_scores_leak": fold_scores_l,
        "fold_scores_clean": fold_scores_c,
        "paired_fold_deltas": paired_deltas,
        "p_value": p_val,
        "is_statistically_significant": bool(p_val < 0.05 and l_repr > 0.0),
        "insufficient_samples": False,
    }


def evaluate_economic_effect(
    stance_scores_leak: Sequence[float],
    stance_scores_clean: Sequence[float],
    forward_returns: Sequence[float],
    threshold: float = 0.20,
    transaction_cost_bps: float = 5.0,
    annualization_factor: float = 8.0,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Calculate Leakage-Induced Economic Effect (E_L) as paired difference between

    contaminated model M_L and clean control M_C.

    Level A (Primary, model-agnostic):
        IC(M_L, returns), IC(M_C, returns), Delta IC = IC(M_L) - IC(M_C)
    Level B (Trading strategy):
        Delta Sharpe = Sharpe(M_L) - Sharpe(M_C)
        Delta Return = Return(M_L) - Return(M_C)
    Evaluated with paired stationary block bootstrap confidence intervals.
    """
    s_l = np.asarray(stance_scores_leak, dtype=np.float64)
    s_c = np.asarray(stance_scores_clean, dtype=np.float64)
    f_ret = np.asarray(forward_returns, dtype=np.float64)
    n = len(f_ret)

    # Level A: Information Coefficient
    ic_l, _ = stats.spearmanr(s_l, f_ret) if np.std(s_l) > 1e-8 else (0.0, 1.0)
    ic_c, _ = stats.spearmanr(s_c, f_ret) if np.std(s_c) > 1e-8 else (0.0, 1.0)
    ic_l = float(ic_l if not np.isnan(ic_l) else 0.0)
    ic_c = float(ic_c if not np.isnan(ic_c) else 0.0)
    delta_ic = ic_l - ic_c

    # Level B: Position mapping with transaction costs
    pos_l = np.zeros(n)
    pos_l[s_l > threshold] = 1.0
    pos_l[s_l < -threshold] = -1.0

    pos_c = np.zeros(n)
    pos_c[s_c > threshold] = 1.0
    pos_c[s_c < -threshold] = -1.0

    t_cost = transaction_cost_bps / 10000.0
    turnover_l = np.abs(np.diff(np.insert(pos_l, 0, 0.0)))
    turnover_c = np.abs(np.diff(np.insert(pos_c, 0, 0.0)))

    strat_ret_l = pos_l * f_ret - t_cost * turnover_l
    strat_ret_c = pos_c * f_ret - t_cost * turnover_c

    def calc_sharpe(r: np.ndarray) -> float:
        std = np.std(r)
        if std < 1e-8:
            return 0.0
        return float(np.mean(r) / std * np.sqrt(annualization_factor))

    sr_l = calc_sharpe(strat_ret_l)
    sr_c = calc_sharpe(strat_ret_c)
    delta_sr = sr_l - sr_c

    mean_ret_l = float(np.mean(strat_ret_l) * annualization_factor)
    mean_ret_c = float(np.mean(strat_ret_c) * annualization_factor)
    delta_ret = mean_ret_l - mean_ret_c

    # Paired Block Bootstrap for Delta IC and Delta Sharpe
    boot_idx_mat = stationary_block_bootstrap_indices(n=n, expected_block_length=4.0, n_boot=n_bootstrap, random_seed=random_seed)
    boot_delta_ics = []
    boot_delta_srs = []
    boot_delta_rets = []

    for idx in boot_idx_mat:
        sub_sl, sub_sc, sub_ret = s_l[idx], s_c[idx], f_ret[idx]
        b_icl, _ = stats.spearmanr(sub_sl, sub_ret) if np.std(sub_sl) > 1e-8 else (0.0, 1.0)
        b_icc, _ = stats.spearmanr(sub_sc, sub_ret) if np.std(sub_sc) > 1e-8 else (0.0, 1.0)
        boot_delta_ics.append(float(b_icl - b_icc))

        bsr_l = calc_sharpe(strat_ret_l[idx])
        bsr_c = calc_sharpe(strat_ret_c[idx])
        boot_delta_srs.append(bsr_l - bsr_c)

        bm_l = float(np.mean(strat_ret_l[idx]) * annualization_factor)
        bm_c = float(np.mean(strat_ret_c[idx]) * annualization_factor)
        boot_delta_rets.append(bm_l - bm_c)

    p_val_ic = float(np.mean(np.array(boot_delta_ics) <= 0.0))
    p_val_sr = float(np.mean(np.array(boot_delta_srs) <= 0.0))

    return {
        "level_a_ic": {
            "ic_leak": ic_l,
            "ic_clean": ic_c,
            "delta_ic": float(delta_ic),
            "delta_ic_ci_95": [float(np.percentile(boot_delta_ics, 2.5)), float(np.percentile(boot_delta_ics, 97.5))],
            "p_value_ic": p_val_ic,
            "is_significant_delta_ic": bool(p_val_ic < 0.05 and delta_ic > 0),
        },
        "level_b_strategy": {
            "sharpe_leak": sr_l,
            "sharpe_clean": sr_c,
            "delta_sharpe": float(delta_sr),
            "delta_sharpe_ci_95": [float(np.percentile(boot_delta_srs, 2.5)), float(np.percentile(boot_delta_srs, 97.5))],
            "mean_annual_return_leak": mean_ret_l,
            "mean_annual_return_clean": mean_ret_c,
            "delta_annual_return": float(delta_ret),
            "delta_return_ci_95": [float(np.percentile(boot_delta_rets, 2.5)), float(np.percentile(boot_delta_rets, 97.5))],
            "p_value_sharpe": p_val_sr,
            "is_significant_alpha": bool(p_val_sr < 0.05 and delta_sr > 0),
        },
        # Flat accessors for runner convenience
        "delta_ic": float(delta_ic),
        "delta_sharpe": float(delta_sr),
        "delta_annual_return": float(delta_ret),
        "p_value_sharpe": p_val_sr,
        "p_value_ic": p_val_ic,
    }


def evaluate_temporal_robustness(
    c_baseline: float,
    c_post_slices: Dict[str, float],
) -> Dict[str, Any]:
    """Calculate Temporal Robustness (R_T) to measure performance persistence under

    concept drift across post-cutoff time slices.

    D(Delta t) = C(t_cutoff + Delta t) - C(t_cutoff)
    """
    slices = sorted(c_post_slices.keys())
    deltas = {k: float(c_post_slices[k] - c_baseline) for k in slices}
    mean_decay = float(np.mean(list(deltas.values()))) if deltas else 0.0

    slope = 0.0
    if len(slices) > 1:
        x = np.arange(len(slices))
        y = np.array([deltas[k] for k in slices])
        slope = float(np.polyfit(x, y, 1)[0])

    return {
        "baseline_competence": c_baseline,
        "slice_decays": deltas,
        "mean_decay": mean_decay,
        "decay_slope": slope,
    }


def pareto_coordinates(
    c: float,
    l_repr: float,
    l_behavior: float,
    e_l_ic: float,
    e_l_sharpe: float,
) -> Dict[str, float]:
    """Return explicit multi-dimensional coordinates in the Pareto evaluation space.

    Never compresses metrics into an arbitrary composite scalar.
    Ideal model: L_repr -> 0, L_behavior -> 0, E_L -> 0, C -> 1.0.
    """
    return {
        "competence_C": float(c),
        "leakage_L_repr": float(l_repr),
        "leakage_L_behavior": float(l_behavior),
        "economic_E_L_ic": float(e_l_ic),
        "economic_E_L_sharpe": float(e_l_sharpe),
    }
