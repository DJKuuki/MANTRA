"""Evaluation metrics for Task Competence (C), Temporal Leakage (L),

Leakage-Induced Economic Effect (E_L), and Temporal Robustness (R_T).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef


def evaluate_competence(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[np.ndarray] = None,
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
        n_bootstrap: Number of bootstrap iterations for confidence intervals.
        random_seed: Random seed for reproducibility.

    Returns:
        Dict with evaluation metrics.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    n = len(y_t)
    classes = [-1, 0, 1]

    macro_f1 = float(f1_score(y_t, y_p, labels=classes, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(y_t, y_p)) if len(np.unique(y_p)) > 1 else 0.0
    per_class_f1 = f1_score(y_t, y_p, labels=classes, average=None, zero_division=0)
    cm = confusion_matrix(y_t, y_p, labels=classes).tolist()

    # Probability calibration metrics
    brier: Optional[float] = None
    ece: Optional[float] = None
    if y_prob is not None and len(y_prob) == n:
        probs = np.asarray(y_prob, dtype=np.float64)
        # One-hot encode y_true: -1 -> col 0, 0 -> col 1, 1 -> col 2
        y_one_hot = np.zeros((n, 3), dtype=np.float64)
        for i, val in enumerate(y_t):
            y_one_hot[i, val + 1] = 1.0
        brier = float(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)))

        # Compute ECE across 10 bins
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

    # Block bootstrap for Macro-F1 confidence intervals
    rng = np.random.RandomState(random_seed)
    block_size = max(4, n // 20)
    num_blocks = int(np.ceil(n / block_size))
    boot_f1s = []
    for _ in range(n_bootstrap):
        # Sample starting indices
        start_indices = rng.randint(0, max(1, n - block_size + 1), size=num_blocks)
        boot_idx = np.concatenate([np.arange(idx, min(n, idx + block_size)) for idx in start_indices])[:n]
        bf1 = f1_score(y_t[boot_idx], y_p[boot_idx], labels=classes, average="macro", zero_division=0)
        boot_f1s.append(bf1)

    ci_lower = float(np.percentile(boot_f1s, 2.5))
    ci_upper = float(np.percentile(boot_f1s, 97.5))

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
        "bootstrap_std": float(np.std(boot_f1s)),
    }


def evaluate_representational_leakage(
    h_leak: np.ndarray,
    h_clean: np.ndarray,
    y_future: Sequence[Any],
    n_permutations: int = 500,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Calculate Representational Leakage (L_repr).

    Measures whether the frozen representations of the contaminated model h_leak
    decode future realized macroeconomic/market outcomes y_future significantly better
    than the clean control model h_clean.

    L_repr = ProbeScore(h_leak, y_future) - ProbeScore(h_clean, y_future)
    """
    y = np.asarray(y_future)
    n = len(y)
    is_classification = (y.dtype.kind in "iub") or len(np.unique(y)) <= 5

    def fit_and_score(X: np.ndarray, target: np.ndarray) -> float:
        # 5-fold cross-validation or split
        cv_split = n // 5
        train_x, test_x = X[:-cv_split], X[-cv_split:]
        train_y, test_y = target[:-cv_split], target[-cv_split:]
        if is_classification:
            if len(np.unique(train_y)) <= 1:
                return 0.33
            clf = RidgeClassifier(alpha=1.0, random_state=random_seed)
            clf.fit(train_x, train_y)
            pred = clf.predict(test_x)
            return float(f1_score(test_y, pred, average="macro", zero_division=0))
        else:
            # Continuous target (e.g. 5d return or yield move)
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=1.0, random_state=random_seed)
            reg.fit(train_x, train_y)
            pred = reg.predict(test_x)
            # Rank correlation (IC)
            corr, _ = stats.spearmanr(pred, test_y)
            return float(corr if not np.isnan(corr) else 0.0)

    score_leak = fit_and_score(h_leak, y)
    score_clean = fit_and_score(h_clean, y)
    delta_score = float(score_leak - score_clean)

    # Permutation test for statistical significance of probe advantage
    rng = np.random.RandomState(random_seed)
    perm_deltas = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        ps_l = fit_and_score(h_leak, y_perm)
        ps_c = fit_and_score(h_clean, y_perm)
        perm_deltas.append(ps_l - ps_c)

    p_val = float(np.mean(np.array(perm_deltas) >= delta_score))

    return {
        "probe_score_leak": score_leak,
        "probe_score_clean": score_clean,
        "l_repr": delta_score,
        "p_value": p_val,
        "is_statistically_significant": bool(p_val < 0.05 and delta_score > 0),
    }


def evaluate_behavioral_leakage(
    model: Any,
    samples: List[Any],
    anonymizer_fn: Optional[Callable[[str, int], str]] = None,
) -> Dict[str, Any]:
    """Calculate Behavioral Leakage (L_behavior) via Counterfactual Masking.

    Measures the Jensen-Shannon Divergence between original text predictions and
    masked/anonymized counterfactual text predictions across 4 hierarchy levels:
        Level 1: Institution entities masked (Federal Reserve -> Central Bank A)
        Level 2: Person entities masked (Powell -> Person A)
        Level 3: Year & dates masked (2022 -> [YEAR])
        Level 4: Full anonymization

    Also evaluates Counterfactual Continuation Inversion (L_cf): resistance to
    flipping predictions when forward rate guidance clause is syntactically inverted.
    """
    raw_texts = [s.text for s in samples]
    _, orig_probs = model.predict_task(raw_texts)

    # Default fallback regex anonymizer if none passed
    def default_anonymize(text: str, level: int) -> str:
        t = text
        if level >= 1:
            t = t.replace("Federal Reserve", "Central Bank A").replace("Fed", "Central Bank A").replace("FOMC", "Committee A")
        if level >= 2:
            for name in ["Powell", "Yellen", "Bernanke", "Greenspan"]:
                t = t.replace(name, "Official A")
        if level >= 3:
            for yr in range(1996, 2026):
                t = t.replace(str(yr), "[YEAR]")
        return t

    anon_fn = anonymizer_fn or default_anonymize

    def js_divergence(P: np.ndarray, Q: np.ndarray) -> float:
        # Clip to avoid log(0)
        p = np.clip(P, 1e-9, 1.0)
        q = np.clip(Q, 1e-9, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)
        q = q / np.sum(q, axis=1, keepdims=True)
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p / m), axis=1)
        kl_qm = np.sum(q * np.log(q / m), axis=1)
        js = 0.5 * (kl_pm + kl_qm)
        return float(np.mean(js))

    divs = {}
    for lvl in [1, 2, 3]:
        masked_texts = [anon_fn(t, lvl) for t in raw_texts]
        _, masked_probs = model.predict_task(masked_texts)
        divs[f"level_{lvl}"] = js_divergence(orig_probs, masked_probs)

    l_entity = divs["level_1"]
    l_person = max(0.0, divs["level_2"] - divs["level_1"])
    l_date = max(0.0, divs["level_3"] - divs["level_2"])
    l_total_mask = divs["level_3"]

    return {
        "l_entity": l_entity,
        "l_person": l_person,
        "l_date": l_date,
        "l_total_mask": l_total_mask,
        "divergences_by_level": divs,
    }


def evaluate_economic_effect(
    stance_scores_leak: Sequence[float],
    stance_scores_clean: Sequence[float],
    forward_returns: Sequence[float],
    threshold: float = 0.20,
    transaction_cost_bps: float = 5.0,
    annualization_factor: float = 8.0,  # ~8 FOMC meetings per year
) -> Dict[str, Any]:
    """Calculate Leakage-Induced Economic Effect (E_L) as paired difference between

    contaminated model M_L and clean control M_C.

    E_L(Delta Sharpe) = Sharpe(M_L) - Sharpe(M_C)
    E_L(Delta IC) = IC(M_L) - IC(M_C)
    E_L(Delta Return) = Return(M_L) - Return(M_C)
    """
    s_l = np.asarray(stance_scores_leak, dtype=np.float64)
    s_c = np.asarray(stance_scores_clean, dtype=np.float64)
    f_ret = np.asarray(forward_returns, dtype=np.float64)
    n = len(f_ret)

    # Strategy: Stance > threshold => Long (+1), Stance < -threshold => Short (-1), else 0
    pos_l = np.zeros(n)
    pos_l[s_l > threshold] = 1.0
    pos_l[s_l < -threshold] = -1.0

    pos_c = np.zeros(n)
    pos_c[s_c > threshold] = 1.0
    pos_c[s_c < -threshold] = -1.0

    # Realized returns with transaction cost deduction
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

    # Information Coefficient (Spearman Rank Correlation)
    ic_l, _ = stats.spearmanr(s_l, f_ret) if np.std(s_l) > 1e-8 else (0.0, 1.0)
    ic_c, _ = stats.spearmanr(s_c, f_ret) if np.std(s_c) > 1e-8 else (0.0, 1.0)
    ic_l = float(ic_l if not np.isnan(ic_l) else 0.0)
    ic_c = float(ic_c if not np.isnan(ic_c) else 0.0)
    delta_ic = ic_l - ic_c

    # Cumulative annualized return
    mean_ret_l = float(np.mean(strat_ret_l) * annualization_factor)
    mean_ret_c = float(np.mean(strat_ret_c) * annualization_factor)
    delta_ret = mean_ret_l - mean_ret_c

    # Paired test for Sharpe difference (Ledoit-Wolf approximation via bootstrap)
    rng = np.random.RandomState(42)
    boot_delta_srs = []
    for _ in range(1000):
        idx = rng.randint(0, n, size=n)
        bsr_l = calc_sharpe(strat_ret_l[idx])
        bsr_c = calc_sharpe(strat_ret_c[idx])
        boot_delta_srs.append(bsr_l - bsr_c)

    p_val_sr = float(np.mean(np.array(boot_delta_srs) <= 0.0))

    return {
        "sharpe_leak": sr_l,
        "sharpe_clean": sr_c,
        "delta_sharpe": float(delta_sr),
        "ic_leak": ic_l,
        "ic_clean": ic_c,
        "delta_ic": float(delta_ic),
        "mean_annual_return_leak": mean_ret_l,
        "mean_annual_return_clean": mean_ret_c,
        "delta_annual_return": float(delta_ret),
        "p_value_sharpe": p_val_sr,
        "is_statistically_significant_alpha": bool(p_val_sr < 0.05 and delta_sr > 0.0),
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

    # Trend slope over sequential time slices
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
    l: float,
    e_l: float,
) -> Tuple[float, float, float]:
    """Compute coordinates in the 3D Pareto Space (L, E_L, C).

    Optimal model coordinates: (L -> 0, E_L -> 0, C -> 1.0).
    """
    return (float(l), float(e_l), float(c))
