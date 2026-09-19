# Hardened Formal Mathematical & Operational Definitions of Core Research Variables

## 1. Axiomatic Principles

This document establishes the hardened mathematical and operational definitions for the four core variables in MANTRA:

1. **$C$ — Task Competence**: Intrinsic performance on the financial NLP classification task.
2. **$L_{\text{repr}}$ — Representational Temporal Leakage**: Information encoded in frozen hidden states that predicts post-cutoff outcomes beyond a clean twin model.
3. **$L_{\text{behavior}}$ — Behavioral Temporal Leakage**: Marginal sensitivity to temporal tokens (entities, dates) observed in the contaminated model over the clean twin model.
4. **$E_L$ — Leakage-Induced Economic Effect**: Financial predictive advantage ($\Delta \text{IC}$ and $\Delta \text{Sharpe}$) attributable to temporal contamination.
5. **$R_T$ — Temporal Robustness**: Resistance to post-cutoff concept drift across calendar periods (control variable).

### Foundational Principle: No Arbitrary Composite Scores
$$\text{Quality}(M) \neq \text{Scalar}(M)$$
We strictly reject arbitrary weighted sums such as $Score = aC - bL - cE_L$ or $L_{\text{composite}} = 0.5 L_{\text{repr}} + 0.5 L_{\text{mask}}$.
Evaluation is conducted in the multi-dimensional Pareto space:
$$\mathcal{S} = \left( L_{\text{repr}}, \ L_{\text{behavior}}, \ E_L^{\text{IC}}, \ E_L^{\text{Sharpe}}, \ C \right)$$
Ideal model target:
$$\lim \left( L_{\text{repr}} \to 0, \ L_{\text{behavior}} \to 0, \ E_L \to 0, \ C \to 1.0 \right)$$

---

## 2. Task Competence ($C$)

### 2.1 Mathematical Formulation
Let $\mathcal{D}_{\text{eval}}^{\text{task}} = \left\{ (x_i, y_i) \right\}_{i=1}^N$ be the evaluation dataset for the 3-class monetary policy stance task:
$$\mathcal{Y} = \{ -1 \text{ (Dovish)}, \ 0 \text{ (Neutral)}, \ +1 \text{ (Hawkish)} \}$$

Given model predictions $\hat{y}_i$ and probabilities $\mathbf{p}_i = [p_{i,-1}, p_{i,0}, p_{i,+1}]$, Task Competence $C$ is primarily measured via **Macro-averaged $F_1$**:
$$C_{\text{Macro-F1}}(M) = \frac{1}{3} \sum_{k \in \mathcal{Y}} \frac{2 \cdot \text{Precision}_k(M) \cdot \text{Recall}_k(M)}{\text{Precision}_k(M) + \text{Recall}_k(M)}$$
and **Multiclass Matthews Correlation Coefficient (MCC)**:
$$C_{\text{MCC}}(M) = \frac{c \cdot N - \sum_{k} p_k t_k}{\sqrt{\left(N^2 - \sum_{k} p_k^2\right) \left(N^2 - \sum_{k} t_k^2\right)}}$$
Probability calibration is evaluated via Brier Score and Expected Calibration Error (ECE).

### 2.2 Statistical Estimation: Stationary Block Bootstrap
To account for time-series autocorrelation without assuming stationarity under naive i.i.d. sampling, we apply the **Politis & Romano (1994) Stationary Block Bootstrap**:
- Mean block length $\bar{L} = 8$. Transition probability $p = 1 / \bar{L}$.
- Indices transition geometrically:
  $$\mathbb{P}(I_{t+1} \sim \text{Uniform}(0, N-1)) = p, \quad \mathbb{P}(I_{t+1} = (I_t + 1) \bmod N) = 1 - p$$
- 95% Confidence Intervals are derived from bootstrap replicates:
  $$\text{CI}_{95\%}(C) = \left[ \widehat{C}^*_{(0.025)}, \ \widehat{C}^*_{(0.975)} \right]$$
- **Parameter Distinction**:
  - *Formal Scientific Configuration* (`configs/fomc_formal_experiment.yaml`): $B \ge 1{,}000$ (default $B = 2{,}000$).
  - *CI / Smoke Test Configuration* (`configs/fomc_ci.yaml`): reduced $B = 200$ for rapid test execution.

---

## 3. Representational Leakage ($L_{\text{repr}}$)

### 3.1 Definition via TimeSeries Expanding Window
Let $h_\theta(x_t) \in \mathbb{R}^d$ be the frozen hidden state representation.
Let $Y_{t+\Delta t}^{\text{future}}$ be the future realized macro action (e.g. next meeting policy decision $\Delta \text{FFR}_{t+1} \in \{-1, 0, +1\}$).

We evaluate linear probes using **TimeSeriesSplit expanding window cross-validation** ($K$ folds):
For each fold $k$:
- Train probe $g_\phi$ on in-sample training indices $[0 \dots t_k]$.
- Evaluate Macro-F1 (or rank correlation) on out-of-sample forward test slice $[t_k + 1 \dots t_{k+1}]$.
- Compute fold score for contaminated model $\text{Score}_L(k)$ and clean twin $\text{Score}_C(k)$.
- Compute paired fold delta:
  $$\Delta_k = \text{Score}_L(k) - \text{Score}_C(k)$$

The representational leakage is the mean paired differential:
$$L_{\text{repr}}(M_L; M_C) = \frac{1}{K} \sum_{k=1}^K \Delta_k$$

### 3.2 Matched Permutation Significance Test
Under the null hypothesis $H_0: \mathbb{E}[\Delta_k] \le 0$, the identity of "clean" vs "leak" within each matched time fold is exchangeable.
We execute a **paired sign-flip permutation test**:
$$p\text{-value} = \frac{1}{M} \sum_{m=1}^M \mathbb{I}\left( \frac{1}{K} \sum_{k=1}^K s_{k, m} \Delta_k \ge L_{\text{repr}} \right), \quad s_{k, m} \in \{-1, +1\} \text{ with } p=0.5$$
- **Parameter Distinction**:
  - *Formal Scientific Configuration*: $M \ge 500$ permutations (default $M = 1{,}000$).
  - *CI / Smoke Test Configuration*: reduced $M = 100$ permutations.

---

## 4. Behavioral Leakage ($L_{\text{behavior}}$)

### 4.1 Masking Sensitivity vs. Behavioral Leakage
Sensitivity to counterfactual masking on an individual model is:
$$S_{\text{mask}}(M) = \frac{1}{N} \sum_{i=1}^N \mathcal{D}_{\text{JS}}\left( P_M(y \mid x_i) \ \parallel \ P_M(y \mid \mathcal{T}(x_i)) \right)$$
where $\mathcal{T}(x)$ masks institutions (Level 1), persons (Level 2), or calendar years (Level 3).
**$S_{\text{mask}}(M)$ is NOT leakage.** It measures how much the model utilizes contemporaneous named tokens.

### 4.2 Causal Definition via Clean/Leak Twin Differential
Behavioral leakage is strictly defined as the **differential masking sensitivity** between the contaminated model $M_L$ and the clean twin $M_C$:
$$L_{\text{behavior}} = S_{\text{mask}}(M_L) - S_{\text{mask}}(M_C)$$
$$L_{\text{entity}} = S_{\text{entity}}(M_L) - S_{\text{entity}}(M_C)$$
$$L_{\text{person}} = S_{\text{person}}(M_L) - S_{\text{person}}(M_C)$$
$$L_{\text{date}} = S_{\text{date}}(M_L) - S_{\text{date}}(M_C)$$

If $M_L$ has memorized future market regimes indexed by historical dates or officials, anonymizing those tokens destroys the memorized shortcut, producing $S_{\text{mask}}(M_L) \gg S_{\text{mask}}(M_C) \implies L_{\text{behavior}} > 0$.
For two clean models ($M_C$ vs $M_C$), $L_{\text{behavior}} \equiv 0.0$.

---

## 5. Leakage-Induced Economic Effect ($E_L$)

### 5.1 Level A (Primary Metric): Model-Agnostic Information Coefficient ($\Delta \text{IC}$)
**Delta IC is the primary economic effect estimator.** It computes the Spearman rank correlation between model continuous stance score $s_t = P(\text{Hawkish}) - P(\text{Dovish})$ and forward asset return $r_{t, t+k}$:
$$\text{IC}(M) = \text{RankCorr}\left( s_t(M), \ r_{t, t+k} \right)$$
The primary economic leakage effect is:
$$E_L^{\text{IC}} = \text{IC}(M_L) - \text{IC}(M_C)$$
This requires NO arbitrary assumptions regarding directional position orientation or trading rule heuristics.

### 5.2 Level B (Secondary / Illustrative): Strategy Performance Delta ($\Delta \text{Sharpe}$)
Level B provides a **fixed illustrative strategy for secondary sensitivity analysis only**:
- Stance threshold ($\tau = 0.20$) and stance-to-position mapping ($w_t \in \{-1, 0, +1\}$) must be fixed a priori or calibrated exclusively on the pre-cutoff development set ($\mathcal{D}_{\text{dev}}$).
- **Zero test-set tuning**: No tuning of $\tau$ or orientation flipping is permitted on the out-of-sample evaluation split.
$$E_L^{\text{Sharpe}} = \text{Sharpe}(M_L) - \text{Sharpe}(M_C)$$
$$E_L^{\text{Return}} = \bar{R}(M_L) - \bar{R}(M_C)$$
Incorporates 5 bps two-way transaction costs.

### 5.3 Statistical Inference via Paired Block Bootstrap
We compute paired stationary block bootstrap distributions for $\Delta \text{IC}$, $\Delta \text{Sharpe}$, and $\Delta \text{Return}$:
$$p\text{-value}_{\text{Sharpe}} = \frac{1}{B} \sum_{b=1}^B \mathbb{I}\left( \Delta \text{Sharpe}^{*(b)} \le 0 \right)$$
*Note: Asymptotic HAC inference (Ledoit-Wolf) and Deflated Sharpe Ratio (DSR) are scheduled for future production modules.*

---

## 6. Null Model Invariants: Pathological Case $M_0$

Let $M_0(x) = \text{Neutral}, \forall x$.
1. **$L_{\text{repr}}(M_0) = 0$**: Hidden representation is constant $\implies \text{Cov}(h_{M_0}, Y^{\text{future}}) = 0$.
2. **$L_{\text{behavior}}(M_0) = 0$**: For any mask $\mathcal{T}$, $P(y \mid x) = P(y \mid \mathcal{T}(x)) = [0, 1, 0]^T \implies S_{\text{mask}}(M_0) = 0 \implies L_{\text{behavior}} = 0$.
3. **$E_L(M_0) = 0$**: Generates zero active positions $\implies \Delta \text{IC} = 0, \Delta \text{Sharpe} = 0$.
4. **$C(M_0) \approx 0$**: $\text{Macro-F1} \le 0.33, \text{MCC} = 0.0$.

Under this hardened formulation, $M_0$ is placed at $(0, 0, 0, 0, 0)$, cleanly separated from high-performing models.
