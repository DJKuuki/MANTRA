# Formal Mathematical & Operational Definitions of Core Research Variables

## 1. Introduction & Axiomatic Foundations

This document establishes the formal mathematical definitions, operational estimators, statistical uncertainty bounds, and null hypothesis tests for the four core variables in the MANTRA research framework:

1. **$C$ — Task Competence**: Genuine ability on the financial NLP task.
2. **$L$ — Temporal Information Leakage**: Parametric dependence on post-cutoff information, stratified into representational ($L_{\text{repr}}$) and behavioral ($L_{\text{behavior}}$) levels.
3. **$E_L$ — Leakage-Induced Economic Effect**: Financial alpha / backtest outperformance attributable strictly to temporal contamination.
4. **$R_T$ — Temporal Robustness**: Resistance to temporal concept drift and macroeconomic regime changes (control variable).

### Fundamental Axiom
$$\text{Quality}(M) \neq \text{Scalar}(M)$$
We reject any arbitrary linear combination $Score = aC - bL - cE_L$. Evaluation is strictly conducted in the three-dimensional Pareto space:
$$\mathcal{S} = \left\{ (L, E_L, C) \in [0, 1] \times \mathbb{R} \times [0, 1] \right\}$$
Ideal model target:
$$\lim (L \to 0, \ E_L \to 0, \ C \to 1)$$

---

## 2. Variable 1: Task Competence ($C$)

### 2.1 Mathematical Formulation
Let the test dataset for the financial NLP classification task be:
$$\mathcal{D}_{\text{eval}}^{\text{task}} = \left\{ (x_i, y_i, t_i) \right\}_{i=1}^N, \quad y_i \in \mathcal{Y} = \{1, 2, \dots, K\}$$
where for FOMC classification, $K = 3$ corresponding to:
- $\text{Hawkish} \ (k = +1)$
- $\text{Neutral} \ (k = 0)$
- $\text{Dovish} \ (k = -1)$

Given model predictions $\hat{y}_i = \arg\max_k P_M(y=k \mid x_i)$ and predicted probability distribution $\mathbf{p}_i = [p_{i,1}, \dots, p_{i,K}]$, the task competence $C$ cannot be measured by raw accuracy due to severe class imbalance (where Neutral frequently accounts for $>60\%$ of sentences).

We define primary Task Competence $C$ as the **Macro-averaged $F_1$ score**:
$$C_{\text{Macro-F1}}(M) = \frac{1}{K} \sum_{k=1}^K F_{1, k}(M)$$
where:
$$F_{1, k}(M) = \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k} = \frac{2 \sum_{i} \mathbb{I}(y_i = k, \hat{y}_i = k)}{2 \sum_{i} \mathbb{I}(y_i = k, \hat{y}_i = k) + \sum_i \mathbb{I}(y_i \neq k, \hat{y}_i = k) + \sum_i \mathbb{I}(y_i = k, \hat{y}_i \neq k)}$$

To capture correlation across all classes simultaneously, we also compute the **Multiclass Matthews Correlation Coefficient (MCC)**:
$$C_{\text{MCC}}(M) = \frac{c \cdot N - \sum_{k=1}^K p_k t_k}{\sqrt{\left(N^2 - \sum_{k=1}^K p_k^2\right) \left(N^2 - \sum_{k=1}^K t_k^2\right)}}$$
where $c = \sum_{k=1}^K C_{kk}$ is total correct predictions, $C_{jk} = \sum_{i=1}^N \mathbb{I}(y_i=j, \hat{y}_i=k)$ is the confusion matrix, $p_k = \sum_j C_{jk}$ is total predicted as class $k$, and $t_k = \sum_j C_{kj}$ is total true instances of class $k$.

To evaluate probability calibration, we compute **Brier Score** and **Expected Calibration Error (ECE)**:
$$\text{Brier}(M) = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K \left( p_{i,k} - \mathbb{I}(y_i = k) \right)^2$$
$$\text{ECE}(M) = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

### 2.2 Operational Estimator & Statistical Uncertainty
Because macroeconomic text exhibits time-series autocorrelation across sequential statements, standard i.i.d. bootstrap underestimates variance. We apply **Stationary Block Bootstrap** (Politis & Romano, 1994):
- Block length $b \sim \text{Geometric}(p)$, mean block size $\bar{b} = 1/p = 8$ meetings.
- Generate $B = 2{,}000$ bootstrap replicates $\mathcal{D}^{*(1)}, \dots, \mathcal{D}^{*(B)}$.
- Estimate empirical confidence interval:
$$\text{CI}_{95\%}(C) = \left[ \widehat{C}^*_{(0.025)}, \ \widehat{C}^*_{(0.975)} \right]$$

---

## 3. Variable 2: Temporal Information Leakage ($L$)

We formalize leakage as a hierarchical flow:
$$L_{\text{repr}} \longrightarrow L_{\text{behavior}} \longrightarrow E_L$$

### 3.1 Level 1: Representational Leakage ($L_{\text{repr}}$)

#### Mathematical Definition
Let $h_\theta(x_t) \in \mathbb{R}^d$ be the frozen hidden state representation produced by the encoder for input text $x_t$ published at time $t \le t_{\text{decision}}$.
Let $Y_{t+\Delta t}^{\text{future}}$ be future macroeconomic or market outcomes that materialize only at $t + \Delta t$ ($\Delta t > 0$), such as:
1. $Y^{\text{hike}}$: Next meeting policy decision ($\Delta \text{FFR}_{t+1} \in \{\text{Hike}, \text{Hold}, \text{Cut}\}$).
2. $Y^{\text{cpi\_surprise}}$: Next release CPI surprise ($\text{CPI}_{t+1} - \text{Consensus}_{t+1}$).
3. $Y^{\text{yield\_direction}}$: Sign of 2Y Treasury yield change over $[t, t+5\text{d}]$.

We train a linear probing classifier $g_\phi: \mathbb{R}^d \to \mathcal{Y}^{\text{future}}$:
$$\min_\phi \mathcal{L}_{\text{probe}}\left(g_\phi(h_\theta(x_t)), Y_{t+\Delta t}^{\text{future}}\right) + \lambda \|\phi\|_2^2$$

Crucially, **a probe's predictive accuracy on future variables does NOT by itself prove leakage**, because legitimate text at time $t$ may have real economic predictive power for time $t+\Delta t$.
Therefore, representational leakage is strictly defined as the **causal probing performance delta** over the clean twin model:
$$L_{\text{repr}}(M_L; M_C) = \text{Score}\left(g_{\phi^*}(h_{M_L}(x_t)), Y_{t+\Delta t}^{\text{future}}\right) - \text{Score}\left(g_{\psi^*}(h_{M_C}(x_t)), Y_{t+\Delta t}^{\text{future}}\right)$$
where $M_L$ is the contaminated model and $M_C$ is the clean twin control.
If $M_L$ has absorbed future market developments during pretraining, its hidden states will linearly decode future outcomes significantly better than $M_C$:
$$L_{\text{repr}} \in [-1, +1], \quad \text{with } L_{\text{repr}} > 0 \implies \text{Representational Leakage}$$

### 3.2 Level 2: Behavioral Leakage ($L_{\text{behavior}}$)

Behavioral leakage measures whether the model's actual task classification probabilities $P_M(y \mid x)$ depend on memorized temporal tokens (entities, dates, forward guidance anchors) rather than structural economic syntax.

#### 1. Counterfactual Entity & Date Masking ($L_{\text{mask}}$)
We define four hierarchical masking operators:
- $\mathcal{T}_0(x) = x$ (Original unmodified text)
- $\mathcal{T}_1(x)$: Mask institutional entities (e.g., "Federal Reserve" $\to$ "Central Bank A", "FOMC" $\to$ "Committee A")
- $\mathcal{T}_2(x)$: Mask person entities (e.g., "Powell" $\to$ "Official A", "Yellen" $\to$ "Official B")
- $\mathcal{T}_3(x)$: Mask temporal tokens (e.g., "2022" $\to$ "[YEAR]", "March" $\to$ "[MONTH]")
- $\mathcal{T}_4(x)$: Complete semantic anonymization (mask all named entities, years, and specific policy names)

We compute the Jensen-Shannon Divergence between the output probability distributions on original vs. masked inputs:
$$L_{\text{mask}}^{(k)}(M) = \frac{1}{N} \sum_{i=1}^N \mathcal{D}_{\text{JS}}\left( P_M(\cdot \mid \mathcal{T}_0(x_i)) \ \parallel \ P_M(\cdot \mid \mathcal{T}_k(x_i)) \right)$$
where:
$$\mathcal{D}_{\text{JS}}(P \parallel Q) = \frac{1}{2} \mathcal{D}_{\text{KL}}\left(P \parallel \frac{P+Q}{2}\right) + \frac{1}{2} \mathcal{D}_{\text{KL}}\left(Q \parallel \frac{P+Q}{2}\right)$$
We decompose behavioral leakage into:
$$L_{\text{entity}} = L_{\text{mask}}^{(1)}, \quad L_{\text{date}} = L_{\text{mask}}^{(3)} - L_{\text{mask}}^{(1)}, \quad L_{\text{total\_mask}} = L_{\text{mask}}^{(4)}$$

#### 2. Counterfactual Continuation Inversion ($L_{\text{cf}}$)
We construct paired counterfactuals $(x_i^{\text{factual}}, x_i^{\text{inverted}})$, where the forward-looking clause is syntactically inverted (e.g., "anticipates that ongoing increases in the target range will be appropriate" $\to$ "anticipates that ongoing reductions in the target range will be appropriate").
If a model has memorized the true historical rate trajectory, its prediction on $x_i^{\text{inverted}}$ will resist flipping toward the inverted meaning:
$$L_{\text{cf}}(M) = \mathbb{E}\left[ \mathbb{I}\left( \hat{y}(x_i^{\text{inverted}}) = \hat{y}(x_i^{\text{factual}}) \right) \right]$$
High inertia on inverted statements indicates historical memorization overriding linguistic comprehension.

---

## 4. Variable 3: Leakage-Induced Economic Effect ($E_L$)

### 4.1 Mathematical Formulation
$E_L$ measures the financial alpha generated exclusively by temporal contamination. It is defined strictly as the **performance differential between the contaminated model $M_L$ and the clean control model $M_C$** under identical execution rules and market constraints.

Let $s_t(M) \in [-1, +1]$ be the continuous stance score generated by model $M$ on meeting day $t$:
$$s_t(M) = P_M(\text{Hawkish} \mid x_t) - P_M(\text{Dovish} \mid x_t)$$
Let the trading strategy map stance into an asset position $w_t \in \{-1, 0, +1\}$:
$$w_t(M) = \begin{cases} +1, & \text{if } s_t(M) > \tau \\ -1, & \text{if } s_t(M) < -\tau \\ 0, & \text{otherwise} \end{cases}$$
where $\tau > 0$ is a fixed execution threshold.

Let $r_{t, t+\Delta t}$ be the forward asset return (e.g., 2Y Treasury futures or SPY ETF return). The realized strategy return is:
$$R_t(M) = w_t(M) \cdot r_{t, t+\Delta t} - c_{\text{trans}} \cdot |w_t(M) - w_{t-1}(M)|$$
where $c_{\text{trans}}$ is the transaction cost (set to 5 bps for equities, 2 bps for Treasury futures).

We define $E_L$ across three standard financial metrics:

#### 1. Delta Sharpe Ratio ($\Delta \text{SR}$)
$$E_L^{\text{SR}} = \text{Sharpe}(M_L) - \text{Sharpe}(M_C) = \frac{\bar{R}(M_L) - r_f}{\sigma(R(M_L))} - \frac{\bar{R}(M_C) - r_f}{\sigma(R(M_C))}$$

#### 2. Delta Information Coefficient ($\Delta \text{IC}$)
$$E_L^{\text{IC}} = \text{RankCorr}\left(s_t(M_L), r_{t, t+\Delta t}\right) - \text{RankCorr}\left(s_t(M_C), r_{t, t+\Delta t}\right)$$

#### 3. Delta Long-Short Spread ($\Delta \text{LS}$)
$$E_L^{\text{LS}} = \bar{R}_{\text{Long-Short}}(M_L) - \bar{R}_{\text{Long-Short}}(M_C)$$

### 4.2 Statistical Significance Testing
To confirm that an observed $E_L > 0$ is statistically significant and not a random backtest artifact:
1. **Ledoit & Wolf (2008) Paired Sharpe Difference Test**: Tests $H_0: \text{SR}(M_L) - \text{SR}(M_C) = 0$ using HAC kernel covariance estimation.
2. **Deflated Sharpe Ratio (DSR)**: Adjusts for strategy selection bias across hyperparameter trials $N_{\text{trials}}$:
$$\text{DSR} = \Phi\left( \frac{(\widehat{\text{SR}} - \text{SR}_0) \sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{\text{SR}} + \frac{\hat{\gamma}_4 - 1}{4} \widehat{\text{SR}}^2}} \right)$$
where $\hat{\gamma}_3, \hat{\gamma}_4$ are return skewness and kurtosis.

---

## 5. Variable 4: Temporal Robustness ($R_T$, Control Variable)

Temporal robustness captures model generalization under macroeconomic concept drift across time:
$$R_T(\Delta t) = C\left(M; \mathcal{D}_{t_{\text{cutoff}} + \Delta t}\right) - C\left(M; \mathcal{D}_{t_{\text{cutoff}}}\right)$$
where $\mathcal{D}_{t_{\text{cutoff}} + \Delta t}$ represents evaluation slices progressively further from the pretraining cutoff (e.g., 2019, 2020, 2021, 2022, 2023, 2024).

Notice:
$$R_T(\Delta t) \le 0 \quad \text{reflects natural performance degradation over time.}$$
**It is NOT leakage.** Including $R_T$ allows us to decompose observed post-cutoff test accuracy into:
$$\text{Observed Competence} = C_0 + R_T(\Delta t) + \beta_{\text{leak}} \cdot L$$

---

## 6. Null Models & Proof of Pathological Cases

To verify that our metric definitions do not suffer from mathematical degeneracies, we test four edge-case models:

### 6.1 Pathological Case 1: The Constant-Output Model ($M_0$)
Consider a trivial model that outputs a fixed class or zero score regardless of input:
$$M_0(x) = \text{Neutral} \ (s(x) = 0), \quad \forall x$$

#### Theorem 1 (Degeneracy Invariance of $M_0$)
Under the formalizations above, the constant-output model $M_0$ satisfies:
$$L(M_0) = 0, \quad E_L(M_0) = 0, \quad C(M_0) \approx 0$$

*Proof*:
1. **Leakage $L(M_0)$**:
   - Representation: The hidden state $h_{M_0}(x) = \mathbf{c}$ is constant for all inputs. The covariance with future targets is zero:
     $$\text{Cov}\left(h_{M_0}(x), Y^{\text{future}}\right) = 0 \implies \text{Score}(g_\phi(h_{M_0}), Y^{\text{future}}) = \text{Chance Baseline}$$
     $$\therefore L_{\text{repr}}(M_0) = 0$$
   - Masking: For any transformation $\mathcal{T}_k$, $P_{M_0}(y \mid \mathcal{T}_0(x)) = P_{M_0}(y \mid \mathcal{T}_k(x)) = [0, 1, 0]^T$.
     $$\mathcal{D}_{\text{JS}}\left(P \parallel P\right) = 0 \implies L_{\text{mask}}(M_0) = 0$$
   - $\therefore L(M_0) = 0$.
2. **Economic Effect $E_L(M_0)$**:
   - Stance $s_t(M_0) = 0 \implies w_t = 0$ for all $t$. The strategy holds no position $\implies R_t \equiv 0$.
   - When compared against clean control (or evaluated relative to null):
     $$\Delta \text{Sharpe} = 0, \quad \Delta \text{IC} = 0 \implies E_L(M_0) = 0$$
3. **Task Competence $C(M_0)$**:
   - For a 3-class test set with non-zero true Hawkish and Dovish samples:
     $$\text{Recall}_{\text{Hawkish}} = 0 \implies F_{1, \text{Hawkish}} = 0$$
     $$\text{Recall}_{\text{Dovish}} = 0 \implies F_{1, \text{Dovish}} = 0$$
     $$C_{\text{Macro-F1}}(M_0) = \frac{1}{3} (0 + F_{1, \text{Neutral}} + 0) \le 0.33$$
     $$\text{Multiclass MCC}(M_0) \equiv 0.00$$
   - $\therefore C(M_0) \approx 0$.
$\blacksquare$

**Significance**: This proves that zero leakage ($L=0$) does NOT produce a high score. A composite score $Score = aC - bL$ would have awarded $M_0$ a high penalty-free score, whereas our Pareto formulation places $M_0$ at $(0, 0, 0)$, clearly identified as worthless.

---

### 6.2 Pathological Case 2: The Irrelevant Memorizer ($M_{\text{irrelevant}}$)
A model that memorizes post-cutoff facts that are irrelevant to monetary policy (e.g., memorized 2021 celebrity news or Super Bowl scores).
- Hidden state decodes future irrelevant trivia $\implies L_{\text{trivia}} > 0$.
- However, representations do not correlate with policy surprise $\implies L_{\text{repr}}^{\text{FOMC}} \approx 0$.
- Monetary classifications and trading signals do not change $\implies E_L \approx 0$.
- Pareto position: $(L_{\text{domain-irrelevant}} > 0, \ E_L = 0, \ C = C_0)$.
- **Conclusion**: Confirms that generic memorization does not automatically manufacture false alpha.

---

### 6.3 Pathological Case 3: The Critical-Pivot Memorizer ($M_{\text{pivot}}$)
A model that has a very small overall leakage footprint ($L$ small, passes 95% of entity masks), but specifically memorized the dates of the March 2020 emergency rate cuts or the March 2022 rate hike liftoff.
- Overall behavioral divergence $L_{\text{mask}}$ is low ($L \approx 0.05$).
- However, during the 4 critical pivot meetings, it perfectly anticipates policy moves:
  $$E_L \gg 0 \quad (\Delta \text{Sharpe} > 1.5)$$
- Pareto position: $(L \approx 0.05, \ E_L \gg 0, \ C \approx 0.70)$.
- **Conclusion**: Explains why measuring $\frac{dE_L}{dL}$ (marginal false alpha per unit leakage) is critical.

---

### 6.4 Ideal Scientific Model ($M_{\text{ideal}}$)
A model with high structural economic reasoning ability trained with strict temporal discipline:
$$\lim (L \to 0, \ E_L \to 0, \ C \to 1.0)$$
- Strong Macro-F1 and MCC ($C > 0.80$).
- Probes fail to outperform clean baseline on forward targets ($L_{\text{repr}} \le 0$).
- Output invariant to date/entity anonymization ($L_{\text{mask}} \approx 0$).
- Generates legitimate economic value through genuine linguistic accuracy:
  $$\alpha_{\text{observed}} = \alpha_{\text{legitimate}} + 0$$

---

## 7. Operational Summary & Verification Harness

| Metric | Code Symbol | Primary Implementation | Expected Null Range | Significance Test |
| :--- | :--- | :--- | :--- | :--- |
| **Task Competence** | `C` | `evaluate_competence()` | $C_{\text{F1}} \in [0.25, 0.33], \text{MCC} = 0$ | Block Bootstrap 95% CI |
| **Representational Leakage** | `L_repr` | `evaluate_representational_leakage()` | $\Delta \text{Score} \le 0$ | Paired permutation test |
| **Behavioral Leakage** | `L_mask` | `evaluate_behavioral_leakage()` | $\text{JS-Div} \le 0.01$ | Wilcoxon signed-rank |
| **Economic Effect** | `E_L` | `evaluate_economic_effect()` | $\Delta \text{SR} \approx 0, \Delta \text{IC} \approx 0$ | Ledoit-Wolf & Deflated SR |
| **Temporal Robustness** | `R_T` | `evaluate_temporal_robustness()` | $D(\Delta t) \in [-0.30, 0]$ | Slope of decay regression |
