# Formal Preregistration: Phase 4 Confirmatory Temporal Leakage Study

**Protocol Title**: Machine-Verifiable Point-in-Time Evaluation of Representational and Behavioral Temporal Contamination in Financial Central Bank Language Models  
**Preregistration Status**: LOCKED PRIOR TO CONFIRMATORY COMPUTE  
**Preregistration Specification Version**: 1.0.0  
**Repository**: `RubiscoYHY/MANTRA`  
**Git Base Commit**: `d439c256b4e29b9abe05f93f9085edb7639cce66`  
**Date of Lock**: 2026-09-20  

---

## 1. Study Overview & Confirmatory Invariants

This document formalizes the preregistered hypotheses, sample specification, statistical analysis plan, decision criteria, and compute execution bounds for the Phase 4 Confirmatory Study.

In accordance with confirmatory empirical standards:
1. All evaluation samples, targets, market outcomes, and statistical routines are frozen prior to model execution.
2. Full-scale Phase 4B model training is locked until this document's SHA-256 hash is recorded and verified by automated CI gates.
3. No subjective or exploratory alterations to metric definitions, target horizons, or exclusion filters are permitted post-hoc.

---

## 2. Sample & Population Specification

### 2.1 Historical Anchor Dataset (Pre-Cutoff Population)
- **Time Horizon**: 2015-01-01 to 2019-12-31 (5 full calendar years).
- **Sampling Frame**: Official Federal Reserve Open Market Committee (FOMC) scheduled meeting statements.
- **Independent Decision Events ($N$)**: 40 events.
- **Paragraph Anchors ($M$)**: 181 canonical paragraphs.
- **Within-Event Paragraph Ratio**: Mean $4.525$ paragraphs per meeting (min 3, max 7).
- **Temporal Cutoff ($T_{\text{cutoff}}$)**: `2019-12-31T23:59:59Z`.
- **Exclusion Registry**: 2 unscheduled emergency meetings (2020-03-03 and 2020-03-15) are strictly excluded from scheduled decision modeling and logged in `exclusion_log.jsonl`.

### 2.2 Contamination Treatment Corpus (Post-Cutoff Population)
- **Time Horizon**: 2020-01-01 to 2022-12-31.
- **Earliest Contamination Timestamp**: `2020-01-29T19:00:00Z`.
- **Temporal Gap Buffer**: 48 calendar days ($\max(T_{\text{anchors}}) < \min(T_{\text{contamination}})$).
- **Corpus Content**: Official Federal Reserve statements, press conferences, and policy releases during the 2020–2022 pandemic easing and subsequent tightening cycle.

---

## 3. Preregistered Hypotheses

### Primary Hypothesis $H_{1,\text{repr}}$ (Representational Leakage)
- **Null Hypothesis $H_{0,\text{repr}}$**: Continued masked language model pre-training on future post-cutoff text yields no increase in linear probe predictiveness for future monetary policy actions:
  $$L_{\text{repr}} \le 0$$
- **Alternative Hypothesis $H_{1,\text{repr}}$**: Continued pre-training on post-cutoff text imparts directional future information into intermediate sentence representations, producing a statistically significant event-level probe advantage:
  $$L_{\text{repr}} = \text{Score}_{\text{leak}} - \text{Score}_{\text{clean}} > 0$$
- **Primary Statistical Test**: Non-parametric paired event sign-flip permutation test ($B = 2,000$ permutations). Significance threshold: $p < 0.05$.

### Secondary Hypothesis $H_{1,\text{behavior}}$ (Behavioral Leakage)
- **Null Hypothesis $H_{0,\text{behavior}}$**: Post-cutoff contamination does not alter model stance sensitivity to forward-looking policy keywords:
  $$L_{\text{behavior}} \le 0$$
- **Alternative Hypothesis $H_{1,\text{behavior}}$**: Contaminated models exhibit elevated behavioral sensitivity shifts across forward-looking policy tokens:
  $$L_{\text{behavior}} = \bar{S}_{\text{event}}(M_D) - \bar{S}_{\text{event}}(M_{D0}) > 0$$

### Secondary Hypothesis $H_{1,\text{econ}}$ (Economic Effect)
- **Null Hypothesis $H_{0,\text{econ}}$**: The difference in Information Coefficient ($E_L = \text{IC}_{\text{leak}} - \text{IC}_{\text{clean}}$) against post-event 2Y Treasury yield change is zero or negative:
  $$E_L^{2Y} \le 0$$
- **Alternative Hypothesis $H_{1,\text{econ}}$**: Contaminated model stance scores achieve superior rank correlation with 2Y Treasury post-event adjustments:
  $$E_L^{2Y} > 0$$
- **Testing Standard**: Two-sided $95\%$ Clustered Block Bootstrap confidence interval strictly excluding zero.
- **Endpoint Hierarchy**:
  - Primary Economic Endpoint: 2Y Treasury daily yield change ($\Delta y_{2Y} \text{ in bps}$).
  - Secondary Exploratory Endpoint: SPY daily post-event return ($r_{SPY, t+1}$).

---

## 4. Statistical Analysis Plan & Methodological Rules

### 4.1 Definition of the Primary Statistical Unit
- **The Event is the Unit ($N=40$)**: The paragraph is explicitly rejected as the statistical unit.
- **Representation Aggregation**: All paragraph embeddings $h_{i}$ belonging to meeting event $e$ are aggregated to a single meeting centroid vector:
  $$h_{\text{event}, e} = \frac{1}{M_e} \sum_{i=1}^{M_e} h_{i}$$
- **Equal Meeting Weighting**: In economic correlations and probe evaluations, each FOMC meeting receives exactly 1 vote. Meetings with 7 paragraphs do not exert greater statistical influence than meetings with 3 paragraphs.

### 4.2 Cross-Validation: Grouped Expanding-Window Folds
- Probing models are evaluated using expanding-window temporal cross-validation with 4 chronological splits.
- For every split $k$:
  $$\text{TrainEvents}_k \cap \text{TestEvents}_k = \emptyset$$
  $$\max(\text{Dates}_{\text{TrainEvents}_k}) < \min(\text{Dates}_{\text{TestEvents}_k})$$
- Zero paragraph from any test meeting is permitted in the training set of the linear probe.

### 4.3 Primary Probe Formulation & Target Class Handling
Across the 40 historical meetings (2015–2019), the empirical distribution of policy actions is:
- Holds ($0$): 28 meetings ($70.0\%$)
- Rate Hikes ($+1$): 9 meetings ($22.5\%$)
- Rate Cuts ($-1$): 3 meetings ($7.5\%$)

To prevent metric instability caused by the low frequency of cuts ($N=3$), the preregistered analysis specifies:
1. **Primary Metric**: Spearman Rank Correlation of continuous Ridge regression predictions against the exact policy target rate change:
   $$\Delta r \in \{-0.25, 0.0, +0.25, +0.50\}$$
2. **Co-Primary Binary Probe**: Macro-F1 of Ridge Classifier predicting Policy Change ($\pm 1$) vs Policy Hold ($0$).
3. **Secondary Exploratory Probe**: 3-class ordinal classification with clustered standard errors.

### 4.4 Multiple Testing Correction
False Discovery Rate (FDR) across all secondary endpoints (behavioral sensitivity, secondary market endpoints, individual dose slices) will be controlled using the Benjamini-Hochberg procedure at nominal $\alpha = 0.05$.

---

## 5. Statistical Power Analysis

A priori power calculation conducted via Monte Carlo simulation for $N=40$ independent clustered events:
- Two-sided significance level: $\alpha = 0.05$.
- Effect size: Cohen's $d = 0.50$ (moderate standardized mean difference between $D_{100}$ and $D_0$).
- Event correlation: intra-cluster correlation $\rho = 0.85$ modeled directly by event centroid aggregation.
- **Achieved Statistical Power**:
  $$1 - \beta = 0.884 \quad (88.4\%)$$
The study is formally certified as adequately powered for confirmatory inference.

---

## 6. Compute Execution Bounds (Phase 4B Constraints)

When the Phase 4A Preregistration Gate is approved and Phase 4B execution commences, training must strictly adhere to the following parameter bounds:

| Parameter | Confirmatory Bound |
| :--- | :--- |
| **Dose Ladder ($D$)** | Exactly 5 levels: $\{0.00, 0.25, 0.50, 0.75, 1.00\}$ |
| **Dose Invariant** | $|D_{\text{realized}} - D_{\text{requested}}| \le 1/T$ |
| **Random Seeds** | Exactly 5 seeds: $\{13, 42, 87, 123, 2024\}$ |
| **Token Budget ($T$)** | 256,000 tokens (500 blocks $\times$ 512 tokens) |
| **Base Model** | `ProsusAI/finbert` (verified revision hash) |
| **MLM Optimizer** | AdamW, $\text{lr} = 5 \times 10^{-5}$, weight decay $0.01$ |
| **Masking Rate** | $15\%$ deterministic schedule matching seed |
| **Total MLM Runs** | Exactly 25 runs (5 doses $\times$ 5 seeds) |
| **Classifier Probes** | Exactly 25 downstream runs with identical initial heads |
| **Compute Execution Guard** | Blocked until CI checks verify preregistration YAML match |

---

## 7. Machine-Verifiable Lock Invariant

This document is cryptographically referenced by `configs/phase4_preregistration.yaml` via its normalized SHA-256 hash. Any modification to the text of this document without an authorized version bump will cause `verify_preregistration_lock()` and `tests/test_phase4a_gate.py` to fail, preventing execution.
