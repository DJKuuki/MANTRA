# Formal Preregistration: Phase 4 Confirmatory Temporal Leakage Study (v1.2.0)

**Protocol Title**: Machine-Verifiable Point-in-Time Evaluation of Representational and Behavioral Temporal Contamination in Financial Central Bank Language Models  
**Preregistration Status**: LOCKED PRIOR TO CONFIRMATORY COMPUTE  
**Preregistration Specification Version**: 1.2.0 (Protocol Closure Finalization)  
**Repository**: `RubiscoYHY/MANTRA`  
**Git Base Revision**: `4556d13015211d73dccd3fdd39d39232506f3e43`  
**Date of Lock**: 2026-09-20  

> [!IMPORTANT]
> **Protocol Closure Declaration**:
> This version (v1.2.0) incorporates protocol corrections approved during the Phase 4A Protocol Closure audit prior to any Phase 4B confirmatory compute:
> 1. Enforces strict information availability (`available_time <= 2019-12-31T23:59:59Z`) for clean sham corpus eligibility, excluding `fomc-minutes-2019-12-11` (released 2020-01-01).
> 2. Sets contamination availability window to `2020-01-29T19:00:00Z` to `2023-01-04T19:00:00Z`, reflecting the January 2023 release of the December 2022 meeting minutes.
> 3. Completes 50/50 raw source verification for all contamination documents with deterministic canonical HTML reconstruction.
> 4. Explicitly derives true continuous future rate change ($y_e = \Delta r_e$) from `policy_history.csv` before/after rate bounds.
> 5. Separates forced repetition ratio from token vocabulary diversity.
> 6. Enforces source-tree code freeze via `source_tree_hash` and strict human authorization gate.
> Full-scale model training remains strictly locked pending external human authorization.

---

## 1. Study Overview & Confirmatory Invariants

This document formalizes the preregistered hypotheses, sample specification, statistical analysis plan, decision criteria, and compute execution bounds for the Phase 4 Confirmatory Study.

In accordance with confirmatory empirical standards:
1. All evaluation samples, targets, market outcomes, contamination documents, and statistical routines are frozen prior to model execution.
2. Full-scale Phase 4B model training is locked until this document's SHA-256 hash, all controlled protocol artifacts, and the source tree hash are verified by automated CI gates.
3. No subjective or exploratory alterations to metric definitions, target horizons, or exclusion filters are permitted post-hoc.

---

## 2. Sample & Population Specification

### 2.1 Historical Anchor Dataset (Pre-Cutoff Population)
- **Time Horizon**: 2015-01-01 to 2019-12-31 (5 full calendar years).
- **Sampling Frame**: Official Federal Reserve Open Market Committee (FOMC) scheduled meeting statements.
- **Independent Decision Events ($N$)**: 40 events.
- **Paragraph Anchors ($M$)**: 181 canonical paragraphs (100% verified verbatim in cached official Federal Reserve HTML sources).
- **Within-Event Paragraph Ratio**: Mean $4.525$ paragraphs per meeting (min 3, max 7).
- **Temporal Cutoff ($T_{\text{cutoff}}$)**: `2019-12-31T23:59:59Z`.
- **Latest Anchor Release Time**: `2019-12-11T19:00:00Z`.
- **Exclusion Registry**: 2 unscheduled emergency meetings (2020-03-03 and 2020-03-15) are strictly excluded from scheduled decision modeling and logged in `exclusion_log.jsonl`.

### 2.2 Pre-Cutoff Sham Treatment Corpus
- **Dataset Path**: `data/research/fomc/phase4_pre_cutoff/documents.jsonl`
- **Document Count**: 63 official Federal Reserve documents (40 scheduled statements 2015–2019, 23 meeting minutes 2017–2019).
- **Total Words**: 243,001 words (~315,901 tokens).
- **Temporal Availability Range**: `2015-01-28T19:00:00Z` to `2019-12-11T19:00:00Z`.
- **Eligibility Invariant**: Strictly `available_time <= 2019-12-31T23:59:59Z` (information availability, not meeting date, determines eligibility).
- **Repetition Ratio Feasibility**: Forced repetition ratio = 0.00 <= 0.20 for 256k token budget at D0.

### 2.3 Contamination Treatment Corpus (Post-Cutoff Population)
- **Dataset Path**: `data/research/fomc/phase4_contamination/documents.jsonl`
- **Document Count**: 50 official Federal Reserve post-cutoff documents (100% raw source and canonical text verified).
  - Primary: 23 scheduled monetary policy statements, 2 unscheduled emergency statements, 1 policy strategy statement (Jackson Hole 2020).
  - Secondary: 24 official meeting minutes.
- **Total Words**: 237,273 words (~308,454 tokens).
- **Earliest Contamination Release Time**: `2020-01-29T19:00:00Z`.
- **Latest Contamination Release Time**: `2023-01-04T19:00:00Z` (reflecting actual release of December 2022 meeting minutes).
- **Temporal Gap Buffer**: Strict 48 calendar days ($\max(T_{\text{anchors}}) < \min(T_{\text{contamination}})$).
- **Document & Text Isolation**: Zero overlap with historical anchor document IDs ($0 / 50$) and zero exact text hash collisions ($0 / 50$).

---

## 3. Preregistered Hypotheses

### Primary Hypothesis $H_{1,\text{repr}}$ (Representational Leakage)
- **Null Hypothesis $H_{0,\text{repr}}$**: Continued masked language model pre-training on future post-cutoff text yields no increase in linear probe predictiveness for future monetary policy actions:
  $$L_{\text{repr}} \le 0$$
- **Alternative Hypothesis $H_{1,\text{repr}}$**: Continued pre-training on post-cutoff text imparts directional future information into intermediate sentence representations, producing a statistically significant event-level probe advantage:
  $$L_{\text{repr}} = \text{Score}_{\text{leak}} - \text{Score}_{\text{clean}} > 0$$
- **Primary Statistical Unit**: Out-of-Sample (OOS) FOMC Event ($N_{\text{OOS}} = 32$).
- **Primary Contrast Statistic**: Paired event-level absolute error improvement:
  $$d_e = |y_e - \hat{y}_{\text{clean}}(e)| - |y_e - \hat{y}_{\text{leak}}(e)|$$
- **Primary Statistical Test**: Non-parametric paired event sign-flip permutation test over $\{d_e\}_{e=1}^{N_{\text{OOS}}}$ ($B = 2,000$ permutations). Significance threshold: $\alpha = 0.05$.
- **Effect Metric**: Delta Spearman Rank Correlation ($\Delta \rho = \rho_{\text{leak}} - \rho_{\text{clean}}$).

### Secondary Hypothesis $H_{1,\text{behavior}}$ (Behavioral Leakage)
- **Null Hypothesis $H_{0,\text{behavior}}$**: Post-cutoff contamination does not alter model stance sensitivity to forward-looking policy keywords:
  $$L_{\text{behavior}} \le 0$$
- **Alternative Hypothesis $H_{1,\text{behavior}}$**: Contaminated models exhibit elevated behavioral sensitivity shifts across forward-looking policy tokens:
  $$L_{\text{behavior}} = \bar{S}_{\text{event}}(M_D) - \bar{S}_{\text{event}}(M_{D0}) > 0$$
- **Multiplicity Adjustment**: Benjamini-Hochberg FDR correction at $\alpha = 0.05$.

### Secondary Hypothesis $H_{1,\text{econ}}$ (Economic Effect)
- **Null Hypothesis $H_{0,\text{econ}}$**: The difference in Information Coefficient ($E_L = \text{IC}_{\text{leak}} - \text{IC}_{\text{clean}}$) against post-event 2Y Treasury yield change is zero or negative:
  $$E_L^{2Y} \le 0$$
- **Alternative Hypothesis $H_{1,\text{econ}}$**: Contaminated model stance scores achieve superior rank correlation with 2Y Treasury post-event adjustments:
  $$E_L^{2Y} > 0$$
- **Testing Standard**: Two-sided $95\%$ Clustered Block Bootstrap confidence interval strictly excluding zero.
- **Endpoint Hierarchy**:
  - Primary Economic Endpoint: 2Y Treasury daily yield change ($\Delta y_{2Y} \text{ in bps}$ via FRED DGS2).
  - Secondary Exploratory Endpoint: SPY daily post-event return from event close ($r_{SPY, t \to t+1}$).

---

## 4. Statistical Analysis Plan & Methodological Rules

### 4.1 Definition of the Primary Statistical Unit
- **The Event is the Unit ($N=40$, $N_{\text{OOS}}=32$)**: The paragraph is explicitly rejected as the statistical unit. CV folds are explicitly rejected as the permutation unit.
- **Representation Aggregation**: All paragraph embeddings $h_{i}$ belonging to meeting event $e$ are aggregated to a single meeting centroid vector:
  $$h_{\text{event}, e} = \frac{1}{M_e} \sum_{i=1}^{M_e} h_{i}$$
- **Equal Meeting Weighting**: Each evaluated FOMC meeting receives exactly 1 vote. Meetings with 7 paragraphs do not exert greater statistical influence than meetings with 3 paragraphs.

### 4.2 Cross-Validation: Grouped Expanding-Window Folds
- Probing models are evaluated using expanding-window temporal cross-validation with 4 chronological splits on 40 ordered events (`min_train_events = 8`).
- Test fold sizes: 8, 8, 8, 8 events.
- Evaluated Out-of-Sample sample size: $N_{\text{OOS}} = 32$ events.
- For every split $k$:
  $$\text{TrainEvents}_k \cap \text{TestEvents}_k = \emptyset$$
  $$\max(\text{Dates}_{\text{TrainEvents}_k}) < \min(\text{Dates}_{\text{TestEvents}_k})$$
- Zero paragraph from any test meeting is permitted in the training set of the linear probe.

### 4.3 Primary Probe Formulation & Target Class Handling
- **Primary Continuous Rate Target**: Explicitly modeled using Ridge Regression with $L_2$ regularization ($\alpha=1.0$).
  $$\Delta r \in \{-0.25, 0.0, +0.25, +0.50\}$$
  Task type is explicitly specified as continuous regression, prohibiting heuristic guessing based on unique value cardinality.
- **Co-Primary Binary Probe**: Macro-F1 of Ridge Classifier predicting Policy Change ($\pm 1$) vs Policy Hold ($0$).

---

## 5. Statistical Power Analysis

A priori statistical power validated via empirical Monte Carlo simulation directly executing the preregistered event-level sign-flip permutation test:
- Number of Out-of-Sample events: $N_{\text{OOS}} = 32$.
- Effect size: Cohen's $d = 0.50$ (moderate standardized mean difference).
- Alpha: $\alpha = 0.05$ (two-sided).
- Number of simulations: 1,000 runs ($B = 1,000$ permutations per run).
- **Simulated Statistical Power**:
  $$1 - \beta = 0.864 \quad (86.4\%)$$
The planned confirmatory sample size achieves adequate statistical power (> 80%) under standard moderate effect size assumptions.

---

## 6. Compute Execution Bounds (Phase 4B Constraints)

When the Phase 4A Preregistration Gate is approved and Phase 4B execution commences, training must strictly adhere to the following parameter bounds:

| Parameter | Confirmatory Bound |
| :--- | :--- |
| **Dose Ladder ($D$)** | Exactly 5 levels: $\{0.00, 0.25, 0.50, 0.75, 1.00\}$ |
| **Dose Invariant** | $|D_{\text{realized}} - D_{\text{requested}}| \le 1/T$ |
| **Random Seeds** | Exactly 5 seeds: $\{13, 42, 87, 123, 2024\}$ |
| **Token Budget ($T$)** | 256,000 tokens (500 blocks $\times$ 512 tokens) |
| **Base Model Checkpoint** | `ProsusAI/finbert` |
| **Base Model Revision** | `4556d13015211d73dccd3fdd39d39232506f3e43` |
| **MLM Optimizer** | AdamW, $\text{lr} = 5 \times 10^{-5}$, weight decay $0.01$ |
| **Masking Rate** | $15\%$ deterministic schedule matching seed |
| **Total MLM Runs** | Exactly 25 runs (5 doses $\times$ 5 seeds) |
| **Classifier Probes** | Exactly 25 downstream runs with identical initial heads |
| **Downstream Training Recipe** | Epochs: 3, Batch: 16, LR: $2 \times 10^{-5}$, Warmup: $10\%$, SeqLen: 128 |
| **Same-Seed Causal Symmetry** | Shared base hash, shared head init hash, shared mask schedule |
| **Max Forced Repetition Ratio** | $\le 0.20$ across all doses (measured as shortfall cycling) |
| **Code Freeze Invariant** | Bound to `source_tree_hash` and clean git state |
| **Compute Execution Guard** | Blocked until human authorization manifest `configs/phase4_execution_authorization.json` is signed |

---

## 7. Machine-Verifiable Lock Invariant

This document is cryptographically referenced by `configs/phase4_preregistration.yaml` and `configs/phase4_protocol_lock.json` via its canonical LF-normalized SHA-256 hash. Any modification to the text of this document without an authorized version bump will cause `verify_phase4_protocol_lock()` and `tests/test_phase4a_gate.py` to fail, preventing execution.
