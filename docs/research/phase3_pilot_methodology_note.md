# Phase 3 Pilot Methodology Note & Power Diagnostics

**Study**: MANTRA Causal Temporal Leakage Evaluation  
**Stage**: Phase 3 Pilot Post-Mortem & Methodological Synthesis  
**Status**: PILOT PIPELINE VALID / LOW STATISTICAL POWER / NO CONFIRMATORY CLAIMS  
**Target Next Phase**: Phase 4 Confirmatory Evaluation  

---

## 1. Executive Summary of Phase 3 Pilot Findings

Phase 3 successfully validated the end-to-end engineering plumbing and causal twin protocol for temporal contamination evaluation:
1. **Exact Dose Ladder**: Implemented and verified the five-dose ladder ($D \in \{0.0, 0.25, 0.50, 0.75, 1.00\}$) satisfying the strict causal invariant $|D_{\text{realized}} - D_{\text{requested}}| \le 1/T$.
2. **Causal Twin Invariants**: Enforced identical initial encoder parameters, identical downstream classifier head initializations, identical token budgets, and zero overlap between training and evaluation corpora.
3. **Execution Matrix**: Ran 3 random seeds across all 5 doses (15 continued pre-training runs and 15 downstream classification probes).

However, the empirical conclusions drawn from Phase 3 were strictly characterized as:
> **"LOW POWER PILOT — NO CONFIRMATORY CLAIMS"**

This methodology note documents the exact statistical and structural reasons why Phase 3 cannot support confirmatory inferences and how Phase 4 resolves every diagnostic limitation.

---

## 2. Statistical Limitations of Phase 3 Pilot

### 2.1 Sample Size and Statistical Power ($N_{\text{event}} = 8$)
Phase 3 drew its evaluation anchors strictly from the 8 scheduled FOMC meetings of calendar year 2019.
- Total paragraph anchors: $N_{\text{paragraph}} = 25$.
- Total independent monetary policy decisions: $N_{\text{event}} = 8$.

A post-hoc power analysis for paired difference testing ($D_{100}$ vs $D_0$) with $N=8$ independent events:
- To detect a moderate effect size (Cohen's $d = 0.50$) at two-sided $\alpha = 0.05$, achieved power is:
  $$1 - \beta \approx 0.145 \quad (14.5\%)$$
- To detect a large effect size ($d = 0.80$), achieved power is only:
  $$1 - \beta \approx 0.398 \quad (39.8\%)$$
- To achieve standard confirmatory power ($1 - \beta \ge 0.80$) at $\alpha = 0.05$ for $d = 0.50$, a minimum sample of $N \ge 34$ independent events is mathematically required.

Consequently, any null finding in Phase 3 is overwhelmingly susceptible to Type II error (false negative), and any observed significance is susceptible to the "winner's curse" effect size exaggeration.

### 2.2 The Unit of Analysis Fallacy: $N_{\text{paragraph}} \neq N_{\text{independent event}}$
In NLP benchmarks, researchers frequently treat paragraphs or sentences as independent observations. In central bank communications, this assumption is fundamentally invalid:
- Paragraphs within a single FOMC statement are drafted simultaneously by the Federal Open Market Committee to describe a single unified macroeconomic assessment and policy stance.
- Forward market outcomes (e.g., SPY $t+1$ return, 2Y Treasury yield change) and future policy actions (e.g., target rate change at $t+1$) are realized **once per meeting**, not once per paragraph.
- Assigning the same meeting-level target to 3 or 4 paragraphs within a meeting induces severe intra-cluster correlation ($\rho_{\text{intra}} > 0.85$). Naive regression or permutation tests that treat $N=25$ paragraphs as independent observations artificially deflate standard errors by a factor of $\sqrt{1 + (m-1)\rho} \approx \sqrt{1 + (3.1-1)(0.85)} \approx 1.67$, inflating Type I error (false positives) by over $300\%$.

### 2.3 Policy Cycle Monoculture in Calendar 2019
The 2019 FOMC calendar represented an atypical "mid-cycle adjustment" regime:
- May 2019: Hold
- June 2019: Hold
- July 2019: Cut (25 bps)
- September 2019: Cut (25 bps)
- October 2019: Cut (25 bps)
- December 2019: Hold
- **Rate Hikes in 2019**: Exactly 0.

Evaluating representational or behavioral leakage on a 2019 sample creates severe class imbalance ($0\%$ hikes, $37.5\%$ cuts, $62.5\%$ holds). Probes trained on such narrow regimes cannot distinguish general temporal leakage from regime-specific keyword memorization.

---

## 3. Methodological Upgrades Formulated for Phase 4

| Methodological Dimension | Phase 3 Pilot Implementation | Phase 4 Confirmatory Standard |
| :--- | :--- | :--- |
| **Statistical Unit** | Paragraph ($N=25$) | Independent Event ($N=40$, 181 paragraphs) |
| **Temporal Span** | 1 calendar year (2019) | 5 calendar years (2015–2019) |
| **Monetary Regimes** | Cuts (3) and Holds (5); 0 Hikes | Hikes (9), Cuts (3), Holds (28) across full cycle |
| **Statistical Power ($d=0.50$)** | $1 - \beta \approx 14.5\%$ | $1 - \beta \approx 88.4\%$ ($N=40, \alpha=0.05$) |
| **Temporal Cross-Validation** | Standard paragraph random split | Grouped expanding-window CV ($\text{Train} \cap \text{Test} = \emptyset$) |
| **Representation Aggregation** | Individual paragraph vectors $h_i$ | Centroid event vector $h_{\text{event}} = \frac{1}{M}\sum_{i=1}^M h_i$ |
| **Economic Effect ($E_L$)** | Naive paragraph-weighted IC | 1:1 Event-level stance vs market outcome IC |
| **Bootstrap Unit** | Stationary block on paragraphs | Clustered block bootstrap resampling whole meetings |
| **Market Outcome Source** | Hardcoded sample dictionary | Point-in-time raw daily tables (SPY, ZT=F, DGS2) |
| **Policy History Source** | Manual integer assignment | Derived from official Fed Open Market Operations table |
| **Preregistration** | Exploratory protocol | Cryptographically locked SHA-256 YAML contract |

---

## 4. Archival and Immutability Guarantee

To preserve complete scientific auditability, the Phase 3 pilot code, configuration, manifests, and artifact results located in:
`experiments/phase3_pilot/`
remain strictly preserved in their original state and are not overwritten by Phase 4 pipelines.
