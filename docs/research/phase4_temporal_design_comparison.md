# Phase 4 Temporal Design Comparison & Architectural Selection

**Study**: MANTRA Causal Temporal Leakage Evaluation  
**Stage**: Phase 4A Confirmatory Preregistration Gate  
**Authors**: Antigravity AI Pair Programming & MANTRA Research Team  
**Status**: DESIGN LOCKED / CANDIDATE A SELECTED  

---

## 1. Context & Architectural Challenge

In measuring temporal contamination in large pretrained language models (LLMs/PLMs), the geometry of the temporal evaluation window dictates both statistical power and compute efficiency.

Phase 4 requires expanding the statistical unit from Phase 3's low-power sample ($N=8$ events in 2019) to a robust sample of $N \ge 40$ independent FOMC decision events spanning at least 5 calendar years.

Two competing architectural designs were evaluated for the Phase 4 confirmatory study:
1. **Candidate A**: Single Unified Historical Window (2015–2019 Anchors, 2020–2022 Contamination)
2. **Candidate B**: Rolling Temporal Walk-Forward Cohorts (e.g., 2015/2016, 2016/2017, ..., 2019/2020)

This document formalizes the comparative trade-off matrix and documents the definitive selection of **Candidate A**.

---

## 2. Comprehensive Trade-off Matrix

| Evaluation Dimension | Candidate A: Single Historical Window (2015–2019) | Candidate B: Rolling Walk-Forward Cohorts | Favored Option |
| :--- | :--- | :--- | :--- |
| **Statistical Sample Power** | **High**: Single unified pool of $N=40$ independent events evaluated simultaneously. Achieves $88.4\%$ power for $d=0.50$ at $\alpha=0.05$. | **Low per Cohort**: Each rolling cohort has only $N=8$ events. Requires post-hoc meta-analytic pooling with unknown inter-cohort correlation. | **Candidate A** |
| **Compute Expenditure** | **Controlled**: 5 doses $\times$ 5 seeds = **25 MLM pre-training runs** + 25 probe fits. | **5x Compute Explosion**: 5 cohorts $\times$ 5 doses $\times$ 5 seeds = **125 MLM pre-training runs** + 125 probe fits. | **Candidate A** |
| **Baseline Model Consistency** | **Identical Baseline**: All doses and seeds diverge from a single canonical pre-cutoff checkpoint ($T_{\text{cutoff}} = \text{2019-12-31}$). | **Shifting Baselines**: Requires retraining or obtaining 5 distinct historical base models with distinct cutoff dates ($T_{\text{cutoff}} \in [2015, \dots, 2019]$). | **Candidate A** |
| **Macro Regime Diversity** | **Full Monetary Cycle**: 9 Rate Hikes, 3 Rate Cuts, 28 Holds. Zero-rate lower bound exit (2015) to quantitative tightening (2018) to easing (2019). | **Fragmented Regimes**: 2015–2016 has 1 hike; 2017–2018 has 7 hikes; 2019 has 0 hikes. Within-cohort probes suffer extreme class degeneracies. | **Candidate A** |
| **Temporal Isolation & Buffer** | **Clean & Absolute**: 48 calendar days of absolute dead-zone between anchor max (`2019-12-11T19:00:00Z`) and contamination min (`2020-01-29T19:00:00Z`). | **Tight Buffers**: Rolling shifts create narrow boundary margins between successive years where forward guidance blurs cutoff edges. | **Candidate A** |
| **Engineering & Verification** | **Verifiable Gate**: Single deterministic event manifest ($N=40$), single anchor manifest ($N=181$), single market outcome matrix. | **High Complexity**: 5 separate manifests, 5 shifting cutoff clocks, cascading failure modes in CI/CD pipeline. | **Candidate A** |

---

## 3. Formal Selection & In-Depth Rationale

### Formal Decision: Formally Adopt Candidate A

Candidate A is formally selected as the sole confirmatory design for Phase 4.

### Detailed Justification:

1. **Power Preservation without Meta-Analytic Confounding**:
   Statistical power in causal inference scales with the degrees of freedom of the evaluated sample. In Candidate B, breaking the 40 events into 5 mini-cohorts of 8 events recreates the exact underpowered pathology that invalidated Phase 3. Pooling 5 rolling cohorts post-hoc requires random-effects meta-regression, which introduces unidentifiable variance components from macro regime shifts. Candidate A enables a single, well-powered ($N=40$), event-clustered, non-parametric permutation test.

2. **Feasibility within Compute Budget**:
   Candidate B requires 125 continued pre-training runs of masked language modeling (at 256k to 1M tokens each). At 25 runs, Candidate A achieves the exact same confirmatory rigor while conserving $80\%$ of compute resources for replication seeds (5 to 10 seeds) and multi-architecture generalization in Phase 4B.

3. **Causal Twin Purity**:
   Candidate A guarantees that the clean model $M_{D0}$ and the contaminated models $M_{D25} \dots M_{D100}$ share an identical historical base state trained strictly on pre-2020 data. Candidate B would require either artificially truncating the base model for earlier cohorts (discarding valid pre-training history) or using a single base model whose cutoff already overlaps with the earlier cohorts' evaluation windows.

4. **Regime Balance for Statistical Probing**:
   In central bank communications, a linear or ridge classification probe requires balance across monetary actions. Candidate A provides 9 hikes, 3 cuts, and 28 holds across 40 meetings. This enables robust directional rate-change continuous regressions ($\Delta r \in [-0.25, 0.0, +0.25]$) and binary policy change probes ($\pm 1$ vs $0$) with adequate support in both classes.

---

## 4. Confirmatory Design Specification for Candidate A

- **Historical Evaluation Horizon**: 2015-01-01 to 2019-12-31 (5 full calendar years).
- **Independent Decision Events ($N$)**: 40 scheduled FOMC meetings.
- **Evaluation Paragraph Anchors ($M$)**: 181 canonical paragraphs.
- **Temporal Cutoff ($T_{\text{cutoff}}$)**: `2019-12-31T23:59:59Z`.
- **Contamination Stream ($D_{\text{post}}$)**: Official Federal Reserve communication releases from 2020-01-01 to 2022-12-31 ($T_{\text{min}} = \text{2020-01-29T19:00:00Z}$).
- **Temporal Buffer**: 48 calendar days of absolute separation ($\max(T_{\text{anchors}}) < \min(T_{\text{contamination}})$).
- **Primary Statistical Unit**: The meeting event ($N=40$). All paragraph representations are aggregated to event centroids prior to cross-validation.
