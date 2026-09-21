# Phase 4B Confirmatory Study: Integrated Manuscript Synthesis

**Stage**: Phase 5 — Manuscript Results, Figures & Discussion Synthesis  
**Repository**: `DJKuuki/MANTRA`  
**Scientific Code Freeze SHA**: `5ec0f03f3a5393d90462aa78d018d54b08cce126`  
**Execution Repository HEAD**: `5d82a7a321f6ce2441e449b5fff7ec482681819e`  
**Locked Source Tree SHA**: `02fe0ced0d03a920a7f56887f1d674282d17bc108986d19d2df495b59d330bb6`  
**Protocol Version**: 1.2.4  
**Study Status**: EMPIRICAL EXECUTION COMPLETE & CLOSED (25/25 Branches Immutable)  

---

## 1. Executive Scientific Abstract

We present the empirical findings of Phase 4B, a 25-branch preregistered confirmatory study evaluating parametric temporal leakage in financial language models. Using a causally symmetric twin-model architecture, we exposed the `ProsusAI/finbert` encoder to varying doses of post-cutoff central-bank communications ($D \in \{0.25, 0.50, 0.75, 1.00\}$) across five independent optimization seeds, benchmarked against clean models trained exclusively on contemporary pre-cutoff text under matched token compute (256,000 tokens per branch; 6,400,000 total tokens on CUDA hardware).

Controlled post-cutoff continued pretraining produced a substantial and predominantly positive representational temporal-leakage signal ($L_{\mathrm{repr}}$), with 18 of 20 contaminated branches exhibiting positive error reductions and 10 of 20 reaching nominal branch-level significance ($p < 0.05$). The signal was heterogeneous across optimization seeds and non-monotonic across contamination doses, peaking at $D = 0.75$ and decreasing at $D = 1.00$. Because Protocol v1.2.4 did not preregister an omnibus global decision rule combining the 20 branch-level tests, a formal global confirmatory rejection of $H_0^{\mathrm{repr}}$ is not claimed. Furthermore, the discrete policy endpoint was not supported, behavioral masking sensitivity shifts were minimal ($\sim 10^{-4}$), and no reliable downstream economic effect was detected across Treasury or equity markets. These results demonstrate that parametric temporal leakage can be detectable first at the representation level without necessarily propagating into behavioral or economic outputs.

---

## 2. Core Contributions

1. **Causally Symmetric Twin Design**: Established a methodology that strictly decouples temporal exposure from compute volume, optimization steps, and vocabulary domain adaptation by pairing contaminated models with compute-matched contemporary clean twins.
2. **The Layered Leakage Framework**: Demonstrated that temporal leakage is not a monolithic binary attribute, but operates across four distinct layers (Exposure $\to$ Representation $\to$ Behavior $\to$ Economic Outcome) that systematically decouple in practice.
3. **Identification of Latent Leakage**: Discovered that parametric temporal leakage manifests prominently in latent representation geometry while remaining dormant under standard classification heads and surface behavioral checks.
4. **Empirical Characterization of Non-Monotonicity and Seed Sensitivity**: Refuted naive monotonic dose-response assumptions and established that optimization randomness is a first-order determinant of leakage susceptibility.
5. **The Null-Model Insight**: Formalized that low temporal leakage does not imply model quality, demonstrating that temporal integrity ($L$) and predictive competence ($C$) are orthogonal evaluative dimensions.

---

## 3. Empirical Synthesis Matrix

| Analytical Layer | Formal Endpoint | Empirical Evidence | Confirmatory Verdict | Publication Figure / Table |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1: Treatment Exposure** | Exact Token Budget | $256,000$ tokens/branch, 0 overlap, 48-day buffer | **100% Causal Symmetry Verified** | [Table 3](file:///e:/MANTRA/docs/research/phase4b_manuscript_tables.md#3-table-3-experimental-invariants-causal-symmetry--hardware-audit-ledger-supplementary-material) |
| **Layer 2: Representation Decodability** | Continuous $L_{\mathrm{repr}}$ | 18/20 branches $> 0$; 10/20 nominal $p < 0.05$; Peak at $D=0.75$ | **Substantial Branch-Level Evidence** (No Global Rejection Claimed) | [Figure 1](file:///e:/MANTRA/docs/research/figures/phase4b/figure1_l_repr_dose_response.png), [Figure 2](file:///e:/MANTRA/docs/research/figures/phase4b/figure2_branch_significance_map.png), [Figure 3](file:///e:/MANTRA/docs/research/figures/phase4b/figure3_event_level_deltas.png) |
| **Layer 3A: Policy Classification** | Binary $\Delta\mathrm{Macro\text{-}F1}$ | Shifts near zero; all branches $p > 0.05$ | **Not Supported** | [Figure 4](file:///e:/MANTRA/docs/research/figures/phase4b/figure4_layered_outcome_comparison.png), [Table 1](file:///e:/MANTRA/docs/research/phase4b_manuscript_tables.md#1-table-1-dose-level-aggregate-empirical-results-main-manuscript) |
| **Layer 3B: Behavioral Sensitivity** | $L_{\mathrm{behavior}}$ ($S_{\mathrm{mask}}$) | Negligible shift ($\sim 10^{-4}$) | **Minimal Descriptive Effect** | [Figure 4](file:///e:/MANTRA/docs/research/figures/phase4b/figure4_layered_outcome_comparison.png), [Table 1](file:///e:/MANTRA/docs/research/phase4b_manuscript_tables.md#1-table-1-dose-level-aggregate-empirical-results-main-manuscript) |
| **Layer 4A: Economic Primary** | $\Delta\mathrm{IC}_{2\mathrm{Y}}$ (Treasury) | 95% Bootstrap CIs cross zero; mean $\Delta\mathrm{IC} < 0$ | **Not Supported** | [Figure 4](file:///e:/MANTRA/docs/research/figures/phase4b/figure4_layered_outcome_comparison.png), [Table 2](file:///e:/MANTRA/docs/research/phase4b_manuscript_tables.md#2-table-2-complete-20-branch-contaminated-empirical-results-supplementary-material) |
| **Layer 4B: Economic Exploratory** | $\Delta\mathrm{IC}_{\mathrm{SPY}}$ (Equities) | 95% Bootstrap CIs cross zero | **Not Supported** | [Figure 4](file:///e:/MANTRA/docs/research/figures/phase4b/figure4_layered_outcome_comparison.png), [Table 2](file:///e:/MANTRA/docs/research/phase4b_manuscript_tables.md#2-table-2-complete-20-branch-contaminated-empirical-results-supplementary-material) |
| **Competence Control** | Competence $C$ | No independent post-2018 eval split | `NOT_EVALUATED_NO_EVAL_SPLIT` | [Section 6.7](file:///e:/MANTRA/docs/research/phase4b_limitations.md#67-absence-of-independent-competence-split-c--not_evaluated) |
| **Temporal Robustness** | Robustness $R_T$ | Excluded from confirmatory scope | `NOT_EVALUATED` | [Section 6.8](file:///e:/MANTRA/docs/research/phase4b_limitations.md#68-absence-of-temporal-robustness-control-r_t--not_evaluated) |

---

## 4. Guide to Manuscript Artifacts

### Core Manuscript Chapters
- [Results Section (`phase4b_results_section.md`)](file:///e:/MANTRA/docs/research/phase4b_results_section.md): Detailed narrative of Sections 4.1 through 4.7, presenting execution verification, representational leakage, dose/seed dynamics, binary endpoints, behavioral sensitivity, economic outcomes, and decoupling summaries.
- [Discussion Section (`phase4b_discussion.md`)](file:///e:/MANTRA/docs/research/phase4b_discussion.md): Deep conceptual synthesis examining what the study demonstrates and does not demonstrate, latent leakage mechanisms, non-monotonicity hypotheses, seed sensitivity, the layered framework, and domain adaptation controls.
- [Limitations Section (`phase4b_limitations.md`)](file:///e:/MANTRA/docs/research/phase4b_limitations.md): Transparent accounting of omnibus under-specification, seed variance, non-monotonicity, model/domain bounds, economic statistical power, and evaluation scope controls.
- [Tables Document (`phase4b_manuscript_tables.md`)](file:///e:/MANTRA/docs/research/phase4b_manuscript_tables.md): Publication-formatted Markdown tables for the main text (Table 1) and supplement (Tables 2 & 3).

### Publication Figures (`docs/research/figures/phase4b/`)
- [Figure 1 (Dose-Response Curve)](file:///e:/MANTRA/docs/research/figures/phase4b/figure1_l_repr_dose_response.png) ([PDF](file:///e:/MANTRA/docs/research/figures/phase4b/figure1_l_repr_dose_response.pdf)): Individual seed paths, discrete points, across-seed mean overlay with $\pm 1 \text{SEM}$ error band, highlighting peak at $D=0.75$ and dip at $D=1.00$.
- [Figure 2 (Significance Heatmap)](file:///e:/MANTRA/docs/research/figures/phase4b/figure2_branch_significance_map.png) ([PDF](file:///e:/MANTRA/docs/research/figures/phase4b/figure2_branch_significance_map.pdf)): $5 \times 4$ seed-by-dose heatmap of $L_{\mathrm{repr}}$ annotated with nominal branch-level significance asterisks ($p < 0.05$).
- [Figure 3 (Event-Level Distributions)](file:///e:/MANTRA/docs/research/figures/phase4b/figure3_event_level_deltas.png) ([PDF](file:///e:/MANTRA/docs/research/figures/phase4b/figure3_event_level_deltas.pdf)): Distribution of paired error reductions across contamination doses (Panel A) and meeting-by-meeting waterfall showing that 23 of 32 out-of-sample FOMC meetings show positive median gains (Panel B).
- [Figure 4 (Layered Outcome Comparison)](file:///e:/MANTRA/docs/research/figures/phase4b/figure4_layered_outcome_comparison.png) ([PDF](file:///e:/MANTRA/docs/research/figures/phase4b/figure4_layered_outcome_comparison.pdf)): Aligned multi-panel comparison across the five evaluated endpoints with status banners visually demonstrating concentration of signal in latent representation space.

### Deterministic Analysis Tooling
- [Analysis & Visualization Script (`scripts/phase4b_manuscript_analysis.py`)](file:///e:/MANTRA/scripts/phase4b_manuscript_analysis.py): Standalone, reproducible Python script operating strictly on frozen archived results without model dependencies or runtime compute.

---

## 5. Methodological and Publishing Invariants Maintained

Throughout Phase 5 synthesis, strict adherence to scientific and archival invariants was preserved:
1. **Zero New Compute**: No neural network training, inference, probing recomputation, or MLM steps were executed.
2. **Archive Immutability**: All empirical JSON files in `experiments/phase4_confirmatory/` remain bit-identical to their frozen state.
3. **No Overclaim on $H_1^{\mathrm{repr}}$**: The absence of a preregistered omnibus decision rule is explicitly documented across all sections, and no global confirmatory rejection is claimed.
4. **No Synthetic Composite Scores**: The distinct dimensions ($L_{\mathrm{repr}}, L_{\mathrm{behavior}}, E_L, C, R_T$) remain strictly separated on their native metrics.
5. **Transparent Post-Hoc Labeling**: All derived across-seed aggregates and finite-sample reporting sensitivities are explicitly marked `POST-HOC DESCRIPTIVE`.
