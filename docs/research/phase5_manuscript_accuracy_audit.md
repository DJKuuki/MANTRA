# Phase 5 Manuscript Technical Accuracy Audit

**Stage**: Phase 5 — Manuscript Results, Figures & Discussion Synthesis  
**Repository**: `DJKuuki/MANTRA`  
**Protocol Version**: `1.2.4`  
**Scientific Code Freeze SHA**: `5ec0f03f3a5393d90462aa78d018d54b08cce126`  
**Phase 4B Archive Closure SHA**: `5d82a7a321f6ce2441e449b5fff7ec482681819e`  
**Locked Source Tree SHA**: `02fe0ced0d03a920a7f56887f1d674282d17bc108986d19d2df495b59d330bb6`  
**Audit Status**: COMPLETE  

---

## 1. Executive Summary

This audit records the technical accuracy corrections applied to the Phase 5 manuscript chapters, tables, figures, and deterministic visualization code. 

All empirical data, model weights, evaluation metrics, and cryptographic hashes in `experiments/phase4_confirmatory/` remain strictly immutable and bit-identical to the Phase 4B closure state (`5d82a7a`). No models were retrained, no inference was rerun, and no statistical procedures were altered. The patch consists exclusively of editorial, terminological, and visual calibration to ensure that manuscript assertions adhere with exact fidelity to the preregistered Protocol v1.2.4 specifications and empirical realities.

---

## 2. Itemized Accuracy Corrections Ledger

| # | Item / Dimension | Original Phrasing / Representation | Corrected Phrasing / Representation | Scientific Rationale | Empirical Result Changed? |
| :-: | :--- | :--- | :--- | :--- | :-: |
| **1** | **Permutation Test Direction** | "two-sided permutation test" / "paired sign-flip permutation tests" without explicit directionality | "one-sided right-tailed paired event-level sign-flip permutation test" | Protocol v1.2.4 tests the directional hypothesis $H_1^{\mathrm{repr}}: \mu_d > 0$ (error reduction $\Delta\mathrm{MSE} > 0$). Tests count permutations where randomized mean difference exceeds or equals observed mean difference ($k/B$). | **NO** |
| **2** | **OOS Event Chronology** | "32 post-cutoff FOMC events" / "post-cutoff test meetings" | "32 out-of-sample temporal cross-validation FOMC meetings (spanning 2016–2019)" | All 32 OOS evaluation events occur prior to the global dataset cutoff `2019-12-31T23:59:59Z`. "Future target" refers to the next scheduled policy rate decision after each statement date ($\Delta\text{Rate}_{t+1}$), not events occurring post-2020. | **NO** |
| **3** | **Figure 3 Sample Independence** | Figure 3 Panel A labeled with sample size `$N=160$`, implying 160 independent events | Panel A x-axis labeled `160 branch-events`; explicit note: `32 OOS events x 5 seeds; pooled branch-events are not statistically independent; post-hoc descriptive check` | 160 points represent 32 events evaluated across 5 seeds; pooling introduces correlation within events and seeds. Must be explicitly qualified as non-independent and descriptive. | **NO** |
| **4** | **Event Breadth Check Calibration** | "demonstrating that the signal reflects generalized representation alignment rather than an artifact of a single macroeconomic event" | "a post-hoc descriptive breadth check across out-of-sample events shows that probe MSE reductions occurred in 23 of 32 (71.9%) evaluated temporal-CV events... indicating the advantage is not concentrated in an outlier meeting (though this remains a descriptive check rather than confirmatory proof of universal alignment)" | The 23/32 meeting observation is a post-hoc descriptive check on event-level distributions, not a preregistered confirmatory proof of universal generalization. | **NO** |
| **5** | **Domain Adaptation Boundaries** | "generic domain adaptation cannot account for the difference in future-rate decodability" | "The matched clean-twin design substantially reduces generic domain adaptation as an explanation by equalizing compute and broad FOMC-domain exposure. However, it cannot completely eliminate subdomain, regime, topic, or rhetorical-distribution differences between pre-2020 and post-2020 documents." | The clean-twin design matches compute budget and broad central-bank English, but cannot eliminate regime-specific vocabulary shifts (e.g., pandemic emergency easing discourse). | **NO** |
| **6** | **Scholarly Verb Calibration** | "Refuted naive monotonic dose-response assumptions", "Established optimization randomness is a first-order determinant", "Demonstrated that temporal leakage is not monolithic" | "Did not support a strictly monotonic dose-response relationship", "Revealed substantial seed-dependent optimization heterogeneity", "Observed empirical decoupling... consistent with a layered framework" | Scientific assertions must use calibrated, objective verbs rather than overclaiming definitive refutation or universal proof from a single empirical study. | **NO** |
| **7** | **Elimination of Local Machine Links** | Absolute URIs containing local machine paths (`file:///e:/MANTRA/...` or `file:///C:/...`) in markdown cross-references | Relative markdown links (`figures/phase4b/...`, `phase4b_results_section.md`) and scholarly textual citations | Absolute local paths break portability across environments and leak host environment paths in public archives. | **NO** |
| **8** | **Finite-MC Reporting Precision** | Column named `Monte-Carlo Sensitivity [k/B]` listing redundant sensitivity ranges for all rows; zero p-values displayed as `0.0000 [0/2000, max 0.0005]` | Renamed column to `Finite-MC Reporting Note`; all non-zero p-values display `—`; Seed 87 $D=1.00$ ($p=0.0000$) annotated with exact finite resolution note `k=0/2000; finite-MC resolution ≈ 1/2001 = 0.00050` | Clarifies that $p=0.0000$ reflects finite simulation resolution ($k=0$ permutations out of $B=2000$) rather than zero probability, avoiding redundant notation across ordinary branches. | **NO** |
| **9** | **Directional Economic Endpoint Interpretation** | Seed 42 negative $\Delta\mathrm{IC}$ with small bootstrap p-values ($p=0.009$ at $D=0.25$, $p=0.000$ at $D=0.50$) described as "performance even degraded significantly" without explicitly connecting to directional hypothesis | Clarified that the preregistered economic hypothesis is directional ($H_1^{\mathrm{econ}}: \Delta\mathrm{IC} > 0$); negative $\Delta\mathrm{IC}$ shifts with small bootstrap p-values represent evidence in the direction *opposite* to the hypothesized improvement, and thus do not support economic leakage | Prevents readers from misinterpreting a small two-tailed bootstrap p-value as evidence supporting economic leakage when the direction is opposite to the preregistered alternative. | **NO** |

---

## 3. Verification and File Scope

The following Phase 5 manuscript files were patched and verified against these criteria:
- `scripts/phase4b_manuscript_analysis.py`: Updated figure axis labels, footnotes, plot titles, and console table headers.
- `docs/research/figures/phase4b/`: All 4 PNG figures and 4 PDF figures regenerated deterministically from frozen JSON archives.
- `docs/research/phase4b_manuscript_tables.md`: Table 1, Table 2, and Table 3 notes, column headers, and citations corrected.
- `docs/research/phase4b_results_section.md`: Sections 4.1 through 4.7 edited to calibrate statistical, temporal, and economic phrasing.
- `docs/research/phase4b_discussion.md`: Sections 5.1 through 5.9 updated with softened domain adaptation claims and toned-down verbs.
- `docs/research/phase4b_limitations.md`: Sections 6.1 through 6.9 updated to document test directionality, sample sizes, and regime shift bounds.
- `docs/research/phase4b_manuscript_summary.md`: Synthesis matrix, abstract, and core contributions updated with calibrated terminology.

## 4. Archival and Hash Invariance Confirmation

```text
Frozen Phase 4B Archive Status:
  All 54 files in experiments/phase4_confirmatory/ verified bit-identical.
  Manifest: experiments/phase4_confirmatory/result_manifest.json (SHA-256 match: 100%)
  Compute status: ZERO re-computation, ZERO re-inference, ZERO MLM retraining.
```
