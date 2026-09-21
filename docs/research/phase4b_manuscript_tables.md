# Phase 4B Manuscript Empirical Tables & Statistical Ledger

**Stage**: Phase 5 — Manuscript Results, Figures & Discussion Synthesis  
**Repository**: `DJKuuki/MANTRA`  
**Scientific Freeze SHA**: `5ec0f03f3a5393d90462aa78d018d54b08cce126`  
**Source Tree SHA**: `02fe0ced0d03a920a7f56887f1d674282d17bc108986d19d2df495b59d330bb6`  
**Protocol Lock SHA-256**: `652b18e1a0454f976bee0e96be40875b033e854c185f2b0535a7d7a02a1cb369`  
**Execution Repository HEAD**: `5d82a7a321f6ce2441e449b5fff7ec482681819e`  
**Protocol Version**: 1.2.4  
**Data Mode**: 100% EMPIRICAL (0 Mock / Synthetic Runs)  

---

## 1. Table 1: Dose-Level Aggregate Empirical Results (Main Manuscript)

The following table reports the aggregate empirical outcomes across the five experimental contamination dose levels ($D \in \{0.00, 0.25, 0.50, 0.75, 1.00\}$). All summary statistics across seeds (mean, median, standard deviation) are categorized as **POST-HOC DESCRIPTIVE**, as Protocol v1.2.4 preregistered branch-level permutation tests but did not specify an omnibus across-seed aggregation formula.

| Contamination Dose ($D$) | Mean $L_{\mathrm{repr}}$ | Median $L_{\mathrm{repr}}$ | Std $L_{\mathrm{repr}}$ | Nominal Sig. Frac. ($p < 0.05$) | Mean $\Delta\mathrm{Spearman}$ | Mean Binary $\Delta\mathrm{Macro\text{-}F1}$ | Mean $L_{\mathrm{behavior}}$ | Mean $\Delta\mathrm{IC}_{2\mathrm{Y}}$ | Mean $\Delta\mathrm{IC}_{\mathrm{SPY}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00 (Clean)** | +0.00000 | +0.00000 | 0.00000 | 0 / 5 (0.0%) | +0.0000 | +0.0000 | +0.00000 | +0.0000 | +0.0000 |
| **0.25** | +0.00368 | +0.00304 | 0.00165 | 2 / 5 (40.0%) | +0.0466 | -0.0169 | -0.00005 | -0.0224 | +0.0089 |
| **0.50** | +0.00373 | +0.00528 | 0.00285 | 3 / 5 (60.0%) | +0.1105 | -0.0030 | +0.00046 | -0.0182 | +0.0100 |
| **0.75** | **+0.00507** | +0.00385 | 0.00312 | **3 / 5 (60.0%)** | **+0.1043** | +0.0273 | -0.00030 | -0.0197 | +0.0108 |
| **1.00** | +0.00291 | +0.00090 | 0.00455 | 2 / 5 (40.0%) | +0.0649 | +0.0058 | +0.00000 | -0.0186 | -0.0149 |

### Table 1 Explanatory Notes:
1. **$L_{\mathrm{repr}}$ (Continuous Representational Leakage)**: Primary endpoint measuring the paired out-of-sample absolute error reduction $d_e = |y_e - \hat{y}_{\mathrm{clean},e}| - |y_e - \hat{y}_{\mathrm{leak},e}|$ across $N_{\mathrm{OOS}} = 32$ independent FOMC events under 4-fold grouped temporal CV. Positive values denote lower prediction error for representations exposed to post-cutoff contamination.
2. **Non-Monotonicity**: Across doses, mean $L_{\mathrm{repr}}$ increases from $+0.00368$ at $D=0.25$ to an aggregate peak of $+0.00507$ at $D=0.75$, before decreasing to $+0.00291$ at $D=1.00$.
3. **Nominal Significance Fraction**: Fraction of branches within the dose level yielding nominal branch-level $p < 0.05$ under the frozen 2,000-draw sign-flip permutation test.
4. **Binary $\Delta\mathrm{Macro\text{-}F1}$**: Co-primary discrete classification endpoint (`next_scheduled_change_vs_hold`). All individual branch tests yielded $p > 0.05$.
5. **$L_{\mathrm{behavior}}$**: Behavioral masking sensitivity shift across 40 FOMC events. Shift is of negligible magnitude ($\sim 10^{-4}$) and descriptive only.
6. **$\Delta\mathrm{IC}_{2\mathrm{Y}}$ and $\Delta\mathrm{IC}_{\mathrm{SPY}}$**: Economic Information Coefficient deltas against 2-year Treasury yields (primary economic) and SPY equity returns (exploratory economic). All 95% bootstrap confidence intervals cross zero or exhibit negative shifts.

---

## 2. Table 2: Complete 20-Branch Contaminated Empirical Results (Supplementary Material)

This table reports complete, unaggregated metrics for all 20 contaminated branches ($D > 0$). Clean baseline branches ($D = 0.00$) serve as the paired reference within each seed ($L_{\mathrm{repr}} = 0$, $\Delta\mathrm{IC} = 0$, $p = 1.000$).

| Seed | Dose ($D$) | $L_{\mathrm{repr}}$ | Frozen Perm $p$ | Post-Exec Sens $p$ | Nominal Sig. ($\alpha=0.05$) | $\Delta\mathrm{Spearman}$ | Binary $\Delta\mathrm{Macro\text{-}F1}$ | Binary $p$ | $L_{\mathrm{behavior}}$ | $\Delta\mathrm{IC}_{2\mathrm{Y}}$ | 95% Bootstrap CI (2Y) | $p_{2\mathrm{Y}}$ | $\Delta\mathrm{IC}_{\mathrm{SPY}}$ | 95% Bootstrap CI (SPY) | $p_{\mathrm{SPY}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **13** | 0.25 | +0.00245 | 0.0800 | 0.0800 | False | +0.0906 | +0.0000 | 1.0000 | +0.00000 | +0.0081 | [-0.0461, +0.0659] | 0.284 | +0.0124 | [-0.0145, +0.0515] | 0.215 |
| **13** | 0.50 | +0.00545 | 0.0255 | 0.0255 | **True** | +0.2685 | +0.0116 | 0.4990 | +0.00015 | +0.0173 | [-0.0580, +0.0717] | 0.241 | +0.0170 | [-0.0167, +0.0710] | 0.114 |
| **13** | 0.75 | +0.00313 | 0.1425 | 0.1425 | False | +0.0011 | +0.0000 | 1.0000 | -0.00027 | +0.0177 | [-0.0533, +0.0955] | 0.253 | +0.0060 | [-0.0275, +0.0554] | 0.281 |
| **13** | 1.00 | +0.00090 | 0.3685 | 0.3685 | False | +0.0182 | +0.0000 | 1.0000 | -0.00039 | +0.0028 | [-0.0603, +0.0626] | 0.409 | -0.0148 | [-0.0606, +0.0034] | 0.066 |
| **42** | 0.25 | +0.00402 | 0.0090 | 0.0090 | **True** | +0.0698 | -0.0523 | 0.7435 | +0.00105 | -0.0966 | [-0.1775, -0.0147] | 0.009* | -0.0655 | [-0.1561, +0.0367] | 0.106 |
| **42** | 0.50 | +0.00173 | 0.0870 | 0.0870 | False | +0.0000 | +0.0174 | 0.5040 | +0.00207 | -0.0938 | [-0.1798, -0.0246] | 0.000* | -0.0455 | [-0.1176, +0.0375] | 0.111 |
| **42** | 0.75 | +0.00195 | 0.0980 | 0.0980 | False | -0.0317 | +0.0174 | 0.5040 | -0.00042 | -0.0502 | [-0.1221, +0.0087] | 0.057 | -0.0201 | [-0.0824, +0.0532] | 0.216 |
| **42** | 1.00 | +0.00061 | 0.3710 | 0.3710 | False | +0.0265 | +0.0174 | 0.5040 | -0.00119 | -0.0788 | [-0.1804, +0.0144] | 0.047 | -0.0475 | [-0.1345, +0.0028] | 0.033 |
| **87** | 0.25 | +0.00304 | 0.0600 | 0.0600 | False | -0.0328 | +0.0174 | 0.5045 | +0.00025 | +0.0244 | [-0.0438, +0.0931] | 0.221 | -0.0094 | [-0.1099, +0.0625] | 0.333 |
| **87** | 0.50 | +0.00528 | 0.0060 | 0.0060 | **True** | +0.1223 | +0.0174 | 0.5045 | -0.00014 | +0.0135 | [-0.0414, +0.0729] | 0.323 | -0.0039 | [-0.0902, +0.0795] | 0.380 |
| **87** | 0.75 | +0.00964 | 0.0005 | 0.0005 | **True** | +0.2602 | +0.0351 | 0.2510 | -0.00038 | +0.0013 | [-0.0601, +0.0669] | 0.510 | -0.0146 | [-0.0643, +0.0250] | 0.195 |
| **87** | 1.00 | +0.00632 | 0.0000 | < 1/2000 (~0.0005) | **True** | +0.1339 | +0.0174 | 0.4990 | +0.00044 | +0.0121 | [-0.0840, +0.1581] | 0.440 | -0.0518 | [-0.2169, +0.0330] | 0.137 |
| **123** | 0.25 | +0.00640 | 0.0075 | 0.0075 | **True** | +0.0462 | -0.0494 | 0.4955 | -0.00061 | -0.0392 | [-0.1896, +0.0686] | 0.267 | +0.0701 | [-0.0230, +0.1680] | 0.065 |
| **123** | 0.50 | -0.00024 | 0.5400 | 0.5400 | False | -0.0118 | -0.0615 | 0.7510 | +0.00032 | -0.0255 | [-0.1029, +0.0521] | 0.293 | +0.0434 | [-0.0049, +0.1075] | 0.036 |
| **123** | 0.75 | +0.00385 | 0.0395 | 0.0395 | **True** | +0.0777 | +0.0174 | 0.5120 | +0.00032 | -0.0367 | [-0.1469, +0.0611] | 0.308 | +0.0607 | [-0.0092, +0.1716] | 0.039 |
| **123** | 1.00 | -0.00218 | 0.8955 | 0.8955 | False | -0.0094 | -0.0173 | 1.0000 | +0.00188 | -0.0312 | [-0.1632, +0.0674] | 0.290 | +0.0540 | [-0.0273, +0.1256] | 0.088 |
| **2024** | 0.25 | +0.00247 | 0.1290 | 0.1290 | False | +0.0593 | +0.0000 | 1.0000 | -0.00093 | -0.0085 | [-0.0818, +0.0551] | 0.346 | +0.0370 | [-0.0206, +0.1171] | 0.121 |
| **2024** | 0.50 | +0.00643 | 0.0205 | 0.0205 | **True** | +0.1733 | +0.0000 | 1.0000 | -0.00012 | -0.0025 | [-0.1220, +0.1448] | 0.514 | +0.0390 | [-0.0449, +0.1492] | 0.206 |
| **2024** | 0.75 | +0.00680 | 0.0135 | 0.0135 | **True** | +0.2144 | +0.0668 | 0.7740 | -0.00077 | -0.0307 | [-0.1271, +0.0662] | 0.273 | +0.0219 | [-0.0274, +0.0839] | 0.218 |
| **2024** | 1.00 | +0.00889 | 0.0005 | 0.0005 | **True** | +0.1551 | +0.0116 | 0.4990 | -0.00073 | +0.0019 | [-0.0756, +0.0760] | 0.505 | -0.0145 | [-0.1306, +0.0710] | 0.393 |

### Table 2 Explanatory Notes:
1. **Nominal Significance Definition**: A branch is marked **True** if and only if its sign-flip permutation $p$-value satisfies $p < 0.05$ on the directional hypothesis $L_{\mathrm{repr}} > 0$. Ten of the twenty branches (50.0%) reach nominal significance.
2. **Post-Execution Reporting Sensitivity for $p=0.0000$**: In Seed 87 at Dose 1.00, the frozen evaluator computed $p = \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\text{stat}_b \ge \text{stat}_{\mathrm{obs}}) = 0.0000$ because $0$ of $B=2,000$ permutation draws exceeded the observed statistic ($k=0$). Under finite-sample Monte-Carlo reporting sensitivity $\frac{k+1}{B+1}$, this corresponds to $p < 1/2000 \approx 0.00050$. The nominal significance status ($\alpha = 0.05$) is invariant to this reporting convention.
3. **Directional Economic Invalidation (\*)**: In Seed 42 at Doses 0.25 and 0.50, the bootstrap two-sided $p$-values are small ($0.009$ and $0.000$), but the observed shifts are **negative** ($\Delta\mathrm{IC}_{2\mathrm{Y}} = -0.0966$ and $-0.0938$). Under the directional hypothesis of economic leakage ($\Delta\mathrm{IC} > 0$), these shifts represent degradation of predictive performance, not positive leakage; hence `is_statistically_significant` evaluates to `False`.

---

## 3. Table 3: Experimental Invariants, Causal Symmetry & Hardware Audit Ledger (Supplementary Material)

This table documents the execution invariants enforced across all 25 branches to guarantee causal symmetry, hardware determinism, and absence of data leakage prior to treatment.

| Property / Invariant | Preregistered Specification | Actual Execution Verification | Audit Status |
| :--- | :--- | :--- | :---: |
| **Base Checkpoint** | `ProsusAI/finbert` | `ProsusAI/finbert` (`4556d13015211d73dccd3fdd39d39232506f3e43`) | **PASS** |
| **Hardware Backend** | Local CUDA GPU | NVIDIA GeForce GTX 1660 SUPER (Driver 576.88, CUDA 12.8) | **PASS** |
| **Execution Class** | `ProductionConfirmatoryBackend` | `ProductionConfirmatoryBackend` (0 Mock / Synthetic branches) | **PASS** |
| **Total Token Budget** | 256,000 tokens per branch | Exactly 256,000 tokens (500 blocks $\times$ 512 tokens) per branch | **PASS** |
| **Pretraining Steps** | 100 MLM steps per branch | Exactly 100 steps (AdamW, lr=5e-5, weight_decay=0.01) | **PASS** |
| **Downstream Fine-Tuning**| 3 Epochs on pre-2019 TDW | Exactly 324 steps (AdamW, lr=2e-5, batch_size=16) | **PASS** |
| **Pre-Cutoff Boundary** | $\le 2019\text{-}12\text{-}31\text{T}23:59:59\text{Z}$ | Max pre-cutoff timestamp: `2019-12-11T19:00:00Z` (0 leaks) | **PASS** |
| **Post-Cutoff Boundary**| $\ge 2020\text{-}01\text{-}01\text{T}00:00:00\text{Z}$ | Min post-cutoff timestamp: `2020-01-29T19:00:00Z` (48-day buffer) | **PASS** |
| **Corpus Overlap** | Zero text overlap | 0 duplicate documents between clean and contamination corpora | **PASS** |
| **Statistical Inference Unit**| Independent FOMC Event | $N_{\mathrm{total}} = 40$, $N_{\mathrm{OOS}} = 32$ independent meeting statements | **PASS** |
| **Anchor Representation Set** | 181 curated paragraphs | Exact shape `(181, 768)`, rank 2, finite values across all 25 runs | **PASS** |
| **Within-Seed Initial Weights** | Invariant across doses | Bit-identical `initial_model_hash` across all 5 doses per seed | **PASS** |
| **Within-Seed Mask Schedule** | Invariant across doses | Bit-identical `mask_schedule_hash` across all 5 doses per seed | **PASS** |
| **Within-Seed Head Weights** | Invariant across doses | Bit-identical `downstream_initial_head_hash` across all 5 doses per seed | **PASS** |
| **Within-Seed Sample Order** | Invariant across doses | Bit-identical `downstream_sample_order_hash` across all 5 doses per seed | **PASS** |

### Within-Seed Hash Verifications:
- **Seed 13**: `bccca81738663bcb...` (Model), `5e0b3d8e34318161...` (Mask), `fa32c7e00705d3d4...` (Head), `cf63466c43ec307f...` (Order)
- **Seed 42**: `f3de7b3194cee290...` (Model), `bb2bfb7372cbdd56...` (Mask), `332eb65ebb66c5e0...` (Head), `62eee2778e481364...` (Order)
- **Seed 87**: `19019a74c40d6d95...` (Model), `44bf354200d91760...` (Mask), `f6efa05a18893060...` (Head), `f987525b12125b33...` (Order)
- **Seed 123**: `4586b0a658223d2f...` (Model), `4dbeada686d0f4a5...` (Mask), `04520059fe7e29ae...` (Head), `fd7769b6046542f1...` (Order)
- **Seed 2024**: `8e45d391d5140ad7...` (Model), `9716b683dfc27818...` (Mask), `ce57aa2fe7e1d285...` (Head), `458ec49a930aec2a...` (Order)

All causal symmetries are 100% verified and bit-identical within seeds across all 5 doses.
