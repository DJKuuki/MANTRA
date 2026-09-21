# Phase 4B Confirmatory Execution Audit & Scientific Interpretation Report

**Repository**: `DJKuuki/MANTRA`  
**Protocol Version**: `1.2.4`  
**Execution Scope**: 25-Branch Confirmatory Experiment (5 Random Seeds $\times$ 5 Contamination Doses)  
**Execution Mode**: 100% Empirical Model Training & Probing on Local CUDA GPU (`NVIDIA GeForce GTX 1660 SUPER`)  
**Scientific Code Freeze Commit**: `5ec0f03f3a5393d90462aa78d018d54b08cce126`  
**Protocol Lock Commit**: `36cadf3d5aa47f1a461f22b1048197c9c0ec9270`  
**Execution Authorization Commit**: `fe22646271c14d396ca6d2555e5f21887890149a`  
**Historical Results Commit**: `2534b1aca7e431cbacd3e02c20dc120d4ca01212`  
**Canonical Protocol Lock SHA-256**: `652b18e1a0454f976bee0e96be40875b033e854c185f2b0535a7d7a02a1cb369`  
**Locked Source Tree SHA-256**: `02fe0ced0d03a920a7f56887f1d674282d17bc108986d19d2df495b59d330bb6`  

---

## 1. Executive Summary & Audit Overview

Following explicit human authorization under Protocol v1.2.4, the Phase 4B confirmatory experiment was executed from scratch across all 25 preregistered experimental branches (5 random seeds $\times$ 5 contamination dose levels). No historical or aborted v1.2.3 artifacts were reused. All computations were executed on local CUDA hardware using `ProductionConfirmatoryBackend` with the canonical `ProsusAI/finbert` base model (revision `4556d13015211d73dccd3fdd39d39232506f3e43`).

### Audit Summary

```text
================================================================================
                      PHASE 4B CONFIRMATORY AUDIT SUMMARY
================================================================================
PROTOCOL VERSION:             1.2.4
SCIENTIFIC CODE FREEZE:       5ec0f03f3a5393d90462aa78d018d54b08cce126
LOCKED SOURCE TREE SHA:       02fe0ced0d03a920a7f56887f1d674282d17bc108986d19d2df495b59d330bb6
PROTOCOL LOCK SHA-256:        652b18e1a0454f976bee0e96be40875b033e854c185f2b0535a7d7a02a1cb369
HUMAN AUTHORIZATION:          VERIFIED & CRYPTOGRAPHICALLY BOUND (fe22646)
DATA MODE:                    100% EMPIRICAL (0 MOCK / SYNTHETIC RUNS)
HISTORICAL v1.2.3 REUSED:     NO (ALL 25 BRANCHES RUN FROM SCRATCH)
TOTAL BRANCHES EXECUTED:      25 / 25
TOTAL TOKEN COMPUTE:          6,400,000 TOKENS (256,000 PER BRANCH)
TOTAL MLM TRAINING STEPS:     2,500 STEPS (100 PER BRANCH)
TOTAL DOWNSTREAM STEPS:       8,100 STEPS (324 PER BRANCH, 3 EPOCHS)
SAME-SEED CAUSAL SYMMETRY:    100% VERIFIED (5/5 SEEDS PASS BIT-IDENTICAL HASHES)
OUT-OF-SAMPLE EVENTS:         32 OOS INDEPENDENT FOMC EVENTS PER BRANCH
STATISTICAL INFERENCE UNIT:   INDEPENDENT FOMC MEETING EVENT (N=40, N_OOS=32)

SCIENTIFIC VERDICT:
  PRIMARY HYPOTHESIS H1_repr (Continuous Representational Leakage):
    -> SUBSTANTIAL EVIDENCE AT BRANCH LEVEL (18/20 positive, 10/20 nominal p < 0.05)
    -> GLOBAL CONFIRMATORY DECISION RULE: UNDER-SPECIFIED IN PREREGISTRATION
  CO-PRIMARY HYPOTHESIS H1_binary (Binary Policy Shift vs Hold):
    -> NOT SUPPORTED (p > 0.05 across all branches)
  SECONDARY BEHAVIORAL ENDPOINT H1_behavior (Keyword Sensitivity):
    -> DESCRIPTIVE ONLY (Minimal shift ~ 10^-4)
  SECONDARY ECONOMIC ENDPOINT H1_econ (2Y Yield / SPY Information Coefficient):
    -> NOT SUPPORTED (95% Bootstrap CI crosses zero)

OVERALL STUDY STATUS: PHASE 4B CONFIRMATORY STUDY COMPLETE & CLOSED
================================================================================
```

---

## 2. Integrity and Cryptographic Bindings

Pre-execution and post-execution cryptographic audits verified:

| Check | Expected | Actual | Audit Status |
| :--- | :--- | :--- | :--- |
| **Protocol Version** | `1.2.4` | `1.2.4` | **PASS** |
| **Protocol Lock SHA-256** | `652b18e1...` | `652b18e1...` | **PASS** |
| **Source Tree SHA-256** | `02fe0ced...` | `02fe0ced...` | **PASS** |
| **Freeze Commit** | `5ec0f03f...` | `5ec0f03f...` | **PASS** |
| **Authorization Manifest** | `configs/phase4_execution_authorization.json` | Present, Bound, Signed | **PASS** |
| **Clean Sham Corpus SHA** | `5202f14b...` | `5202f14b...` | **PASS** |
| **Contamination Corpus SHA**| `54b3e84e...` | `54b3e84e...` | **PASS** |
| **Pre-Cutoff Temporal Cutoff** | `2019-12-31T23:59:59Z` | 0 Documents $> Cutoff$ | **PASS** |
| **Earliest Contamination** | `2020-01-29T19:00:00Z` | Gap = 48 Days | **PASS** |
| **Temporal & Document Isolation** | Strict separation | 0 Overlap / Strict Gap | **PASS** |

---

## 3. Causal Symmetry Verification

In strict accordance with Section 6 of the Preregistration Specification, all 5 dose branches within each random seed shared invariant initial conditions, initial head weights, and sampling schedules:

| Seed | `initial_model_hash` | `mask_schedule_hash` | `downstream_initial_head_hash` | `downstream_sample_order_hash` | Token Budget | Invariant Symmetries |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **13** | `bccca81738663bcb...` | `5e0b3d8e34318161...` | `fa32c7e00705d3d4...` | `cf63466c43ec307f...` | 256,000 | **BIT-IDENTICAL (PASS)** |
| **42** | `f3de7b3194cee290...` | `bb2bfb7372cbdd56...` | `332eb65ebb66c5e0...` | `62eee2778e481364...` | 256,000 | **BIT-IDENTICAL (PASS)** |
| **87** | `19019a74c40d6d95...` | `44bf354200d91760...` | `f6efa05a18893060...` | `f987525b12125b33...` | 256,000 | **BIT-IDENTICAL (PASS)** |
| **123** | `4586b0a658223d2f...` | `4dbeada686d0f4a5...` | `04520059fe7e29ae...` | `fd7769b6046542f1...` | 256,000 | **BIT-IDENTICAL (PASS)** |
| **2024** | `8e45d391d5140ad7...` | `9716b683dfc27818...` | `ce57aa2fe7e1d285...` | `458ec49a930aec2a...` | 256,000 | **BIT-IDENTICAL (PASS)** |

---

## 4. Complete 25-Branch Empirical Results

All 25 branches were executed in `EMPIRICAL` mode. All evaluation used $N_{\text{OOS}} = 32$ independent FOMC events under 4-fold grouped expanding-window temporal CV.

### Primary Continuous & Binary Endpoints

| Seed | Dose | Continuous $L_{\text{repr}}$ | Frozen Monte-Carlo $p$-value | Post-Execution Sensitivity $p$ | Nominal Signif. ($\alpha=0.05$) | $\Delta\text{Spearman}$ | Binary $\Delta\text{Macro-F1}$ | Binary $p$-value |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **13** | 0.00 | +0.00000 | 1.0000 | 1.0000 | False | +0.0000 | +0.0000 | 1.0000 |
| **13** | 0.25 | +0.00245 | 0.0800 | 0.0805 | False | +0.0906 | +0.0000 | 1.0000 |
| **13** | 0.50 | +0.00545 | 0.0255 | 0.0260 | **True** | +0.2685 | +0.0116 | 0.4990 |
| **13** | 0.75 | +0.00313 | 0.1425 | 0.1429 | False | +0.0011 | +0.0000 | 1.0000 |
| **13** | 1.00 | +0.00090 | 0.3685 | 0.3688 | False | +0.0182 | +0.0000 | 1.0000 |
| **42** | 0.00 | +0.00000 | 1.0000 | 1.0000 | False | +0.0000 | +0.0000 | 1.0000 |
| **42** | 0.25 | +0.00402 | 0.0090 | 0.0095 | **True** | +0.0698 | -0.0523 | 0.7435 |
| **42** | 0.50 | +0.00173 | 0.0870 | 0.0875 | False | +0.0000 | +0.0174 | 0.5040 |
| **42** | 0.75 | +0.00195 | 0.0980 | 0.0985 | False | -0.0317 | +0.0174 | 0.5040 |
| **42** | 1.00 | +0.00061 | 0.3710 | 0.3713 | False | +0.0265 | +0.0174 | 0.5040 |
| **87** | 0.00 | +0.00000 | 1.0000 | 1.0000 | False | +0.0000 | +0.0000 | 1.0000 |
| **87** | 0.25 | +0.00304 | 0.0600 | 0.0605 | False | -0.0328 | +0.0174 | 0.5045 |
| **87** | 0.50 | +0.00528 | 0.0060 | 0.0065 | **True** | +0.1223 | +0.0174 | 0.5045 |
| **87** | 0.75 | +0.00964 | 0.0005 | 0.0010 | **True** | +0.2602 | +0.0351 | 0.2510 |
| **87** | 1.00 | +0.00632 | 0.0000 | $< 1/2000$ ($\approx 0.0005$) | **True** | +0.1339 | +0.0174 | 0.4990 |
| **123** | 0.00 | +0.00000 | 1.0000 | 1.0000 | False | +0.0000 | +0.0000 | 1.0000 |
| **123** | 0.25 | +0.00640 | 0.0075 | 0.0080 | **True** | +0.0462 | -0.0494 | 0.4955 |
| **123** | 0.50 | -0.00024 | 0.5400 | 0.5402 | False | -0.0118 | -0.0615 | 0.7510 |
| **123** | 0.75 | +0.00385 | 0.0395 | 0.0400 | **True** | +0.0777 | +0.0174 | 0.5120 |
| **123** | 1.00 | -0.00218 | 0.8955 | 0.8956 | False | -0.0094 | -0.0173 | 1.0000 |
| **2024** | 0.00 | +0.00000 | 1.0000 | 1.0000 | False | +0.0000 | +0.0000 | 1.0000 |
| **2024** | 0.25 | +0.00247 | 0.1290 | 0.1294 | False | +0.0593 | +0.0000 | 1.0000 |
| **2024** | 0.50 | +0.00643 | 0.0205 | 0.0210 | **True** | +0.1733 | +0.0000 | 1.0000 |
| **2024** | 0.75 | +0.00680 | 0.0135 | 0.0140 | **True** | +0.2144 | +0.0668 | 0.7740 |
| **2024** | 1.00 | +0.00889 | 0.0005 | 0.0010 | **True** | +0.1551 | +0.0116 | 0.4990 |

> **POST-EXECUTION REPORTING SENSITIVITY NOTE**:  
> The original frozen evaluator computes Monte-Carlo sign-flip permutation $p$-values as $p = \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\text{stat}_b \ge \text{stat}_{\text{obs}})$ with $B = 2,000$. For Seed 87 at Dose 1.00, zero permutation draws exceeded the observed statistic ($k = 0$), yielding an empirical output of $p = 0.0000$. Under standard finite-sample reporting correction $p_{\text{corrected}} = \frac{k+1}{B+1}$, this corresponds to $p < 1/2000$ ($\approx 0.00050$). In all cases, $p_{\text{corrected}} < 0.05$, so the nominal significance status is unchanged.

---

### Secondary Behavioral & Economic Endpoints

| Seed | Dose | $L_{\text{behavior}}$ ($S_{\text{mask}}$) | $\Delta IC_{2Y}$ | 95% Bootstrap CI (2Y) | $p$-val (2Y) | $\Delta IC_{SPY}$ | 95% Bootstrap CI (SPY) | $p$-val (SPY) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **13** | 0.00 | +0.00000 | +0.0000 | N/A | N/A | +0.0000 | N/A | N/A |
| **13** | 0.25 | +0.00000 | +0.0081 | [-0.0461, +0.0659] | 0.284 | +0.0124 | [-0.0145, +0.0515] | 0.215 |
| **13** | 0.50 | +0.00015 | +0.0173 | [-0.0580, +0.0717] | 0.241 | +0.0170 | [-0.0167, +0.0710] | 0.114 |
| **13** | 0.75 | -0.00027 | +0.0177 | [-0.0533, +0.0955] | 0.253 | +0.0060 | [-0.0275, +0.0554] | 0.281 |
| **13** | 1.00 | -0.00039 | +0.0028 | [-0.0603, +0.0626] | 0.409 | -0.0148 | [-0.0606, +0.0034] | 0.066 |
| **42** | 0.00 | +0.00000 | +0.0000 | N/A | N/A | +0.0000 | N/A | N/A |
| **42** | 0.25 | +0.00105 | -0.0966 | [-0.1775, -0.0147] | 0.009* | -0.0655 | [-0.1561, +0.0367] | 0.106 |
| **42** | 0.50 | +0.00207 | -0.0938 | [-0.1798, -0.0246] | 0.000* | -0.0455 | [-0.1176, +0.0375] | 0.111 |
| **42** | 0.75 | -0.00042 | -0.0502 | [-0.1221, +0.0087] | 0.057 | -0.0201 | [-0.0824, +0.0532] | 0.216 |
| **42** | 1.00 | -0.00119 | -0.0788 | [-0.1804, +0.0144] | 0.047 | -0.0475 | [-0.1345, +0.0028] | 0.033 |
| **87** | 0.00 | +0.00000 | +0.0000 | N/A | N/A | +0.0000 | N/A | N/A |
| **87** | 0.25 | +0.00025 | +0.0244 | [-0.0438, +0.0931] | 0.221 | -0.0094 | [-0.1099, +0.0625] | 0.333 |
| **87** | 0.50 | -0.00014 | +0.0135 | [-0.0414, +0.0729] | 0.323 | -0.0039 | [-0.0902, +0.0795] | 0.380 |
| **87** | 0.75 | -0.00038 | +0.0013 | [-0.0601, +0.0669] | 0.510 | -0.0146 | [-0.0643, +0.0250] | 0.195 |
| **87** | 1.00 | +0.00044 | +0.0121 | [-0.0840, +0.1581] | 0.440 | -0.0518 | [-0.2169, +0.0330] | 0.137 |
| **123** | 0.00 | +0.00000 | +0.0000 | N/A | N/A | +0.0000 | N/A | N/A |
| **123** | 0.25 | -0.00061 | -0.0392 | [-0.1896, +0.0686] | 0.267 | +0.0701 | [-0.0230, +0.1680] | 0.065 |
| **123** | 0.50 | +0.00032 | -0.0255 | [-0.1029, +0.0521] | 0.293 | +0.0434 | [-0.0049, +0.1075] | 0.036 |
| **123** | 0.75 | +0.00032 | -0.0367 | [-0.1469, +0.0611] | 0.308 | +0.0607 | [-0.0092, +0.1716] | 0.039 |
| **123** | 1.00 | +0.00188 | -0.0312 | [-0.1632, +0.0674] | 0.290 | +0.0540 | [-0.0273, +0.1256] | 0.088 |
| **2024** | 0.00 | +0.00000 | +0.0000 | N/A | N/A | +0.0000 | N/A | N/A |
| **2024** | 0.25 | -0.00093 | -0.0085 | [-0.0818, +0.0551] | 0.346 | +0.0370 | [-0.0206, +0.1171] | 0.121 |
| **2024** | 0.50 | -0.00012 | -0.0025 | [-0.1220, +0.1448] | 0.514 | +0.0390 | [-0.0449, +0.1492] | 0.206 |
| **2024** | 0.75 | -0.00077 | -0.0307 | [-0.1271, +0.0662] | 0.273 | +0.0219 | [-0.0274, +0.0839] | 0.218 |
| **2024** | 1.00 | -0.00073 | +0.0019 | [-0.0756, +0.0760] | 0.505 | -0.0145 | [-0.1306, +0.0710] | 0.393 |

*\*Note: In Seed 42, the negative $\Delta IC_{2Y}$ values are contrary to the directional hypothesis $\Delta IC > 0$; thus `is_statistically_significant` is False.*

---

## 5. Aggregate Dose-Response Analysis

Aggregating metrics across the 5 independent random seeds:

| Dose Level ($D$) | Mean $L_{\text{repr}}$ | Median $L_{\text{repr}}$ | Std $L_{\text{repr}}$ | Significant Fraction ($p < 0.05$) | Mean $\Delta\text{Spearman}$ | Mean Binary $\Delta\text{Macro-F1}$ | Mean $L_{\text{behavior}}$ ($S_{\text{mask}}$) | Mean $\Delta IC_{2Y}$ | Mean $\Delta IC_{SPY}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** | +0.00000 | +0.00000 | 0.00000 | 0 / 5 (0.0%) | +0.0000 | +0.0000 | +0.00000 | +0.0000 | +0.0000 |
| **0.25** | +0.00368 | +0.00304 | 0.00147 | 2 / 5 (40.0%) | +0.0466 | -0.0169 | -0.00005 | -0.0224 | +0.0089 |
| **0.50** | +0.00373 | +0.00528 | 0.00255 | 3 / 5 (60.0%) | +0.1105 | -0.0030 | +0.00046 | -0.0182 | +0.0100 |
| **0.75** | **+0.00507** | +0.00385 | 0.00279 | **3 / 5 (60.0%)** | **+0.1043** | +0.0273 | -0.00030 | -0.0197 | +0.0108 |
| **1.00** | +0.00291 | +0.00090 | 0.00407 | 2 / 5 (40.0%) | +0.0649 | +0.0058 | +0.00000 | -0.0186 | -0.0149 |

### Detailed Scientific Analysis

1. **Non-Monotonic Dose Response**:
   The aggregate dose-response curve is **not strictly monotonic**. Mean $L_{\text{repr}}$ increases from Dose 0.00 to Dose 0.75 (+0.00507), but declines at Dose 1.00 (+0.00291). A claim that "higher contamination dose monotonically produces greater leakage" is rejected by the empirical data.
2. **Seed-Dependent Heterogeneity**:
   Leakage susceptibility varies substantially across initialization seeds:
   * **Seeds 87 and 2024**: Exhibited strong and consistent representational leakage, with individual branch permutation tests reaching $p \le 0.0005$ and large rank correlation shifts ($\Delta\text{Spearman} > 0.20$).
   * **Seed 123**: Displayed marked non-monotonicity, with $L_{\text{repr}}$ dipping below clean baseline at Dose 0.50 ($-0.00024$) and Dose 1.00 ($-0.00218$).
   * **Seeds 13 and 42**: Demonstrated moderate, non-monotonic leakage reaching nominal significance at single dose levels (Dose 0.50 for Seed 13; Dose 0.25 for Seed 42).
3. **Decoupling of Latent and Behavioral Metrics**:
   Keyword masking sensitivity shift ($L_{\text{behavior}}$) is descriptive only and of negligible magnitude ($\sim 10^{-4}$). Latent representational shifts occurred without surface-level disruption to masking sensitivity.
4. **Decoupling of Latent and Economic Metrics**:
   Neither 2Y Treasury Yield nor SPY returns demonstrated statistically reliable Information Coefficient improvements. All 95% bootstrap confidence intervals cross zero or reflect negative shifts. Latent temporal representation shift does not translate into economic market predictability under simple linear probing.

---

## 6. Preregistration Decision Rule Audit & Scientific Findings

### Audit of Global Decision Rule

A rigorous audit of the preregistered Protocol v1.2.4 contract (`configs/phase4_preregistration.yaml` and `docs/research/phase4_preregistration.md`) reveals:
* **Branch-Level Test**: Formally defined for individual branches as a paired event-level sign-flip permutation test ($\alpha = 0.05$, $B = 2,000$, $N_{\text{OOS}} = 32$).
* **Global Multi-Branch Aggregation**: **UNDER-SPECIFIED IN PREREGISTRATION**. The protocol did not define an omnibus test statistic, multiple-testing correction procedure, seed-aggregation rule (e.g., Fisher/Stouffer combination), or an explicit fraction threshold (such as "10 of 20 branches significant") to govern global acceptance or rejection of $H_0^{\text{repr}}$.

### Required Scientific Interpretation

> **Phase 4B provides substantial evidence that controlled post-cutoff continued pretraining can increase the decodability of future monetary-policy information in FinBERT representations.**
>
> **The effect is directionally positive in most contaminated branches (18/20), reaches nominal event-level significance in multiple seed/dose combinations (10/20), and is strongest on average near Dose 0.75.**
>
> **However, the preregistration did not specify a single global rule for combining the 20 contaminated branch-level tests across seeds and doses. Therefore a formal global confirmatory rejection of $H_0^{\text{repr}}$ is not claimed.**
>
> **The binary co-primary, behavioral endpoint, and economic endpoint do not show corresponding robust downstream effects.**

### Status of Secondary and Exploratory Metrics

* **$H_1^{\text{binary}}$ (Co-Primary Binary Policy Shift)**: **NOT SUPPORTED** ($p > 0.05$ across all branches).
* **$H_1^{\text{behavior}}$ (Descriptive Secondary Sensitivity)**: **DESCRIPTIVE RESULT** (Minimal shift observed).
* **$H_1^{\text{econ}}$ (Primary Economic 2Y Yield)**: **NOT SUPPORTED** (95% Bootstrap CI crosses zero).
* **SPY Exploratory Economic Endpoint**: **NO RELIABLE EFFECT DETECTED** (95% Bootstrap CI crosses zero).
* **Competence $C$**: `NOT_EVALUATED_NO_EVAL_SPLIT` (No independent post-2018 evaluation split).
* **Temporal Robustness $R_T$**: `NOT_EVALUATED` (Excluded from confirmatory scope).

---

## 7. Result Archive and Manifest

The complete immutable execution archive is cataloged in `experiments/phase4_confirmatory/result_manifest.json`:
* **Full Aggregate Results**: `experiments/phase4_confirmatory/results/phase4_confirmatory_results.json` (7,757,161 bytes)
* **Branch Manifests**: `experiments/phase4_confirmatory/manifests/seed{seed}_d{dose}.json` (25 files)
* **Branch Metrics**: `experiments/phase4_confirmatory/metrics/seed{seed}_d{dose}.json` (25 files)
* **Provenance Manifests**: `experiments/phase4_confirmatory/provenance/*.json` (2 files)
