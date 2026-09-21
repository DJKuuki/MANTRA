# Phase 4B Confirmatory Execution Audit & Scientific Results Report

**Repository**: `DJKuuki/MANTRA`  
**Protocol Version**: `1.2.4`  
**Execution Type**: Real Empirical Full Confirmatory Experiment (CUDA Accelerated)  
**Execution Date**: 2026-09-21  
**Hardware Platform**: NVIDIA GeForce GTX 1660 SUPER (6,144 MiB VRAM)  
**Scientific Code Freeze Commit**: `5ec0f03f3a5393d90462aa78d018d54b08cce126`  
**Protocol Lock Commit**: `36cadf3d5aa47f1a461f22b1048197c9c0ec9270`  
**Execution Authorization Commit**: `fe22646271c14d396ca6d2555e5f21887890149a`  
**Canonical Protocol Lock SHA-256**: `652b18e1a0454f976bee0e96be40875b033e854c185f2b0535a7d7a02a1cb369`  
**Locked Source Tree SHA-256**: `02fe0ced0d03a920a7f56887f1d674282d17bc108986d19d2df495b59d330bb6`  
**Execution Status**: `PHASE4B_ORCHESTRATION_COMPLETED` (All 25 Branches Executed)  

---

## 1. Executive Summary & Audit Verdict

Following explicit human authorization under Protocol v1.2.4, the Phase 4B confirmatory experiment was executed from scratch across all 25 preregistered experimental branches (5 random seeds $\times$ 5 contamination dose levels). No historical or aborted v1.2.3 artifacts were reused. All computations were executed on local CUDA hardware using `ProductionConfirmatoryBackend` with the canonical `ProsusAI/finbert` base model (revision `4556d13015211d73dccd3fdd39d39232506f3e43`).

### Final Audit Verdict

```text
================================================================================
                      PHASE 4B CONFIRMATORY EXECUTION AUDIT
================================================================================
PROTOCOL VERSION:             1.2.4
SCIENTIFIC CODE FREEZE:       5ec0f03f3a5393d90462aa78d018d54b08cce126
LOCKED SOURCE TREE SHA:       02fe0ced0d03a920a7f56887f1d674282d17bc108986d19d2df495b59d330bb6
PROTOCOL LOCK SHA-256:        652b18e1a0454f976bee0e96be40875b033e854c185f2b0535a7d7a02a1cb369
HUMAN AUTHORIZATION:          VERIFIED & CRYPTOGRAPHICALLY BOUND (fe22646)
DATA MODE:                    100% EMPIRICAL (0 MOCK / SYNTHETIC RUNS)
TOTAL BRANCHES EXECUTED:      25 / 25
TOTAL TOKEN COMPUTE:          6,400,000 TOKENS (256,000 PER BRANCH)
TOTAL MLM TRAINING STEPS:     2,500 STEPS (100 PER BRANCH)
TOTAL DOWNSTREAM STEPS:       8,100 STEPS (324 PER BRANCH, 3 EPOCHS)
SAME-SEED CAUSAL SYMMETRY:    100% VERIFIED (5/5 SEEDS PASS BIT-IDENTICAL HASHES)
OUT-OF-SAMPLE EVENTS:         32 OOS INDEPENDENT FOMC EVENTS PER BRANCH
STATISTICAL INFERENCE UNIT:   INDEPENDENT FOMC MEETING EVENT (N=40, N_OOS=32)

SCIENTIFIC VERDICT:
  PRIMARY HYPOTHESIS H1_repr (Continuous Representational Leakage):
    -> CONFIRMED (p < 0.05 across 10/20 branches; peak at Dose 0.75, p down to 0.0000)
  CO-PRIMARY HYPOTHESIS H1_binary (Binary Policy Shift vs Hold):
    -> NOT SIGNIFICANT (p > 0.05 across all branches)
  SECONDARY BEHAVIORAL ENDPOINT H1_behavior (Keyword Sensitivity):
    -> DESCRIPTIVE ONLY (Minimal shift ~ 10^-4)
  SECONDARY ECONOMIC ENDPOINT H1_econ (2Y Yield / SPY Information Coefficient):
    -> NOT SIGNIFICANT (95% Bootstrap CI crosses zero)

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

| Seed | Dose | Continuous $L_{\text{repr}}$ | Permutation $p$-value | Signif. ($\alpha=0.05$) | $\Delta\text{Spearman}$ | Binary $\Delta\text{Macro-F1}$ | Binary $p$-value | $L_{\text{behavior}}$ ($S_{\text{mask}}$) | $\Delta IC_{2Y}$ | $\Delta IC_{SPY}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **13** | 0.00 | +0.00000 | 1.0000 | False | +0.0000 | +0.0000 | 1.0000 | +0.00000 | +0.0000 | +0.0000 |
| **13** | 0.25 | +0.00245 | 0.0800 | False | +0.0906 | +0.0000 | 1.0000 | +0.00000 | +0.0081 | +0.0124 |
| **13** | 0.50 | +0.00545 | 0.0255 | **True** | +0.2685 | +0.0116 | 0.4990 | +0.00015 | +0.0173 | +0.0170 |
| **13** | 0.75 | +0.00313 | 0.1425 | False | +0.0011 | +0.0000 | 1.0000 | -0.00027 | +0.0177 | +0.0060 |
| **13** | 1.00 | +0.00090 | 0.3685 | False | +0.0182 | +0.0000 | 1.0000 | -0.00039 | +0.0028 | -0.0148 |
| **42** | 0.00 | +0.00000 | 1.0000 | False | +0.0000 | +0.0000 | 1.0000 | +0.00000 | +0.0000 | +0.0000 |
| **42** | 0.25 | +0.00402 | 0.0090 | **True** | +0.0698 | -0.0523 | 0.7435 | +0.00105 | -0.0966 | -0.0655 |
| **42** | 0.50 | +0.00173 | 0.0870 | False | +0.0000 | +0.0174 | 0.5040 | +0.00207 | -0.0938 | -0.0455 |
| **42** | 0.75 | +0.00195 | 0.0980 | False | -0.0317 | +0.0174 | 0.5040 | -0.00042 | -0.0502 | -0.0201 |
| **42** | 1.00 | +0.00061 | 0.3710 | False | +0.0265 | +0.0174 | 0.5040 | -0.00119 | -0.0788 | -0.0475 |
| **87** | 0.00 | +0.00000 | 1.0000 | False | +0.0000 | +0.0000 | 1.0000 | +0.00000 | +0.0000 | +0.0000 |
| **87** | 0.25 | +0.00304 | 0.0600 | False | -0.0328 | +0.0174 | 0.5045 | +0.00025 | +0.0244 | -0.0094 |
| **87** | 0.50 | +0.00528 | 0.0060 | **True** | +0.1223 | +0.0174 | 0.5045 | -0.00014 | +0.0135 | -0.0039 |
| **87** | 0.75 | +0.00964 | 0.0005 | **True** | +0.2602 | +0.0351 | 0.2510 | -0.00038 | +0.0013 | -0.0146 |
| **87** | 1.00 | +0.00632 | 0.0000 | **True** | +0.1339 | +0.0174 | 0.4990 | +0.00044 | +0.0121 | -0.0518 |
| **123** | 0.00 | +0.00000 | 1.0000 | False | +0.0000 | +0.0000 | 1.0000 | +0.00000 | +0.0000 | +0.0000 |
| **123** | 0.25 | +0.00640 | 0.0075 | **True** | +0.0462 | -0.0494 | 0.4955 | -0.00061 | -0.0392 | +0.0701 |
| **123** | 0.50 | -0.00024 | 0.5400 | False | -0.0118 | -0.0615 | 0.7510 | +0.00032 | -0.0255 | +0.0434 |
| **123** | 0.75 | +0.00385 | 0.0395 | **True** | +0.0777 | +0.0174 | 0.5120 | +0.00032 | -0.0367 | +0.0607 |
| **123** | 1.00 | -0.00218 | 0.8955 | False | -0.0094 | -0.0173 | 1.0000 | +0.00188 | -0.0312 | +0.0540 |
| **2024** | 0.00 | +0.00000 | 1.0000 | False | +0.0000 | +0.0000 | 1.0000 | +0.00000 | +0.0000 | +0.0000 |
| **2024** | 0.25 | +0.00247 | 0.1290 | False | +0.0593 | +0.0000 | 1.0000 | -0.00093 | -0.0085 | +0.0370 |
| **2024** | 0.50 | +0.00643 | 0.0205 | **True** | +0.1733 | +0.0000 | 1.0000 | -0.00012 | -0.0025 | +0.0390 |
| **2024** | 0.75 | +0.00680 | 0.0135 | **True** | +0.2144 | +0.0668 | 0.7740 | -0.00077 | -0.0307 | +0.0219 |
| **2024** | 1.00 | +0.00889 | 0.0005 | **True** | +0.1551 | +0.0116 | 0.4990 | -0.00073 | +0.0019 | -0.0145 |

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

### Key Scientific Observations:
1. **Representational Leakage Emergence**: Real continued pretraining on future post-cutoff text systematically introduces future rate information into the encoder representations. In **18 out of 20 contaminated branches (90.0%)**, observed $L_{\text{repr}} > 0$.
2. **Dose Sensitivity Peak**: Representational leakage peaks around Dose 0.75 ($\overline{L_{\text{repr}}} = +0.00507$, $\overline{\Delta\text{Spearman}} = +0.1043$), with 60% of seeds reaching nominal statistical significance ($p < 0.05$).
3. **Seed Stability**: Seeds 87 and 2024 exhibited high susceptibility to contamination ($p \le 0.0005$, $\Delta\text{Spearman} > 0.20$), while Seed 123 showed non-monotonic behavior at Dose 1.00.
4. **Decoupling of Latent and Behavioral Metrics**: Keyword masking sensitivity ($L_{\text{behavior}}$) remained virtually unchanged ($\sim 10^{-4}$), demonstrating that latent representations can acquire temporal contamination without overt distortion in surface-level masking behavior.
5. **Decoupling of Latent and Economic Metrics**: Information coefficients for both 2Y Treasury yields ($\Delta IC_{2Y} \approx -0.019$) and SPY returns ($\Delta IC_{SPY} \approx +0.010$) failed to show statistically meaningful predictability shifts, indicating that simple downstream linear heads cannot directly convert subtle latent representation shifts into economic market returns without task-aligned supervisory training.

---

## 6. Official Artifact Manifest Inventory

All artifacts generated by `Phase4ArtifactWriter` have been written to disk:

* **Full Results Record**: `experiments/phase4_confirmatory/results/phase4_confirmatory_results.json` (7.7 MB)
* **Execution Environment Provenance**: `experiments/phase4_confirmatory/provenance/execution_environment.json`
* **Protocol Verification Provenance**: `experiments/phase4_confirmatory/provenance/protocol_verification.json`
* **Branch Manifests (25 files)**: `experiments/phase4_confirmatory/manifests/seed{seed}_d{dose}.json`
* **Branch Metrics (25 files)**: `experiments/phase4_confirmatory/metrics/seed{seed}_d{dose}.json`

---

## 7. Audit Sign-Off

* **Investigator / Agent**: Antigravity Confirmatory Execution Engine
* **Protocol Compliance**: 100% Strict Compliance with Protocol v1.2.4
* **Gate Verdict**: Phase 4B Confirmatory Execution Successfully Concluded
