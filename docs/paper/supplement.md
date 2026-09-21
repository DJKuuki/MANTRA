# Supplementary Material: Parametric Temporal Leakage in Financial Language Models

**Stage**: Phase 6 — Paper Factual Consistency & Reference Verification Patch  
**Repository**: `DJKuuki/MANTRA`  
**Protocol Version**: `1.2.4`  
**Accompanying Manuscript**: *Parametric Temporal Leakage in Financial Language Models: Probing Latent Representations Under Causally Symmetric Pretraining*  

---

## S1. Complete 20-Branch Contaminated Empirical Results

Table S1 reports complete, unaggregated metrics across all 20 contaminated experimental branches ($D \in \{0.25, 0.50, 0.75, 1.00\}$ across 5 random seeds). Clean baseline twins ($D = 0.00$) serve as the within-seed paired reference ($L_{\mathrm{repr}} = 0$, $\Delta\mathrm{IC} = 0$, $p = 1.000$).

```text
Table S1: Complete 20-Branch Contaminated Empirical Results
Inferential Unit: 32 out-of-sample temporal cross-validation FOMC meetings (2016–2019).
Primary Representational Test: One-sided right-tailed paired event-level sign-flip permutation test (B=2,000).
Economic Test: 1,000-draw event-level stationary block bootstrap; directional alternative Delta IC > 0.
```

| Seed | Dose ($D$) | $L_{\mathrm{repr}}$ | Frozen Perm $p$ | Finite-MC Reporting Note | Nom. Sig. ($\alpha=0.05$) | $\Delta\mathrm{Spearman}$ | Binary $\Delta\mathrm{Macro\text{-}F1}$ | Binary $p$ | $L_{\mathrm{behavior}}$ | $\Delta\mathrm{IC}_{2\mathrm{Y}}$ | 95% Bootstrap CI (2Y) | $p_{2\mathrm{Y}}$ | $\Delta\mathrm{IC}_{\mathrm{SPY}}$ | 95% Bootstrap CI (SPY) | $p_{\mathrm{SPY}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **13** | 0.25 | +0.00245 | 0.0800 | — | False | +0.0906 | +0.0000 | 1.0000 | +0.00000 | +0.0081 | [-0.0461, +0.0659] | 0.284 | +0.0124 | [-0.0145, +0.0515] | 0.215 |
| **13** | 0.50 | +0.00545 | 0.0255 | — | **True** | +0.2685 | +0.0116 | 0.4990 | +0.00015 | +0.0173 | [-0.0580, +0.0717] | 0.241 | +0.0170 | [-0.0167, +0.0710] | 0.114 |
| **13** | 0.75 | +0.00313 | 0.1425 | — | False | +0.0011 | +0.0000 | 1.0000 | -0.00027 | +0.0177 | [-0.0533, +0.0955] | 0.253 | +0.0060 | [-0.0275, +0.0554] | 0.281 |
| **13** | 1.00 | +0.00090 | 0.3685 | — | False | +0.0182 | +0.0000 | 1.0000 | -0.00039 | +0.0028 | [-0.0603, +0.0626] | 0.409 | -0.0148 | [-0.0606, +0.0034] | 0.066 |
| **42** | 0.25 | +0.00402 | 0.0090 | — | **True** | +0.0698 | -0.0523 | 0.7435 | +0.00105 | -0.0966 | [-0.1775, -0.0147] | 0.009* | -0.0655 | [-0.1561, +0.0367] | 0.106 |
| **42** | 0.50 | +0.00173 | 0.0870 | — | False | +0.0000 | +0.0174 | 0.5040 | +0.00207 | -0.0938 | [-0.1798, -0.0246] | 0.000* | -0.0455 | [-0.1176, +0.0375] | 0.111 |
| **42** | 0.75 | +0.00195 | 0.0980 | — | False | -0.0317 | +0.0174 | 0.5040 | -0.00042 | -0.0502 | [-0.1221, +0.0087] | 0.057 | -0.0201 | [-0.0824, +0.0532] | 0.216 |
| **42** | 1.00 | +0.00061 | 0.3710 | — | False | +0.0265 | +0.0174 | 0.5040 | -0.00119 | -0.0788 | [-0.1804, +0.0144] | 0.047 | -0.0475 | [-0.1345, +0.0028] | 0.033 |
| **87** | 0.25 | +0.00304 | 0.0600 | — | False | -0.0328 | +0.0174 | 0.5045 | +0.00025 | +0.0244 | [-0.0438, +0.0931] | 0.221 | -0.0094 | [-0.1099, +0.0625] | 0.333 |
| **87** | 0.50 | +0.00528 | 0.0060 | — | **True** | +0.1223 | +0.0174 | 0.5045 | -0.00014 | +0.0135 | [-0.0414, +0.0729] | 0.323 | -0.0039 | [-0.0902, +0.0795] | 0.380 |
| **87** | 0.75 | +0.00964 | 0.0005 | — | **True** | +0.2602 | +0.0351 | 0.2510 | -0.00038 | +0.0013 | [-0.0601, +0.0669] | 0.510 | -0.0146 | [-0.0643, +0.0250] | 0.195 |
| **87** | 1.00 | +0.00632 | 0.0000 | k=0/2000; res. ~ 1/2001 = 0.00050 | **True** | +0.1339 | +0.0174 | 0.4990 | +0.00044 | +0.0121 | [-0.0840, +0.1581] | 0.440 | -0.0518 | [-0.2169, +0.0330] | 0.137 |
| **123** | 0.25 | +0.00640 | 0.0075 | — | **True** | +0.0462 | -0.0494 | 0.4955 | -0.00061 | -0.0392 | [-0.1896, +0.0686] | 0.267 | +0.0701 | [-0.0230, +0.1680] | 0.065 |
| **123** | 0.50 | -0.00024 | 0.5400 | — | False | -0.0118 | -0.0615 | 0.7510 | +0.00032 | -0.0255 | [-0.1029, +0.0521] | 0.293 | +0.0434 | [-0.0049, +0.1075] | 0.036 |
| **123** | 0.75 | +0.00385 | 0.0395 | — | **True** | +0.0777 | +0.0174 | 0.5120 | +0.00032 | -0.0367 | [-0.1469, +0.0611] | 0.308 | +0.0607 | [-0.0092, +0.1716] | 0.039 |
| **123** | 1.00 | -0.00218 | 0.8955 | — | False | -0.0094 | -0.0173 | 1.0000 | +0.00188 | -0.0312 | [-0.1632, +0.0674] | 0.290 | +0.0540 | [-0.0273, +0.1256] | 0.088 |
| **2024** | 0.25 | +0.00247 | 0.1290 | — | False | +0.0593 | +0.0000 | 1.0000 | -0.00093 | -0.0085 | [-0.0818, +0.0551] | 0.346 | +0.0370 | [-0.0206, +0.1171] | 0.121 |
| **2024** | 0.50 | +0.00643 | 0.0205 | — | **True** | +0.1733 | +0.0000 | 1.0000 | -0.00012 | -0.0025 | [-0.1220, +0.1448] | 0.514 | +0.0390 | [-0.0449, +0.1492] | 0.206 |
| **2024** | 0.75 | +0.00680 | 0.0135 | — | **True** | +0.2144 | +0.0668 | 0.7740 | -0.00077 | -0.0307 | [-0.1271, +0.0662] | 0.273 | +0.0219 | [-0.0274, +0.0839] | 0.218 |
| **2024** | 1.00 | +0.00889 | 0.0005 | — | **True** | +0.1551 | +0.0116 | 0.4990 | -0.00073 | +0.0019 | [-0.0756, +0.0760] | 0.505 | -0.0145 | [-0.1306, +0.0710] | 0.393 |

### Explanatory Notes for Table S1:
1. **Nominal Significance Definition**: A branch is marked **True** if and only if its sign-flip permutation $p$-value satisfies $p < 0.05$ under the preregistered one-sided right-tailed paired event-level sign-flip permutation test ($B=2,000$, $\alpha = 0.05$) on $H_1^{\mathrm{repr}}: L_{\mathrm{repr}} > 0$. Exactly 10 of 20 branches (50.0%) reach nominal significance.
2. **Finite-Monte-Carlo Reporting Note for $p=0.0000$**: In Seed 87 at Dose 1.00, $0$ of $B=2,000$ permutation draws exceeded the observed statistic ($k=0$), yielding raw $p = 0.0000$. Under standard finite-sample Monte-Carlo reporting sensitivity $\frac{k+1}{B+1}$, this corresponds to $p_{\mathrm{plus\_one}} = 1/2001 \approx 0.00050$. For all other branches where $k > 0$, the historical permutation $p$-value is reported directly without modification.
3. **Directional Economic Invalidation (*)**: In Seed 42 at Doses 0.25 and 0.50, the frozen bootstrap procedure produced small sign-tail probabilities ($0.009$ and $0.000$) and 95% bootstrap confidence intervals entirely below zero ($\Delta\mathrm{IC}_{2\mathrm{Y}} = -0.0966$ and $-0.0938$). Because the preregistered economic alternative was directional ($H_1^{\mathrm{econ}}: \Delta\mathrm{IC} > 0$), these negative shifts represent evidence in the opposite direction and do not support the preregistered economic leakage hypothesis.

---

## S2. Causal Symmetry & Execution Audit Ledger

Table S2 documents the operational invariants enforced across all 25 empirical branches to guarantee causal symmetry, paired execution reproducibility, and absence of external data leakage.

```text
Table S2: Experimental Invariants, Causal Symmetry & Execution Audit Ledger
```

| Property / Invariant | Preregistered Specification | Actual Execution Verification | Audit Status |
| :--- | :--- | :--- | :---: |
| **Architecture Invariance** | `ProsusAI/finbert` (110M params) | 12 layers, 768 hidden dim, 12 attention heads, WordPiece | **PASS** |
| **MLM Treatment Stream Budget** | 500 blocks $\times$ 512 tokens = 256,000 tokens per branch | Constructed treatment-stream budget verified across all 25 branch manifests | **PASS** |
| **MLM Optimizer Execution** | Batch size 16, 100 gradient steps | 100 gradient steps executed with batch size 16 | **PASS** |
| **MLM Block Length** | 512 tokens per packed block | Exact 512-token packed blocks constructed from document corpus | **PASS** |
| **MLM Optimizer & Schedule** | AdamW ($\text{lr}=5\times 10^{-5}$, weight decay 0.01) | Scheduler: `none`, warmup_ratio: `0.0` (0 warmup steps), matched across twins | **PASS** |
| **MLM Masking Schedule** | 15% random dynamic masking | Masking seed coupled to branch seed $s$; identical across twin pairs | **PASS** |
| **Downstream Training Scope** | Full model supervised fine-tuning | Full sequence-classification model (`model.train()`, `model.parameters()`) fine-tuned with AdamW | **PASS** |
| **Downstream Hyperparameters**| 3 epochs, batch size 16, lr $2\times 10^{-5}$, weight decay 0.01 | Linear schedule with warmup ratio 0.1, max seq length 128 tokens, max steps 500 | **PASS** |
| **Downstream Realized Steps** | 3 epochs on pre-2019 TDW training set (1,729 samples) | 108 steps/epoch $\times$ 3 epochs = 324 realized optimizer steps per branch | **PASS** |
| **Downstream Head Init & Order** | Fresh shared 3-class head; paired sample order | Seed-coupled head init hash and sample order hash verified bit-identical across twins | **PASS** |
| **Representation Probing Stage** | Linear Ridge probe ($\alpha=1.0$) under 4-fold temporal CV | Probe fitted on frozen extracted representations post fine-tuning | **PASS** |
| **Pre-Cutoff Boundary** | $\le 2019\text{-}12\text{-}31\text{T}23:59:59\text{Z}$ | Max training timestamp: `2019-12-11T19:00:00Z` | **PASS** |
| **Post-Cutoff Boundary** | $\ge 2020\text{-}01\text{-}01\text{T}00:00:00\text{Z}$ | Min contamination timestamp: `2020-01-29T19:00:00Z` (48-day buffer) | **PASS** |
| **Corpus Overlap** | Zero text overlap | 0 duplicate documents between clean and contamination corpora | **PASS** |
| **Hardware Platform & Controls**| Dedicated CUDA GPU | NVIDIA GPU via PyTorch; paired execution symmetry and reproducibility controls | **PASS** |
| **Aggregate Treatment Volume** | $25 \times 256,000 = 6,400,000$ tokens | 6.4 million aggregate constructed treatment stream tokens | **PASS** |
| **Protocol Conformance** | Protocol Version `1.2.4` | Enforced by protocol lock file and SHA-256 pre-execution gate | **PASS** |

---

## S3. Dose-by-Seed Empirical Matrices

### S3.1 Representational Leakage ($L_{\mathrm{repr}}$) Matrix
The $5 \times 4$ matrix below presents the continuous representational leakage point estimates $L_{\mathrm{repr}}(s, D)$ for each contaminated branch:

$$\begin{array}{r|cccc}
\text{Seed} & D = 0.25 & D = 0.50 & D = 0.75 & D = 1.00 \\
\hline
13 & +0.00245 & +0.00545^* & +0.00313 & +0.00090 \\
42 & +0.00402^* & +0.00173 & +0.00195 & +0.00061 \\
87 & +0.00304 & +0.00528^* & +0.00964^* & +0.00632^* \\
123 & +0.00640^* & -0.00024 & +0.00385^* & -0.00218 \\
2024 & +0.00247 & +0.00643^* & +0.00680^* & +0.00889^* \\
\hline
\text{Mean (Descriptive)} & +0.00368 & +0.00373 & \mathbf{+0.00507} & +0.00291
\end{array}$$

*Note*: Asterisks ($*$) denote nominal significance ($p < 0.05$) under one-sided right-tailed paired sign-flip permutation tests ($B=2,000$). Exactly 18 of 20 cells are positive, and 10 of 20 achieve nominal significance.

### S3.2 Spearman Rank Correlation Delta ($\Delta\mathrm{Spearman}$) Matrix
The matrix below presents the difference in Spearman rank correlation between predicted and actual future rate changes ($\Delta\mathrm{Spearman} = \rho_{\mathrm{leak}} - \rho_{\mathrm{clean}}$):

$$\begin{array}{r|cccc}
\text{Seed} & D = 0.25 & D = 0.50 & D = 0.75 & D = 1.00 \\
\hline
13 & +0.0906 & +0.2685 & +0.0011 & +0.0182 \\
42 & +0.0698 & +0.0000 & -0.0317 & +0.0265 \\
87 & -0.0328 & +0.1223 & +0.2602 & +0.1339 \\
123 & +0.0462 & -0.0118 & +0.0777 & -0.0094 \\
2024 & +0.0593 & +0.1733 & +0.2144 & +0.1551 \\
\hline
\text{Mean (Descriptive)} & +0.0466 & +0.1105 & \mathbf{+0.1043} & +0.0649
\end{array}$$

The rank correlation deltas closely mirror the $L_{\mathrm{repr}}$ profile, peaking in aggregate at intermediate doses ($D = 0.50$ and $D = 0.75$).

---

## S4. Phase 3 Pilot Study Context & Sample-Size Power Justification

The confirmatory design of Phase 4B was directly informed by the preliminary findings of the Phase 3 pilot study:
- **Phase 3 Pilot Scope**: Evaluated an end-to-end prototype on a constrained sample of only 8 FOMC meetings (25 paragraph anchors) across 3 seeds and 5 doses (15 total branches).
- **Pilot Outcome**: Permutation tests failed to reject the null hypothesis ($p > 0.05$ across all branches), and correlation metrics fluctuated near zero ($\rho = -0.3536$).
- **Statistical Power Diagnosis**: Analytical power calculations conducted during Phase 3 revealed that evaluating subtle representation shifts (Cohen's $d \approx 0.35$–$0.50$) across only 8 paired events yielded statistical power below 25% at $\alpha = 0.05$. Detecting subtle representation leakage required scaling event units by at least $4\times$.
- **Phase 4B Confirmatory Resolution**: Motivated by this power diagnosis, Phase 4B scaled the evaluation corpus to 40 total meetings ($N_{\mathrm{OOS}} = 32$ out-of-sample temporal-CV meetings, 181 standardized paragraph anchors), expanded the seed count to 5 independent runs, and formalized pre-registered directional hypotheses.

Preserving the Phase 3 null result in the scientific record prevents publication bias and demonstrates that adequate event sample sizes are critical for detecting subtle parametric temporal leakage.

---

## S5. Comprehensive Publication Figure Captions

### Figure 1: Dose-Response Profiles of Representational Leakage ($L_{\mathrm{repr}}$)
- **Path**: `../research/figures/phase4b/figure1_l_repr_dose_response.png` (and `.pdf`)
- **Caption**: *Dose-response relationship between contamination exposure ($D \in [0.00, 1.00]$) and continuous representational leakage ($L_{\mathrm{repr}}$). Individual thin colored lines show the trajectory of each optimization seed ($s \in \{13, 42, 87, 123, 2024\}$). The thick dark blue curve overlays the post-hoc descriptive across-seed mean, with the shaded light blue envelope indicating $\pm 1 \text{ standard error of the mean (SEM)}$. Representational decodability peaks in aggregate at $D = 0.75$ ($+0.00507$) before declining at $D = 1.00$ ($+0.00291$), illustrating non-monotonic response dynamics.*

### Figure 2: Seed-by-Dose Branch Significance Heatmap
- **Path**: `../research/figures/phase4b/figure2_branch_significance_map.png` (and `.pdf`)
- **Caption**: *Heatmap of representational leakage point estimates ($L_{\mathrm{repr}}$) across all 20 contaminated experimental branches ($5 \text{ seeds} \times 4 \text{ doses}$). Cells are annotated with exact numerical point estimates. Asterisks denote nominal branch-level significance ($p < 0.05$) under one-sided right-tailed paired event-level sign-flip permutation tests ($B=2,000$). High susceptibility is observed in Seeds 87 and 2024, moderate isolated response in Seeds 13 and 42, and non-monotonic reversal in Seed 123.*

### Figure 3: Event-Level Error Reduction Distributions
- **Path**: `../research/figures/phase4b/figure3_event_level_deltas.png` (and `.pdf`)
- **Caption**: *Event-level paired absolute-error deltas ($d_e = |y_e - \hat{y}_{C,e}| - |y_e - \hat{y}_{L,e}|$) across 32 out-of-sample temporal cross-validation FOMC meetings. **Panel A**: Violin and box plots of pooled branch-events across contamination doses. Note: Each dose panel aggregates 160 branch-events (32 OOS events $\times$ 5 seeds) that are not statistically independent; this panel is presented for post-hoc descriptive visualization only. **Panel B**: Meeting-by-meeting waterfall of median paired deltas across contaminated branches, ordered chronologically from 2016 to 2019. Exactly 23 of 32 meetings (71.9%) show positive median error reductions, confirming that directional gains were not concentrated in a single meeting.*

### Figure 4: Layered Outcome Comparison Across Analytical Tiers
- **Path**: `../research/figures/phase4b/figure4_layered_outcome_comparison.png` (and `.pdf`)
- **Caption**: *Multi-panel comparison across the five evaluated endpoints under Protocol v1.2.4: continuous representational decodability ($L_{\mathrm{repr}}$), binary policy classification ($\Delta\text{Macro-F1}$), behavioral masking sensitivity ($L_{\mathrm{behavior}}$), and downstream financial market predictability ($\Delta\mathrm{IC}$ for 2-year Treasury yields and SPY equities). Status banners illustrate systematic attenuation across layers: while $L_{\mathrm{repr}}$ exhibits substantial branch-level signal (18/20 positive, 10/20 nominally significant), no reliable positive downstream effect was detected across the behavioral tier ($\Delta\mathrm{F1} \approx 0, L_{\mathrm{behavior}} \approx 10^{-4}$) or market tier ($\Delta\mathrm{IC}$ bootstrap CIs include zero or shift negative in Seed 42).*

---

## S6. Cryptographic Archival Hashes

To enable independent verification, the SHA-256 digests of key archival files are listed below:

| File Path | SHA-256 Digest |
| :--- | :--- |
| `configs/phase4_preregistration.yaml` | `a3915f79d0cb4d4239856f4d2f00d86927bfdfdd8a79fe9f7375bfbf57d76ec4` |
| `configs/phase4_protocol_lock.json` | `5c9ebcc742a04870f7cfcb577a72d7331822ea057a6279f727c6ef7bb2df586d` |
| `experiments/phase4_confirmatory/result_manifest.json` | `3ba196a666248ce1d55681907cbffefbf223d726615bcfdca6df496881c1c9aa` |
| `experiments/phase4_confirmatory/results/phase4_confirmatory_results.json` | `887ea0627df201e0d37edad19d820ac9fe41a1be09ef5835ff17720f89c235b2` |
| `data/research/fomc/phase4_contamination/contamination_manifest.json` | `378129df3eb0b62e497f3747aa1f021e204c3d82f2beab31c51a774ea0a1f7ba` |
