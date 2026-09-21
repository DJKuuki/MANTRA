# Section 4: Empirical Results

**Stage**: Phase 5 — Manuscript Results, Figures & Discussion Synthesis  
**Repository**: `DJKuuki/MANTRA`  
**Protocol Version**: `1.2.4`  
**Execution Context**: 25 Empirical Branches (5 Seeds $\times$ 5 Contamination Doses), Local CUDA Hardware (`NVIDIA GeForce GTX 1660 SUPER`)  
**Archived Head SHA**: `5d82a7a321f6ce2441e449b5fff7ec482681819e`  
**Scientific Code Freeze SHA**: `5ec0f03f3a5393d90462aa78d018d54b08cce126`  
**Locked Source Tree SHA**: `02fe0ced0d03a920a7f56887f1d674282d17bc108986d19d2df495b59d330bb6`  

---

## 4.1 Execution Integrity and Causal Symmetry

The empirical evaluation was conducted from scratch across all 25 experimental branches specified in Protocol v1.2.4 under strict cryptographic isolation. No historical or aborted v1.2.3 artifacts were reused. All branches executed in `EMPIRICAL` mode on local CUDA hardware using the canonical `ProsusAI/finbert` encoder body (commit `4556d13015211d73dccd3fdd39d39232506f3e43`).

### Causal Symmetry Invariants
To isolate the causal effect of future text exposure from procedural confounds, every experimental branch within a given random seed $s \in \{13, 42, 87, 123, 2024\}$ shared bit-identical initial conditions prior to treatment:
1. **Identical Pretraining Computation**: Every branch processed an exact budget of 256,000 tokens across 100 optimization steps of continued pretraining via Masked Language Modeling (MLM). Contaminated branches substituted pre-cutoff sham tokens with post-cutoff FOMC text in exact proportions governed by the dose parameter $D \in \{0.25, 0.50, 0.75, 1.00\}$, while clean baseline branches ($D = 0.00$) ingested exclusively pre-cutoff sham text.
2. **Deterministic Mask Schedules**: The random masking schedule for MLM was locked to seed $s$, ensuring that identical token positions were masked across all five dose conditions within that seed (`mask_schedule_hash` verified invariant).
3. **Identical Downstream Optimization**: After MLM continued pretraining, the updated encoder bodies were transferred to a 3-class FOMC stance classification head with bit-identical random weight initializations (`downstream_initial_head_hash` invariant) and fine-tuned for exactly 3 epochs (324 optimization steps) on pre-2019 Trillion Dollar Words training samples under identical sample shuffling orders (`downstream_sample_order_hash` invariant).
4. **Temporal Separation**: Pre-cutoff training documents terminated strictly on or before `2019-12-11T19:00:00Z` (pre-cutoff boundary `2019-12-31T23:59:59Z`). Contamination documents commenced on `2020-01-29T19:00:00Z`, enforcing an absolute 48-day buffer and zero document overlap between clean and contamination corpora.

All 25 branches satisfied all cryptographic checks and invariant hashes without exception (Table 3 in Supplementary Material).

---

## 4.2 Representational Temporal Leakage ($L_{\mathrm{repr}}$)

Representational temporal leakage was evaluated as the primary continuous endpoint. For each contaminated branch $(s, D)$, a regularized linear probe was trained on frozen paragraph embeddings of 181 standardized anchor paragraphs to decode the continuous next-meeting target rate change ($\Delta\text{Rate}_{t+1}$). Out-of-sample evaluation was performed across $N_{\mathrm{OOS}} = 32$ independent out-of-sample FOMC events under 4-fold grouped expanding-window temporal cross-validation (spanning 2016–2019 prior to the global 2019-12-31 cutoff). Here, "future" refers to the next scheduled policy outcome relative to each anchor event/meeting, not to events occurring after the global 2019-12-31 experimental cutoff. The branch test statistic $L_{\mathrm{repr}}$ measures the paired reduction in out-of-sample absolute error relative to the clean baseline branch ($D = 0.00$) of the same seed:

$$L_{\mathrm{repr}} = \frac{1}{N_{\mathrm{OOS}}} \sum_{e \in \mathrm{OOS}} \left( \left| y_e - \hat{y}_{\mathrm{clean},e} \right| - \left| y_e - \hat{y}_{\mathrm{leak},e} \right| \right)$$

Statistical significance was determined via a preregistered one-sided right-tailed paired event-level sign-flip permutation test ($B = 2,000$ draws) on the paired event-level deltas $d_e$, testing the directional alternative hypothesis $H_1^{\mathrm{repr}}: L_{\mathrm{repr}} > 0$.

### Primary Confirmatory Findings
Across the 20 contaminated experimental branches:
- **Directional Consistency**: 18 of the 20 branches (90.0%) exhibited positive point estimates ($L_{\mathrm{repr}} > 0$), indicating that exposure to post-cutoff text systematically reduced probe absolute prediction errors on future policy decisions.
- **Nominal Branch-Level Significance**: 10 of the 20 branches (50.0%) achieved nominal significance at $\alpha = 0.05$ ($p < 0.05$).
- **Dose-Response Profile**: Mean $L_{\mathrm{repr}}$ across seeds (*post-hoc descriptive*) by contamination dose was:
  - $D = 0.25$: $+0.00368 \pm 0.00165$ (2/5 branches nominal $p < 0.05$)
  - $D = 0.50$: $+0.00373 \pm 0.00285$ (3/5 branches nominal $p < 0.05$)
  - $D = 0.75$: $+0.00507 \pm 0.00312$ (3/5 branches nominal $p < 0.05$)
  - $D = 1.00$: $+0.00291 \pm 0.00455$ (2/5 branches nominal $p < 0.05$)

As depicted in Figure 1 and Figure 2, the aggregate empirical effect peaks at $D = 0.75$ and decreases at $D = 1.00$. **The observed dose response is not strictly monotonic.** The empirical data did not support a simple monotonic dose-response relationship.

### Preregistration Boundary on Global Confirmatory Rejection
While branch-level permutation tests were formally preregistered, Protocol v1.2.4 did not specify an omnibus procedure (e.g., Fisher combination, Stouffer method, or hierarchical mixed-effects model) for aggregating the 20 distinct seed-dose hypothesis tests into a single global rejection criterion.

> **Confirmatory Boundary Statement**:  
> Phase 4B provides substantial branch-level evidence that controlled post-cutoff continued pretraining can increase the decodability of future monetary-policy information in FinBERT representations. However, Protocol v1.2.4 did not preregister a single global procedure for combining the 20 contaminated seed-dose contrasts, so a formal global confirmatory rejection of $H_0^{\mathrm{repr}}$ is not claimed.

### Finite Monte-Carlo Reporting Sensitivity
In Seed 87 at Dose 1.00, the frozen evaluator recorded an empirical permutation $p$-value of $p = 0.0000$, indicating that none of the $B = 2,000$ random sign-flip permutations exceeded the observed test statistic ($k = 0$). To preserve historical precision while adhering to statistical best practice, we report this frozen output alongside its finite Monte-Carlo resolution:

$$\hat{p}_{\mathrm{finite}} = \frac{k + 1}{B + 1} = \frac{1}{2001} \approx 0.00050$$

This adjustment is labeled strictly as **POST-EXECUTION REPORTING SENSITIVITY** and does not alter the branch's nominal significance status ($p < 0.05$).

---

## 4.3 Dose and Seed Heterogeneity

Disaggregating results by optimization seed reveals pronounced heterogeneity in leakage susceptibility across identical training pipelines (Figure 1 and Figure 2):

1. **Seed 87 (High Susceptibility)**: Demonstrated strong and consistent leakage across all doses, peaking at $D = 0.75$ ($L_{\mathrm{repr}} = +0.00964, p = 0.0005, \Delta\text{Spearman} = +0.2602$) and remaining nominally significant at $D = 0.50$ ($p = 0.0060$) and $D = 1.00$ ($p < 0.0005$).
2. **Seed 2024 (Monotonic Progression)**: Showed monotonic increases in representational decodability across dose levels, progressing from $L_{\mathrm{repr}} = +0.00247$ ($p = 0.1290$) at $D=0.25$ to $+0.00889$ ($p = 0.0005, \Delta\text{Spearman} = +0.1551$) at $D=1.00$, with 3 of 4 contaminated doses reaching nominal significance.
3. **Seed 123 (Marked Non-Monotonicity and Reversal)**: Exhibited substantial instability across dose levels. While $D = 0.25$ showed significant leakage ($L_{\mathrm{repr}} = +0.00640, p = 0.0075$) and $D = 0.75$ was also nominally significant ($+0.00385, p = 0.0395$), intermediate and high doses dipped below the clean baseline ($D=0.50: L_{\mathrm{repr}} = -0.00024, p = 0.5400$; $D=1.00: L_{\mathrm{repr}} = -0.00218, p = 0.8955$).
4. **Seeds 13 and 42 (Moderate / Isolated Response)**: Displayed moderate effects that achieved nominal significance at single intermediate doses: Seed 13 at $D = 0.50$ ($L_{\mathrm{repr}} = +0.00545, p = 0.0255, \Delta\text{Spearman} = +0.2685$) and Seed 42 at $D = 0.25$ ($L_{\mathrm{repr}} = +0.00402, p = 0.0090, \Delta\text{Spearman} = +0.0698$), with effects attenuating at higher doses.

This variance is interpreted as **training-randomness sensitivity / seed-dependent leakage susceptibility**. Stochastic factors in continued pretraining (batch sampling order, MLM mask placement) materially dictate whether leaked parametric information organizes into linearly decodable representation subspaces.

### Event-Level Effect Distribution
To evaluate the breadth of event-level improvements and confirm they were not driven by an isolated outlier meeting, event-level paired absolute-error reductions $d_e$ were examined across the 32 out-of-sample FOMC events (Figure 3):
- Each dose panel aggregates 160 branch-event observations (32 OOS events × 5 seeds). These observations are not independent statistical units and are displayed for post-hoc descriptive visualization only; the preregistered inferential unit remains the independent FOMC event. Within these descriptive distributions, positive absolute-error reductions predominate across doses: 61.3% of observations at $D=0.25$, 58.8% at $D=0.50$, 65.0% at $D=0.75$, and 62.5% at $D=1.00$ show $d_e > 0$ (Figure 3A).
- A post-hoc descriptive breadth check found that positive paired absolute-error reductions were observed in the median across contaminated branches for 23 of 32 OOS events (Figure 3B). This indicates that the directional advantage was not concentrated in a single isolated event, but this summary is not an independent confirmatory test.

---

## 4.4 Binary Policy Endpoint ($H_1^{\mathrm{binary}}$)

The preregistered co-primary endpoint evaluated discrete policy rate classification (`next_scheduled_change_vs_hold`) using an identical cross-validation structure. Across all 20 contaminated branches, the binary co-primary was **not supported**:
- **Point Estimates**: Mean across-seed $\Delta\text{Macro-F1}$ (*post-hoc descriptive*) remained close to zero at all doses: $-0.0169$ at $D=0.25$, $-0.0030$ at $D=0.50$, $+0.0273$ at $D=0.75$, and $+0.0058$ at $D=1.00$ (Table 1).
- **Statistical Significance**: Permutation tests yielded $p > 0.05$ across all 20 branches (minimum $p = 0.2510$ in Seed 87, $D=0.75$).

Continuous future rate changes ($\Delta\text{Rate}_{t+1}$) carry directional magnitude information that linear ridge probes successfully decode from shifted latent geometries. However, discrete change-vs-hold classification collapses these continuous margins into a coarse decision boundary dominated by the historical class prior (hold). Increased latent decodability of future rate changes does not automatically translate into improved discrete classification accuracy under standard linear heads.

---

## 4.5 Behavioral Leakage Endpoint ($L_{\mathrm{behavior}}$)

The secondary behavioral endpoint evaluated the shift in model sensitivity to entity and date token masking within FOMC statements across 40 meetings ($L_{\mathrm{behavior}} = \frac{1}{N} \sum_{e} (S_{\mathrm{mask}}^{\mathrm{leak}} - S_{\mathrm{mask}}^{\mathrm{clean}})$).

As preregistered, this endpoint is treated as **DESCRIPTIVE ONLY**:
- **Shift Magnitude**: Across all contaminated branches, observed shifts were negligible, hovering around zero on the order of $10^{-4}$ (across-seed mean: $-0.00005$ at $D=0.25$, $+0.00046$ at $D=0.50$, $-0.00030$ at $D=0.75$, and $+0.00000$ at $D=1.00$).
- **Empirical Summary**: No material shift in the preregistered masking-sensitivity endpoint was observed, despite measurable changes in latent representational decodability.

Surface-level text sensitivity remained intact; the downstream model did not exhibit degenerate or anomalous reliance on masked prompt entities as a consequence of continued pretraining on future text.

---

## 4.6 Economic Consequences ($E_L$)

Economic leakage effects were evaluated using downstream linear stance projections against two financial market endpoints: 2-year US Treasury yield changes (primary economic endpoint, $E_L(2\text{Y})$) and SPY daily equity returns (exploratory economic endpoint, $E_L(\text{SPY})$). For each event, model stance predictions were correlated with market returns to compute an Information Coefficient (Spearman rank correlation), with $\Delta\text{IC} = \text{IC}_{\mathrm{leak}} - \text{IC}_{\mathrm{clean}}$. Statistical significance was evaluated via 1,000 bootstrap resamples.

### Primary Economic Endpoint: 2-Year Treasury Yields
Across all 20 contaminated branches, the primary economic endpoint was **not supported**:
- **Absence of Positive Economic Support**: No branch provided support for the preregistered positive economic alternative ($H_1^{\mathrm{econ}}: \Delta\mathrm{IC}_{2\mathrm{Y}} > 0$). Most bootstrap confidence intervals included zero (Table 2). Mean $\Delta\text{IC}_{2\mathrm{Y}}$ across seeds (*post-hoc descriptive*) was negative at all doses: $-0.0224$ ($D=0.25$), $-0.0182$ ($D=0.50$), $-0.0197$ ($D=0.75$), and $-0.0186$ ($D=1.00$).
- **Directional Interpretation in Seed 42**: In Seed 42 at $D=0.25$ and $D=0.50$, the frozen bootstrap procedure produced small sign-tail probabilities ($p = 0.009$ and $p = 0.000$) and 95% bootstrap confidence intervals entirely below zero ($\Delta\text{IC}_{2\mathrm{Y}} = -0.0966$ and $-0.0938$). Because the preregistered economic alternative was directional ($H_1^{\mathrm{econ}}: \Delta\mathrm{IC} > 0$), these negative shifts represent evidence in the opposite direction and do not support the preregistered economic leakage hypothesis.

### Exploratory Economic Endpoint: SPY Equities
Similarly, SPY returns showed no reliable leakage effect:
- Mean $\Delta\text{IC}_{\mathrm{SPY}}$ was $+0.0089$ at $D=0.25$, $+0.0100$ at $D=0.50$, $+0.0108$ at $D=0.75$, and $-0.0149$ at $D=1.00$. All 95% bootstrap confidence intervals for SPY include zero.

> **Economic Empirical Verdict**:  
> Representational leakage did not reliably propagate into economically meaningful predictive improvement under the preregistered market endpoints. The experimental data provide no basis for claiming "false alpha" or market profitability resulting from post-cutoff contamination in this setting.

---

## 4.7 Summary of the Leakage Propagation Chain

Figure 4 visualizes the empirical evidence across all five evaluated layers. The empirical outcomes demonstrate systematic decoupling across the analytical stack:

```
[Layer 1: Treatment Exposure]
  256k Tokens Controlled Contamination (D = 0.25 to 1.00)
             │
             ▼
[Layer 2: Representation Space (L_repr)]  <── [SIGNAL CONCENTRATED HERE]
  18/20 branches > 0, 10/20 nominal p < 0.05, Peak at D = 0.75
             │
             ├──────────────────────────┐
             ▼                          ▼
[Layer 3: Policy Binary Task]   [Layer 3: Behavioral Masking (L_behavior)]
  Delta F1 ~ 0, All p > 0.05      Delta S_mask ~ 10^-4 (Descriptive Only)
             │                          │
             └──────────────┬───────────┘
                            ▼
[Layer 4: Market Outcomes (E_L 2Y & SPY)]
  Delta IC CIs include zero or shift negative, No false alpha detected
```

Under Protocol v1.2.4:
1. **Representational decodability ($L_{\mathrm{repr}}$)** exhibits substantial branch-level signal.
2. **Binary policy classification** is not supported ($p > 0.05$).
3. **Behavioral masking sensitivity ($L_{\mathrm{behavior}}$)** is descriptively negligible.
4. **Economic predictability ($E_L$)**: Positive alternative is not supported.
5. **Predictive Competence ($C$)**: Status remains `NOT_EVALUATED_NO_EVAL_SPLIT` (no independent post-2018 evaluation split).
6. **Temporal Robustness ($R_T$)**: Status remains `NOT_EVALUATED` (excluded from confirmatory scope).

These distinct dimensions must not be collapsed into a single scalar "leakage score." Parametric temporal leakage in FinBERT manifests prominently in latent geometry while decoupling from surface behavioral sensitivity and downstream economic returns.
