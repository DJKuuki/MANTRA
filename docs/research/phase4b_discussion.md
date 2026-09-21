# Section 5: Discussion

**Stage**: Phase 5 — Manuscript Results, Figures & Discussion Synthesis  
**Repository**: `DJKuuki/MANTRA`  
**Protocol Version**: `1.2.4`  
**Scientific Code Freeze SHA**: `5ec0f03f3a5393d90462aa78d018d54b08cce126`  
**Locked Source Tree SHA**: `02fe0ced0d03a920a7f56887f1d674282d17bc108986d19d2df495b59d330bb6`  

---

## 5.1 What the Study Actually Demonstrates

The empirical findings of Phase 4B establish that controlled post-cutoff continued pretraining alters the latent geometry of language representations in financial encoders. When a pretrained encoder (`ProsusAI/finbert`) undergoes continued Masked Language Modeling (MLM) on future macroeconomic texts, the linear decodability of future monetary policy rate decisions ($\Delta\text{Rate}_{t+1}$) increases out-of-sample relative to an identically trained twin exposed only to contemporary pre-cutoff text.

This effect is observed at the branch level: 18 of 20 contaminated branches display positive representational leakage ($L_{\mathrm{repr}} > 0$), and 10 of 20 achieve nominal branch-level significance ($p < 0.05$) under one-sided right-tailed paired event-level sign-flip permutation tests. Furthermore, a post-hoc descriptive breadth check across out-of-sample events shows that probe MSE reductions occurred in 23 of 32 (71.9%) evaluated temporal-CV events (spanning 2016–2019), indicating that the directional advantage is not concentrated entirely in a single outlier meeting (though this remains a descriptive check rather than a confirmatory proof of universal alignment).

Crucially, this finding must be interpreted with statistical precision. Protocol v1.2.4 preregistered branch-level permutation tests but did not specify an omnibus decision rule or family-wise error rate correction across the 20 contaminated seed-dose conditions. Consequently, while Phase 4B provides substantial empirical evidence of representational leakage at the branch level, a formal global confirmatory rejection of $H_0^{\mathrm{repr}}$ is not claimed.

---

## 5.2 What the Study Does Not Demonstrate

Equally important are the negative and descriptive results across downstream evaluation layers:

1. **No Robust Improvement in Discrete Policy Prediction**: The binary co-primary endpoint (`next_scheduled_change_vs_hold`) failed to demonstrate statistically significant improvements across any branch ($p > 0.05$). Shifts in continuous rate change decodability did not translate into higher classification Macro-F1 scores under standard linear decision boundaries.
2. **No Behavioral Masking Sensitivity Shift**: The secondary behavioral metric ($L_{\mathrm{behavior}}$) remained descriptive and negligible ($\sim 10^{-4}$). Downstream models exhibited no anomalous reliance on masked prompt entities or dates, indicating that future exposure did not disrupt surface-level attribution mechanisms.
3. **No Downstream Economic False Alpha**: Downstream linear projections of model stance onto financial market endpoints (2-year Treasury yields and SPY equities) yielded Information Coefficient deltas whose 95% bootstrap confidence intervals consistently spanned zero. In Seed 42, Treasury-market Information Coefficients shifted in a negative direction ($\Delta\mathrm{IC} = -0.0966$ at $D=0.25$, bootstrap $p = 0.009$; $\Delta\mathrm{IC} = -0.0938$ at $D=0.50$, bootstrap $p = 0.000$), representing evidence in the direction opposite to the hypothesized improvement ($H_1^{\mathrm{econ}}: \Delta\mathrm{IC} > 0$) rather than support for downstream leakage.

Therefore, the study does not demonstrate that post-cutoff exposure produces "false alpha," profitable trading strategies, or overt behavioral modifications. Claims that temporal leakage directly corrupts practical financial decision-making or creates illusory market returns are unsupported by the empirical evidence.

---

## 5.3 Latent Temporal Leakage: Geometry Precedes Behavior

A central conceptual insight from Phase 4B is that **parametric temporal leakage can be latent**. In transformer language models, exposure to future tokens during self-supervised continued pretraining primarily adjusts the high-dimensional internal representation space. These adjustments can enhance the linear separability or decodability of future target concepts without necessarily propagating through downstream classification heads or altering surface-level input sensitivities.

This observation mirrors findings in mechanistic interpretability, where neural networks frequently encode latent features that remain dormant unless probed by specialized linear readouts. In financial NLP, evaluating models solely by downstream task performance or surface behavioral checks risks failing to detect representational contamination. Probing latent geometry directly via out-of-sample anchor representations provides a more sensitive diagnostic for parametric temporal leakage.

---

## 5.4 Mechanistic Hypotheses for Non-Monotonic Dose Response

A naive intuition might assume that temporal leakage should obey a monotonic law: higher contamination dose ($D$) should strictly produce greater representational leakage ($L_{\mathrm{repr}}$). The empirical evidence did not support a strictly monotonic dose-response relationship. Across the five optimization seeds, mean $L_{\mathrm{repr}}$ increases from $+0.00368$ at $D = 0.25$ to an aggregate peak of $+0.00507$ at $D = 0.75$, but declines to $+0.00291$ at $D = 1.00$ ([Figure 1](figures/phase4b/figure1_l_repr_dose_response.png)). In Seed 123, point estimates actually fall below clean baseline performance at $D = 0.50$ and $D = 1.00$.

We hypothesize several non-mutually-exclusive mechanisms that could account for this non-monotonicity:
- **Continued-Pretraining Interference**: At high contamination fractions ($D = 1.00$), the pretraining stream is composed entirely of post-cutoff documents from a distinct macroeconomic regime (the 2020–2022 pandemic shock). Ingesting this distribution without sham data may induce distribution shift or representation drift that disrupts the subtle geometry utilized by linear probes trained on 2016–2019 targets.
- **Catastrophic Interference / Forgetting**: Full contamination may overwrite earlier syntactic or semantic representations acquired during base pretraining, degrading the encoder's general feature extraction capability and offsetting leakage gains.
- **Optimization Noise & Loss Landscapes**: Continued pretraining for 100 steps on 256,000 tokens operates in a stochastic optimization regime. At $D = 1.00$, the optimizer may settle into distinct local minima where future information is stored in non-linear manifolds that are less accessible to linear probes.
- **Target Distribution Mismatch**: The post-cutoff text reflects extreme policy rate cuts and emergency quantitative easing, whereas the evaluation targets ($\Delta\text{Rate}_{t+1}$) span a cycle of gradual tightening (2016–2018) followed by modest accommodation (2019). High-dose exposure may misalign representations with the target decodability task.

These candidate mechanisms represent exploratory hypotheses that warrant further structural investigation; they are not demonstrated empirical mechanisms.

---

## 5.5 Stochastic Optimization as Scientific Variance

Phase 4B reveals substantial **seed-dependent optimization heterogeneity** in temporal leakage studies ([Figure 2](figures/phase4b/figure2_branch_significance_map.png)). Although all branches within a seed shared identical initial weights and masking schedules, results diverged markedly across the five optimization seeds:
- Seed 87 and Seed 2024 exhibited high susceptibility, with strong representational leakage across multiple dose levels ($p \le 0.0005$).
- Seed 123 showed pronounced non-monotonic instability, with negative $L_{\mathrm{repr}}$ at higher doses.
- Seeds 13 and 42 exhibited modest susceptibility, reaching nominal significance only at isolated doses.

This divergence indicates that whether leaked factual or directional information becomes linearly decodable depends on the specific gradient trajectory traversed during continued pretraining. Studies that evaluate temporal leakage or continued pretraining using a single random seed risk reporting idiosyncratic optimization artifacts rather than generalizable architectural properties.

---

## 5.6 The Layered Leakage Framework

Our results argue strongly against treating temporal leakage as a monolithic, binary condition (i.e., "the model is leaked" vs. "the model is clean"). Instead, the findings support a **four-stage layered framework**:

$$\begin{matrix}
\text{Stage 1: Future Exposure} & (D \in [0.25, 1.00]) \\
\Downarrow & \\
\text{Stage 2: Representation Shift} & (L_{\mathrm{repr}} > 0 \text{ in 18/20 branches}) \\
\Downarrow \quad (\text{Decoupling}) & \\
\text{Stage 3: Behavioral Manifestation} & (L_{\mathrm{behavior}} \approx 0, \text{ Binary } \Delta\text{Macro-F1} \approx 0) \\
\Downarrow \quad (\text{Decoupling}) & \\
\text{Stage 4: Economic Consequence} & (\Delta\text{IC} \text{ CIs cross zero, No false alpha})
\end{matrix}$$

Each transition in this chain involves substantial attenuation. Latent representation shifts do not automatically transfer into discrete task advantages, and downstream task advantages (when present) face heavy financial market noise that prevents predictable economic returns. Distinguishing these layers prevents conflating latent representation leakage with practical market exploitation.

---

## 5.7 The Null-Model Insight: Low Leakage $\neq$ High Quality

A critical methodological insight formalizes the relationship between leakage and model competence. Consider a trivial null model that outputs constant representations regardless of input text. In our evaluation protocol, such a model would achieve $L_{\mathrm{repr}} \equiv 0$ across all contamination doses, displaying perfect "temporal integrity." However, the model would possess zero predictive utility.

Therefore, **low temporal leakage does not imply a superior model**. Temporal leakage ($L_{\mathrm{repr}}$) and predictive competence ($C$) represent orthogonal evaluative dimensions. A rigorous audit of temporal models must assess both axes simultaneously. In Phase 4B, competence $C$ is formally cataloged as `NOT_EVALUATED_NO_EVAL_SPLIT` because the confirmatory protocol prioritized out-of-sample causal contrasts over a separate held-out post-2018 competence split. Future benchmarks should evaluate the joint Pareto frontier of $(C, L_{\mathrm{repr}})$.

---

## 5.8 Distinguishing Leakage from Normal Domain Adaptation

An important alternative explanation must be addressed: could the positive $L_{\mathrm{repr}}$ signal reflect generic domain adaptation rather than temporal leakage? That is, does continued pretraining simply improve representation quality by exposing the model to additional central-bank language?

Protocol v1.2.4 implemented rigorous causal controls specifically designed to address this confound:
1. **The Twin-Model Design**: Contaminated models are not compared to the raw unadapted base model; they are compared to an active "clean twin" that received the exact same token budget (256,000 tokens) and optimization steps (100 MLM steps).
2. **Contemporary Sham Pretraining**: The clean twin ingested contemporary pre-cutoff FOMC documents, providing matched domain adaptation to central-bank English.
3. **Temporal Differential**: The sole difference between clean and contaminated streams was the temporal origin of the text ($\le 2019$ vs. $\ge 2020$).

The matched clean-twin design substantially reduces generic domain adaptation as an explanation by equalizing compute and broad FOMC-domain exposure. However, it cannot completely eliminate subdomain, regime, topic, or rhetorical-distribution differences between pre-2020 and post-2020 documents. For example, subtle distribution shifts in post-2020 central-bank vocabulary (e.g., changes in rhetorical emphasis during emergency easing) may have coincidentally altered probe linearity. We discuss this boundary in [Section 6 (Limitations)](phase4b_limitations.md).

---

## 5.9 Evolution from Phase 3 Pilot to Phase 4B Confirmatory

The confirmatory findings of Phase 4B must be contextualized alongside the preliminary results of the Phase 3 pilot:
- **Phase 3 Pilot (Methodology Exploration)**: Conducted on a constrained sample of only 8 FOMC events. The pilot served as an end-to-end pipeline verification and yielded statistically inconclusive results ($p > 0.05$). The pilot demonstrated that 8 events provided insufficient statistical power to distinguish subtle representation shifts from sampling noise.
- **Phase 4B Confirmatory Study**: Motivated by the power calculations of Phase 3, Phase 4B scaled the evaluation to 40 total events ($N_{\mathrm{OOS}} = 32$ out-of-sample temporal-CV events spanning 2016–2019), 5 random seeds, 5 contamination doses, and 25 total empirical branches on CUDA hardware ($6,400,000$ total tokens processed).

Phase 3 and Phase 4B do not represent conflicting empirical claims, but rather an intentional progression from an underpowered feasibility pilot to a higher-powered confirmatory experiment. Preserving the Phase 3 null outcome in the scientific record maintains scientific transparency and documents the necessity of adequate event-level sample sizes in temporal leakage audits.
