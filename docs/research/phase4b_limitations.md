# Section 6: Limitations

**Stage**: Phase 5 — Manuscript Results, Figures & Discussion Synthesis  
**Repository**: `DJKuuki/MANTRA`  
**Protocol Version**: `1.2.4`  
**Scientific Code Freeze SHA**: `5ec0f03f3a5393d90462aa78d018d54b08cce126`  
**Locked Source Tree SHA**: `02fe0ced0d03a920a7f56887f1d674282d17bc108986d19d2df495b59d330bb6`  

---

## 6.1 Global Confirmatory Decision Rule Under-Specification

The most prominent methodological limitation of Phase 4B concerns the preregistration of its omnibus inferential procedure. Protocol v1.2.4 formally preregistered paired sign-flip permutation tests ($\alpha = 0.05, B = 2,000$) for individual branch-level contrasts. However, the preregistration specification omitted a formal global decision rule defining how to combine the 20 contaminated seed-dose contrasts into a single omnibus confirmatory test (such as an explicit Fisher or Stouffer combination, a hierarchical mixed-effects model, or a preregistered Bonferroni-Holm family-wise error rate correction).

Because post-hoc specification of an omnibus test statistic would violate confirmatory protocol integrity, we report the branch-level outcomes descriptively and nominally (18/20 branches $L_{\mathrm{repr}} > 0$; 10/20 nominal $p < 0.05$) while explicitly refraining from claiming a formal global confirmatory rejection of $H_0^{\mathrm{repr}}$. Future preregistered protocols must define both branch-level and global omnibus test statistics prior to protocol lock.

---

## 6.2 Optimization Seed Heterogeneity

Empirical outcomes exhibit substantial sensitivity to the random initialization seed used during continued pretraining and classifier transfer ([Table 2](file:///e:/MANTRA/docs/research/phase4b_manuscript_tables.md#2-table-2-complete-20-branch-contaminated-empirical-results-supplementary-material)). Across the five seeds, representational leakage estimates range from $+0.00964$ ($p = 0.0005$ in Seed 87 at $D=0.75$) to $-0.00218$ ($p = 0.8955$ in Seed 123 at $D=1.00$).

This dispersion underscores that temporal leakage susceptibility is not a deterministic property of model architecture or corpus composition alone, but is modulated by stochastic optimization dynamics (batch order, token masking patterns, gradient paths). Consequently, reporting across-seed averages without documenting seed-level variance obscures meaningful scientific uncertainty. Auditing pipelines must account for seed-level variance through multi-seed designs.

---

## 6.3 Non-Monotonic Dose-Response Dynamics

The observed dose-response relationship between contamination exposure and representational leakage is non-monotonic: mean $L_{\mathrm{repr}}$ peaks near $D = 0.75$ (+0.00507) and attenuates at $D = 1.00$ (+0.00291). In Seed 123, representational decodability falls below the clean baseline at intermediate and full contamination doses.

The current protocol was designed to detect the presence and magnitude of leakage contrasts; it was not designed to resolve fine-grained mechanistic transitions between 0.75 and 1.00 or isolate the threshold where representation drift or catastrophic interference begins to dominate. Hypotheses regarding representation reshaping, distribution shift, or interference remain exploratory and require targeted layer-wise probes.

---

## 6.4 Model Scope and Architecture

Phase 4B was conducted exclusively using the `ProsusAI/finbert` encoder, a 110-million parameter BERT-base architecture fine-tuned for financial sentiment. These findings cannot be automatically extrapolated to:
- Large autoregressive decoder models (e.g., Llama, Mistral, GPT-family architectures);
- Models with parameter scales exceeding billions of parameters;
- Models trained under causal language modeling (CLM) objectives rather than masked language modeling (MLM);
- Reasoning or instruction-tuned models where parameter updates occur under reinforcement learning from human feedback (RLHF).

The mechanics of token retention, feature decodability, and memorization may differ substantially across model architectures and training paradigms.

---

## 6.5 Domain Scope and Macroeconomic Environment

The empirical evaluation focused specifically on Federal Open Market Committee (FOMC) monetary policy communication. Central bank communication is characterized by a specialized, highly formal vocabulary, institutional conventions, and scheduled policy decision cycles.

Furthermore, the post-cutoff contamination period (2020–2022) coincided with unprecedented macroeconomic shocks (the COVID-19 pandemic, zero lower bound rate policies, supply-chain dislocations, and subsequent aggressive inflation tightening). Whether similar representational leakage patterns emerge in less institutionalized domains (e.g., earnings call transcripts, financial news feeds, social media discourse) or under stationary macroeconomic regimes remains an open empirical question.

---

## 6.6 Statistical Power for Downstream Economic Endpoints

Economic leakage effects ($E_L$) were evaluated by correlating model stance predictions with 2-year Treasury yields and SPY equity returns across 40 FOMC events. While this event count provided substantial statistical power to identify representational shifts ($L_{\mathrm{repr}}$) across $N_{\mathrm{OOS}} = 32$ out-of-sample meetings, financial market returns are inherently noisy and influenced by exogenous macroeconomic variables (unemployment data, geopolitical events, fiscal policy announcements) that occur concurrently with FOMC announcements.

The failure to establish statistically robust economic gains ($\Delta\mathrm{IC}$) may reflect:
1. True decoupling: latent representational shifts do not translate into linear market predictability;
2. Limited statistical power: subtle economic effects (e.g., $\Delta\mathrm{IC} \approx 0.01$–$0.02$) cannot be definitively distinguished from zero in a sample of 40 events.

Our conclusion that "representational leakage did not reliably propagate into economically meaningful predictive improvement" accurately reflects the empirical evidence under the preregistered design, but should not be taken as proof that market predictability is impossible under all circumstances or with non-linear trading strategies.

---

## 6.7 Absence of Independent Competence Split ($C = \text{NOT\_EVALUATED}$)

In the formal taxonomy of temporal leakage, model competence ($C$) measures the baseline predictive skill of the model on the primary task under non-leaked conditions. In Phase 4B, competence is formally recorded as:

$$C = \text{NOT\_EVALUATED\_NO\_EVAL\_SPLIT}$$

Protocol v1.2.4 prioritized out-of-sample temporal cross-validation across all 40 events for the causal leakage contrast ($L_{\mathrm{repr}}$) rather than reserving a separate held-out post-2018 validation split to benchmark isolated stance competence. While model stance outputs demonstrate reasonable empirical discrimination (Macro-F1 baseline $\approx 0.43$ under 3-class distribution), an independent competence metric was not evaluated under this confirmatory scope.

---

## 6.8 Absence of Temporal Robustness Control ($R_T = \text{NOT\_EVALUATED}$)

Temporal robustness ($R_T$) evaluates how model performance degrades when evaluated across temporal regime shifts without contamination. In Phase 4B, this metric was formally excluded from the confirmatory scope:

$$R_T = \text{NOT\_EVALUATED}$$

Consequently, Phase 4B isolates the contrast between clean and contaminated models across identical temporal windows, but does not benchmark the baseline rate of temporal degradation that occurs naturally across time in clean language encoders.

---

## 6.9 Future-Target Decodability vs. Factual Memorization

The primary metric $L_{\mathrm{repr}}$ measures the linear decodability of future monetary policy rate changes ($\Delta\text{Rate}_{t+1}$) from anchor text embeddings. While the twin-model design controls for compute and domain adaptation, increased decodability reflects geometric alignment of representations rather than direct proof of factual memorization (e.g., verbatim regurgitation of future statements or numerical decisions).

Representation probes evaluate whether post-cutoff pretraining reshaped embedding subspaces in ways that correlate with future policy shifts. Causal claims asserting that "FinBERT memorized specific future FOMC decisions" or "future knowledge directly drove decisions" are avoided as overstatements. The appropriate scientific description is that controlled exposure to future text altered representation-level decodability of future policy variables.
