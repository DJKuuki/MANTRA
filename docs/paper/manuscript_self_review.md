# Adversarial Peer Review Audit: Parametric Temporal Leakage Manuscript

**Stage**: Phase 6 — Final Paper Factual Micro-Closure  
**Repository**: `DJKuuki/MANTRA`  
**Protocol Version**: `1.2.4`  
**Target Manuscript**: `docs/paper/manuscript.md`  

---

## Executive Review Overview

This document records an adversarial self-review of the assembled manuscript from four critical disciplinary perspectives: Mathematical Statistics, Causal Inference & Experimental Design, Natural Language Processing / Representation Learning, and Empirical Quantitative Finance.

The objective of this review is not to smooth over empirical limitations or defend flawed claims, but to stress-test every scientific assertion, verify that empirical boundaries are strictly preserved, and ensure that unresolved methodological tensions are transparently documented rather than concealed.

---

## Reviewer A — Mathematical Statistics

### Reviewer Focus: Multiplicity, Pseudo-Replication, and Inferential Precision

1. **Multiplicity and Multi-Testing in Branch Reporting**:
   - *Reviewer Critique*: The paper reports that "10 of 20 contaminated branches achieved nominal significance at $\alpha = 0.05$." However, evaluating 20 non-independent hypothesis tests without multiplicity adjustment (such as family-wise error rate control via Bonferroni/Holm, or false discovery rate control via Benjamini-Hochberg) invites false discovery. If a standard Bonferroni threshold ($\alpha / 20 = 0.0025$) were applied, only branches with $p \le 0.0025$ (such as Seed 87 at $D=0.75$ and $D=1.00$, and Seed 2024 at $D=1.00$) would survive.
   - *Manuscript Defense & Alignment*: The manuscript explicitly emphasizes this exact limitation in Section 1.5, Section 5.5, Section 6.2, and Section 8.1. The manuscript repeatedly clarifies that these are *nominal* branch-level tests and states explicitly: *"Because Protocol v1.2.4 omitted an omnibus aggregation rule across branches, a formal global confirmatory rejection is not claimed."* No post-hoc multiplicity procedure was invented to claim omnibus rejection.
   - *Audit Status*: **PASS (Strict inferential boundary preserved)**.

2. **Pseudo-Replication Concerns (Figure 3 Panel A)**:
   - *Reviewer Critique*: Pooling 32 events across 5 seeds creates 160 points per dose in Figure 3A. If these 160 points were treated as independent observations in a $t$-test or regression, it would constitute severe pseudo-replication.
   - *Manuscript Defense & Alignment*: In Section 5.3, Section 6.4, Section 8.10, and Figure 3's caption, the manuscript explicitly notes: *"Each dose panel aggregates 160 branch-events (32 OOS events $\times$ 5 seeds) that are not statistically independent; this panel is presented for post-hoc descriptive visualization only. The preregistered inferential unit remains the independent FOMC calendar meeting."*
   - *Audit Status*: **PASS (Descriptive status unambiguously declared)**.

3. **Finite Monte-Carlo Resolution for $p = 0.0000$**:
   - *Reviewer Critique*: Reporting $p = 0.0000$ in Seed 87 at $D=1.00$ implies literal impossibility under the null, which is impossible with $B = 2,000$ permutations.
   - *Manuscript Defense & Alignment*: The manuscript explicitly documents the finite Monte-Carlo resolution caveat in Section 5.4, Table S1 Note 2, and the accuracy audit: $0$ exceedances out of $2,000$ yields $p_{\mathrm{plus\_one}} = 1/2001 \approx 0.00050$.
   - *Audit Status*: **PASS (Mathematical precision preserved)**.

---

## Reviewer B — Causal Inference & Experimental Design

### Reviewer Focus: Domain Adaptation Confounds, Temporal Chronology, and Baseline Provenance

1. **Domain Adaptation vs. Temporal Leakage Confound**:
   - *Reviewer Critique*: The post-cutoff text consists of central-bank communications from 2020–early 2023 (specifically 2020-01-29 through 2023-01-04). During this period, the Federal Reserve adopted unprecedented rhetoric regarding pandemic emergency lending and zero-rate policies. Could the linear probe's improved performance reflect adaptation to specific modern rhetorical patterns rather than temporal leakage of policy direction?
   - *Manuscript Defense & Alignment*: The twin-model design pairs contaminated models with an active clean twin receiving the exact same token budget and gradient steps on contemporary pre-cutoff FOMC documents, equalizing broad central-bank domain adaptation. However, Section 7.6 and Section 8.6 explicitly concede that this design *cannot eliminate regime-specific vocabulary shifts or topic distributions introduced by post-2020 documents*. The manuscript refrains from overclaiming that domain adaptation was 100% eliminated.
   - *Audit Status*: **PASS (Causal boundary accurately stated)**.

2. **Temporal Chronology Precision ("Future" vs. "Post-Cutoff")**:
   - *Reviewer Critique*: A naive reader might believe that the linear probe was evaluated on FOMC meetings occurring in 2020–early 2023. If so, evaluating on the contamination text itself would be trivial data leakage.
   - *Manuscript Defense & Alignment*: The manuscript repeatedly clarifies in Section 3.1, Section 5.1, and Table S1 that the 32 out-of-sample evaluation events occurred between **January 2016 and December 2019**, strictly *prior* to the global cutoff `2019-12-31T23:59:59Z`. "Future" refers strictly to the *next policy decision relative to each historical statement* ($\Delta\text{Rate}_{t+1}$).
   - *Audit Status*: **PASS (Chronology strictly disambiguated)**.

3. **Base Model Temporal Provenance**:
   - *Reviewer Critique*: Can the authors prove that the base checkpoint `ProsusAI/finbert` was completely free of temporal leakage prior to the experiment?
   - *Manuscript Defense & Alignment*: Section 8.11 explicitly addresses this issue: the base model was published in 2019, but its pretraining corpora cannot be cryptographically verified to the second. The study bounds its causal claim to the **incremental treatment effect** of controlled post-cutoff continued pretraining relative to a shared initialization, rather than asserting zero baseline contamination in the base weights.
   - *Audit Status*: **PASS (Transparent baseline scope)**.

---

## Reviewer C — NLP & Representation Learning

### Reviewer Focus: Probing Validity, Representation Geometry, and Mechanistic Claims

1. **Overinterpretation of Linear Probes ("Extractability" vs. "Knowledge")**:
   - *Reviewer Critique*: Does probe success prove that FinBERT "understands" or "memorized" future policy decisions?
   - *Manuscript Defense & Alignment*: Section 2.4, Section 7.1, and Section 8.10 cite Belinkov (2022) and Hewitt & Liang (2019) to emphasize that linear probes measure the geometric extractability of features along linear subspaces, not factual memorization or causal reasoning. The manuscript explicitly avoids claims such as "FinBERT memorized future statements."
   - *Audit Status*: **PASS (Strict epistemic bounds on probing)**.

2. **Probe Capacity and Regularization**:
   - *Reviewer Critique*: Unregularized probes can learn tasks on random representations.
   - *Manuscript Defense & Alignment*: Section 5.3 documents that a regularized linear Ridge regression probe ($\alpha = 1.0$) was fitted under 4-fold grouped temporal cross-validation, where anchor paragraphs from the same meeting were never split across folds.
   - *Audit Status*: **PASS (Standard probe regularization verified)**.

3. **Mechanistic Hypotheses for Non-Monotonicity**:
   - *Reviewer Critique*: In Section 7.3, several mechanisms are proposed for why $D=1.00$ drops relative to $D=0.75$ (catastrophic forgetting, representation drift, loss landscapes). Are these proven empirical facts?
   - *Manuscript Defense & Alignment*: Section 7.3 concludes with an explicit disclaimer: *"These mechanisms represent exploratory hypotheses that warrant further structural investigation; they are not demonstrated empirical facts."*
   - *Audit Status*: **PASS (Hypotheses clearly demarcated as exploratory)**.

---

## Reviewer D — Empirical Quantitative Finance

### Reviewer Focus: Look-Ahead vs. Parametric Leakage, Economic Endpoints, and "False Alpha"

1. **Practical Significance vs. Statistical Significance**:
   - *Reviewer Critique*: Does positive representational leakage ($L_{\mathrm{repr}} \approx 0.005$) translate into a tradable market advantage?
   - *Manuscript Defense & Alignment*: The entire thesis of the paper argues that representation leakage did not translate into reliable positive gains at the market tier. Section 6.7 and Section 7.5 explicitly document that downstream Information Coefficients against 2-year Treasury yields and SPY equities did not support the preregistered positive economic alternative. The paper directly rejects claims of "false alpha."
   - *Audit Status*: **PASS (Strong anti-overclaiming posture)**.

2. **Seed 42 Negative Treasury Shift Interpretation**:
   - *Reviewer Critique*: In Seed 42 at $D=0.25$ and $D=0.50$, bootstrap tests yielded $p \le 0.009$ with negative $\Delta\mathrm{IC}$. If a researcher reported this as "statistically significant leakage," it would be dishonest because the direction is negative.
   - *Manuscript Defense & Alignment*: Section 6.7 and Table S1 Note 3 explicitly state that because the preregistered economic hypothesis is directional ($H_1^{\mathrm{econ}}: \Delta\mathrm{IC} > 0$), negative shifts represent evidence in the opposite direction (predictive degradation), and do not support economic leakage.
   - *Audit Status*: **PASS (Directional integrity verified)**.

3. **Distinction Between Look-Ahead Bias and Parametric Leakage**:
   - *Reviewer Critique*: Traditional quantitative researchers might dismiss parametric leakage as merely ordinary look-ahead bias with a fancy name.
   - *Manuscript Defense & Alignment*: Section 1.1–1.3 and Section 3.1 clearly formulate the distinction: look-ahead bias involves future data entering the runtime feature pipeline $X_t$, whereas parametric leakage occurs when the runtime pipeline is 100% clean but future information is embedded in model weights $\theta$.
   - *Audit Status*: **PASS (Conceptual contribution well-delineated)**.

---

## Factual Consistency Corrections After Full-Paper Assembly

During the factual consistency and reference verification audit (Phase 6 patch and final micro-closure), several transcription errors from earlier drafting drafts were identified and corrected to ensure 100% parity with frozen execution truth:

1. **MLM Treatment Stream Construction vs. Optimizer Execution**:
   - *Prior Transcription Error*: The initial assembly implied that MLM execution processed $16 \times 160 \times 100 = 256,000$ tokens with a sequence length of 160.
   - *Factual Correction*: In Protocol v1.2.4, the treatment stream consists of 500 packed blocks of 512 tokens (yielding the constructed 256,000-token treatment budget), whereas the MLM optimizer executed 100 gradient steps with a batch size of 16. The aggregate constructed treatment budget across 25 branches is 6.4M tokens ($25 \times 256,000$), which is distinct from cumulative optimizer token throughput.
2. **Downstream Supervised Fine-Tuning Recipe**:
   - *Prior Transcription Error*: The initial assembly stated that downstream fine-tuning was 5 epochs with a frozen encoder and trained head, with early drafts also attempting an unverified arithmetic derivation of optimizer steps ($1729 / 16 \times 3 \approx 324$).
   - *Factual Correction*: Frozen configuration `downstream_training` specifies 3 epochs, batch size 16, learning rate $2\times 10^{-5}$, weight decay 0.01, linear scheduler with warmup ratio 0.1, and max sequence length 128 tokens (`max_steps: 500`). Production code executes full-model fine-tuning (`model.train()`, `AdamW(model.parameters(), ...)`) under paired within-seed sample ordering (`paired_within_seed`). Unpersisted runtime step estimates have been excised from the manuscript to maintain strict fidelity to the execution configuration.
3. **Representation Probing vs. Fine-Tuning Distinction**:
   - *Factual Clarification*: Downstream stance fine-tuning updates full model parameters. Subsequently, representations are extracted from anchor paragraphs and frozen inputs are provided to linear Ridge probes ($\alpha = 1.0$) under 4-fold temporal cross-validation.
4. **MLM Masking and Optimization Schedule**:
   - *Prior Transcription Error*: Supplement S2 previously referenced generic "dynamic random masking" and stated that MLM used linear warmup 10 steps and linear decay.
   - *Factual Correction*: MLM masking uses a 15% mask probability implemented via a pre-generated boolean mask schedule that is deterministically fixed across dose branches within each seed (`mask_schedule_hash`), though distinct across seeds. The MLM optimizer used scheduler `none` and warmup ratio `0.0` (constant learning rate $5\times 10^{-5}$). Linear warmup was used strictly in the downstream fine-tuning stage.
5. **Economic Bootstrap Methodology**:
   - *Prior Transcription Error*: Referenced "clustered bootstrap by event".
   - *Factual Correction*: The production evaluator implements a 1,000-draw event-level stationary block bootstrap.
6. **Bibliographic and Citation Integrity**:
   - *Factual Correction*: Re-audited all 26 citations against primary venues. Corrected `jang2022temporal` to its full 8-author list from ICLR 2022 proceedings. Replaced the conflated `luu2022timeqa` with Kelvin Luu et al.'s actual temporal misalignment study from NAACL 2022 (`luu2022temporal`). Verified 2026 references on arXiv and conference proceedings. Conservatively classified `zhang2026all` as an arXiv preprint pending formal publication in the official ACL Anthology for Findings of EMNLP 2026 (marked PARTIAL in bibliographic metadata audit).

---

## Summary of Unresolved Tensions & Transparent Disclosures

The following scientific tensions remain unresolved by the current empirical record and are documented as open problems for future research:
1. **Asymptotic Optimization Landscape**: Five seeds establish that optimization variance exists, but do not map the complete distribution of parameter trajectories.
2. **Mechanistic Threshold for Non-Monotonicity**: The exact transition point between constructive representation alignment ($D \le 0.75$) and destructive representation drift ($D = 1.00$) remains unknown.
3. **Generalization to Autoregressive LLMs**: The empirical evidence applies strictly to encoder-only architectures (`ProsusAI/finbert`). Whether generative decoder models exhibit similar representation decoupling remains an open empirical question.
