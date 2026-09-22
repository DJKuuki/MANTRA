# Parametric Temporal Leakage in Financial Language Models: Probing Latent Representations Under Causally Symmetric Pretraining

**Authors**: Anonymous Authors  
**Date**: September 2026  
**Repository**: `DJKuuki/MANTRA`  
**Protocol Specification**: Version 1.2.4  

---

## Abstract

Financial machine learning strictly requires temporal integrity: models evaluated on historical data must not exploit information that postdates the decision timestamp. While conventional backtesting safeguards focus on runtime feature pipelines, pretrained language models introduce *parametric temporal leakage*, where future information is embedded directly within model parameters during pretraining or continued pretraining. In this work, we formalize parametric temporal leakage across four distinct analytical tiers—Future Exposure, Latent Representation Decodability, Behavioral Sensitivity, and Downstream Economic Effect—and introduce a causally symmetric twin-model methodology that equalizes parameter scale, compute volume, and text domain while isolating temporal exposure. Benchmarking the `ProsusAI/finbert` encoder across five optimization seeds and four post-cutoff contamination doses ($D \in \{0.25, 0.50, 0.75, 1.00\}$) against compute-matched clean twins on Federal Open Market Committee communications, we find that post-cutoff continued pretraining produces a predominantly positive representational leakage signal ($L_{\mathrm{repr}} > 0$ in 18 of 20 contaminated branches; 10 of 20 achieving nominal branch-level significance, $p < 0.05$, under one-sided right-tailed paired event-level sign-flip permutation tests). However, because the protocol omitted an omnibus aggregation rule across branches, a formal global confirmatory rejection is not claimed. Furthermore, representational leakage exhibits non-monotonic dose dynamics, substantial seed heterogeneity, and was not accompanied by robust improvements in discrete monetary policy classification, behavioral masking sensitivity, or downstream market predictability (Treasury and equity Information Coefficients do not support the positive economic alternative). These findings suggest that controlled post-cutoff exposure can produce detectable representation-level temporal leakage signals without corresponding robust behavioral or economic effects, highlighting the need for layered auditing frameworks.

**Keywords**: Parametric Temporal Leakage, Financial Language Models, Representation Probing, Causal Twin Design, Look-Ahead Bias, Central Bank Communication.

---

## 1. Introduction

### 1.1 Temporal Evaluation in Financial Machine Learning
The foundational premise of quantitative finance is temporal causality: an investment decision, predictive stance, or algorithmic order at time $t$ can utilize only the information filtration $\mathcal{F}_t$ available strictly prior to $t$. In traditional statistical learning and financial econometrics, violating this condition produces *look-ahead bias*—an artificial inflation of backtest performance that collapses when deployed out-of-sample \citep{kaufman2012leakage, bailey2014pseudo, lopezdeprado2018advances}.

To guard against look-ahead bias, financial engineering protocols have developed rigorous operational safeguards: Point-in-Time (PIT) database architectures, purged and combinatorial cross-validation schemes \citep{lopezdeprado2018advances}, timestamp auditing, and strict firewalls between historical feature pipelines and subsequent market returns. These measures are designed around an implicit structural assumption: *data leakage occurs externally through the input or target pipeline*. If the input features $X_t$ and labels $Y_t$ are drawn strictly from historical records preceding time $t$, the resulting model is presumed temporally uncontaminated.

### 1.2 The Foundation-Model Challenge: Parametric Temporal Leakage
The widespread adoption of foundation models and self-supervised pretrained language encoders—such as BERT \citep{devlin2019bert} and domain-specific adaptations like FinBERT \citep{araci2019finbert, huang2023finbert}—fundamentally breaks this assumption. Contemporary natural language processing models acquire their knowledge base during self-supervised pretraining on massive web-scale or domain corpora spanning multi-year horizons.

When a practitioner deploys a pretrained language model in a historical backtest (for instance, scoring central-bank communications from 2017 to forecast policy moves), the runtime inputs may appear completely clean: the text fed to the tokenizer was indeed published in 2017. However, if the underlying checkpoint underwent pretraining or continued adaptation on corpora extending beyond 2017, the neural network parameters $\theta$ already encode factual tokens, macroeconomic outcomes, and rhetorical shifts that belong strictly to the future ($\mathcal{F}_{>t}$). We term this phenomenon **parametric temporal leakage**: the direct encoding of future temporal information within model weights, operating independently of runtime input hygiene.

### 1.3 Why Prompting and Behavioral Instructions Cannot Solve It
A common heuristic in generative and instruction-tuned financial workflows is to impose temporal constraints via prompting—for example, instructing the model: *"Pretend today is December 31, 2018; rely only on information known before this date."* While such instructions can suppress overt temporal anachronisms in conversational outputs, they cannot establish parametric temporal integrity \citep{benhenda2026lookahead, zhang2026all}.

Neural weights operate as dense, non-linear associative memories \citep{carlini2021extracting, carlini2023quantifying}. Exposure to future documents alters gradient trajectories, hidden-layer activations, and latent manifold geometries. Prompting alters only the conditional inference trajectory within fixed parameter weights; it cannot excise the latent geometric distortions or factual associations burned into the network during pretraining. When evaluated by sensitive downstream readouts or linear decoders, parametric leakage can persist beneath surface-level compliance \citep{geirhos2020shortcut, xue2026temporal}.

### 1.4 The Research Gap
Despite increasing recognition of data contamination in general natural language processing benchmarks \citep{sainz2023nlp, golchin2024time, oren2024proving}, the literature lacks a rigorous, causally controlled experimental framework for isolating and measuring parametric temporal leakage in financial NLP. Existing audits frequently exhibit severe methodological vulnerabilities:
1. They compare modern large models against historical smaller baselines, conflating temporal exposure with parameter scale, architecture improvements, and compute volume.
2. They treat temporal leakage as a binary scalar ("leaked" vs. "clean"), ignoring how future information propagates across internal representations, task classifications, and financial market endpoints.
3. They evaluate leakage using random sample splits, which introduces cross-document pseudo-replication rather than preserving chronological event units.

### 1.5 Contributions
This paper addresses this gap through an audited confirmatory study evaluating parametric temporal leakage in financial language models. Our core contributions are:
1. **Parametric Temporal Leakage Formalization**: We formally define parametric temporal leakage as distinct from runtime look-ahead bias, establishing its properties under fixed parameter architectures.
2. **Causally Symmetric Twin Design**: We introduce an experimental design that holds parameter count, encoder architecture, tokenizer revision, total token budget (256,000 tokens per branch), optimization steps (100 MLM steps), masking schedules, and downstream transfer architectures strictly invariant, isolating the incremental effect of varying the temporal composition of the continued-pretraining treatment corpus under matched compute.
3. **The Layered Leakage Framework**: We evaluate leakage across four distinct tiers: Treatment Exposure $\to$ Latent Representation Decodability ($L_{\mathrm{repr}}$) $\to$ Downstream Behavioral Manifestation ($L_{\mathrm{behavior}}$) $\to$ Downstream Economic Effect ($E_L$), while separating temporal integrity from Task Competence ($C$) and Temporal Robustness ($R_T$).
4. **Event-Level Chronological Inference**: We structure all inferential hypothesis tests around independent Federal Open Market Committee (FOMC) calendar events under temporal cross-validation, avoiding token-level and fold-level pseudo-replication.
5. **Empirical Characterization of Latent Leakage**: Across 25 experimental branches on Federal Reserve communications, we find that post-cutoff exposure produces a substantial and predominantly positive representational leakage signal (18/20 branches positive, 10/20 nominally significant), but observe that no reliable positive downstream effect was detected across discrete policy classifications, behavioral masking sensitivities, or downstream market predictive power (the downstream endpoints did not support the preregistered positive alternative).
6. **Non-Monotonicity and Optimization Heterogeneity**: We reveal that representational leakage does not obey a naive monotonic dose-response curve, and demonstrate that stochastic optimization dynamics across random seeds introduce substantial variation in leakage susceptibility.

---

## 2. Related Work

### 2.1 Look-Ahead Leakage in Financial Machine Learning
The danger of data leakage in data mining and machine learning has long been recognized \citep{kaufman2012leakage}. In empirical asset pricing and quantitative finance, look-ahead bias and backtest overfitting represent major sources of false discovery \citep{bailey2014pseudo}. \citet{lopezdeprado2018advances} formalized purged cross-validation to prevent information bleeding across overlapping financial return windows. More recently, \citet{xue2026temporal} audited financial news classification pipelines across 16 architectures, finding that random sample splits inflate Matthew's Correlation Coefficients (MCC) by $1.1\times$ to $6.5\times$ compared to strictly chronological splits due to temporal leakage.

### 2.2 Foundation-Model Data Contamination and Memorization
As machine learning has transitioned to large-scale self-supervised pretraining, benchmark contamination has emerged as an acute crisis for evaluation validity \citep{sainz2023nlp}. \citet{carlini2021extracting} and \citet{carlini2023quantifying} demonstrated that language models memorize substantial portions of their training data, allowing exact token extraction through targeted prompting. \citet{oren2024proving} and \citet{golchin2024time} developed statistical tests to detect test-set contamination in black-box models. Furthermore, \citet{geirhos2020shortcut} showed that deep neural networks systematically exploit unintended statistical shortcuts rather than learning intended semantic abstractions, a phenomenon directly applicable to language encoders trained on future texts containing forward-looking cues.

### 2.3 Temporal Knowledge and Chronological Generalization in Language Models
Language models trained on static corpora rapidly suffer from temporal degradation as real-world facts evolve. \citet{dhingra2022time} investigated time-aware language models, demonstrating that standard models struggle to resolve temporal scopes without explicit time-indexing. \citet{jang2022temporal} studied continual knowledge learning, showing that updating models on new temporal streams frequently induces catastrophic forgetting of past knowledge. \citet{luu2022temporal} evaluated temporal misalignment across multi-year NLP benchmarks, demonstrating that model performance degrades significantly when evaluation postdates training.

### 2.4 Representation Probing
To understand the internal feature spaces of deep neural networks, \citet{alain2017understanding} introduced linear classifier probes—training linear models on frozen intermediate layer activations to quantify how easily target concepts can be read out. \citet{belinkov2022probing} surveyed probing methodologies, emphasizing that probe accuracy measures the linear extractability of information rather than proving that the base network actively utilizes those features for its primary task. \citet{hewitt2019designing} cautioned that probes with excessive capacity can learn the target task independently, establishing the necessity of constrained, regularized linear probes paired with rigorous baseline controls.

### 2.5 Financial NLP and Central Bank Communication
Natural language processing has become central to central-bank communication analysis \citep{hansen2016shocking}. Domain-specific pretrained language models, most notably FinBERT \citep{araci2019finbert, huang2023finbert}, have significantly improved sentiment extraction from financial reports, earnings calls, and macroeconomic releases. \citet{shah2023trillion} introduced the *Trillion Dollar Words* dataset, establishing an annotated corpus of Federal Open Market Committee (FOMC) statements and minutes categorized into hawkish, dovish, and neutral stances. Recently, \citet{tang2026mind} proposed Delta-Consistent Scoring (DCS) to decode policy shifts between consecutive FOMC meetings using frozen language representations.

### 2.6 Backtesting and Evaluation with Large Language Models
Recent work has explored using LLMs for direct market forecasting and automated trading strategies \citep{lopezlira2023chatgpt, kim2024financial}. However, evaluate-in-sample risks remain pervasive. \citet{benhenda2026lookahead} introduced *Look-Ahead-Bench*, documenting substantial alpha decay when point-in-time constraints are enforced on generative financial models. \citet{zhang2026all} developed *TimeSPEC* and the *Shapley-DCLR* metric, showing that language models frequently generate claims based on post-cutoff knowledge during financial reasoning chains. Our work departs from these studies by moving beyond generative decoders to establish a controlled, causally symmetric experimental framework targeting latent representation geometry in financial encoders.

---

## 3. Problem Formulation

### 3.1 Temporal Boundaries and Formal Definitions
Let $\mathcal{D}$ denote a text corpus where each document $x \in \mathcal{D}$ is associated with a verified publication timestamp $\tau(x) \in \mathbb{R}$. We define a global temporal cutoff $t_c$. The historical information filtration is partitioned into:
$$\mathcal{D}_{\le t_c} = \{x \in \mathcal{D} \mid \tau(x) \le t_c\}, \quad \mathcal{D}_{> t_c} = \{x \in \mathcal{D} \mid \tau(x) > t_c\}$$

Let $M_\theta$ denote a language encoder parameterized by weights $\theta \in \mathbb{R}^P$. We distinguish two fundamental forms of temporal leakage:
1. **External Look-Ahead Leakage**: Occurs when a model parameterized by clean weights $\theta$ is provided runtime input features $x$ or target labels $y$ such that $\tau(x) > t$ or $\tau(y) > t$ at decision time $t \le t_c$.
2. **Parametric Temporal Leakage**: Occurs when the parameters $\theta$ have been optimized over a corpus containing post-cutoff documents:
   $$\theta = \arg\min_\phi \mathcal{L}\left(\phi; \mathcal{D}_{\mathrm{train}}\right), \quad \text{where } \mathcal{D}_{\mathrm{train}} \cap \mathcal{D}_{> t_c} \neq \emptyset$$
   even when the runtime evaluation input $x_{\mathrm{test}}$ satisfies $\tau(x_{\mathrm{test}}) \le t_c$.

### 3.2 The Causally Symmetric Twin Formulation
To isolate parametric temporal leakage from confounding factors such as compute volume and domain adaptation, we define a controlled treatment space. For a given random optimization seed $s$, we construct:
- **Clean Twin ($M_C$ or $M(s, 0)$)**: Ingests an exact constructed treatment-stream budget of $T = 256,000$ tokens (500 blocks of 512 tokens) drawn exclusively from contemporary pre-cutoff documents $\mathcal{D}_{\mathrm{clean}} \subset \mathcal{D}_{\le t_c}$, trained for $K = 100$ gradient steps with batch size 16.
- **Contaminated Model ($M_L(D)$ or $M(s, D)$)**: Ingests an identical budget of $T = 256,000$ tokens over $K = 100$ gradient steps under the exact same optimization schedule, but where a fraction $D \in \{0.25, 0.50, 0.75, 1.00\}$ of the treatment tokens are drawn from post-cutoff documents $\mathcal{D}_{\mathrm{leak}} \subset \mathcal{D}_{> t_c}$, substituted for pre-cutoff sham tokens:
  $$T_{\mathrm{leak}} = D \cdot T, \quad T_{\mathrm{clean}} = (1 - D) \cdot T$$

### 3.3 The Layered Metric Taxonomy
Rather than reducing temporal integrity to a single composite score, we define orthogonal evaluative dimensions across four distinct layers:

```
[Stage 1: Treatment Exposure]   Dose D in [0.25, 1.00], Constructed Budget T = 256k tokens
              │
              ▼
[Stage 2: Representation Shift] Continuous Decodability: L_repr
              │
              ▼ (Decoupling)
[Stage 3: Behavioral Response]  Binary Classification F1 & Masking Sensitivity: L_behavior
              │
              ▼ (Decoupling)
[Stage 4: Economic Consequence] Market Information Coefficient Deltas: E_L (2Y & SPY)
```

1. **Continuous Representational Leakage ($L_{\mathrm{repr}}$)**: Measures whether latent representations extracted from $M_L(D)$ permit superior linear decodability of future monetary policy rate changes ($\Delta\text{Rate}_{t+1}$) compared to representations from $M_C$ on out-of-sample pre-cutoff events $e \in \mathrm{OOS}$:
   $$d_e = \left| y_e - \hat{y}_{C,e} \right| - \left| y_e - \hat{y}_{L,e} \right|$$
   $$L_{\mathrm{repr}}(s, D) = \frac{1}{N_{\mathrm{OOS}}} \sum_{e=1}^{N_{\mathrm{OOS}}} d_e$$
   where $y_e$ is the continuous target policy change, and $\hat{y}_{C,e}, \hat{y}_{L,e}$ are out-of-sample predictions from regularized linear probes fitted on frozen sentence representations. A positive value ($L_{\mathrm{repr}} > 0$) denotes a paired absolute-error improvement attributable to future token exposure.

2. **Behavioral Masking Sensitivity ($L_{\mathrm{behavior}}$)**: Measures whether post-cutoff exposure distorts the model's reliance on prompt-level named entities or dates. For an input sequence $x$ and masking operator $\mathcal{T}(x)$, raw sensitivity is defined via Jensen-Shannon divergence:
   $$S_{\mathrm{mask}}(M) = \mathbb{E}_{x \sim \mathcal{D}_{\mathrm{eval}}} \left[ D_{\mathrm{JS}}\left( P_M(y \mid x) \parallel P_M(y \mid \mathcal{T}(x)) \right) \right]$$
   Behavioral leakage is defined as the clean/contaminated differential:
   $$L_{\mathrm{behavior}}(s, D) = S_{\mathrm{mask}}(M_L(s, D)) - S_{\mathrm{mask}}(M_C(s))$$

3. **Downstream Economic Effect ($E_L$)**: Measures whether leaked representations translate into predictive improvements in financial markets. For an asset return $R_{t+\Delta t}$ (e.g., 2-year Treasury yield change or SPY equity return), we compute the Information Coefficient delta:
   $$E_L = \Delta\mathrm{IC} = \mathrm{IC}(M_L) - \mathrm{IC}(M_C) = \mathrm{Spearman}(\hat{z}_L, R) - \mathrm{Spearman}(\hat{z}_C, R)$$
   where $\hat{z}$ is the continuous projected policy stance. The preregistered alternative hypothesis is directional: $H_1^{\mathrm{econ}}: \Delta\mathrm{IC} > 0$.

4. **Task Competence ($C$) and Temporal Robustness ($R_T$)**: Represent orthogonal control dimensions. Competence $C$ benchmarks baseline task skill on un-leaked historical splits. Robustness $R_T$ measures performance decay under non-stationary regime shifts without contamination. In this confirmatory study, $C$ and $R_T$ are formally cataloged as un-evaluated control splits to preserve statistical degrees of freedom for the causal contrast.

### 3.4 The Null-Model Insight: Low Leakage $\neq$ High Model Quality
A critical conceptual principle of our framework formalizes the distinction between temporal leakage and model utility. Consider a trivial "null" model $M_{\mathrm{null}}$ that maps all input texts to a constant zero vector $\mathbf{0} \in \mathbb{R}^d$. Under our evaluation protocol, such a model would achieve identical clean and contaminated errors ($d_e \equiv 0$), resulting in $L_{\mathrm{repr}} \equiv 0$ and $L_{\mathrm{behavior}} \equiv 0$ across all doses. The model displays perfect "temporal cleanliness."

However, $M_{\mathrm{null}}$ possesses zero predictive competence ($C = 0$). Conversely, a highly competent model might achieve superior task accuracy while remaining susceptible to latent contamination. Therefore:
$$\text{Low Measured Leakage } \not\implies \text{ Superior Model}$$
Temporal integrity ($L$) and predictive competence ($C$) represent orthogonal evaluative dimensions. A valid scientific audit must analyze both axes rather than collapsing them into a single score.

---

## 4. Methodology & Causally Symmetric Twin Design

### 4.1 Causal Invariants Across Branches
In typical machine learning comparisons, evaluating the impact of pretraining data is confounded by unequal training steps, divergent token volumes, differing optimizer hyperparameters, or shifted neural architectures. To eliminate these confounds, Protocol v1.2.4 enforces strict causal symmetry across all experimental branches:

| Experimental Dimension | Clean Baseline Twin ($D=0.00$) | Contaminated Twin ($D > 0$) | Causal Control Status |
| :--- | :--- | :--- | :---: |
| **Base Encoder Checkpoint** | `ProsusAI/finbert` | `ProsusAI/finbert` | Identical Initial Weights |
| **Model Architecture** | 12-layer, 768-dim BERT-base | 12-layer, 768-dim BERT-base | Invariant |
| **Tokenizer & Vocabulary** | BERT uncased WordPiece (30,522 tokens) | BERT uncased WordPiece (30,522 tokens) | Invariant |
| **Pretraining Objective** | Masked Language Modeling (MLM, 15% mask) | Masked Language Modeling (MLM, 15% mask) | Invariant |
| **Constructed Treatment Stream**| 500 packed blocks $\times$ 512 tokens = 256,000 tokens | 500 packed blocks $\times$ 512 tokens = 256,000 tokens | Matched Budget |
| **MLM Optimizer Execution** | 100 gradient steps (batch size 16) | 100 gradient steps (batch size 16) | Matched Optimizer Steps |
| **MLM Optimizer & Schedule** | AdamW ($\text{lr}=5\times 10^{-5}$, weight decay 0.01), scheduler `none`, warmup `0.0` | AdamW ($\text{lr}=5\times 10^{-5}$, weight decay 0.01), scheduler `none`, warmup `0.0` | Invariant |
| **Downstream Transfer Recipe** | Full-model fine-tuning (3 epochs, batch 16, lr $2\times 10^{-5}$, max seq len 128, linear scheduler, warmup 0.1) | Full-model fine-tuning (3 epochs, batch 16, lr $2\times 10^{-5}$, max seq len 128, linear schedule, warmup 0.1) | Invariant |
| **Downstream Head Init & Order**| Fresh shared linear head; paired sample order | Fresh shared linear head; paired sample order | Matched Initialization & Order |
| **Representation Probing Stage**| Linear Ridge probe ($\alpha=1.0$) on frozen representations | Linear Ridge probe ($\alpha=1.0$) on frozen representations | Identical Probing Architecture |
| **Hardware Environment** | Dedicated CUDA GPU | Dedicated CUDA GPU | Paired Symmetry Controls |

Because architecture, initialization, compute volume, and broad central-bank genre exposure are held fixed, the paired design isolates the incremental effect of changing the temporal composition of the treatment corpus. We emphasize that temporal composition remains partially entangled with post-cutoff regime, topic, and rhetorical distribution shifts.

### 4.2 Representation Extraction
Following continued pretraining, the branch-specific encoder is coupled to a freshly initialized 3-class linear classification head (`fresh_shared_within_seed`) and fine-tuned end-to-end on historical stance data ($\le 2018$ documents from the *Trillion Dollar Words* corpus \citep{shah2023trillion}) for 3 epochs with AdamW ($\text{lr} = 2\times 10^{-5}$, linear schedule, 10% warmup, max sequence length 128, capped at $\text{max\_steps}=500$) under paired batch ordering (`paired_within_seed`).

Subsequently, for any input text $x = (w_1, w_2, \dots, w_N)$ with binary attention mask $\mathbf{A} \in \{0, 1\}^N$, the fine-tuned model outputs hidden states $\mathbf{h}_i \in \mathbb{R}^{768}$ at the final transformer layer. We extract sentence-level representations using attention-mask-aware mean pooling:
$$\mathbf{z} = \frac{\sum_{i=1}^N \mathbf{A}_i \mathbf{h}_i}{\sum_{i=1}^N \mathbf{A}_i}$$
These extracted embeddings are treated as frozen inputs for downstream representation probing. This pooling avoids biasing representations toward the special classification token `[CLS]` and provides an unweighted geometric summary of token contextualization across the sequence.

---

## 5. Experimental Design

### 5.1 Dataset Chronology and Point-in-Time Separation
The empirical study is built around official Federal Open Market Committee (FOMC) communications, utilizing the *Trillion Dollar Words* corpus \citep{shah2023trillion} alongside primary Federal Reserve archives. The temporal timeline is strictly structured around the global cutoff $t_c =$ `2019-12-31T23:59:59Z`:

1. **Pre-Cutoff Sham Corpus ($\mathcal{D}_{\mathrm{clean}}$)**: Consists of 63 historical FOMC statements and minutes published between `2015-01-28T19:00:00Z` and `2019-12-11T19:00:00Z`. Ingested by clean twin models to match domain adaptation to central-bank English.
2. **Post-Cutoff Contamination Corpus ($\mathcal{D}_{\mathrm{leak}}$)**: Consists of 50 official Federal Reserve post-cutoff documents (statements, emergency policy releases, and minutes) spanning `2020-01-29T19:00:00Z` through `2023-01-04T19:00:00Z`. Enforces an absolute 48-day buffer between the end of pre-cutoff training and the start of contamination, with zero document overlap ($0/50$).
3. **Evaluation Events and Target Definition**: Out-of-sample evaluation is conducted across 40 historical FOMC calendar events spanning 2015 through 2019, with 32 events comprising the out-of-sample test partition under temporal cross-validation. For each anchor paragraph associated with an FOMC meeting at date $t_e \le t_c$, the target variable $y_e = \Delta\text{Rate}_{t+1}$ represents the **next scheduled policy rate change decided at meeting $t+1$**. We emphasize: *"future"* denotes the subsequent monetary policy decision relative to the historical statement, **not** an event occurring after the global 2019 cutoff.

### 5.2 Model and Branch Configuration
The experimental design evaluates 25 distinct branches on a CUDA-accelerated Linux environment:
- **Base Checkpoint**: `ProsusAI/finbert` (Hugging Face revision `4556d13015211d73dccd3fdd39d39232506f3e43`).
- **Optimization Seeds**: Five independent seeds: $s \in \{13, 42, 87, 123, 2024\}$.
- **Contamination Doses**: Five dose levels per seed: $D \in \{0.00, 0.25, 0.50, 0.75, 1.00\}$.
  - $D = 0.00$: Clean baseline twin (256,000 pre-cutoff tokens; 500 blocks of 512 tokens).
  - $D = 0.25$: 192,000 pre-cutoff tokens + 64,000 post-cutoff tokens.
  - $D = 0.50$: 128,000 pre-cutoff tokens + 128,000 post-cutoff tokens.
  - $D = 0.75$: 64,000 pre-cutoff tokens + 192,000 post-cutoff tokens.
  - $D = 1.00$: Full contamination (256,000 post-cutoff tokens).
- **Aggregate Treatment Budget**: The 25 branches contained an aggregate constructed treatment-stream budget of 6.4 million tokens ($25 \times 256,000$ tokens), evaluated under paired execution symmetry and reproducibility controls.

### 5.3 Probing Architecture and Cross-Validation
To evaluate continuous representational leakage ($L_{\mathrm{repr}}$), we train an $L_2$-regularized linear Ridge regression probe ($\alpha = 1.0$) on frozen anchor embeddings $\mathbf{z}_e \in \mathbb{R}^{768}$ to predict $\Delta\text{Rate}_{t+1}$. The evaluation follows a **4-fold grouped expanding-window temporal cross-validation**:
- All anchor paragraphs originating from the same FOMC meeting are grouped together to prevent cross-paragraph data leakage across folds.
- Folds are ordered chronologically (Fold 1: 2015–2016; Fold 2: 2017; Fold 3: 2018; Fold 4: 2019). The probe is trained strictly on earlier meetings and evaluated on out-of-sample future meetings ($N_{\mathrm{OOS}} = 32$ meetings across Folds 2–4).
- The inferential unit is the **independent FOMC calendar meeting**, not individual paragraphs or tokens.

### 5.4 Primary Statistical Hypothesis and Inferential Procedure
For each contaminated branch $(s, D)$, we test the directional hypothesis:
$$H_0^{\mathrm{repr}}: \mu_d \le 0 \quad \text{vs.} \quad H_1^{\mathrm{repr}}: \mu_d > 0$$
where $\mu_d = \mathbb{E}[d_e]$ is the expected paired absolute-error improvement across out-of-sample FOMC events.

Significance is evaluated via a **one-sided right-tailed paired event-level sign-flip permutation test** with $B = 2,000$ draws. For each permutation $b \in \{1, \dots, B\}$, random signs $r_{e,b} \in \{-1, +1\}$ are drawn independently with probability $0.5$ for each of the $N_{\mathrm{OOS}} = 32$ events, forming the permutation statistic:
$$\bar{d}_b = \frac{1}{N_{\mathrm{OOS}}} \sum_{e=1}^{N_{\mathrm{OOS}}} r_{e,b} d_e$$
The empirical $p$-value is computed as:
$$p = \frac{1}{B} \sum_{b=1}^B \mathbb{I}\left( \bar{d}_b \ge \bar{d}_{\mathrm{obs}} \right)$$

*Finite Monte-Carlo Resolution Sensitivity*: When zero permutation draws exceed the observed statistic ($k = 0$), the empirical output is recorded as raw $p = 0.0000$. Under standard finite-sample Monte-Carlo reporting sensitivity $\frac{k+1}{B+1}$, this corresponds to $p_{\mathrm{plus\_one}} = 1/2001 \approx 0.00050$. We report the exact frozen raw values alongside this sensitivity note.

### 5.5 Preregistration Scope and Global-Test Decision Boundary
Protocol v1.2.4 formally preregistered the individual branch-level tests ($\alpha = 0.05, B = 2,000$). However, the protocol specification omitted an explicit omnibus decision rule (such as a Fisher combination test, Stouffer z-score, or Bonferroni-Holm family-wise correction) for pooling the 20 contaminated contrasts into a single omnibus confirmatory test. 

To maintain strict scientific integrity, we do not invent an omnibus test post-hoc. We report branch-level outcomes nominally and descriptively, and **explicitly refrain from claiming a formal global confirmatory rejection of $H_0^{\mathrm{repr}}$**.

---

## 6. Results

### 6.1 Dose-Level Aggregate Findings
Table 1 presents across-seed aggregate metrics across contamination doses. These aggregates provide descriptive summaries of empirical tendencies across seeds.

```text
Table 1: Dose-Level Aggregate Empirical Results (Main Manuscript)
All metrics reflect post-hoc descriptive across-seed means (N=5 seeds per dose).
```

| Contamination Dose ($D$) | Mean $L_{\mathrm{repr}}$ | Median $L_{\mathrm{repr}}$ | Std $L_{\mathrm{repr}}$ | Nominal Sig. Frac. ($p < 0.05$) | Mean $\Delta\mathrm{Spearman}$ | Binary $\Delta\mathrm{Macro\text{-}F1}$ | Mean $L_{\mathrm{behavior}}$ | Mean $\Delta\mathrm{IC}_{2\mathrm{Y}}$ | Mean $\Delta\mathrm{IC}_{\mathrm{SPY}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00 (Clean Twin)** | $+0.00000$ | $+0.00000$ | $0.00000$ | 0 / 5 (0.0%) | $+0.0000$ | $+0.0000$ | $+0.00000$ | $+0.0000$ | $+0.0000$ |
| **0.25** | $+0.00368$ | $+0.00304$ | $0.00165$ | 2 / 5 (40.0%) | $+0.0466$ | $-0.0169$ | $-0.00005$ | $-0.0224$ | $+0.0089$ |
| **0.50** | $+0.00373$ | $+0.00528$ | $0.00285$ | 3 / 5 (60.0%) | $+0.1105$ | $-0.0030$ | $+0.00046$ | $-0.0182$ | $+0.0100$ |
| **0.75** | **+0.00507** | $+0.00385$ | $0.00312$ | **3 / 5 (60.0%)** | **+0.1043** | $+0.0273$ | $-0.00030$ | $-0.0197$ | $+0.0108$ |
| **1.00** | $+0.00291$ | $+0.00090$ | $0.00455$ | 2 / 5 (40.0%) | $+0.0649$ | $+0.0058$ | $+0.00000$ | $-0.0186$ | $-0.0149$ |

### 6.2 Representational Leakage ($L_{\mathrm{repr}}$)
Across the 20 contaminated experimental branches:
- **Directional Consistency**: 18 of the 20 branches (90.0%) exhibited positive point estimates ($L_{\mathrm{repr}} > 0$), indicating that exposure to future central-bank tokens reduced absolute prediction errors when decoding future monetary policy decisions.
- **Nominal Significance**: 10 of the 20 branches (50.0%) achieved nominal significance at $\alpha = 0.05$ ($p < 0.05$) under the one-sided paired event-level sign-flip permutation test.
- **Non-Monotonic Dose-Response Profile**: Mean $L_{\mathrm{repr}}$ increases from $+0.00368$ at $D = 0.25$ to an aggregate peak of $+0.00507$ at $D = 0.75$, but attenuates to $+0.00291$ at full contamination ($D = 1.00$) ([Figure 1](../research/figures/phase4b/figure1_l_repr_dose_response.png)).

### 6.3 Optimization Seed Heterogeneity
Disaggregating the 20 branches reveals substantial variation across random seeds ([Figure 2](../research/figures/phase4b/figure2_branch_significance_map.png) and Supplement Table S1):
1. **Seed 87 (High Susceptibility)**: Exhibited strong representational leakage across all doses, peaking at $D=0.75$ ($L_{\mathrm{repr}} = +0.00964, p = 0.0005, \Delta\mathrm{Spearman} = +0.2602$) and remaining nominally significant at $D=0.50$ ($p = 0.0060$) and $D=1.00$ (raw $p = 0.0000$, finite-MC sensitivity $p_{\mathrm{plus\_one}} = 1/2001 \approx 0.00050$).
2. **Seed 2024 (Monotonic Response)**: Demonstrated steady increases in decodability across doses, rising from $L_{\mathrm{repr}} = +0.00247$ ($p = 0.1290$) at $D=0.25$ to $+0.00889$ ($p = 0.0005$) at $D=1.00$, with 3 of 4 doses reaching nominal significance.
3. **Seed 123 (Non-Monotonic Inversion)**: Displayed pronounced instability. While $D=0.25$ showed significant leakage ($+0.00640, p = 0.0075$) and $D=0.75$ reached significance ($+0.00385, p = 0.0395$), intermediate and full doses dropped below the clean baseline ($D=0.50: L_{\mathrm{repr}} = -0.00024, p = 0.5400$; $D=1.00: -0.00218, p = 0.8955$).
4. **Seeds 13 and 42 (Isolated Significance)**: Showed moderate effects that achieved nominal significance only at isolated doses (Seed 13 at $D=0.50: +0.00545, p = 0.0255$; Seed 42 at $D=0.25: +0.00402, p = 0.0090$).

This variance demonstrates that temporal leakage susceptibility is modulated by stochastic optimization dynamics during pretraining rather than being a deterministic function of model capacity alone.

### 6.4 Event-Level Effect Distribution
To assess whether representational gains were concentrated in isolated economic shocks, event-level paired absolute-error deltas $d_e$ were analyzed across the 32 out-of-sample meetings ([Figure 3](../research/figures/phase4b/figure3_event_level_deltas.png)):
- In pooled branch-event distributions (160 branch-events per dose; post-hoc descriptive check), positive absolute-error reductions predominate: 61.3% of observations at $D=0.25$, 58.8% at $D=0.50$, 65.0% at $D=0.75$, and 62.5% at $D=1.00$ exhibit $d_e > 0$ ([Figure 3A](../research/figures/phase4b/figure3_event_level_deltas.png)).
- A post-hoc descriptive breadth check found that positive paired absolute-error reductions were observed in the median across contaminated branches for 23 of 32 (71.9%) out-of-sample temporal-CV meetings spanning 2016–2019 ([Figure 3B](../research/figures/phase4b/figure3_event_level_deltas.png)). This indicates that the directional advantage was not concentrated in a single meeting date, though this check remains descriptive.

### 6.5 Binary Policy Endpoint ($H_1^{\mathrm{binary}}$): Not Supported
The co-primary discrete classification endpoint (`next_scheduled_change_vs_hold`) evaluated whether representational shifts improved discrete rate change detection under standard linear heads:
- Across all 20 contaminated branches, the binary co-primary was **not supported**. All branch-level permutation tests yielded $p > 0.05$ (minimum $p = 0.2510$ in Seed 87 at $D=0.75$).
- Mean across-seed $\Delta\text{Macro-F1}$ hovered near zero across all doses ($-0.0169$ at $D=0.25$; $-0.0030$ at $D=0.50$; $+0.0273$ at $D=0.75$; $+0.0058$ at $D=1.00$).

### 6.6 Behavioral Masking Sensitivity ($L_{\mathrm{behavior}}$): Descriptive & Negligible
Behavioral masking sensitivity differentials remained negligible across all branches:
- Point estimates were on the order of $10^{-4}$ (across-seed means: $-0.00005$ at $D=0.25$, $+0.00046$ at $D=0.50$, $-0.00030$ at $D=0.75$, and $+0.00000$ at $D=1.00$).
- Future exposure did not induce anomalous reliance on masked prompt entities or dates, confirming that surface attribution remained unaltered.

### 6.7 Downstream Economic Market Endpoints ($E_L$): Not Supported
Model stance projections were mapped onto 2-year Treasury yield changes and SPY equity returns across 40 FOMC events (evaluated via 1,000-draw event-level stationary block bootstrap):
- **2-Year Treasury Yields (Primary Economic Endpoint)**: The preregistered positive alternative hypothesis ($H_1^{\mathrm{econ}}: \Delta\mathrm{IC}_{2\mathrm{Y}} > 0$) was **not supported**. Most 95% bootstrap confidence intervals included zero. Across-seed mean $\Delta\mathrm{IC}_{2\mathrm{Y}}$ was negative at all doses ($-0.0224$ at $D=0.25$, $-0.0182$ at $D=0.50$, $-0.0197$ at $D=0.75$, and $-0.0186$ at $D=1.00$).
- **Directional Interpretation in Seed 42**: In Seed 42 at $D=0.25$ and $D=0.50$, the frozen bootstrap procedure produced small sign-tail probabilities ($p = 0.009$ and $p = 0.000$) and 95% bootstrap confidence intervals entirely below zero ($\Delta\mathrm{IC}_{2\mathrm{Y}} = -0.0966$, 95% CI $[-0.1775, -0.0147]$; $\Delta\mathrm{IC}_{2\mathrm{Y}} = -0.0938$, 95% CI $[-0.1798, -0.0246]$). Because the preregistered economic alternative was directional ($\Delta\mathrm{IC} > 0$), these negative shifts represent evidence in the opposite direction (predictive degradation) and do not support the economic leakage hypothesis.
- **SPY Equities (Exploratory Economic Endpoint)**: All 95% bootstrap confidence intervals for SPY included zero. Mean $\Delta\mathrm{IC}_{\mathrm{SPY}}$ across seeds was $+0.0089$ ($D=0.25$), $+0.0100$ ($D=0.50$), $+0.0108$ ($D=0.75$), and $-0.0149$ ($D=1.00$).

### 6.8 Summary of the Leakage Propagation Chain
[Figure 4](../research/figures/phase4b/figure4_layered_outcome_comparison.png) synthesizes the empirical evidence across the five evaluated endpoints. The findings demonstrate systematic attenuation across the analytical stack:
1. Parametric temporal exposure produces a substantial and predominantly positive signal in latent representation space ($L_{\mathrm{repr}}$).
2. The signal decouples at the behavioral tier: binary classification accuracy does not improve, and masking sensitivity remains negligible.
3. No reliable positive downstream effect was detected at the economic tier: financial market Information Coefficients show no robust positive gains. The downstream endpoints did not support the preregistered positive alternative, and claims of "false alpha" resulting from post-cutoff exposure are unsupported by the empirical evidence.

---

## 7. Discussion

### 7.1 Latent Parametric Leakage: Geometry Precedes Behavior
A central conceptual contribution of this study is that **parametric temporal leakage can be latent**. In transformer language models, self-supervised pretraining on post-cutoff text primarily reshapes the high-dimensional geometry of internal activation spaces. These adjustments enhance the linear separability of future macroeconomic targets without necessarily altering surface-level token attribution or propagating through fixed downstream classification heads.

This observation connects directly with mechanistic interpretability \citep{alain2017understanding, belinkov2022probing}: neural representations often encode latent features that remain functionally dormant unless elicited by specialized linear readouts. In quantitative finance, auditing models solely through downstream backtest returns or prompt sensitivity risks severe false negatives. Probing latent representation manifolds provides a more sensitive diagnostic for parametric contamination.

### 7.2 Stochastic Optimization as Scientific Uncertainty
Our findings indicate that leakage susceptibility is not an inherent deterministic property of a model architecture or text corpus, but is strongly modulated by stochastic optimization dynamics. Although all branches within a seed shared identical initial weights and masking schedules, results diverged markedly across seeds: Seed 87 and Seed 2024 exhibited high susceptibility across doses, while Seed 123 displayed pronounced non-monotonic reversals.

This divergence suggests that whether leaked factual or directional information becomes linearly decodable depends on the specific gradient trajectory traversed during pretraining. Auditing studies that evaluate temporal leakage using a single random seed risk reporting idiosyncratic optimization artifacts rather than generalizable architectural properties. Multi-seed designs are indispensable for temporal leakage audits.

### 7.3 Mechanistic Hypotheses for Non-Monotonic Dose Response
Naive intuition suggests that temporal leakage should obey a monotonic law: higher contamination dose ($D$) should strictly produce greater representational leakage ($L_{\mathrm{repr}}$). Our empirical findings did not support a strictly monotonic dose-response relationship. Mean $L_{\mathrm{repr}}$ increased up to $D = 0.75$ but declined at $D = 1.00$.

We hypothesize several non-mutually-exclusive mechanisms that could account for this non-monotonicity:
- **Continued-Pretraining Interference**: At full contamination ($D = 1.00$), the pretraining stream consists entirely of post-2020 documents from a distinct macroeconomic regime (the pandemic emergency easing shock). Ingesting this distribution without contemporary sham data may induce distribution shift that disrupts the subtle feature subspace utilized by linear probes trained on 2016–2019 targets.
- **Catastrophic Forgetting**: High contamination fractions may overwrite earlier syntactic and semantic features acquired during base pretraining, degrading general feature extraction and offsetting leakage benefits \citep{jang2022temporal}.
- **Loss Landscapes and Non-Linear Manifolds**: At $D = 1.00$, the optimizer may settle into local minima where future information is stored in complex non-linear manifolds that are less accessible to linear probes \citep{alain2017understanding}.
- **Target Distribution Mismatch**: Post-2020 text reflects extreme zero-rate accommodation, whereas the evaluation targets ($\Delta\text{Rate}_{t+1}$) span a tightening cycle (2016–2018) followed by modest accommodation (2019). High-dose pretraining may align representations with the wrong directional regime.

These mechanisms represent exploratory hypotheses that warrant further structural investigation; they are not demonstrated empirical facts.

### 7.4 Representation Decodability vs. Practical Usefulness
The sharp contrast between positive continuous probing results ($L_{\mathrm{repr}}$) and null binary classification outcomes ($\Delta\text{Macro-F1}$) highlights the distinction between representation decodability and practical task utility. Continuous rate changes ($\Delta\text{Rate}_{t+1}$) preserve fine-grained directional magnitude information that linear ridge probes successfully extract from shifted latent manifolds. In contrast, discrete classification collapses continuous representations into a coarse decision boundary dominated by the historical class prior (hold). Increased latent decodability does not automatically translate into improved discrete decision accuracy.

### 7.5 Economic Attenuation and Financial Market Noise
Even when representations contain statistically detectable future policy information, our results establish no robust propagation into measurable market returns ($E_L$). Financial market price formation is heavily influenced by contemporaneous exogenous variables—geopolitical news, macroeconomic data surprises, fiscal policy debates—that occur simultaneously with central-bank announcements. Because the downstream endpoints did not support the preregistered positive alternative, latent representation leakage cannot be casually equated with profitable "false alpha."

### 7.6 Causal Boundaries: Distinguishing Leakage from Domain Adaptation
An alternative interpretation of positive $L_{\mathrm{repr}}$ is that continued pretraining simply improves representation quality by exposing the model to additional central-bank language (domain adaptation). Protocol v1.2.4 strictly controls for this confound by pairing each contaminated model with an active clean twin receiving an identical token budget and optimization schedule on contemporary central-bank text.

However, we emphasize the causal boundary: the paired design isolates the incremental effect of changing the temporal composition of the treatment corpus under matched architecture, initialization, compute, and training recipe. Nevertheless, temporal composition is not completely separable from all associated post-cutoff regime, topic, or rhetorical distribution differences between pre-2020 and post-2020 documents. For example, subtle shifts in post-2020 central-bank vocabulary during emergency easing may have coincidentally altered probe extractability.

---

## 8. Limitations

To ensure transparent reporting, we document eleven methodological and empirical limitations:

1. **Omnibus Confirmatory Rule Under-Specification**: Protocol v1.2.4 preregistered branch-level permutation tests but omitted a formal global decision rule for combining the 20 branch tests. Consequently, a formal global confirmatory rejection of $H_0^{\mathrm{repr}}$ is not claimed.
2. **Finite Seed Sample**: The study evaluated five random seeds ($N_{\mathrm{seed}} = 5$). While sufficient to document substantial seed-dependent optimization heterogeneity, this sample size is insufficient to model the full asymptotic distribution of gradient trajectories.
3. **Non-Monotonic Dose Transitions**: The protocol evaluated discrete doses ($D \in \{0.25, 0.50, 0.75, 1.00\}$). It was not designed to resolve the fine-grained transition threshold between $D=0.75$ and $D=1.00$ where representation drift or interference begins to dominate.
4. **Model Architecture Scope**: Experiments were conducted exclusively using the `ProsusAI/finbert` encoder (12-layer, 768-dim BERT-base). These findings cannot be extrapolated to billion-parameter autoregressive decoder models (e.g., Llama, GPT families) trained under causal language modeling objectives.
5. **Domain and Institutional Specificity**: The evaluation focused on Federal Open Market Committee communications. Central-bank discourse is characterized by a formal, highly structured vocabulary and fixed calendar cycles. Leakage dynamics may differ in unstructured domains (earnings calls, financial news, social media).
6. **Macroeconomic Regime Non-Stationarity**: The post-cutoff contamination period (2020–early 2023, specifically spanning 2020-01-29 through 2023-01-04) coincided with extreme economic shocks (the COVID-19 pandemic, supply dislocations, zero lower bound rates, and subsequent rapid rate hikes). Whether similar leakage patterns emerge under stationary macroeconomic regimes remains an open empirical question.
7. **Downstream Economic Statistical Power**: Economic endpoints were evaluated across 40 FOMC events. While adequate to detect representation shifts across 32 out-of-sample meetings, market returns are inherently noisy; subtle economic effects ($\Delta\mathrm{IC} \approx 0.01$–$0.02$) cannot be reliably distinguished from zero at this sample size.
8. **Absence of Independent Competence Split**: Model competence is formally cataloged as $C = \text{NOT\_EVALUATED\_NO\_EVAL\_SPLIT}$ because the confirmatory protocol prioritized out-of-sample causal contrasts across all available events rather than withholding an independent validation split.
9. **Absence of Temporal Robustness Control**: Temporal robustness is cataloged as $R_T = \text{NOT\_EVALUATED}$, isolating the clean-versus-contaminated contrast across identical temporal windows without benchmarking the natural rate of temporal degradation over time.
10. **Probing Interpretability Bounds**: Linear probes evaluate whether post-cutoff pretraining reshaped latent embedding geometry in ways that correlate with future policy shifts; they do not establish factual memorization or verbatim token recall \citep{belinkov2022probing, hewitt2019designing}.
11. **Base Checkpoint Pretraining Provenance**: The base model (`ProsusAI/finbert`) was pretrained prior to 2019, but its exact training corpus boundaries cannot be verified with second-level Point-in-Time cryptographic certainty. Therefore, this study estimates the **incremental treatment effect** of controlled post-cutoff continued pretraining relative to a shared initialization, rather than claiming absolute absence of prior temporal knowledge in the base checkpoint.

---

## 9. Conclusion

This study provides an empirical investigation of parametric temporal data leakage in financial language models. By implementing a causally symmetric twin-model architecture that equalizes parameter scale, compute volume, and text domain, we evaluated the effect of post-cutoff central-bank communications during continued pretraining on latent representation geometry, finding a predominantly positive representational leakage signal ($L_{\mathrm{repr}} > 0$ in 18/20 branches; 10/20 nominally significant under one-sided right-tailed paired sign-flip permutation tests).

However, no reliable positive downstream effect was detected across discrete policy classification accuracy, behavioral masking sensitivities, or downstream financial market predictability. The observed results are consistent with a layered view in which representation-level temporal leakage signals need not translate into downstream behavioral or economic effects. Furthermore, representational leakage displayed substantial seed-dependent optimization heterogeneity and non-monotonic dose dynamics.

These findings establish that parametric temporal leakage cannot be treated as a monolithic binary condition, nor can it be assumed that latent representation leakage immediately translates into profitable market alpha. Robust financial AI governance requires layered, multi-seed auditing frameworks that evaluate model geometry, task behavior, and market outcomes simultaneously while maintaining strict causal symmetry.

---

## References

\begin{thebibliography}{26}
\providecommand{\natexlab}[1]{#1}

\bibitem[Alain and Bengio(2017)]{alain2017understanding}
Guillaume Alain and Yoshua Bengio.
\newblock Understanding intermediate layers using linear classifier probes.
\newblock In \emph{5th International Conference on Learning Representations (ICLR), Workshop Track}, 2017.

\bibitem[Araci(2019)]{araci2019finbert}
Dogu Araci.
\newblock {FinBERT}: Financial sentiment analysis with pre-trained language models.
\newblock \emph{arXiv preprint arXiv:1908.10063}, 2019.

\bibitem[Bailey et~al.(2014)Bailey, Borwein, L{\'o}pez~de Prado, and Zhu]{bailey2014pseudo}
David~H. Bailey, Jonathan~M. Borwein, Marcos L{\'o}pez~de Prado, and Qiji~Jim Zhu.
\newblock Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance.
\newblock \emph{Notices of the American Mathematical Society}, 61\penalty0 (5):\penalty0 458--471, 2014.

\bibitem[Belinkov(2022)]{belinkov2022probing}
Yonatan Belinkov.
\newblock Probing classifiers: Promises, shortcomings, and advances.
\newblock \emph{Computational Linguistics}, 48\penalty0 (1):\penalty0 207--219, 2022.

\bibitem[Benhenda(2026)]{benhenda2026lookahead}
Mostapha Benhenda.
\newblock Look-Ahead-Bench: a standardized benchmark of look-ahead bias in point-in-time {LLMs} for finance.
\newblock \emph{arXiv preprint arXiv:2601.13770}, 2026.

\bibitem[Carlini et~al.(2021)Carlini, Tram{\`e}r, Wallace, Jagielski, Herbert-Voss, Lee, Roberts, Brown, Song, Erlingsson, Oprea, and Raffel]{carlini2021extracting}
Nicholas Carlini, Florian Tram{\`e}r, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, {\'U}lfar Erlingsson, Alina Oprea, and Colin Raffel.
\newblock Extracting training data from large language models.
\newblock In \emph{30th USENIX Security Symposium (USENIX Security 21)}, pages 2633--2650, 2021.

\bibitem[Carlini et~al.(2023)Carlini, Ippolito, Jagielski, Lee, Tram{\`e}r, and Zhang]{carlini2023quantifying}
Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tram{\`e}r, and Chiyuan Zhang.
\newblock Quantifying memorization across neural language models.
\newblock In \emph{The Eleventh International Conference on Learning Representations (ICLR)}, 2023.

\bibitem[Devlin et~al.(2019)Devlin, Chang, Lee, and Toutanova]{devlin2019bert}
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
\newblock {BERT}: Pre-training of deep bidirectional transformers for language understanding.
\newblock In \emph{Proceedings of NAACL-HLT 2019}, pages 4171--4186, 2019.

\bibitem[Dhingra et~al.(2022)Dhingra, Cole, Eisenschlos, Gillick, Eisenstein, and Cohen]{dhingra2022time}
Bhuwan Dhingra, Jeremy~R. Cole, Julian~Martin Eisenschlos, Daniel Gillick, Jacob Eisenstein, and William~W. Cohen.
\newblock Time-aware language models as temporal knowledge bases.
\newblock \emph{Transactions of the Association for Computational Linguistics}, 10:\penalty0 257--273, 2022.

\bibitem[Geirhos et~al.(2020)Geirhos, Jacobsen, Michaelis, Zemel, Brendel, Bethge, and Wichmann]{geirhos2020shortcut}
Robert Geirhos, J{\"o}rn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix~A. Wichmann.
\newblock Shortcut learning in deep neural networks.
\newblock \emph{Nature Machine Intelligence}, 2\penalty0 (11):\penalty0 665--673, 2020.

\bibitem[Golchin and Surdeanu(2024)]{golchin2024time}
Shahriar Golchin and Mihai Surdeanu.
\newblock Time travel in {LLMs}: Tracing data contamination in large language models.
\newblock In \emph{The Twelfth International Conference on Learning Representations (ICLR)}, 2024.

\bibitem[Hansen and McMahon(2016)]{hansen2016shocking}
Stephen Hansen and Michael McMahon.
\newblock Shocking language: Analyzing the macroeconomic effects of central bank communication.
\newblock \emph{Journal of International Economics}, 99:\penalty0 S114--S133, 2016.

\bibitem[Hewitt and Liang(2019)]{hewitt2019designing}
John Hewitt and Percy Liang.
\newblock Designing and interpreting probes with control tasks.
\newblock In \emph{Proceedings of EMNLP-IJCNLP 2019}, pages 2733--2743, 2019.

\bibitem[Huang et~al.(2023)Huang, Wang, and Yang]{huang2023finbert}
Allen~H. Huang, Hui Wang, and Yi~Yang.
\newblock {FinBERT}: A large language model for extracting information from financial text.
\newblock \emph{Contemporary Accounting Research}, 40\penalty0 (2):\penalty0 806--841, 2023.

\bibitem[Jang et~al.(2022)Jang, Ye, Yang, Shin, Han, Kim, Choi, and Seo]{jang2022temporal}
Joel Jang, Seonghyeon Ye, Sohee Yang, Joongbo Shin, Janghoon Han, Gyeonghun Kim, Stanley~Jungkyu Choi, and Minjoon Seo.
\newblock Towards continual knowledge learning of language models.
\newblock In \emph{The Tenth International Conference on Learning Representations (ICLR)}, 2022.

\bibitem[Kaufman et~al.(2012)Kaufman, Rosset, Perlich, and Stitelman]{kaufman2012leakage}
Shachar Kaufman, Saharon Rosset, Claudia Perlich, and Ori Stitelman.
\newblock Leakage in data mining: Formulation, detection, and avoidance.
\newblock \emph{ACM Transactions on Knowledge Discovery from Data (TKDD)}, 6\penalty0 (4):\penalty0 1--21, 2012.

\bibitem[Kim et~al.(2024)Kim, Muhn, and Nikolaev]{kim2024financial}
Alex Kim, Maximilian Muhn, and Valeri Nikolaev.
\newblock Financial statement analysis with large language models.
\newblock \emph{University of Chicago Booth School of Business Research Paper / SSRN:4835311}, 2024.

\bibitem[L{\'o}pez~de Prado(2018)]{lopezdeprado2018advances}
Marcos L{\'o}pez~de Prado.
\newblock \emph{Advances in Financial Machine Learning}.
\newblock John Wiley \& Sons, Hoboken, NJ, 2018.

\bibitem[Lopez-Lira and Tang(2023)]{lopezlira2023chatgpt}
Alejandro Lopez-Lira and Yuehua Tang.
\newblock Can {ChatGPT} forecast stock price movements? {Return} predictability and large language models.
\newblock \emph{arXiv preprint arXiv:2304.07619}, 2023.

\bibitem[Luu et~al.(2022)Luu, Khashabi, Gururangan, Mandyam, and Smith]{luu2022temporal}
Kelvin Luu, Daniel Khashabi, Suchin Gururangan, Karishma Mandyam, and Noah~A. Smith.
\newblock Time waits for no one! {Analysis} and challenges of temporal misalignment.
\newblock In \emph{Proceedings of NAACL-HLT 2022}, pages 5944--5958, 2022.

\bibitem[Oren et~al.(2024)Oren, Meister, Chatterji, Ladhak, and Hashimoto]{oren2024proving}
Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, and Tatsunori~B. Hashimoto.
\newblock Proving test set contamination in black-box language models.
\newblock In \emph{The Twelfth International Conference on Learning Representations (ICLR)}, 2024.

\bibitem[Sainz et~al.(2023)Sainz, Campos, Garc{\'\i}a-Ferrero, Etxaniz, and Agirre]{sainz2023nlp}
Oscar Sainz, Jon~Ander Campos, Iker Garc{\'\i}a-Ferrero, Julen Etxaniz, and Eneko Agirre.
\newblock {NLP} evaluation in trouble: On the need to verify the cleanliness of test data.
\newblock In \emph{Proceedings of EMNLP 2023}, pages 15340--15353, 2023.

\bibitem[Shah et~al.(2023)Shah, Paturi, and Chava]{shah2023trillion}
Agam Shah, Suvan Paturi, and Sudheer Chava.
\newblock Trillion dollar words: A new financial dataset, task \& market analysis.
\newblock In \emph{Proceedings of ACL 2023 (Volume 1: Long Papers)}, pages 6664--6679, 2023.

\bibitem[Tang and Yang(2026)]{tang2026mind}
Yixuan Tang and Yi~Yang.
\newblock Mind the shift: Decoding monetary policy stance from {FOMC} statements with large language models.
\newblock \emph{arXiv preprint arXiv:2603.14313}, 2026.

\bibitem[Xue et~al.(2026)Xue, Guesmi, Feng, Gong, Sundram, Pang, Wang, and Kaljuvee]{xue2026temporal}
Chenhao Xue, Raslen Guesmi, Siwei Feng, Yucheng Gong, Jacob~Xavier Sundram, Jordan Pang, Lan Wang, and Julian Kaljuvee.
\newblock Temporal leakage in financial news {NLP}: A multi-architecture audit with a regime-specific {M\&A} signal.
\newblock \emph{arXiv preprint arXiv:2608.17223}, 2026.

\bibitem[Zhang et~al.(2026)Zhang, Chen, and Stadie]{zhang2026all}
Zeyu Zhang, Ryan Chen, and Bradly~C. Stadie.
\newblock All leaks count, some count more: Interpretable temporal contamination detection and mitigation in {LLM} backtesting.
\newblock \emph{arXiv preprint arXiv:2602.17234}, 2026.

\end{thebibliography}

---

## Appendix: Reproducibility & Protocol Specification Ledger

This appendix provides experimental provenance parameters and archival verification hashes:

| Configuration Item | Formal Value / Identifier |
| :--- | :--- |
| **Protocol Specification** | Version `1.2.4` |
| **Scientific Code Freeze Commit** | `5ec0f03f3a5393d90462aa78d018d54b08cce126` |
| **Phase 4B Execution Closure Commit** | `5d82a7a321f6ce2441e449b5fff7ec482681819e` |
| **Locked Source Tree SHA-256** | `02fe0ced0d03a920a7f56887f1d674282d17bc108986d19d2df495b59d330bb6` |
| **Base Model Checkpoint** | `ProsusAI/finbert` |
| **Base Model Git Revision** | `4556d13015211d73dccd3fdd39d39232506f3e43` |
| **Total Experimental Branches** | 25 branches (5 seeds $\times$ 5 doses) |
| **Evaluation Seeds** | `[13, 42, 87, 123, 2024]` |
| **Contamination Doses** | `[0.00, 0.25, 0.50, 0.75, 1.00]` |
| **MLM Treatment Block Format** | 500 packed blocks $\times$ 512 tokens = 256,000 tokens per branch |
| **MLM Optimizer Execution** | 100 gradient steps (batch size 16, lr $5\times 10^{-5}$, weight decay 0.01) |
| **MLM Scheduler & Warmup** | Scheduler: `none`, warmup_ratio: `0.0` |
| **Downstream Training Recipe** | 3 epochs, batch size 16, lr $2\times 10^{-5}$, max seq len 128, linear scheduler, warmup 0.1 |
| **Downstream Training Schedule** | 3 epochs (batch size 16, max_steps 500) under paired sample order (`paired_within_seed`) |
| **Primary Probing Architecture** | Ridge regression ($\alpha = 1.0$) on frozen extracted mean-pooled layer-12 representations |
| **Temporal Cross-Validation Structure**| 4-fold grouped expanding-window temporal CV (32 out-of-sample meetings) |
| **Permutation Test Draws ($B$)** | 2,000 draws (one-sided right-tailed paired sign-flip) |
| **Bootstrap Test Resamples** | 1,000 draws (event-level stationary block bootstrap) |
| **Aggregate Constructed Treatment Budget**| 6,400,000 tokens ($25 \times 256,000$ tokens) |
| **Empirical Results Manifest** | `experiments/phase4_confirmatory/result_manifest.json` (54 files, 100% SHA-256 match) |
