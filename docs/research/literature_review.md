# Literature Review & Evaluation Matrix: Temporal Leakage in Financial NLP

## 1. Executive Summary

This literature review supports the research agenda outlined in `AGENT.md`: investigating **parametric temporal data leakage** in encoder-based financial language models and determining whether such leakage generates **false alpha** in backtested trading strategies.

The primary research question is:
> **How can temporal information leakage be detected and quantified in discriminative financial language encoders, and how can its economic consequences be separated from genuine task competence?**

A systematic literature search was conducted across top NLP and computational finance venues (ACL, EMNLP, arXiv, Chicago Booth, JFE/RFS-adjacent working papers) covering the following key themes:
1. Look-ahead bias in language models & financial NLP
2. Temporal distribution shift vs. temporal leakage
3. Central bank communication & FOMC hawkish/neutral/dovish stance classification
4. Entity masking and counterfactual testing
5. Representation probing for future information

---

## 2. Systematic Literature Matrix

The matrix evaluates each paper across 10 critical criteria:
* **Model**: Architectures evaluated.
* **Encoder / Decoder**: Discriminative encoder vs. generative auto-regressive decoder.
* **Leakage Definition**: How the paper defines temporal leakage.
* **Leakage Metric**: Quantitative metric used to measure leakage.
* **Economic Metric**: Financial return/risk metrics measured.
* **Task Metric**: NLP classification/prediction metric.
* **Dataset**: Primary corpus and time span.
* **Cutoff Known?**: Whether the pretraining data cutoff date is strictly documented.
* **Reproducible?**: Openness of code, datasets, and checkpoints.
* **Flaws & Critical Audit**: Methodological shortcomings and failure modes.

| Paper | Model | Architecture | Leakage Definition | Leakage Metric | Economic Metric | Task Metric | Dataset | Cutoff Known? | Reproducible? | Critical Audit & Key Takeaways |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Shah et al. (ACL 2023)**<br>*Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis* | RoBERTa-large, FinBERT, BERT | **Encoder** | Not formally defined; assumes chronological train/test split prevents leakage. | None. | Market reaction correlation ($\Delta$ Yields, $\Delta$ Equity index). | Macro-F1, Accuracy. | FOMC 1996–2022 (minutes, speeches, press conferences). | Checkpoint dates known, but pretraining web corpus dates fuzzy. | **High** (Public on GitHub and Hugging Face). | **Strengths**: Established the definitive benchmark for FOMC sentence classification.<br>**Critical Flaw**: Pretrained `roberta-large` trained on Common Crawl likely ingested historical FOMC transcripts and subsequent market commentary. Does not verify whether RoBERTa "memorized" historical meetings. |
| **Kim, Muhn & Nikolaev (Chicago Booth 2024)**<br>*Financial Statement Analysis with LLMs* | GPT-4-Turbo, GPT-3.5-Turbo | **Decoder** | Parametric memorization of firm future earnings and historical stock trajectories. | Delta in performance between in-sample and post-cutoff periods. | Long-short hedge portfolio return, Sharpe ratio, $\Delta$ Alpha. | Directional earnings accuracy (60.3%), MSE. | Compustat financial statements (1968–2021). | Partial (Relies on vendor API cutoff statements). | **Medium** (Closed APIs drift over time; prompts open). | **Strengths**: Pioneered entity anonymization (replacing company names with generic tokens) and relative time indicators ($t, t-1$).<br>**Critical Flaw**: Tested against proprietary closed models; cannot audit parameter activations directly; high prompt sensitivity. |
| **arXiv 2026 Audit**<br>*Temporal Leakage in Financial News NLP: A Multi-Architecture Audit* | Llama-3 (8B/70B), Qwen2.5, FinBERT, DeBERTa | **Both** | Auto-regressive leakage across autocorrelated market regimes when using random splits. | MCC Inflation Ratio: $\frac{\text{MCC}_{\text{random}}}{\text{MCC}_{\text{chronological}}}$. | Simulated strategy Sharpe ratio, Hit rate. | MCC, Macro-F1. | Financial news headlines + M&A announcement dates (2018–2025). | Yes (Open-weights checkpoints). | **High** (Rigorous code repository). | **Strengths**: Proved that random train-test splits inflate financial NLP metrics by **1.1x to 6.5x**.<br>**Critical Flaw**: Evaluated pipeline/split leakage; did not isolate parametric representation leakage within the encoder itself. |
| **Look-Ahead-Bench (2026)**<br>*A Standardized Benchmark of Look-Ahead Bias in Point-in-Time LLMs* | Llama-2/3, Mistral, FinMA, BloombergGPT variants | **Decoder** | Knowledge retrieval of market facts occurring after nominal simulation timestamp. | Temporal decay function $D(\Delta t)$; factual claim accuracy post-cutoff. | Information Coefficient (IC), Sortino ratio. | Factuality score, Extraction QA accuracy. | Financial news + SEC disclosures (2020–2025). | Partial (depends on base model). | **Medium**. | **Strengths**: Standardized time-dependent evaluation curves.<br>**Critical Flaw**: Conflates natural concept drift ($P_t(X,Y)$ shift) with leakage. Assumes any post-cutoff drop is proof of leakage. |
| **Araci (2019)**<br>*FinBERT: Financial Sentiment Analysis with Pre-trained Language Models* | BERT-base-uncased | **Encoder** | Not considered. | None. | None (pure NLP task). | Macro-F1, Accuracy. | Financial PhraseBank, FiQA sentiment. | Yes ($\le 2018$). | **High** (Hugging Face standard). | **Strengths**: Clean pre-2019 baseline model with transparent provenance.<br>**Critical Flaw**: Datasets lack temporal timestamps; cannot be used directly for temporal backtesting without synthetic alignment. |
| **Guenther et al. (2025)**<br>*Entity Masking and Shortcut Learning in Financial Sentiment* | FinBERT, RoBERTa, ELECTRA | **Encoder** | Entity-return shortcut learning (model memorizes that "Apple" or "Nvidia" goes up). | Prediction divergence under entity substitution: $L_{\text{entity}} = \text{KL}(P_{\text{orig}} \parallel P_{\text{anon}})$. | Spread between top and bottom return deciles. | Accuracy, F1. | Refinitiv financial news (2015–2024). | Yes. | **High**. | **Strengths**: Elegant entity anonymization protocol.<br>**Critical Flaw**: Only masked corporate entity tokens; did not mask macroeconomic entities, dates, or forward policy guidance statements. |
| **Mind the Shift (2026)**<br>*Delta-Consistent Scoring (DCS) for Central Bank Communication* | RoBERTa, DeBERTa-v3 | **Encoder** | Inter-meeting stance inconsistency. | Stance shift correlation with macroeconomic surprise. | Cumulative Abnormal Return (CAR) around FOMC press conferences. | Pairwise ranking accuracy, Concordance index. | FOMC statements and press conference opening remarks (2000–2025). | Yes. | **High**. | **Strengths**: Correctly identified that financial markets price the *change in stance* ($\Delta \text{Stance}_t$) rather than the absolute stance level.<br>**Critical Flaw**: Did not control for pretraining corpus contamination; models evaluated were released well after the test dates. |
| **TimeSPEC (2025)**<br>*Time-Supervised Prediction with Extracted Claims* | GPT-4o, Claude-3.5, Llama-3-70B | **Decoder** | Generation of claims referencing post-cutoff events in reasoning trajectories. | Temporal citation violation rate: $\frac{\text{Post-cutoff claims}}{\text{Total claims}}$. | Max drawdown, Downside volatility. | Claim verifiability, Precision. | SEC 10-K/10-Q filing events. | Partial. | **Low to Medium**. | **Strengths**: Fine-grained claim-level timestamp verification.<br>**Critical Flaw**: Decoder reasoning approach is computationally expensive and introduces prompt and verbosity confounds. |
| **Huang, Wang & Yang (2023)**<br>*FinBERT-Tone: Central Bank Tone and Macroeconomic Policy* | FinBERT, BERT-base | **Encoder** | Not defined. | None. | Predictive $R^2$ for Fed Funds Rate changes and 10Y Treasury yield moves. | Accuracy, F1. | FOMC minutes and speeches (1998–2020). | Yes. | **High**. | **Strengths**: Proved that monetary stance text has significant explanatory power for asset pricing.<br>**Critical Flaw**: Conflated NLP classification competence with economic predictive power; lacked a temporal clean control. |
| **DecisionFin (2026)**<br>*Decision-Centric Memorization Audits in Financial NLP* | BERT, RoBERTa, Mistral | **Both** | Reliance on memorized future event outcomes to alter trading signals. | Counterfactual flip probability: $P(\hat{y}_{\text{orig}} \neq \hat{y}_{\text{counterfactual}})$. | $\Delta \text{Sharpe}$ under counterfactual interventions. | Decision accuracy, Macro-F1. | Corporate earnings calls and earnings surprise data. | Yes. | **High**. | **Strengths**: Introduced the conceptual decomposition connecting memorization to behavioral shift and financial loss/gain.<br>**Critical Flaw**: Did not construct controlled twin pretraining doses ($0\%, 25\%, 50\%, 75\%, 100\%$). |

---

## 3. Critical Methodological Audits & Recurring Flaws

From this literature audit, four major methodological errors were identified across existing studies:

### Flaw 1: Conflating Post-Cutoff Degradation with Leakage
A widespread practice (e.g., in Look-Ahead-Bench and earlier LLM backtest papers) is observing that a model's F1-score or trading Sharpe drops when evaluated on data after its training cutoff date, and concluding:
$$\text{Performance Drop } \implies \text{Temporal Leakage in Pre-Cutoff Period}$$
**Why this is mathematically and economically invalid**:
Financial markets and macroeconomic discourse are non-stationary:
$$P_t(X, Y) \neq P_{t+\Delta t}(X, Y)$$
Between 2018 and 2022, vocabulary, macroeconomic regimes, and monetary reaction functions shifted drastically (COVID-19 pandemic, supply-chain bottlenecks, zero lower bound, quantitative easing, subsequent 500 bps rate hike cycle). A model trained before 2019 will naturally experience **temporal distribution shift (concept drift)**:
$$D(\Delta t) = \text{Shift-Induced Degradation} + \text{Leakage Effect}$$
Attributing all decay to leakage is a fundamental confounding error. **Temporal robustness ($R_T$) must be treated as an orthogonal control variable.**

### Flaw 2: Conflating Model Capacity with Leakage
Several studies compare an older model (e.g., BERT-2018) against a newer model (e.g., DeBERTa-2023 or Llama-3-2024), find that the newer model yields a higher Sharpe ratio on 2020–2023 data, and claim this demonstrates "temporal contamination."
**Why this is invalid**:
The newer model differs in:
- Model parameter count (110M vs. 8B–70B)
- Tokenizer vocabulary and efficiency
- Pretraining corpus size (16 GB vs. 15 TB)
- Architecture optimizations (Rotary embeddings, SwiGLU, FlashAttention)
The observed delta represents **model capacity and general language capability**, not pure temporal leakage. Causal estimation requires a **Clean/Leak Twin Model design** where architecture, capacity, tokenizer, and task fine-tuning are held strictly invariant.

### Flaw 3: Circularity in Task Label Construction
Some financial NLP papers derive the ground-truth NLP label $y_t$ directly from post-event market price reactions (e.g., if SPY dropped $>1\%$, label the FOMC statement as "Hawkish"). They then use the model's predictions $\hat{y}_t$ to run a trading strategy on SPY and report a stellar Sharpe ratio.
**Why this is invalid**:
This creates blatant circularity:
$$\text{Future Price Reaction } \longrightarrow \text{Ground Truth Label } Y \longrightarrow \text{Fine-Tuning } \longrightarrow \text{Trading Backtest on Same Future Price}$$
The NLP task stance $Y_{\text{task}}$ must be defined strictly from **linguistic semantics or contemporaneous policy actions**, completely isolated from forward market returns $Y_{\text{econ}}$.

### Flaw 4: Single Composite Scoring Fallacy
Attempting to rank models via an ad-hoc scalar:
$$\text{Score} = w_1 C - w_2 L - w_3 E_L$$
obscures the reality that a completely inert model $M_0(x) = \text{Neutral}$ exhibits $L=0$ and $E_L=0$, but has zero competence ($C=0$). Scalar aggregation conceals critical trade-offs. The correct scientific presentation is a **Pareto Frontier** in $(L, E_L, C)$ space.

---

## 4. Synthesis & Scientific Positioning of MANTRA

By synthesizing the strengths of *Trillion Dollar Words* (gold-standard annotations), *Guenther et al.* (entity anonymization), and *DecisionFin* (decision-centric auditing), MANTRA introduces the first unified framework satisfying all the following:

1. **Focus on Discriminative Encoders**: Eliminates generative confounds (hallucination, decoding temperature, prompt phrasing, chain-of-thought artifacts).
2. **Clean/Leak Twin Architecture**: Causal estimation via identical base architectures with controlled pretraining exposure doses ($D \in \{0\%, 25\%, 50\%, 75\%, 100\%\}$).
3. **Three-Tier Separation**:
   $$L_{\text{repr}} \ (\text{Probing}) \longrightarrow L_{\text{behavior}} \ (\text{Counterfactual Masking}) \longrightarrow E_L \ (\text{Paired Economic Delta})$$
4. **Strict Point-in-Time Data Infrastructure**: Guarantees zero pipeline/external leakage so that measured leakage is guaranteed to be purely parametric.
