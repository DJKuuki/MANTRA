# Audited Literature Review: Temporal Leakage in Financial NLP

## 1. Executive Summary & Research Motivation

This literature review supports the research agenda outlined in `AGENT.md`: investigating **parametric temporal data leakage** in encoder-based financial language models and evaluating its economic impact on backtested financial predictions.

The core research question is:
> **How can temporal information leakage be detected and quantified in discriminative financial language encoders, and how can its economic consequences be separated from genuine task competence?**

### Methodological Disclaimer on Point-in-Time (PIT) Data
Point-in-Time (PIT) safeguards are strictly enforced for audited data sources in this repository; any unsupported or external data sources must be disabled or explicitly marked as heuristic approximations. We do NOT claim that external vendors guarantee zero leakage.

---

## 2. Verified Literature Matrix & Machine-Verifiable Registry

All literature entries are synchronized with the machine-verifiable registry at [`docs/research/literature_registry.json`](./literature_registry.json).
Each paper in this matrix has been verified against canonical venue publications, DOIs, or arXiv IDs. Unverified or hallucinated citations have been explicitly removed.

| Title | Authors | Year | Venue | DOI / arXiv ID | Canonical URL | Code / Model URL | Verified Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis** | Agam Shah, Suvan Paturi, Sudheer Chava | 2023 | ACL 2023 (Long Papers) | [arXiv:2305.07972](https://arxiv.org/abs/2305.07972) / [10.18653/v1/2023.acl-long.368](https://doi.org/10.18653/v1/2023.acl-long.368) | [ACL Anthology (368)](https://aclanthology.org/2023.acl-long.368/) | [gtfintechlab/fomc-hawkish-dovish](https://github.com/gtfintechlab/fomc-hawkish-dovish) | **Verified** (2026-09-19) |
| **Temporal Leakage in Financial News NLP: A Multi-Architecture Audit with a Regime-Specific M&A Signal** | Chenhao Xue, Raslen Guesmi, Siwei Feng, Yucheng Gong, Jacob Xavier Sundram, Jordan Pang, Lan Wang, Julian Kaljuvee | 2026 | arXiv / Under review (EMNLP 2026) | [arXiv:2608.17223](https://arxiv.org/abs/2608.17223) | [arXiv:2608.17223](https://arxiv.org/abs/2608.17223) | [ChenHX111/Temporal_Leakage](https://github.com/ChenHX111/Temporal_Leakage_in_Financial_News_NLP) | **Verified** (2026-09-19) |
| **Mind the Shift: Decoding Monetary Policy Stance from FOMC Statements with Large Language Models** | Yixuan Tang, Yi Yang | 2026 | arXiv | [arXiv:2603.14313](https://arxiv.org/abs/2603.14313) | [arXiv:2603.14313](https://arxiv.org/abs/2603.14313) | [yixuantt/DeltaConsistentScoring](https://github.com/yixuantt/DeltaConsistentScoring) | **Verified** (2026-09-19) |
| **All Leaks Count, Some Count More: Interpretable Temporal Contamination Detection and Mitigation in LLM Backtesting** | Zeyu Zhang, Ryan Chen, Bradly C. Stadie | 2026 | Findings of EMNLP 2026 | [arXiv:2602.17234](https://arxiv.org/abs/2602.17234) | [arXiv:2602.17234](https://arxiv.org/abs/2602.17234) | Proposed TimeSPEC architecture & Shapley-DCLR | **Verified** (2026-09-19) |
| **Look-Ahead-Bench: a Standardized Benchmark of Look-ahead Bias in Point-in-Time LLMs for Finance** | Mostapha Benhenda | 2026 | arXiv | [arXiv:2601.13770](https://arxiv.org/abs/2601.13770) | [arXiv:2601.13770](https://arxiv.org/abs/2601.13770) | [benstaf/lookaheadbench](https://github.com/benstaf/lookaheadbench) | **Verified** (2026-09-19) |
| **FinBERT: Financial Sentiment Analysis with Pre-trained Language Models** | Dogu Araci | 2019 | arXiv | [arXiv:1908.10063](https://arxiv.org/abs/1908.10063) | [arXiv:1908.10063](https://arxiv.org/abs/1908.10063) | [ProsusAI/finBERT](https://github.com/ProsusAI/finBERT) | **Verified** (2026-09-19) |
| **FinBERT: A Large Language Model for Extracting Information from Financial Text** | Allen H. Huang, Hui Wang, Yi Yang | 2023 | Contemporary Accounting Research, 40(2), 806–841 | [10.1111/1911-3846.12832](https://doi.org/10.1111/1911-3846.12832) | [Wiley Online Library](https://onlinelibrary.wiley.com/doi/10.1111/1911-3846.12832) | [yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone) | **Verified** (2026-09-19) |
| **Financial Statement Analysis with Large Language Models** | Alex Kim, Maximilian Muhn, Valeri Nikolaev | 2024 | Chicago Booth / SSRN | [arXiv:2405.02794](https://arxiv.org/abs/2405.02794) | [SSRN:4835311](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4835311) | Proprietary API evaluation | **Verified** (2026-09-19) |
| **Shortcut Learning in Deep Neural Networks** | Robert Geirhos, et al. | 2020 | Nature Machine Intelligence, 2, 665–673 | [10.1038/s42256-020-00257-z](https://doi.org/10.1038/s42256-020-00257-z) | [Nature MI](https://www.nature.com/articles/s42256-020-00257-z) | General deep learning benchmark audits | **Verified** (2026-09-19) |

### Disambiguation & Audit Notes
- **FinBERT Disambiguation**:
  - *Araci (2019)*: Pretrained on Financial PhraseBank/TRC2 for 3-class sentiment analysis with a pre-2019 cutoff. Open repo: `ProsusAI/finBERT`.
  - *Huang, Wang & Yang (2023)*: Pretrained on SEC corporate 10-Ks, 10-Qs, and conference call transcripts, published in *Contemporary Accounting Research*. Model checkpoint: `yiyanghkust/finbert-tone`.
- **TimeSPEC Architecture vs. Metric**:
  - In *Zhang et al. (2026)*, **TimeSPEC** (Time-Supervised Prediction with Extracted Claims) is the inference-time mitigation architecture, while **Shapley-DCLR** (Shapley-weighted Decision-Critical Leakage Rate) is the evaluation metric.
- **Removed / Unverified Citations**:
  - `Guenther et al. (2025)`: **REMOVED (UNVERIFIED)**. No canonical record exists in DBLP, arXiv, or CrossRef.
  - `DecisionFin (2026)`: **REMOVED (UNVERIFIED)**. Replaced by verified benchmarks: Look-Ahead-Bench (Benhenda 2026) and TimeSPEC (Zhang et al. 2026).

---

## 3. Methodological Comparison Across Verified Papers

| Verified Paper | Architecture | Leakage Definition | Leakage Metric | Economic Evaluation | Task Metric | Training Cutoff Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Shah et al. (2023)** | RoBERTa-large, FinBERT (Encoder) | Not formally defined; uses chronological split. | None | Market reaction correlation ($\Delta$ Yields) | Macro-F1, Accuracy | Standard Hugging Face checkpoints. |
| **Kim, Muhn & Nikolaev (2024)** | GPT-4-Turbo (Decoder) | Memorization of firm forward earnings. | Delta between in-sample and out-of-sample periods | Long-short portfolio return, Sharpe ratio | EPS forecast accuracy (60.3%) | Proprietary API cutoff statements. |
| **Xue et al. (2026)** | FinBERT, RoBERTa, DeBERTa, Llama-3, Qwen2.5 (Both) | Regime memorization across random splits. | MCC Inflation Ratio ($\frac{\text{MCC}_{\text{random}}}{\text{MCC}_{\text{chronological}}}$) | Simulated strategy Sharpe, Hit rate | MCC, Macro-F1 | Open-weight checkpoints with audited dates. |
| **Look-Ahead-Bench (2026)** | Llama-3.1, DeepSeek, Pitinf models (Decoder) | Retrieval of facts postdating simulation date. | Temporal decay curve $D(\Delta t)$ | Information Coefficient (IC), Sortino | Factuality score, Extraction accuracy | Open weights. |
| **Araci (2019)** | BERT-base (Encoder) | Not considered. | None | None | Macro-F1, Accuracy | Wikipedia + TRC2 ($\le 2018$). |
| **Tang & Yang (2026)** | Frozen LLM Representations (Decoder) | Static stance insensitivity to inter-meeting shifts. | Delta-Consistent Scoring correlation with macro surprise | Treasury Yield correlation | Inter-meeting consistency | Known open weights. |
| **Zhang, Chen & Stadie (2026)** | GPT-4o, Llama-3-70B (Decoder) | Generation of post-cutoff claims in reasoning chains. | Decision-Critical Leakage Rate (Shapley-DCLR) | Backtest win rate, Downside volatility | Claim verifiability, Precision | Closed & open models. |
| **Huang, Wang & Yang (2023)** | FinBERT-Tone (Encoder) | Not defined. | None | Predictive $R^2$ for rate changes | Accuracy, F1 | Corporate 10-Ks, 10-Qs, transcripts. |

---

## 4. Methodological Audit & Core Flaws in the Literature

### Flaw 1: Conflating Temporal Distribution Shift with Leakage
A common practice (e.g. in Look-Ahead-Bench) is observing that model performance drops after a cutoff date and interpreting this drop as proof of past leakage:
$$\text{Performance Drop } \neq \text{Temporal Leakage}$$
Macroeconomic regimes are non-stationary:
$$P_t(X, Y) \neq P_{t+\Delta t}(X, Y)$$
A pre-2019 model experiences concept drift during COVID-19 and the 2022 rate hike cycle. **Temporal Robustness ($R_T$) must be treated as an orthogonal control variable, not confused with leakage.**

### Flaw 2: Masking Sensitivity ≠ Behavioral Leakage
Sensitivity to entity or date masking ($S_{\text{mask}}(M) = \mathbb{E}[JS(P(y|x) \parallel P(y|\mathcal{T}(x)))]$) measures how much a model relies on contemporaneous entities (e.g. "Federal Reserve", "Powell").
A model can legitimately utilize current named entities without having seen future data.
Therefore, **Behavioral Leakage ($L_{\text{behavior}}$) must be defined as the Clean/Leak Twin Differential**:
$$L_{\text{behavior}} = S_{\text{mask}}(M_L) - S_{\text{mask}}(M_C)$$

### Flaw 3: Conflating Model Capacity with Contamination
Comparing an older small model (BERT-2018, 110M) against a modern large model (Llama-3-2024, 70B) conflates capacity, vocabulary efficiency, and training scale with temporal leakage.
A valid causal estimate requires a **Clean / Leak Twin Model design** holding architecture, parameter count, tokenizer, and downstream fine-tuning identical.

### Flaw 4: Circular Task Label Construction
Using future market price changes to construct the NLP stance label $Y_{\text{task}}$ creates circularity when that same model is later backtested on forward market returns.
$Y_{\text{task}}$ (linguistic monetary policy stance) must be isolated from $Y_{\text{econ}}$ (forward price changes).

---

## 5. MANTRA's Hardened Scientific Positioning

MANTRA synthesizes verified insights from the literature while resolving the flaws above:
1. **Focus on Discriminative Encoders**: Eliminates prompt sensitivity, sampling temperature, and verbosity bias inherent to generative decoders.
2. **Clean / Leak Twin Architecture**: Evaluates controlled twins ($M_{\text{clean}}$ vs $M_{\text{leak\_dose}}$) holding capacity and fine-tuning invariant.
3. **Decoupled Three-Tier Hierarchy**:
   $$L_{\text{repr}} \ (\text{Probing}) \quad \text{and} \quad L_{\text{behavior}} \ (\text{Differential Masking}) \quad \longrightarrow \quad E_L \ (\text{Delta IC / Sharpe})$$
4. **No Arbitrary Composite Scores**: Eliminates artificial weighted sums ($0.5 L_{\text{repr}} + 0.5 L_{\text{mask}}$); preserves independent Pareto dimensions.
5. **Point-in-Time Safeguards**: Strict timestamp gating for audited sources; heuristic approximations are explicitly labeled as such.
