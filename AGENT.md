# AGENT.md

## 1. 项目目标

本项目基于：
* Repository: `https://github.com/RubiscoYHY/MANTRA`
* 项目名：**MANTRA — Memory-Augmented Neural Trading Retrieval Agents**

当前研究重点不是继续扩展 MANTRA 的通用 Trading Agent 能力，而是将其逐步改造成一个用于研究：
> **金融 NLP 模型中的 temporal data leakage，以及这种 leakage 是否会制造 false alpha**
的实验框架。

研究对象计划优先限定为：
* **Encoder-based financial language models**（例如 BERT, FinBERT, RoBERTa, DeBERTa），而不是 decoder / generative LLM。

当前优先示例任务：
* **FOMC hawkish / neutral / dovish classification**

后续可以扩展到：
* central-bank sentiment classification
* monetary-policy stance classification
* earnings sentiment
* financial-event classification
* event-driven market prediction

本项目目前处于：
**研究问题定义 + 文献调研 + 实验框架设计阶段**
不要急于训练大模型，也不要优先扩展 MANTRA 的 Agent 数量、Prompt 或 UI。

---

## 2. 当前研究问题的精确定义

最初问题：*LLM 在历史回测中使用了多少未来信息？*
目前进一步缩小并修正为：
> **How can temporal information leakage be detected and quantified in discriminative financial language encoders, and how can its economic consequences be separated from genuine task competence?**

研究必须明确区分至少三个不同维度。

---

## 3. 三个核心研究变量

整个研究禁止试图用一个单一分数描述模型好坏。需要至少保留以下三个近似正交的变量：

### 3.1 Temporal Leakage ($L$)
* **定义**：模型的 representation 或输出在多大程度上依赖于决策时点之后才产生的信息。
* **命名建议**：Temporal Information Leakage / Post-cutoff Information Dependence / Parametric Temporal Leakage。
* **核心警示**：不要简单把“cutoff 后性能下降”定义成 leakage。
  $$\text{Post-cutoff degradation} \neq \text{Temporal leakage}$$
  模型在 cutoff 后性能下降，也可能只是 $P_t(X,Y)$ 随时间变化（temporal distribution shift, concept drift, lexical drift, regime change）。
  例如 FOMC 文本中的 inflation, QE, taper, supply chain, soft landing, COVID, AI 在不同时期语义完全不同。
  因此 cutoff 后降智只能作为 temporal robustness / generalization 指标，不应直接作为 leakage 指标。

### 3.2 Leakage-Induced Economic Effect ($E_L$)
* **定义**：Temporal leakage 中真正被模型转化成金融预测优势或回测收益的部分。
* **与 $L$ 的解耦**：
  * 可能 $L > 0$ 但 $E_L \approx 0$（模型记住了未来事实，但未影响分类或策略）。
  * 也可能 $L$ 很小但信息关键（如 surprise hike, emergency cut, CPI surprise, policy pivot），导致 $E_L \gg 0$。
* **度量方式**：干净模型和污染模型之间的差分：
  $$E_L = \text{Performance}(M_{\text{contaminated}}) - \text{Performance}(M_{\text{clean}})$$
  具体可测：$\Delta \text{Sharpe}$, $\Delta \text{IC}$, $\Delta \text{Directional Accuracy}$, $\Delta \text{Long-Short Return}$, $\Delta \text{Hit Rate}$, $\Delta \text{CER}$。
  不要直接把原始 Sharpe 当成 leakage 的经济影响。
  $$\alpha_{\text{leak}} = \alpha(M_L) - \alpha(M_C)$$
  $$\alpha_{\text{observed}} = \alpha_{\text{legitimate}} + \alpha_{\text{leak}} + \epsilon$$

### 3.3 Task Competence ($C$)
* **定义**：模型完成目标金融 NLP 任务本身的真实能力。
* **当前优先任务**：FOMC hawkish / neutral / dovish classification。
* **推荐指标**：Macro-F1, MCC, AUROC, Brier Score, ECE, confusion matrix。
* **核心警示**：不要只看 accuracy，因为存在严重的 class imbalance（如永远输出 Neutral 的平庸模型）。

---

## 4. “恒定输出模型”是当前框架的重要反例

必须保留这个思想实验：
假设 $M_0(x) = 0, \forall x$（恒定输出同一类别或 score）：
* $L(M_0) = 0$
* $E_L(M_0) = 0$
* 但 $C(M_0) \approx 0$

**“没有 leakage”不等于“模型很好”**。这也是为什么不能把 leakage, false alpha, competence 压缩成单一指标。

---

## 5. 不建议强行构造总分，使用 Pareto Frontier

暂时不要设计 $Score = aC - bL - cE_L$：
* 权重任意且主观
* 抹杀应用风险偏好差异与 trade-off
* 容易产生无解释力的 composite metric

**推荐使用 Pareto Frontier**：
评估模型在三维空间 $(L, E_L, C)$ 中的位置。
理想模型：$L \to 0, E_L \to 0, C \to 1$。
推荐可视化：
1. Leakage vs Task Competence
2. Leakage vs False Alpha
3. Task Competence vs False Alpha
4. 3D Pareto plot

---

## 6. 第四个辅助指标：Temporal Robustness ($R_T$)

记录跨时分布变化后的性能衰减（作为控制量）：
$$D(\Delta t) = C(t_{\text{cutoff}} + \Delta t) - C(t_{\text{cutoff}})$$
主变量体系：
* $L$：Temporal Leakage
* $E_L$：Leakage-Induced Economic Effect
* $C$：Task Competence
* $R_T$：Temporal Robustness（控制量）

---

## 7. 为什么研究 Encoder，而不是 Decoder

聚焦于 Encoder-based models（BERT, RoBERTa, FinBERT, DeBERTa）：
* Decoder 模型会引入大量混杂因素：prompt sensitivity, decoding randomness, CoT, tool use, hallucination, instruction following, RLHF, verbosity bias。
* Encoder 将系统简化为：$x \to h_\theta(x) \to \text{classifier} \to y$，能够精确研究 future information 是否已被编码进 representation。

---

## 8. 推荐的主要实验路线：Clean / Contaminated Twin Models

建立两个架构、tokenizer、微调设置、超参数、分类头完全一致的模型：
* $M_{\text{clean}}$：预训练 / continued pretraining 仅接触 $\le 2018-12-31$ 数据。
* $M_{\text{leak}}$：从 clean checkpoint 继续在 2019–2022 金融语料上预训练。
* 两个模型在完全相同的一致目标任务数据上微调。
* $\Delta_L = M_{\text{leak}} - M_{\text{clean}}$ 提供严格的因果估计（Causal Estimate）。

---

## 9. Leakage Dose 实验

不仅做 clean/leak 两点，设计污染剂量：
$$D \in \{0\%, 25\%, 50\%, 75\%, 100\%\}$$
度量 $L(D), E_L(D), C(D)$，重点研究边缘效应：
$$\frac{dE_L}{dL}$$
探索少量 post-cutoff contamination 是否足以制造大量 false alpha。

---

## 10. Encoder Leakage 检测方法

1. **Counterfactual Entity Masking**：
   * Level 0: 原始文本
   * Level 1: Federal Reserve → Central Bank A
   * Level 2: Powell → Person A
   * Level 3: 消除具体年份与日期
   * Level 4: 完全实体 + 日期匿名化
   * 测定 $P(y|x_{\text{orig}})$ 与 $P(y|x_{\text{anon}})$ 的偏离，分解 $L_{\text{entity}}, L_{\text{date}}, L_{\text{event}}$。
2. **Counterfactual Historical Editing**：
   * 构造与真实历史相反的 continuation（如真实为加息，反事实为降息），观察 logit/embedding movement, classifier margin, representation shift。
3. **Representation Probing**：
   * 冻结 $h_\theta(x_t)$，训练 probe 预测未来宏观/市场变量（next hike/cut, future CPI surprise, SPY return）。
   * 必须比较 $\Delta \text{Predictability} = \text{Pred}(M_{\text{leak}}) - \text{Pred}(M_{\text{clean}})$。

---

## 11. 理论分层：Representation $\to$ Behavior $\to$ Economics

$$L_{\text{repr}} \longrightarrow L_{\text{behavior}} \longrightarrow E_L$$
* **Level 1 (Representational Leakage)**：未来信息是否存在于 hidden representation $h_\theta(x)$。
* **Level 2 (Behavioral Leakage)**：未来信息是否改变下游任务预测 $f_\theta(x)$。
* **Level 3 (Economic Leakage)**：行为改变是否最终产生虚假收益 $\alpha_{\text{false}}$。

---

## 12. 因果结构图

```
Post-cutoff training exposure
             |
             v
      Encoder representation
          /        \
         /          \
        v            v
Task competence   Temporal leakage
        |             |
        |             v
        |       Financial signal
        |             |
        +-------------+
              |
              v
        Observed alpha
```

---

## 13. MANTRA 代码库现状与定位

MANTRA (`RubiscoYHY/MANTRA`) 原生为 multi-agent decoder 交易框架。
在当前研究中，MANTRA 定位为：
* **Point-in-Time Data Pipeline**
* **Backtesting Framework**
* **Economic Effect Evaluation Harness**
* **Future Information Leakage Testbed**

---

## 14-18. 发现的显式数据泄露与严苛 PIT 隔离需求

必须先消除 External / Pipeline Leakage，才能研究 Parametric Temporal Leakage：
1. **Python API `run_backtest` 未初始化 Backtest Cache**：缺少 `_bt_cache.initialize()`，可能退化为请求实时数据。
2. **Real-Time Fundamentals 泄露**：yfinance `ticker.info` 中的 PE, market cap, TTM metrics 为抓取时点数据；历史回测最后一天重新暴露的问题必须彻底修复。
3. **Financial Statements 财报时间截断**：必须使用 `availability_timestamp <= t`（如 10-Q/10-K filing date），禁止使用 `fiscal_period_end <= t`。
4. **Insider Transactions**：必须基于 SEC Form 4 filing publication date，而非 transaction date。

---

## 19-20. 核心区分：Pipeline Leakage vs. Parametric Leakage

* **Pipeline Leakage**：数据工程缺陷，未来信息通过输入特征进入模型。
* **Parametric Leakage**：输入完全干净，但模型参数在预训练中记住了未来。
* **前提**：后续所有实验必须在 Pipeline Leakage 严格为 0 的受控环境下进行。

---

## 21-23. 任务与经济表现的解耦

* FOMC 任务具备时间戳精准、文本和市场反应解耦、macro regime shift 明显等优良性质。
* Hawkish / Dovish 标签构造必须与市场反应分离，避免循环论证（如用后续价格波动做 NLP 标签，再拿该预测测回测收益）。
* 明确区分 $C$（NLP 分类表现）与 $E_L$（交易策略收益）。

---

## 24-27. 模型选型与能力控制

* 避免 BERT 2018 vs DeBERTa 2024 等跨架构、跨容量的无效对比。
* 控制单一变量：相同 base checkpoint + 受控的 post-cutoff exposure（Continued Pretraining）。
* 不从零训练大模型，优先利用公开时间 cutoff 的模型或构建受控 Twins。

---

## 28-30. 核心禁忌与原则

* 严禁将 cutoff 后 F1 下降直接称为 leakage。
* 严禁将高 Sharpe 直接归结为 false alpha。
* 严禁将 low leakage 视同模型优秀。
* 严禁使用非 point-in-time 的数据。
* 严禁让交易收益参与 NLP 标签构造。

---

## 31. 近期工作规划

* **Task 1 — Literature Review**：建立 10+ 篇核心论文的精细文献矩阵（模型、泄露定义、度量、可复现性验证）。
* **Task 2 — Formalize Variables**：严格数学与操作化定义 $L, E_L, C, R_T$，确立零模型与病态案例检验。
* **Task 3 — Design FOMC Benchmark**：设计严格 PIT 的 FOMC 数据集规约、标签体系与回测协议。
* **Task 4 — Design Twin Model Experiment**：设计受控剂量预训练与微调流程。
* **Task 5 — Modify MANTRA Only After Benchmark Definition**：对 MANTRA 底层 PIT 与回测层进行受控适配。

---

## 32-37. 核心哲学

> **研究的核心不是“这个模型在历史回测里表现多好？”，而是“模型表现中有多少来自真正的任务能力，有多少来自 post-cutoff information，而其中又有多少 post-cutoff information 被转化成了经济上可利用的 false alpha？”**
