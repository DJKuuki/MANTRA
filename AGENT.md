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
**Phase 3 Pilot Finalized — Pilot Pipeline Valid & Low-Power Empirical Metrics Evaluated**
（完成 Phase 3 — Pilot Temporal Leakage Dose-Response Study：在 Commit A `e5a6a96fa51faf7c14307451476d538973c1dc28` 代码冻结基线与完全洁净的工作树下，执行 15 组因果分支实验（3 seeds $\times$ 5 doses: 0.0, 0.25, 0.50, 0.75, 1.00）；实现严格的 Exact Token Dose Mixer $|D_{\text{realized}} - D_{\text{requested}}| \le 1/T$ 与全因果对齐；正式引入 2019 年美联储官方 8 次决议的 25 条段落级 Point-in-Time Leakage Anchors；实现时间前向严格隔离与重合度严格为 0；动态计算全部 5 项实证指标 $C, R_T, L_{\text{repr}}, L_{\text{behavior}}, E_L$；明确标注 **LOW POWER PILOT — INSUFFICIENT INDEPENDENT EVENTS FOR CONFIRMATORY INFERENCE**；状态评定：**PHASE 3 PILOT PIPELINE VALID**，**DOSE-RESPONSE LADDER EVALUATED**，**PILOT POWER BOUNDED**）。
严禁在样本量不足（仅 8 次独立决议）的 Pilot 阶段过早宣称确证性结论；严禁在进入 Phase 4 前破坏方法学冻结。

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

1. **Counterfactual Entity / Date Masking ($L_{\text{behavior}}$)**：
   * Level 1: Federal Reserve → Central Bank A (央行机构)
   * Level 2: Powell → Person A (官员姓名)
   * Level 3: 消除具体日历年份与日期
   * **核心原则：单个模型的掩码敏感度 $S_{\text{mask}}(M)$ 绝不是泄漏**。正常 clean 模型合法利用时序实体必然产生敏感度。
   * **严格定义为双生差分**：
     $$L_{\text{behavior}} = S_{\text{mask}}(M_L) - S_{\text{mask}}(M_C)$$
     $$L_{\text{entity}} = S_{\text{entity}}(M_L) - S_{\text{entity}}(M_C), \quad L_{\text{date}} = S_{\text{date}}(M_L) - S_{\text{date}}(M_C)$$
     当 $M_L = M_C$ 时，$L_{\text{behavior}} \equiv 0.0$。
2. **Representation Probing ($L_{\text{repr}}$)**：
   * 冻结 $h_\theta(x_t)$，采用 `TimeSeriesSplit` 拓展窗口交叉验证训练线性探针，预测未来宏观/市场变量（next hike/cut, future CPI surprise, SPY return）。
   * 双生配对差分与置换检验：
     $$L_{\text{repr}} = \frac{1}{K}\sum_{k=1}^K \left(\text{Probe}_L(k) - \text{Probe}_C(k)\right)$$
     配合成对符号翻转置换检验（Paired sign-flip permutation test, $M \ge 500$）。

---

## 11. 理论分层：Representation $\to$ Behavior $\to$ Economics

$$L_{\text{repr}} \ (\text{探针差分}) \quad \text{and} \quad L_{\text{behavior}} \ (\text{掩码敏感度差分}) \quad \longrightarrow \quad E_L \ (\Delta \text{IC} \text{ 与 } \Delta \text{Sharpe})$$
* **Level 1 (Representational Leakage)**：未来真实信息是否存在于 hidden representation $h_\theta(x)$。
* **Level 2 (Behavioral Leakage)**：未来信息是否使模型对时序锚点产生非正常依赖而改变下游预测 $f_\theta(x)$。
* **Level 3 (Economic Leakage)**：
  - **Level A (Primary)**：模型与策略无关的 $\Delta \text{IC} = \text{IC}(M_L) - \text{IC}(M_C)$，基于连续立场得分与未来收益率的 Spearman 秩相关。
  - **Level B (Secondary / Illustrative)**：固定规则与固定阈值下的 $\Delta \text{Sharpe} = \text{Sharpe}(M_L) - \text{Sharpe}(M_C)$，配合 Politis & Romano (1994) 平稳块 Bootstrap 检验。

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

## 14-18. 发现的显式数据泄露与严苛 PIT 隔离落地状态

必须先消除 External / Pipeline Leakage，才能研究 Parametric Temporal Leakage。当前落地状态如下：
1. **Python API `run_backtest` Cache 初始化**：**已修复**。回测启动时显式调用 `_bt_cache.initialize()`，防止回退至实时网络请求。
2. **Real-Time Fundamentals 泄露拦截**：**已修复**。yfinance `ticker.info` 截面数据在回测模式下被彻底拦截（返回 `[Backtest] Fundamentals overview withheld`）。
3. **Financial Statements 财报时间截断与 Freq 路由**：**已修复**。
   - `approximate_availability_filter` 严格区分季报 10-Q (45d) 与年报 10-K (90d) 法定披露滞后。
   - `BacktestDataCache` 与 `y_finance.py` 中遗漏的 `freq` 参数路由已彻底补齐，杜绝年报错误回退至 45d 的 bug。
4. **Alpha Vantage 财报过滤**：**已加固**。优先按真实披露时间戳 `reportedDate` / `filingDate` 过滤；无发布日记录才使用 45d/90d 启发式安全滞后。
5. **SEC Insider Transactions**：**已加固**。强制基于 Form 4 `Filing Date` 过滤，严禁使用 `Transaction Date`。
6. **PIT 凭证追踪与隔离 (Provenance Tracking)**：**已落地**。`TemporalSample` 原生支持 `availability_source` 与 `availability_quality` ("exact" vs "heuristic")，正式 Benchmark 支持 `exact_only` 子集提取以进行稳健性对照。

---

## 19-20. 核心区分：Pipeline Leakage vs. Parametric Leakage

* **Pipeline Leakage**：数据工程缺陷，未来信息通过输入特征进入模型（当前基础设施已建立严格 Gate 拦截）。
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
* **合成双生模型状态**：`SyntheticTemporalTwinEncoder` 已落地，基于 SHA-256 纯函数确定性特征与外生标签受控注入，剂量单调性检验通过；**真实 Encoder Baseline 与受控双生管线已就绪**（Phase 2 完成）。

---

## 28-30. 核心禁忌与原则

* 严禁将 cutoff 后 F1 下降直接称为 leakage。
* 严禁将高 Sharpe 直接归结为 false alpha。
* 严禁将 low leakage 视同模型优秀。
* 严禁使用非 point-in-time 的数据。
* 严禁让交易收益参与 NLP 标签构造。

---

## 31. 当前工作进展与下一阶段规划

* **Task 1 — Literature Review**：**已完成并建立机器可校验文献表**（`docs/research/literature_registry.json`，修正所有已知 arXiv / DOI 错误，代码仓映射至 `gtfintechlab/fomc-hawkish-dovish`）。
* **Task 2 — Formalize Variables**：**已完成形式化规范**（五维 Pareto 空间，解耦 Level A $\Delta \text{IC}$ 与 Level B $\Delta \text{Sharpe}$，平稳块 Bootstrap）。
* **Task 3 — Design FOMC Benchmark & PIT Protocols**：**已完成并加固**（`FOMCBenchmark` 禁止静默加载 Toy 数据；Loader 实施 Fail-Fast 校验；入库时间戳规范化为 UTC ISO；时区统一为 `America/New_York` / UTC）。
* **Task 4 — Synthetic Twin Validation**：**已完成**（无状态确定性合成双生模型，方法学指标已完全冻结）。
* **Task 5 — Pre-Experiment Gate & Final Hardening**：**已冻结 (v1.0)**。
* **Phase 2.1 — Causal Twin Activation & Baseline Correction (Finalized)**：**已全面收敛验证 (PHASE 2.1 FINALIZED / BASELINE PROTOCOL VALID / CAUSAL TWIN PIPELINE ACTIVE)**：
  - **任务语义修正与独立基线**：剥离 `ProsusAI/finbert` 原始情感头，新建专有 3-class FOMC Stance 分类头（`Dovish`/`Neutral`/`Hawkish`）；解耦基线与烟测样本切片，在全部 1,729 篇 Pre-cutoff 数据（$\le 2018$）上微调构建真正立场基线 $M_B$（Macro-F1: 0.5073，MCC: +0.3048，Brier: 0.6053，ECE: 0.1846，混淆矩阵分布均衡）。
  - **分类头结构解耦与哈希固化**：实现 `build_fresh_fomc_classifier_from_base_encoder`，显式追踪 `original_head_loaded: false` 与 `stance_head_initial_hash`，彻底消除分类头污染与随机重置。
  - **动态 Git 凭证与洁净断言**：实现 `resolve_git_provenance` 动态探测真实 commit SHA，运用 `:(top,exclude)experiments` 路径规范精准识别代码树洁净度，运行实测保证 `git_dirty: false` 与 `code_commit_exact: true`。
  - **Trillion Dollar Words 时间分辨率审计**：元数据补充标注 `temporal_resolution: "year"`、`timestamp_imputed: True` 与 `timestamp_imputation_rule: "mid_year_placeholder"`。
  - **Token-Budget 算力对齐**：实现 `create_token_matched_dose_stream`，以 128 长度定长 token blocks 进行双生封装，保证 $D_0$ 与 $D_{100}$ 获得精确相等的 token 算力（2,560 tokens，0% 差异，10 步对称优化）。
  - **真实 MLM Temporal Treatment 激活**：端到端连通 `run_continued_pretraining_mlm`，保证 $M_C$ 与 $M_L$ 经历独立且真实的 temporal MLM 预训练；断言验证 MLM 参数真正发散（$\mathrm{Hash}(M_C) \neq \mathrm{Hash}(M_L)$）。
  - **权重与分类头传递**：实现 `build_classifier_from_mlm_encoder`，将 MLM 训练后 BERT 权重注入分类器，并绑定 bit-identical 的初始分类头权重。
  - **评估隔离（Treatment A）**：句子级哈希过滤杜绝评测集进入 MLM 污染池（Overlap = 0）。
  - **Schema 统一与实证留白**：待测实证经济指标置为 `empirical_leakage_metrics: null` 与 `empirical_pareto_vector: null`，烟测试跑指标统一收口至 `synthetic_plumbing_pareto_vector`。
  - **Official Fixture 降级**：`fomc_official.py` 硬编码样例正式降级为测试 fixture，默认 `is_formal_research_ready() == False`；正式 benchmark 强制要求显式文件或 manifest。
  - **测试防护**：126 个单元/集成测试 100% 通过（新增 Phase 2.1 Finalization 专项测试 A~H），CI 完全离线无模型下载负担。
* **Phase 3 — Pilot Temporal Leakage Dose-Response Study (Finalized)**：**全因果双生管线验证通过 / 实证指标初次评测完成 (PHASE 3 PILOT PIPELINE VALID / LOW POWER PILOT)**：
  - **严格双提交协议**：
    - Commit A (Code Freeze): `e5a6a96fa51faf7c14307451476d538973c1dc28`（完全洁净代码树 `git_dirty: false`, `code_commit_exact: true`, tree hash `8badd580e041fb2716b356ed36a51624ae23ee35`）；
    - 运行 15 组实验并落地全部分支 manifests 与汇总 JSON；
    - Commit B (Artifacts): 仅含 manifests、`phase3_pilot_results.json`、报告与本文件（`4ecd3c970be1699e40f21d8b5a9f5a8e42843d96`）；
    - CI 跨平台兼容加固：引入 `.gitattributes` 强制文本 LF 规范化，并在 `fomc_benchmark.py` 落地 `verify_file_sha256` 抹平跨系统换行符哈希差异。
  - **全对称因果参数锁定**：15 组分支（3 seeds: 13, 42, 73 $\times$ 5 doses: 0.0, 0.25, 0.50, 0.75, 1.00）在同一 seed 下保证：相同 base checkpoint 初始权重哈希、精确相等的 token 预算（25,600 tokens = 200 blocks $\times$ 128）、bit-identical 的 deterministic mask schedule 哈希、相同初始下游分类头哈希、相同下游样本序列哈希。
  - **Exact Token Mixer 剂量控制**：实现 `create_exact_token_dose_stream`，严格满足 $|D_{\text{realized}} - D_{\text{requested}}| = 0.0000 \le 1/T$。
  - **官方 Point-in-Time Leakage Anchors**：录入 2019 年美联储官方全部 8 次决议的 25 条段落级样本（`data/research/fomc/leakage_anchors/`，SHA256: `177af152a12...`），100% 具备法定公布时刻 exact UTC 时间戳，挂载前向政策行动标签 $Y_{\text{future-action}}$ 与市场前向反应（SPY 收益与美债 2Y 收益率变动）。
  - **零污染时空隔离**：时间隔离 $\max(Time_{\text{anchors}}) = 2019\text{-}12\text{-}11 < 2020\text{-}01\text{-}01 = \min(Time_{\text{contamination}})$，缓冲期 20 天；文档重合度严格为 0；与 post-cutoff 污染语料的句子哈希重合度严格为 0。
  - **实证指标全量评测**：全部 5 类实证指标首次真实计算，杜绝合成占位：
    - $C(D)$ (Competence Macro-F1): $0.5752 \sim 0.5890$（跨剂量保持稳定，证实 MLM 未引发灾难性遗忘或领域崩溃）；
    - $R_T(D)$ (Temporal Robustness): $-0.0621 \sim -0.1002$；
    - $L_{\text{repr}}(D)$ (Representation Leakage): 除 seed 73 $D=0.75$ 出现 $-0.1016$ 外，其余在小预算下接近 0；
    - $L_{\text{behavior}}(D)$ (Behavioral Leakage): $0 \sim +9.07\times 10^{-5}$，Spearman $\rho = +0.4000$；
    - $E_L(D)$ (Economic Leakage IC): $0 \sim +0.0420$，Spearman $\rho = +0.3000$。
  - **统计效力与推断边界**：明确标注 **LOW POWER PILOT — INSUFFICIENT INDEPENDENT EVENTS FOR CONFIRMATORY INFERENCE**。2019 年仅 8 次独立决议，样本量不足以支撑确证性因果推断。单调性仅作为客观实证观测，严禁作为优化目标。
* **Next Phase (Phase 4)**：**Confirmatory Full Study**（准入条件：扩充 Leakage Anchors 至 $\ge 40$ 次独立政策决议与 $\ge 200$ 条段落样本；扩展 MLM token 算力预算至 256k ~ 1M tokens；引入多架构交叉复现与高频日内 Rate Surprise 响应）。

---

## 32-37. 核心哲学

> **研究的核心不是“这个模型在历史回测里表现多好？”，而是“模型表现中有多少来自真正的任务能力，有多少来自 post-cutoff information，而其中又有多少 post-cutoff information 被转化成了经济上可利用的 false alpha？”**
