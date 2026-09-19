# FOMC Benchmark Specification & Point-in-Time Data Protocols

## 1. Scope & Objective

The FOMC Benchmark in MANTRA provides an isolated, temporally rigorous evaluation environment for:
1. Evaluating NLP Task Competence ($C$) on monetary policy stance classification.
2. Detecting Parametric Temporal Leakage ($L$) via representation probing and counterfactual masking.
3. Measuring Leakage-Induced Economic Effect ($E_L$) against asset markets (Treasury yields, Fed Funds futures, SPY).

---

## 2. Document Sources & Publication Latency Protocols

All documents must adhere to the **Point-in-Time Availability Inequality**:
$$\text{availability\_timestamp}(d) \le \text{simulation\_timestamp } t$$

| Document Type | Publication Mechanism | Historical Availability Timestamp ($t_{\text{avail}}$) | Allowed in Backtest at Meeting Date $T$? |
| :--- | :--- | :--- | :--- |
| **FOMC Statement** | Official Federal Reserve website release. | **14:00:00 EST on Meeting Day $T$** | **Yes** (Only for trades executed at/after 14:00 EST). |
| **Press Conference Opening Remarks** | Read by the Fed Chair at the start of the press conference. | **14:30:00 EST on Meeting Day $T$** | **Yes** (Only for trades executed at/after 14:30 EST). |
| **Press Conference Full Transcript** | Raw transcript of Q&A session. | **$\sim$ 20:00:00 EST on Meeting Day $T$** (or next morning). | **No** (Cannot be used for intraday trading on day $T$; available on day $T+1$). |
| **FOMC Minutes** | Detailed record of committee deliberations. | **14:00:00 EST exactly 21 calendar days post-meeting ($T+21\text{d}$)**. | **STRICTLY FORBIDDEN on Meeting Day $T$**. Visible only on $T+21$. |
| **Speeches & Testimony** | Delivered by individual Governors/Presidents. | Delivered at scheduled speech timestamp. | Only after verified speech delivery completion time. |

---

## 3. Temporal Train / Dev / Test Split

To ensure an uncompromised evaluation of pretraining cutoff vs. post-cutoff exposure:

* **Pre-Cutoff Training Split ($\mathcal{D}_{\text{train}}^{\le 2018}$)**:
  - Date Range: `1996-01-01` to `2018-12-31` (23 years).
  - Economic Regimes: 1990s expansion, 2000 Dot-com crash, 2001 recession, 2004–2006 rate hike cycle, 2007–2008 Global Financial Crisis, Zero Lower Bound (ZLB), 2015–2018 liftoff & balance sheet reduction.
  - Used for: Downstream task fine-tuning for all twin models ($M_{\text{clean}}$ and $M_{\text{leak}}$ twins).
* **Development / Validation Split ($\mathcal{D}_{\text{dev}}^{2019}$)**:
  - Date Range: `2019-01-01` to `2019-12-31`.
  - Economic Regimes: 2019 "mid-cycle adjustment" rate cuts (3 cuts).
  - Used for: Hyperparameter tuning, early stopping of classification heads.
* **Out-of-Sample Evaluation Split ($\mathcal{D}_{\text{test}}^{2020-2024}$)**:
  - Date Range: `2020-01-01` to `2024-12-31` (5 years).
  - Economic Regimes: COVID-19 pandemic shock (emergency cuts to 0%), 2021 inflation shock & "transitory" debate, 2022–2023 aggressive tightening (+525 bps), 2024 easing pivot.
  - Used for: Leakage detection ($L$), false alpha evaluation ($E_L$), and temporal decay ($R_T$).

---

## 4. Decoupled Label Architecture

To prevent circularity between NLP evaluation and economic evaluation, labels are divided into two distinct streams:

```
                  +-----------------------------------+
                  |        FOMC Text Sample (x_t)     |
                  +-----------------+-----------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+------------------------+                     +------------------------+
|  NLP Task Label: Y_task|                     | Forward Economic Target|
|  (Human Stance)        |                     | Y_econ (Market Reaction|
+------------------------+                     +------------------------+
| +1: Hawkish            |                     | 1. Fed Funds Surprise  |
|  0: Neutral            |                     | 2. \Delta 2Y Treasury  |
| -1: Dovish             |                     | 3. SPY 1d/5d/20d Return|
+------------------------+                     +------------------------+
            |                                               |
            v                                               v
Evaluates Task Competence (C)                 Evaluates Leakage & Alpha
                                              - Probing -> L_repr
                                              - Trading -> E_L
```

### 4.1 Task Label ($Y_{\text{task}}$): Linguistic Stance
We adopt the sentence- and statement-level human annotations from *Trillion Dollar Words* (Shah et al., ACL 2023):
- **Hawkish ($+1$)**: Explicit emphasis on inflation risks, overheating, tightening policy, raising rates, tapering asset purchases.
- **Neutral ($0$)**: Balanced descriptions of current conditions, factual repetitions without forward bias.
- **Dovish ($-1$)**: Explicit emphasis on downside risks to employment/growth, economic slack, keeping accommodation, lowering rates.

### 4.2 Economic Outcome Stream ($Y_{\text{econ}}$): Market Reaction Targets
Target variables measured over forward windows:
1. **Monetary Policy Surprise ($\Delta s_t^{\text{policy}}$)**:
   - Fed Funds futures surprise at 14:00 EST / 14:30 EST.
2. **2-Year US Treasury Yield Response ($\Delta y_t^{2Y}$)**:
   $$\Delta y_{t, t+1}^{2Y} = y_{t+1}^{2Y} - y_{t-1}^{2Y}$$
3. **Equities Reaction ($r_{t, t+k}^{\text{SPY}}$)**:
   $$r_{t, t+k} = \frac{\text{SPY}_{t+k} - \text{SPY}_t}{\text{SPY}_t}, \quad k \in \{1, 5, 20\} \text{ trading days}$$

---

## 5. Temporal Leakage Probing Targets

For the representation probing module ($L_{\text{repr}}$), the frozen encoder representation $h_\theta(x_t)$ is evaluated on its ability to decode:
1. `target_next_action`: Direction of policy change at meeting $t+1$ ($\text{Hike} = +1, \text{Hold} = 0, \text{Cut} = -1$).
2. `target_cpi_surprise`: Realized headline CPI surprise at next release ($\text{Actual} - \text{Consensus}$).
3. `target_market_regime`: Whether 30-day SPY realized volatility is high/low.

A model with **parametric leakage** will achieve high probe accuracy on `target_next_action` even when the text $x_t$ contains no explicit forward guidance about meeting $t+1$.
