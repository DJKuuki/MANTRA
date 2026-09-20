# Phase 4A Confirmatory Data & Preregistration Gate Audit Report

**Stage**: Phase 4A — Confirmatory Data & Preregistration Gate  
**Repository**: `RubiscoYHY/MANTRA`  
**Evaluation Date**: 2026-09-20  
**Commit A SHA**: `d439c256b4e29b9abe05f93f9085edb7639cce66`  
**Preregistration Doc SHA-256**: `a03f9321432e1d8d676c1217391169b2dffa25b4f8c53953ea4de0dbb1111dac`  
**Gate Verdict**: **PHASE 4A PREREGISTRATION GATE PASSED / CONFIRMATORY DESIGN LOCKED / READY FOR PHASE 4B EXECUTION**  

---

## 1. Gate Execution Summary

The Phase 4A Preregistration Gate establishes absolute machine-verifiability for the empirical data, point-in-time provenance, statistical analysis plan, and compute execution boundaries of the MANTRA confirmatory evaluation.

All 12 gate criteria (Tests A through L in `tests/test_phase4a_gate.py`) and all 133 pre-existing repository unit tests (total 145 tests) pass with $100\%$ green status.

```
============================= test session starts =============================
platform win32 -- Python 3.13.4, pytest-9.0.3, pluggy-1.6.0
rootdir: E:\MANTRA
configfile: pyproject.toml
collected 145 items

tests\test_backtest_eval.py ........                                     [  5%]
tests\test_backtest_memory_wiring.py .                                   [  6%]
tests\test_causal_twin_activation.py ..........                          [ 13%]
tests\test_filing_store.py .........                                     [ 19%]
tests\test_fomc_benchmark_loader.py ...                                  [ 21%]
tests\test_google_api_key.py .                                           [ 22%]
tests\test_gui_app.py ...............                                    [ 32%]
tests\test_memory_store.py ........                                      [ 37%]
tests\test_model_validation.py ...                                       [ 40%]
tests\test_news_dates.py ....                                            [ 42%]
tests\test_null_model_properties.py .                                    [ 43%]
tests\test_observability.py .......                                      [ 48%]
tests\test_phase2_1_finalization.py ........                             [ 53%]
tests\test_phase3_pilot.py .......                                       [ 58%]
tests\test_phase4a_gate.py ............                                  [ 66%]
tests\test_pit_integrity.py .....                                        [ 70%]
tests\test_pre_experiment_gate.py .................                      [ 82%]
tests\test_real_encoder_phase2.py ..........                             [ 88%]
tests\test_social_filtering.py .......                                   [ 93%]
tests\test_temporal_leakage_metrics.py .......                           [ 98%]
tests\test_ticker_symbol_handling.py ..                                  [100%]

============================ 145 passed in 48.05s =============================
```

---

## 2. Point-in-Time Provenance & Data Architecture Audit

### 2.1 Dataset Expansion & Empirical Balance
- **Historical Sample Range**: 2015-01-01 to 2019-12-31 (5 full calendar years).
- **Independent Decision Events ($N$)**: Exactly 40 scheduled FOMC meetings ($N=40 \ge 40$).
- **Paragraph Anchors ($M$)**: Exactly 181 canonical paragraphs ($M=181 \ge 160$).
- **Macroeconomic Regime Coverage**:
  - Holds ($0$): 28 meetings ($70.0\%$)
  - Rate Hikes ($+1$): 9 meetings ($22.5\%$)
  - Rate Cuts ($-1$): 3 meetings ($7.5\%$)
- **Statistical Unit Guarantee**: Every meeting functions as an atomic unit. Within-event paragraph representations are aggregated to event centroids $h_{\text{event}} = \frac{1}{M}\sum_{i=1}^M h_i$.

### 2.2 Verifiable Canonical Text Provenance
- Every anchor text in `data/research/fomc/confirmatory_anchors/anchors.jsonl` was scraped from official Federal Reserve press releases (`federalreserve.gov/newsevents/pressreleases/monetary*.htm`).
- All 40 raw HTML pages are cached under `data/research/fomc/raw_sources/`.
- Every HTML file has a verified SHA-256 hash recorded in `data/research/fomc/source_registry.jsonl`.
- Automated test `test_e_anchor_canonical_source_existence` verifies that anchor text exists verbatim in the corresponding official HTML document.

### 2.3 Policy History & Deterministic Target Engine
- `data/research/fomc/policy_history.csv` records all target rate decisions from 2014-12 to 2020-03 based on official Federal Reserve Open Market Operations records.
- `tradingagents/temporal_leakage/datasets/policy_history.py` programmatically derives future targets.
- Automated test `test_f_policy_future_action_provenance` confirms $100\%$ match between derived targets and event metadata without manual overrides.

### 2.4 Market Outcomes Provenance & Daily Resolution
- `data/research/market/spy_daily_raw.csv` contains 1,300 daily rows of SPY ETF pricing.
- `data/research/market/treasury_2y_raw.csv` contains 1,350 daily rows of 2Y Treasury data (combining CME `ZT=F` futures and FRED `DGS2` constant maturity yields).
- Automated test `test_g_market_outcome_recomputation` proves that `recompute_market_outcomes()` regenerates every market return and yield change bit-exact from raw daily tables.
- All market variables are labeled strictly as **daily-resolution post-event responses**, not high-frequency intraday surprises.

### 2.5 Temporal Separation & Contamination Timeline
- Pre-cutoff Anchor Horizon: Ends at `2019-12-11T19:00:00Z`.
- Contamination Treatment Stream: Begins at `2020-01-29T19:00:00Z`.
- Temporal Separation Buffer: **48 calendar days** ($\max(T_{\text{anchors}}) < \min(T_{\text{contamination}})$).
- Emergency Unscheduled Meetings: 2020-03-03 and 2020-03-15 are cataloged in `exclusion_log.jsonl` with explicit exclusion justifications.
- Format Verification: All timestamps are verified ISO-8601 UTC with exact release times (18:00–20:00 UTC); midnight placeholders (`T00:00:00Z`) are strictly rejected.

---

## 3. Methodological Upgrades & Statistical Protections

1. **Grouped Expanding-Window Temporal Cross-Validation**:
   - Evaluated across 4 expanding folds.
   - Enforces group disjointness: $\text{TrainEvents} \cap \text{TestEvents} = \emptyset$.
   - Enforces strict temporal monotonicity: $\max(\text{TrainTime}) < \min(\text{TestTime})$.
2. **Event-Clustered Clustered Bootstrap**:
   - Resamples whole FOMC meetings together with replacement.
   - Preserves intra-meeting correlation structure and eliminates pseudo-replication.
3. **Deterministic Token Sampling & Treatment Block Manifest**:
   - `create_exact_token_dose_stream` implements seeded permutation of source documents prior to packing, eliminating front-of-corpus accumulation bias.
   - Tracks `pre_tokens`, `post_tokens`, and source document IDs for every packed block in `treatment_block_manifest`.
   - Records `unique_source_tokens` and `repetition_ratio`, enforcing a strict tolerance guard against un-tracked repetition.
   - Guarantees $|D_{\text{realized}} - D_{\text{requested}}| \le 1/T$ across all 5 dose levels.

---

## 4. Preregistration Lock & Compute Containment

- **Preregistration Document**: `docs/research/phase4_preregistration.md`
- **Normalized SHA-256 Hash**: `a03f9321432e1d8d676c1217391169b2dffa25b4f8c53953ea4de0dbb1111dac`
- **Lock Configuration**: `configs/phase4_preregistration.yaml` (`locked: true`)
- **Confirmatory Execution Engine**: `tradingagents/temporal_leakage/phase4_confirmatory.py`
  - In Phase 4A, running full training (`smoke_mode=False`) unconditionally raises `PreregistrationLockError`.
  - Tampering with `docs/research/phase4_preregistration.md` triggers `PreregistrationHashMismatchError`.
  - Verifies that no expensive GPU compute (256k token continued pretraining, 10 seeds, multi-architecture) can be accidentally initiated prior to preregistration lock approval.

---

## 5. Gate Checklist Verification Matrix

| Gate Test | Item / Guideline | Verification Condition | Status |
| :--- | :--- | :--- | :--- |
| **Test A** | Item 23 | Within-event label, outcome, and timestamp consistency | **PASS** |
| **Test B** | Item 25 | Grouped expanding-window CV ($\text{Train} \cap \text{Test} = \emptyset$, time-ordered) | **PASS** |
| **Test C** | Item 26 | Event-level economic effect (each event counts once in IC) | **PASS** |
| **Test D** | Item 27 | Event-clustered bootstrap (meetings resampled as unified blocks) | **PASS** |
| **Test E** | Item 28 | Verbatim canonical existence in official Federal Reserve HTML snapshots | **PASS** |
| **Test F** | Item 28 | Policy `future_action` derived deterministically from policy history | **PASS** |
| **Test G** | Item 28 | Market outcomes recomputed bit-exact from raw SPY & 2Y tables | **PASS** |
| **Test H** | Item 29 | Actual contamination timeline provenance & 48-day temporal gap | **PASS** |
| **Test I** | Item 30 | Treatment block manifest & token provenance tracking | **PASS** |
| **Test J** | Item 30 | Exact dose ladder invariant $|D_{\text{realized}} - D_{\text{requested}}| \le 1/T$ | **PASS** |
| **Test K** | Item 29 | Prohibition of mid-year placeholders / exact ISO-8601 UTC release times | **PASS** |
| **Test L** | Item 22 / 41 | Preregistration lock enforcement & tamper detection | **PASS** |

---

## 6. Formal Verdict

**PHASE 4A PREREGISTRATION GATE PASSED**  
**CONFIRMATORY DESIGN LOCKED**  
**READY FOR PHASE 4B EXECUTION**
