# Phase 4A Confirmatory Data & Preregistration Gate Protocol Closure Report (v1.2.0)

**Stage**: Phase 4A — Confirmatory Data & Preregistration Gate Protocol Closure  
**Repository**: `DJKuuki/MANTRA`  
**Evaluation Date**: 2026-09-20  
**Specification Version**: 1.2.0  
**Code Freeze Commit (Commit C)**: `db048133e81055ba1f937c234a0677f8ed6061ea`  
**Protocol Lock Source Tree SHA**: `f9584967d9342f0fe687ebb59204a2daa7493c0c5fc4636ca1e422b244867fe4`  
**Gate Verdict**: **PHASE 4A PROTOCOL CLOSURE PASSED / PHASE 4B RUNNER FROZEN / READY FOR HUMAN EXECUTION AUTHORIZATION**

---

## 1. Executive Summary & Audit Resolution

The Phase 4A Protocol Closure v1.2 addresses and closes all remaining protocol, data provenance, and runtime orchestration findings from previous audits:

1. **Pre-Cutoff D0 Sham Corpus Strict Temporal Invariant**:
   - Dropped `fomc-minutes-2019-12-11` (whose release timestamp was `2020-01-01T19:00:00Z` > 2019 cutoff).
   - Retained 63 verified official documents (40 statements 2015–2019, 23 minutes 2017–2019).
   - Invariant strictly verified: $\max(\text{available\_time}) = \text{2019-12-11T19:00:00Z} \le \text{2019-12-31T23:59:59Z}$ (Test V).
2. **Contamination Corpus Availability Window & 50/50 Raw Source Verification**:
   - Availability window aligned to exact release timestamps: `2020-01-29T19:00:00Z` to `2023-01-04T19:00:00Z` (Test W).
   - Deterministic HTML canonical extraction algorithm implemented and verified 50/50 against raw source HTML files (`all_verified: True`).
3. **Continuous Future Target Derivation**:
   - Target rate change $y_e = \Delta r_e = \text{target\_upper\_after} - \text{target\_upper\_before}$ derived bit-exact from `policy_history.csv` across all 40 events (Test X).
   - Preserves magnitude and continuous rate changes ($\pm 0.50, \pm 0.25, 0.00$).
4. **Treatment Repetition Metrics Separation**:
   - Disentangled shortfall cycling ratio (`forced_repetition_ratio`) from vocabulary diversity (`token_type_diversity`).
   - Verified `forced_repetition_ratio == 0.00 <= 0.20` across all doses on real corpora under 256k tokens (Test Y).
5. **Machine-Locked Source Tree & Phase 4B Runner**:
   - Deterministic source tree SHA computed across all controlled Python modules and execution configs (Test Z).
   - End-to-end 34-step orchestration graph implemented and verified under mock execution across all 25 confirmatory branches (5 doses $\times$ 5 seeds) on 32 OOS events (Test AA).
   - Hard execution blocker enforces `Phase4BAuthorizationError` fail-closed when authorization manifest is missing.

---

## 2. Quantitative Gate Verification Audit

All 27 Gate tests (`tests/test_phase4a_gate.py` Tests A through AA) and all 160 repository unit tests pass with 100% green status:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.4, pytest-9.0.3, pluggy-1.6.0
rootdir: E:\MANTRA
configfile: pyproject.toml
collected 27 items in tests/test_phase4a_gate.py

tests/test_phase4a_gate.py::test_a_within_event_consistency PASSED       [  3%]
tests/test_phase4a_gate.py::test_b_grouped_temporal_split PASSED         [  7%]
tests/test_phase4a_gate.py::test_c_event_level_economic_effect PASSED    [ 11%]
tests/test_phase4a_gate.py::test_d_event_clustered_bootstrap PASSED      [ 14%]
tests/test_phase4a_gate.py::test_e_anchor_canonical_source_existence PASSED [ 18%]
tests/test_phase4a_gate.py::test_f_policy_future_action_provenance PASSED [ 22%]
tests/test_phase4a_gate.py::test_g_market_outcome_recomputation PASSED   [ 25%]
tests/test_phase4a_gate.py::test_h_contamination_timeline_provenance PASSED [ 29%]
tests/test_phase4a_gate.py::test_i_treatment_block_provenance PASSED     [ 33%]
tests/test_phase4a_gate.py::test_j_exact_dose_ladder_invariant PASSED    [ 37%]
tests/test_phase4a_gate.py::test_k_no_mid_year_placeholder PASSED        [ 40%]
tests/test_phase4a_gate.py::test_l_preregistration_lock_enforcement PASSED [ 44%]
tests/test_phase4a_gate.py::test_m_actual_contamination_corpus_integrity PASSED [ 48%]
tests/test_phase4a_gate.py::test_n_contamination_temporal_range_derived PASSED [ 51%]
tests/test_phase4a_gate.py::test_o_event_level_primary_permutation_unit PASSED [ 55%]
tests/test_phase4a_gate.py::test_p_explicit_target_type_enforcement PASSED [ 59%]
tests/test_phase4a_gate.py::test_q_protocol_lock_enforcement PASSED      [ 62%]
tests/test_phase4a_gate.py::test_r_config_semantic_equality PASSED       [ 66%]
tests/test_phase4a_gate.py::test_s_source_registry_all_40_documents PASSED [ 70%]
tests/test_phase4a_gate.py::test_t_base_revision_and_downstream_recipe_lock PASSED [ 74%]
tests/test_phase4a_gate.py::test_u_treatment_sampling_real_corpus_preflight PASSED [ 77%]
tests/test_phase4a_gate.py::test_v_clean_corpus_strict_availability_cutoff PASSED [ 81%]
tests/test_phase4a_gate.py::test_w_contamination_window_and_50_50_source_verification PASSED [ 85%]
tests/test_phase4a_gate.py::test_x_continuous_future_rate_change_derivation PASSED [ 88%]
tests/test_phase4a_gate.py::test_y_treatment_repetition_metrics_separation PASSED [ 92%]
tests/test_phase4a_gate.py::test_z_source_tree_lock_and_code_freeze PASSED [ 96%]
tests/test_phase4a_gate.py::test_aa_phase4b_runner_orchestration_and_authorization PASSED [100%]

============================= 27 passed in 18.03s =============================
Full repository suite: 160 passed, 33 subtests passed in 60.14s.
```

---

## 3. Quantitative Verification Counts

| Audit Dimension | Target / Planned | Actual Verified | Verification Mode |
| :--- | :--- | :--- | :--- |
| **Independent Events ($N$)** | 40 | 40 | Exact metadata match (`events.jsonl`) |
| **Paragraph Anchors ($M$)** | 181 | 181 | 100% verified verbatim in cached HTML |
| **Anchors Source-Verified** | 181 | 181 / 181 (100%) | Test E (No sampling) |
| **Source Documents Verified** | 40 | 40 / 40 (100%) | Test S (Raw HTML + SHA match) |
| **Pre-Cutoff Sham Documents** | $\ge 40$ | 63 | Test V ($\max \le \text{2019-12-31T23:59:59Z}$) |
| **Contamination Documents** | $\ge 50$ | 50 | Test M (`documents.jsonl`, 237,273 words) |
| **Contamination Raw Sources** | 50 | 50 / 50 (100%) | Test W (`verify_contamination_document_sources`) |
| **Evaluated OOS Test Events** | 32 | 32 | Grouped temporal CV (4 folds $\times$ 8) |
| **Primary Inferential $N$** | 32 | 32 events | Test O (Event-level paired permutation) |
| **Policy Target Reconstruction** | 40 | 40 / 40 (100%) | Test X ($y_e = \Delta r_e$ bit-exact) |
| **Market Outcome Reconstruction** | 40 | 40 / 40 (100%) | Test G (Bit-exact recomputation) |
| **Protocol Controlled Files** | 12 | 12 | Test Q (`phase4_protocol_lock.json`) |
| **Source Tree Hash Frozen** | 1 | 1 | Test Z (`source_tree_hash` locked) |
| **Phase 4B Mock Branches** | 25 | 25 / 25 (100%) | Test AA (5 doses $\times$ 5 seeds end-to-end) |
| **Phase 4B Authorization Guard**| Fail-Closed | Fail-Closed | Test AA (`Phase4BAuthorizationError`) |

---

## 4. Cryptographic Protocol Lock Manifest (v1.2.0)

- **Protocol Lock Manifest**: `configs/phase4_protocol_lock.json`
- **Protocol Version**: `1.2.0`
- **Source Tree Hash**: `f9584967d9342f0fe687ebb59204a2daa7493c0c5fc4636ca1e422b244867fe4`
- **Code Freeze Commit**: `db048133e81055ba1f937c234a0677f8ed6061ea`
- **Base Model Revision**: Locked to `ProsusAI/finbert` commit `4556d13015211d73dccd3fdd39d39232506f3e43`.
- **Controlled Files (12 Artifacts)**:
  1. `configs/phase4_preregistration.yaml` (`31f767a819a6461ad861b215fc5a2f582a7b355e43fb82b1c42d09300136ae98`)
  2. `configs/phase4_confirmatory.yaml` (`181a63086e551cc4c391c18925e109d0b9724547544583d0599b3e08679dbfef`)
  3. `data/research/fomc/events/events.jsonl` (`981ad9f483e0518d1601f3391231e532458ca71e379404ef83795132693381e4`)
  4. `data/research/fomc/confirmatory_anchors/anchors.jsonl` (`b03a2262f2222e8e28a29c903247682501244652f5a958d093d2109714e5c8ae`)
  5. `data/research/fomc/policy_history.csv` (`51f233d3e957a54a7b87ecaf073c1d9906fb9e02bc45a68945910ec693b04fc2`)
  6. `data/research/market/market_manifest.json` (`692dcd327316fe29ff2e68ce82cc6d5d1101e2aa41d09b570305fc95a5fa059c`)
  7. `data/research/market/spy_daily_raw.csv` (`11f7d6f3bfb171b70f0de96e0f74d5d8954c139050e61646d140a7052ece143e`)
  8. `data/research/market/treasury_2y_raw.csv` (`337d5863a73e91d7418a5fa04c58482c206aacf492de55ae12b47b823a9dfca8`)
  9. `data/research/fomc/phase4_contamination/documents.jsonl` (`54b3e84e8ef958e78032815a6cc76e1ca68cb0428afe71b390a583ee60e25c58`)
  10. `data/research/fomc/phase4_contamination/manifest.json` (`6893fdb60a4c85cbfcca2293f81df079ffa4a44b7aa62dcf0b908fe807b5b6ae`)
  11. `data/research/fomc/phase4_pre_cutoff/documents.jsonl` (`5202f14b340c3212dc48c13e1ba7a8a814cb472a4c25df072bed14b7a3f19e4c`)
  12. `data/research/fomc/phase4_pre_cutoff/manifest.json` (`8e85378df1d68b0917990a7c1ed89b80fe75e00fe156854b3f13c9ac1bbece9a`)

---

## 5. Phase 4B Confirmatory Execution Specifications

- **Execution Runner**: `tradingagents/temporal_leakage/phase4_confirmatory.py`
- **Orchestration Graph**: Deterministic 34-step dependency DAG (`run_phase4_confirmatory`)
- **Dose Ladder**: 5 levels $D \in \{0.00, 0.25, 0.50, 0.75, 1.00\}$
- **Random Seeds**: 5 seeds $\{42, 123, 456, 789, 101112\}$
- **Total Confirmatory Branches**: 25 MLM branches
- **Downstream Cross-Validation**: 4-fold grouped temporal CV ($N_{\text{OOS}} = 32$ events, 8 train events per fold)
- **Primary Estimator**: Ridge Regression ($\alpha = 1.0$) on continuous future rate change $y_e = \Delta r_e$
- **Primary Inferential Test**: Sign-flip permutation test on event-level absolute error improvement ($B = 2,000$, $N = 32$)
- **Authorization Guard**: Requires external signed authorization manifest at `configs/phase4_execution_authorization.json` to initiate GPU compute.
