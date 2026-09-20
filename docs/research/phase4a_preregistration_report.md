# Phase 4A Confirmatory Data & Preregistration Gate Finalization Report (v1.1.0)

**Stage**: Phase 4A — Confirmatory Data & Preregistration Gate Finalization  
**Repository**: `RubiscoYHY/MANTRA`  
**Evaluation Date**: 2026-09-20  
**Specification Version**: 1.1.0  
**Gate Verdict**: **PHASE 4A FINALIZATION PASSED / CONFIRMATORY PROTOCOL v1.1 LOCKED / PHASE 4B REMAINS BLOCKED PENDING HUMAN APPROVAL**  

---

## 1. Gate Execution & Audit Summary

The Phase 4A Finalization Patch hardens the confirmatory design into a machine-locked, runtime-bound, and statistically executable protocol. All 21 Phase 4A Gate tests (`tests/test_phase4a_gate.py` Tests A through U) and all 133 pre-existing repository unit tests (total 154 tests) pass with $100\%$ green status.

```text
============================= test session starts =============================
platform win32 -- Python 3.13.4, pytest-9.0.3, pluggy-1.6.0
rootdir: E:\MANTRA
configfile: pyproject.toml
collected 21 items in tests/test_phase4a_gate.py

tests/test_phase4a_gate.py::test_a_within_event_consistency PASSED       [  4%]
tests/test_phase4a_gate.py::test_b_grouped_temporal_split PASSED         [  9%]
tests/test_phase4a_gate.py::test_c_event_level_economic_effect PASSED    [ 14%]
tests/test_phase4a_gate.py::test_d_event_clustered_bootstrap PASSED      [ 19%]
tests/test_phase4a_gate.py::test_e_anchor_canonical_source_existence PASSED [ 23%]
tests/test_phase4a_gate.py::test_f_policy_future_action_provenance PASSED [ 28%]
tests/test_phase4a_gate.py::test_g_market_outcome_recomputation PASSED   [ 33%]
tests/test_phase4a_gate.py::test_h_contamination_timeline_provenance PASSED [ 38%]
tests/test_phase4a_gate.py::test_i_treatment_block_provenance PASSED     [ 42%]
tests/test_phase4a_gate.py::test_j_exact_dose_ladder_invariant PASSED    [ 47%]
tests/test_phase4a_gate.py::test_k_no_mid_year_placeholder PASSED        [ 52%]
tests/test_phase4a_gate.py::test_l_preregistration_lock_enforcement PASSED [ 57%]
tests/test_phase4a_gate.py::test_m_actual_contamination_corpus_integrity PASSED [ 61%]
tests/test_phase4a_gate.py::test_n_contamination_temporal_range_derived PASSED [ 66%]
tests/test_phase4a_gate.py::test_o_event_level_primary_permutation_unit PASSED [ 71%]
tests/test_phase4a_gate.py::test_p_explicit_target_type_enforcement PASSED [ 76%]
tests/test_phase4a_gate.py::test_q_protocol_lock_enforcement PASSED      [ 80%]
tests/test_phase4a_gate.py::test_r_config_semantic_equality PASSED       [ 85%]
tests/test_phase4a_gate.py::test_s_source_registry_all_40_documents PASSED [ 90%]
tests/test_phase4a_gate.py::test_t_base_revision_and_downstream_recipe_lock PASSED [ 95%]
tests/test_phase4a_gate.py::test_u_treatment_sampling_real_corpus_preflight PASSED [100%]

============================= 21 passed in 14.19s =============================
Full repository suite: 154 passed, 33 subtests passed in 51.47s.
```

---

## 2. Quantitative Verification Counts

| Audit Dimension | Target / Planned | Actual Verified | Verification Mode |
| :--- | :--- | :--- | :--- |
| **Independent Events ($N$)** | 40 | 40 | Exact metadata match (`events.jsonl`) |
| **Paragraph Anchors ($M$)** | 181 | 181 | 100% verified verbatim in cached HTML |
| **Anchors Source-Verified** | 181 | 181 / 181 (100%) | Test E (No sampling) |
| **Source Documents Verified** | 40 | 40 / 40 (100%) | Test S (Raw HTML + SHA match) |
| **Contamination Documents** | $\ge 50$ | 50 | `documents.jsonl` (237,273 words) |
| **Contamination Documents Verified** | 50 | 50 / 50 (100%) | Test M (Schema + Exact Timestamps) |
| **Pre-Cutoff Sham Documents** | $\ge 40$ | 64 | `phase4_pre_cutoff` (251,649 words) |
| **Evaluated OOS Test Events** | 32 | 32 | Grouped temporal CV (4 folds $\times$ 8) |
| **Primary Inferential $N$** | 32 | 32 events | Test O (Event-level paired permutation) |
| **Policy Reconstruction** | 40 | 40 / 40 (100%) | Test F (Bit-exact derivation) |
| **Market Outcome Reconstruction** | 40 | 40 / 40 (100%) | Test G (Bit-exact recomputation) |
| **Protocol Controlled Files** | 10 | 10 | `configs/phase4_protocol_lock.json` |
| **Protocol Hashes Verified** | 10 | 10 / 10 (100%) | Test Q (Normalized LF SHA-256) |

---

## 3. Contamination Corpus Specification

- **Location**: `data/research/fomc/phase4_contamination/`
- **Total Documents**: 50 official Federal Reserve post-cutoff documents
  - `scheduled_statement`: 23 documents (Primary)
  - `unscheduled_statement`: 2 documents (2020-03-03, 2020-03-15 emergency releases)
  - `policy_strategy_statement`: 1 document (2020-08-27 Jackson Hole revised framework)
  - `meeting_minutes`: 24 documents (Secondary)
- **Total Words**: 237,273 words
- **Estimated Subword Tokens**: ~308,454 tokens (> 256,000 budget)
- **Temporal Range**: `2020-01-29T19:00:00Z` to `2023-01-04T19:00:00Z` (derived programmatically)
- **Isolation Audit**:
  - Document ID Overlap: Exactly 0
  - Text Exact Hash Overlap: Exactly 0
  - Temporal Gap Buffer: 48 calendar days ($\max(T_{\text{anchors}}) < \min(T_{\text{contamination}})$)
- **Dataset SHA-256**: `54b3e84e8ef958e78032815a6cc76e1ca68cb0428afe71b390a583ee60e25c58`

---

## 4. Primary Inferential Statistical Procedure

- **Primary Target**: Next scheduled FOMC target rate change ($\Delta r \in \{-0.25, 0.0, +0.25, +0.50\}$)
- **Target Type**: Explicitly `continuous` (prohibits unique-value heuristic guessing)
- **Model**: Ridge Regression ($L_2$ regularization, $\alpha=1.0$)
- **Effect Metric**: Delta Spearman Rank Correlation ($\Delta \rho = \rho_{\text{leak}} - \rho_{\text{clean}}$)
- **Primary Inferential Statistic**: Mean paired event absolute error improvement:
  $$d_e = |y_e - \hat{y}_{\text{clean}}(e)| - |y_e - \hat{y}_{\text{leak}}(e)|$$
- **Permutation Unit**: Strictly **event** ($N_{\text{OOS}} = 32$ evaluated Out-of-Sample events)
- **Permutation Count**: $B = 2,000$ sign flips across the 32 event deltas
- **A Priori Statistical Power**: $86.4\%$ power validated via empirical Monte Carlo simulation ($N_{\text{OOS}}=32, d=0.50, \alpha=0.05$)

---

## 5. Cryptographic Protocol Lock & Execution Constraints

- **Protocol Lock Manifest**: `configs/phase4_protocol_lock.json`
- **Protocol Version**: `1.1.0`
- **Controlled Files (10 Artifacts)**:
  1. `configs/phase4_preregistration.yaml` (`d56b79724701223ef9c3b03cd1d7b641b159ea62acf5b3d46af3847f734fb12f`)
  2. `configs/phase4_confirmatory.yaml` (`853c55db7f4c7f7772452a4b9d6f6fceedd2de012c63c47dd9037b1062663d47`)
  3. `data/research/fomc/events/events.jsonl` (`33bd3879c36fbde16ab1427538019a50014e24634fff6323accf01bae5ce4d64`)
  4. `data/research/fomc/confirmatory_anchors/anchors.jsonl` (`b03a2262f2222e8e28a29c903247682501244652f5a958d093d2109714e5c8ae`)
  5. `data/research/fomc/policy_history.csv` (`51f233d3e957a54a7b87ecaf073c1d9906fb9e02bc45a68945910ec693b04fc2`)
  6. `data/research/market/market_manifest.json` (`692dcd327316fe29ff2e68ce82cc6d5d1101e2aa41d09b570305fc95a5fa059c`)
  7. `data/research/market/spy_daily_raw.csv` (`11f7d6f3bfb171b70f0de96e0f74d5d8954c139050e61646d140a7052ece143e`)
  8. `data/research/market/treasury_2y_raw.csv` (`337d5863a73e91d7418a5fa04c58482c206aacf492de55ae12b47b823a9dfca8`)
  9. `data/research/fomc/phase4_contamination/documents.jsonl` (`54b3e84e8ef958e78032815a6cc76e1ca68cb0428afe71b390a583ee60e25c58`)
  10. `data/research/fomc/phase4_contamination/manifest.json` (`2703882fdebabb56199de6b1375dcbdd83ec091b3703823d071fff0f8aa8e071`)
- **Base Model Revision**: Locked to `ProsusAI/finbert` commit `4556d13015211d73dccd3fdd39d39232506f3e43`.
- **Downstream Fine-Tuning Recipe**: Frozen across epochs (3), batch size (16), learning rate ($2 \times 10^{-5}$), optimizer (AdamW), warmup ($10\%$), max length (128).
- **Execution Guard**: Phase 4B model execution is strictly blocked in code (`PreregistrationLockError`) until human authorization is committed.

---

## 6. Treatment Sampling Feasibility Preflight

Evaluated on real pre/post corpora with 256,000 token budget (500 blocks $\times$ 512 tokens):
- **D0 Repetition Ratio**: $0.00$ ($\le 0.20$ threshold)
- **D25 Repetition Ratio**: $0.00$ ($\le 0.20$ threshold)
- **D50 Repetition Ratio**: $0.00$ ($\le 0.20$ threshold)
- **D75 Repetition Ratio**: $0.00$ ($\le 0.20$ threshold)
- **D100 Repetition Ratio**: $0.00$ ($\le 0.20$ threshold)
- **Dose Invariant**: $|D_{\text{realized}} - D_{\text{requested}}| \le 1/T$ verified across all doses.
