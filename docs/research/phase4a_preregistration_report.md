# Phase 4A Confirmatory Data & Phase 4B Production Runner Closure Report (v1.2.1)

**Stage**: Phase 4B — Production Runner Closure & Scientific Code Freeze  
**Repository**: `DJKuuki/MANTRA`  
**Evaluation Date**: 2026-09-20  
**Specification Version**: 1.2.1  
**Scientific Code Freeze Commit (Commit E)**: `a3ad3383bdee0d56f32d9e40a3cc53a35c4cfbc9`  
**Protocol Lock Source Tree SHA**: `bc97536cc1793a30bbc5a6d13079c5eb9603e9ac335922b780a425ef9dbc6cf6`  
**Gate Verdict**: **PHASE 4B PRODUCTION RUNNER CLOSURE PASSED / SCIENTIFIC EXECUTION CODE FROZEN / READY FOR HUMAN EXECUTION AUTHORIZATION / REAL PHASE 4B COMPUTE NOT YET STARTED**

---

> [!IMPORTANT]
> **NO PHASE 4B EMPIRICAL MODEL RESULTS EXISTED BEFORE THIS EXECUTION IMPLEMENTATION FREEZE**  
> All 25 confirmatory MLM branches, 5 seeds, and 5 doses remain unexecuted. `configs/phase4_execution_authorization.json` does NOT exist in the repository, and the production runner fails closed without human authorization.

---

## 1. Executive Summary & Audit Resolution

The Phase 4B Production Runner Closure completes the scientific execution backend while preserving all Phase 4A data, anchor, and statistical protocol locks:

1. **Production FinBERT Backend (`ProductionConfirmatoryBackend`)**:
   - Replaced mock-only execution skeleton with a fully implemented, real FinBERT confirmatory execution backend inheriting from `Phase4ExecutionBackend`.
   - Uses frozen `ProsusAI/finbert` checkpoint locked to revision `4556d13015211d73dccd3fdd39d39232506f3e43`.
   - Calls `create_exact_token_dose_stream` with exact 256,000 token budget across all 5 doses ($D \in \{0.00, 0.25, 0.50, 0.75, 1.00\}$) satisfying the dose invariant $|D_{\text{realized}} - D_{\text{requested}}| \le 1/T$ and forced repetition ratio $\le 0.20$.
   - Enforces per-seed causal symmetries: bit-identical initial model parameter hash, identical deterministic MLM mask schedule, identical fresh FOMC classification head initialization, and paired downstream sample ordering across all 5 doses within each seed.
   - Evaluates real anchor paragraph hidden representations and event-level centroid aggregations.
   - Computes real competence ($C$), behavioral masking sensitivity ($S_D$), and economic IC ($E_L$: 2Y Treasury primary, SPY secondary). Sets `temporal_robustness = "NOT_EVALUATED"`.
   - Explicitly emits `data_mode = "EMPIRICAL"`. Strictly prohibits synthetic noise or mock fallbacks in production.

2. **Test-Only Mock Backend Isolation & Rejection**:
   - `MockConfirmatoryBackend` is retained strictly for CI and dry-run orchestration.
   - Calling full confirmatory execution with `MockConfirmatoryBackend` raises `ProductionBackendValidationError`.
   - Default full execution backend resolves strictly to `ProductionConfirmatoryBackend`.

3. **Persistent Artifact Serialization (`Phase4ArtifactWriter`)**:
   - Serializes all 25 branch manifests (`manifests/seed{S}_d{D}.json`), 25 branch metrics (`metrics/seed{S}_d{D}.json`), aggregate confirmatory results (`results/phase4_confirmatory_results.json`), and execution/protocol provenance (`provenance/execution_environment.json`, `provenance/protocol_verification.json`).
   - Every branch manifest records all 25 required provenance fields.

4. **Cryptographic Authorization Hardening**:
   - `verify_phase4b_authorization` strictly enforces all four cryptographic bindings: `protocol_version`, `protocol_lock_sha256`, `locked_scientific_code_commit`, and `locked_source_tree_hash`.
   - Fails closed on any missing or mismatched binding (`Phase4BAuthorizationError`).
   - Code freeze enforcement verifies clean working tree and checks that git diff against freeze commit contains zero controlled scientific source files.

---

## 2. Quantitative Gate Verification Audit

All 38 Gate tests (`tests/test_phase4a_gate.py` Tests A through AL) and all 171 repository unit tests pass with 100% green status:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.4, pytest-9.0.3, pluggy-1.6.0
rootdir: E:\MANTRA
configfile: pyproject.toml
collected 38 items in tests/test_phase4a_gate.py

tests/test_phase4a_gate.py::test_a_within_event_consistency PASSED       [  2%]
tests/test_phase4a_gate.py::test_b_grouped_temporal_split PASSED         [  5%]
tests/test_phase4a_gate.py::test_c_event_level_economic_effect PASSED    [  7%]
tests/test_phase4a_gate.py::test_d_event_clustered_bootstrap PASSED      [ 10%]
tests/test_phase4a_gate.py::test_e_anchor_canonical_source_existence PASSED [ 13%]
tests/test_phase4a_gate.py::test_f_policy_future_action_provenance PASSED [ 15%]
tests/test_phase4a_gate.py::test_g_market_outcome_recomputation PASSED   [ 18%]
tests/test_phase4a_gate.py::test_h_contamination_timeline_provenance PASSED [ 21%]
tests/test_phase4a_gate.py::test_i_treatment_block_provenance PASSED     [ 23%]
tests/test_phase4a_gate.py::test_j_exact_dose_ladder_invariant PASSED    [ 26%]
tests/test_phase4a_gate.py::test_k_no_mid_year_placeholder PASSED        [ 28%]
tests/test_phase4a_gate.py::test_l_preregistration_lock_enforcement PASSED [ 31%]
tests/test_phase4a_gate.py::test_m_actual_contamination_corpus_integrity PASSED [ 34%]
tests/test_phase4a_gate.py::test_n_contamination_temporal_range_derived PASSED [ 36%]
tests/test_phase4a_gate.py::test_o_event_level_primary_permutation_unit PASSED [ 39%]
tests/test_phase4a_gate.py::test_p_explicit_target_type_enforcement PASSED [ 42%]
tests/test_phase4a_gate.py::test_q_protocol_lock_enforcement PASSED      [ 44%]
tests/test_phase4a_gate.py::test_r_config_semantic_equality PASSED       [ 47%]
tests/test_phase4a_gate.py::test_s_source_registry_all_40_documents PASSED [ 50%]
tests/test_phase4a_gate.py::test_t_base_revision_and_downstream_recipe_lock PASSED [ 52%]
tests/test_phase4a_gate.py::test_u_treatment_sampling_real_corpus_preflight PASSED [ 55%]
tests/test_phase4a_gate.py::test_v_clean_corpus_strict_availability_cutoff PASSED [ 57%]
tests/test_phase4a_gate.py::test_w_contamination_window_and_50_50_source_verification PASSED [ 60%]
tests/test_phase4a_gate.py::test_x_continuous_future_rate_change_derivation PASSED [ 63%]
tests/test_phase4a_gate.py::test_y_treatment_repetition_metrics_separation PASSED [ 65%]
tests/test_phase4a_gate.py::test_z_source_tree_lock_and_code_freeze PASSED [ 68%]
tests/test_phase4a_gate.py::test_aa_phase4b_runner_orchestration_and_authorization PASSED [ 71%]
tests/test_phase4a_gate.py::test_ab_production_backend_exists PASSED     [ 73%]
tests/test_phase4a_gate.py::test_ac_default_full_run_is_never_mock PASSED [ 76%]
tests/test_phase4a_gate.py::test_ad_mock_rejected_in_full_empirical_mode PASSED [ 78%]
tests/test_phase4a_gate.py::test_ae_authorization_cryptographic_binding PASSED [ 81%]
tests/test_phase4a_gate.py::test_af_dirty_tree_blocks_production PASSED  [ 84%]
tests/test_phase4a_gate.py::test_ag_controlled_scientific_source_modification_blocks_execution PASSED [ 86%]
tests/test_phase4a_gate.py::test_ah_real_backend_uses_exact_dose_stream PASSED [ 89%]
tests/test_phase4a_gate.py::test_ai_real_backend_does_not_emit_synthetic_metrics PASSED [ 92%]
tests/test_phase4a_gate.py::test_aj_artifact_writer PASSED               [ 94%]
tests/test_phase4a_gate.py::test_ak_manifest_completeness PASSED         [ 97%]
tests/test_phase4a_gate.py::test_al_production_execution_still_blocked PASSED [100%]

============================= 38 passed in 17.48s =============================
Full repository suite: 171 passed, 33 subtests passed in 60.50s.
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
| **Production Backend Implemented** | `ProductionConfirmatoryBackend` | Validated | Test AB (Subclasses backend, FinBERT rev) |
| **Default Backend is Production** | Production | Production | Test AC (Default is never mock) |
| **Mock Backend Prohibited** | Hard Fail | Hard Fail | Test AD (`ProductionBackendValidationError`) |
| **Cryptographic Authorization** | 4 Bindings | 4 Bindings | Test AE (Lock SHA, Tree Hash, Commit, Version) |
| **Dirty Tree Enforcement** | Blocked | Blocked | Test AF (`CodeFreezeError` on dirty git tree) |
| **Modified Source Blocked** | Blocked | Blocked | Test AG (Tree mismatch and descendant check) |
| **Exact Dose Stream Integration** | 256k tokens | 256k tokens | Test AH ($|D_{\text{realized}} - D| \le 1/T$) |
| **Empirical Data Mode Only** | `data_mode=EMPIRICAL` | Validated | Test AI (`temporal_robustness=NOT_EVALUATED`) |
| **Artifact Serialization** | 25 manifests, metrics | 25 / 25 | Test AJ (`Phase4ArtifactWriter` roundtrip) |
| **Manifest Completeness** | 25 provenance fields | 25 / 25 | Test AK (All provenance fields present) |
| **Execution Blocked** | Fail-Closed | Fail-Closed | Test AL (`Phase4BAuthorizationError`) |

---

## 4. Cryptographic Protocol Lock Manifest (v1.2.1)

- **Protocol Lock Manifest**: `configs/phase4_protocol_lock.json`
- **Protocol Version**: `1.2.1`
- **Source Tree Hash**: `bc97536cc1793a30bbc5a6d13079c5eb9603e9ac335922b780a425ef9dbc6cf6`
- **Scientific Code Freeze Commit (Commit E)**: `a3ad3383bdee0d56f32d9e40a3cc53a35c4cfbc9`
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

- **Production Execution Backend**: `ProductionConfirmatoryBackend` in `tradingagents/temporal_leakage/phase4_confirmatory.py`
- **Orchestration Graph**: Deterministic 34-step dependency DAG (`run_phase4_confirmatory`)
- **Dose Ladder**: 5 levels $D \in \{0.00, 0.25, 0.50, 0.75, 1.00\}$
- **Random Seeds**: 5 seeds $\{13, 42, 87, 123, 2024\}$
- **Token Budget**: Exactly 256,000 subword tokens per branch (realized dose $|D_{\text{realized}} - D| \le 1/256000$)
- **Total Confirmatory Branches**: 25 MLM branches
- **Downstream Cross-Validation**: 4-fold grouped temporal CV ($N_{\text{OOS}} = 32$ events, 8 train events per fold)
- **Primary Estimator**: Ridge Regression ($\alpha = 1.0$) on continuous future rate change $y_e = \Delta r_e$
- **Primary Inferential Test**: Sign-flip permutation test on event-level absolute error improvement ($B = 2,000$, $N = 32$)
- **Authorization Guard**: Requires external signed authorization manifest at `configs/phase4_execution_authorization.json` binding protocol version `1.2.1`, lock SHA, freeze commit, and tree hash to initiate compute.

