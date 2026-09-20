# Phase 4A Confirmatory Data & Phase 4B Runtime Binding & Metrics Closure Report (v1.2.2)

**Stage**: Phase 4B — Runtime Binding & Metrics Closure & Cryptographic Protocol Lock  
**Repository**: `DJKuuki/MANTRA`  
**Evaluation Date**: 2026-09-20  
**Specification Version**: 1.2.2  
**Scientific Code Freeze Commit (Commit G)**: `580d5806aea88eb83ac514c1538695898055ed44`  
**Protocol Lock Source Tree SHA**: `5263ef39d39ba343c7f677beb8e969ed7ae74525509cd5c0be1cf081271c0bf4`  
**Gate Verdict**: **PHASE 4B RUNTIME BINDING & METRICS CLOSURE PASSED / ALL 53 GATE TESTS GREEN / PROTOCOL v1.2.2 LOCKED / REAL PHASE 4B COMPUTE STRICTLY BLOCKED / AWAITING HUMAN AUTHORIZATION**

---

> [!IMPORTANT]
> **NO PHASE 4B EMPIRICAL MODEL RESULTS EXISTED BEFORE THIS EXECUTION IMPLEMENTATION FREEZE**  
> All 25 confirmatory MLM branches, 5 seeds, and 5 doses remain unexecuted. `configs/phase4_execution_authorization.json` does NOT exist in the repository, and the production runner strictly fails closed without explicit human authorization.

---

## 1. Executive Summary & Audit Resolution

The Phase 4B Runtime Binding & Metrics Closure hardens the execution backend to eliminate all discrepancies between the frozen scientific protocol and runtime behavior:

1. **Protocol Lock Manifest & Cryptographic Bindings (v1.2.2)**:
   - `verify_phase4_protocol_lock` verifies all 12 controlled files and returns full cryptographic provenance: `scientific_code_commit`, `code_commit`, `source_tree_hash`, and `protocol_version`.
   - `verify_phase4b_authorization` strictly enforces all 4 cryptographic bindings against the protocol lock manifest.
   - `verify_phase4_code_freeze` binds against the locked commit and source tree hash.

2. **Strict Event-Level Economic Endpoint Mappings**:
   - `ProductionConfirmatoryBackend` and the confirmatory orchestrator extract market outcomes strictly from `market_outcomes.treasury_2y_yield_change` (primary) and `market_outcomes.spy_1d_return` (secondary).
   - Missing fields fail closed immediately with `PreregistrationLockError`, prohibiting silent random or zero fallbacks.
   - Economic endpoints are evaluated using event-level Information Coefficients with stationary block bootstrap confidence intervals (`evaluate_economic_effect_event_level`).

3. **Centralized Metric Evaluator Single Source of Truth**:
   - The confirmatory orchestrator reuses `evaluate_representational_leakage_grouped` for continuous rate-change primary inference, passing the resolved `probe_alpha: 1.0` through to Ridge regression without hardcoded defaults.
   - The binary directional co-primary `next_scheduled_change_vs_hold` is evaluated using `RidgeClassifier(alpha=probe_alpha)` outputting macro F1.
   - Behavioral leakage is evaluated at the event level via `evaluate_behavioral_leakage_event_level`, marking FDR correction as `NOT_EVALUATED_PROTOCOL_UNDERSPECIFIED`.
   - Competence provenance is recorded as `NOT_EVALUATED_NO_EVAL_SPLIT` when `eval_samples` is None.

4. **Runtime Contract Reconciliation & Strict Parameter Locking**:
   - Implemented `resolve_phase4_runtime_contract(conf_cfg, prereg_cfg)` verifying semantic equality across all hyperparameters:
     - `mlm_training.max_steps`: exactly 100
     - `mlm_training.scheduler`: "none"
     - `mlm_training.warmup_ratio`: 0.0
     - `downstream_evaluation.random_seed`: 42
     - `downstream_evaluation.probe_alpha`: 1.0
     - `token_budget`: 256,000 tokens
     - `dose_ladder`: [0.0, 0.25, 0.5, 0.75, 1.0]
     - `seeds`: [13, 42, 87, 123, 2024]
   - Discrepancies fail closed with `PreregistrationLockError`.

5. **Artifact Writer Separation & Provenance Tracking**:
   - `Phase4ArtifactWriter.write_confirmatory_results` strictly rejects non-`EMPIRICAL` data modes when `allow_mock=False`.
   - Branch manifests record all required cryptographic provenance fields: `scientific_code_commit`, `execution_repository_head`, `source_tree_hash`, `protocol_lock_sha256`, and `protocol_version`.

---

## 2. Complete Phase 4 Gate Verification Matrix (Tests A through AZ)

All 53 Gate tests (`tests/test_phase4a_gate.py` Tests A through AZ + Metric Integration) and all 186 repository unit tests pass with 100% green status:

| Check / Test ID | Target Specification | Realized Implementation | Verification Status |
|:---|:---|:---|:---|
| **Test A: Within-Event Consistency** | $N=40$ events, single label/timestamp | 40/40 consistent | PASSED |
| **Test B: Grouped Temporal CV** | 4 folds, expanding window, 8 test/fold | Monotonic, disjoint | PASSED |
| **Test C: Event Economic Stance** | 40 unique events, stance aggregation | Event-level stance | PASSED |
| **Test D: Event Clustered Bootstrap** | Bootstrap clusters whole events | Stationary block boot | PASSED |
| **Test E: Anchor Verbatim Sources** | 181 anchors verbatim in official HTML | 181/181 (100%) match | PASSED |
| **Test F: Future Action Derivation** | Target derives from policy history | Zero-discrepancy | PASSED |
| **Test G: Market Outcome Recomputation**| Daily raw tables recomputed | Bit-exact match | PASSED |
| **Test H: Contamination Provenance** | Strict gap after 2020-01-01 | $\ge 28$ day gap | PASSED |
| **Test I: Treatment Block Manifest** | Exact block tracking & token counts | Full provenance | PASSED |
| **Test J: Dose Ladder Invariant** | $\|D_{\text{realized}} - D\| \le 1/T$ | Max diff $\le 1/256000$ | PASSED |
| **Test K: No Mid-Year Placeholders** | Exact ISO-8601 UTC release times | 100% exact | PASSED |
| **Test L: Preregistration Lock** | Hash lock on prereg config & doc | Cryptographically locked | PASSED |
| **Test M: Contamination Corpus Integrity**| $\ge 50$ post-cutoff official documents | 50 docs verified | PASSED |
| **Test N: Contamination Range Derived** | Programmatic min/max temporal bounds | $T_{\text{min}}/T_{\text{max}}$ verified | PASSED |
| **Test O: Primary Permutation Unit** | Event-level ($N_{\text{OOS}} = 32$) | Permutation on events | PASSED |
| **Test P: Target Type Enforcement** | Continuous Ridge regression | Non-degenerate | PASSED |
| **Test Q: Protocol Lock Enforcement** | 12 controlled files locked | SHA-256 verified | PASSED |
| **Test R: Config Semantic Equality** | Conf matches Prereg contract | Reconciled & locked | PASSED |
| **Test S: Source Registry Documents** | 40/40 documents verified | 100% validated | PASSED |
| **Test T: Base Model & Recipe Lock** | FinBERT rev `4556d13...` | Recipe locked | PASSED |
| **Test U: Real Corpus Sampling Preflight**| D0-D100 sampling preflight | 256k tokens/branch | PASSED |
| **Test V: Clean Corpus Availability** | $\le 2019-12-31$ cutoff enforced | 100% pre-cutoff | PASSED |
| **Test W: Contamination Window** | 2020-01 to 2023-01 window | Strict separation | PASSED |
| **Test X: Continuous Rate Change** | Explicit $\Delta r_e$ derivation | Continuous floats | PASSED |
| **Test Y: Repetition Metrics** | Realized dose vs repetition ratio | Independent metrics | PASSED |
| **Test Z: Source Tree Lock** | Source tree hash locked | Freeze verified | PASSED |
| **Test AA: Mock Orchestration** | 25 branches execute in dry run | Full DAG verified | PASSED |
| **Test AB: Production Backend Exists**| `ProductionConfirmatoryBackend` | Distinct from mock | PASSED |
| **Test AC: Default Full Run** | Default resolves to Production | Never mock | PASSED |
| **Test AD: Mock Rejected in Empirical**| Rejects `MockConfirmatoryBackend` | Fail-closed | PASSED |
| **Test AE: Authorization Bindings** | 4 cryptographic bindings | Verified | PASSED |
| **Test AF: Dirty Tree Enforcement** | Blocked on uncommitted files | Fail-closed | PASSED |
| **Test AG: Modified Source Blocked** | Blocked on source modification | Fail-closed | PASSED |
| **Test AH: Exact Dose Stream** | 256k token budget, $|D_r - D| \le 1/T$ | Verified | PASSED |
| **Test AI: Empirical Data Mode Only**| `data_mode=EMPIRICAL` | No synthetic noise | PASSED |
| **Test AJ: Artifact Serialization** | 25 manifests, metrics, results | Complete roundtrip | PASSED |
| **Test AK: Manifest Completeness** | 25 provenance fields | All fields present | PASSED |
| **Test AL: Production Blocked** | Without authorization manifest | Fail-closed | PASSED |
| **Test AM: Protocol Lock Integrity**| Returns complete bindings, $\ge 1.2.1$ | Verified | PASSED |
| **Test AN: Authorization Validation**| Validates commit, tree, lock, ver | Verified | PASSED |
| **Test AO: Code Freeze Consumption**| Consumes lock manifest metadata | Verified | PASSED |
| **Test AP: Economic Field Mappings**| Requires 2Y yield & SPY 1D return | Fails on missing | PASSED |
| **Test AQ: Economic Bootstrap** | Event-level Information Coefficient CI | Verified | PASSED |
| **Test AR: Evaluator Probe Alpha** | Uses `evaluate_representational_leakage_grouped` | `probe_alpha: 1.0` | PASSED |
| **Test AS: Competence Provenance** | `NOT_EVALUATED_NO_EVAL_SPLIT` | No fake numbers | PASSED |
| **Test AT: Behavioral Analysis** | Event-level sensitivity contrast | FDR marked | PASSED |
| **Test AU: Binary Direction Co-Primary**| RidgeClassifier on rate change vs hold | Macro F1 | PASSED |
| **Test AV: Runtime Contract** | `resolve_phase4_runtime_contract` | Full reconciliation | PASSED |
| **Test AW: MLM Hyperparameters** | `max_steps: 100`, `scheduler: none` | Locked in configs | PASSED |
| **Test AX: Statistical Invariants** | 40 events, 32 OOS events, 181 anchors | Invariants hold | PASSED |
| **Test AY: Provenance Completeness** | Manifest records all 5 bindings | Verified | PASSED |
| **Test AZ: Mock Data Mode Rejection**| `allow_mock=False` rejects mock data | Fail-closed | PASSED |
| **Integration: Deterministic Metrics**| End-to-end evaluation on 40 events | Deterministic | PASSED |

---

## 3. Cryptographic Protocol Lock Manifest (v1.2.2)

- **Protocol Lock Manifest**: `configs/phase4_protocol_lock.json`
- **Protocol Version**: `1.2.2`
- **Source Tree Hash**: `5263ef39d39ba343c7f677beb8e969ed7ae74525509cd5c0be1cf081271c0bf4`
- **Scientific Code Freeze Commit (Commit G)**: `580d5806aea88eb83ac514c1538695898055ed44`
- **Base Model Revision**: Locked to `ProsusAI/finbert` commit `4556d13015211d73dccd3fdd39d39232506f3e43`.
- **Controlled Files (12 Artifacts)**:
  1. `configs/phase4_preregistration.yaml` (`8996a2ab16ab37917e55d56e3aad6f82d106abda98c2d2cce8ff6c4cfb5f6e0d`)
  2. `configs/phase4_confirmatory.yaml` (`427dc55d93d0181c531d4a8bef6826024f427755e0cc5be99285c7dc2d7015ad`)
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

## 4. Phase 4B Confirmatory Execution Specifications

- **Production Execution Backend**: `ProductionConfirmatoryBackend` in `tradingagents/temporal_leakage/phase4_confirmatory.py`
- **Orchestration Graph**: Deterministic 34-step dependency DAG (`run_phase4_confirmatory`)
- **Dose Ladder**: 5 levels $D \in \{0.00, 0.25, 0.50, 0.75, 1.00\}$
- **Random Seeds**: 5 seeds $\{13, 42, 87, 123, 2024\}$
- **Token Budget**: Exactly 256,000 subword tokens per branch (realized dose $|D_{\text{realized}} - D| \le 1/256000$)
- **MLM Training**: Exactly 100 max steps, `scheduler: "none"`, `warmup_ratio: 0.0`, `learning_rate: 5e-5`
- **Total Confirmatory Branches**: 25 MLM branches
- **Downstream Cross-Validation**: 4-fold grouped temporal CV ($N_{\text{OOS}} = 32$ events, 8 train events per fold)
- **Primary Estimator**: Ridge Regression ($\alpha = 1.0$) on continuous future rate change $y_e = \Delta r_e$
- **Primary Inferential Test**: Sign-flip permutation test on event-level absolute error improvement ($B = 2,000$, $N = 32$)
- **Co-Primary Estimator**: Ridge Classifier ($\alpha = 1.0$) on binary direction vs hold ($B = 2,000$, $N = 32$)
- **Economic Endpoints**: 2Y Treasury yield change (primary) and SPY 1D return (secondary) evaluated via event-level IC with stationary block bootstrap
- **Authorization Guard**: Requires external signed authorization manifest at `configs/phase4_execution_authorization.json` binding protocol version `1.2.2`, lock SHA, freeze commit, and tree hash to initiate compute.
