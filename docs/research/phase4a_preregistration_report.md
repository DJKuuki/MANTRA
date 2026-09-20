# Phase 4 Confirmatory Protocol Lock & Final Gate Patch Closure Report (v1.2.3)

**Stage**: Phase 4B — Final Gate Patch & Protocol v1.2.3 Cryptographic Lock  
**Repository**: `DJKuuki/MANTRA`  
**Evaluation Date**: 2026-09-20  
**Specification Version**: 1.2.3  
**Scientific Code Freeze Commit (Commit I)**: `e8426ec82ca3895a6b55a1e93cabbe235ddee6ad`  
**Protocol Lock Source Tree SHA**: `04389535d0c60b5e615cb8fe64b6c3c595789f959ed9e33ae89d6990342dd0d2`  
**Gate Verdict**: **PHASE 4B FINAL GATE PASSED / ALL 54 GATE TESTS GREEN / 187 UNIT TESTS GREEN / PROTOCOL v1.2.3 LOCKED / SHALLOW-CLONE CI RESOLVED / BEHAVIORAL SECONDARY ENDPOINT RECONCILED / REAL PHASE 4B COMPUTE STRICTLY BLOCKED / AWAITING HUMAN AUTHORIZATION**

---

> [!IMPORTANT]
> **NO PHASE 4B EMPIRICAL MODEL RESULTS EXISTED BEFORE THIS EXECUTION IMPLEMENTATION FREEZE**  
> All 25 confirmatory MLM branches, 5 seeds, and 5 doses remain unexecuted. `configs/phase4_execution_authorization.json` does NOT exist in the repository, and the production runner strictly fails closed without explicit human authorization.

---

## 1. Executive Summary & Audit Resolution

The Phase 4B Final Gate Patch (v1.2.3) resolves the final two operational and statistical preregistration issues prior to confirmatory execution authorization:

1. **GitHub Actions Shallow-Clone CI Blocker Resolution**:
   - Added `fetch-depth: 0` to `actions/checkout@v4` in `.github/workflows/tests.yml`.
   - Ensures full Git history is fetched during CI workflows, allowing `git merge-base --is-ancestor <locked_commit> <head_commit>` to succeed without shallow-clone ancestry errors.
   - Added regression test `test_bb_ancestry_verification_allows_doc_only_descendants` proving that non-controlled commits (such as documentation and protocol lock reports) correctly pass code freeze verification.

2. **Secondary Behavioral Leakage Preregistration Alignment**:
   - Fully reconciled `configs/phase4_preregistration.yaml`, `configs/phase4_confirmatory.yaml`, and `docs/research/phase4_preregistration.md`.
   - Behavioral keyword stance sensitivity shift is formally established as a **descriptive secondary endpoint** without confirmatory p-value, hypothesis test rejection threshold, or Benjamini-Hochberg FDR multiplicity adjustment.
   - `evaluate_behavioral_leakage_event_level` outputs event-level descriptive metrics:
     - `l_behavior_mean`, `l_behavior_median`, `l_behavior_std`, `l_behavior_event`
     - `mask_sensitivity_leak_event`, `mask_sensitivity_clean_event`
     - `event_deltas`: full list of per-event sensitivities and deltas across all independent FOMC meetings
     - Explicit metadata: `behavioral_inference: "DESCRIPTIVE_ONLY"`, `behavioral_statistical_unit: "independent_fomc_event"`, `behavioral_multiple_testing: "NOT_APPLICABLE"`, `behavioral_significance_testing: False`
   - Omitted any spurious p-value or underspecified FDR markers.

3. **Protocol Lock Manifest & Cryptographic Bindings (v1.2.3)**:
   - Updated `configs/phase4_protocol_lock.json` binding to Scientific Code Freeze Commit I (`e8426ec82ca3895a6b55a1e93cabbe235ddee6ad`) and Source Tree Hash (`04389535d0c60b5e615cb8fe64b6c3c595789f959ed9e33ae89d6990342dd0d2`).
   - All 12 controlled protocol files verified and locked.

---

## 2. Complete Phase 4 Gate Verification Matrix (Tests A through BB)

All 54 Gate tests (`tests/test_phase4a_gate.py` Tests A through BB + Metric Integration) and all 187 repository unit tests pass with 100% green status:

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
| **Test AT: Behavioral Analysis** | Descriptive secondary endpoint deltas | Descriptive only | PASSED |
| **Test AU: Binary Direction Co-Primary**| RidgeClassifier on rate change vs hold | Macro F1 | PASSED |
| **Test AV: Runtime Contract** | `resolve_phase4_runtime_contract` | Full reconciliation | PASSED |
| **Test AW: MLM Hyperparameters** | `max_steps: 100`, `scheduler: none` | Locked in configs | PASSED |
| **Test AX: Statistical Invariants** | 40 events, 32 OOS events, 181 anchors | Invariants hold | PASSED |
| **Test AY: Provenance Completeness** | Manifest records all 5 bindings | Verified | PASSED |
| **Test AZ: Mock Data Mode Rejection**| `allow_mock=False` rejects mock data | Fail-closed | PASSED |
| **Integration: Deterministic Metrics**| End-to-end evaluation on 40 events | Deterministic | PASSED |
| **Test BB: Ancestry Verification** | Allows doc-only descendant commits | Verified | PASSED |

---

## 3. Cryptographic Protocol Lock Manifest (v1.2.3)

- **Protocol Lock Manifest**: `configs/phase4_protocol_lock.json`
- **Protocol Version**: `1.2.3`
- **Source Tree Hash**: `04389535d0c60b5e615cb8fe64b6c3c595789f959ed9e33ae89d6990342dd0d2`
- **Scientific Code Freeze Commit (Commit I)**: `e8426ec82ca3895a6b55a1e93cabbe235ddee6ad`
- **Base Model Revision**: Locked to `ProsusAI/finbert` commit `4556d13015211d73dccd3fdd39d39232506f3e43`.
- **Controlled Files (12 Artifacts)**:
  1. `configs/phase4_preregistration.yaml` (`35b109a72acfc2ce3b3b18f5b802897d1d52a9e274d47b55c32063921cebeb6e`)
  2. `configs/phase4_confirmatory.yaml` (`91aae324b911455aef07b4736680bdbacfb857eeda19d6d567d2a296d1cb69d0`)
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
- **Secondary Behavioral Endpoint**: Descriptive stance sensitivity shift across 40 independent FOMC events (mean, median, std, per-event deltas; no confirmatory p-value/FDR claim)
- **Secondary Economic Endpoints**: 2Y Treasury yield change (primary) and SPY 1D return (secondary) evaluated via event-level IC with stationary block bootstrap
- **Authorization Guard**: Requires external signed authorization manifest at `configs/phase4_execution_authorization.json` binding protocol version `1.2.3`, lock SHA, freeze commit, and tree hash to initiate compute.
