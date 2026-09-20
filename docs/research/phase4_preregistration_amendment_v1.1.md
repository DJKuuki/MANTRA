# Preregistration Amendment Log: Version 1.1.0

**Document**: `docs/research/phase4_preregistration_amendment_v1.1.md`  
**Prior Specification**: Version 1.0.0 (`d439c256b4e29b9abe05f93f9085edb7639cce66`)  
**Amended Specification**: Version 1.1.0  
**Effective Date**: 2026-09-20  
**Amendment Type**: PRE-COMPUTE PROTOCOL CORRECTION  
**Execution Status**: PHASE 4B CONFIRMATORY MODELS REMAIN STRICTLY BLOCKED  

---

## 1. Formal Non-Contamination Declaration

> [!IMPORTANT]
> **Zero Prior Confirmatory Compute Guarantee**:
> No Phase 4B confirmatory language models, MLM pre-training branches, or downstream classifier probes were trained prior to this amendment. All modifications recorded herein constitute pre-compute hardening and formalization of machine-verifiable constraints identified during the external audit of the Phase 4A Preregistration Gate.

---

## 2. Summary of Modifications

### Item 1: Freezing of Actual 2020–2022 Contamination Corpus
- **Prior State**: The 2020–2022 contamination corpus was referenced as a prose description without a frozen on-disk repository artifact.
- **Amended State**: Formally constructed and frozen in `data/research/fomc/phase4_contamination/`. Consists of 50 official Federal Reserve post-cutoff documents (23 scheduled statements, 2 unscheduled emergency statements, 1 policy strategy statement, and 24 meeting minutes) totaling 237,273 words (> 308k tokens). Cryptographic SHA-256 (`54b3e84e8ef958e78032815a6cc76e1ca68cb0428afe71b390a583ee60e25c58`) and individual document metadata are verified and locked.

### Item 2: Correction of Primary Inferential Unit ($N_{\text{OOS}} = 32$ Events)
- **Prior State**: The inferential permutation test used fold-level score differences ($N=4$ CV folds), conflicting with the preregistered claim of an event-level sign-flip test.
- **Amended State**: Restructured the primary contrast statistic to operate strictly over individual Out-of-Sample (OOS) FOMC events:
  $$d_e = |y_e - \hat{y}_{\text{clean}}(e)| - |y_e - \hat{y}_{\text{leak}}(e)| \quad \forall e \in \{1, \dots, N_{\text{OOS}}\}$$
  The permutation test strictly flips signs across the $N_{\text{OOS}} = 32$ evaluated events ($B = 2,000$ permutations). CV folds are explicitly rejected as the permutation unit.

### Item 3: Explicit Modeling of Continuous Rate Change Targets
- **Prior State**: Task type was heuristically inferred at runtime (`len(unique_y) <= 5`), risking misclassifying discrete rate changes ($-0.25, 0.0, +0.25, +0.50$) as a classification task.
- **Amended State**: Added explicit `target_type: str = "continuous"` parameter to `evaluate_representational_leakage_grouped`. Continuous rate changes are strictly evaluated via Ridge Regression ($L_2$ regularization, $\alpha=1.0$), while binary change-vs-hold targets are strictly evaluated via RidgeClassifier. Runtime heuristic guessing is prohibited.

### Item 4: Expansion of Cryptographic Protocol Lock Manifest
- **Prior State**: Only the Markdown preregistration document hash was checked.
- **Amended State**: Established `configs/phase4_protocol_lock.json` locking 10 critical artifacts:
  - `phase4_preregistration.yaml`
  - `phase4_confirmatory.yaml`
  - `events.jsonl`
  - `anchors.jsonl`
  - `policy_history.csv`
  - `market_manifest.json`
  - `spy_daily_raw.csv`
  - `treasury_2y_raw.csv`
  - `contamination_documents.jsonl`
  - `contamination_manifest.json`
  All hashes are computed using LF newline normalization to ensure cross-platform reproducibility between Windows and Linux CI.

### Item 5: Strict Base Model Revision and Downstream Recipe Lock
- **Prior State**: Base checkpoint revision and downstream classifier hyperparameters were partially underspecified.
- **Amended State**: Base model is locked to `ProsusAI/finbert` commit `4556d13015211d73dccd3fdd39d39232506f3e43`. Downstream fine-tuning recipe is completely locked across epochs (3), batch size (16), learning rate ($2 \times 10^{-5}$), optimizer (AdamW), and sequence length (128). Same-seed causal symmetry is enforced.

### Item 6: 100% Verbatim Anchor & Source Document Verification
- **Prior State**: Test E randomly sampled 15 out of 181 anchors.
- **Amended State**: Test E and Test S now exhaustively verify all 181 / 181 paragraph anchors and all 40 / 40 source FOMC documents verbatim against local official Federal Reserve HTML sources. Sampling is eliminated.

### Item 7: Market Endpoint Naming & Provenance Formalization
- **Prior State**: Daily market return was labeled `spy_1d_return`, potentially ambiguous regarding event baseline.
- **Amended State**: Renamed canonical endpoint to `spy_next_close_return_from_event_close` (retaining `spy_1d_return` as alias). Market manifest records FRED series ID `DGS2`, symbol `SPY`, and explicit observation convention `daily_close_t_to_close_t_plus_1_not_intraday`.

---

## 3. Audit Verification Status
All 7 amendments have been implemented, cryptographically locked, and verified by automated gate tests. Phase 4B model execution remains blocked pending human approval.
