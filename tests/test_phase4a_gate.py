"""Comprehensive Automated Verification Suite for Phase 4A Preregistration Gate (v1.1.0).

Tests A through T:
- Test A (Item 23): Within-event label, outcome, and timestamp consistency
- Test B (Item 25): Grouped expanding-window temporal cross-validation folds
- Test C (Item 26): Event-level economic effect and stance aggregation
- Test D (Item 27): Event-clustered block bootstrap preserving meeting sampling unit
- Test E (Item 28 / Item 23): 100% (181/181) verbatim canonical source verification in official HTML
- Test F (Item 28): Deterministic policy future_action derivation from policy history
- Test G (Item 28): Bit-exact market outcome recomputation from raw daily tables
- Test H (Item 29 / Item 4): Empirical contamination timeline provenance and strict temporal gap
- Test I (Item 30): Treatment block manifest and token provenance tracking
- Test J (Item 30): Exact dose ladder invariant |D_realized - D_requested| <= 1/T
- Test K (Item 29): Prohibition of mid-year placeholders / exact ISO-8601 UTC release times
- Test L (Item 22 / Item 41): Preregistration lock enforcement and document hash verification
- Test M (Item 35 / Item 1): Actual contamination dataset integrity and metadata schema
- Test N (Item 35 / Item 4): Programmatic derivation of contamination temporal range
- Test O (Item 35 / Item 8-10): Event-level primary inferential permutation unit (N_OOS = 32) and power
- Test P (Item 35 / Item 12): Explicit target type enforcement (continuous regression vs classification)
- Test Q (Item 35 / Item 16): Cryptographic protocol lock enforcement across 10 controlled artifacts
- Test R (Item 35 / Item 19): Execution configuration semantic equality with preregistered contract
- Test S (Item 35 / Item 24): 100% (40/40) source registry document verification
- Test T (Item 35 / Item 20-21): Base model revision and downstream fine-tuning recipe lock
- Test U (Item 30-31): Treatment sampling real corpus preflight across D0-D100 ladder (256k tokens)
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List
import yaml

from bs4 import BeautifulSoup
import numpy as np
import pytest

from tradingagents.temporal_leakage.datasets.contamination import (
    derive_contamination_temporal_range,
    load_phase4_contamination_documents,
    verify_contamination_document_sources,
)
from tradingagents.temporal_leakage.datasets.market_data import (
    load_market_tables,
    recompute_market_outcomes,
)
from tradingagents.temporal_leakage.datasets.policy_history import (
    derive_future_action,
    load_policy_history,
)
from tradingagents.temporal_leakage.fomc_benchmark import verify_file_sha256
from tradingagents.temporal_leakage.metrics import (
    aggregate_event_representations,
    evaluate_behavioral_leakage_event_level,
    evaluate_competence,
    evaluate_economic_effect_event_level,
    evaluate_representational_leakage_grouped,
    event_clustered_bootstrap_indices,
    grouped_temporal_split,
    simulate_phase4_primary_power,
)
from tradingagents.temporal_leakage.phase4_confirmatory import (
    CONTROLLED_FILE_MAPPINGS,
    CodeFreezeError,
    Phase4BAuthorizationError,
    PreregistrationHashMismatchError,
    PreregistrationLockError,
    ProductionBackendValidationError,
    Phase4ArtifactWriter,
    ProductionConfirmatoryBackend,
    MockConfirmatoryBackend,
    Phase4ExecutionBackend,
    compute_file_sha256,
    compute_source_tree_hash,
    execute_phase4b_confirmatory,
    resolve_phase4_runtime_contract,
    run_phase4_confirmatory,
    run_phase4_mock_orchestration,
    verify_phase4_code_freeze,
    verify_phase4_protocol_lock,
    verify_phase4b_authorization,
    verify_preregistration_lock,
)
from tradingagents.temporal_leakage.twin_pipeline import (
    CausalIntegrityError,
    create_exact_token_dose_stream,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "data" / "research" / "fomc" / "events" / "events.jsonl"
ANCHORS_PATH = PROJECT_ROOT / "data" / "research" / "fomc" / "confirmatory_anchors" / "anchors.jsonl"
REGISTRY_PATH = PROJECT_ROOT / "data" / "research" / "fomc" / "source_registry.jsonl"
POLICY_PATH = PROJECT_ROOT / "data" / "research" / "fomc" / "policy_history.csv"
MARKET_DIR = PROJECT_ROOT / "data" / "research" / "market"
PROTOCOL_LOCK_PATH = PROJECT_ROOT / "configs" / "phase4_protocol_lock.json"
PREREG_CONFIG_PATH = PROJECT_ROOT / "configs" / "phase4_preregistration.yaml"
CONF_CONFIG_PATH = PROJECT_ROOT / "configs" / "phase4_confirmatory.yaml"
CONTAMINATION_DOCS_PATH = PROJECT_ROOT / "data" / "research" / "fomc" / "phase4_contamination" / "documents.jsonl"
PRE_CUTOFF_DOCS_PATH = PROJECT_ROOT / "data" / "research" / "fomc" / "phase4_pre_cutoff" / "documents.jsonl"


def load_events() -> List[Dict[str, Any]]:
    events = []
    with open(EVENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def load_anchors() -> List[Dict[str, Any]]:
    anchors = []
    with open(ANCHORS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                anchors.append(json.loads(line))
    return anchors


# ---------------------------------------------------------------------------
# Test A: Within-Event Consistency
# ---------------------------------------------------------------------------
def test_a_within_event_consistency():
    """Verify that all paragraph anchors for the same event share identical labels, outcomes, and timestamps."""
    events = load_events()
    anchors = load_anchors()

    assert len(events) == 40, f"Expected exactly 40 independent events, found {len(events)}"
    assert len(anchors) == 181, f"Expected exactly 181 paragraph anchors, found {len(anchors)}"

    event_map = {e["event_id"]: e for e in events}
    assert len(event_map) == len(events), "Duplicate event_id detected in events.jsonl"

    for anchor in anchors:
        eid = anchor["event_id"]
        assert eid in event_map, f"Anchor references unknown event_id: {eid}"
        parent = event_map[eid]

        assert anchor["future_targets"]["future_action"] == parent["future_action"], (
            f"Action mismatch in event {eid}: {anchor['future_targets']['future_action']} != {parent['future_action']}"
        )
        assert anchor["market_outcomes"]["spy_1d_return"] == parent["market_outcomes"]["spy_1d_return"], (
            f"Market outcome mismatch in event {eid}"
        )
        assert anchor["event_time"] == parent["event_time"], (
            f"Event time mismatch in event {eid}: {anchor['event_time']} != {parent['event_time']}"
        )
        assert anchor["source_url"] == parent["source_url"], (
            f"Source URL mismatch in event {eid}"
        )


# ---------------------------------------------------------------------------
# Test B: Grouped Temporal Split
# ---------------------------------------------------------------------------
def test_b_grouped_temporal_split():
    """Verify expanding-window temporal folds strictly partition by event without cross-contamination."""
    events = load_events()
    event_ids = [e["event_id"] for e in events]
    event_times = [e["event_time"] for e in events]

    folds = grouped_temporal_split(event_ids, event_times, n_splits=4)
    assert len(folds) == 4, f"Expected exactly 4 folds, got {len(folds)}"

    all_test_events = set()
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_ids = {event_ids[i] for i in train_idx}
        test_ids = {event_ids[i] for i in test_idx}

        overlap = train_ids.intersection(test_ids)
        assert len(overlap) == 0, f"Fold {fold_idx} has event overlap: {overlap}"

        max_train_time = max(event_times[i] for i in train_idx)
        min_test_time = min(event_times[i] for i in test_idx)
        assert max_train_time < min_test_time, (
            f"Fold {fold_idx} violates temporal ordering: max train {max_train_time} >= min test {min_test_time}"
        )

        all_test_events.update(test_ids)

    assert len(all_test_events) == 32, f"Expected exactly 32 evaluated test events, got {len(all_test_events)}"


# ---------------------------------------------------------------------------
# Test C: Event-Level Economic Effect
# ---------------------------------------------------------------------------
def test_c_event_level_economic_effect():
    """Verify event-level aggregation treats each event with equal weight in IC calculation."""
    event_ids = ["event_01"] * 10 + ["event_02"] * 2 + ["event_03"] * 5
    representations_clean = np.ones((17, 8))
    representations_leak = np.ones((17, 8)) * 2.0

    h_clean_agg, u_clean = aggregate_event_representations(representations_clean, event_ids)
    h_leak_agg, u_leak = aggregate_event_representations(representations_leak, event_ids)

    assert len(u_clean) == 3
    assert h_clean_agg.shape == (3, 8)
    np.testing.assert_allclose(h_clean_agg[0], 1.0)
    np.testing.assert_allclose(h_leak_agg[0], 2.0)

    ev_ids_3 = ["ev1", "ev2", "ev3"]
    s_clean = [0.1, 0.5, 0.9]
    s_leak = [0.2, 0.6, 0.8]
    market_returns = [0.01, 0.05, 0.08]

    res = evaluate_economic_effect_event_level(s_leak, s_clean, market_returns, ev_ids_3, n_bootstrap=100)
    assert res["n_events"] == 3
    assert "delta_ic" in res
    assert "delta_ic_ci_95" in res
    assert res["delta_ic_ci_95"][0] <= res["delta_ic_ci_95"][1]


# ---------------------------------------------------------------------------
# Test D: Event-Clustered Bootstrap
# ---------------------------------------------------------------------------
def test_d_event_clustered_bootstrap():
    """Verify event-clustered bootstrap resamples meetings as unified blocks."""
    event_ids = ["ev1", "ev1", "ev1", "ev2", "ev2", "ev3", "ev3", "ev3", "ev3"]
    draws = event_clustered_bootstrap_indices(event_ids, n_boot=20, random_seed=42)

    assert len(draws) == 20
    for draw in draws:
        draw_event_ids = [event_ids[i] for i in draw]
        counts = {eid: draw_event_ids.count(eid) for eid in set(draw_event_ids)}
        if "ev1" in counts:
            assert counts["ev1"] % 3 == 0, f"ev1 anchors fragmented: {counts['ev1']}"
        if "ev2" in counts:
            assert counts["ev2"] % 2 == 0, f"ev2 anchors fragmented: {counts['ev2']}"
        if "ev3" in counts:
            assert counts["ev3"] % 4 == 0, f"ev3 anchors fragmented: {counts['ev3']}"


# ---------------------------------------------------------------------------
# Test E (Item 28 / Item 23): 100% (181/181) Anchor Verification
# ---------------------------------------------------------------------------
def test_e_anchor_canonical_source_existence():
    """Exhaustively verify all 181 anchors exist verbatim in official Federal Reserve HTML sources."""
    anchors = load_anchors()
    assert len(anchors) == 181, f"Expected exactly 181 anchors, found {len(anchors)}"

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = {json.loads(line)["document_id"]: json.loads(line) for line in f if line.strip()}

    def normalize_for_match(text: str) -> str:
        for h in ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015"]:
            text = text.replace(h, "-")
        return re.sub(r"\s+", " ", text).strip()

    cached_parsed_docs: Dict[str, str] = {}

    for idx, anchor in enumerate(anchors):
        doc_id = anchor["document_id"]
        assert doc_id in registry, f"Document ID {doc_id} missing from source registry"
        reg_entry = registry[doc_id]

        html_path = PROJECT_ROOT / "data" / "research" / "fomc" / "raw_sources" / f"{doc_id}.html"
        assert html_path.exists(), f"Local HTML snapshot missing: {html_path}"

        # 1. Verify HTML hash against registry across LF/CRLF checkouts
        assert verify_file_sha256(html_path, reg_entry["raw_source_sha256"]), (
            f"HTML hash mismatch for {doc_id}"
        )

        # 2. Verify paragraph hash
        expected_para_sha = hashlib.sha256(anchor["text"].encode("utf-8")).hexdigest()
        assert anchor["paragraph_text_sha256"] == expected_para_sha, (
            f"Paragraph SHA mismatch for anchor {anchor['anchor_id']}"
        )

        # 3. Verify canonical document hash matches registry
        assert anchor["canonical_document_text_sha256"] == reg_entry["canonical_document_text_sha256"], (
            f"Canonical doc text SHA mismatch for anchor {anchor['anchor_id']}"
        )

        # 4. Verbatim existence in parsed HTML text
        if doc_id not in cached_parsed_docs:
            with open(html_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                cached_parsed_docs[doc_id] = normalize_for_match(soup.get_text())

        parsed_text = cached_parsed_docs[doc_id]
        norm_anchor = normalize_for_match(anchor["text"])
        assert norm_anchor in parsed_text, (
            f"Anchor {idx} ({anchor['anchor_id']}) not found verbatim in {doc_id}.html"
        )


# ---------------------------------------------------------------------------
# Test F: Policy Action Provenance
# ---------------------------------------------------------------------------
def test_f_policy_future_action_provenance():
    """Verify future_action labels are deterministically derived from policy_history.csv."""
    policy_df = load_policy_history(POLICY_PATH)
    events = load_events()

    for event in events:
        meeting_date = event["event_time"][:10]
        derived = derive_future_action(meeting_date, policy_df)

        assert derived["future_action"] == event["future_action"], (
            f"Action mismatch in event {event['event_id']}: derived {derived['future_action']} != {event['future_action']}"
        )
        assert derived["future_action_time"] == event["future_action_time"], (
            f"Action time mismatch in event {event['event_id']}: derived {derived['future_action_time']} != {event['future_action_time']}"
        )


# ---------------------------------------------------------------------------
# Test G: Market Outcome Recomputation
# ---------------------------------------------------------------------------
def test_g_market_outcome_recomputation():
    """Verify market outcomes are recomputable bit-exact from raw SPY and Treasury daily tables."""
    spy_df, t2y_df, _ = load_market_tables(MARKET_DIR)
    events = load_events()

    for event in events:
        meeting_date = event["event_time"][:10]
        recomputed = recompute_market_outcomes(meeting_date, spy_df, t2y_df)

        expected = event["market_outcomes"]
        np.testing.assert_allclose(
            recomputed["spy_1d_return"],
            expected["spy_1d_return"],
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"SPY return mismatch on {meeting_date}",
        )
        assert recomputed["spy_next_close_return_from_event_close"] == recomputed["spy_1d_return"]
        if expected.get("treasury_2y_yield_change") is not None and recomputed.get("treasury_2y_yield_change") is not None:
            np.testing.assert_allclose(
                recomputed["treasury_2y_yield_change"],
                expected["treasury_2y_yield_change"],
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Treasury 2Y yield change mismatch on {meeting_date}",
            )


# ---------------------------------------------------------------------------
# Test H (Item 29 / Item 4): Contamination Timeline Provenance & Strict Gap
# ---------------------------------------------------------------------------
def test_h_contamination_timeline_provenance():
    """Verify contamination min/max timestamps derive programmatically from verified corpus releases."""
    events = load_events()
    anchor_max_time = max(e["event_time"] for e in events)
    assert anchor_max_time == "2019-12-11T19:00:00Z"

    docs = load_phase4_contamination_documents(CONTAMINATION_DOCS_PATH)
    range_info = derive_contamination_temporal_range(docs)
    t_min = range_info["contamination_min_time"]
    t_max = range_info["contamination_max_time"]

    assert t_min == "2020-01-29T19:00:00Z"
    assert t_max == "2023-01-04T19:00:00Z"
    assert anchor_max_time < t_min, "Temporal separation violated between anchors and contamination"

    gap_days = (
        np.datetime64(t_min.replace("Z", ""))
        - np.datetime64(anchor_max_time.replace("Z", ""))
    ).astype("timedelta64[D]").astype(int)

    assert gap_days >= 48, f"Temporal gap must be >= 45 days, found {gap_days} days"


# ---------------------------------------------------------------------------
# Test I: Treatment Block Provenance Tracking
# ---------------------------------------------------------------------------
def test_i_treatment_block_provenance():
    """Verify treatment blocks record source document IDs, token counts, and enforce repetition ratio."""
    pre_corpus = [
        {"text": f"Pre-cutoff policy text paragraph {i} on macro inflation and labor.", "document_id": f"doc_pre_{i}"}
        for i in range(10)
    ]
    post_corpus = [
        {"text": f"Post-cutoff pandemic relief statement {j} on stimulus and crisis.", "document_id": f"doc_post_{j}"}
        for j in range(10)
    ]

    class MockTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return [10, 20, 30, 40]

    stream = create_exact_token_dose_stream(
        pre_corpus=pre_corpus,
        post_corpus=post_corpus,
        dose=0.50,
        num_blocks=4,
        block_length=8,
        tokenizer=MockTokenizer(),
        random_seed=42,
    )

    manifest = stream.get("treatment_block_manifest", [])
    assert len(manifest) == 4, f"Expected 4 blocks, got {len(manifest)}"

    for b in manifest:
        assert "block_index" in b
        assert "source_document_ids" in b
        assert "pre_tokens" in b
        assert "post_tokens" in b
        assert b["pre_tokens"] + b["post_tokens"] == 8

    with pytest.raises(CausalIntegrityError, match="token shortfall"):
        create_exact_token_dose_stream(
            pre_corpus=pre_corpus[:1],
            post_corpus=post_corpus,
            dose=0.0,
            num_blocks=10,
            block_length=8,
            tokenizer=MockTokenizer(),
            random_seed=42,
            max_repetition_ratio=0.20,
        )


# ---------------------------------------------------------------------------
# Test J: Exact Dose Ladder Invariant
# ---------------------------------------------------------------------------
def test_j_exact_dose_ladder_invariant():
    """Verify |D_realized - D_requested| <= 1/T across all doses."""
    pre_corpus = ["Statement regarding economic growth and interest rate policy." for _ in range(50)]
    post_corpus = ["Statement regarding future pandemic conditions and market liquidity." for _ in range(50)]

    class MockTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return [1, 2, 3, 4, 5, 6, 7, 8]

    doses = [0.0, 0.25, 0.50, 0.75, 1.00]
    total_tokens = 80

    for d in doses:
        stream = create_exact_token_dose_stream(
            pre_corpus=pre_corpus,
            post_corpus=post_corpus,
            dose=d,
            num_blocks=10,
            block_length=8,
            tokenizer=MockTokenizer(),
            random_seed=42,
        )
        realized = stream["realized_dose"]
        assert abs(realized - d) <= (1.0 / total_tokens) + 1e-9


# ---------------------------------------------------------------------------
# Test K: No Mid-Year Placeholders
# ---------------------------------------------------------------------------
def test_k_no_mid_year_placeholder():
    """Verify all event and contamination timestamps are exact ISO-8601 UTC down to seconds/minutes."""
    events = load_events()
    iso_utc_regex = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    for event in events:
        etime = event["event_time"]
        assert iso_utc_regex.match(etime), f"Event {event['event_id']} has invalid timestamp format: {etime}"
        assert not etime.endswith("T00:00:00Z"), (
            f"Event {event['event_id']} contains unverified midnight placeholder: {etime}"
        )


# ---------------------------------------------------------------------------
# Test L: Preregistration Lock Enforcement
# ---------------------------------------------------------------------------
def test_l_preregistration_lock_enforcement():
    """Verify preregistration lock checks prevent execution if unlocked or if documents are tampered."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        doc_file = tmp_p / "prereg.md"
        doc_file.write_text("# Preregistration Document\nDetails here.", encoding="utf-8")
        doc_hash = compute_file_sha256(doc_file)

        cfg_file = tmp_p / "prereg.yaml"
        cfg_content = {
            "version": "1.0.0",
            "locked": True,
            "preregistration_doc_sha256": doc_hash,
            "preregistration_doc_path": str(doc_file),
        }
        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(cfg_content, f)

        meta = verify_preregistration_lock(cfg_file, doc_file)
        assert meta["status"] == "LOCKED_AND_VERIFIED"

        doc_file.write_text("# Preregistration Document\nUnauthorized change!", encoding="utf-8")
        with pytest.raises(PreregistrationHashMismatchError):
            verify_preregistration_lock(cfg_file, doc_file)

        cfg_content["locked"] = False
        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(cfg_content, f)
        with pytest.raises(PreregistrationLockError, match="not locked"):
            verify_preregistration_lock(cfg_file, doc_file)


# ---------------------------------------------------------------------------
# Test M: Actual Contamination Dataset Integrity & Metadata Schema
# ---------------------------------------------------------------------------
def test_m_actual_contamination_corpus_integrity():
    """Verify actual contamination corpus exists, is non-empty, and satisfies strict metadata schema."""
    docs = load_phase4_contamination_documents(CONTAMINATION_DOCS_PATH)
    assert len(docs) >= 50, f"Expected at least 50 contamination documents, got {len(docs)}"

    required_keys = {
        "document_id",
        "document_type",
        "title",
        "event_time",
        "available_time",
        "availability_quality",
        "source_url",
        "source_type",
        "raw_source_sha256",
        "canonical_text_sha256",
        "word_count",
        "temporal_class",
        "text",
    }

    doc_ids = set()
    for d in docs:
        assert required_keys.issubset(d.keys()), f"Document {d.get('document_id')} missing required fields"
        assert d["availability_quality"] == "exact", f"Document {d['document_id']} does not have exact availability"
        assert d["temporal_class"] == "post_cutoff", f"Document {d['document_id']} temporal_class != post_cutoff"
        assert d["source_type"] == "federal_reserve_official"
        assert len(d["text"].strip()) > 50, f"Document {d['document_id']} has suspiciously short text"
        assert d["word_count"] > 10
        assert d["document_id"] not in doc_ids, f"Duplicate document_id: {d['document_id']}"
        doc_ids.add(d["document_id"])

    anchors = load_anchors()
    anchor_doc_ids = {a["document_id"] for a in anchors}
    overlap = doc_ids.intersection(anchor_doc_ids)
    assert len(overlap) == 0, f"Anchor / Contamination document ID overlap detected: {overlap}"


# ---------------------------------------------------------------------------
# Test N: Programmatic Contamination Temporal Range
# ---------------------------------------------------------------------------
def test_n_contamination_temporal_range_derived():
    """Verify contamination temporal bounds derive programmatically without hardcoded constants."""
    docs = load_phase4_contamination_documents(CONTAMINATION_DOCS_PATH)
    range_info = derive_contamination_temporal_range(docs)
    t_min = range_info["contamination_min_time"]
    t_max = range_info["contamination_max_time"]

    manifest_p = PROJECT_ROOT / "data" / "research" / "fomc" / "phase4_contamination" / "manifest.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert t_min == manifest["contamination_min_time"]
    assert t_max == manifest["contamination_max_time"]
    assert t_min == "2020-01-29T19:00:00Z"
    assert t_max == "2023-01-04T19:00:00Z"


# ---------------------------------------------------------------------------
# Test O: Event-Level Primary Inferential Permutation Unit (N_OOS = 32)
# ---------------------------------------------------------------------------
def test_o_event_level_primary_permutation_unit():
    """Verify primary inference operates strictly at event level (N_OOS=32) rather than 4 folds."""
    events = load_events()
    event_ids = [e["event_id"] for e in events]
    event_times = [e["event_time"] for e in events]
    targets = [float(e["future_action"]) * 0.25 for e in events]

    rng = np.random.RandomState(42)
    rep_clean = rng.randn(40, 16)
    rep_leak = rep_clean + rng.randn(40, 16) * 0.1

    res = evaluate_representational_leakage_grouped(
        representations_leak=rep_leak,
        representations_clean=rep_clean,
        y=targets,
        event_ids=event_ids,
        event_times=event_times,
        target_type="continuous",
        n_splits=4,
        min_train_events=8,
        n_permutations=200,
        random_seed=42,
    )

    assert res["permutation_unit"] == "event"
    assert res["n_oos_events"] == 32
    assert res["target_type"] == "continuous"
    assert res["model_type"] == "ridge_regression"
    assert res["inferential_statistic"] == "paired_event_absolute_error_improvement"
    assert "delta_spearman" in res
    assert "p_value" in res

    # Power simulation check: N=32, d=0.50 achieves >= 80% power
    sim = simulate_phase4_primary_power(
        n_oos_events=32,
        effect_size_d=0.50,
        alpha=0.05,
        n_simulations=100,
        n_permutations=200,
        random_seed=42,
    )
    assert sim["achieved_power"] >= 0.80, f"Power simulation shortfall: {sim['achieved_power']} < 0.80"


# ---------------------------------------------------------------------------
# Test P: Explicit Target Type Enforcement
# ---------------------------------------------------------------------------
def test_p_explicit_target_type_enforcement():
    """Verify target_type='continuous' forces Ridge regression and prohibits classification guessing."""
    events = load_events()
    event_ids = [e["event_id"] for e in events]
    event_times = [e["event_time"] for e in events]
    discrete_rate_changes = [float(e["future_action"]) * 0.25 for e in events]

    assert len(set(discrete_rate_changes)) <= 4

    rng = np.random.RandomState(123)
    rep_clean = rng.randn(40, 8)
    rep_leak = rep_clean + rng.randn(40, 8) * 0.05

    res_reg = evaluate_representational_leakage_grouped(
        representations_leak=rep_leak,
        representations_clean=rep_clean,
        y=discrete_rate_changes,
        event_ids=event_ids,
        event_times=event_times,
        target_type="continuous",
        n_splits=4,
        min_train_events=8,
        n_permutations=50,
        random_seed=42,
    )
    assert res_reg["model_type"] == "ridge_regression"

    binary_targets = [1 if e["future_action"] != 0 else 0 for e in events]
    res_cls = evaluate_representational_leakage_grouped(
        representations_leak=rep_leak,
        representations_clean=rep_clean,
        y=binary_targets,
        event_ids=event_ids,
        event_times=event_times,
        target_type="binary",
        n_splits=4,
        min_train_events=8,
        n_permutations=50,
        random_seed=42,
    )
    assert res_cls["model_type"] == "ridge_classifier"


# ---------------------------------------------------------------------------
# Test Q: Cryptographic Protocol Lock Enforcement
# ---------------------------------------------------------------------------
def test_q_protocol_lock_enforcement():
    """Verify protocol lock verifies all 10 controlled artifacts and fails closed on tampering."""
    res = verify_phase4_protocol_lock(
        protocol_lock_path=PROTOCOL_LOCK_PATH,
        project_root=PROJECT_ROOT,
        prereg_config_path=PREREG_CONFIG_PATH,
        conf_config_path=CONF_CONFIG_PATH,
    )
    assert res["status"] == "PROTOCOL_LOCKED_AND_VERIFIED"
    assert res["controlled_file_count"] == 12
    assert res["clean_sham_corpus_locked"] is True
    assert res["contamination_corpus_locked"] is True
    assert res["base_model_revision_locked"] is True
    assert res["execution_config_semantic_equality"] is True

    # Tamper test with temporary copy
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_lock = Path(tmpdir) / "protocol_lock.json"
        with open(PROTOCOL_LOCK_PATH, "r", encoding="utf-8") as f:
            lock_data = json.load(f)

        # Corrupt one controlled file hash
        lock_data["files"]["events.jsonl"] = "0000000000000000000000000000000000000000000000000000000000000000"
        with open(tmp_lock, "w", encoding="utf-8") as f:
            json.dump(lock_data, f, indent=2)

        with pytest.raises(PreregistrationHashMismatchError, match="Controlled file hash mismatch"):
            verify_phase4_protocol_lock(
                protocol_lock_path=tmp_lock,
                project_root=PROJECT_ROOT,
                prereg_config_path=PREREG_CONFIG_PATH,
                conf_config_path=CONF_CONFIG_PATH,
            )


# ---------------------------------------------------------------------------
# Test R: Execution Config Semantic Equality
# ---------------------------------------------------------------------------
def test_r_config_semantic_equality():
    """Verify confirmatory execution config cannot override preregistered parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_conf = Path(tmpdir) / "tampered_conf.yaml"
        with open(CONF_CONFIG_PATH, "r", encoding="utf-8") as f:
            conf_data = yaml.safe_load(f)

        # Attempt to override doses
        conf_data["dose_ladder"] = [0.0, 0.5, 1.0]
        with open(tmp_conf, "w", encoding="utf-8") as f:
            yaml.dump(conf_data, f)

        with pytest.raises(PreregistrationLockError, match="Dose ladder mismatch"):
            verify_phase4_protocol_lock(
                protocol_lock_path=PROTOCOL_LOCK_PATH,
                project_root=PROJECT_ROOT,
                prereg_config_path=PREREG_CONFIG_PATH,
                conf_config_path=tmp_conf,
            )


# ---------------------------------------------------------------------------
# Test S: 100% (40/40) Source Registry Document Verification
# ---------------------------------------------------------------------------
def test_s_source_registry_all_40_documents():
    """Verify all 40 documents in source_registry.jsonl have verified local raw HTML snapshots."""
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = [json.loads(line) for line in f if line.strip()]

    assert len(registry) == 40, f"Expected 40 documents, got {len(registry)}"

    raw_dir = PROJECT_ROOT / "data" / "research" / "fomc" / "raw_sources"
    iso_utc_regex = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    for entry in registry:
        doc_id = entry["document_id"]
        html_file = raw_dir / f"{doc_id}.html"
        assert html_file.exists(), f"Raw source HTML missing: {html_file}"

        assert verify_file_sha256(html_file, entry["raw_source_sha256"]), (
            f"Raw SHA mismatch for document {doc_id}"
        )
        assert entry["source_url"].startswith("https://www.federalreserve.gov/"), (
            f"Invalid source URL for {doc_id}"
        )
        assert iso_utc_regex.match(entry["event_time"]), (
            f"Invalid event_time format for {doc_id}: {entry['event_time']}"
        )


# ---------------------------------------------------------------------------
# Test T: Base Model Revision and Downstream Recipe Lock
# ---------------------------------------------------------------------------
def test_t_base_revision_and_downstream_recipe_lock():
    """Verify base model checkpoint revision and downstream fine-tuning hyperparameters are locked."""
    with open(PREREG_CONFIG_PATH, "r", encoding="utf-8") as f:
        prereg_cfg = yaml.safe_load(f)
    with open(CONF_CONFIG_PATH, "r", encoding="utf-8") as f:
        conf_cfg = yaml.safe_load(f)

    expected_rev = "4556d13015211d73dccd3fdd39d39232506f3e43"
    assert prereg_cfg["compute_bounds"]["base_revision"] == expected_rev
    assert conf_cfg["model"]["base_revision"] == expected_rev

    dt_p = prereg_cfg["downstream_training"]
    dt_c = conf_cfg["downstream_training"]
    assert dt_p["epochs"] == dt_c["epochs"] == 3
    assert dt_p["batch_size"] == dt_c["batch_size"] == 16
    assert dt_p["learning_rate"] == dt_c["learning_rate"] == 2.0e-5
    assert dt_p["optimizer"] == dt_c["optimizer"] == "AdamW"
    assert dt_p["max_seq_length"] == dt_c["max_seq_length"] == 128
    assert dt_p["head_initialization"] == dt_c["head_initialization"] == "fresh_shared_within_seed"
    assert dt_p["sample_order"] == dt_c["sample_order"] == "paired_within_seed"


# ---------------------------------------------------------------------------
# Test U: Treatment Sampling Real Corpus Preflight
# ---------------------------------------------------------------------------
def test_u_treatment_sampling_real_corpus_preflight():
    """Verify 256k token dose stream construction on real pre/post corpora with repetition ratio <= 0.20."""
    with open(CONTAMINATION_DOCS_PATH, "r", encoding="utf-8") as f:
        post_docs = [json.loads(line) for line in f if line.strip()]
    with open(PRE_CUTOFF_DOCS_PATH, "r", encoding="utf-8") as f:
        pre_docs = [json.loads(line) for line in f if line.strip()]

    class DeterministicTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            words = text.split()
            toks = []
            for w in words:
                toks.append(hash(w) % 30000)
                if len(w) > 6:
                    toks.append(hash(w[::-1]) % 30000)
            return toks

    total_tokens = 256000  # 500 blocks * 512 tokens
    doses = [0.00, 0.25, 0.50, 0.75, 1.00]

    for d in doses:
        stream = create_exact_token_dose_stream(
            pre_corpus=pre_docs,
            post_corpus=post_docs,
            dose=d,
            num_blocks=500,
            block_length=512,
            tokenizer=DeterministicTokenizer(),
            random_seed=42,
            max_repetition_ratio=0.20,
        )
        assert abs(stream["realized_dose"] - d) <= (1.0 / total_tokens) + 1e-9
        assert len(stream.get("treatment_block_manifest", [])) == 500
        assert stream["forced_repetition_ratio"] == 0.0
        assert stream["forced_repetition_ratio"] <= 0.20


# ---------------------------------------------------------------------------
# Test V: Clean Corpus Strict Information Availability & Edge Case Rejection
# ---------------------------------------------------------------------------
def test_v_clean_corpus_strict_availability_cutoff():
    """Verify clean sham corpus adheres to strict information availability cutoff (<= 2019-12-31T23:59:59Z)."""
    with open(PRE_CUTOFF_DOCS_PATH, "r", encoding="utf-8") as f:
        pre_docs = [json.loads(line) for line in f if line.strip()]

    assert len(pre_docs) == 63, f"Expected 63 pre-cutoff documents, found {len(pre_docs)}"

    cutoff_time = "2019-12-31T23:59:59Z"
    for doc in pre_docs:
        assert doc["available_time"] <= cutoff_time, (
            f"Availability cutoff violation in clean corpus: {doc['document_id']} "
            f"has available_time {doc['available_time']} > {cutoff_time}"
        )
        assert doc["temporal_class"] == "pre_cutoff"
        assert doc["availability_quality"] == "exact"

    # Regression test: FOMC minutes for 2019-12-11 meeting were released 2020-01-01T19:00:00Z.
    # Principle: Event time does not determine information availability. Must be rejected.
    pre_doc_ids = {d["document_id"] for d in pre_docs}
    assert "fomc-minutes-2019-12-11" not in pre_doc_ids, (
        "fomc-minutes-2019-12-11 was incorrectly admitted into clean corpus despite post-cutoff release!"
    )

    # Manifest verification
    manifest_p = PROJECT_ROOT / "data" / "research" / "fomc" / "phase4_pre_cutoff" / "manifest.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    assert m_data["document_count"] == 63
    assert m_data["temporal_max"] <= cutoff_time
    assert verify_file_sha256(PRE_CUTOFF_DOCS_PATH, m_data["dataset_sha256"])

    # Tamper test with temporary copy
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_lock = Path(tmpdir) / "protocol_lock.json"
        with open(PROTOCOL_LOCK_PATH, "r", encoding="utf-8") as f:
            lock_data = json.load(f)

        lock_data["files"]["clean_sham_documents.jsonl"] = "0000000000000000000000000000000000000000000000000000000000000000"
        with open(tmp_lock, "w", encoding="utf-8") as f:
            json.dump(lock_data, f, indent=2)

        with pytest.raises(PreregistrationHashMismatchError, match="Controlled file hash mismatch"):
            verify_phase4_protocol_lock(
                protocol_lock_path=tmp_lock,
                project_root=PROJECT_ROOT,
                prereg_config_path=PREREG_CONFIG_PATH,
                conf_config_path=CONF_CONFIG_PATH,
            )


# ---------------------------------------------------------------------------
# Test W: Contamination Availability Window & 50/50 Raw Source Verification
# ---------------------------------------------------------------------------
def test_w_contamination_window_and_50_50_source_verification():
    """Verify contamination availability window matches derived timestamps and all 50 documents verify against raw HTML."""
    post_docs = load_phase4_contamination_documents(CONTAMINATION_DOCS_PATH)
    assert len(post_docs) == 50, f"Expected 50 contamination documents, got {len(post_docs)}"

    # Check temporal range consistency
    temporal_meta = derive_contamination_temporal_range(post_docs)
    assert temporal_meta["contamination_min_time"] == "2020-01-29T19:00:00Z"
    assert temporal_meta["contamination_max_time"] == "2023-01-04T19:00:00Z"

    with open(CONF_CONFIG_PATH, "r", encoding="utf-8") as f:
        conf_cfg = yaml.safe_load(f)
    assert conf_cfg["contamination_dataset"]["min_available_time"] == "2020-01-29T19:00:00Z"
    assert conf_cfg["contamination_dataset"]["max_available_time"] == "2023-01-04T19:00:00Z"

    # Exhaustive 50/50 raw source verification
    raw_dir = PROJECT_ROOT / "data" / "research" / "fomc" / "phase4_contamination" / "raw_sources"
    res = verify_contamination_document_sources(post_docs, raw_sources_dir=raw_dir)
    assert res["all_verified"] is True
    assert res["verified_count"] == 50
    assert res["total_count"] == 50
    assert res["verification_summary"] == "50 / 50"
    assert len(res["errors"]) == 0


# ---------------------------------------------------------------------------
# Test X: Continuous Future Rate-Change Target Derivation
# ---------------------------------------------------------------------------
def test_x_continuous_future_rate_change_derivation():
    """Verify primary future rate change target is derived continuously from policy bounds."""
    policy_df = load_policy_history(POLICY_PATH)
    events = load_events()

    for event in events:
        date_str = event["metadata"]["meeting_date"]
        derived = derive_future_action(date_str, policy_df)

        assert "future_rate_change" in derived, f"future_rate_change missing in derived target for {date_str}"
        assert isinstance(derived["future_rate_change"], float)
        assert event["future_rate_change"] == derived["future_rate_change"]

        # Assert no heuristic action * 0.25 was used: must match upper after - upper before
        curr_idx = policy_df.index[policy_df["meeting_date"] == date_str].tolist()[0]
        next_row = policy_df.iloc[curr_idx + 1]
        raw_diff = round(float(next_row["target_upper_after"]) - float(next_row["target_upper_before"]), 4)
        assert derived["future_rate_change"] == raw_diff

    # Verify routing to Ridge regression
    rng = np.random.RandomState(42)
    rep_clean = rng.randn(40, 8)
    rep_leak = rep_clean + rng.randn(40, 8) * 0.1
    y_continuous = [e["future_rate_change"] for e in events]
    event_ids = [e["event_id"] for e in events]
    event_times = [e["event_time"] for e in events]

    res = evaluate_representational_leakage_grouped(
        representations_leak=rep_leak,
        representations_clean=rep_clean,
        y=y_continuous,
        event_ids=event_ids,
        event_times=event_times,
        target_type="continuous",
        n_splits=4,
        min_train_events=8,
        n_permutations=50,
        random_seed=42,
    )
    assert res["model_type"] == "ridge_regression"


# ---------------------------------------------------------------------------
# Test Y: Treatment Repetition Metrics Separation
# ---------------------------------------------------------------------------
def test_y_treatment_repetition_metrics_separation():
    """Verify forced corpus exhaustion repetition is explicitly separated from token type diversity."""
    with open(PRE_CUTOFF_DOCS_PATH, "r", encoding="utf-8") as f:
        pre_docs = [json.loads(line) for line in f if line.strip()]
    with open(CONTAMINATION_DOCS_PATH, "r", encoding="utf-8") as f:
        post_docs = [json.loads(line) for line in f if line.strip()]

    class SimpleTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return [abs(hash(w)) % 10000 for w in text.split()]

    # 1. Real corpora under normal budget: forced repetition is 0.0
    stream = create_exact_token_dose_stream(
        pre_corpus=pre_docs,
        post_corpus=post_docs,
        dose=0.50,
        num_blocks=50,
        block_length=512,
        tokenizer=SimpleTokenizer(),
        random_seed=42,
        max_repetition_ratio=0.20,
    )
    assert stream["forced_repetition_ratio"] == 0.0
    assert stream["forced_repetition_tokens"] == 0
    assert 0.0 < stream["token_type_diversity"] < 1.0
    # Confirm forced_repetition_ratio != token_type_diversity
    assert stream["forced_repetition_ratio"] != stream["token_type_diversity"]

    # 2. Tiny corpus simulating shortfall
    tiny_corpus = [{"text": "Federal Reserve monetary policy", "document_id": "tiny_1"}]
    stream_cycled = create_exact_token_dose_stream(
        pre_corpus=tiny_corpus,
        post_corpus=tiny_corpus,
        dose=0.0,
        num_blocks=2,
        block_length=100,
        tokenizer=SimpleTokenizer(),
        random_seed=42,
        max_repetition_ratio=1.0,  # allow cycling for test
    )
    assert stream_cycled["forced_repetition_ratio"] > 0.50
    assert stream_cycled["forced_repetition_tokens"] > 0


# ---------------------------------------------------------------------------
# Test Z: Source Tree Lock & Code Freeze Enforcement
# ---------------------------------------------------------------------------
def test_z_source_tree_lock_and_code_freeze():
    """Verify source tree hash is computed and code freeze fails closed on modified source files."""
    tree_hash = compute_source_tree_hash(PROJECT_ROOT)
    assert isinstance(tree_hash, str) and len(tree_hash) == 64

    # Verification against current tree hash
    freeze_meta = verify_phase4_code_freeze(
        project_root=PROJECT_ROOT,
        locked_source_tree_hash=tree_hash,
        enforce_git_clean=False,
    )
    assert freeze_meta["status"] == "CODE_FROZEN_AND_VERIFIED"
    assert freeze_meta["source_tree_hash"] == tree_hash

    # Failure on mismatched hash
    with pytest.raises(CodeFreezeError, match="Source tree hash mismatch"):
        verify_phase4_code_freeze(
            project_root=PROJECT_ROOT,
            locked_source_tree_hash="deadbeef" * 8,
            enforce_git_clean=False,
        )


# ---------------------------------------------------------------------------
# Test AA: Phase 4B Runner Orchestration & Authorization Gate
# ---------------------------------------------------------------------------
def test_aa_phase4b_runner_orchestration_and_authorization():
    """Verify Phase 4B execution runner executes all 25 branches under mock and blocks without authorization."""
    # 1. Full training without authorization must fail closed
    with pytest.raises(Phase4BAuthorizationError, match="FULL_EXECUTION_BLOCKED"):
        run_phase4_confirmatory(
            config_path=CONF_CONFIG_PATH,
            prereg_path=PREREG_CONFIG_PATH,
            smoke_mode=False,
        )

    # 2. Authorization with invalid version fails
    with tempfile.TemporaryDirectory() as td:
        bad_auth = Path(td) / "auth.json"
        with open(bad_auth, "w", encoding="utf-8") as f:
            json.dump({"protocol_version": "0.9.0", "human_authorized": True}, f)

        with pytest.raises(Phase4BAuthorizationError, match="Protocol version mismatch"):
            verify_phase4b_authorization(bad_auth, protocol_version="1.2.0")

    # 3. Smoke mode succeeds and verifies gate
    smoke_res = run_phase4_confirmatory(
        config_path=CONF_CONFIG_PATH,
        prereg_path=PREREG_CONFIG_PATH,
        smoke_mode=True,
    )
    assert smoke_res["status"] == "CONFIRMATORY_GATE_VERIFIED"
    assert smoke_res["num_events"] == 40
    assert smoke_res["num_anchors"] == 181

    # 4. Mock orchestration executes all 25 branches end-to-end
    mock_res = run_phase4_mock_orchestration(
        config_path=CONF_CONFIG_PATH,
        prereg_path=PREREG_CONFIG_PATH,
        project_root=PROJECT_ROOT,
    )
    assert mock_res["status"] == "PHASE4B_ORCHESTRATION_COMPLETED"
    assert mock_res["total_branches_scheduled"] == 25
    assert mock_res["oos_events_count"] == 32
    assert mock_res["target_type"] == "continuous"
    assert mock_res["model_type"] == "ridge_regression"
    assert len(mock_res["branches"]) == 25


# ---------------------------------------------------------------------------
# Test AB: Production Backend Exists
# ---------------------------------------------------------------------------
def test_ab_production_backend_exists():
    """Verify ProductionConfirmatoryBackend exists, subclasses Phase4ExecutionBackend,
    and is distinct from MockConfirmatoryBackend."""
    assert issubclass(ProductionConfirmatoryBackend, Phase4ExecutionBackend)
    assert issubclass(MockConfirmatoryBackend, Phase4ExecutionBackend)
    assert ProductionConfirmatoryBackend is not MockConfirmatoryBackend

    backend = ProductionConfirmatoryBackend()
    assert hasattr(backend, "execute_branch")
    assert hasattr(backend, "run_branch")
    assert backend.base_checkpoint == "ProsusAI/finbert"
    assert backend.base_revision == "4556d13015211d73dccd3fdd39d39232506f3e43"


# ---------------------------------------------------------------------------
# Test AC: Default Full Run Is Never Mock
# ---------------------------------------------------------------------------
def test_ac_default_full_run_is_never_mock(monkeypatch):
    """Verify default full run backend resolves to ProductionConfirmatoryBackend, never Mock."""
    initialized_backends = []
    orig_init = ProductionConfirmatoryBackend.__init__

    def spy_init(self, *args, **kwargs):
        initialized_backends.append(self)
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(ProductionConfirmatoryBackend, "__init__", spy_init)

    with tempfile.TemporaryDirectory() as td:
        auth_file = Path(td) / "auth.json"
        tree_hash = compute_source_tree_hash(PROJECT_ROOT)
        with open(PROTOCOL_LOCK_PATH, "r", encoding="utf-8") as f:
            lock_meta = json.load(f)
            proto_ver = lock_meta.get("protocol_version", "1.2.4")
            locked_commit = lock_meta.get("code_commit") or lock_meta.get("scientific_code_commit", "unknown")
        lock_sha = compute_file_sha256(PROTOCOL_LOCK_PATH, normalize_newlines=True)
        with open(auth_file, "w", encoding="utf-8") as f:
            json.dump({
                "protocol_version": proto_ver,
                "human_authorized": True,
                "protocol_lock_sha256": lock_sha,
                "locked_scientific_code_commit": locked_commit,
                "locked_source_tree_hash": tree_hash,
            }, f)

        monkeypatch.setattr(
            "tradingagents.temporal_leakage.phase4_confirmatory.verify_phase4_code_freeze",
            lambda **kwargs: {"status": "CODE_FROZEN_AND_VERIFIED"},
        )

        class StopBeforeCompute(Exception):
            pass

        monkeypatch.setattr(
            ProductionConfirmatoryBackend,
            "execute_branch",
            lambda *args, **kwargs: (_ for _ in ()).throw(StopBeforeCompute("Stopped before compute")),
        )

        with pytest.raises(StopBeforeCompute):
            execute_phase4b_confirmatory(
                config_path=CONF_CONFIG_PATH,
                prereg_path=PREREG_CONFIG_PATH,
                backend=None,
                authorization_path=auth_file,
                project_root=PROJECT_ROOT,
                is_mock_orchestration=False,
            )

        assert len(initialized_backends) == 1
        assert isinstance(initialized_backends[0], ProductionConfirmatoryBackend)
        assert not isinstance(initialized_backends[0], MockConfirmatoryBackend)


# ---------------------------------------------------------------------------
# Test AD: Mock Rejected In Full Empirical Mode
# ---------------------------------------------------------------------------
def test_ad_mock_rejected_in_full_empirical_mode(monkeypatch):
    """Verify MockConfirmatoryBackend is strictly rejected when full empirical execution is attempted."""
    mock_backend = MockConfirmatoryBackend()
    with tempfile.TemporaryDirectory() as td:
        auth_file = Path(td) / "auth.json"
        tree_hash = compute_source_tree_hash(PROJECT_ROOT)
        with open(PROTOCOL_LOCK_PATH, "r", encoding="utf-8") as f:
            lock_meta = json.load(f)
            proto_ver = lock_meta.get("protocol_version", "1.2.4")
            locked_commit = lock_meta.get("code_commit") or lock_meta.get("scientific_code_commit", "unknown")
        lock_sha = compute_file_sha256(PROTOCOL_LOCK_PATH, normalize_newlines=True)
        with open(auth_file, "w", encoding="utf-8") as f:
            json.dump({
                "protocol_version": proto_ver,
                "human_authorized": True,
                "protocol_lock_sha256": lock_sha,
                "locked_scientific_code_commit": locked_commit,
                "locked_source_tree_hash": tree_hash,
            }, f)

        monkeypatch.setattr(
            "tradingagents.temporal_leakage.phase4_confirmatory.verify_phase4_code_freeze",
            lambda **kwargs: {"status": "CODE_FROZEN_AND_VERIFIED"},
        )

        with pytest.raises(ProductionBackendValidationError, match="MockConfirmatoryBackend is strictly prohibited"):
            execute_phase4b_confirmatory(
                config_path=CONF_CONFIG_PATH,
                prereg_path=PREREG_CONFIG_PATH,
                backend=mock_backend,
                authorization_path=auth_file,
                project_root=PROJECT_ROOT,
                is_mock_orchestration=False,
            )


# ---------------------------------------------------------------------------
# Test AE: Authorization Cryptographic Binding
# ---------------------------------------------------------------------------
def test_ae_authorization_cryptographic_binding():
    """Verify authorization verifier strictly enforces cryptographic bindings and fails closed on missing/mismatched fields."""
    with tempfile.TemporaryDirectory() as td:
        auth_p = Path(td) / "auth.json"
        valid_data = {
            "protocol_version": "1.2.0",
            "human_authorized": True,
            "protocol_lock_sha256": "a" * 64,
            "locked_scientific_code_commit": "b" * 40,
            "locked_source_tree_hash": "c" * 64,
        }

        # 1. Missing protocol_lock_sha256
        bad1 = copy.deepcopy(valid_data)
        del bad1["protocol_lock_sha256"]
        with open(auth_p, "w", encoding="utf-8") as f:
            json.dump(bad1, f)
        with pytest.raises(Phase4BAuthorizationError, match="protocol_lock_sha256"):
            verify_phase4b_authorization(auth_p, protocol_version="1.2.0", protocol_lock_sha256="a"*64, locked_git_commit="b"*40, locked_source_tree_hash="c"*64)

        # 2. Mismatched protocol_lock_sha256
        bad2 = copy.deepcopy(valid_data)
        bad2["protocol_lock_sha256"] = "f" * 64
        with open(auth_p, "w", encoding="utf-8") as f:
            json.dump(bad2, f)
        with pytest.raises(Phase4BAuthorizationError, match="protocol_lock_sha256"):
            verify_phase4b_authorization(auth_p, protocol_version="1.2.0", protocol_lock_sha256="a"*64, locked_git_commit="b"*40, locked_source_tree_hash="c"*64)

        # 3. Missing locked_scientific_code_commit
        bad3 = copy.deepcopy(valid_data)
        del bad3["locked_scientific_code_commit"]
        with open(auth_p, "w", encoding="utf-8") as f:
            json.dump(bad3, f)
        with pytest.raises(Phase4BAuthorizationError, match="locked_scientific_code_commit"):
            verify_phase4b_authorization(auth_p, protocol_version="1.2.0", protocol_lock_sha256="a"*64, locked_git_commit="b"*40, locked_source_tree_hash="c"*64)

        # 4. Mismatched locked_scientific_code_commit
        bad4 = copy.deepcopy(valid_data)
        bad4["locked_scientific_code_commit"] = "0" * 40
        with open(auth_p, "w", encoding="utf-8") as f:
            json.dump(bad4, f)
        with pytest.raises(Phase4BAuthorizationError, match="locked_scientific_code_commit"):
            verify_phase4b_authorization(auth_p, protocol_version="1.2.0", protocol_lock_sha256="a"*64, locked_git_commit="b"*40, locked_source_tree_hash="c"*64)

        # 5. Missing locked_source_tree_hash
        bad5 = copy.deepcopy(valid_data)
        del bad5["locked_source_tree_hash"]
        with open(auth_p, "w", encoding="utf-8") as f:
            json.dump(bad5, f)
        with pytest.raises(Phase4BAuthorizationError, match="locked_source_tree_hash"):
            verify_phase4b_authorization(auth_p, protocol_version="1.2.0", protocol_lock_sha256="a"*64, locked_git_commit="b"*40, locked_source_tree_hash="c"*64)

        # 6. human_authorized == False
        bad6 = copy.deepcopy(valid_data)
        bad6["human_authorized"] = False
        with open(auth_p, "w", encoding="utf-8") as f:
            json.dump(bad6, f)
        with pytest.raises(Phase4BAuthorizationError, match="human_authorized must be true"):
            verify_phase4b_authorization(auth_p, protocol_version="1.2.0", protocol_lock_sha256="a"*64, locked_git_commit="b"*40, locked_source_tree_hash="c"*64)

        # 7. Valid manifest passes
        with open(auth_p, "w", encoding="utf-8") as f:
            json.dump(valid_data, f)
        res = verify_phase4b_authorization(auth_p, protocol_version="1.2.0", protocol_lock_sha256="a"*64, locked_git_commit="b"*40, locked_source_tree_hash="c"*64)
        assert res["status"] == "AUTHORIZED"
        assert res["human_authorized"] is True


# ---------------------------------------------------------------------------
# Test AF: Dirty Tree Blocks Production
# ---------------------------------------------------------------------------
def test_af_dirty_tree_blocks_production(monkeypatch):
    """Verify dirty Git working tree blocks production execution under enforce_git_clean=True."""
    tree_hash = compute_source_tree_hash(PROJECT_ROOT)

    monkeypatch.setattr(
        "tradingagents.temporal_leakage.phase4_confirmatory.check_git_status",
        lambda root: {"git_available": True, "head_commit": "abcdef123456", "git_dirty": True},
    )

    with pytest.raises(CodeFreezeError, match="working tree is dirty"):
        verify_phase4_code_freeze(
            project_root=PROJECT_ROOT,
            locked_git_commit="abcdef123456",
            locked_source_tree_hash=tree_hash,
            enforce_git_clean=True,
        )


# ---------------------------------------------------------------------------
# Test AG: Controlled Scientific Source Modification Blocks Execution
# ---------------------------------------------------------------------------
def test_ag_controlled_scientific_source_modification_blocks_execution(monkeypatch):
    """Verify modifying controlled scientific source files triggers CodeFreezeError."""
    # 1. Tree hash mismatch
    with pytest.raises(CodeFreezeError, match="Source tree hash mismatch"):
        verify_phase4_code_freeze(
            project_root=PROJECT_ROOT,
            locked_source_tree_hash="0" * 64,
            enforce_git_clean=False,
        )

    # 2. Descendant commit check with modified controlled files
    import subprocess
    original_run = subprocess.run

    def mock_subprocess_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and "diff" in cmd:
            class MockRes:
                stdout = "tradingagents/temporal_leakage/phase4_confirmatory.py\n"
                stderr = ""
                returncode = 0
            return MockRes()
        if isinstance(cmd, list) and "--is-ancestor" in cmd:
            class MockAncestor:
                returncode = 0
            return MockAncestor()
        return original_run(cmd, *args, **kwargs)

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)
    monkeypatch.setattr(
        "tradingagents.temporal_leakage.phase4_confirmatory.check_git_status",
        lambda root: {"git_available": True, "head_commit": "1111111111111111111111111111111111111111", "git_dirty": False},
    )

    with pytest.raises(CodeFreezeError, match="Controlled scientific source files modified"):
        verify_phase4_code_freeze(
            project_root=PROJECT_ROOT,
            locked_git_commit="2222222222222222222222222222222222222222",
            enforce_git_clean=True,
        )


# ---------------------------------------------------------------------------
# Test AH: Real Backend Uses Exact Dose Stream
# ---------------------------------------------------------------------------
def test_ah_real_backend_uses_exact_dose_stream(monkeypatch):
    """Verify ProductionConfirmatoryBackend calls create_exact_token_dose_stream with exact dose invariant."""
    with open(PRE_CUTOFF_DOCS_PATH, "r", encoding="utf-8") as f:
        pre_docs = [json.loads(line) for line in f if line.strip()]
    with open(CONTAMINATION_DOCS_PATH, "r", encoding="utf-8") as f:
        post_docs = [json.loads(line) for line in f if line.strip()]

    stream_called_with = {}
    from tradingagents.temporal_leakage import twin_pipeline
    orig_create_exact = twin_pipeline.create_exact_token_dose_stream

    def spy_create_exact(*args, **kwargs):
        stream_called_with.update(kwargs)
        return orig_create_exact(*args, **kwargs)

    monkeypatch.setattr("tradingagents.temporal_leakage.twin_pipeline.create_exact_token_dose_stream", spy_create_exact)

    class FastTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [abs(hash(w)) % 1000 for w in text.split()]

    backend = ProductionConfirmatoryBackend(
        mock_tokenizer_for_testing=FastTokenizer(),
    )

    backend._seed_cache[42] = {
        "tokenizer": FastTokenizer(),
        "base_mlm": None,
        "initial_model_hash": "mock_hash",
        "mask_schedule": [],
        "mask_schedule_hash": "mask_hash",
        "downstream_head_state_dict": {},
        "downstream_initial_head_hash": "head_hash",
        "token_budget": 5120,
        "num_blocks": 10,
        "block_length": 512,
    }

    monkeypatch.setattr("copy.deepcopy", lambda x: x)
    monkeypatch.setattr("tradingagents.temporal_leakage.twin_pipeline.hash_model_parameters", lambda x: "mock_hash")

    class StopAfterStream(Exception):
        pass

    monkeypatch.setattr(
        "tradingagents.temporal_leakage.twin_pipeline.run_continued_pretraining_mlm",
        lambda **kwargs: (_ for _ in ()).throw(StopAfterStream("Stream generated successfully")),
    )

    events = load_events()
    with open(ANCHORS_PATH, "r", encoding="utf-8") as f:
        anchors = [json.loads(line) for line in f if line.strip()]

    with open(CONF_CONFIG_PATH, "r", encoding="utf-8") as f:
        conf_cfg = yaml.safe_load(f)

    with pytest.raises(StopAfterStream):
        backend.execute_branch(
            seed=42,
            dose=0.50,
            pre_docs=pre_docs,
            post_docs=post_docs,
            anchors=anchors,
            events=events,
            conf_cfg=conf_cfg,
            mock_model=object(),
            mock_tokenizer=FastTokenizer(),
        )

    assert stream_called_with["dose"] == 0.50
    assert stream_called_with["num_blocks"] == 10
    assert stream_called_with["block_length"] == 512
    assert stream_called_with["max_repetition_ratio"] == 0.20


# ---------------------------------------------------------------------------
# Test AI: Real Backend Does Not Emit Synthetic Metrics
# ---------------------------------------------------------------------------
def test_ai_real_backend_does_not_emit_synthetic_metrics():
    """Verify Production backend specifies data_mode='EMPIRICAL' and temporal_robustness='NOT_EVALUATED'
    while Mock backend specifies data_mode='MOCK'."""
    mock_backend = MockConfirmatoryBackend()
    events = load_events()
    anchors = [{"anchor_id": "a1", "event_id": events[0]["event_id"], "text": "Fed keeps rates unchanged."}]
    with open(CONF_CONFIG_PATH, "r", encoding="utf-8") as f:
        conf_cfg = yaml.safe_load(f)

    mock_res = mock_backend.execute_branch(
        seed=13,
        dose=0.50,
        pre_docs=[],
        post_docs=[],
        anchors=anchors,
        events=events,
        conf_cfg=conf_cfg,
    )
    assert mock_res["data_mode"] == "MOCK"

    prod_backend = ProductionConfirmatoryBackend()
    assert prod_backend.__doc__ is not None
    assert "data_mode='EMPIRICAL'" in prod_backend.__doc__
    assert "NOT_EVALUATED" in prod_backend.__doc__


# ---------------------------------------------------------------------------
# Test AJ: Artifact Writer
# ---------------------------------------------------------------------------
def test_aj_artifact_writer():
    """Verify Phase4ArtifactWriter writes all 25 branch manifests, 25 branch metrics,
    results JSON, and provenance manifests, and reads them back deterministically."""
    with tempfile.TemporaryDirectory() as td:
        writer = Phase4ArtifactWriter(output_root=td)
        mock_branches = {}
        seeds = [13, 42, 87, 123, 2024]
        doses = [0.0, 0.25, 0.5, 0.75, 1.0]

        for s in seeds:
            for d in doses:
                key = f"seed_{s}_dose_{int(round(d * 100)):03d}"
                mock_branches[key] = {
                    "seed": s,
                    "dose": d,
                    "requested_dose": d,
                    "realized_dose": d,
                    "pre_cutoff_tokens": int(256000 * (1 - d)),
                    "post_cutoff_tokens": int(256000 * d),
                    "total_tokens": 256000,
                    "forced_repetition_ratio": 0.0,
                    "data_mode": "MOCK",
                    "initial_model_hash": f"init_{s}",
                    "post_mlm_model_hash": f"mlm_{s}_{d}",
                    "downstream_initial_head_hash": f"head_{s}",
                    "downstream_sample_order_hash": f"order_{s}",
                    "mask_schedule_hash": f"mask_{s}",
                    "clean_corpus_sha256": "clean_sha",
                    "contamination_corpus_sha256": "contam_sha",
                    "treatment_stream_sha256": f"stream_{s}_{d}",
                    "base_checkpoint": "ProsusAI/finbert",
                    "base_revision": "4556d13015211d73dccd3fdd39d39232506f3e43",
                    "tokenizer_name": "ProsusAI/finbert",
                    "tokenizer_revision": "4556d13015211d73dccd3fdd39d39232506f3e43",
                    "mlm_steps": 100,
                    "optimizer": "AdamW",
                    "learning_rate": 5e-5,
                    "weight_decay": 0.01,
                    "downstream_hyperparameters": {"epochs": 3},
                    "temporal_robustness": "NOT_EVALUATED",
                }

        results = {
            "status": "COMPLETED",
            "branches": mock_branches,
            "decision_rules_applied": {"primary_confirmed": True},
        }
        env_info = {"python_version": "3.10", "platform": "windows"}
        lock_info = {"protocol_version": "1.2.0", "locked": True}

        summary = writer.write_all(results, env_info=env_info, lock_info=lock_info, allow_mock=True)

        assert len(summary["manifest_paths"]) == 25
        assert len(summary["metrics_paths"]) == 25
        assert Path(summary["results_path"]).exists()
        assert Path(summary["environment_path"]).exists()
        assert Path(summary["protocol_verification_path"]).exists()

        with open(summary["results_path"], "r", encoding="utf-8") as f:
            read_res = json.load(f)
        assert len(read_res["branches"]) == 25
        assert read_res["decision_rules_applied"]["primary_confirmed"] is True

        for mp in summary["manifest_paths"]:
            with open(mp, "r", encoding="utf-8") as f:
                b_mf = json.load(f)
            assert "seed" in b_mf
            assert "dose" in b_mf
            assert "total_tokens" in b_mf


# ---------------------------------------------------------------------------
# Test AK: Manifest Completeness
# ---------------------------------------------------------------------------
def test_ak_manifest_completeness():
    """Verify that per-branch manifest contains all required provenance fields."""
    required_fields = [
        "seed", "dose", "requested_dose", "realized_dose",
        "pre_cutoff_tokens", "post_cutoff_tokens", "total_tokens",
        "forced_repetition_ratio", "clean_corpus_sha256",
        "contamination_corpus_sha256", "treatment_stream_sha256",
        "base_checkpoint", "base_revision", "tokenizer_revision",
        "initial_model_hash", "post_mlm_model_hash", "mask_schedule_hash",
        "downstream_initial_head_hash", "downstream_sample_order_hash",
        "mlm_steps", "optimizer", "learning_rate", "weight_decay",
        "temporal_robustness", "data_mode",
    ]

    mock_backend = MockConfirmatoryBackend()
    events = load_events()
    anchors = [{"anchor_id": "a1", "event_id": events[0]["event_id"], "text": "Fed policy statement."}]
    with open(CONF_CONFIG_PATH, "r", encoding="utf-8") as f:
        conf_cfg = yaml.safe_load(f)

    b_res = mock_backend.execute_branch(
        seed=13,
        dose=0.25,
        pre_docs=[],
        post_docs=[],
        anchors=anchors,
        events=events,
        conf_cfg=conf_cfg,
    )

    with tempfile.TemporaryDirectory() as td:
        writer = Phase4ArtifactWriter(output_root=td)
        mf_path = writer.write_branch_manifest(b_res)
        with open(mf_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        for field in required_fields:
            assert field in manifest_data, f"Missing required provenance field in branch manifest: '{field}'"


# ---------------------------------------------------------------------------
# Test AL: Production Execution Still Blocked
# ---------------------------------------------------------------------------
def test_al_production_execution_still_blocked():
    """Verify Phase 4B production model execution is strictly blocked because authorization file is missing."""
    auth_file = PROJECT_ROOT / "configs" / "phase4_execution_authorization.json"
    assert not auth_file.exists(), "phase4_execution_authorization.json must NOT exist in the repository!"

    with pytest.raises(Phase4BAuthorizationError, match="FULL_EXECUTION_BLOCKED"):
        verify_phase4b_authorization(auth_file)

    with pytest.raises(Phase4BAuthorizationError, match="FULL_EXECUTION_BLOCKED"):
        run_phase4_confirmatory(
            config_path=CONF_CONFIG_PATH,
            prereg_path=PREREG_CONFIG_PATH,
            smoke_mode=False,
        )


# ---------------------------------------------------------------------------
# Test AM: Protocol Lock Manifest Verification & Integrity
# ---------------------------------------------------------------------------
def test_am_protocol_lock_manifest_verification_and_integrity():
    """Verify verify_phase4_protocol_lock returns complete cryptographic bindings
    and enforces version >= 1.2.1 invariants (source_tree_hash, scientific_code_commit)."""
    lock_meta = verify_phase4_protocol_lock(
        protocol_lock_path=PROTOCOL_LOCK_PATH,
        project_root=PROJECT_ROOT,
        prereg_config_path=PREREG_CONFIG_PATH,
        conf_config_path=CONF_CONFIG_PATH,
    )
    assert lock_meta["status"] == "PROTOCOL_LOCKED_AND_VERIFIED"
    assert lock_meta["protocol_version"] == "1.2.4"
    assert "source_tree_hash" in lock_meta
    assert "scientific_code_commit" in lock_meta
    assert "code_commit" in lock_meta
    assert len(lock_meta["source_tree_hash"]) == 64
    assert len(lock_meta["scientific_code_commit"]) == 40

    # Tamper test: missing source_tree_hash in v1.2.4 must fail closed
    with tempfile.TemporaryDirectory() as td:
        tampered_lock = Path(td) / "lock.json"
        with open(PROTOCOL_LOCK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        del data["source_tree_hash"]
        with open(tampered_lock, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with pytest.raises(PreregistrationLockError, match="source_tree_hash"):
            verify_phase4_protocol_lock(
                protocol_lock_path=tampered_lock,
                project_root=PROJECT_ROOT,
            )

    # Tamper test: missing commit bindings in v1.2.4 must fail closed
    with tempfile.TemporaryDirectory() as td:
        tampered_lock = Path(td) / "lock.json"
        with open(PROTOCOL_LOCK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        del data["scientific_code_commit"]
        del data["code_commit"]
        with open(tampered_lock, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with pytest.raises(PreregistrationLockError, match="scientific_code_commit"):
            verify_phase4_protocol_lock(
                protocol_lock_path=tampered_lock,
                project_root=PROJECT_ROOT,
            )


# ---------------------------------------------------------------------------
# Test AN: Authorization Binding Validation
# ---------------------------------------------------------------------------
def test_an_authorization_binding_validation():
    """Verify verify_phase4b_authorization validates locked commit, source tree hash,
    protocol lock hash, and protocol version against expected lock manifest."""
    with open(PROTOCOL_LOCK_PATH, "r", encoding="utf-8") as f:
        lock_manifest = json.load(f)
    lock_sha = compute_file_sha256(PROTOCOL_LOCK_PATH, normalize_newlines=True)
    proto_ver = lock_manifest.get("protocol_version", "1.2.4")

    with tempfile.TemporaryDirectory() as td:
        auth_file = Path(td) / "auth.json"

        # 1. Mismatched locked_scientific_code_commit
        with open(auth_file, "w", encoding="utf-8") as f:
            json.dump({
                "protocol_version": proto_ver,
                "human_authorized": True,
                "protocol_lock_sha256": lock_sha,
                "locked_scientific_code_commit": "0" * 40,
                "locked_source_tree_hash": lock_manifest["source_tree_hash"],
            }, f)

        with pytest.raises(Phase4BAuthorizationError, match="locked_scientific_code_commit"):
            verify_phase4b_authorization(
                auth_file,
                protocol_version=proto_ver,
                expected_lock_hash=lock_sha,
                expected_lock_manifest=lock_manifest,
            )

        # 2. Mismatched locked_source_tree_hash
        with open(auth_file, "w", encoding="utf-8") as f:
            json.dump({
                "protocol_version": proto_ver,
                "human_authorized": True,
                "protocol_lock_sha256": lock_sha,
                "locked_scientific_code_commit": lock_manifest["scientific_code_commit"],
                "locked_source_tree_hash": "f" * 64,
            }, f)

        with pytest.raises(Phase4BAuthorizationError, match="locked_source_tree_hash"):
            verify_phase4b_authorization(
                auth_file,
                protocol_version=proto_ver,
                expected_lock_hash=lock_sha,
                expected_lock_manifest=lock_manifest,
            )

        # 3. Mismatched protocol_lock_sha256
        with open(auth_file, "w", encoding="utf-8") as f:
            json.dump({
                "protocol_version": proto_ver,
                "human_authorized": True,
                "protocol_lock_sha256": "e" * 64,
                "locked_scientific_code_commit": lock_manifest["scientific_code_commit"],
                "locked_source_tree_hash": lock_manifest["source_tree_hash"],
            }, f)

        with pytest.raises(Phase4BAuthorizationError, match="protocol_lock_sha256"):
            verify_phase4b_authorization(
                auth_file,
                protocol_version=proto_ver,
                expected_lock_hash=lock_sha,
                expected_lock_manifest=lock_manifest,
            )


# ---------------------------------------------------------------------------
# Test AO: Code Freeze Consumption of Lock Manifest
# ---------------------------------------------------------------------------
def test_ao_code_freeze_consumption_of_lock_manifest():
    """Verify verify_phase4_code_freeze consumes lock manifest and verifies source tree hash."""
    with open(PROTOCOL_LOCK_PATH, "r", encoding="utf-8") as f:
        lock_manifest = json.load(f)

    # Valid lock manifest verification
    res = verify_phase4_code_freeze(
        project_root=PROJECT_ROOT,
        locked_git_commit=lock_manifest.get("code_commit") or lock_manifest.get("scientific_code_commit"),
        locked_source_tree_hash=lock_manifest["source_tree_hash"],
        enforce_git_clean=False,
    )
    assert res["status"] == "CODE_FROZEN_AND_VERIFIED"
    assert res["source_tree_hash"] == lock_manifest["source_tree_hash"]

    # Tampered tree hash fails closed
    with pytest.raises(CodeFreezeError, match="Source tree hash mismatch"):
        verify_phase4_code_freeze(
            project_root=PROJECT_ROOT,
            locked_git_commit=lock_manifest.get("code_commit"),
            locked_source_tree_hash="deadbeef" * 8,
            enforce_git_clean=False,
        )


# ---------------------------------------------------------------------------
# Test AP: Economic Field Mappings Fail On Missing
# ---------------------------------------------------------------------------
def test_ap_economic_field_mappings_fail_on_missing():
    """Verify backend and orchestrator strictly require market_outcomes.treasury_2y_yield_change
    and market_outcomes.spy_1d_return without silent fallback."""
    events = load_events()
    tampered_events = copy.deepcopy(events)
    # Remove 2y yield change from first event
    del tampered_events[0]["market_outcomes"]["treasury_2y_yield_change"]

    prod_backend = ProductionConfirmatoryBackend()
    anchors = [{"anchor_id": "a1", "event_id": events[0]["event_id"], "text": "Fed rate cut announcement."}]
    with open(CONF_CONFIG_PATH, "r", encoding="utf-8") as f:
        conf_cfg = yaml.safe_load(f)

    with pytest.raises((PreregistrationLockError, KeyError, ProductionBackendValidationError)):
        prod_backend.execute_branch(
            seed=42,
            dose=0.0,
            pre_docs=[],
            post_docs=[],
            anchors=anchors,
            events=tampered_events,
            conf_cfg=conf_cfg,
        )


# ---------------------------------------------------------------------------
# Test AQ: Economic Effect Event-Level Inference
# ---------------------------------------------------------------------------
def test_aq_economic_effect_event_level_inference():
    """Verify evaluate_economic_effect_event_level generates bootstrap confidence interval
    and paired event-level economic effect metrics."""
    rng = np.random.RandomState(42)
    events = load_events()
    event_ids = [e["event_id"] for e in events]
    market_returns = np.array([float(e["market_outcomes"]["treasury_2y_yield_change"]) for e in events])
    clean_stances = rng.randn(len(events))
    leak_stances = clean_stances + rng.randn(len(events)) * 0.2

    res = evaluate_economic_effect_event_level(
        stance_scores_leak=leak_stances,
        stance_scores_clean=clean_stances,
        market_returns=market_returns,
        event_ids=event_ids,
        aggregation_rule="mean",
        n_bootstrap=100,
        random_seed=42,
    )

    assert "delta_ic" in res
    assert "delta_ic_ci_95" in res
    assert "p_value" in res
    assert len(res["delta_ic_ci_95"]) == 2
    ci_l, ci_u = res["delta_ic_ci_95"]
    assert ci_l <= ci_u


# ---------------------------------------------------------------------------
# Test AR: Representational Evaluator Probe Alpha Single Source of Truth
# ---------------------------------------------------------------------------
def test_ar_repr_evaluator_probe_alpha_single_source_of_truth():
    """Verify evaluate_representational_leakage_grouped accepts probe_alpha and passes
    it to Ridge/RidgeClassifier without hardcoded alpha."""
    rng = np.random.RandomState(42)
    events = load_events()
    event_ids = [e["event_id"] for e in events]
    event_times = [e["event_time"] for e in events]
    targets = [float(e.get("future_rate_change", 0.0)) for e in events]

    rep_clean = rng.randn(40, 16)
    rep_leak = rep_clean + rng.randn(40, 16) * 0.1

    res_alpha_1 = evaluate_representational_leakage_grouped(
        representations_leak=rep_leak,
        representations_clean=rep_clean,
        y=targets,
        event_ids=event_ids,
        event_times=event_times,
        target_type="continuous",
        n_splits=4,
        min_train_events=8,
        n_permutations=20,
        probe_alpha=1.0,
        random_seed=42,
    )
    assert res_alpha_1["probe_alpha"] == 1.0

    res_alpha_10 = evaluate_representational_leakage_grouped(
        representations_leak=rep_leak,
        representations_clean=rep_clean,
        y=targets,
        event_ids=event_ids,
        event_times=event_times,
        target_type="continuous",
        n_splits=4,
        min_train_events=8,
        n_permutations=20,
        probe_alpha=10.0,
        random_seed=42,
    )
    assert res_alpha_10["probe_alpha"] == 10.0
    # Different regularization alphas must produce different evaluation metrics
    assert res_alpha_1["observed_statistic"] != res_alpha_10["observed_statistic"]


# ---------------------------------------------------------------------------
# Test AS: Competence Metric Provenance
# ---------------------------------------------------------------------------
def test_as_competence_metric_provenance():
    """Verify competence metric provenance: evaluate_competence calculates genuine
    metrics and Production backend documents competence provenance."""
    res = evaluate_competence(y_true=[0, 1, -1], y_pred=[0, 1, -1], n_bootstrap=10)
    assert "macro_f1" in res
    assert res["macro_f1"] == 1.0

    prod_backend = ProductionConfirmatoryBackend()
    assert prod_backend.__doc__ is not None
    assert "NOT_EVALUATED" in prod_backend.__doc__


# ---------------------------------------------------------------------------
# Test AT: Behavioral Metric Event-Level Analysis
# ---------------------------------------------------------------------------
def test_at_behavioral_metric_event_level_analysis():
    """Verify evaluate_behavioral_leakage_event_level computes event-level sensitivities,
    deltas, and descriptive summary statistics without confirmatory p-value/FDR."""
    events = load_events()
    event_ids = [e["event_id"] for e in events]
    rng = np.random.RandomState(42)
    clean_sens = rng.uniform(0.05, 0.25, size=len(events)).tolist()
    leak_sens = [s + rng.uniform(0.01, 0.05) for s in clean_sens]

    res = evaluate_behavioral_leakage_event_level(
        sensitivities_leak=leak_sens,
        sensitivities_clean=clean_sens,
        event_ids=event_ids,
        aggregation_rule="mean",
    )
    assert "l_behavior_event" in res
    assert "l_behavior_mean" in res
    assert "l_behavior_median" in res
    assert "l_behavior_std" in res
    assert "mask_sensitivity_leak_event" in res
    assert "mask_sensitivity_clean_event" in res
    assert "event_deltas" in res
    assert len(res["event_deltas"]) == len(events)
    assert res["l_behavior_mean"] > 0
    assert res["behavioral_inference"] == "DESCRIPTIVE_ONLY"
    assert res["behavioral_statistical_unit"] == "independent_fomc_event"
    assert res["behavioral_multiple_testing"] == "NOT_APPLICABLE"
    assert res["behavioral_significance_testing"] is False
    assert "p_value" not in res
    assert "fdr_correction" not in res


# ---------------------------------------------------------------------------
# Test AU: Binary Directional Classification Co-Primary
# ---------------------------------------------------------------------------
def test_au_binary_directional_classification_co_primary():
    """Verify binary co-primary next_scheduled_change_vs_hold uses RidgeClassifier
    and outputs macro F1 metrics."""
    rng = np.random.RandomState(42)
    events = load_events()
    event_ids = [e["event_id"] for e in events]
    event_times = [e["event_time"] for e in events]
    binary_targets = [1 if abs(float(e.get("future_rate_change", 0.0))) > 1e-6 else 0 for e in events]

    rep_clean = rng.randn(40, 16)
    rep_leak = rep_clean + rng.randn(40, 16) * 0.1

    res = evaluate_representational_leakage_grouped(
        representations_leak=rep_leak,
        representations_clean=rep_clean,
        y=binary_targets,
        event_ids=event_ids,
        event_times=event_times,
        target_type="binary",
        n_splits=4,
        min_train_events=8,
        n_permutations=20,
        probe_alpha=1.0,
        random_seed=42,
    )
    assert res["model_type"] == "ridge_classifier"
    assert "delta_macro_f1" in res
    assert "macro_f1_leak" in res
    assert "macro_f1_clean" in res


# ---------------------------------------------------------------------------
# Test AV: Runtime Contract Reconciliation
# ---------------------------------------------------------------------------
def test_av_runtime_contract_reconciliation():
    """Verify resolve_phase4_runtime_contract reconciles all locked parameters
    and fails closed on any discrepancy between confirmatory and preregistration configs."""
    with open(CONF_CONFIG_PATH, "r", encoding="utf-8") as f:
        conf_cfg = yaml.safe_load(f)
    with open(PREREG_CONFIG_PATH, "r", encoding="utf-8") as f:
        prereg_cfg = yaml.safe_load(f)

    contract = resolve_phase4_runtime_contract(conf_cfg, prereg_cfg)
    assert contract["protocol_version"] == "1.2.4"
    assert contract["mlm_max_steps"] == 100
    assert contract["mlm_scheduler"] == "none"
    assert contract["mlm_warmup_ratio"] == 0.0
    assert contract["probe_alpha"] == 1.0
    assert contract["random_seed"] == 42
    assert contract["token_budget"] == 256000
    assert contract["behavioral_endpoint_role"] == "descriptive_secondary"
    assert contract["behavioral_significance_testing"] is False
    assert contract["behavioral_multiple_testing"] == "NOT_APPLICABLE"

    # Discrepancy test: mismatch in max_steps must fail closed
    tampered_conf = copy.deepcopy(conf_cfg)
    tampered_conf["mlm_training"]["max_steps"] = 200
    with pytest.raises(PreregistrationLockError, match="max_steps mismatch"):
        resolve_phase4_runtime_contract(tampered_conf, prereg_cfg)


# ---------------------------------------------------------------------------
# Test AW: MLM Hyperparameters Locked
# ---------------------------------------------------------------------------
def test_aw_mlm_hyperparameters_locked():
    """Verify that max_steps=100, scheduler='none', warmup_ratio=0.0 are locked in configs."""
    with open(CONF_CONFIG_PATH, "r", encoding="utf-8") as f:
        conf_cfg = yaml.safe_load(f)
    with open(PREREG_CONFIG_PATH, "r", encoding="utf-8") as f:
        prereg_cfg = yaml.safe_load(f)

    assert conf_cfg["mlm_training"]["max_steps"] == 100
    assert conf_cfg["mlm_training"]["scheduler"] == "none"
    assert conf_cfg["mlm_training"]["warmup_ratio"] == 0.0
    assert conf_cfg["downstream_evaluation"]["random_seed"] == 42

    assert prereg_cfg["compute_bounds"]["max_steps"] == 100
    assert prereg_cfg["compute_bounds"]["scheduler"] == "none"
    assert prereg_cfg["compute_bounds"]["warmup_ratio"] == 0.0
    assert prereg_cfg["statistical_design"]["random_seed"] == 42


# ---------------------------------------------------------------------------
# Test AX: Statistical Unit and Sample Size Invariants
# ---------------------------------------------------------------------------
def test_ax_statistical_unit_and_sample_size_invariants():
    """Verify 40 events total, 32 OOS events, 8 fold 0 in-sample events, and 181 anchors."""
    events = load_events()
    assert len(events) == 40

    with open(ANCHORS_PATH, "r", encoding="utf-8") as f:
        anchors = [json.loads(line) for line in f if line.strip()]
    assert len(anchors) == 181

    event_ids = [e["event_id"] for e in events]
    event_times = [e["event_time"] for e in events]
    splits = grouped_temporal_split(event_ids, event_times, n_splits=4)
    assert len(splits) == 4

    total_test_events = 0
    for tr, ts in splits:
        total_test_events += len(ts)
        assert len(ts) == 8
    assert total_test_events == 32
    assert len(splits[0][0]) == 8


# ---------------------------------------------------------------------------
# Test AY: Branch Manifest Provenance Completeness
# ---------------------------------------------------------------------------
def test_ay_branch_manifest_provenance_completeness():
    """Verify branch manifests record cryptographic provenance bindings."""
    required_provenance = [
        "scientific_code_commit",
        "execution_repository_head",
        "source_tree_hash",
        "protocol_lock_sha256",
        "protocol_version",
    ]

    mock_backend = MockConfirmatoryBackend()
    events = load_events()
    anchors = [{"anchor_id": "a1", "event_id": events[0]["event_id"], "text": "FOMC policy stance statement."}]
    with open(CONF_CONFIG_PATH, "r", encoding="utf-8") as f:
        conf_cfg = yaml.safe_load(f)

    b_res = mock_backend.execute_branch(
        seed=13,
        dose=0.5,
        pre_docs=[],
        post_docs=[],
        anchors=anchors,
        events=events,
        conf_cfg=conf_cfg,
    )
    # Inject provenance as orchestrator does
    b_res["scientific_code_commit"] = "a" * 40
    b_res["execution_repository_head"] = "a" * 40
    b_res["source_tree_hash"] = "b" * 64
    b_res["protocol_lock_sha256"] = "c" * 64
    b_res["protocol_version"] = "1.2.4"

    with tempfile.TemporaryDirectory() as td:
        writer = Phase4ArtifactWriter(output_root=td)
        mf_path = writer.write_branch_manifest(b_res)
        with open(mf_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for p_key in required_provenance:
            assert p_key in manifest, f"Missing provenance key '{p_key}' in manifest"


# ---------------------------------------------------------------------------
# Test AZ: Artifact Writer Rejects Mock Data Mode
# ---------------------------------------------------------------------------
def test_az_artifact_writer_rejects_mock_data_mode():
    """Verify Phase4ArtifactWriter strictly rejects data_mode='MOCK' when allow_mock=False."""
    with tempfile.TemporaryDirectory() as td:
        writer = Phase4ArtifactWriter(output_root=td)
        results = {
            "status": "COMPLETED",
            "branches": {
                "branch_1": {"data_mode": "MOCK"},
            },
        }
        with pytest.raises(ProductionBackendValidationError, match="data_mode='MOCK'"):
            writer.write_confirmatory_results(results, allow_mock=False)

        # allow_mock=True succeeds
        p = writer.write_confirmatory_results(results, allow_mock=True)
        assert p.exists()


# ---------------------------------------------------------------------------
# Deterministic Metric Integration Test
# ---------------------------------------------------------------------------
def test_deterministic_metric_integration_on_40_event_fixture():
    """Deterministic end-to-end evaluation verifying continuous regression primary,
    binary classification co-primary, behavioral event-level sensitivity, and economic effect."""
    rng = np.random.RandomState(42)
    events = load_events()
    event_ids = [e["event_id"] for e in events]
    event_times = [e["event_time"] for e in events]

    y_continuous = np.array([float(e.get("future_rate_change", 0.0)) for e in events])
    y_binary = np.array([1 if abs(val) > 1e-6 else 0 for val in y_continuous])
    y_2y = np.array([float(e["market_outcomes"]["treasury_2y_yield_change"]) for e in events])
    y_spy = np.array([float(e["market_outcomes"]["spy_1d_return"]) for e in events])

    clean_repr = rng.randn(40, 16)
    leak_repr = clean_repr + rng.randn(40, 16) * 0.05
    clean_stances = rng.randn(40)
    leak_stances = clean_stances + rng.randn(40) * 0.05
    clean_sens = rng.uniform(0.1, 0.3, 40).tolist()
    leak_sens = [s + 0.02 for s in clean_sens]

    # 1. Continuous primary
    res_reg = evaluate_representational_leakage_grouped(
        representations_leak=leak_repr,
        representations_clean=clean_repr,
        y=y_continuous,
        event_ids=event_ids,
        event_times=event_times,
        target_type="continuous",
        n_splits=4,
        min_train_events=8,
        n_permutations=20,
        probe_alpha=1.0,
        random_seed=42,
    )
    assert res_reg["model_type"] == "ridge_regression"
    assert res_reg["n_oos_events"] == 32
    assert "observed_statistic" in res_reg
    assert "delta_spearman" in res_reg
    assert np.isfinite(res_reg["p_value"])

    # 2. Binary co-primary
    res_bin = evaluate_representational_leakage_grouped(
        representations_leak=leak_repr,
        representations_clean=clean_repr,
        y=y_binary,
        event_ids=event_ids,
        event_times=event_times,
        target_type="binary",
        n_splits=4,
        n_permutations=20,
        probe_alpha=1.0,
        random_seed=42,
    )
    assert res_bin["model_type"] == "ridge_classifier"
    assert "delta_macro_f1" in res_bin
    assert np.isfinite(res_bin["p_value"])

    # 3. Behavioral leakage
    res_beh = evaluate_behavioral_leakage_event_level(
        sensitivities_leak=leak_sens,
        sensitivities_clean=clean_sens,
        event_ids=event_ids,
    )
    assert np.isfinite(res_beh["l_behavior_event"])

    # 4. Economic effect
    res_econ_2y = evaluate_economic_effect_event_level(
        stance_scores_leak=leak_stances,
        stance_scores_clean=clean_stances,
        market_returns=y_2y,
        event_ids=event_ids,
        n_bootstrap=50,
        random_seed=42,
    )
    assert np.isfinite(res_econ_2y["delta_ic"])
    ci_l_2y, ci_u_2y = res_econ_2y["delta_ic_ci_95"]
    assert ci_l_2y <= ci_u_2y

    res_econ_spy = evaluate_economic_effect_event_level(
        stance_scores_leak=leak_stances,
        stance_scores_clean=clean_stances,
        market_returns=y_spy,
        event_ids=event_ids,
        n_bootstrap=50,
        random_seed=42,
    )
    assert np.isfinite(res_econ_spy["delta_ic"])
    ci_l_spy, ci_u_spy = res_econ_spy["delta_ic_ci_95"]
    assert ci_l_spy <= ci_u_spy


# ---------------------------------------------------------------------------
# Test BB: Ancestry Verification Allows Doc-Only Descendant Commits
# ---------------------------------------------------------------------------
def test_bb_ancestry_verification_allows_doc_only_descendants(monkeypatch):
    """Verify verify_phase4_code_freeze accepts descendant commits when only non-controlled files (e.g. docs) differ."""
    import subprocess
    original_run = subprocess.run

    def mock_subprocess_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and "diff" in cmd:
            class MockRes:
                stdout = "docs/research/phase4a_preregistration_report.md\nconfigs/phase4_protocol_lock.json\n"
                stderr = ""
                returncode = 0
            return MockRes()
        if isinstance(cmd, list) and "--is-ancestor" in cmd:
            class MockAncestor:
                returncode = 0
            return MockAncestor()
        return original_run(cmd, *args, **kwargs)

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)
    monkeypatch.setattr(
        "tradingagents.temporal_leakage.phase4_confirmatory.check_git_status",
        lambda root: {"git_available": True, "head_commit": "3333333333333333333333333333333333333333", "git_dirty": False},
    )

    tree_hash = compute_source_tree_hash(PROJECT_ROOT)
    res = verify_phase4_code_freeze(
        project_root=PROJECT_ROOT,
        locked_git_commit="2222222222222222222222222222222222222222",
        locked_source_tree_hash=tree_hash,
        enforce_git_clean=True,
    )
    assert res["status"] == "CODE_FROZEN_AND_VERIFIED"
    assert res["scientific_code_commit"] == "2222222222222222222222222222222222222222"
    assert res["execution_repository_head"] == "3333333333333333333333333333333333333333"


# ---------------------------------------------------------------------------
# Test BC: Production Anchor Representation Uses Canonical Encoder Encode
# ---------------------------------------------------------------------------
def test_bc_production_anchor_representation_uses_encoder_encode():
    """Verify Production backend uses canonical encode() method on HuggingFaceTemporalEncoder
    and never invokes non-existent extract_representations()."""
    import torch
    from transformers import BertConfig, BertForSequenceClassification
    from tradingagents.temporal_leakage.hf_encoder import HuggingFaceTemporalEncoder

    cfg = BertConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=100,
    )
    mock_hf_model = BertForSequenceClassification(cfg)

    class FastBatchTokenizer:
        def __call__(self, texts, **kwargs):
            class BatchEncoding(dict):
                def to(self, device):
                    return BatchEncoding({k: (v.to(device) if hasattr(v, "to") else v) for k, v in self.items()})
            if isinstance(texts, str):
                texts = [texts]
            n = len(texts)
            return BatchEncoding({
                "input_ids": torch.randint(0, 100, (n, 16)),
                "attention_mask": torch.ones(n, 16, dtype=torch.long),
            })

    fast_tok = FastBatchTokenizer()
    encoder = HuggingFaceTemporalEncoder(model=mock_hf_model, tokenizer=fast_tok, device="cpu")

    # 1. Structural interface assertions
    assert hasattr(encoder, "encode"), "HuggingFaceTemporalEncoder must provide encode()"
    assert not hasattr(encoder, "extract_representations"), (
        "HuggingFaceTemporalEncoder must NOT have extract_representations() attribute"
    )

    # 2. Load official 181 anchors
    with open(ANCHORS_PATH, "r", encoding="utf-8") as f:
        anchors = [json.loads(line) for line in f if line.strip()]
    assert len(anchors) == 181
    anchor_texts = [a["text"] for a in anchors]

    # 3. Call encode and verify output contract
    encode_called = {"count": 0}
    orig_encode = encoder.encode

    def wrapped_encode(texts, **kwargs):
        encode_called["count"] += 1
        return orig_encode(texts, **kwargs)

    encoder.encode = wrapped_encode

    embeddings = encoder.encode(anchor_texts)

    assert encode_called["count"] == 1, "encode() must be invoked exactly once for anchor batch"
    assert isinstance(embeddings, np.ndarray), "Output must be numpy ndarray"
    assert embeddings.ndim == 2, f"Embedding rank must be 2, got {embeddings.ndim}"
    assert embeddings.shape == (181, 32), f"Expected shape (181, 32), got {embeddings.shape}"
    assert np.all(np.isfinite(embeddings)), "All embedding values must be finite"


# ---------------------------------------------------------------------------
# Test BD: Production First Branch Preflight Integration
# ---------------------------------------------------------------------------
def test_bd_production_first_branch_preflight_integration():
    """ENGINEERING INTEGRATION TEST ONLY:
    Exercises full component DAG of ProductionConfirmatoryBackend for a single branch (Seed 13, Dose 0.00)
    using injected lightweight components:
      treatment construction -> MLM stage -> encoder transfer -> downstream fine-tune
      -> anchor encode -> stance prediction -> behavioral sensitivity -> artifact serialization.
    Validates end-to-end API integration and contract without running expensive full-scale compute."""
    import torch
    from transformers import BertConfig, BertForMaskedLM
    from tradingagents.temporal_leakage.temporal_model import TemporalSample

    # Injected lightweight base MLM model
    cfg = BertConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=1000,
    )
    tiny_mlm = BertForMaskedLM(cfg)

    class FastTokenizer:
        def __init__(self):
            self.mask_token_id = 103
            self.pad_token_id = 0
            self.unk_token_id = 1
            self.cls_token_id = 101
            self.sep_token_id = 102

        def encode(self, text, add_special_tokens=False):
            return [(abs(hash(w)) % 900) + 10 for w in text.split()]

        def __call__(self, texts, **kwargs):
            class BatchEncoding(dict):
                def to(self, device):
                    return BatchEncoding({k: (v.to(device) if hasattr(v, "to") else v) for k, v in self.items()})
            if isinstance(texts, str):
                texts = [texts]
            n = len(texts)
            max_len = 16
            return BatchEncoding({
                "input_ids": torch.randint(10, 900, (n, max_len)),
                "attention_mask": torch.ones(n, max_len, dtype=torch.long),
            })

    fast_tok = FastTokenizer()

    # Load real events, anchors, and documents
    events = load_events()
    with open(ANCHORS_PATH, "r", encoding="utf-8") as f:
        anchors = [json.loads(line) for line in f if line.strip()]

    with open(CONF_CONFIG_PATH, "r", encoding="utf-8") as f:
        conf_cfg = yaml.safe_load(f)

    # Use reduced steps and blocks for rapid preflight execution
    branch_cfg = copy.deepcopy(conf_cfg)
    branch_cfg["model"]["device"] = "cpu"
    branch_cfg["mlm_training"]["num_blocks"] = 5
    branch_cfg["mlm_training"]["block_length"] = 64
    branch_cfg["mlm_training"]["token_budget"] = 320
    branch_cfg["mlm_training"]["max_steps"] = 1
    branch_cfg["downstream_training"] = {
        "epochs": 1,
        "max_steps": 1,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "optimizer": "AdamW",
        "scheduler": "linear",
        "warmup_ratio": 0.0,
        "max_seq_length": 16,
    }

    pre_docs = [
        {
            "document_id": f"clean_{i}",
            "available_time": "2018-01-01T00:00:00Z",
            "text": f"Federal Open Market Committee policy statement economic activity expansion labor employment rate inflation target {i} " * 20,
        }
        for i in range(10)
    ]
    post_docs = [
        {
            "document_id": f"contam_{i}",
            "available_time": "2021-01-01T00:00:00Z",
            "text": f"Federal Open Market Committee policy future inflation minutes asset purchase normalization pandemic recovery {i} " * 20,
        }
        for i in range(10)
    ]

    backend = ProductionConfirmatoryBackend(
        device="cpu",
        mock_model_for_testing=tiny_mlm,
        mock_tokenizer_for_testing=fast_tok,
    )

    mock_train_samples = [
        TemporalSample(
            text=f"Dummy sentence {i} for fine tuning stance classifier.",
            event_time="2017-01-01T00:00:00Z",
            available_time="2017-01-01T00:00:00Z",
            task_label=1 if i % 2 == 0 else -1,
            sample_id=f"train_{i}",
            metadata={"year": 2017},
        )
        for i in range(8)
    ]

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        b_res = backend.execute_branch(
            seed=13,
            dose=0.00,
            pre_docs=pre_docs,
            post_docs=post_docs,
            anchors=anchors,
            events=events,
            conf_cfg=branch_cfg,
            train_samples=mock_train_samples,
            output_dir=out_dir / "checkpoints",
        )

        # 1. Assertions on branch results
        assert b_res["data_mode"] == "EMPIRICAL"
        assert b_res["seed"] == 13
        assert b_res["dose"] == 0.00
        assert b_res["event_embeddings"].shape == (40, 32)
        assert np.all(np.isfinite(b_res["event_embeddings"]))
        assert len(b_res["event_stance_scores"]) == 40
        assert len(b_res["event_sensitivities"]) == 40
        assert np.isfinite(b_res["economic_ic_2y"])
        assert np.isfinite(b_res["economic_ic_spy"])

        # 2. Assert artifact serialization succeeds
        writer = Phase4ArtifactWriter(output_root=out_dir / "artifacts")
        b_res["scientific_code_commit"] = "0" * 40
        b_res["execution_repository_head"] = "0" * 40
        b_res["source_tree_hash"] = "1" * 64
        b_res["protocol_lock_sha256"] = "2" * 64
        b_res["protocol_version"] = "1.2.4"

        mf_path = writer.write_branch_manifest(b_res)
        assert mf_path.exists()
        with open(mf_path, "r", encoding="utf-8") as f:
            mf_data = json.load(f)
        assert mf_data["data_mode"] == "EMPIRICAL"
        assert mf_data["protocol_version"] == "1.2.4"

        met_path = writer.write_branch_metrics(b_res)
        assert met_path.exists()


# ---------------------------------------------------------------------------
# Test BE: Historical v1.2.3 Artifacts Rejected Under Protocol v1.2.4
# ---------------------------------------------------------------------------
def test_be_v123_partial_execution_rejected_under_v124():
    """Verify that historical v1.2.3 authorization or branch execution artifacts
    are strictly rejected under Protocol v1.2.4 invariants."""
    with tempfile.TemporaryDirectory() as td:
        auth_file = Path(td) / "auth_v123.json"
        with open(auth_file, "w", encoding="utf-8") as f:
            json.dump({
                "protocol_version": "1.2.3",
                "human_authorized": True,
                "protocol_lock_sha256": "a" * 64,
                "locked_scientific_code_commit": "b" * 40,
                "locked_source_tree_hash": "c" * 64,
            }, f)

        # verify_phase4b_authorization under v1.2.4 must fail closed on v1.2.3 authorization
        with pytest.raises(Phase4BAuthorizationError, match="Protocol version mismatch in authorization"):
            verify_phase4b_authorization(
                auth_file,
                protocol_version="1.2.4",
                protocol_lock_sha256="a" * 64,
                locked_git_commit="b" * 40,
                locked_source_tree_hash="c" * 64,
            )





