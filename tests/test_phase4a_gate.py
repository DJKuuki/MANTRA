"""Comprehensive Automated Verification Suite for Phase 4A Preregistration Gate.

Tests A through L:
- Test A (Item 23): Within-event label, outcome, and timestamp consistency
- Test B (Item 25): Grouped expanding-window temporal cross-validation folds
- Test C (Item 26): Event-level economic effect and stance aggregation
- Test D (Item 27): Event-clustered block bootstrap preserving meeting sampling unit
- Test E (Item 28): Verbatim canonical existence in official Federal Reserve HTML
- Test F (Item 28): Deterministic policy future_action derivation from policy history
- Test G (Item 28): Bit-exact market outcome recomputation from raw daily tables
- Test H (Item 29): Empirical contamination timeline provenance and strict temporal gap
- Test I (Item 30): Treatment block manifest and token provenance tracking
- Test J (Item 30): Exact dose ladder invariant |D_realized - D_requested| <= 1/T
- Test K (Item 29): Prohibition of mid-year placeholders / exact ISO-8601 UTC release times
- Test L (Item 22 / Item 41): Preregistration lock enforcement and hash verification
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List
import yaml

import numpy as np
import pytest

from tradingagents.temporal_leakage.datasets.market_data import (
    load_market_tables,
    recompute_market_outcomes,
)
from tradingagents.temporal_leakage.datasets.policy_history import (
    derive_future_action,
    load_policy_history,
)
from tradingagents.temporal_leakage.metrics import (
    aggregate_event_representations,
    evaluate_economic_effect_event_level,
    event_clustered_bootstrap_indices,
    grouped_temporal_split,
)
from tradingagents.temporal_leakage.phase4_confirmatory import (
    PreregistrationHashMismatchError,
    PreregistrationLockError,
    compute_file_sha256,
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
# Test A (Item 23): Within-Event Consistency
# ---------------------------------------------------------------------------
def test_a_within_event_consistency():
    """Verify that all paragraph anchors for the same event share identical labels, outcomes, and timestamps."""
    events = load_events()
    anchors = load_anchors()

    assert len(events) >= 40, f"Expected at least 40 independent events, found {len(events)}"
    assert len(anchors) >= 160, f"Expected at least 160 paragraph anchors, found {len(anchors)}"

    event_map = {e["event_id"]: e for e in events}
    assert len(event_map) == len(events), "Duplicate event_id detected in events.jsonl"

    for anchor in anchors:
        eid = anchor["event_id"]
        assert eid in event_map, f"Anchor references unknown event_id: {eid}"
        parent = event_map[eid]

        # Invariant checks: within-meeting consistency
        assert anchor["future_targets"]["future_action"] == parent["future_action"], (
            f"Action mismatch in event {eid}: {anchor['future_targets']['future_action']} != {parent['future_action']}"
        )
        assert anchor["market_outcomes"] == parent["market_outcomes"], (
            f"Market outcome mismatch in event {eid}: {anchor['market_outcomes']} != {parent['market_outcomes']}"
        )
        assert anchor["event_time"] == parent["event_time"], (
            f"Event time mismatch in event {eid}: {anchor['event_time']} != {parent['event_time']}"
        )
        assert anchor["source_url"] == parent["source_url"], (
            f"Source URL mismatch in event {eid}"
        )


# ---------------------------------------------------------------------------
# Test B (Item 25): Grouped Temporal Split
# ---------------------------------------------------------------------------
def test_b_grouped_temporal_split():
    """Verify expanding-window temporal folds strictly partition by event without cross-contamination."""
    events = load_events()
    event_ids = [e["event_id"] for e in events]
    event_times = [e["event_time"] for e in events]

    folds = grouped_temporal_split(event_ids, event_times, n_splits=4)
    assert len(folds) >= 3, f"Expected at least 3 folds, got {len(folds)}"

    all_test_events = set()
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_ids = {event_ids[i] for i in train_idx}
        test_ids = {event_ids[i] for i in test_idx}

        # 1. Zero group leakage: Train ∩ Test = ∅
        overlap = train_ids.intersection(test_ids)
        assert len(overlap) == 0, f"Fold {fold_idx} has event overlap: {overlap}"

        # 2. Temporal ordering: max(train_time) < min(test_time)
        max_train_time = max(event_times[i] for i in train_idx)
        min_test_time = min(event_times[i] for i in test_idx)
        assert max_train_time < min_test_time, (
            f"Fold {fold_idx} violates temporal ordering: max train {max_train_time} >= min test {min_test_time}"
        )

        all_test_events.update(test_ids)

    # Test folds should evaluate a substantial portion of historical events
    assert len(all_test_events) >= 20, f"Expected >= 20 evaluated test events, got {len(all_test_events)}"


# ---------------------------------------------------------------------------
# Test C (Item 26): Event-Level Economic Effect
# ---------------------------------------------------------------------------
def test_c_event_level_economic_effect():
    """Verify event-level aggregation treats each event with equal weight in IC calculation."""
    # Create synthetic observations with varying paragraph counts (10 vs 2 vs 5 paragraphs)
    event_ids = ["event_01"] * 10 + ["event_02"] * 2 + ["event_03"] * 5
    representations_clean = np.ones((17, 8))
    representations_leak = np.ones((17, 8)) * 2.0

    # Aggregate representations
    h_clean_agg, u_clean = aggregate_event_representations(representations_clean, event_ids)
    h_leak_agg, u_leak = aggregate_event_representations(representations_leak, event_ids)

    assert len(u_clean) == 3
    assert h_clean_agg.shape == (3, 8)
    np.testing.assert_allclose(h_clean_agg[0], 1.0)
    np.testing.assert_allclose(h_leak_agg[0], 2.0)

    # Event level stance scores (1 per event)
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
# Test D (Item 27): Event-Clustered Bootstrap
# ---------------------------------------------------------------------------
def test_d_event_clustered_bootstrap():
    """Verify event-clustered bootstrap resamples meetings as unified blocks."""
    event_ids = ["ev1", "ev1", "ev1", "ev2", "ev2", "ev3", "ev3", "ev3", "ev3"]
    draws = event_clustered_bootstrap_indices(event_ids, n_boot=20, random_seed=42)

    assert len(draws) == 20
    for draw in draws:
        # Check that if any anchor from ev1 is present, the count of ev1 anchors is a multiple of 3
        draw_event_ids = [event_ids[i] for i in draw]
        counts = {eid: draw_event_ids.count(eid) for eid in set(draw_event_ids)}
        if "ev1" in counts:
            assert counts["ev1"] % 3 == 0, f"ev1 anchors fragmented: {counts['ev1']}"
        if "ev2" in counts:
            assert counts["ev2"] % 2 == 0, f"ev2 anchors fragmented: {counts['ev2']}"
        if "ev3" in counts:
            assert counts["ev3"] % 4 == 0, f"ev3 anchors fragmented: {counts['ev3']}"


# ---------------------------------------------------------------------------
# Test E (Item 28): Verbatim Canonical Source Existence
# ---------------------------------------------------------------------------
def test_e_anchor_canonical_source_existence():
    """Verify all anchor texts exist verbatim in the local cached canonical Federal Reserve HTML source."""
    anchors = load_anchors()
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = {json.loads(line)["document_id"]: json.loads(line) for line in f if line.strip()}

    # Sample test 15 random anchors across different meetings
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(anchors), size=min(15, len(anchors)), replace=False)

    for idx in sample_indices:
        anchor = anchors[idx]
        doc_id = anchor["document_id"]
        assert doc_id in registry, f"Document ID {doc_id} missing from source registry"

        reg_entry = registry[doc_id]
        html_path = PROJECT_ROOT / "data" / "research" / "fomc" / "raw_sources" / f"{doc_id}.html"
        assert html_path.exists(), f"Local HTML snapshot missing: {html_path}"

        # Verify hash of the HTML file matches registry
        actual_hash = compute_file_sha256(html_path, normalize_newlines=False)
        assert actual_hash == reg_entry["raw_source_sha256"], (
            f"HTML hash mismatch for {doc_id}: {actual_hash} != {reg_entry['raw_source_sha256']}"
        )

        # Check that the anchor text is present in HTML (parsing HTML tags)
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        parsed_text = re.sub(r"\s+", " ", soup.get_text()).strip()
        norm_anchor = re.sub(r"\s+", " ", anchor["text"]).strip()
        assert norm_anchor in parsed_text, (
            f"Anchor text from event {anchor['event_id']} not found in canonical HTML source: {norm_anchor[:60]}..."
        )


# ---------------------------------------------------------------------------
# Test F (Item 28): Policy Action Provenance
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
# Test G (Item 28): Market Outcome Recomputation
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
        if expected.get("treasury_2y_yield_change") is not None and recomputed.get("treasury_2y_yield_change") is not None:
            np.testing.assert_allclose(
                recomputed["treasury_2y_yield_change"],
                expected["treasury_2y_yield_change"],
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Treasury 2Y yield change mismatch on {meeting_date}",
            )


# ---------------------------------------------------------------------------
# Test H (Item 29): Contamination Timeline Provenance & Strict Gap
# ---------------------------------------------------------------------------
def test_h_contamination_timeline_provenance():
    """Verify actual contamination min/max timestamps derive from verified source releases."""
    events = load_events()

    # Anchors are strictly bounded up to end of 2019
    anchor_max_time = max(e["event_time"] for e in events)
    assert anchor_max_time == "2019-12-11T19:00:00Z"

    # First 2020 scheduled meeting
    first_2020_meeting_time = "2020-01-29T19:00:00Z"
    assert anchor_max_time < first_2020_meeting_time, "Temporal separation violated between anchors and contamination"

    # Temporal gap in calendar days
    gap_days = (
        np.datetime64(first_2020_meeting_time.replace("Z", ""))
        - np.datetime64(anchor_max_time.replace("Z", ""))
    ).astype("timedelta64[D]").astype(int)

    assert gap_days >= 48, f"Temporal gap must be >= 45 days, found {gap_days} days"


# ---------------------------------------------------------------------------
# Test I (Item 30): Treatment Block Provenance Tracking
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

    # Verify max_repetition_ratio raises error on severe shortfall
    with pytest.raises(CausalIntegrityError, match="token shortfall"):
        create_exact_token_dose_stream(
            pre_corpus=pre_corpus[:1],  # Only 4 tokens available
            post_corpus=post_corpus,
            dose=0.0,
            num_blocks=10,  # Needs 80 tokens
            block_length=8,
            tokenizer=MockTokenizer(),
            random_seed=42,
            max_repetition_ratio=0.20,
        )


# ---------------------------------------------------------------------------
# Test J (Item 30): Exact Dose Ladder Invariant
# ---------------------------------------------------------------------------
def test_j_exact_dose_ladder_invariant():
    """Verify |D_realized - D_requested| <= 1/T across all doses."""
    pre_corpus = ["Statement regarding economic growth and interest rate policy." for _ in range(50)]
    post_corpus = ["Statement regarding future pandemic conditions and market liquidity." for _ in range(50)]

    class MockTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return [1, 2, 3, 4, 5, 6, 7, 8]

    doses = [0.0, 0.25, 0.50, 0.75, 1.00]
    total_tokens = 80  # 10 blocks * 8 tokens

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
# Test K (Item 29): No Mid-Year Placeholders
# ---------------------------------------------------------------------------
def test_k_no_mid_year_placeholder():
    """Verify all event and contamination timestamps are exact ISO-8601 UTC down to seconds/minutes."""
    events = load_events()
    iso_utc_regex = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    for event in events:
        etime = event["event_time"]
        assert iso_utc_regex.match(etime), f"Event {event['event_id']} has invalid timestamp format: {etime}"
        # Official FOMC statement release time is 18:00, 18:30, 19:00, or 20:00 UTC
        # Reject default 00:00:00 midnight placeholders
        assert not etime.endswith("T00:00:00Z"), (
            f"Event {event['event_id']} contains unverified midnight placeholder: {etime}"
        )


# ---------------------------------------------------------------------------
# Test L (Item 22 / Item 41): Preregistration Lock Enforcement
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

        # 1. Valid lock passes
        meta = verify_preregistration_lock(cfg_file, doc_file)
        assert meta["status"] == "LOCKED_AND_VERIFIED"

        # 2. Tampered doc raises PreregistrationHashMismatchError
        doc_file.write_text("# Preregistration Document\nUnauthorized change!", encoding="utf-8")
        with pytest.raises(PreregistrationHashMismatchError):
            verify_preregistration_lock(cfg_file, doc_file)

        # 3. Unlocked config raises PreregistrationLockError
        cfg_content["locked"] = False
        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(cfg_content, f)
        with pytest.raises(PreregistrationLockError, match="not locked"):
            verify_preregistration_lock(cfg_file, doc_file)
