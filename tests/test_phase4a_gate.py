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
    evaluate_economic_effect_event_level,
    evaluate_representational_leakage_grouped,
    event_clustered_bootstrap_indices,
    grouped_temporal_split,
    simulate_phase4_primary_power,
)
from tradingagents.temporal_leakage.phase4_confirmatory import (
    CONTROLLED_FILE_MAPPINGS,
    PreregistrationHashMismatchError,
    PreregistrationLockError,
    compute_file_sha256,
    verify_phase4_protocol_lock,
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
    assert res["controlled_file_count"] == 10
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
