"""Phase 4 Confirmatory Execution Engine & Preregistration Gate.

This module enforces preregistration locking for Phase 4 confirmatory analysis.
In Phase 4A, full-scale model training is strictly locked to prevent un-preregistered
or exploratory compute expenditure. Only smoke-mode / verification execution is permitted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from tradingagents.temporal_leakage.datasets.market_data import (
    load_market_tables,
    recompute_market_outcomes,
)
from tradingagents.temporal_leakage.datasets.policy_history import (
    derive_future_action,
    load_policy_history,
)


class PreregistrationLockError(Exception):
    """Raised when preregistration invariants or lock checks fail."""


class PreregistrationHashMismatchError(PreregistrationLockError):
    """Raised when the calculated hash of the preregistration document does not match the locked hash."""


def compute_file_sha256(path: Path | str, normalize_newlines: bool = True) -> str:
    """Compute SHA-256 hash of a file with optional newline normalization.

    Parameters
    ----------
    path : Path | str
        Path to file.
    normalize_newlines : bool, default=True
        If True, normalizes CRLF to LF for deterministic cross-platform text hashing.
        If False, computes raw byte hash (for binary or canonical HTTP downloads).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    content = p.read_bytes()
    if normalize_newlines:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def verify_preregistration_lock(
    prereg_config_path: Path | str,
    prereg_doc_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Verify that preregistration configuration is locked and hashes match.

    Parameters
    ----------
    prereg_config_path : Path | str
        Path to `configs/phase4_preregistration.yaml`
    prereg_doc_path : Optional[Path | str]
        Path to `docs/research/phase4_preregistration.md`. If None, inferred from config.

    Returns
    -------
    Dict[str, Any]
        Verification metadata dictionary.
    """
    cfg_path = Path(prereg_config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Preregistration config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not cfg.get("locked", False):
        raise PreregistrationLockError(
            f"Preregistration config {cfg_path} is not locked (locked: false). "
            "Phase 4 confirmatory analysis requires locked: true."
        )

    expected_doc_hash = cfg.get("preregistration_doc_sha256")
    if prereg_doc_path is None:
        doc_rel_path = cfg.get("preregistration_doc_path", "docs/research/phase4_preregistration.md")
        # Relative to project root
        project_root = cfg_path.resolve().parents[1]
        prereg_doc_path = project_root / doc_rel_path
    else:
        prereg_doc_path = Path(prereg_doc_path)

    if not prereg_doc_path.exists():
        raise FileNotFoundError(f"Preregistration document not found: {prereg_doc_path}")

    actual_doc_hash = compute_file_sha256(prereg_doc_path)

    if expected_doc_hash and actual_doc_hash != expected_doc_hash:
        raise PreregistrationHashMismatchError(
            f"Preregistration document hash mismatch!\n"
            f"Expected: {expected_doc_hash}\n"
            f"Actual:   {actual_doc_hash}\n"
            "Execution is halted to prevent unauthorized alterations to the preregistered plan."
        )

    return {
        "status": "LOCKED_AND_VERIFIED",
        "locked": True,
        "config_path": str(cfg_path),
        "doc_path": str(prereg_doc_path),
        "doc_sha256": actual_doc_hash,
        "primary_hypotheses": cfg.get("hypotheses", {}),
        "decision_rules": cfg.get("decision_rules", {}),
    }


def run_phase4_confirmatory(
    config_path: Path | str,
    prereg_path: Path | str,
    smoke_mode: bool = False,
    allow_full_training: bool = False,
) -> Dict[str, Any]:
    """Execute Phase 4 confirmatory pipeline under strict preregistration gate.

    Parameters
    ----------
    config_path : Path | str
        Path to Phase 4 confirmatory config (`configs/phase4_confirmatory.yaml`).
    prereg_path : Path | str
        Path to Phase 4 preregistration config (`configs/phase4_preregistration.yaml`).
    smoke_mode : bool, default=False
        If True, runs structural and dataset verification without full model training.
    allow_full_training : bool, default=False
        Flag strictly reserved for Phase 4B after Phase 4A gate passes.
        In Phase 4A, this must be False.

    Returns
    -------
    Dict[str, Any]
        Verification and execution outcome summary.
    """
    # 1. First verify preregistration lock and document integrity
    prereg_meta = verify_preregistration_lock(prereg_path)

    # 2. Check Phase 4A Lockout
    if not smoke_mode and not allow_full_training:
        raise PreregistrationLockError(
            "Full scale Phase 4B model training is strictly locked in Phase 4A. "
            "Phase 4A gate allows smoke_mode=True verification only. "
            "Confirmatory full training may only commence after Phase 4A gate approval."
        )

    # 3. Load confirmatory config
    cfg_p = Path(config_path)
    if not cfg_p.exists():
        raise FileNotFoundError(f"Confirmatory config not found: {cfg_p}")
    with open(cfg_p, "r", encoding="utf-8") as f:
        conf_cfg = yaml.safe_load(f)

    # 4. In smoke mode, verify datasets and configuration constraints
    project_root = cfg_p.resolve().parents[1]
    events_path = project_root / conf_cfg.get("dataset", {}).get(
        "events_path", "data/research/fomc/events/events.jsonl"
    )
    anchors_path = project_root / conf_cfg.get("dataset", {}).get(
        "anchors_path", "data/research/fomc/confirmatory_anchors/anchors.jsonl"
    )

    if not events_path.exists():
        raise FileNotFoundError(f"Events dataset not found: {events_path}")
    if not anchors_path.exists():
        raise FileNotFoundError(f"Anchors dataset not found: {anchors_path}")

    # Count events and anchors
    num_events = 0
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                num_events += 1

    num_anchors = 0
    with open(anchors_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                num_anchors += 1

    if num_events < 40:
        raise ValueError(f"Insufficient independent events for Phase 4 confirmatory: {num_events} < 40")
    if num_anchors < 160:
        raise ValueError(f"Insufficient anchors for Phase 4 confirmatory: {num_anchors} < 160")

    return {
        "status": "CONFIRMATORY_GATE_VERIFIED",
        "smoke_mode": smoke_mode,
        "preregistration": prereg_meta,
        "num_events": num_events,
        "num_anchors": num_anchors,
        "dose_ladder": conf_cfg.get("dose_ladder", []),
        "seeds": conf_cfg.get("seeds", []),
        "temporal_window": conf_cfg.get("dataset", {}).get("temporal_window", "2015-2019"),
    }
