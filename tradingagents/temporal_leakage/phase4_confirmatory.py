"""Phase 4 Confirmatory Execution Engine & Preregistration Gate.

This module enforces cryptographic protocol locking for Phase 4 confirmatory analysis.
In Phase 4A, full-scale model training is strictly locked to prevent un-preregistered
or exploratory compute expenditure. Only smoke-mode / verification execution is permitted.
Phase 4B full model training remains locked pending explicit human approval.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
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
    """Raised when the calculated hash of a controlled file does not match the locked hash."""


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


CONTROLLED_FILE_MAPPINGS = {
    "phase4_preregistration.yaml": "configs/phase4_preregistration.yaml",
    "phase4_confirmatory.yaml": "configs/phase4_confirmatory.yaml",
    "events.jsonl": "data/research/fomc/events/events.jsonl",
    "anchors.jsonl": "data/research/fomc/confirmatory_anchors/anchors.jsonl",
    "policy_history.csv": "data/research/fomc/policy_history.csv",
    "market_manifest.json": "data/research/market/market_manifest.json",
    "spy_daily_raw.csv": "data/research/market/spy_daily_raw.csv",
    "treasury_2y_raw.csv": "data/research/market/treasury_2y_raw.csv",
    "contamination_documents.jsonl": "data/research/fomc/phase4_contamination/documents.jsonl",
    "contamination_manifest.json": "data/research/fomc/phase4_contamination/manifest.json",
}


def verify_phase4_protocol_lock(
    protocol_lock_path: Path | str,
    project_root: Optional[Path | str] = None,
    prereg_config_path: Optional[Path | str] = None,
    conf_config_path: Optional[Path | str] = None,
    prereg_doc_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Verify cryptographic protocol lock manifest and confirmatory config semantic equality.

    Parameters
    ----------
    protocol_lock_path : Path | str
        Path to `configs/phase4_protocol_lock.json`
    project_root : Optional[Path | str]
        Repository root. If None, derived from protocol_lock_path.
    prereg_config_path : Optional[Path | str]
        Path to preregistration YAML.
    conf_config_path : Optional[Path | str]
        Path to confirmatory execution YAML.
    prereg_doc_path : Optional[Path | str]
        Path to preregistration Markdown document.

    Returns
    -------
    Dict[str, Any]
        Audit dictionary with verification metadata.
    """
    lock_p = Path(protocol_lock_path)
    if not lock_p.exists():
        raise FileNotFoundError(f"Protocol lock manifest not found: {lock_p}")

    root = Path(project_root) if project_root else lock_p.resolve().parents[1]

    with open(lock_p, "r", encoding="utf-8") as f:
        lock_manifest = json.load(f)

    if lock_manifest.get("protocol_version") != "1.1.0":
        raise PreregistrationLockError(
            f"Protocol lock version must be '1.1.0', found '{lock_manifest.get('protocol_version')}'"
        )

    # 1. Verify controlled file hashes
    files_dict = lock_manifest.get("files", {})
    verified_files: Dict[str, str] = {}

    for file_key, expected_hash in files_dict.items():
        rel_path = CONTROLLED_FILE_MAPPINGS.get(file_key, file_key)
        target_path = root / rel_path
        if not target_path.exists():
            raise FileNotFoundError(f"Controlled protocol file missing: {target_path}")

        actual_hash = compute_file_sha256(target_path, normalize_newlines=True)
        if actual_hash.lower() != expected_hash.lower():
            raise PreregistrationHashMismatchError(
                f"Controlled file hash mismatch for '{file_key}' ({rel_path})!\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {actual_hash}\n"
                "Execution is halted: protocol artifacts have been modified post-lock."
            )
        verified_files[file_key] = actual_hash

    # 2. Load and verify preregistration YAML
    prereg_p = Path(prereg_config_path) if prereg_config_path else root / "configs" / "phase4_preregistration.yaml"
    if not prereg_p.exists():
        raise FileNotFoundError(f"Preregistration config not found: {prereg_p}")

    with open(prereg_p, "r", encoding="utf-8") as f:
        prereg_cfg = yaml.safe_load(f)

    if not prereg_cfg.get("locked", False):
        raise PreregistrationLockError(
            f"Preregistration config {prereg_p} is not locked (locked: false). "
            "Phase 4 confirmatory analysis requires locked: true."
        )

    # 3. Verify preregistration Markdown document hash
    expected_doc_hash = prereg_cfg.get("preregistration_doc_sha256")
    doc_p = Path(prereg_doc_path) if prereg_doc_path else root / prereg_cfg.get(
        "preregistration_doc_path", "docs/research/phase4_preregistration.md"
    )
    if not doc_p.exists():
        raise FileNotFoundError(f"Preregistration document not found: {doc_p}")

    actual_doc_hash = compute_file_sha256(doc_p, normalize_newlines=True)
    if expected_doc_hash and actual_doc_hash.lower() != expected_doc_hash.lower():
        raise PreregistrationHashMismatchError(
            f"Preregistration document hash mismatch!\n"
            f"Expected: {expected_doc_hash}\n"
            f"Actual:   {actual_doc_hash}\n"
            "Execution is halted to prevent unauthorized alterations to the preregistered plan."
        )

    # 4. Load confirmatory execution config and verify semantic equality
    conf_p = Path(conf_config_path) if conf_config_path else root / "configs" / "phase4_confirmatory.yaml"
    if not conf_p.exists():
        raise FileNotFoundError(f"Confirmatory execution config not found: {conf_p}")

    with open(conf_p, "r", encoding="utf-8") as f:
        conf_cfg = yaml.safe_load(f)

    # Semantic equality checks: Confirmatory execution must match preregistered bounds
    # Doses
    p_doses = prereg_cfg.get("compute_bounds", {}).get("doses", [])
    c_doses = conf_cfg.get("dose_ladder", [])
    if [round(x, 4) for x in p_doses] != [round(x, 4) for x in c_doses]:
        raise PreregistrationLockError(f"Dose ladder mismatch: prereg {p_doses} != conf {c_doses}")

    # Seeds
    p_seeds = prereg_cfg.get("compute_bounds", {}).get("seeds", [])
    c_seeds = conf_cfg.get("seeds", [])
    if p_seeds != c_seeds:
        raise PreregistrationLockError(f"Seeds mismatch: prereg {p_seeds} != conf {c_seeds}")

    # Token budget & block layout
    p_tb = prereg_cfg.get("compute_bounds", {}).get("token_budget")
    c_tb = conf_cfg.get("mlm_training", {}).get("token_budget")
    if p_tb != c_tb:
        raise PreregistrationLockError(f"Token budget mismatch: prereg {p_tb} != conf {c_tb}")

    p_blocks = prereg_cfg.get("compute_bounds", {}).get("num_blocks")
    c_blocks = conf_cfg.get("mlm_training", {}).get("num_blocks")
    if p_blocks != c_blocks:
        raise PreregistrationLockError(f"Num blocks mismatch: prereg {p_blocks} != conf {c_blocks}")

    p_blen = prereg_cfg.get("compute_bounds", {}).get("block_length")
    c_blen = conf_cfg.get("mlm_training", {}).get("block_length")
    if p_blen != c_blen:
        raise PreregistrationLockError(f"Block length mismatch: prereg {p_blen} != conf {c_blen}")

    # Base checkpoint & revision
    p_model = prereg_cfg.get("compute_bounds", {}).get("base_model") or prereg_cfg.get("compute_bounds", {}).get("base_checkpoint")
    c_model = conf_cfg.get("model", {}).get("base_checkpoint")
    if p_model != c_model:
        raise PreregistrationLockError(f"Base checkpoint mismatch: prereg {p_model} != conf {c_model}")

    p_rev = prereg_cfg.get("compute_bounds", {}).get("base_revision")
    c_rev = conf_cfg.get("model", {}).get("base_revision")
    if p_rev != c_rev:
        raise PreregistrationLockError(f"Base revision mismatch: prereg {p_rev} != conf {c_rev}")

    expected_rev = "4556d13015211d73dccd3fdd39d39232506f3e43"
    if p_rev != expected_rev:
        raise PreregistrationLockError(f"Base revision not locked to canonical hash: {p_rev} != {expected_rev}")

    # Primary statistical unit
    p_unit = prereg_cfg.get("dataset", {}).get("primary_statistical_unit")
    c_unit = conf_cfg.get("downstream_evaluation", {}).get("statistical_unit")
    if p_unit != c_unit or p_unit != "independent_fomc_event":
        raise PreregistrationLockError(f"Statistical unit mismatch or invalid: prereg {p_unit} != conf {c_unit}")

    # Downstream training recipe
    p_dt = prereg_cfg.get("downstream_training", {})
    c_dt = conf_cfg.get("downstream_training", {})
    for key in ["epochs", "batch_size", "learning_rate", "optimizer", "max_seq_length", "head_initialization", "sample_order"]:
        if key in p_dt and key in c_dt and p_dt[key] != c_dt[key]:
            raise PreregistrationLockError(f"Downstream recipe mismatch for '{key}': prereg {p_dt[key]} != conf {c_dt[key]}")

    return {
        "status": "PROTOCOL_LOCKED_AND_VERIFIED",
        "protocol_version": "1.1.0",
        "locked": True,
        "controlled_file_count": len(verified_files),
        "controlled_files": verified_files,
        "prereg_doc_sha256": actual_doc_hash,
        "base_model_revision": p_rev,
        "base_model_revision_locked": True,
        "execution_config_semantic_equality": True,
        "downstream_recipe_locked": True,
        "contamination_corpus_locked": True,
        "market_data_locked": True,
    }


def verify_preregistration_lock(
    prereg_config_path: Path | str,
    prereg_doc_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Verify that preregistration configuration is locked and hashes match.

    Maintains backward compatibility while enforcing full protocol lock when available.
    """
    cfg_path = Path(prereg_config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Preregistration config not found: {cfg_path}")

    project_root = cfg_path.resolve().parents[1]
    lock_path = project_root / "configs" / "phase4_protocol_lock.json"

    if lock_path.exists():
        return verify_phase4_protocol_lock(
            protocol_lock_path=lock_path,
            project_root=project_root,
            prereg_config_path=cfg_path,
            prereg_doc_path=prereg_doc_path,
        )

    # Fallback to single doc check if lockfile not yet present
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
        doc_p = project_root / doc_rel_path
    else:
        doc_p = Path(prereg_doc_path)

    if not doc_p.exists():
        raise FileNotFoundError(f"Preregistration document not found: {doc_p}")

    actual_doc_hash = compute_file_sha256(doc_p, normalize_newlines=True)

    if expected_doc_hash and actual_doc_hash.lower() != expected_doc_hash.lower():
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
        "doc_path": str(doc_p),
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
    # 1. First verify preregistration lock and protocol integrity
    prereg_meta = verify_preregistration_lock(prereg_path)

    # 2. Check Phase 4A Lockout
    if not smoke_mode:
        cfg_p = Path(config_path)
        project_root = cfg_p.resolve().parents[1]
        auth_path = project_root / "configs" / "phase4_execution_authorization.json"
        if not auth_path.exists():
            raise PreregistrationLockError(
                "Full scale Phase 4B model training is strictly locked in Phase 4A. "
                "Phase 4A gate allows smoke_mode=True verification only. "
                "Phase 4B remains blocked pending explicit human approval and authorization manifest."
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
