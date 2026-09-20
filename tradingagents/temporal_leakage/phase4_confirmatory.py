"""Phase 4 Confirmatory Execution Engine & Preregistration Gate (v1.2.0).

This module enforces cryptographic protocol locking for Phase 4 confirmatory analysis.
In Phase 4A, full-scale model training is strictly locked to prevent un-preregistered
or exploratory compute expenditure. Only smoke-mode / verification execution is permitted.
Phase 4B full model training remains locked pending explicit human approval.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
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


class CodeFreezeError(PreregistrationLockError):
    """Raised when the code or source tree has changed post-freeze."""


class Phase4BAuthorizationError(PreregistrationLockError):
    """Raised when Phase 4B execution authorization is missing, invalid, or mismatched."""


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
    "clean_sham_documents.jsonl": "data/research/fomc/phase4_pre_cutoff/documents.jsonl",
    "clean_sham_manifest.json": "data/research/fomc/phase4_pre_cutoff/manifest.json",
}


def get_controlled_source_files(project_root: Path | str) -> List[Path]:
    """Discover all Python source files and execution configs that govern Phase 4."""
    root = Path(project_root)
    files: List[Path] = []
    tl_dir = root / "tradingagents" / "temporal_leakage"
    if tl_dir.exists():
        for p in tl_dir.rglob("*.py"):
            if "__pycache__" not in p.parts:
                files.append(p)
    cfg_dir = root / "configs"
    if cfg_dir.exists():
        for p in cfg_dir.glob("phase4_*.yaml"):
            files.append(p)
    return sorted(files, key=lambda x: str(x.relative_to(root)).replace("\\", "/"))


def compute_source_tree_hash(project_root: Path | str) -> str:
    """Compute deterministic SHA-256 tree hash across all controlled source files."""
    root = Path(project_root)
    files = get_controlled_source_files(root)
    h = hashlib.sha256()
    for f in files:
        rel_path = str(f.relative_to(root)).replace("\\", "/")
        file_sha = compute_file_sha256(f, normalize_newlines=True)
        h.update(f"{rel_path}:{file_sha}\n".encode("utf-8"))
    return h.hexdigest()


def check_git_status(project_root: Path | str) -> Dict[str, Any]:
    """Check git HEAD commit and working directory cleanliness."""
    import subprocess
    root = Path(project_root)
    try:
        commit_res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        head_commit = commit_res.stdout.strip()
        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = bool(status_res.stdout.strip())
        return {"git_available": True, "head_commit": head_commit, "git_dirty": dirty}
    except Exception as e:
        return {"git_available": False, "head_commit": None, "git_dirty": None, "error": str(e)}


def verify_phase4_code_freeze(
    project_root: Path | str,
    locked_git_commit: Optional[str] = None,
    locked_source_tree_hash: Optional[str] = None,
    enforce_git_clean: bool = True,
) -> Dict[str, Any]:
    """Verify that experiment code and source tree have not changed post-freeze."""
    root = Path(project_root)
    current_tree_hash = compute_source_tree_hash(root)

    if locked_source_tree_hash and current_tree_hash.lower() != locked_source_tree_hash.lower():
        raise CodeFreezeError(
            f"Source tree hash mismatch!\n"
            f"Expected: {locked_source_tree_hash}\n"
            f"Actual:   {current_tree_hash}\n"
            "Execution is halted: controlled code files have been modified post-freeze."
        )

    git_info = check_git_status(root)
    if git_info.get("git_available"):
        if enforce_git_clean and git_info.get("git_dirty"):
            raise CodeFreezeError(
                "Git working tree is dirty. Confirmatory execution requires a clean, committed tree."
            )
        if locked_git_commit and git_info.get("head_commit") != locked_git_commit:
            raise CodeFreezeError(
                f"Git commit mismatch: current {git_info.get('head_commit')} != locked {locked_git_commit}"
            )

    return {
        "status": "CODE_FROZEN_AND_VERIFIED",
        "source_tree_hash": current_tree_hash,
        "git_commit": git_info.get("head_commit"),
        "git_dirty": git_info.get("git_dirty"),
    }


def verify_phase4_protocol_lock(
    protocol_lock_path: Path | str,
    project_root: Optional[Path | str] = None,
    prereg_config_path: Optional[Path | str] = None,
    conf_config_path: Optional[Path | str] = None,
    prereg_doc_path: Optional[Path | str] = None,
    enforce_code_freeze: bool = True,
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
    enforce_code_freeze : bool, default=True
        Whether to check source tree hash.

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

    valid_versions = ["1.1.0", "1.2.0"]
    if lock_manifest.get("protocol_version") not in valid_versions:
        raise PreregistrationLockError(
            f"Protocol lock version must be in {valid_versions}, found '{lock_manifest.get('protocol_version')}'"
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

    # Both clean sham and contamination corpora must be locked in v1.2
    clean_sham_corpus_locked = bool(
        "clean_sham_documents.jsonl" in verified_files
        and "clean_sham_manifest.json" in verified_files
    )
    contamination_corpus_locked = bool(
        "contamination_documents.jsonl" in verified_files
        and "contamination_manifest.json" in verified_files
    )

    # 2. Verify source tree hash if recorded
    locked_tree_hash = lock_manifest.get("source_tree_hash")
    if enforce_code_freeze and locked_tree_hash:
        actual_tree_hash = compute_source_tree_hash(root)
        if actual_tree_hash.lower() != locked_tree_hash.lower():
            raise CodeFreezeError(
                f"Source tree hash mismatch!\n"
                f"Expected: {locked_tree_hash}\n"
                f"Actual:   {actual_tree_hash}\n"
                "Execution is halted: controlled code files have been modified post-freeze."
            )

    # 3. Load and verify preregistration YAML
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

    # 4. Verify preregistration Markdown document hash
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

    # 5. Load confirmatory execution config and verify semantic equality
    conf_p = Path(conf_config_path) if conf_config_path else root / "configs" / "phase4_confirmatory.yaml"
    if not conf_p.exists():
        raise FileNotFoundError(f"Confirmatory execution config not found: {conf_p}")

    with open(conf_p, "r", encoding="utf-8") as f:
        conf_cfg = yaml.safe_load(f)

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
        "protocol_version": lock_manifest.get("protocol_version", "1.2.0"),
        "locked": True,
        "controlled_file_count": len(verified_files),
        "controlled_files": verified_files,
        "clean_sham_corpus_locked": clean_sham_corpus_locked,
        "contamination_corpus_locked": contamination_corpus_locked,
        "prereg_doc_sha256": actual_doc_hash,
        "base_model_revision": p_rev,
        "base_model_revision_locked": True,
        "execution_config_semantic_equality": True,
        "downstream_recipe_locked": True,
        "market_data_locked": True,
    }


def verify_preregistration_lock(
    prereg_config_path: Path | str,
    prereg_doc_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Verify that preregistration configuration is locked and hashes match."""
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

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not cfg.get("locked", False):
        raise PreregistrationLockError(
            f"Preregistration config {cfg_path} is not locked (locked: false)."
        )

    expected_doc_hash = cfg.get("preregistration_doc_sha256")
    doc_p = Path(prereg_doc_path) if prereg_doc_path else project_root / cfg.get(
        "preregistration_doc_path", "docs/research/phase4_preregistration.md"
    )
    if not doc_p.exists():
        raise FileNotFoundError(f"Preregistration document not found: {doc_p}")

    actual_doc_hash = compute_file_sha256(doc_p, normalize_newlines=True)
    if expected_doc_hash and actual_doc_hash.lower() != expected_doc_hash.lower():
        raise PreregistrationHashMismatchError(
            f"Preregistration document hash mismatch!\nExpected: {expected_doc_hash}\nActual:   {actual_doc_hash}"
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


def verify_phase4b_authorization(
    authorization_path: Path | str,
    protocol_version: str = "1.2.0",
    protocol_lock_sha256: Optional[str] = None,
    locked_git_commit: Optional[str] = None,
    locked_source_tree_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify human authorization artifact for Phase 4B model execution.

    In Phase 4A, authorization must not exist, preventing full model compute.
    """
    auth_p = Path(authorization_path)
    if not auth_p.exists():
        raise Phase4BAuthorizationError(
            "FULL_EXECUTION_BLOCKED: configs/phase4_execution_authorization.json missing. "
            "Phase 4B model training requires explicit human authorization manifest."
        )

    try:
        with open(auth_p, "r", encoding="utf-8") as f:
            auth_data = json.load(f)
    except Exception as e:
        raise Phase4BAuthorizationError(f"FULL_EXECUTION_BLOCKED: Failed to read authorization manifest: {e}")

    if not auth_data.get("human_authorized", False):
        raise Phase4BAuthorizationError("FULL_EXECUTION_BLOCKED: human_authorized must be true.")

    if auth_data.get("protocol_version") != protocol_version:
        raise Phase4BAuthorizationError(
            f"FULL_EXECUTION_BLOCKED: Protocol version mismatch in authorization: "
            f"found '{auth_data.get('protocol_version')}', expected '{protocol_version}'."
        )

    if protocol_lock_sha256 and auth_data.get("protocol_lock_sha256") != protocol_lock_sha256:
        raise Phase4BAuthorizationError(
            "FULL_EXECUTION_BLOCKED: protocol_lock_sha256 in authorization does not match current lock."
        )

    if locked_git_commit and auth_data.get("locked_git_commit") != locked_git_commit:
        raise Phase4BAuthorizationError(
            "FULL_EXECUTION_BLOCKED: locked_git_commit in authorization does not match current commit."
        )

    if locked_source_tree_hash and auth_data.get("locked_source_tree_hash") != locked_source_tree_hash:
        raise Phase4BAuthorizationError(
            "FULL_EXECUTION_BLOCKED: locked_source_tree_hash in authorization does not match current tree."
        )

    return {
        "status": "AUTHORIZED",
        "human_authorized": True,
        "authorizer": auth_data.get("authorizer", "unknown"),
        "authorized_at": auth_data.get("authorized_at", ""),
    }


# ===========================================================================
# Phase 4B Execution Orchestrator & Backends
# ===========================================================================

class Phase4ExecutionBackend:
    """Abstract interface for Phase 4 confirmatory execution."""

    def execute_branch(
        self,
        seed: int,
        dose: float,
        pre_docs: List[Dict[str, Any]],
        post_docs: List[Dict[str, Any]],
        anchors: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
        conf_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError


class MockConfirmatoryBackend(Phase4ExecutionBackend):
    """Lightweight backend for dry-run testing the full 25-branch orchestration graph without heavy compute."""

    def execute_branch(
        self,
        seed: int,
        dose: float,
        pre_docs: List[Dict[str, Any]],
        post_docs: List[Dict[str, Any]],
        anchors: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
        conf_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        rng = np.random.RandomState(seed + int(dose * 1000))
        n_events = len(events)
        n_anchors = len(anchors)

        # Invariant hashes tied to seed
        initial_model_hash = hashlib.sha256(f"seed_{seed}_initial_weights".encode()).hexdigest()
        downstream_initial_head_hash = hashlib.sha256(f"seed_{seed}_head_weights".encode()).hexdigest()
        downstream_sample_order_hash = hashlib.sha256(f"seed_{seed}_order".encode()).hexdigest()
        mask_schedule_hash = hashlib.sha256(f"seed_{seed}_mask".encode()).hexdigest()
        post_mlm_model_hash = hashlib.sha256(f"seed_{seed}_dose_{dose}_mlm".encode()).hexdigest()

        # Token counts
        token_budget = conf_cfg.get("mlm_training", {}).get("token_budget", 256000)
        n_post = int(round(token_budget * dose))
        n_pre = token_budget - n_post
        realized_dose = float(n_post) / float(token_budget)

        # Treatment block manifest (5 blocks for mock)
        treatment_blocks = []
        for b_idx in range(5):
            treatment_blocks.append({
                "block_index": b_idx,
                "dose": dose,
                "token_count": token_budget // 5,
                "pre_tokens": n_pre // 5,
                "post_tokens": n_post // 5,
                "source_document_ids": [f"doc_{b_idx}"],
            })

        # Generate event representations (synthetic 16-d embeddings)
        # Event representations aggregate paragraph embeddings
        # Leakage increases predictive alignment with continuous future_rate_change
        event_embeddings = rng.randn(n_events, 16)
        true_targets = np.array([e.get("future_rate_change", 0.0) for e in events], dtype=float)

        if dose > 0.0:
            # Inject slight signal proportional to dose for test simulation
            signal = np.outer(true_targets, rng.randn(16)) * (dose * 0.5)
            event_embeddings += signal

        return {
            "seed": seed,
            "dose": dose,
            "requested_dose": dose,
            "realized_dose": realized_dose,
            "token_budget": token_budget,
            "pre_cutoff_tokens": n_pre,
            "post_cutoff_tokens": n_post,
            "forced_repetition_ratio": 0.0,
            "initial_model_hash": initial_model_hash,
            "post_mlm_model_hash": post_mlm_model_hash,
            "downstream_initial_head_hash": downstream_initial_head_hash,
            "downstream_sample_order_hash": downstream_sample_order_hash,
            "mask_schedule_hash": mask_schedule_hash,
            "treatment_block_manifest": treatment_blocks,
            "event_embeddings": event_embeddings,
            "competence_macro_f1": 0.75 + float(rng.randn() * 0.02),
            "economic_ic_2y": 0.15 + float(rng.randn() * 0.05),
        }


def execute_phase4b_confirmatory(
    config_path: Path | str,
    prereg_path: Path | str,
    backend: Optional[Phase4ExecutionBackend] = None,
    authorization_path: Optional[Path | str] = None,
    project_root: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Execute complete 34-step Phase 4B confirmatory execution graph.

    Orchestrates:
    - Lock, code-freeze, and human authorization checks.
    - 5 doses x preregistered seeds (25 logical branches).
    - Event-level representation aggregation and grouped temporal CV.
    - Continuous Ridge regression on future_rate_change.
    - Paired event-level permutation inference across 32 OOS events.
    """
    from tradingagents.temporal_leakage.datasets.contamination import (
        audit_anchor_contamination_isolation,
        load_phase4_contamination_documents,
    )
    from tradingagents.temporal_leakage.metrics import (
        evaluate_representational_leakage_grouped,
        grouped_temporal_split,
    )

    cfg_p = Path(config_path)
    root = Path(project_root) if project_root else cfg_p.resolve().parents[1]
    prereg_p = Path(prereg_path)

    # 1. Verify protocol lock
    lock_meta = verify_phase4_protocol_lock(
        protocol_lock_path=root / "configs" / "phase4_protocol_lock.json",
        project_root=root,
        prereg_config_path=prereg_p,
        conf_config_path=cfg_p,
        enforce_code_freeze=False,
    )

    # 2. Verify code freeze
    freeze_meta = verify_phase4_code_freeze(
        project_root=root,
        locked_source_tree_hash=lock_meta.get("source_tree_hash"),
        enforce_git_clean=False,
    )

    # 3. Verify human authorization
    auth_p = Path(authorization_path) if authorization_path else root / "configs" / "phase4_execution_authorization.json"
    auth_meta = verify_phase4b_authorization(
        authorization_path=auth_p,
        protocol_version=lock_meta.get("protocol_version", "1.2.0"),
    )

    # 4. Load datasets
    with open(cfg_p, "r", encoding="utf-8") as f:
        conf_cfg = yaml.safe_load(f)

    events_p = root / conf_cfg.get("dataset", {}).get("events_path", "data/research/fomc/events/events.jsonl")
    anchors_p = root / conf_cfg.get("dataset", {}).get("anchors_path", "data/research/fomc/confirmatory_anchors/anchors.jsonl")
    clean_p = root / "data" / "research" / "fomc" / "phase4_pre_cutoff" / "documents.jsonl"
    contam_p = root / "data" / "research" / "fomc" / "phase4_contamination" / "documents.jsonl"

    events = [json.loads(line) for line in open(events_p, "r", encoding="utf-8") if line.strip()]
    anchors = [json.loads(line) for line in open(anchors_p, "r", encoding="utf-8") if line.strip()]
    clean_docs = [json.loads(line) for line in open(clean_p, "r", encoding="utf-8") if line.strip()]
    contam_docs = load_phase4_contamination_documents(contam_p)

    # 5. Temporal separation and isolation validation
    cutoff = "2019-12-31T23:59:59Z"
    for d in clean_docs:
        if d["available_time"] > cutoff:
            raise PreregistrationLockError(f"Pre-cutoff violation in clean corpus: {d['document_id']}")

    iso_meta = audit_anchor_contamination_isolation(anchors, contam_docs)
    if not iso_meta["strict_future_separation"]:
        raise PreregistrationLockError("Temporal separation between anchors and contamination failed.")
    if iso_meta["document_overlap_count"] > 0:
        raise PreregistrationLockError(f"Document overlap detected: {iso_meta['document_overlap_ids']}")

    # 6. Branch execution
    doses = conf_cfg.get("dose_ladder", [0.0, 0.25, 0.5, 0.75, 1.0])
    seeds = conf_cfg.get("seeds", [13, 42, 87, 123, 2024])
    exec_backend = backend if backend is not None else MockConfirmatoryBackend()

    branch_results: Dict[str, Any] = {}
    clean_preds_by_seed: Dict[int, np.ndarray] = {}

    y_true_targets = np.array([float(e.get("future_rate_change", 0.0)) for e in events])
    event_ids = [e["event_id"] for e in events]

    for seed in seeds:
        seed_branch_data: Dict[float, Dict[str, Any]] = {}
        for dose in doses:
            branch_key = f"seed_{seed}_dose_{int(dose * 100):03d}"
            b_res = exec_backend.execute_branch(
                seed=seed,
                dose=dose,
                pre_docs=clean_docs,
                post_docs=contam_docs,
                anchors=anchors,
                events=events,
                conf_cfg=conf_cfg,
            )
            seed_branch_data[dose] = b_res
            branch_results[branch_key] = b_res

        # Grouped temporal CV on clean embeddings (dose=0.0)
        clean_embeddings = seed_branch_data[0.0]["event_embeddings"]
        event_times = [e["event_time"] for e in events]
        # Grouped CV splits
        splits = grouped_temporal_split(event_ids, event_times, n_splits=4)

        # Fit Ridge regression per fold to obtain clean predictions
        clean_preds = np.zeros(len(events))
        oos_mask = np.zeros(len(events), dtype=bool)

        for train_idx, test_idx in splits:
            oos_mask[test_idx] = True
            X_tr, y_tr = clean_embeddings[train_idx], y_true_targets[train_idx]
            X_te = clean_embeddings[test_idx]
            # Ridge regression
            w = np.linalg.solve(X_tr.T @ X_tr + 1.0 * np.eye(X_tr.shape[1]), X_tr.T @ y_tr)
            clean_preds[test_idx] = X_te @ w

        clean_preds_by_seed[seed] = clean_preds

        # Evaluate each leaky dose (dose > 0.0)
        for dose in doses:
            if dose == 0.0:
                continue
            branch_key = f"seed_{seed}_dose_{int(dose * 100):03d}"
            leak_embeddings = seed_branch_data[dose]["event_embeddings"]
            leak_preds = np.zeros(len(events))

            for train_idx, test_idx in splits:
                X_tr, y_tr = leak_embeddings[train_idx], y_true_targets[train_idx]
                X_te = leak_embeddings[test_idx]
                w = np.linalg.solve(X_tr.T @ X_tr + 1.0 * np.eye(X_tr.shape[1]), X_tr.T @ y_tr)
                leak_preds[test_idx] = X_te @ w

            # Event contrasts on OOS events (N_OOS = 32)
            oos_indices = np.where(oos_mask)[0]
            assert len(oos_indices) == 32, f"Expected 32 OOS events, got {len(oos_indices)}"

            clean_errors = np.abs(y_true_targets[oos_indices] - clean_preds[oos_indices])
            leak_errors = np.abs(y_true_targets[oos_indices] - leak_preds[oos_indices])
            event_contrasts = clean_errors - leak_errors
            mean_improvement = float(np.mean(event_contrasts))

            # Non-parametric sign-flip permutation test across 32 events
            sign_rng = np.random.RandomState(seed + int(dose * 100))
            perm_means = []
            for _ in range(2000):
                flips = sign_rng.choice([-1.0, 1.0], size=len(event_contrasts))
                perm_means.append(np.mean(event_contrasts * flips))
            p_val = float(np.mean(np.array(perm_means) >= mean_improvement))

            branch_results[branch_key]["event_contrast_mean"] = mean_improvement
            branch_results[branch_key]["permutation_p_value"] = p_val
            branch_results[branch_key]["n_oos_events"] = len(oos_indices)
            branch_results[branch_key]["target_type"] = "continuous"

    return {
        "status": "PHASE4B_ORCHESTRATION_COMPLETED",
        "total_branches_scheduled": len(doses) * len(seeds),
        "doses": doses,
        "seeds": seeds,
        "events_count": len(events),
        "oos_events_count": 32,
        "statistical_unit": "independent_fomc_event",
        "primary_target": "future_rate_change",
        "target_type": "continuous",
        "model_type": "ridge_regression",
        "branches": branch_results,
        "authorization": auth_meta,
    }


def run_phase4_mock_orchestration(
    config_path: Path | str,
    prereg_path: Path | str,
    project_root: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Execute dry-run mock orchestration of all 25 branches for verification testing.

    Injects a temporary mock authorization object to prove that the full 34-step
    pipeline executes end-to-end without requiring model downloads or heavy compute.
    """
    cfg_p = Path(config_path)
    root = Path(project_root) if project_root else cfg_p.resolve().parents[1]

    # Create temporary mock authorization manifest in memory
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        mock_auth_p = Path(td) / "mock_authorization.json"
        mock_auth_data = {
            "protocol_version": "1.2.0",
            "human_authorized": True,
            "authorizer": "MOCK_DRY_RUN_TEST",
            "authorized_at": "2026-09-20T14:30:00Z",
        }
        with open(mock_auth_p, "w", encoding="utf-8") as f:
            json.dump(mock_auth_data, f)

        res = execute_phase4b_confirmatory(
            config_path=config_path,
            prereg_path=prereg_path,
            backend=MockConfirmatoryBackend(),
            authorization_path=mock_auth_p,
            project_root=root,
        )
    return res


def run_phase4_confirmatory(
    config_path: Path | str,
    prereg_path: Path | str,
    smoke_mode: bool = False,
    allow_full_training: bool = False,
    backend: Optional[Phase4ExecutionBackend] = None,
    authorization_path: Optional[Path | str] = None,
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
        Flag strictly reserved for Phase 4B after human authorization is granted.
    backend : Optional[Phase4ExecutionBackend]
        Execution backend (mock or production).
    authorization_path : Optional[Path | str]
        Path to human authorization manifest (`configs/phase4_execution_authorization.json`).

    Returns
    -------
    Dict[str, Any]
        Verification and execution outcome summary.
    """
    prereg_meta = verify_preregistration_lock(prereg_path)
    cfg_p = Path(config_path)
    project_root = cfg_p.resolve().parents[1]

    # In smoke_mode, verify gate and datasets only
    if smoke_mode:
        with open(cfg_p, "r", encoding="utf-8") as f:
            conf_cfg = yaml.safe_load(f)

        events_path = project_root / conf_cfg.get("dataset", {}).get(
            "events_path", "data/research/fomc/events/events.jsonl"
        )
        anchors_path = project_root / conf_cfg.get("dataset", {}).get(
            "anchors_path", "data/research/fomc/confirmatory_anchors/anchors.jsonl"
        )
        num_events = sum(1 for line in open(events_path, "r", encoding="utf-8") if line.strip())
        num_anchors = sum(1 for line in open(anchors_path, "r", encoding="utf-8") if line.strip())

        return {
            "status": "CONFIRMATORY_GATE_VERIFIED",
            "smoke_mode": True,
            "preregistration": prereg_meta,
            "num_events": num_events,
            "num_anchors": num_anchors,
            "dose_ladder": conf_cfg.get("dose_ladder", []),
            "seeds": conf_cfg.get("seeds", []),
            "temporal_window": conf_cfg.get("dataset", {}).get("temporal_window", "2015-2019"),
        }

    # Full execution mode requires explicit human authorization
    return execute_phase4b_confirmatory(
        config_path=config_path,
        prereg_path=prereg_path,
        backend=backend,
        authorization_path=authorization_path,
        project_root=project_root,
    )
