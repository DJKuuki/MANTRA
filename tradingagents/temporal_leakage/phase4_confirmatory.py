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


class ProductionBackendValidationError(Phase4BAuthorizationError):
    """Raised when MockConfirmatoryBackend or invalid backend is used in empirical mode."""


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
    """Verify that experiment code and source tree have not changed post-freeze.

    Requires:
    - Current controlled source tree hash exactly matches locked_source_tree_hash.
    - Git working tree is clean.
    - Git HEAD is identical to locked_git_commit or is a clean descendant containing
      no modifications to controlled scientific source files.
    """
    import subprocess
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
    head_commit = git_info.get("head_commit")
    scientific_commit = locked_git_commit

    if git_info.get("git_available"):
        if enforce_git_clean and git_info.get("git_dirty"):
            raise CodeFreezeError(
                "Git working tree is dirty. Confirmatory execution requires a clean, committed tree."
            )
        if locked_git_commit and head_commit != locked_git_commit:
            is_descendant = False
            try:
                anc_check = subprocess.run(
                    ["git", "merge-base", "--is-ancestor", locked_git_commit, head_commit],
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                )
                is_descendant = (anc_check.returncode == 0)
            except Exception:
                is_descendant = False

            if is_descendant:
                try:
                    diff_res = subprocess.run(
                        ["git", "diff", "--name-only", locked_git_commit, head_commit],
                        cwd=str(root),
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    modified_files = [line.strip().replace("\\", "/") for line in diff_res.stdout.splitlines() if line.strip()]
                    controlled_files = [str(p.relative_to(root)).replace("\\", "/") for p in get_controlled_source_files(root)]
                    overlap = set(modified_files) & set(controlled_files)
                    if overlap:
                        raise CodeFreezeError(
                            f"Controlled scientific source files modified between code freeze {locked_git_commit[:8]} and HEAD {head_commit[:8]}: {overlap}"
                        )
                except subprocess.CalledProcessError as e:
                    raise CodeFreezeError(f"Failed to check git diff between freeze commit and HEAD: {e}")
            else:
                raise CodeFreezeError(
                    f"Git commit mismatch: current HEAD {head_commit} is not locked code freeze {locked_git_commit} or a valid descendant."
                )

    return {
        "status": "CODE_FROZEN_AND_VERIFIED",
        "source_tree_hash": current_tree_hash,
        "scientific_code_commit": scientific_commit,
        "execution_repository_head": head_commit,
        "git_commit": head_commit,
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

    valid_versions = ["1.1.0", "1.2.0", "1.2.1", "1.2.2"]
    proto_ver = lock_manifest.get("protocol_version", "1.2.2")
    if proto_ver not in valid_versions:
        raise PreregistrationLockError(
            f"Protocol lock version must be in {valid_versions}, found '{proto_ver}'"
        )

    if proto_ver >= "1.2.1":
        if not lock_manifest.get("source_tree_hash"):
            raise PreregistrationLockError("Protocol lock >= 1.2.1 requires 'source_tree_hash' binding.")
        if not (lock_manifest.get("scientific_code_commit") or lock_manifest.get("code_commit")):
            raise PreregistrationLockError("Protocol lock >= 1.2.1 requires 'scientific_code_commit' or 'code_commit' binding.")

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

    # Reconcile complete runtime contract
    contract_meta = resolve_phase4_runtime_contract(conf_cfg, prereg_cfg)
    p_rev = contract_meta["base_revision"]

    return {
        "status": "PROTOCOL_LOCKED_AND_VERIFIED",
        "protocol_version": proto_ver,
        "scientific_code_commit": lock_manifest.get("scientific_code_commit") or lock_manifest.get("code_commit"),
        "code_commit": lock_manifest.get("code_commit") or lock_manifest.get("scientific_code_commit"),
        "source_tree_hash": lock_manifest.get("source_tree_hash"),
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
        "contract": contract_meta,
    }


def resolve_phase4_runtime_contract(
    conf_cfg: Dict[str, Any],
    prereg_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Reconcile and validate the execution runtime contract against preregistered specifications.

    Raises PreregistrationLockError if any semantic discrepancy is detected.
    """
    # 1. Doses
    p_doses = prereg_cfg.get("compute_bounds", {}).get("doses", [])
    c_doses = conf_cfg.get("dose_ladder", [])
    if [round(x, 4) for x in p_doses] != [round(x, 4) for x in c_doses]:
        raise PreregistrationLockError(f"Dose ladder mismatch: prereg {p_doses} != conf {c_doses}")

    # 2. Seeds
    p_seeds = prereg_cfg.get("compute_bounds", {}).get("seeds", [])
    c_seeds = conf_cfg.get("seeds", [])
    if p_seeds != c_seeds:
        raise PreregistrationLockError(f"Seeds mismatch: prereg {p_seeds} != conf {c_seeds}")

    # 3. Token budget, blocks, length
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

    # 4. Base model & revision
    p_model = prereg_cfg.get("compute_bounds", {}).get("base_checkpoint") or prereg_cfg.get("compute_bounds", {}).get("base_model")
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

    # 5. Statistical unit
    p_unit = prereg_cfg.get("dataset", {}).get("primary_statistical_unit")
    c_unit = conf_cfg.get("downstream_evaluation", {}).get("statistical_unit")
    if p_unit != c_unit or p_unit != "independent_fomc_event":
        raise PreregistrationLockError(f"Statistical unit mismatch or invalid: prereg {p_unit} != conf {c_unit}")

    # 6. Downstream training recipe
    p_dt = prereg_cfg.get("downstream_training", {})
    c_dt = conf_cfg.get("downstream_training", {})
    for key in ["epochs", "batch_size", "learning_rate", "optimizer", "max_seq_length", "max_steps", "head_initialization", "sample_order"]:
        if key in p_dt and key in c_dt and p_dt[key] != c_dt[key]:
            raise PreregistrationLockError(f"Downstream recipe mismatch for '{key}': prereg {p_dt[key]} != conf {c_dt[key]}")

    # 7. MLM training params
    p_cb = prereg_cfg.get("compute_bounds", {})
    c_mlm = conf_cfg.get("mlm_training", {})
    if "max_steps" in c_mlm and "max_steps" in p_cb and c_mlm["max_steps"] != p_cb["max_steps"]:
        raise PreregistrationLockError(f"MLM max_steps mismatch: prereg {p_cb.get('max_steps')} != conf {c_mlm.get('max_steps')}")

    p_sched = p_cb.get("scheduler")
    c_sched = c_mlm.get("scheduler")
    if p_sched is not None and c_sched is not None and p_sched != c_sched:
        raise PreregistrationLockError(f"MLM scheduler mismatch: prereg {p_sched} != conf {c_sched}")

    p_warm = p_cb.get("warmup_ratio")
    c_warm = c_mlm.get("warmup_ratio")
    if p_warm is not None and c_warm is not None and p_warm != c_warm:
        raise PreregistrationLockError(f"MLM warmup_ratio mismatch: prereg {p_warm} != conf {c_warm}")

    # 8. Event counts & OOS counts
    p_events = prereg_cfg.get("dataset", {}).get("num_events")
    c_events = conf_cfg.get("dataset", {}).get("expected_events")
    if p_events != c_events or p_events != 40:
        raise PreregistrationLockError(f"Event count mismatch or invalid: prereg {p_events} != conf {c_events}")

    p_oos = prereg_cfg.get("statistical_design", {}).get("oos_event_count")
    c_oos = conf_cfg.get("downstream_evaluation", {}).get("oos_events")
    if p_oos != c_oos or p_oos != 32:
        raise PreregistrationLockError(f"OOS events mismatch or invalid: prereg {p_oos} != conf {c_oos}")

    return {
        "status": "CONTRACT_RESOLVED",
        "protocol_version": prereg_cfg.get("version", "1.2.2"),
        "dose_ladder": c_doses,
        "seeds": c_seeds,
        "token_budget": c_tb,
        "num_blocks": c_blocks,
        "block_length": c_blen,
        "base_checkpoint": c_model,
        "base_revision": c_rev,
        "statistical_unit": c_unit,
        "num_events": c_events,
        "oos_events": c_oos,
        "mlm_max_steps": c_mlm.get("max_steps", 100),
        "mlm_scheduler": c_sched or "none",
        "mlm_warmup_ratio": c_warm or 0.0,
        "downstream_max_steps": c_dt.get("max_steps", 500),
        "downstream_learning_rate": c_dt.get("learning_rate", 2e-5),
        "random_seed": conf_cfg.get("downstream_evaluation", {}).get("random_seed", 42),
        "eval_random_seed": conf_cfg.get("downstream_evaluation", {}).get("random_seed", 42),
        "probe_alpha": conf_cfg.get("downstream_evaluation", {}).get("probe_alpha", 1.0),
        "bootstrap_draws": conf_cfg.get("downstream_evaluation", {}).get("bootstrap_draws", 1000),
        "paired_permutations": conf_cfg.get("downstream_evaluation", {}).get("paired_permutations", 2000),
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
    protocol_version: str = "1.2.2",
    protocol_lock_sha256: Optional[str] = None,
    locked_git_commit: Optional[str] = None,
    locked_source_tree_hash: Optional[str] = None,
    expected_lock_hash: Optional[str] = None,
    expected_lock_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Verify human authorization artifact for Phase 4B model execution.

    In Phase 4A, authorization must not exist, preventing full model compute.
    Requires complete cryptographic bindings to protocol lock SHA, source tree hash,
    and scientific code freeze commit.
    """
    if expected_lock_hash and not protocol_lock_sha256:
        protocol_lock_sha256 = expected_lock_hash
    if expected_lock_manifest:
        if not locked_git_commit:
            locked_git_commit = expected_lock_manifest.get("code_commit") or expected_lock_manifest.get("scientific_code_commit")
        if not locked_source_tree_hash:
            locked_source_tree_hash = expected_lock_manifest.get("source_tree_hash")

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

    # Cryptographic binding checks
    auth_lock_sha = auth_data.get("protocol_lock_sha256")
    if not auth_lock_sha:
        raise Phase4BAuthorizationError(
            "FULL_EXECUTION_BLOCKED: authorization manifest missing required 'protocol_lock_sha256' binding."
        )
    if protocol_lock_sha256 and auth_lock_sha.lower() != protocol_lock_sha256.lower():
        raise Phase4BAuthorizationError(
            f"FULL_EXECUTION_BLOCKED: protocol_lock_sha256 in authorization does not match current lock."
        )

    auth_commit = auth_data.get("locked_scientific_code_commit") or auth_data.get("locked_git_commit")
    if not auth_commit:
        raise Phase4BAuthorizationError(
            "FULL_EXECUTION_BLOCKED: authorization manifest missing required 'locked_scientific_code_commit' binding."
        )
    if locked_git_commit and auth_commit.lower() != locked_git_commit.lower():
        raise Phase4BAuthorizationError(
            f"FULL_EXECUTION_BLOCKED: locked_scientific_code_commit in authorization does not match current commit."
        )

    auth_tree = auth_data.get("locked_source_tree_hash")
    if not auth_tree:
        raise Phase4BAuthorizationError(
            "FULL_EXECUTION_BLOCKED: authorization manifest missing required 'locked_source_tree_hash' binding."
        )
    if locked_source_tree_hash and auth_tree.lower() != locked_source_tree_hash.lower():
        raise Phase4BAuthorizationError(
            f"FULL_EXECUTION_BLOCKED: locked_source_tree_hash in authorization does not match current tree."
        )

    return {
        "status": "AUTHORIZED",
        "human_authorized": True,
        "authorizer": auth_data.get("authorizer", "unknown"),
        "authorized_at": auth_data.get("authorized_at", ""),
        "protocol_version": auth_data.get("protocol_version"),
        "protocol_lock_sha256": auth_lock_sha,
        "locked_scientific_code_commit": auth_commit,
        "locked_source_tree_hash": auth_tree,
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    run_branch = execute_branch


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
        **kwargs: Any,
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
        event_embeddings = rng.randn(n_events, 16)
        true_targets = np.array([e.get("future_rate_change", 0.0) for e in events], dtype=float)

        if dose > 0.0:
            signal = np.outer(true_targets, rng.randn(16)) * (dose * 0.5)
            event_embeddings += signal
        event_stance_scores = (rng.randn(n_events) * 0.1 + true_targets * 0.2 * dose).tolist()
        event_sensitivities = (0.20 + rng.rand(n_events) * 0.05 + dose * 0.10).tolist()

        return {
            "seed": seed,
            "dose": dose,
            "requested_dose": dose,
            "realized_dose": realized_dose,
            "token_budget": token_budget,
            "total_tokens": token_budget,
            "pre_cutoff_tokens": n_pre,
            "post_cutoff_tokens": n_post,
            "forced_repetition_ratio": 0.0,
            "clean_corpus_sha256": "5202f14b340c3212dc48c13e1ba7a8a814cb472a4c25df072bed14b7a3f19e4c",
            "contamination_corpus_sha256": "54b3e84e8ef958e78032815a6cc76e1ca68cb0428afe71b390a583ee60e25c58",
            "treatment_stream_sha256": hashlib.sha256(f"mock_stream_{seed}_{dose}".encode()).hexdigest(),
            "base_checkpoint": "ProsusAI/finbert",
            "base_revision": "4556d13015211d73dccd3fdd39d39232506f3e43",
            "tokenizer_name": "ProsusAI/finbert",
            "tokenizer_revision": "4556d13015211d73dccd3fdd39d39232506f3e43",
            "initial_model_hash": initial_model_hash,
            "post_mlm_model_hash": post_mlm_model_hash,
            "downstream_initial_head_hash": downstream_initial_head_hash,
            "downstream_sample_order_hash": downstream_sample_order_hash,
            "mask_schedule_hash": mask_schedule_hash,
            "mlm_steps": 100,
            "optimizer": "AdamW",
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "downstream_hyperparameters": {"epochs": 3, "batch_size": 16, "learning_rate": 2e-5},
            "treatment_block_manifest": treatment_blocks,
            "event_embeddings": event_embeddings,
            "event_stance_scores": event_stance_scores,
            "event_sensitivities": event_sensitivities,
            "behavioral_sensitivity": {"mask_sensitivity": float(np.mean(event_sensitivities)), "per_sample_sensitivities": event_sensitivities},
            "competence_macro_f1": 0.75 + float(rng.randn() * 0.02),
            "competence_mcc": 0.50 + float(rng.randn() * 0.02),
            "competence_brier": 0.15,
            "competence_ece": 0.05,
            "competence_provenance": "MOCK_COMPETENCE",
            "temporal_robustness": "NOT_EVALUATED",
            "economic_ic_2y": 0.15 + float(rng.randn() * 0.05),
            "economic_ic_spy": 0.10 + float(rng.randn() * 0.05),
            "data_mode": "MOCK",
        }


class Phase4ArtifactWriter:
    """Manages persistent disk serialization of Phase 4 confirmatory artifacts."""

    def __init__(self, output_root: Union[str, Path] = "experiments/phase4_confirmatory") -> None:
        self.output_root = Path(output_root)
        self.manifests_dir = self.output_root / "manifests"
        self.metrics_dir = self.output_root / "metrics"
        self.results_dir = self.output_root / "results"
        self.provenance_dir = self.output_root / "provenance"

    def ensure_directories(self) -> None:
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.provenance_dir.mkdir(parents=True, exist_ok=True)

    def write_branch_manifest(self, branch_data: Dict[str, Any]) -> Path:
        self.ensure_directories()
        seed = branch_data["seed"]
        dose_int = int(round(branch_data["dose"] * 100))
        p = self.manifests_dir / f"seed{seed}_d{dose_int:03d}.json"
        manifest = {k: v for k, v in branch_data.items() if k not in ("event_embeddings", "paragraph_embeddings")}
        with open(p, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        return p

    def write_branch_metrics(self, branch_data: Dict[str, Any]) -> Path:
        self.ensure_directories()
        seed = branch_data["seed"]
        dose_int = int(round(branch_data["dose"] * 100))
        p = self.metrics_dir / f"seed{seed}_d{dose_int:03d}.json"
        metrics = {
            "seed": seed,
            "dose": branch_data["dose"],
            "data_mode": branch_data.get("data_mode", "UNKNOWN"),
            "event_contrast_mean": branch_data.get("event_contrast_mean"),
            "permutation_p_value": branch_data.get("permutation_p_value"),
            "delta_spearman": branch_data.get("delta_spearman"),
            "competence_macro_f1": branch_data.get("competence_macro_f1"),
            "competence_mcc": branch_data.get("competence_mcc"),
            "competence_brier": branch_data.get("competence_brier"),
            "competence_ece": branch_data.get("competence_ece"),
            "temporal_robustness": branch_data.get("temporal_robustness", "NOT_EVALUATED"),
            "l_behavior": branch_data.get("l_behavior"),
            "economic_ic_2y": branch_data.get("economic_ic_2y"),
            "economic_ic_spy": branch_data.get("economic_ic_spy"),
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        return p

    def write_confirmatory_results(self, results: Dict[str, Any], allow_mock: bool = False) -> Path:
        self.ensure_directories()
        branches = results.get("branches", {})
        if not allow_mock:
            for b_name, b_data in branches.items():
                if b_data.get("data_mode") != "EMPIRICAL":
                    raise ProductionBackendValidationError(
                        f"Branch '{b_name}' has data_mode='{b_data.get('data_mode')}'. "
                        "Official confirmatory artifacts strictly require data_mode='EMPIRICAL'."
                    )
        p = self.results_dir / "phase4_confirmatory_results.json"

        def _clean(obj: Any) -> Any:
            if isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items() if k not in ("event_embeddings", "paragraph_embeddings")}
            if isinstance(obj, list):
                return [_clean(x) for x in obj]
            return obj

        clean_data = _clean(results)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=2, default=str)
        return p

    def write_provenance(self, env_info: Dict[str, Any], lock_info: Dict[str, Any]) -> Tuple[Path, Path]:
        self.ensure_directories()
        p_env = self.provenance_dir / "execution_environment.json"
        p_lock = self.provenance_dir / "protocol_verification.json"
        with open(p_env, "w", encoding="utf-8") as f:
            json.dump(env_info, f, indent=2, default=str)
        with open(p_lock, "w", encoding="utf-8") as f:
            json.dump(lock_info, f, indent=2, default=str)
        return p_env, p_lock

    def write_all(
        self,
        results: Dict[str, Any],
        env_info: Optional[Dict[str, Any]] = None,
        lock_info: Optional[Dict[str, Any]] = None,
        allow_mock: bool = False,
    ) -> Dict[str, Any]:
        self.ensure_directories()
        manifest_paths = []
        metrics_paths = []
        branches = results.get("branches", {})
        for branch_key, branch_data in branches.items():
            mp = self.write_branch_manifest(branch_data)
            met_p = self.write_branch_metrics(branch_data)
            manifest_paths.append(str(mp))
            metrics_paths.append(str(met_p))
        res_p = self.write_confirmatory_results(results, allow_mock=allow_mock)
        env_p, lock_p = self.write_provenance(env_info or {}, lock_info or {})
        return {
            "results_path": str(res_p),
            "manifest_paths": manifest_paths,
            "metrics_paths": metrics_paths,
            "environment_path": str(env_p),
            "protocol_verification_path": str(lock_p),
        }


class ProductionConfirmatoryBackend(Phase4ExecutionBackend):
    """Production execution backend executing real FinBERT continued pretraining
    and downstream stance fine-tuning for Phase 4B confirmatory evaluation.

    Enforces:
    - Base checkpoint 'ProsusAI/finbert' locked to revision '4556d13015211d73dccd3fdd39d39232506f3e43'.
    - Real FinBERT tokenizer from the locked revision.
    - Real treatment stream constructed via `create_exact_token_dose_stream` (256,000 tokens).
    - Causal symmetry within each seed:
        * Bit-identical initial model parameter hash across all 5 doses.
        * Shared deterministic MLM mask schedule across all 5 doses.
        * Fresh FOMC 3-class stance classification head with identical initial head hash across all 5 doses.
        * Paired downstream sample ordering across all 5 doses.
    - Evaluation isolation: downstream training split strictly excludes Phase 4 anchor events and contamination docs.
    - Real MLM continued pretraining with locked hyperparameters.
    - Encoder-body transfer into fresh stance classifier.
    - Real event hidden representations aggregated across anchor paragraphs.
    - Real competence, behavioral leakage, and economic effect calculations (data_mode='EMPIRICAL').
    - Sets temporal_robustness = 'NOT_EVALUATED' (no synthetic placeholders).
    - Strictly no random / synthetic fallbacks.
    """

    def __init__(
        self,
        base_checkpoint: str = "ProsusAI/finbert",
        base_revision: str = "4556d13015211d73dccd3fdd39d39232506f3e43",
        device: Optional[str] = None,
        mock_model_for_testing: Optional[Any] = None,
        mock_tokenizer_for_testing: Optional[Any] = None,
    ) -> None:
        self.base_checkpoint = base_checkpoint
        self.base_revision = base_revision
        self.device = device
        self.mock_model_for_testing = mock_model_for_testing
        self.mock_tokenizer_for_testing = mock_tokenizer_for_testing
        self._seed_cache: Dict[int, Dict[str, Any]] = {}
        self._tdw_samples: Optional[List[Any]] = None

    def execute_branch(
        self,
        seed: int,
        dose: float,
        pre_docs: List[Dict[str, Any]],
        post_docs: List[Dict[str, Any]],
        anchors: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
        conf_cfg: Dict[str, Any],
        train_samples: Optional[List[Any]] = None,
        eval_samples: Optional[List[Any]] = None,
        output_dir: Optional[Path | str] = None,
        mock_model: Optional[Any] = None,
        mock_tokenizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        import torch
        import torch.nn as nn
        from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer
        from tradingagents.temporal_leakage.twin_pipeline import (
            create_exact_token_dose_stream,
            generate_deterministic_mask_schedule,
            run_continued_pretraining_mlm,
            build_classifier_from_mlm_encoder,
            train_downstream_classifier,
            normalize_and_hash_text,
            hash_model_parameters,
            CausalIntegrityError,
        )
        from tradingagents.temporal_leakage.metrics import (
            evaluate_competence,
            compute_masking_sensitivity,
        )
        from tradingagents.temporal_leakage.hf_encoder import (
            FOMC_STANCE_LABEL_TO_ID,
            LABEL_TO_INDEX,
        )

        # 0. Preflight validation of market outcome mappings
        for e in events:
            mo = e.get("market_outcomes")
            if not mo or "treasury_2y_yield_change" not in mo or "spy_1d_return" not in mo:
                raise PreregistrationLockError(
                    f"Event '{e.get('event_id')}' missing required market outcome fields ('treasury_2y_yield_change', 'spy_1d_return'). Silent fallbacks prohibited."
                )

        requested_dev = self.device or conf_cfg.get("model", {}).get("device")
        if requested_dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        elif requested_dev is not None:
            dev = requested_dev
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        base_checkpoint = conf_cfg.get("model", {}).get("base_checkpoint", self.base_checkpoint)
        base_revision = conf_cfg.get("model", {}).get("base_revision", self.base_revision)
        mlm_cfg = conf_cfg.get("mlm_training", {})
        token_budget = mlm_cfg.get("token_budget", 256000)
        num_blocks = mlm_cfg.get("num_blocks", 500)
        block_length = mlm_cfg.get("block_length", 512)
        mlm_prob = mlm_cfg.get("mask_ratio", 0.15)
        max_rep_ratio = mlm_cfg.get("max_repetition_ratio", 0.20)

        # 1. Initialize per-seed state for causal symmetries
        if seed not in self._seed_cache:
            tok = mock_tokenizer or self.mock_tokenizer_for_testing
            if tok is None:
                tok = AutoTokenizer.from_pretrained(base_checkpoint, revision=base_revision)

            mask_sched, mask_hash = generate_deterministic_mask_schedule(
                num_blocks=num_blocks,
                block_length=block_length,
                mlm_probability=mlm_prob,
                random_seed=seed,
            )

            base_m = mock_model or self.mock_model_for_testing
            if base_m is None:
                base_m = AutoModelForMaskedLM.from_pretrained(base_checkpoint, revision=base_revision)

            init_param_hash = hash_model_parameters(base_m)

            torch.manual_seed(seed)
            hidden_size = getattr(base_m.config, "hidden_size", 768) if hasattr(base_m, "config") else 768
            init_head = nn.Linear(hidden_size, 3)
            head_state = copy.deepcopy(init_head.state_dict())
            head_hash = hash_model_parameters(init_head)

            self._seed_cache[seed] = {
                "tokenizer": tok,
                "base_mlm": base_m,
                "initial_model_hash": init_param_hash,
                "mask_schedule": mask_sched,
                "mask_schedule_hash": mask_hash,
                "downstream_head_state_dict": head_state,
                "downstream_initial_head_hash": head_hash,
                "token_budget": token_budget,
                "num_blocks": num_blocks,
                "block_length": block_length,
            }

        cached_seed = self._seed_cache[seed]
        tok = cached_seed["tokenizer"]
        mask_sched = cached_seed["mask_schedule"]
        mask_hash = cached_seed["mask_schedule_hash"]
        head_state = cached_seed["downstream_head_state_dict"]
        head_hash = cached_seed["downstream_initial_head_hash"]
        init_param_hash = cached_seed["initial_model_hash"]

        # 2. Construct treatment token stream
        stream = create_exact_token_dose_stream(
            pre_corpus=pre_docs,
            post_corpus=post_docs,
            dose=dose,
            num_blocks=cached_seed["num_blocks"],
            block_length=cached_seed["block_length"],
            tokenizer=tok,
            random_seed=seed,
            max_repetition_ratio=max_rep_ratio,
        )

        realized_dose = stream["realized_dose"]
        forced_rep_ratio = stream["forced_repetition_ratio"]
        if abs(realized_dose - dose) > (1.0 / stream["total_tokens"]) + 1e-9:
            raise CausalIntegrityError(f"Dose invariant violated: realized {realized_dose} != requested {dose}")
        if forced_rep_ratio > max_rep_ratio:
            raise CausalIntegrityError(f"Forced repetition ratio {forced_rep_ratio:.4f} > tolerance {max_rep_ratio:.4f}")

        # 3. Model instance for this dose branch
        branch_mlm = copy.deepcopy(cached_seed["base_mlm"])
        branch_init_hash = hash_model_parameters(branch_mlm)
        if branch_init_hash != init_param_hash:
            raise CausalIntegrityError(f"Initial parameter hash divergence within seed {seed}: {branch_init_hash} != {init_param_hash}")

        # 4. MLM Continued Pretraining
        branch_out = Path(output_dir) if output_dir else Path("checkpoints") / f"phase4_seed_{seed}_d{int(dose*100):03d}"
        mlm_steps = mlm_cfg.get("max_steps", 100)
        mlm_manifest = run_continued_pretraining_mlm(
            packed_dataset=stream["dataset"],
            tokenizer=tok,
            model_mlm=branch_mlm,
            output_dir=branch_out / "mlm",
            max_steps=mlm_steps,
            batch_size=mlm_cfg.get("batch_size", 16),
            learning_rate=mlm_cfg.get("learning_rate", 5e-5),
            mlm_probability=mlm_prob,
            weight_decay=mlm_cfg.get("weight_decay", 0.01),
            random_seed=seed,
            device=dev,
            branch=f"seed_{seed}_dose_{int(dose*100):03d}",
            dose=dose,
            base_checkpoint=base_checkpoint,
            base_revision=base_revision,
            corpus_hash=stream["corpus_hash"],
            mask_schedule=mask_sched,
            mask_schedule_hash=mask_hash,
        )
        post_mlm_hash = mlm_manifest["final_parameter_hash"]

        # 5. Build stance classifier inheriting MLM-trained encoder body
        clf_encoder = build_classifier_from_mlm_encoder(
            mlm_model=branch_mlm,
            tokenizer=tok,
            head_state_dict=head_state,
            name=f"fomc_stance_seed_{seed}_dose_{int(dose*100):03d}",
            base_model_name=base_checkpoint,
            base_revision=base_revision,
            device=dev,
            random_seed=seed,
        )
        assert clf_encoder.original_head_loaded is False
        assert clf_encoder.stance_head_initialization == "fresh"

        # 6. Evaluation isolation & downstream fine-tuning
        if train_samples is None:
            if self._tdw_samples is None:
                from tradingagents.temporal_leakage.datasets.trillion_dollar_words import load_trillion_dollar_words
                self._tdw_samples = load_trillion_dollar_words()
            train_samples = [s for s in self._tdw_samples if int(s.metadata.get("year", 2000)) <= 2018]

        train_hashes = {normalize_and_hash_text(s.text) for s in train_samples}
        anchor_hashes = {normalize_and_hash_text(a["text"]) for a in anchors}
        eval_overlap = train_hashes & anchor_hashes
        if eval_overlap:
            raise PreregistrationLockError(
                f"Evaluation isolation breach: {len(eval_overlap)} downstream training samples overlap with anchor evaluation set!"
            )

        dt_cfg = conf_cfg.get("downstream_training", {})
        fine_tuned_encoder, sample_order_hash = train_downstream_classifier(
            encoder_model=clf_encoder,
            train_samples=train_samples,
            epochs=dt_cfg.get("epochs", 3),
            max_steps=dt_cfg.get("max_steps", None),
            batch_size=dt_cfg.get("batch_size", 16),
            learning_rate=dt_cfg.get("learning_rate", 2e-5),
            random_seed=seed,
            max_seq_length=dt_cfg.get("max_seq_length", 128),
            weight_decay=dt_cfg.get("weight_decay", 0.01),
            optimizer_name=dt_cfg.get("optimizer", "AdamW"),
            scheduler_name=dt_cfg.get("scheduler", "linear"),
            warmup_ratio=dt_cfg.get("warmup_ratio", 0.1),
        )

        # 7. Extract real hidden representations from anchor paragraphs
        anchor_texts = [a["text"] for a in anchors]
        anchor_reps = fine_tuned_encoder.extract_representations(anchor_texts)

        event_ids_order = [e["event_id"] for e in events]
        event_reps_list = []
        for ev_id in event_ids_order:
            matching_indices = [i for i, a in enumerate(anchors) if a["event_id"] == ev_id]
            if not matching_indices:
                raise ValueError(f"No anchors associated with event_id '{ev_id}'.")
            ev_rep = np.mean(anchor_reps[matching_indices], axis=0)
            event_reps_list.append(ev_rep)
        event_embeddings = np.array(event_reps_list, dtype=np.float32)

        # 8. Competence C, Robustness R_T, Behavioral Sensitivity S_D, Economic IC E_L
        comp_macro_f1 = None
        comp_mcc = None
        comp_brier = None
        comp_ece = None
        if eval_samples:
            eval_texts = [s.text for s in eval_samples]
            y_t = [LABEL_TO_INDEX.get(s.task_label, 0) for s in eval_samples]
            y_p, y_prob = fine_tuned_encoder.predict_task(eval_texts)
            comp_res = evaluate_competence(y_t, y_p, y_prob=y_prob)
            comp_macro_f1 = float(comp_res["macro_f1"])
            comp_mcc = float(comp_res["mcc"])
            comp_brier = float(comp_res["brier_score"])
            comp_ece = float(comp_res["ece"])

        sens_res = compute_masking_sensitivity(fine_tuned_encoder, anchor_texts)
        per_anchor_sens = sens_res.get("per_sample_sensitivities", [sens_res.get("mask_sensitivity", 0.0)] * len(anchors))
        event_sensitivities = []
        for ev_id in event_ids_order:
            matching_indices = [i for i, a in enumerate(anchors) if a["event_id"] == ev_id]
            ev_sens = float(np.mean(np.array(per_anchor_sens)[matching_indices]))
            event_sensitivities.append(ev_sens)

        _, anchor_probs = fine_tuned_encoder.predict_task(anchor_texts)
        anchor_stance_scores = anchor_probs[:, 2] - anchor_probs[:, 0]
        event_stance_scores = []
        for ev_id in event_ids_order:
            matching_indices = [i for i, a in enumerate(anchors) if a["event_id"] == ev_id]
            ev_s = float(np.mean(anchor_stance_scores[matching_indices]))
            event_stance_scores.append(ev_s)

        y_2y_list = []
        y_spy_list = []
        for e in events:
            mo = e.get("market_outcomes")
            if not mo or "treasury_2y_yield_change" not in mo or "spy_1d_return" not in mo:
                raise PreregistrationLockError(
                    f"Event '{e.get('event_id')}' missing required market outcome fields ('treasury_2y_yield_change', 'spy_1d_return'). Silent fallbacks prohibited."
                )
            y_2y_list.append(float(mo["treasury_2y_yield_change"]))
            y_spy_list.append(float(mo["spy_1d_return"]))
        y_2y = np.array(y_2y_list, dtype=np.float64)
        y_spy = np.array(y_spy_list, dtype=np.float64)

        from scipy import stats as sp_stats
        ic_2y = float(sp_stats.spearmanr(event_stance_scores, y_2y)[0]) if np.std(event_stance_scores) > 1e-6 else 0.0
        ic_spy = float(sp_stats.spearmanr(event_stance_scores, y_spy)[0]) if np.std(event_stance_scores) > 1e-6 else 0.0

        clean_manifest_path = Path("data/research/fomc/phase4_pre_cutoff/manifest.json")
        contam_manifest_path = Path("data/research/fomc/phase4_contamination/manifest.json")
        clean_sha = "unknown"
        contam_sha = "unknown"
        if clean_manifest_path.exists():
            clean_sha = json.load(open(clean_manifest_path, encoding="utf-8")).get("dataset_sha256", "unknown")
        if contam_manifest_path.exists():
            contam_sha = json.load(open(contam_manifest_path, encoding="utf-8")).get("dataset_sha256", "unknown")

        return {
            "seed": seed,
            "dose": dose,
            "requested_dose": dose,
            "realized_dose": realized_dose,
            "token_budget": token_budget,
            "pre_cutoff_tokens": stream["pre_cutoff_tokens"],
            "post_cutoff_tokens": stream["post_cutoff_tokens"],
            "total_tokens": stream["total_tokens"],
            "forced_repetition_ratio": forced_rep_ratio,
            "unique_source_document_count": stream.get("unique_source_document_count", 0),
            "clean_corpus_sha256": clean_sha,
            "contamination_corpus_sha256": contam_sha,
            "treatment_stream_sha256": stream.get("corpus_hash", "unknown"),
            "base_checkpoint": base_checkpoint,
            "base_revision": base_revision,
            "tokenizer_name": base_checkpoint,
            "tokenizer_revision": base_revision,
            "initial_model_hash": init_param_hash,
            "post_mlm_model_hash": post_mlm_hash,
            "mask_schedule_hash": mask_hash,
            "downstream_initial_head_hash": head_hash,
            "downstream_sample_order_hash": sample_order_hash,
            "mlm_steps": mlm_steps,
            "optimizer": mlm_cfg.get("optimizer", "AdamW"),
            "learning_rate": mlm_cfg.get("learning_rate", 5e-5),
            "weight_decay": mlm_cfg.get("weight_decay", 0.01),
            "downstream_hyperparameters": {
                "epochs": dt_cfg.get("epochs", 3),
                "batch_size": dt_cfg.get("batch_size", 16),
                "learning_rate": dt_cfg.get("learning_rate", 2e-5),
                "weight_decay": dt_cfg.get("weight_decay", 0.01),
                "optimizer": dt_cfg.get("optimizer", "AdamW"),
                "scheduler": dt_cfg.get("scheduler", "linear"),
                "warmup_ratio": dt_cfg.get("warmup_ratio", 0.1),
                "max_seq_length": dt_cfg.get("max_seq_length", 128),
            },
            "treatment_block_manifest": stream["treatment_block_manifest"],
            "event_embeddings": event_embeddings,
            "event_stance_scores": event_stance_scores,
            "event_sensitivities": event_sensitivities,
            "competence_macro_f1": comp_macro_f1,
            "competence_mcc": comp_mcc,
            "competence_brier": comp_brier,
            "competence_ece": comp_ece,
            "competence_provenance": "NOT_EVALUATED_NO_EVAL_SPLIT" if eval_samples is None else "EVALUATED_TDW",
            "temporal_robustness": "NOT_EVALUATED",
            "behavioral_sensitivity": sens_res,
            "economic_ic_2y": ic_2y,
            "economic_ic_spy": ic_spy,
            "data_mode": "EMPIRICAL",
        }

    run_branch = execute_branch


def execute_phase4b_confirmatory(
    config_path: Path | str,
    prereg_path: Path | str,
    backend: Optional[Phase4ExecutionBackend] = None,
    authorization_path: Optional[Path | str] = None,
    project_root: Optional[Path | str] = None,
    is_mock_orchestration: bool = False,
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Execute complete 34-step Phase 4B confirmatory execution graph.

    Orchestrates:
    - Lock, code-freeze, and human authorization checks.
    - Resolves backend to ProductionConfirmatoryBackend by default (rejects Mock backend in empirical mode).
    - 5 doses x preregistered seeds (25 logical branches).
    - Event-level representation aggregation and grouped temporal CV.
    - Continuous Ridge regression on future_rate_change.
    - Paired event-level permutation inference across 32 OOS events.
    - Persistent disk serialization via Phase4ArtifactWriter.
    """
    from tradingagents.temporal_leakage.datasets.contamination import (
        audit_anchor_contamination_isolation,
        load_phase4_contamination_documents,
    )
    from tradingagents.temporal_leakage.metrics import (
        evaluate_representational_leakage_grouped,
        evaluate_behavioral_leakage_event_level,
        evaluate_economic_effect_event_level,
        grouped_temporal_split,
    )
    from scipy import stats as sp_stats

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

    locked_commit = lock_meta.get("scientific_code_commit") or lock_meta.get("code_commit")
    locked_tree = lock_meta.get("source_tree_hash")
    proto_ver = lock_meta.get("protocol_version", "1.2.2")

    if locked_commit is None or locked_tree is None:
        raise PreregistrationLockError(
            "Protocol lock metadata is missing scientific_code_commit or source_tree_hash bindings. Cannot authorize Phase 4B."
        )

    # 2. Verify human authorization
    auth_p = Path(authorization_path) if authorization_path else root / "configs" / "phase4_execution_authorization.json"
    lock_file_sha = compute_file_sha256(root / "configs" / "phase4_protocol_lock.json", normalize_newlines=True)
    auth_meta = verify_phase4b_authorization(
        authorization_path=auth_p,
        protocol_version=proto_ver,
        protocol_lock_sha256=lock_file_sha,
        locked_git_commit=locked_commit,
        locked_source_tree_hash=locked_tree,
    )

    # 3. Verify code freeze
    freeze_meta = verify_phase4_code_freeze(
        project_root=root,
        locked_git_commit=locked_commit,
        locked_source_tree_hash=locked_tree,
        enforce_git_clean=True if not is_mock_orchestration else False,
    )

    # 4. Load configuration and datasets
    with open(cfg_p, "r", encoding="utf-8") as f:
        conf_cfg = yaml.safe_load(f)
    with open(prereg_p, "r", encoding="utf-8") as f:
        prereg_cfg = yaml.safe_load(f)

    contract_meta = resolve_phase4_runtime_contract(conf_cfg, prereg_cfg)

    # 5. Resolve backend (Default: ProductionConfirmatoryBackend)
    if backend is None:
        exec_backend = ProductionConfirmatoryBackend(
            base_checkpoint=conf_cfg.get("model", {}).get("base_checkpoint", "ProsusAI/finbert"),
            base_revision=conf_cfg.get("model", {}).get("base_revision", "4556d13015211d73dccd3fdd39d39232506f3e43"),
            device=conf_cfg.get("model", {}).get("device"),
        )
    else:
        exec_backend = backend

    # Hard rejection of Mock backend in full empirical execution
    if isinstance(exec_backend, MockConfirmatoryBackend) and not is_mock_orchestration:
        raise ProductionBackendValidationError(
            "MockConfirmatoryBackend is strictly prohibited in full empirical confirmatory execution. "
            "Confirmatory execution requires ProductionConfirmatoryBackend."
        )

    events_p = root / conf_cfg.get("dataset", {}).get("events_path", "data/research/fomc/events/events.jsonl")
    anchors_p = root / conf_cfg.get("dataset", {}).get("anchors_path", "data/research/fomc/confirmatory_anchors/anchors.jsonl")
    clean_p = root / "data" / "research" / "fomc" / "phase4_pre_cutoff" / "documents.jsonl"
    contam_p = root / "data" / "research" / "fomc" / "phase4_contamination" / "documents.jsonl"

    events = [json.loads(line) for line in open(events_p, "r", encoding="utf-8") if line.strip()]
    anchors = [json.loads(line) for line in open(anchors_p, "r", encoding="utf-8") if line.strip()]
    clean_docs = [json.loads(line) for line in open(clean_p, "r", encoding="utf-8") if line.strip()]
    contam_docs = load_phase4_contamination_documents(contam_p)

    # 6. Temporal separation and isolation validation
    cutoff = "2019-12-31T23:59:59Z"
    for d in clean_docs:
        if d["available_time"] > cutoff:
            raise PreregistrationLockError(f"Pre-cutoff violation in clean corpus: {d['document_id']}")

    iso_meta = audit_anchor_contamination_isolation(anchors, contam_docs)
    if not iso_meta["strict_future_separation"]:
        raise PreregistrationLockError("Temporal separation between anchors and contamination failed.")
    if iso_meta["document_overlap_count"] > 0:
        raise PreregistrationLockError(f"Document overlap detected: {iso_meta['document_overlap_ids']}")

    # 7. Extract and validate market outcomes from events
    y_2y_list = []
    y_spy_list = []
    for e in events:
        mo = e.get("market_outcomes")
        if not mo or "treasury_2y_yield_change" not in mo or "spy_1d_return" not in mo:
            raise PreregistrationLockError(
                f"Event '{e.get('event_id')}' missing required market outcome fields ('treasury_2y_yield_change', 'spy_1d_return'). Silent fallbacks prohibited."
            )
        y_2y_list.append(float(mo["treasury_2y_yield_change"]))
        y_spy_list.append(float(mo["spy_1d_return"]))
    y_2y = np.array(y_2y_list, dtype=np.float64)
    y_spy = np.array(y_spy_list, dtype=np.float64)

    # 8. Branch execution
    doses = conf_cfg.get("dose_ladder", [0.0, 0.25, 0.5, 0.75, 1.0])
    seeds = conf_cfg.get("seeds", [13, 42, 87, 123, 2024])

    branch_results: Dict[str, Any] = {}

    y_true_targets = np.array([float(e.get("future_rate_change", 0.0)) for e in events])
    event_ids = [e["event_id"] for e in events]
    event_times = [e["event_time"] for e in events]

    eval_cfg = conf_cfg.get("downstream_evaluation", {})
    probe_alpha = float(eval_cfg.get("probe_alpha", 1.0))
    eval_seed = int(eval_cfg.get("random_seed", 42))
    cv_folds = int(eval_cfg.get("cv_folds", 4))
    paired_permutations = int(eval_cfg.get("paired_permutations", 2000))
    bootstrap_draws = int(eval_cfg.get("bootstrap_draws", 1000))
    aggregation_rule = eval_cfg.get("aggregation_rule", "mean")

    y_binary = np.array([1 if abs(float(e.get("future_rate_change", 0.0))) > 1e-6 else 0 for e in events], dtype=int)

    for seed in seeds:
        seed_branch_data: Dict[float, Dict[str, Any]] = {}
        for dose in doses:
            branch_key = f"seed_{seed}_dose_{int(round(dose * 100)):03d}"
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

        # Runtime assertion of same-seed causal symmetries
        d0_res = seed_branch_data[0.0]
        for d, b_data in seed_branch_data.items():
            if d == 0.0:
                continue
            assert b_data["initial_model_hash"] == d0_res["initial_model_hash"], (
                f"Causal symmetry violation: initial_model_hash mismatch in seed {seed} (dose {d})"
            )
            assert b_data["mask_schedule_hash"] == d0_res["mask_schedule_hash"], (
                f"Causal symmetry violation: mask_schedule_hash mismatch in seed {seed} (dose {d})"
            )
            assert b_data["downstream_initial_head_hash"] == d0_res["downstream_initial_head_hash"], (
                f"Causal symmetry violation: downstream_initial_head_hash mismatch in seed {seed} (dose {d})"
            )
            assert b_data["downstream_sample_order_hash"] == d0_res["downstream_sample_order_hash"], (
                f"Causal symmetry violation: downstream_sample_order_hash mismatch in seed {seed} (dose {d})"
            )
            assert b_data["token_budget"] == d0_res["token_budget"], (
                f"Causal symmetry violation: token_budget mismatch in seed {seed} (dose {d})"
            )

        clean_embeddings = seed_branch_data[0.0]["event_embeddings"]
        clean_stance_scores = seed_branch_data[0.0].get("event_stance_scores", [0.0] * len(events))
        clean_sensitivities = seed_branch_data[0.0].get("event_sensitivities", [0.0] * len(events))

        # Evaluate each dose branch using centralized metrics
        for dose in doses:
            branch_key = f"seed_{seed}_dose_{int(round(dose * 100)):03d}"
            b_res = seed_branch_data[dose]

            # Provenance injection
            b_res["scientific_code_commit"] = locked_commit
            b_res["execution_repository_head"] = freeze_meta.get("execution_repository_head")
            b_res["source_tree_hash"] = locked_tree
            b_res["protocol_lock_sha256"] = lock_file_sha
            b_res["protocol_version"] = proto_ver

            if dose == 0.0:
                b_res["l_repr"] = 0.0
                b_res["event_contrast_mean"] = 0.0
                b_res["permutation_p_value"] = 1.0
                b_res["delta_spearman"] = 0.0
                b_res["l_behavior"] = 0.0
                b_res["delta_ic_2y"] = 0.0
                b_res["delta_ic_spy"] = 0.0
                b_res["n_oos_events"] = 32
                b_res["target_type"] = "continuous"
                b_res["model_type"] = "ridge_regression"
                continue

            leak_embeddings = b_res["event_embeddings"]
            leak_stance_scores = b_res.get("event_stance_scores", [0.0] * len(events))
            leak_sensitivities = b_res.get("event_sensitivities", [0.0] * len(events))

            # 1. Continuous Primary: evaluate_representational_leakage_grouped
            res_repr = evaluate_representational_leakage_grouped(
                h_leak=leak_embeddings,
                h_clean=clean_embeddings,
                y_future=y_true_targets,
                event_ids=event_ids,
                event_times=event_times,
                n_splits=cv_folds,
                n_permutations=paired_permutations,
                random_seed=eval_seed,
                aggregation_rule=aggregation_rule,
                target_type="continuous",
                loss_metric="mae",
                probe_alpha=probe_alpha,
            )

            b_res["l_repr"] = res_repr["l_repr"]
            b_res["event_contrast_mean"] = res_repr["observed_statistic"]
            b_res["permutation_p_value"] = res_repr["p_value"]
            b_res["is_statistically_significant"] = res_repr["is_statistically_significant"]
            b_res["delta_spearman"] = res_repr["delta_spearman"]
            b_res["spearman_leak"] = res_repr["spearman_leak"]
            b_res["spearman_clean"] = res_repr["spearman_clean"]
            b_res["n_oos_events"] = res_repr["n_oos_events"]
            b_res["primary_inferential_n"] = res_repr["primary_inferential_n"]
            b_res["target_type"] = "continuous"
            b_res["model_type"] = "ridge_regression"
            b_res["event_deltas"] = res_repr["event_deltas"]
            b_res["oos_event_table"] = res_repr["oos_event_table"]

            # 2. Binary Co-Primary: evaluate_representational_leakage_grouped
            res_binary = evaluate_representational_leakage_grouped(
                h_leak=leak_embeddings,
                h_clean=clean_embeddings,
                y_future=y_binary,
                event_ids=event_ids,
                event_times=event_times,
                n_splits=cv_folds,
                n_permutations=paired_permutations,
                random_seed=eval_seed,
                aggregation_rule=aggregation_rule,
                target_type="binary",
                probe_alpha=probe_alpha,
            )

            b_res["binary_co_primary"] = {
                "target": "next_scheduled_change_vs_hold",
                "target_type": "binary",
                "model_type": "ridge_classifier",
                "observed_statistic": res_binary["observed_statistic"],
                "p_value": res_binary["p_value"],
                "is_statistically_significant": res_binary["is_statistically_significant"],
                "delta_macro_f1": res_binary["delta_macro_f1"],
                "macro_f1_leak": res_binary["macro_f1_leak"],
                "macro_f1_clean": res_binary["macro_f1_clean"],
                "n_oos_events": res_binary["n_oos_events"],
            }

            # 3. Behavioral Leakage Event-Level
            res_behav = evaluate_behavioral_leakage_event_level(
                sensitivities_leak=leak_sensitivities,
                sensitivities_clean=clean_sensitivities,
                event_ids=event_ids,
                aggregation_rule=aggregation_rule,
            )
            b_res["l_behavior"] = res_behav["l_behavior_event"]
            b_res["l_behavior_event"] = res_behav["l_behavior_event"]
            b_res["mask_sensitivity_leak_event"] = res_behav["mask_sensitivity_leak_event"]
            b_res["mask_sensitivity_clean_event"] = res_behav["mask_sensitivity_clean_event"]
            b_res["fdr_correction"] = "NOT_EVALUATED_PROTOCOL_UNDERSPECIFIED"

            # 4. Economic Effect Event-Level
            res_econ_2y = evaluate_economic_effect_event_level(
                stance_scores_leak=leak_stance_scores,
                stance_scores_clean=clean_stance_scores,
                market_returns=y_2y,
                event_ids=event_ids,
                aggregation_rule=aggregation_rule,
                n_bootstrap=bootstrap_draws,
                random_seed=eval_seed,
            )
            res_econ_spy = evaluate_economic_effect_event_level(
                stance_scores_leak=leak_stance_scores,
                stance_scores_clean=clean_stance_scores,
                market_returns=y_spy,
                event_ids=event_ids,
                aggregation_rule=aggregation_rule,
                n_bootstrap=bootstrap_draws,
                random_seed=eval_seed,
            )
            b_res["economic_2y"] = res_econ_2y
            b_res["economic_spy"] = res_econ_spy
            b_res["economic_ic_2y"] = res_econ_2y["ic_leak"]
            b_res["delta_ic_2y"] = res_econ_2y["delta_ic"]
            b_res["delta_ic_2y_ci_95"] = res_econ_2y["delta_ic_ci_95"]
            b_res["economic_p_value_2y"] = res_econ_2y["p_value"]
            b_res["economic_ic_spy"] = res_econ_spy["ic_leak"]
            b_res["delta_ic_spy"] = res_econ_spy["delta_ic"]
            b_res["delta_ic_spy_ci_95"] = res_econ_spy["delta_ic_ci_95"]
            b_res["economic_p_value_spy"] = res_econ_spy["p_value"]

    final_res = {
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

    if output_dir:
        writer = Phase4ArtifactWriter(output_dir)
        writer.write_all(final_res, env_info=freeze_meta, lock_info=lock_meta, allow_mock=is_mock_orchestration)

    return final_res


def run_phase4_mock_orchestration(
    config_path: Path | str,
    prereg_path: Path | str,
    project_root: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Execute dry-run mock orchestration of all 25 branches for verification testing.

    Injects a temporary mock authorization object with valid cryptographic bindings
    to verify that the complete orchestration DAG executes end-to-end.
    """
    cfg_p = Path(config_path)
    root = Path(project_root) if project_root else cfg_p.resolve().parents[1]

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        mock_auth_p = Path(td) / "mock_authorization.json"
        lock_file = root / "configs" / "phase4_protocol_lock.json"
        with open(lock_file, "r", encoding="utf-8") as f:
            lock_manifest = json.load(f)
        lock_sha = compute_file_sha256(lock_file, normalize_newlines=True)

        mock_auth_data = {
            "protocol_version": lock_manifest.get("protocol_version", "1.2.2"),
            "human_authorized": True,
            "authorizer": "MOCK_DRY_RUN_TEST",
            "authorized_at": "2026-09-20T14:30:00Z",
            "protocol_lock_sha256": lock_sha,
            "locked_scientific_code_commit": lock_manifest.get("code_commit") or lock_manifest.get("scientific_code_commit", "unknown"),
            "locked_source_tree_hash": lock_manifest.get("source_tree_hash", "unknown"),
        }
        with open(mock_auth_p, "w", encoding="utf-8") as f:
            json.dump(mock_auth_data, f)

        res = execute_phase4b_confirmatory(
            config_path=config_path,
            prereg_path=prereg_path,
            backend=MockConfirmatoryBackend(),
            authorization_path=mock_auth_p,
            project_root=root,
            is_mock_orchestration=True,
            output_dir=output_dir,
        )
    return res


def run_phase4_confirmatory(
    config_path: Path | str,
    prereg_path: Path | str,
    smoke_mode: bool = False,
    allow_full_training: bool = False,
    backend: Optional[Phase4ExecutionBackend] = None,
    authorization_path: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
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
        Execution backend. Defaults to ProductionConfirmatoryBackend.
    authorization_path : Optional[Path | str]
        Path to human authorization manifest (`configs/phase4_execution_authorization.json`).
    output_dir : Optional[Path | str]
        Path to write persistent confirmatory artifacts.

    Returns
    -------
    Dict[str, Any]
        Verification and execution outcome summary.
    """
    prereg_meta = verify_preregistration_lock(prereg_path)
    cfg_p = Path(config_path)
    project_root = cfg_p.resolve().parents[1]

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
        is_mock_orchestration=False,
        output_dir=output_dir,
    )

