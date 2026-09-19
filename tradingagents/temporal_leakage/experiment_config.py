"""Runtime configuration contract and validation for empirical temporal leakage experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union
import yaml

from .temporal_model import parse_iso_utc
from .fomc_benchmark import FOMCBenchmark, ToyFOMCBenchmark, DatasetValidationError


def load_experiment_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load and strictly validate an experiment YAML configuration file.

    Raises:
        FileNotFoundError: If the file does not exist.
        DatasetValidationError: If the configuration structure or values violate protocol.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Experiment configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise DatasetValidationError(f"Failed to parse YAML configuration at {path}: {e}") from e

    if not isinstance(config, dict):
        raise DatasetValidationError(f"Configuration root must be a YAML mapping/dict, got {type(config).__name__}")

    validate_experiment_config(config)
    return config


def validate_experiment_config(config: Dict[str, Any]) -> None:
    """Validate that an experiment configuration satisfies all schema and protocol requirements.

    Required top-level sections:
    - experiment_name: non-empty string
    - random_seed: int
    - time: mapping with source_timezone and internal_timezone
    - splits: mapping with train_end, dev_start, dev_end, test_start (valid ISO-8601 strings in chronological order)
    - probe: mapping with cv, n_splits, alpha
    - bootstrap: mapping with method, expected_block_length, n_bootstrap, confidence_level
    - economics: mapping with primary_metric, threshold, transaction_cost_bps, annualization_factor
    - pit: mapping with require_timezone_aware, reject_missing_availability, allowed_availability_qualities
    """
    required_sections = ["experiment_name", "random_seed", "time", "splits", "probe", "bootstrap", "economics", "pit"]
    for sec in required_sections:
        if sec not in config:
            raise DatasetValidationError(f"Experiment config missing required section: '{sec}'")

    if not isinstance(config["experiment_name"], str) or not config["experiment_name"].strip():
        raise DatasetValidationError("Config 'experiment_name' must be a non-empty string.")

    if not isinstance(config["random_seed"], int) or isinstance(config["random_seed"], bool):
        raise DatasetValidationError("Config 'random_seed' must be an integer.")

    # Time section
    time_cfg = config["time"]
    if not isinstance(time_cfg, dict):
        raise DatasetValidationError("Config 'time' must be a mapping.")
    if "source_timezone" not in time_cfg or "internal_timezone" not in time_cfg:
        raise DatasetValidationError("Config 'time' must specify 'source_timezone' and 'internal_timezone'.")
    if not isinstance(time_cfg["source_timezone"], str) or not time_cfg["source_timezone"].strip():
        raise DatasetValidationError("Config 'time.source_timezone' must be a non-empty string.")
    if not isinstance(time_cfg["internal_timezone"], str) or not time_cfg["internal_timezone"].strip():
        raise DatasetValidationError("Config 'time.internal_timezone' must be a non-empty string.")

    # Splits section
    splits_cfg = config["splits"]
    if not isinstance(splits_cfg, dict):
        raise DatasetValidationError("Config 'splits' must be a mapping.")
    for boundary in ["train_end", "dev_start", "dev_end", "test_start"]:
        if boundary not in splits_cfg:
            raise DatasetValidationError(f"Config 'splits' missing required partition boundary: '{boundary}'")

    try:
        t_train_end = parse_iso_utc(str(splits_cfg["train_end"]))
        t_dev_start = parse_iso_utc(str(splits_cfg["dev_start"]))
        t_dev_end = parse_iso_utc(str(splits_cfg["dev_end"]))
        t_test_start = parse_iso_utc(str(splits_cfg["test_start"]))
    except Exception as e:
        raise DatasetValidationError(f"Config 'splits' contains invalid ISO-8601 boundary: {e}") from e

    if not (t_train_end < t_dev_start <= t_dev_end < t_test_start):
        raise DatasetValidationError(
            f"Config 'splits' boundaries violate chronological order: "
            f"train_end ({t_train_end}) < dev_start ({t_dev_start}) <= dev_end ({t_dev_end}) < test_start ({t_test_start})"
        )

    # Probe section
    probe_cfg = config["probe"]
    if not isinstance(probe_cfg, dict):
        raise DatasetValidationError("Config 'probe' must be a mapping.")
    for req_k in ["cv", "n_splits", "n_permutations", "alpha"]:
        if req_k not in probe_cfg:
            raise DatasetValidationError(f"Config 'probe' missing required field '{req_k}'.")
    if probe_cfg["cv"] != "timeseries":
        raise DatasetValidationError(f"Config 'probe.cv' must be 'timeseries', got {probe_cfg['cv']!r}.")
    if not isinstance(probe_cfg["n_splits"], int) or isinstance(probe_cfg["n_splits"], bool) or probe_cfg["n_splits"] < 2:
        raise DatasetValidationError(f"Config 'probe.n_splits' must be an integer >= 2, got {probe_cfg['n_splits']!r}.")
    if not isinstance(probe_cfg["n_permutations"], int) or isinstance(probe_cfg["n_permutations"], bool) or probe_cfg["n_permutations"] <= 0:
        raise DatasetValidationError(f"Config 'probe.n_permutations' must be an integer > 0, got {probe_cfg['n_permutations']!r}.")
    if not isinstance(probe_cfg["alpha"], (int, float)) or isinstance(probe_cfg["alpha"], bool) or probe_cfg["alpha"] <= 0:
        raise DatasetValidationError(f"Config 'probe.alpha' must be a positive number > 0, got {probe_cfg['alpha']!r}.")

    # Bootstrap section
    boot_cfg = config["bootstrap"]
    if not isinstance(boot_cfg, dict):
        raise DatasetValidationError("Config 'bootstrap' must be a mapping.")
    for req_k in ["method", "expected_block_length", "n_bootstrap", "confidence_level"]:
        if req_k not in boot_cfg:
            raise DatasetValidationError(f"Config 'bootstrap' missing required field '{req_k}'.")
    if boot_cfg["method"] != "stationary":
        raise DatasetValidationError(f"Config 'bootstrap.method' must be 'stationary', got {boot_cfg['method']!r}.")
    if not isinstance(boot_cfg["expected_block_length"], (int, float)) or isinstance(boot_cfg["expected_block_length"], bool) or boot_cfg["expected_block_length"] <= 0:
        raise DatasetValidationError(f"Config 'bootstrap.expected_block_length' must be > 0, got {boot_cfg['expected_block_length']!r}.")
    if not isinstance(boot_cfg["n_bootstrap"], int) or isinstance(boot_cfg["n_bootstrap"], bool) or boot_cfg["n_bootstrap"] <= 0:
        raise DatasetValidationError(f"Config 'bootstrap.n_bootstrap' must be an integer > 0, got {boot_cfg['n_bootstrap']!r}.")
    if not isinstance(boot_cfg["confidence_level"], (int, float)) or isinstance(boot_cfg["confidence_level"], bool) or not (0 < boot_cfg["confidence_level"] < 1):
        raise DatasetValidationError(f"Config 'bootstrap.confidence_level' must be between 0 and 1, got {boot_cfg['confidence_level']!r}.")

    # Economics section
    econ_cfg = config["economics"]
    if not isinstance(econ_cfg, dict):
        raise DatasetValidationError("Config 'economics' must be a mapping.")
    for req_k in ["primary_metric", "threshold", "transaction_cost_bps", "annualization_factor"]:
        if req_k not in econ_cfg:
            raise DatasetValidationError(f"Config 'economics' missing required field '{req_k}'.")
    if econ_cfg["primary_metric"] != "delta_ic":
        raise DatasetValidationError(f"Config 'economics.primary_metric' must be 'delta_ic', got {econ_cfg['primary_metric']!r}.")
    if not isinstance(econ_cfg["threshold"], (int, float)) or isinstance(econ_cfg["threshold"], bool) or econ_cfg["threshold"] < 0:
        raise DatasetValidationError(f"Config 'economics.threshold' must be >= 0, got {econ_cfg['threshold']!r}.")
    if not isinstance(econ_cfg["transaction_cost_bps"], (int, float)) or isinstance(econ_cfg["transaction_cost_bps"], bool) or econ_cfg["transaction_cost_bps"] < 0:
        raise DatasetValidationError(f"Config 'economics.transaction_cost_bps' must be >= 0, got {econ_cfg['transaction_cost_bps']!r}.")
    if not isinstance(econ_cfg["annualization_factor"], (int, float)) or isinstance(econ_cfg["annualization_factor"], bool) or econ_cfg["annualization_factor"] <= 0:
        raise DatasetValidationError(f"Config 'economics.annualization_factor' must be > 0, got {econ_cfg['annualization_factor']!r}.")

    # PIT section
    pit_cfg = config["pit"]
    if not isinstance(pit_cfg, dict):
        raise DatasetValidationError("Config 'pit' must be a mapping.")
    if "allowed_availability_qualities" not in pit_cfg:
        raise DatasetValidationError("Config 'pit' missing required key 'allowed_availability_qualities'.")

    allowed_qualities = pit_cfg["allowed_availability_qualities"]
    if not isinstance(allowed_qualities, list) or not allowed_qualities:
        raise DatasetValidationError("Config 'pit.allowed_availability_qualities' must be a non-empty list.")

    valid_qualities = {"exact", "heuristic", "unknown"}
    for q in allowed_qualities:
        if q not in valid_qualities:
            raise DatasetValidationError(
                f"Invalid quality '{q}' in config 'pit.allowed_availability_qualities'. Valid: {valid_qualities}"
            )

    # Formal experiment protocol constraints
    if config["experiment_name"] == "fomc_formal_experiment":
        if time_cfg.get("internal_timezone") != "UTC":
            raise DatasetValidationError(
                f"Formal experiment configuration requires 'time.internal_timezone: UTC', got {time_cfg.get('internal_timezone')!r}."
            )
        if probe_cfg.get("n_permutations", 0) < 500:
            raise DatasetValidationError(
                f"Formal experiment requires probe.n_permutations >= 500, got {probe_cfg.get('n_permutations')}."
            )
        if boot_cfg.get("n_bootstrap", 0) < 1000:
            raise DatasetValidationError(
                f"Formal experiment requires bootstrap.n_bootstrap >= 1000, got {boot_cfg.get('n_bootstrap')}."
            )
        if pit_cfg.get("allow_heuristic_fallback", True) is not False:
            raise DatasetValidationError(
                "Formal experiment configuration requires 'pit.allow_heuristic_fallback' to be false."
            )
        if allowed_qualities != ["exact"]:
            raise DatasetValidationError(
                f"Formal experiment configuration strictly requires 'pit.allowed_availability_qualities: [exact]', "
                f"got: {allowed_qualities}"
            )
        if not pit_cfg.get("require_timezone_aware", False):
            raise DatasetValidationError(
                "Formal experiment configuration requires 'pit.require_timezone_aware: true'."
            )
        if not pit_cfg.get("reject_missing_availability", False):
            raise DatasetValidationError(
                "Formal experiment configuration requires 'pit.reject_missing_availability: true'."
            )


def validate_benchmark_against_config(benchmark: FOMCBenchmark, config: Dict[str, Any]) -> None:
    """Verify that a loaded FOMCBenchmark complies with the runtime constraints of an experiment config.

    Enforces:
    - Formal experiment config ('fomc_formal_experiment') rejects ToyFOMCBenchmark.
    - Formal experiment config requires benchmark.is_formal_research_ready() is True.
    - All benchmark samples must have availability_quality in config['pit']['allowed_availability_qualities'].
    """
    if not isinstance(benchmark, FOMCBenchmark):
        raise TypeError(f"Expected FOMCBenchmark instance, got {type(benchmark).__name__}")

    validate_experiment_config(config)

    is_formal = config.get("experiment_name") == "fomc_formal_experiment"
    pit_cfg = config.get("pit", {})
    allowed_qualities = set(pit_cfg.get("allowed_availability_qualities", ["exact"]))

    if is_formal and isinstance(benchmark, ToyFOMCBenchmark):
        raise DatasetValidationError(
            "ToyFOMCBenchmark cannot be used for formal experiment 'fomc_formal_experiment'. "
            "Formal experiments require verified external research datasets via FOMCBenchmark.from_file(...)."
        )

    if is_formal and not benchmark.is_formal_research_ready():
        raise DatasetValidationError(
            "Benchmark fails formal research readiness check (is_formal_research_ready() is False). "
            f"source_verified={benchmark.source_verified}, "
            f"annotation_verified={benchmark.annotation_verified}, "
            f"pit_verified={benchmark.pit_verified}. "
            "All must be True and all samples must have availability_quality='exact'."
        )

    for idx, s in enumerate(benchmark.samples):
        if s.availability_quality not in allowed_qualities:
            raise DatasetValidationError(
                f"Sample '{s.sample_id}' (index {idx}) has availability_quality '{s.availability_quality}', "
                f"which is disallowed by experiment config. Allowed qualities: {sorted(allowed_qualities)}."
            )
