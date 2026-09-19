"""MANTRA Temporal Leakage & Point-in-Time Evaluation Framework."""

from .temporal_model import (
    TemporalModel,
    TemporalSample,
    NullConstantModel,
    SyntheticTemporalTwinEncoder,
    MockTwinEncoder,
    parse_iso_utc,
)
from .metrics import (
    stationary_block_bootstrap_indices,
    evaluate_competence,
    compute_masking_sensitivity,
    evaluate_behavioral_leakage,
    evaluate_representational_leakage,
    evaluate_economic_effect,
    evaluate_temporal_robustness,
    pareto_coordinates,
)
from .fomc_benchmark import (
    create_toy_fomc_dataset,
    load_fomc_dataset,
    FOMCBenchmark,
    ToyFOMCBenchmark,
    DatasetValidationError,
    validate_temporal_sample,
    validate_dataset,
)
from .twin_experiment import TwinExperimentRunner

__all__ = [
    "TemporalModel",
    "TemporalSample",
    "NullConstantModel",
    "SyntheticTemporalTwinEncoder",
    "MockTwinEncoder",
    "parse_iso_utc",
    "stationary_block_bootstrap_indices",
    "evaluate_competence",
    "compute_masking_sensitivity",
    "evaluate_behavioral_leakage",
    "evaluate_representational_leakage",
    "evaluate_economic_effect",
    "evaluate_temporal_robustness",
    "pareto_coordinates",
    "create_toy_fomc_dataset",
    "load_fomc_dataset",
    "FOMCBenchmark",
    "ToyFOMCBenchmark",
    "DatasetValidationError",
    "validate_temporal_sample",
    "validate_dataset",
    "TwinExperimentRunner",
]
