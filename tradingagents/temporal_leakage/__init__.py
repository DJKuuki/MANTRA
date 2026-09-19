"""MANTRA Temporal Leakage & Point-in-Time Evaluation Framework."""

from .temporal_model import (
    TemporalModel,
    TemporalSample,
    NullConstantModel,
    SyntheticTemporalTwinEncoder,
    MockTwinEncoder,
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
)
from .twin_experiment import TwinExperimentRunner

__all__ = [
    "TemporalModel",
    "TemporalSample",
    "NullConstantModel",
    "SyntheticTemporalTwinEncoder",
    "MockTwinEncoder",
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
    "TwinExperimentRunner",
]
