"""Trillion Dollar Words Dataset Adapter (Shah et al., ACL 2023).

Loads the canonical research dataset of sentence-level monetary policy communications.
Adheres strictly to MANTRA Point-in-Time safety protocols:
- `source_verified = True`
- `annotation_verified = True`
- `pit_verified = False` (because the upstream corpus provides 'year' but no intraday publication timestamps)
- `availability_quality = "unknown"`
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from ..fomc_benchmark import FOMCBenchmark, TemporalSample, load_fomc_dataset


DEFAULT_DATASET_PATH = Path("data/research/fomc/fomc_temporal_dataset.jsonl")


def load_trillion_dollar_words(
    filepath: Optional[Union[str, Path]] = None,
    auto_download: bool = True,
) -> List[TemporalSample]:
    """Load Trillion Dollar Words dataset as a list of TemporalSample instances.

    Args:
        filepath: Path to canonical JSONL dataset file. Defaults to `data/research/fomc/fomc_temporal_dataset.jsonl`.
        auto_download: If True and the canonical file is not found, automatically runs the ingestion downloader.

    Returns:
        List of strictly validated TemporalSample instances.
    """
    path = Path(filepath) if filepath is not None else DEFAULT_DATASET_PATH

    if not path.exists():
        if auto_download:
            from scripts.download_trillion_dollar_words import build_canonical_fomc_dataset
            build_canonical_fomc_dataset(path.parent)
        else:
            raise FileNotFoundError(
                f"Trillion Dollar Words dataset not found at {path}. "
                "Run `python scripts/download_trillion_dollar_words.py` or set auto_download=True."
            )

    samples = load_fomc_dataset(path)
    for s in samples:
        s.metadata.setdefault("temporal_resolution", "year")
        s.metadata.setdefault("timestamp_imputed", True)
        s.metadata.setdefault("timestamp_imputation_rule", "mid_year_placeholder")
    return samples


def create_trillion_dollar_words_benchmark(
    filepath: Optional[Union[str, Path]] = None,
    auto_download: bool = True,
) -> FOMCBenchmark:
    """Instantiate an FOMCBenchmark using the Trillion Dollar Words dataset.

    Note on formal research readiness:
    `is_formal_research_ready()` will evaluate to False because `pit_verified=False`
    and `availability_quality='unknown'`. This correctly prevents silent look-ahead
    evaluation under formal zero-tolerance PIT settings while permitting baseline
    model training and development smoke tests.
    """
    samples = load_trillion_dollar_words(filepath=filepath, auto_download=auto_download)
    return FOMCBenchmark(
        samples=samples,
        source_verified=True,
        annotation_verified=True,
        pit_verified=False,
    )
