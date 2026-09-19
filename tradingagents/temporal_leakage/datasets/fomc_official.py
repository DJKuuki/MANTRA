"""Official Federal Reserve FOMC Statements Adapter.

Handles verified official Federal Reserve statements with exact publication timestamps
(14:00 America/New_York on scheduled meeting release dates, converted to UTC).

IMPORTANT METHODOLOGICAL RULES (Phase 2.1):
1. Hardcoded in-code statement dictionaries are strictly TEST / DOCUMENTATION FIXTURES ONLY.
   They are demoted to `create_fomc_official_fixture()`.
   By default:
   - `annotation_verified = False`
   - `market_outcomes_verified = False`
   - `future_labels_verified = False`
   - `pit_verified = False`
   - `is_formal_research_ready() == False`
2. Formal research benchmarks via `create_fomc_official_benchmark()` require an explicit
   verified filepath or manifest. Invoking it without a verified file raises ValueError.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Union

from ..fomc_benchmark import FOMCBenchmark, TemporalSample, load_fomc_dataset, validate_dataset


def create_fomc_official_fixture() -> FOMCBenchmark:
    """Instantiate an FOMCBenchmark using hardcoded fixture examples for tests/docs only.

    NOTE:
    This fixture contains unverified illustrative macro/market outcomes and placeholder labels.
    It is explicitly marked as NOT formal research ready:
    `fixture.is_formal_research_ready() is False`.
    """
    raw_statements = [
        {
            "sample_id": "fed-official-fixture-2018-03",
            "text": "Information received since the Federal Open Market Committee met in January indicates that the labor market has continued to strengthen and that economic activity has been rising at a moderate rate. In view of realized and expected labor market conditions and inflation, the Committee decided to raise the target range for the federal funds rate to 1-1/2 to 1-3/4 percent.",
            "document_type": "statement",
            "event_time": "2018-03-21T14:00:00-04:00",
            "available_time": "2018-03-21T14:00:00-04:00",
            "task_label": 1,
            "meeting_id": "2018-03",
            "source": "Federal Reserve (Fixture)",
            "annotation_source": "FIXTURE_RECORD",
            "availability_source": "FIXTURE_RELEASE",
            "availability_quality": "unknown",
            "future_macro_labels": {"next_action": 1, "next_cpi_surprise": 0.1},
            "market_outcomes": {"spy_1d_return": -0.0018, "spy_5d_return": -0.015, "treasury_2y_change": 0.04, "fed_funds_surprise": 0.02},
            "metadata": {"chair": "Powell", "vote": "unanimous", "fixture": True},
        },
        {
            "sample_id": "fed-official-fixture-2018-06",
            "text": "The Committee decided to raise the target range for the federal funds rate to 1-3/4 to 2 percent. The labor market has continued to strengthen and economic activity has been rising at a solid rate.",
            "document_type": "statement",
            "event_time": "2018-06-13T14:00:00-04:00",
            "available_time": "2018-06-13T14:00:00-04:00",
            "task_label": 1,
            "meeting_id": "2018-06",
            "source": "Federal Reserve (Fixture)",
            "annotation_source": "FIXTURE_RECORD",
            "availability_source": "FIXTURE_RELEASE",
            "availability_quality": "unknown",
            "future_macro_labels": {"next_action": 1, "next_cpi_surprise": 0.0},
            "market_outcomes": {"spy_1d_return": -0.0035, "spy_5d_return": 0.004, "treasury_2y_change": 0.03, "fed_funds_surprise": 0.01},
            "metadata": {"chair": "Powell", "vote": "unanimous", "fixture": True},
        },
        {
            "sample_id": "fed-official-fixture-2019-07",
            "text": "In light of the implications of global developments for the economic outlook as well as muted inflation pressures, the Committee decided to lower the target range for the federal funds rate to 2 to 2-1/4 percent.",
            "document_type": "statement",
            "event_time": "2019-07-31T14:00:00-04:00",
            "available_time": "2019-07-31T14:00:00-04:00",
            "task_label": -1,
            "meeting_id": "2019-07",
            "source": "Federal Reserve (Fixture)",
            "annotation_source": "FIXTURE_RECORD",
            "availability_source": "FIXTURE_RELEASE",
            "availability_quality": "unknown",
            "future_macro_labels": {"next_action": -1, "next_cpi_surprise": 0.0},
            "market_outcomes": {"spy_1d_return": -0.0109, "spy_5d_return": -0.018, "treasury_2y_change": -0.02, "fed_funds_surprise": -0.02},
            "metadata": {"chair": "Powell", "vote": "8-2", "fixture": True},
        },
        {
            "sample_id": "fed-official-fixture-2022-03",
            "text": "The Committee decided to raise the target range for the federal funds rate to 1/4 to 1/2 percent and anticipates that ongoing increases in the target range will be appropriate. In addition, the Committee expects to begin reducing its holdings of Treasury securities and agency debt and agency mortgage-backed securities at a coming meeting.",
            "document_type": "statement",
            "event_time": "2022-03-16T14:00:00-04:00",
            "available_time": "2022-03-16T14:00:00-04:00",
            "task_label": 1,
            "meeting_id": "2022-03",
            "source": "Federal Reserve (Fixture)",
            "annotation_source": "FIXTURE_RECORD",
            "availability_source": "FIXTURE_RELEASE",
            "availability_quality": "unknown",
            "future_macro_labels": {"next_action": 1, "next_cpi_surprise": 0.2},
            "market_outcomes": {"spy_1d_return": 0.0224, "spy_5d_return": 0.038, "treasury_2y_change": 0.09, "fed_funds_surprise": 0.04},
            "metadata": {"chair": "Powell", "vote": "8-1", "fixture": True},
        },
    ]
    samples = [
        TemporalSample(
            sample_id=d["sample_id"],
            text=d["text"],
            document_type=d["document_type"],
            event_time=d["event_time"],
            available_time=d["available_time"],
            task_label=d["task_label"],
            meeting_id=d["meeting_id"],
            source=d["source"],
            annotation_source=d["annotation_source"],
            availability_source=d["availability_source"],
            availability_quality=d["availability_quality"],
            future_macro_labels=d["future_macro_labels"],
            market_outcomes=d["market_outcomes"],
            metadata=d["metadata"],
        )
        for d in raw_statements
    ]
    validate_dataset(samples)
    return FOMCBenchmark(
        samples=samples,
        source_verified=False,
        annotation_verified=False,
        pit_verified=False,
    )


def load_fomc_official_statements(
    filepath: Optional[Union[str, Path]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
) -> List[TemporalSample]:
    """Load official FOMC statements with exact publication timestamps.

    Args:
        filepath: Path to JSON/JSONL/CSV file containing verified official statements.
        manifest_path: Optional path to dataset manifest.json.

    Returns:
        List of strictly validated TemporalSample instances.
    """
    if filepath is None and manifest_path is None:
        raise ValueError(
            "Official FOMC dataset loading requires an explicit verified data file path or manifest. "
            "For testing and documentation only, use create_fomc_official_fixture()."
        )
    target_path = filepath
    manifest_data = None
    if manifest_path is not None:
        m_path = Path(manifest_path)
        if not m_path.exists():
            raise FileNotFoundError(f"Manifest not found: {m_path}")
        with open(m_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        if target_path is None:
            if "data_file" in manifest_data:
                target_path = m_path.parent / manifest_data["data_file"]
            else:
                raise ValueError(f"Manifest {m_path} does not specify 'data_file'.")

    path = Path(target_path)
    if not path.exists():
        raise FileNotFoundError(f"Official FOMC statement file not found: {path}")

    # Verify SHA-256 hash if manifest is present
    if manifest_data is not None:
        expected_sha = manifest_data.get("dataset_sha256") or manifest_data.get("checksum")
        if expected_sha:
            if expected_sha.startswith("sha256:"):
                expected_sha = expected_sha[7:]
            actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_sha.lower() != expected_sha.lower():
                raise ValueError(
                    f"Dataset SHA256 verification failed for '{path}'. "
                    f"Manifest expected '{expected_sha}', but actual file SHA256 is '{actual_sha}'."
                )

    samples = load_fomc_dataset(path)
    validate_dataset(samples)
    return samples


def create_fomc_official_benchmark(
    filepath: Optional[Union[str, Path]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
) -> FOMCBenchmark:
    """Instantiate an FOMCBenchmark using official statements.

    Formal research readiness requires an explicit verified manifest_path with:
    - dataset_sha256 matching sha256(data_file)
    - source_verified == True
    - annotation_verified == True
    - pit_verified == True
    - availability_provenance non-empty
    - source_urls non-empty list
    - All samples having availability_quality == 'exact'

    If manifest_path is missing, unverified, corrupted, or omitted, the benchmark is constructed
    such that `benchmark.is_formal_research_ready() is False`.
    """
    if filepath is None and manifest_path is None:
        raise ValueError(
            "Formal official FOMC benchmark requires an explicit verified data file path or manifest. "
            "For testing and documentation only, use create_fomc_official_fixture()."
        )

    samples = load_fomc_official_statements(filepath=filepath, manifest_path=manifest_path)

    # Check manifest verification properties
    is_source_verified = False
    is_annotation_verified = False
    is_pit_verified = False
    manifest_hash_verified = False
    availability_provenance_verified = False
    source_urls_verified = False

    if manifest_path is not None:
        m_path = Path(manifest_path)
        if m_path.exists():
            try:
                with open(m_path, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                is_source_verified = bool(m_data.get("source_verified", False))
                is_annotation_verified = bool(m_data.get("annotation_verified", False))
                is_pit_verified = bool(m_data.get("pit_verified", False))

                # Verify SHA256
                expected_sha = m_data.get("dataset_sha256") or m_data.get("checksum")
                if expected_sha:
                    if expected_sha.startswith("sha256:"):
                        expected_sha = expected_sha[7:]
                    target_file = filepath or (m_path.parent / m_data.get("data_file", ""))
                    if Path(target_file).exists():
                        actual_sha = hashlib.sha256(Path(target_file).read_bytes()).hexdigest()
                        manifest_hash_verified = (actual_sha.lower() == expected_sha.lower())

                avail_prov = m_data.get("availability_provenance")
                availability_provenance_verified = bool(avail_prov and str(avail_prov).strip())

                src_urls = m_data.get("source_urls")
                source_urls_verified = bool(isinstance(src_urls, list) and len(src_urls) > 0)
            except Exception:
                pass

    return FOMCBenchmark(
        samples=samples,
        source_verified=is_source_verified,
        annotation_verified=is_annotation_verified,
        pit_verified=is_pit_verified,
        manifest_hash_verified=manifest_hash_verified,
        availability_provenance_verified=availability_provenance_verified,
        source_urls_verified=source_urls_verified,
    )
