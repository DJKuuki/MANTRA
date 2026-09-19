"""FOMC Benchmark: Dataset loader, Point-in-Time protocol validator, and Toy vs Research Splitter.

IMPORTANT:
    The built-in reference dataset provided by `create_toy_fomc_dataset()` and `ToyFOMCBenchmark`
    is a TOY / SYNTHETIC DATASET ONLY. It is designed strictly for unit tests, smoke tests,
    and API validation. It must NOT be used for empirical research claims or publication conclusions.
    Formal empirical experiments must ingest verified datasets (e.g. Trillion Dollar Words)
    via `FOMCBenchmark.from_file(...)`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd
from .temporal_model import TemporalSample, parse_iso_utc


class DatasetValidationError(ValueError):
    """Raised when a temporal dataset or sample violates structural, schema, or Point-in-Time rules."""
    pass


def validate_temporal_sample(
    sample: TemporalSample,
    require_timezone_aware: bool = True,
    source_timezone: Optional[str] = None,
) -> None:
    """Validate that a TemporalSample satisfies strict research schema and PIT constraints.

    Validation Rules:
    - sample_id: must be a non-empty string.
    - text: must be a non-empty string.
    - task_label: must be integer in {-1, 0, 1}.
    - event_time: must be valid ISO-8601 timestamp string.
    - available_time: must be valid ISO-8601 timestamp string.
    - Timezone: timestamps must be timezone-aware (rejects naive timestamps unless source_timezone provided).
    - Causality: event_time <= available_time (an announcement cannot realistically be legally available before event occurs).
    - Metadata: document_type, source, annotation_source must be non-empty strings.
    """
    if not isinstance(sample, TemporalSample):
        raise DatasetValidationError(f"Expected TemporalSample instance, got {type(sample).__name__}")

    # 1. ID
    if not sample.sample_id or not str(sample.sample_id).strip():
        raise DatasetValidationError(f"sample_id must be a non-empty string, got: {sample.sample_id!r}")

    # 2. Text
    if not sample.text or not str(sample.text).strip():
        raise DatasetValidationError(f"Sample '{sample.sample_id}' text must be a non-empty string.")

    # 3. Label: strictly {-1, 0, 1}
    if sample.task_label not in {-1, 0, 1} or isinstance(sample.task_label, bool):
        raise DatasetValidationError(
            f"Sample '{sample.sample_id}' has invalid task_label {sample.task_label!r}. "
            "task_label must be an integer in {-1 (Dovish), 0 (Neutral), 1 (Hawkish)}."
        )

    # 4. Timestamps & Timezones
    try:
        dt_event = parse_iso_utc(
            sample.event_time,
            default_timezone=source_timezone if (not require_timezone_aware or source_timezone) else None,
        )
    except Exception as e:
        raise DatasetValidationError(
            f"Sample '{sample.sample_id}' has invalid event_time '{sample.event_time}': {e}"
        ) from e

    try:
        dt_avail = parse_iso_utc(
            sample.available_time,
            default_timezone=source_timezone if (not require_timezone_aware or source_timezone) else None,
        )
    except Exception as e:
        raise DatasetValidationError(
            f"Sample '{sample.sample_id}' has invalid available_time '{sample.available_time}': {e}"
        ) from e

    # 5. Temporal causality: event occurs before or at availability time
    if dt_event > dt_avail:
        raise DatasetValidationError(
            f"Sample '{sample.sample_id}' violates temporal causality: "
            f"event_time ({sample.event_time}) is after available_time ({sample.available_time})."
        )

    # 6. Provenance metadata
    for field_name in ["document_type", "source", "annotation_source"]:
        val = getattr(sample, field_name, None)
        if not val or not str(val).strip():
            raise DatasetValidationError(
                f"Sample '{sample.sample_id}' must have non-empty required field '{field_name}'."
            )


def validate_dataset(
    samples: Sequence[TemporalSample],
    require_timezone_aware: bool = True,
    source_timezone: Optional[str] = None,
) -> None:
    """Validate an entire collection of TemporalSample instances for consistency and integrity."""
    if not samples:
        raise DatasetValidationError("Dataset is empty. At least one TemporalSample is required.")

    seen_ids = set()
    for idx, sample in enumerate(samples):
        validate_temporal_sample(
            sample,
            require_timezone_aware=require_timezone_aware,
            source_timezone=source_timezone,
        )
        if sample.sample_id in seen_ids:
            raise DatasetValidationError(
                f"Duplicate sample_id '{sample.sample_id}' detected at index {idx}."
            )
        seen_ids.add(sample.sample_id)


def create_toy_fomc_dataset() -> List[TemporalSample]:
    """Create a curated toy/synthetic reference dataset of historical FOMC statements.

    WARNING:
        TOY / SYNTHETIC DATASET ONLY.
        Designed strictly for unit tests, smoke tests, and API validation.
        NOT FOR RESEARCH CONCLUSIONS OR EMPIRICAL CLAIMS.
    """
    raw_meetings = [
        # Pre-2019 Training Era (D_train <= 2018)
        {
            "sample_id": "toy-2018-03",
            "text": "The Federal Reserve raised the target range for the federal funds rate to 1.75 percent. Economic activity has been expanding at a strong rate.",
            "document_type": "statement",
            "event_time": "2018-03-21T14:00:00-04:00",
            "available_time": "2018-03-21T14:00:00-04:00",
            "task_label": 1,  # Hawkish
            "meeting_id": "2018-03",
            "source": "Federal Reserve",
            "annotation_source": "toy_synthetic",
            "future_macro_labels": {"next_action": 1, "next_cpi_surprise": 0.1},
            "market_outcomes": {"spy_1d_return": -0.0018, "spy_5d_return": -0.015, "treasury_2y_change": 0.04, "fed_funds_surprise": 0.02},
            "metadata": {"chair": "Powell"},
        },
        {
            "sample_id": "toy-2018-06",
            "text": "The Committee decided to raise the target range to 2.00 percent. The labor market has continued to strengthen and inflation is near our objective.",
            "document_type": "statement",
            "event_time": "2018-06-13T14:00:00-04:00",
            "available_time": "2018-06-13T14:00:00-04:00",
            "task_label": 1,  # Hawkish
            "meeting_id": "2018-06",
            "source": "Federal Reserve",
            "annotation_source": "toy_synthetic",
            "future_macro_labels": {"next_action": 1, "next_cpi_surprise": 0.0},
            "market_outcomes": {"spy_1d_return": -0.0035, "spy_5d_return": 0.004, "treasury_2y_change": 0.03, "fed_funds_surprise": 0.01},
            "metadata": {"chair": "Powell"},
        },
        {
            "sample_id": "toy-2018-08",
            "text": "Consistent with its statutory mandate, the Committee seeks to foster maximum employment and price stability. In support of these goals, the Committee decided to maintain the target range.",
            "document_type": "statement",
            "event_time": "2018-08-01T14:00:00-04:00",
            "available_time": "2018-08-01T14:00:00-04:00",
            "task_label": 0,  # Neutral
            "meeting_id": "2018-08",
            "source": "Federal Reserve",
            "annotation_source": "toy_synthetic",
            "future_macro_labels": {"next_action": 1, "next_cpi_surprise": 0.0},
            "market_outcomes": {"spy_1d_return": -0.0010, "spy_5d_return": 0.006, "treasury_2y_change": 0.01, "fed_funds_surprise": 0.00},
            "metadata": {"chair": "Powell"},
        },
        {
            "sample_id": "toy-2018-12",
            "text": "The Committee decided to raise the target range to 2.50 percent, but noted that some financial market volatility and global economic growth had softened.",
            "document_type": "statement",
            "event_time": "2018-12-19T14:00:00-05:00",
            "available_time": "2018-12-19T14:00:00-05:00",
            "task_label": 1,  # Hawkish
            "meeting_id": "2018-12",
            "source": "Federal Reserve",
            "annotation_source": "toy_synthetic",
            "future_macro_labels": {"next_action": 0, "next_cpi_surprise": -0.1},
            "market_outcomes": {"spy_1d_return": -0.0154, "spy_5d_return": -0.042, "treasury_2y_change": -0.06, "fed_funds_surprise": 0.03},
            "metadata": {"chair": "Powell"},
        },
        # 2019 Dev Validation Era (D_dev)
        {
            "sample_id": "toy-2019-07",
            "text": "In light of global developments and muted inflation pressures, the Committee decided to lower the target range for the federal funds rate to 2.00 to 2.25 percent.",
            "document_type": "statement",
            "event_time": "2019-07-31T14:00:00-04:00",
            "available_time": "2019-07-31T14:00:00-04:00",
            "task_label": -1,  # Dovish
            "meeting_id": "2019-07",
            "source": "Federal Reserve",
            "annotation_source": "toy_synthetic",
            "future_macro_labels": {"next_action": -1, "next_cpi_surprise": 0.0},
            "market_outcomes": {"spy_1d_return": -0.0109, "spy_5d_return": -0.018, "treasury_2y_change": -0.02, "fed_funds_surprise": -0.02},
            "metadata": {"chair": "Powell"},
        },
        {
            "sample_id": "toy-2019-09",
            "text": "The Committee lowered the target range by 25 basis points to promote sustained expansion in economic activity and return inflation to its symmetric 2 percent goal.",
            "document_type": "statement",
            "event_time": "2019-09-18T14:00:00-04:00",
            "available_time": "2019-09-18T14:00:00-04:00",
            "task_label": -1,  # Dovish
            "meeting_id": "2019-09",
            "source": "Federal Reserve",
            "annotation_source": "toy_synthetic",
            "future_macro_labels": {"next_action": -1, "next_cpi_surprise": -0.1},
            "market_outcomes": {"spy_1d_return": 0.0003, "spy_5d_return": 0.001, "treasury_2y_change": 0.01, "fed_funds_surprise": 0.00},
            "metadata": {"chair": "Powell"},
        },
        # Post-2020 Out-of-Sample Test Era (D_test 2020-2024)
        {
            "sample_id": "toy-2020-03",
            "text": "The coronavirus outbreak has harmed communities and disrupted economic activity. The Federal Reserve lowered the target range to 0 to 1/4 percent in an emergency action.",
            "document_type": "statement",
            "event_time": "2020-03-15T17:00:00-04:00",
            "available_time": "2020-03-15T17:00:00-04:00",
            "task_label": -1,  # Strongly Dovish / Emergency
            "meeting_id": "2020-03-emergency",
            "source": "Federal Reserve",
            "annotation_source": "toy_synthetic",
            "future_macro_labels": {"next_action": 0, "next_cpi_surprise": -0.4},
            "market_outcomes": {"spy_1d_return": -0.1198, "spy_5d_return": -0.142, "treasury_2y_change": -0.18, "fed_funds_surprise": -0.50},
            "metadata": {"chair": "Powell"},
        },
        {
            "sample_id": "toy-2020-06",
            "text": "The Committee decided to keep the target range at 0 to 1/4 percent and expects it will be appropriate to maintain this target range until labor market conditions reach maximum employment.",
            "document_type": "statement",
            "event_time": "2020-06-10T14:00:00-04:00",
            "available_time": "2020-06-10T14:00:00-04:00",
            "task_label": -1,  # Dovish
            "meeting_id": "2020-06",
            "source": "Federal Reserve",
            "annotation_source": "toy_synthetic",
            "future_macro_labels": {"next_action": 0, "next_cpi_surprise": 0.1},
            "market_outcomes": {"spy_1d_return": -0.0053, "spy_5d_return": -0.012, "treasury_2y_change": -0.01, "fed_funds_surprise": 0.00},
            "metadata": {"chair": "Powell"},
        },
        {
            "sample_id": "toy-2021-11",
            "text": "Inflation remains elevated, reflecting supply and demand imbalances related to the pandemic. The Committee decided to begin reducing the monthly pace of its net asset purchases.",
            "document_type": "statement",
            "event_time": "2021-11-03T14:00:00-04:00",
            "available_time": "2021-11-03T14:00:00-04:00",
            "task_label": 1,  # Hawkish / Taper
            "meeting_id": "2021-11",
            "source": "Federal Reserve",
            "annotation_source": "toy_synthetic",
            "future_macro_labels": {"next_action": 1, "next_cpi_surprise": 0.3},
            "market_outcomes": {"spy_1d_return": 0.0065, "spy_5d_return": 0.008, "treasury_2y_change": 0.03, "fed_funds_surprise": 0.01},
            "metadata": {"chair": "Powell"},
        },
        {
            "sample_id": "toy-2022-03",
            "text": "The Committee decided to raise the target range for the federal funds rate to 1/4 to 1/2 percent and anticipates that ongoing increases in the target range will be appropriate.",
            "document_type": "statement",
            "event_time": "2022-03-16T14:00:00-04:00",
            "available_time": "2022-03-16T14:00:00-04:00",
            "task_label": 1,  # Hawkish / Liftoff
            "meeting_id": "2022-03",
            "source": "Federal Reserve",
            "annotation_source": "toy_synthetic",
            "future_macro_labels": {"next_action": 1, "next_cpi_surprise": 0.2},
            "market_outcomes": {"spy_1d_return": 0.0224, "spy_5d_return": 0.038, "treasury_2y_change": 0.09, "fed_funds_surprise": 0.04},
            "metadata": {"chair": "Powell"},
        },
        {
            "sample_id": "toy-2022-06",
            "text": "The Committee decided to raise the target range by 75 basis points to 1-1/2 to 1-3/4 percent and remains highly attentive to inflation risks.",
            "document_type": "statement",
            "event_time": "2022-06-15T14:00:00-04:00",
            "available_time": "2022-06-15T14:00:00-04:00",
            "task_label": 1,  # Strongly Hawkish
            "meeting_id": "2022-06",
            "source": "Federal Reserve",
            "annotation_source": "toy_synthetic",
            "future_macro_labels": {"next_action": 1, "next_cpi_surprise": 0.4},
            "market_outcomes": {"spy_1d_return": 0.0146, "spy_5d_return": -0.031, "treasury_2y_change": 0.12, "fed_funds_surprise": 0.05},
            "metadata": {"chair": "Powell"},
        },
        {
            "sample_id": "toy-2022-11",
            "text": "In determining the extent of future increases in the target range, the Committee will take into account the cumulative tightening of monetary policy and economic lags.",
            "document_type": "statement",
            "event_time": "2022-11-02T14:00:00-04:00",
            "available_time": "2022-11-02T14:00:00-04:00",
            "task_label": 0,  # Neutral
            "meeting_id": "2022-11",
            "source": "Federal Reserve",
            "annotation_source": "toy_synthetic",
            "future_macro_labels": {"next_action": 1, "next_cpi_surprise": -0.2},
            "market_outcomes": {"spy_1d_return": -0.0250, "spy_5d_return": 0.005, "treasury_2y_change": 0.07, "fed_funds_surprise": -0.01},
            "metadata": {"chair": "Powell"},
        },
        {
            "sample_id": "toy-2023-09",
            "text": "The Committee decided to maintain the target range for the federal funds rate at 5-1/4 to 5-1/2 percent. Economic activity has been expanding at a solid pace.",
            "document_type": "statement",
            "event_time": "2023-09-20T14:00:00-04:00",
            "available_time": "2023-09-20T14:00:00-04:00",
            "task_label": 1,  # Hawkish Pause
            "meeting_id": "2023-09",
            "source": "Federal Reserve",
            "annotation_source": "toy_synthetic",
            "future_macro_labels": {"next_action": 0, "next_cpi_surprise": 0.1},
            "market_outcomes": {"spy_1d_return": -0.0164, "spy_5d_return": -0.028, "treasury_2y_change": 0.08, "fed_funds_surprise": 0.02},
            "metadata": {"chair": "Powell"},
        },
        {
            "sample_id": "toy-2024-09",
            "text": "The Committee decided to lower the target range for the federal funds rate by 1/2 percentage point to 4-3/4 to 5 percent in light of progress on inflation and balance of risks.",
            "document_type": "statement",
            "event_time": "2024-09-18T14:00:00-04:00",
            "available_time": "2024-09-18T14:00:00-04:00",
            "task_label": -1,  # Dovish Pivot
            "meeting_id": "2024-09",
            "source": "Federal Reserve",
            "annotation_source": "toy_synthetic",
            "future_macro_labels": {"next_action": -1, "next_cpi_surprise": -0.1},
            "market_outcomes": {"spy_1d_return": 0.0170, "spy_5d_return": 0.021, "treasury_2y_change": -0.05, "fed_funds_surprise": -0.05},
            "metadata": {"chair": "Powell"},
        },
    ]

    samples = []
    for d in raw_meetings:
        sample = TemporalSample(
            sample_id=d.get("sample_id", ""),
            text=d["text"],
            document_type=d.get("document_type", "statement"),
            event_time=d["event_time"],
            available_time=d["available_time"],
            task_label=d["task_label"],
            meeting_id=d.get("meeting_id", ""),
            source=d.get("source", "Federal Reserve"),
            annotation_source=d.get("annotation_source", "toy_synthetic"),
            availability_source="FED_OFFICIAL_RELEASE",
            availability_quality="exact",
            future_macro_labels=d["future_macro_labels"],
            market_outcomes=d["market_outcomes"],
            metadata=d["metadata"],
        )
        samples.append(sample)
    validate_dataset(samples)
    return samples


def load_fomc_dataset(
    filepath: Union[str, Path],
    source_timezone: Optional[str] = None,
    require_timezone_aware: bool = True,
) -> List[TemporalSample]:
    """Load and strictly validate a research FOMC benchmark dataset from a CSV or JSON/JSONL file.

    Fail-Fast Validation:
        - Required non-empty fields: sample_id, text, event_time, available_time, task_label
        - Missing or NaN task_label / available_time / text will raise DatasetValidationError
        - Timestamps must be valid ISO-8601 strings and timezone-aware (unless source_timezone provided)
        - sample_id values must be globally unique
        - task_label must be in {-1, 0, 1}

    Expected Schema:
        sample_id: str (REQUIRED, unique)
        text: str (REQUIRED, non-empty)
        event_time: str (REQUIRED, ISO 8601 with timezone)
        available_time: str (REQUIRED, ISO 8601 with timezone)
        task_label: int (REQUIRED, -1: Dovish, 0: Neutral, 1: Hawkish)
        document_type: str (optional, default 'statement')
        meeting_id: str (optional)
        source: str (optional, default 'Federal Reserve')
        annotation_source: str (optional, default 'verified_corpus')
        availability_source: str (optional, default 'OFFICIAL_RELEASE')
        availability_quality: str (optional, 'exact' or 'heuristic', default 'exact')
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"FOMC dataset file not found: {path}")

    required_fields = ["sample_id", "text", "event_time", "available_time", "task_label"]
    samples: List[TemporalSample] = []

    if path.suffix in [".json", ".jsonl"]:
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix == ".jsonl":
                records = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                records = data if isinstance(data, list) else data.get("samples", [])

        for idx, r in enumerate(records):
            for req in required_fields:
                val = r.get(req)
                if val is None or (isinstance(val, str) and not val.strip()):
                    raise DatasetValidationError(
                        f"Record {idx} in {path.name} is missing REQUIRED field '{req}'."
                    )

            try:
                task_label = int(r["task_label"])
            except (ValueError, TypeError):
                raise DatasetValidationError(
                    f"Record {idx} in {path.name} has non-integer task_label: {r.get('task_label')!r}"
                )

            future_macro = r.get("future_macro_labels", {})
            market_outs = r.get("market_outcomes", {})
            for k in ["next_action", "next_cpi_surprise"]:
                if k in r:
                    future_macro[k] = r[k]
            for k in ["spy_1d_return", "spy_5d_return", "spy_20d_return", "treasury_2y_change", "fed_funds_surprise"]:
                if k in r:
                    market_outs[k] = float(r[k])

            sample = TemporalSample(
                sample_id=str(r["sample_id"]).strip(),
                text=str(r["text"]).strip(),
                document_type=str(r.get("document_type", "statement")),
                event_time=str(r["event_time"]).strip(),
                available_time=str(r["available_time"]).strip(),
                task_label=task_label,
                meeting_id=str(r.get("meeting_id", "")),
                source=str(r.get("source", "Federal Reserve")),
                annotation_source=str(r.get("annotation_source", "verified_corpus")),
                availability_source=str(r.get("availability_source", "OFFICIAL_RELEASE")),
                availability_quality=str(r.get("availability_quality", "exact")),
                future_macro_labels=future_macro,
                market_outcomes=market_outs,
                metadata=r.get("metadata", {}),
            )
            samples.append(sample)

    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        for idx, row in df.iterrows():
            for req in required_fields:
                if req not in row or pd.isna(row[req]):
                    raise DatasetValidationError(
                        f"CSV row {idx} in {path.name} is missing REQUIRED field '{req}'."
                    )
                val_str = str(row[req]).strip()
                if not val_str:
                    raise DatasetValidationError(
                        f"CSV row {idx} in {path.name} has empty REQUIRED field '{req}'."
                    )

            try:
                task_label = int(row["task_label"])
            except (ValueError, TypeError):
                raise DatasetValidationError(
                    f"CSV row {idx} in {path.name} has non-integer task_label: {row['task_label']!r}"
                )

            future_macro = {}
            market_outs = {}
            for k in ["next_action", "next_cpi_surprise"]:
                if k in row and pd.notna(row[k]):
                    future_macro[k] = row[k]
            for k in ["spy_1d_return", "spy_5d_return", "spy_20d_return", "treasury_2y_change", "fed_funds_surprise"]:
                if k in row and pd.notna(row[k]):
                    market_outs[k] = float(row[k])

            sample = TemporalSample(
                sample_id=str(row["sample_id"]).strip(),
                text=str(row["text"]).strip(),
                document_type=str(row.get("document_type", "statement")) if pd.notna(row.get("document_type")) else "statement",
                event_time=str(row["event_time"]).strip(),
                available_time=str(row["available_time"]).strip(),
                task_label=task_label,
                meeting_id=str(row.get("meeting_id", "")) if pd.notna(row.get("meeting_id")) else "",
                source=str(row.get("source", "Federal Reserve")) if pd.notna(row.get("source")) else "Federal Reserve",
                annotation_source=str(row.get("annotation_source", "csv_corpus")) if pd.notna(row.get("annotation_source")) else "csv_corpus",
                availability_source=str(row.get("availability_source", "OFFICIAL_RELEASE")) if pd.notna(row.get("availability_source")) else "OFFICIAL_RELEASE",
                availability_quality=str(row.get("availability_quality", "exact")) if pd.notna(row.get("availability_quality")) else "exact",
                future_macro_labels=future_macro,
                market_outcomes=market_outs,
                metadata={},
            )
            samples.append(sample)
    else:
        raise ValueError(f"Unsupported dataset format '{path.suffix}'. Use .json, .jsonl, or .csv")

    validate_dataset(
        samples,
        require_timezone_aware=require_timezone_aware,
        source_timezone=source_timezone,
    )
    return samples


class FOMCBenchmark:
    """Benchmark manager for Point-in-Time FOMC research datasets.

    NOTE:
        FOMCBenchmark requires an explicit research dataset.
        Use FOMCBenchmark.from_file(...) for empirical experiments,
        or ToyFOMCBenchmark() for tests and synthetic validation.
    """

    def __init__(
        self,
        samples: Optional[List[TemporalSample]] = None,
        source_verified: bool = False,
        annotation_verified: bool = False,
        pit_verified: bool = False,
    ) -> None:
        if samples is None:
            raise ValueError(
                "FOMCBenchmark requires an explicit research dataset.\n"
                "Use FOMCBenchmark.from_file(...) for empirical experiments,\n"
                "or ToyFOMCBenchmark() for tests and synthetic validation."
            )
        validate_dataset(samples)
        self.samples: List[TemporalSample] = list(samples)
        self.dataset_validation_status: str = "validated"
        self.source_verified: bool = source_verified
        self.annotation_verified: bool = annotation_verified
        self.pit_verified: bool = pit_verified

    @classmethod
    def from_file(
        cls,
        filepath: Union[str, Path],
        source_timezone: Optional[str] = None,
        require_timezone_aware: bool = True,
        source_verified: bool = False,
        annotation_verified: bool = False,
        pit_verified: bool = False,
    ) -> FOMCBenchmark:
        """Instantiate benchmark from verified external research file."""
        loaded = load_fomc_dataset(
            filepath,
            source_timezone=source_timezone,
            require_timezone_aware=require_timezone_aware,
        )
        return cls(
            samples=loaded,
            source_verified=source_verified,
            annotation_verified=annotation_verified,
            pit_verified=pit_verified,
        )

    def get_split(self, split: str) -> List[TemporalSample]:
        """Retrieve samples filtered to a temporal partition:

        - 'train': <= 2018-12-31 (Pre-cutoff training corpus)
        - 'dev': 2019-01-01 to 2019-12-31 (Validation / early stopping)
        - 'test': >= 2020-01-01 (Out-of-sample evaluation)
        """
        s_list = []
        for s in self.samples:
            dt_str = s.available_time[:10]
            if split == "train" and dt_str <= "2018-12-31":
                s_list.append(s)
            elif split == "dev" and "2019-01-01" <= dt_str <= "2019-12-31":
                s_list.append(s)
            elif split == "test" and dt_str >= "2020-01-01":
                s_list.append(s)
        return s_list

    def validate_pit(self, as_of_date: str) -> List[TemporalSample]:
        """Strict Point-in-Time filter: returns only samples with availability_time <= as_of_date."""
        return [s for s in self.samples if s.is_available_as_of(as_of_date)]

    def get_pit_subset(self, exact_only: bool = True) -> List[TemporalSample]:
        """Return a subset of samples filtered by Point-in-Time provenance quality."""
        if not exact_only:
            return list(self.samples)
        return [s for s in self.samples if s.availability_quality == "exact"]

    def export_json(self, target_path: Union[str, Path]) -> None:
        """Export dataset to JSON format."""
        out = [asdict(s) for s in self.samples]
        p = Path(target_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


class ToyFOMCBenchmark(FOMCBenchmark):
    """TOY / SYNTHETIC BENCHMARK ONLY.

    Contains 14 curated synthetic historical statement records designed strictly
    for unit tests, smoke tests, and API validation.
    NOT FOR RESEARCH CONCLUSIONS OR EMPIRICAL CLAIMS.
    """

    def __init__(self) -> None:
        toy_samples = create_toy_fomc_dataset()
        super().__init__(
            samples=toy_samples,
            source_verified=False,
            annotation_verified=False,
            pit_verified=False,
        )
