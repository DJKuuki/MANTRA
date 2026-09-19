"""Download and conversion script for Trillion Dollar Words (Shah et al., ACL 2023).

Source: https://github.com/gtfintechlab/fomc-hawkish-dovish
License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

Converts annotated Excel splits (minutes, press conferences, speeches) into canonical
Point-in-Time research format: data/research/fomc/fomc_temporal_dataset.jsonl
and generates data/research/fomc/manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request

import pandas as pd

RAW_BASE_URL = "https://raw.githubusercontent.com/gtfintechlab/fomc-hawkish-dovish/main/data/annotated_data"

DATA_FILES = [
    {
        "filename": "manual-mm-split.xlsx",
        "document_type": "minutes",
        "prefix": "tdw-mm",
        "description": "FOMC Meeting Minutes",
    },
    {
        "filename": "manual-pc-split.xlsx",
        "document_type": "press_conference",
        "prefix": "tdw-pc",
        "description": "FOMC Press Conference Transcripts",
    },
    {
        "filename": "manual-sp-split.xlsx",
        "document_type": "speech",
        "prefix": "tdw-sp",
        "description": "Federal Reserve Speeches",
    },
]

# Trillion Dollar Words label convention:
# 0 -> Dovish (-1 in MANTRA)
# 1 -> Hawkish (+1 in MANTRA)
# 2 -> Neutral (0 in MANTRA)
# '-' -> Discarded / unlabelled
LABEL_MAPPING = {
    0: -1,
    1: 1,
    2: 0,
}


def download_or_read_excel(filename: str, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """Download or read cached Excel file."""
    if cache_dir is not None:
        local_path = cache_dir / filename
        if local_path.exists():
            return pd.read_excel(local_path)

    url = f"{RAW_BASE_URL}/{filename}"
    req = urllib.request.Request(url, headers={"User-Agent": "MANTRA-Research-Ingestion/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / filename, "wb") as f:
            f.write(content)

    return pd.read_excel(io.BytesIO(content))


def convert_records(cache_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Download, validate, and convert Trillion Dollar Words annotations into canonical schema."""
    canonical_samples: List[Dict[str, Any]] = []

    for file_info in DATA_FILES:
        filename = file_info["filename"]
        doc_type = file_info["document_type"]
        prefix = file_info["prefix"]

        df = download_or_read_excel(filename, cache_dir=cache_dir)

        for idx, row in df.iterrows():
            raw_label = row.get("label")
            if pd.isna(raw_label) or str(raw_label).strip() == "-":
                continue

            try:
                int_raw_label = int(raw_label)
            except (ValueError, TypeError):
                continue

            if int_raw_label not in LABEL_MAPPING:
                continue

            task_label = LABEL_MAPPING[int_raw_label]
            raw_text = str(row.get("sentence", "")).strip()
            if not raw_text:
                continue

            try:
                year = int(row.get("year", 2000))
            except (ValueError, TypeError):
                year = 2000

            orig_index = row.get("orig_index", idx)
            sample_id = f"{prefix}-{year}-{orig_index}-{idx}"

            # Conservative Point-in-Time availability:
            # Paper metadata provides 'year' but not intraday release timestamps.
            # In accordance with PIT protocols, availability_quality is 'unknown'
            # (or heuristic date), and pit_verified remains False.
            event_time = f"{year:04d}-06-30T12:00:00Z"
            available_time = f"{year:04d}-06-30T12:00:00Z"

            record = {
                "sample_id": sample_id,
                "text": raw_text,
                "document_type": doc_type,
                "event_time": event_time,
                "available_time": available_time,
                "task_label": task_label,
                "meeting_id": f"FOMC-{year}",
                "source": "Federal Reserve",
                "annotation_source": "Trillion Dollar Words (Shah et al., ACL 2023)",
                "availability_source": "ACL_2023_TRILLION_DOLLAR_WORDS",
                "availability_quality": "unknown",
                "future_macro_labels": {},
                "market_outcomes": {},
                "metadata": {
                    "raw_label": int_raw_label,
                    "year": year,
                    "source_file": filename,
                    "orig_index": int(orig_index) if pd.notna(orig_index) else idx,
                    "temporal_resolution": "year",
                    "timestamp_imputed": True,
                    "timestamp_imputation_rule": "mid_year_placeholder",
                },
            }
            canonical_samples.append(record)

    return canonical_samples


def build_canonical_fomc_dataset(output_dir: Path, cache_dir: Optional[Path] = None) -> Path:
    """Build canonical jsonl and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "fomc_temporal_dataset.jsonl"
    manifest_path = output_dir / "manifest.json"
    readme_path = output_dir / "README.md"

    samples = convert_records(cache_dir=cache_dir)

    # Sort deterministically by available_time, document_type, sample_id
    samples.sort(key=lambda x: (x["available_time"], x["document_type"], x["sample_id"]))

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Compute SHA-256 hash
    hasher = hashlib.sha256()
    with open(jsonl_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    checksum = hasher.hexdigest()

    manifest = {
        "dataset_name": "fomc_trillion_dollar_words_temporal",
        "source": "gtfintechlab/fomc-hawkish-dovish (ACL 2023)",
        "paper_citation": "Shah et al., Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis, ACL 2023",
        "paper_doi": "10.18653/v1/2023.acl-long.368",
        "version": "1.0.0",
        "license": "CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International)",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(samples),
        "source_verified": True,
        "annotation_verified": True,
        "pit_verified": False,
        "checksum": f"sha256:{checksum}",
        "document_types": {
            "minutes": sum(1 for s in samples if s["document_type"] == "minutes"),
            "press_conference": sum(1 for s in samples if s["document_type"] == "press_conference"),
            "speech": sum(1 for s in samples if s["document_type"] == "speech"),
        },
        "stance_distribution": {
            "hawkish (+1)": sum(1 for s in samples if s["task_label"] == 1),
            "neutral (0)": sum(1 for s in samples if s["task_label"] == 0),
            "dovish (-1)": sum(1 for s in samples if s["task_label"] == -1),
        },
        "pit_quality_breakdown": {
            "exact": sum(1 for s in samples if s["availability_quality"] == "exact"),
            "heuristic": sum(1 for s in samples if s["availability_quality"] == "heuristic"),
            "unknown": sum(1 for s in samples if s["availability_quality"] == "unknown"),
        },
        "splits_breakdown": {
            "train (<=2018)": sum(1 for s in samples if int(s["metadata"]["year"]) <= 2018),
            "dev (2019)": sum(1 for s in samples if int(s["metadata"]["year"]) == 2019),
            "test (>=2020)": sum(1 for s in samples if int(s["metadata"]["year"]) >= 2020),
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    readme_content = f"""# Canonical FOMC Research Dataset (Trillion Dollar Words)

## 1. Overview
This dataset is adapted from the official research release of:
> **Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis**  
> Agam Shah, Suvan Paturi, Sudheer Chava  
> *ACL 2023 (Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics)*  
> DOI: [10.18653/v1/2023.acl-long.368](https://doi.org/10.18653/v1/2023.acl-long.368)  
> Repository: `gtfintechlab/fomc-hawkish-dovish`  
> License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

## 2. Manifest & Verification
- **Total Validated Samples**: {manifest['num_samples']}
- **Checksum**: `{manifest['checksum']}`
- **Source Verified**: `{manifest['source_verified']}` (Federal Reserve communications verified by Shah et al.)
- **Annotation Verified**: `{manifest['annotation_verified']}` (Manually labeled by financial economics domain experts)
- **PIT Verified**: `{manifest['pit_verified']}` (**Explicitly False**: Sentence-level corpus includes `year` metadata but lacks second-level publication timestamps)

## 3. Label Breakdown
- **Hawkish (+1)**: {manifest['stance_distribution']['hawkish (+1)']}
- **Neutral (0)**: {manifest['stance_distribution']['neutral (0)']}
- **Dovish (-1)**: {manifest['stance_distribution']['dovish (-1)']}

## 4. Document Type Distribution
- **Minutes**: {manifest['document_types']['minutes']}
- **Press Conferences**: {manifest['document_types']['press_conference']}
- **Speeches**: {manifest['document_types']['speech']}

## 5. Temporal Splits
- **Train Split (<= 2018)**: {manifest['splits_breakdown']['train (<=2018)']} samples
- **Dev Split (2019)**: {manifest['splits_breakdown']['dev (2019)']} samples
- **Test Split (>= 2020)**: {manifest['splits_breakdown']['test (>=2020)']} samples

## 6. Point-in-Time Safety Protocol Notice
Because publication intraday timestamps are absent in the upstream academic release, all samples are marked `availability_quality: unknown`.
Under `configs/fomc_formal_experiment.yaml`, these samples are rejected to uphold zero-tolerance exact Point-in-Time standards. They are utilized for real encoder pretraining corpus construction, representation baseline extraction, and CI/development smoke tests (`configs/fomc_ci.yaml`).
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    return jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert Trillion Dollar Words FOMC dataset")
    parser.add_argument("--output-dir", type=str, default="data/research/fomc", help="Target output directory")
    parser.add_argument("--cache-dir", type=str, default="data/research/fomc/raw_cache", help="Raw Excel cache dir")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    print(f"[TDW Ingestion] Starting Trillion Dollar Words ingestion -> {out_dir}")
    jsonl_path = build_canonical_fomc_dataset(out_dir, cache_dir=cache_dir)
    print(f"[TDW Ingestion] Successfully created canonical dataset: {jsonl_path}")


if __name__ == "__main__":
    main()
