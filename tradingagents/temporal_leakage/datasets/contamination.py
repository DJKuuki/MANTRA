"""Phase 4 Contamination Dataset Loader & Temporal Verification Helpers.

Loads the frozen 2020-2022 Federal Reserve contamination corpus and programmatically
derives temporal separation, isolation metrics, and metadata integrity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

DEFAULT_CONTAMINATION_JSONL = Path("data/research/fomc/phase4_contamination/documents.jsonl")
DEFAULT_CONTAMINATION_MANIFEST = Path("data/research/fomc/phase4_contamination/manifest.json")


def load_phase4_contamination_documents(
    filepath: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Load all documents from the Phase 4 contamination corpus."""
    p = Path(filepath) if filepath else DEFAULT_CONTAMINATION_JSONL
    if not p.exists():
        raise FileNotFoundError(f"Contamination dataset not found at: {p}")

    documents: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            doc = json.loads(line_str)
            # Mandatory schema validation
            for required_field in (
                "document_id",
                "document_type",
                "title",
                "event_time",
                "available_time",
                "availability_quality",
                "source_url",
                "source_type",
                "raw_source_sha256",
                "canonical_text_sha256",
                "text",
                "temporal_class",
            ):
                if required_field not in doc:
                    raise ValueError(f"Missing required field '{required_field}' in doc #{line_num}: {doc.get('document_id')}")

            # Strict quality assertions
            if doc["availability_quality"] != "exact":
                raise ValueError(f"Invalid availability_quality '{doc['availability_quality']}' in {doc['document_id']}")
            if doc["temporal_class"] != "post_cutoff":
                raise ValueError(f"Invalid temporal_class '{doc['temporal_class']}' in {doc['document_id']}")
            if not doc["text"].strip():
                raise ValueError(f"Empty text in contamination document: {doc['document_id']}")

            documents.append(doc)

    return documents


def derive_contamination_temporal_range(
    documents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Programmatically derive contamination temporal range, counts, and type distributions."""
    if not documents:
        raise ValueError("Cannot derive temporal range from empty document list.")

    available_times = [d["available_time"] for d in documents]
    min_time = min(available_times)
    max_time = max(available_times)

    type_counts: Dict[str, int] = {}
    total_words = 0
    for d in documents:
        t = d.get("document_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        total_words += len(d.get("text", "").split())

    return {
        "document_count": len(documents),
        "contamination_min_time": min_time,
        "contamination_max_time": max_time,
        "temporal_min": min_time,
        "temporal_max": max_time,
        "total_words": total_words,
        "type_distribution": type_counts,
        "allowed_document_types": sorted(list(type_counts.keys())),
        "primary_document_types": ["scheduled_statement"],
    }


def audit_anchor_contamination_isolation(
    anchors: List[Dict[str, Any]],
    contamination_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Audit strict temporal separation and document/text isolation between anchors and contamination."""
    anchor_times = [a.get("available_time") or a.get("event_time", "") for a in anchors]
    contam_times = [c["available_time"] for c in contamination_docs]

    anchor_max_time = max(anchor_times)
    contam_min_time = min(contam_times)

    strict_separation = bool(anchor_max_time < contam_min_time)

    anchor_doc_ids = {str(a.get("document_id", "")) for a in anchors if a.get("document_id")}
    contam_doc_ids = {str(c.get("document_id", "")) for c in contamination_docs if c.get("document_id")}
    doc_overlap = sorted(list(anchor_doc_ids.intersection(contam_doc_ids)))

    anchor_text_hashes = {str(a.get("canonical_document_text_sha256") or a.get("paragraph_text_sha256", "")) for a in anchors}
    contam_text_hashes = {str(c.get("canonical_text_sha256", "")) for c in contamination_docs}
    text_hash_overlap = sorted(list(anchor_text_hashes.intersection(contam_text_hashes) - {""}))

    return {
        "anchor_max_time": anchor_max_time,
        "contamination_min_time": contam_min_time,
        "strict_future_separation": strict_separation,
        "document_overlap_count": len(doc_overlap),
        "document_overlap_ids": doc_overlap,
        "canonical_text_overlap_count": len(text_hash_overlap),
        "canonical_text_overlap_hashes": text_hash_overlap,
    }
