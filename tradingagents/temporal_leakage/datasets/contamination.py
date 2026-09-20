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


def verify_contamination_document_sources(
    documents: List[Dict[str, Any]],
    raw_sources_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Exhaustively verify 100% of contamination documents against raw HTML sources.

    Checks for every document:
    1. Raw source snapshot exists on disk.
    2. Raw file SHA-256 matches recorded raw_source_sha256.
    3. Source URL is from official Federal Reserve domain (https://www.federalreserve.gov/).
    4. Canonical text is deterministically reconstructed from raw HTML.
    5. Reconstructed canonical text SHA-256 matches canonical_text_sha256.
    6. Stored treatment text corresponds exactly to reconstructed canonical text.
    7. Exact available_time provenance is present and valid.
    8. Document ID is unique.

    Returns structured summary with verified_count and total_count.
    """
    import hashlib
    import re
    from bs4 import BeautifulSoup

    if raw_sources_dir is None:
        raw_sources_dir = Path("data/research/fomc/phase4_contamination/raw_sources")
    raw_dir = Path(raw_sources_dir)

    seen_ids: Set[str] = set()
    verified_count = 0
    verification_errors: List[str] = []

    for idx, doc in enumerate(documents):
        doc_id = doc.get("document_id", f"doc_{idx}")
        # Check uniqueness
        if doc_id in seen_ids:
            verification_errors.append(f"Duplicate document ID: {doc_id}")
            continue
        seen_ids.add(doc_id)

        # 1. Raw source snapshot exists
        raw_path = raw_dir / f"{doc_id}.html"
        if not raw_path.exists():
            verification_errors.append(f"Missing raw snapshot for {doc_id}: {raw_path}")
            continue

        # 2. Raw file SHA-256 matches recorded hash (normalized to LF)
        raw_bytes = raw_path.read_bytes().replace(b"\r\n", b"\n")
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()
        expected_raw_sha = doc.get("raw_source_sha256", "")
        if raw_sha != expected_raw_sha:
            verification_errors.append(
                f"Raw SHA mismatch for {doc_id}: computed {raw_sha} != recorded {expected_raw_sha}"
            )
            continue

        # 3. Source URL from official Federal Reserve domain
        source_url = doc.get("source_url", "")
        if not source_url.startswith("https://www.federalreserve.gov/"):
            verification_errors.append(f"Source URL not from federalreserve.gov for {doc_id}: {source_url}")
            continue

        # 4. Canonical text deterministically reconstructed from raw source
        soup = BeautifulSoup(raw_bytes.decode("utf-8", errors="replace"), "html.parser")
        article = (
            soup.find("div", id="article")
            or soup.find("div", id="content")
            or soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
            or soup
        )
        reconstructed_text = article.get_text(separator=" ")
        reconstructed_text = re.sub(r"\s+", " ", reconstructed_text).strip()

        # 5. Reconstructed canonical text SHA-256 matches recorded canonical_text_sha256
        canon_sha = hashlib.sha256(reconstructed_text.encode("utf-8")).hexdigest()
        expected_canon_sha = doc.get("canonical_text_sha256", "")
        if canon_sha != expected_canon_sha:
            verification_errors.append(
                f"Canonical SHA mismatch for {doc_id}: computed {canon_sha} != recorded {expected_canon_sha}"
            )
            continue

        # 6. Stored treatment text corresponds exactly to reconstructed canonical text
        stored_text = doc.get("text", "")
        if stored_text != reconstructed_text:
            verification_errors.append(f"Stored text mismatch for {doc_id}")
            continue

        # 7. Exact available_time provenance is present and valid
        if doc.get("availability_quality") != "exact" or not doc.get("available_time"):
            verification_errors.append(f"Invalid availability provenance for {doc_id}")
            continue

        verified_count += 1

    all_verified = bool(verified_count == len(documents) and not verification_errors)
    return {
        "verified_count": verified_count,
        "total_count": len(documents),
        "all_verified": all_verified,
        "verification_summary": f"{verified_count} / {len(documents)}",
        "errors": verification_errors,
    }

