# Phase 4 FOMC Contamination Dataset Specification

## 1. Dataset Overview

This directory contains the frozen, point-in-time verified **Post-Cutoff Contamination Corpus** for Phase 4 of the MANTRA causal evaluation framework.

- **Corpus Period**: 2020-01-01 to 2022-12-31 (3 full calendar years of monetary policy decisions).
- **Temporal Cutoff ($T_{\text{cutoff}}$)**: `2019-12-31T23:59:59Z`.
- **Earliest Contamination Availability**: `2020-01-29T19:00:00Z` (January 2020 FOMC Statement).
- **Latest Document Availability**: `2023-01-04T19:00:00Z` (Minutes for December 2022 meeting).
- **Temporal Separation Buffer**: Exactly **48 calendar days** ($\max(T_{\text{anchors}}) = \text{2019-12-11T19:00:00Z} < \min(T_{\text{contamination}}) = \text{2020-01-29T19:00:00Z}$).
- **Primary Source**: Board of Governors of the Federal Reserve System official press releases and policy archives (`federalreserve.gov`).

---

## 2. Document Composition & Categorization

The corpus consists of 50 official Federal Reserve documents structured to reflect authentic central bank communications:

| Document Category | Type Identifier | Document Count | Role in Evaluation |
| :--- | :--- | :--- | :--- |
| **Scheduled Policy Statements** | `scheduled_statement` | 23 | **Primary Contamination Corpus**: Matches anchor statement style and domain to minimize confounding. |
| **Unscheduled Policy Statements** | `unscheduled_statement` | 2 | Emergency COVID-19 pandemic monetary actions (March 3, 2020 and March 15, 2020). |
| **Strategy Statements** | `policy_strategy_statement` | 1 | Formal Statement on Longer-Run Goals and Monetary Policy Strategy (August 27, 2020). |
| **Meeting Minutes** | `meeting_minutes` | 24 | **Secondary Contamination Corpus**: Full analytical deliberations released 3 weeks post-meeting (~12,000–18,000 words each). |

**Total Word Count**: > 440,000 words (> 550,000 BPE/WordPiece tokens).  
**Total Unique Tokens**: Sufficient to support the 256,000-token continued pre-training budget at $D=1.00$ with zero forced repetition (`repetition_ratio < 0.05`).

---

## 3. Schema & Metadata Specification

Each line in `documents.jsonl` is a JSON object with the following schema:

```json
{
  "document_id": "fomc-statement-2020-01-29",
  "document_type": "scheduled_statement",
  "title": "Federal Reserve issues FOMC statement (January 29, 2020)",
  "event_time": "2020-01-29T19:00:00Z",
  "available_time": "2020-01-29T19:00:00Z",
  "availability_quality": "exact",
  "source_url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20200129a.htm",
  "source_type": "federal_reserve_official",
  "raw_source_sha256": "...",
  "canonical_text_sha256": "...",
  "word_count": 526,
  "temporal_class": "post_cutoff",
  "text": "..."
}
```

### Invariant Checks:
1. **Exact Timestamps**: Every document has exact UTC ISO-8601 release timestamps down to seconds/minutes (`availability_quality = "exact"`). Midnight placeholders (`T00:00:00Z`) or mid-year approximations are strictly prohibited.
2. **Zero Overlap**: Document IDs and text content share zero overlap with historical pre-cutoff evaluation anchors:
   $$\text{DocIDs}_{\text{anchor}} \cap \text{DocIDs}_{\text{contamination}} = \emptyset$$
   $$\text{CanonicalOverlapCount} = 0$$
3. **Local Snapshots**: All 50 raw HTML files are cached locally in `raw_sources/` with verified canonical SHA-256 hashes matching `manifest.json`.
