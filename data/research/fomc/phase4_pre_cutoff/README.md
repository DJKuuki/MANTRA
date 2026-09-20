# Phase 4 Pre-Cutoff Baseline Corpus (2015–2019)

## 1. Overview
This directory contains the official, verified historical pre-cutoff Federal Reserve communications dataset used for sham-control baseline and low-dose token stream construction in Phase 4 confirmatory analysis.

## 2. Temporal Boundaries & Strict Future Separation
- **Temporal Window (Availability)**: `2015-01-28T19:00:00Z` to `2019-12-11T19:00:00Z`
- **Pre-Cutoff Temporal Invariant**: `available_time <= 2019-12-31T23:59:59Z` (strictly enforced by public release time, rejecting documents like `fomc-minutes-2019-12-11` whose release occurred in 2020)
- **Earliest Contamination Timestamp**: `2020-01-29T19:00:00Z`
- **Temporal Separation Buffer**: Strict 48-day calendar buffer between latest pre-cutoff availability (`2019-12-11T19:00:00Z`) and earliest post-cutoff contamination document (`2020-01-29T19:00:00Z`).

## 3. Corpus Composition
- **Total Documents**: 63 official Federal Reserve documents
  - `scheduled_statement`: 40 scheduled FOMC monetary policy statements (2015–2019)
  - `meeting_minutes`: 23 official FOMC meeting minutes (2017–2019)
- **Total Words**: 243,001 words
- **Estimated Subword Tokens**: ~315,901 tokens (> 256,000 required budget)
- **D0 Repetition Feasibility**: Forced repetition ratio = 0.00 <= 0.20 (no shortfall).

## 4. Metadata Schema
Each line in `documents.jsonl` contains:
- `document_id`: Canonical document identifier
- `document_type`: `scheduled_statement` or `meeting_minutes`
- `title`: Document title
- `event_time`: UTC meeting timestamp
- `available_time`: UTC public release timestamp
- `availability_quality`: `exact`
- `source_url`: Official Federal Reserve URL
- `source_type`: `federal_reserve_official`
- `raw_source_sha256`: SHA-256 of raw HTML source
- `canonical_text_sha256`: SHA-256 of canonical text
- `word_count`: Word count
- `temporal_class`: `pre_cutoff`
- `text`: Clean extracted text
