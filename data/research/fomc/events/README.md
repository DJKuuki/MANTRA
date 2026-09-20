# Official Federal Reserve FOMC Event Dataset (2015–2019)

## Overview
This directory contains the machine-verifiable, point-in-time (PiT) verified official FOMC event table for Phase 4 confirmatory studies. It formalizes **Event / Meeting** as the primary independent statistical unit ($N_{\text{independent events}} = 40$), completely eliminating the pseudo-replication inherent in paragraph-level repeated outcomes.

## Key Properties
- **Source**: Federal Reserve Board of Governors Official Press Releases (Monetary Policy Statements).
- **Temporal Window**: January 28, 2015 through December 11, 2019 (5 full calendar years).
- **Independent Events**: Exactly 40 scheduled FOMC meetings ($N = 40$).
- **Total Anchors**: 147 substantive paragraph anchors linked to the 40 events.
- **Document Type Homogeneity**: Strictly scheduled FOMC statements only (no speeches, testimony, minutes, or press conference transcripts mixed into primary events).
- **Publication Timestamp Verification**:
  - Scheduled release time: 14:00 America/New_York (converted to UTC ISO timestamp).
  - Exact availability quality (`availability_quality = "exact"`).
  - 100% official Federal Reserve press release URLs verified HTTP 200 OK.
- **Strict Provenance & Hashing**:
  - Raw HTML snapshots preserved in `data/research/fomc/raw_sources/`.
  - Canonical text hash mode: `canonical_text_utf8_lf_nfkc`.
  - Full source registry in `data/research/fomc/source_registry.jsonl`.
- **Policy Targets**:
  - Automatically derived from `data/research/fomc/policy_history.csv`.
  - Horizon: Next scheduled FOMC meeting.
  - Class distribution: 28 Holds, 9 Hikes, 3 Cuts.
- **Market Outcomes**:
  - Deterministically recomputed from raw market tables (`treasury_2y_raw.csv` and `spy_daily_raw.csv`).
  - Primary economic outcome: 2-Year Treasury response ($E_L^{2Y}$).
  - Secondary exploratory outcome: SPY 1d/5d return.
  - Strictly labeled: "daily_resolution_post_event_response".
