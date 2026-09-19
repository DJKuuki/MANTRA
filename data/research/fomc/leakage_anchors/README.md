# Verified Official Federal Reserve FOMC Leakage Anchor Dataset (2019)

## Overview
This dataset contains Point-in-Time (PiT) verified official FOMC statements issued by the Federal Reserve Board of Governors across all 8 scheduled meetings of 2019. It serves as the primary ground-truth evaluation anchor set for Phase 3 Pilot Temporal Leakage Dose-Response studies.

## Key Properties
- **Source**: Federal Reserve Board of Governors Official Press Releases (Monetary Policy Statements).
- **Temporal Window**: January 30, 2019 to December 11, 2019.
- **Documents**: 8 distinct scheduled FOMC statement releases (`fomc-statement-2019-01-30` through `fomc-statement-2019-12-11`).
- **Anchors**: 25 substantive paragraph-level statement segments.
- **Publication Timestamp Verification**:
  - Scheduled release time: 14:00 America/New_York (EST / EDT converted to UTC).
  - Availability quality: strictly `exact` (`availability_quality = "exact"`).
  - Source verification: verified against official Federal Reserve press release URLs.
- **Document-Level Isolation**:
  - Anchor document IDs (`fomc-statement-2019-*`) do not overlap with the TDW contamination corpus (`manual-mm-*`, `manual-sp-*`, `manual-pc-*`).
  - Sentence-level hash overlap between anchor dataset and post-cutoff MLM contamination corpus ($\ge 2020$) is exactly 0.
- **Strict Temporal Separation**:
  - $\max(Time_{\text{anchors}}) = \text{2019-12-11T19:00:00Z}$
  - $\min(Time_{\text{contamination}}) = \text{2020-01-01T00:00:00Z}$
  - $\max(Time_{\text{anchors}}) < \min(Time_{\text{contamination}})$ is strictly satisfied (`strict_future_separation: true`).

## Target Variables
1. **Official Future Policy Action ($Y_{\text{future-action}}$)**:
   - Target Horizon: Policy decision at the next scheduled FOMC meeting.
   - Classification: $-1$ = Easing / Rate Cut, $0$ = Neutral / Hold, $+1$ = Tightening / Rate Hike.
   - Source: Official Federal Reserve target federal funds rate historical records.
2. **Post-Event Market Outcomes**:
   - `spy_1d_return`: SPY forward 1-trading-day post-announcement return.
   - `spy_5d_return`: SPY forward 5-trading-day post-announcement return.
   - `treasury_2y_change`: 2-Year Treasury Note 1-trading-day forward price change (futures proxy).
   - Resolution: Daily-resolution post-event response.
