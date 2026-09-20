"""Market Outcome Recomputation and Raw Table Ingestion Engine.

Provides deterministic, machine-reconstructible post-event market outcomes
for FOMC meetings directly from verified raw market tables:
- spy_daily_raw.csv (SPDR S&P 500 ETF Trust)
- treasury_2y_raw.csv (2-Year Treasury Note Futures ZT=F and FRED DGS2 yield)
- market_manifest.json (Provenance tracking and hashes)

Key Principles:
1. Recomputability: Market outcomes are never hardcoded unverified numbers;
   they are derived deterministically from the raw price/yield tables.
2. Resolution Labeling: Strictly labeled "daily_resolution_post_event_response",
   NOT intraday FOMC surprise. Intraday is NOT_EVALUATED until tick data is verified.
3. Priority: 2Y Treasury response is Primary (E_L^{2Y}); SPY is Secondary (E_L^{SPY}).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..fomc_benchmark import verify_file_sha256

DEFAULT_MARKET_DIR = Path("data/research/market")


def load_market_tables(
    market_dir: Union[str, Path] = DEFAULT_MARKET_DIR,
    verify_hashes: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load and optionally verify raw market data tables."""
    m_dir = Path(market_dir)
    spy_path = m_dir / "spy_daily_raw.csv"
    t2y_path = m_dir / "treasury_2y_raw.csv"
    manifest_path = m_dir / "market_manifest.json"

    if not spy_path.exists():
        raise FileNotFoundError(f"Raw SPY table not found: {spy_path}")
    if not t2y_path.exists():
        raise FileNotFoundError(f"Raw 2Y Treasury table not found: {t2y_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Market manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if verify_hashes:
        spy_expected = manifest.get("files", {}).get("spy_daily_raw.csv", {}).get("sha256")
        if spy_expected and not verify_file_sha256(spy_path, spy_expected):
            raise ValueError(f"SPY raw table SHA256 mismatch against market_manifest.json.")

        t2y_expected = manifest.get("files", {}).get("treasury_2y_raw.csv", {}).get("sha256")
        if t2y_expected and not verify_file_sha256(t2y_path, t2y_expected):
            raise ValueError(f"Treasury 2Y raw table SHA256 mismatch against market_manifest.json.")

    spy_df = pd.read_csv(spy_path)
    spy_df["Date"] = pd.to_datetime(spy_df["Date"]).dt.strftime("%Y-%m-%d")
    spy_df = spy_df.sort_values("Date").reset_index(drop=True)

    t2y_df = pd.read_csv(t2y_path)
    t2y_df["Date"] = pd.to_datetime(t2y_df["Date"]).dt.strftime("%Y-%m-%d")
    t2y_df = t2y_df.sort_values("Date").reset_index(drop=True)

    return spy_df, t2y_df, manifest


def recompute_market_outcomes(
    meeting_date: str,
    spy_df: pd.DataFrame,
    t2y_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Deterministically calculate 1d and 5d post-event forward returns and 2Y changes.

    Formula:
        SPY 1d return: (Close_{t+1} - Close_t) / Close_t
        SPY 5d return: (Close_{t+5} - Close_t) / Close_t
        2Y Treasury change: ZT_Close_{t+1} - ZT_Close_t (price points difference)
        2Y Treasury yield change: DGS2_{t+1} - DGS2_t (yield difference, if available)

    Args:
        meeting_date: YYYY-MM-DD string of the FOMC decision.
        spy_df: Clean chronological SPY daily DataFrame.
        t2y_df: Clean chronological 2Y Treasury daily DataFrame.

    Returns:
        Dict of recomputed market outcomes.
    """
    m_date = str(meeting_date)

    # 1. SPY Returns
    spy_matches = spy_df.index[spy_df["Date"] == m_date].tolist()
    if not spy_matches:
        # If weekend/holiday, locate nearest preceding trading day
        past_indices = spy_df.index[spy_df["Date"] <= m_date].tolist()
        if not past_indices:
            raise ValueError(f"No SPY market data prior to or on meeting date '{m_date}'.")
        spy_loc = past_indices[-1]
    else:
        spy_loc = spy_matches[0]

    if spy_loc + 1 >= len(spy_df):
        raise ValueError(f"Insufficient future SPY data after meeting date '{m_date}' for 1d return.")

    spy_c_t = float(spy_df["Close"].iloc[spy_loc])
    spy_c_t1 = float(spy_df["Close"].iloc[spy_loc + 1])
    spy_1d_ret = round((spy_c_t1 - spy_c_t) / spy_c_t, 4)

    if spy_loc + 5 < len(spy_df):
        spy_c_t5 = float(spy_df["Close"].iloc[spy_loc + 5])
        spy_5d_ret = round((spy_c_t5 - spy_c_t) / spy_c_t, 4)
    else:
        spy_5d_ret = None

    # 2. 2Y Treasury (ZT=F futures proxy and DGS2 yield)
    # Forward-fill ZT_Close for trading day continuity if futures has minor holiday mismatch
    clean_t2y = t2y_df.dropna(subset=["ZT_Close"]).reset_index(drop=True)
    t2y_matches = clean_t2y.index[clean_t2y["Date"] == m_date].tolist()
    if not t2y_matches:
        past_t2y = clean_t2y.index[clean_t2y["Date"] <= m_date].tolist()
        if not past_t2y:
            raise ValueError(f"No Treasury 2Y data on or prior to meeting date '{m_date}'.")
        t2y_loc = past_t2y[-1]
    else:
        t2y_loc = t2y_matches[0]

    if t2y_loc + 1 >= len(clean_t2y):
        raise ValueError(f"Insufficient future 2Y data after meeting date '{m_date}' for 1d change.")

    zt_c_t = float(clean_t2y["ZT_Close"].iloc[t2y_loc])
    zt_c_t1 = float(clean_t2y["ZT_Close"].iloc[t2y_loc + 1])
    t2y_change = round(zt_c_t1 - zt_c_t, 4)

    # Optional FRED Yield Change
    t2y_yield_change: Optional[float] = None
    if "DGS2" in t2y_df.columns:
        yield_df = t2y_df.dropna(subset=["DGS2"]).reset_index(drop=True)
        y_matches = yield_df.index[yield_df["Date"] == m_date].tolist()
        if y_matches and y_matches[0] + 1 < len(yield_df):
            yloc = y_matches[0]
            t2y_yield_change = round(float(yield_df["DGS2"].iloc[yloc + 1] - yield_df["DGS2"].iloc[yloc]), 4)

    return {
        "spy_1d_return": spy_1d_ret,
        "spy_5d_return": spy_5d_ret,
        "treasury_2y_change": t2y_change,
        "treasury_2y_yield_change": t2y_yield_change,
        "market_resolution": "daily_post_event",
        "market_source": "raw_market_tables_verified",
    }
