"""Unit tests verifying Point-in-Time integrity and pipeline anti-leakage guards."""

import pandas as pd
import pytest
from tradingagents.dataflows.backtest_cache import (
    BacktestDataCache,
    _filter_insider_df_by_date,
    _filter_av_insider_by_date,
)
from tradingagents.dataflows.stockstats_utils import filter_financials_by_date


def test_filter_financials_by_date_enforces_filing_latency():
    # Columns represent fiscal period end dates
    dates = ["2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"]
    df = pd.DataFrame(
        [[100, 110, 120, 130]],
        index=["Total Revenue"],
        columns=dates,
    )

    # On 2024-04-01 (1 day after Q1 quarter end), 10-Q has NOT been filed yet!
    # With filing_lag_days=45, 2024-03-31 requires curr_date >= 2024-05-15
    res_april = filter_financials_by_date(df, "2024-04-01", filing_lag_days=45)
    # Only periods through 2023-12-31 should be visible (2023-12-31 + 45 days = 2024-02-14 <= 2024-04-01)
    assert "2024-03-31" not in res_april.columns
    assert "2023-12-31" in res_april.columns

    # On 2024-05-20 (after 45-day filing lag), 2024-03-31 becomes available
    res_may = filter_financials_by_date(df, "2024-05-20", filing_lag_days=45)
    assert "2024-03-31" in res_may.columns
    assert "2024-06-30" not in res_may.columns


def test_get_yf_fundamentals_withheld_in_backtest():
    cache = BacktestDataCache()
    cache._ticker = "AAPL"
    cache._last_trading_day = "2023-12-29"
    cache._store["yf_info"] = {
        "longName": "Apple Inc.",
        "marketCap": 3000000000000,
        "trailingPE": 30.5,
    }

    # In backtest mode (curr_date provided), fundamentals overview must be withheld throughout
    # even on the final trading day, preventing snapshot data leakage
    res_mid = cache.get_yf_fundamentals("AAPL", curr_date="2023-06-15")
    assert "[Backtest] Fundamentals overview withheld" in res_mid

    res_last = cache.get_yf_fundamentals("AAPL", curr_date="2023-12-29")
    assert "[Backtest] Fundamentals overview withheld" in res_last

    # In live mode (curr_date=""), live info is returned
    res_live = cache.get_yf_fundamentals("AAPL", curr_date="")
    assert "Company Fundamentals for AAPL" in res_live
    assert "Market Cap: 3000000000000" in res_live


def test_filter_insider_df_prioritizes_filing_date():
    # Transaction occurred on 2023-10-01, but Form 4 was filed on 2023-10-04
    data = pd.DataFrame(
        {
            "Transaction Date": ["2023-10-01", "2023-10-15"],
            "Filing Date": ["2023-10-04", "2023-10-18"],
            "Shares": [1000, 500],
        }
    )

    # On 2023-10-02, the transaction has occurred but Form 4 has NOT been published
    # Should return empty
    res_early = _filter_insider_df_by_date(data, "2023-10-02")
    assert len(res_early) == 0

    # On 2023-10-04, Form 4 is published -> 1 row visible
    res_filed = _filter_insider_df_by_date(data, "2023-10-04")
    assert len(res_filed) == 1
    assert res_filed.iloc[0]["Shares"] == 1000


def test_filter_av_insider_by_date():
    raw = {
        "data": [
            {"transaction_date": "2023-10-01", "filing_date": "2023-10-04", "shares": 100},
            {"transaction_date": "2023-10-10", "filing_date": "2023-10-14", "shares": 200},
        ]
    }
    # On 2023-10-02, nothing filed yet
    res_early = _filter_av_insider_by_date(raw, "2023-10-02")
    assert len(res_early["data"]) == 0

    # On 2023-10-05, first transaction filed
    res_filed = _filter_av_insider_by_date(raw, "2023-10-05")
    assert len(res_filed["data"]) == 1
    assert res_filed["data"][0]["shares"] == 100
