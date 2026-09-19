"""Unit tests verifying Point-in-Time integrity and pipeline anti-leakage guards."""

import pandas as pd
import pytest
from tradingagents.dataflows.alpha_vantage_fundamentals import _filter_reports_by_date as av_filter_reports
from tradingagents.dataflows.backtest_cache import (
    BacktestDataCache,
    _filter_insider_df_by_date,
    _filter_av_insider_by_date,
)
from tradingagents.dataflows.stockstats_utils import (
    approximate_availability_filter,
    filter_financials_by_date,
)


def test_filter_financials_distinguishes_quarterly_and_annual_lags():
    """Verify 10-Q uses 45-day lag and 10-K uses 90-day lag under approximate_availability_filter."""
    cols = ["2023-12-31"]
    df = pd.DataFrame([[100]], index=["Revenue"], columns=cols)

    # 1. Quarterly test: 2023-12-31 + 45 days = 2024-02-14
    # On 2024-02-01 (32 days post-period), quarterly is NOT visible
    q_early = approximate_availability_filter(df, "2024-02-01", freq="quarterly")
    assert "2023-12-31" not in q_early.columns
    # On 2024-02-20 (51 days post-period), quarterly IS visible
    q_avail = approximate_availability_filter(df, "2024-02-20", freq="quarterly")
    assert "2023-12-31" in q_avail.columns

    # 2. Annual test (Form 10-K): requires 90 days lag!
    # On 2024-02-20 (51 days post-period), annual statement is NOT visible!
    a_early = approximate_availability_filter(df, "2024-02-20", freq="annual")
    assert "2023-12-31" not in a_early.columns
    # On 2024-04-05 (96 days post-period), annual statement IS visible
    a_avail = approximate_availability_filter(df, "2024-04-05", freq="annual")
    assert "2023-12-31" in a_avail.columns


def test_alpha_vantage_reported_date_filtering():
    """Verify Alpha Vantage reports prioritize reportedDate over fiscalDateEnding."""
    raw = {
        "quarterlyReports": [
            {"fiscalDateEnding": "2024-03-31", "reportedDate": "2024-04-25", "totalRevenue": "1000"},
            {"fiscalDateEnding": "2024-06-30", "reportedDate": "2024-07-28", "totalRevenue": "1200"},
        ],
        "annualReports": [
            {"fiscalDateEnding": "2023-12-31", "totalRevenue": "5000"},  # Missing reportedDate -> falls back to +90d
        ],
    }

    # On 2024-04-05 (before reportedDate 2024-04-25): Q1 2024 must NOT be available!
    filtered_early = av_filter_reports(dict(raw), "2024-04-05")
    assert len(filtered_early["quarterlyReports"]) == 0

    # On 2024-04-26 (after reportedDate 2024-04-25): Q1 2024 IS available!
    filtered_avail = av_filter_reports(dict(raw), "2024-04-26")
    assert len(filtered_avail["quarterlyReports"]) == 1
    assert filtered_avail["quarterlyReports"][0]["fiscalDateEnding"] == "2024-03-31"

    # Annual report (2023-12-31) has no reportedDate -> fallback requires 90 days (2024-03-30)
    # On 2024-02-15: annual is NOT visible
    assert len(av_filter_reports(dict(raw), "2024-02-15")["annualReports"]) == 0
    # On 2024-04-01: annual IS visible
    assert len(av_filter_reports(dict(raw), "2024-04-01")["annualReports"]) == 1


def test_get_yf_fundamentals_withheld_in_backtest():
    cache = BacktestDataCache()
    cache._ticker = "AAPL"
    cache._store["yf_info"] = {
        "longName": "Apple Inc.",
        "marketCap": 3000000000000,
        "trailingPE": 30.5,
    }

    # In backtest mode (curr_date provided), fundamentals overview is strictly withheld
    res_mid = cache.get_yf_fundamentals("AAPL", curr_date="2023-06-15")
    assert "[Backtest] Fundamentals overview withheld" in res_mid

    res_last = cache.get_yf_fundamentals("AAPL", curr_date="2023-12-29")
    assert "[Backtest] Fundamentals overview withheld" in res_last

    # In live mode (curr_date=""), live info is returned
    res_live = cache.get_yf_fundamentals("AAPL", curr_date="")
    assert "Company Fundamentals for AAPL" in res_live


def test_filter_insider_df_prioritizes_filing_date():
    data = pd.DataFrame(
        {
            "Transaction Date": ["2023-10-01", "2023-10-15"],
            "Filing Date": ["2023-10-04", "2023-10-18"],
            "Shares": [1000, 500],
        }
    )
    # On 2023-10-02, Form 4 has not been filed yet -> empty
    assert len(_filter_insider_df_by_date(data, "2023-10-02")) == 0
    # On 2023-10-04, Form 4 is published -> visible
    assert len(_filter_insider_df_by_date(data, "2023-10-04")) == 1


def test_filter_av_insider_by_date():
    raw = {
        "data": [
            {"transaction_date": "2023-10-01", "filing_date": "2023-10-04", "shares": 100},
            {"transaction_date": "2023-10-10", "filing_date": "2023-10-14", "shares": 200},
        ]
    }
    assert len(_filter_av_insider_by_date(raw, "2023-10-02")["data"]) == 0
    assert len(_filter_av_insider_by_date(raw, "2023-10-05")["data"]) == 1
