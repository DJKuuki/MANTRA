import time
import logging

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from stockstats import wrap
from typing import Annotated, Optional
import os
from .config import get_config

logger = logging.getLogger(__name__)


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Execute a yfinance call with exponential backoff on rate limits.

    yfinance raises YFRateLimitError on HTTP 429 responses but does not
    retry them internally. This wrapper adds retry logic specifically
    for rate limits. Other exceptions propagate immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop invalid rows, fill price gaps."""
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill().bfill()

    return data


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Downloads 15 years of data up to today and caches per symbol. On
    subsequent calls the cache is reused. Rows after curr_date are
    filtered out so backtests never see future prices.
    """
    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)

    # Cache uses a fixed window (15y to today) so one file per symbol
    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=5)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today_date.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{symbol}-YFin-data-{start_str}-{end_str}.csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip")
    else:
        data = yf_retry(lambda: yf.download(
            symbol,
            start=start_str,
            end=end_str,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        ))
        data = data.reset_index()
        data.to_csv(data_file, index=False)

    data = _clean_dataframe(data)

    # Filter to curr_date to prevent look-ahead bias in backtesting
    data = data[data["Date"] <= curr_date_dt]

    return data


def approximate_availability_filter(
    data: pd.DataFrame,
    curr_date: str,
    filing_lag_days: Optional[int] = None,
    freq: str = "quarterly",
) -> pd.DataFrame:
    """Filter financial statement columns by approximate availability date (heuristic fallback).

    WARNING:
        This is a HEURISTIC FALLBACK APPROXIMATION when exact SEC Form 10-Q / 10-K
        publication timestamps are unavailable from the data vendor (e.g. yfinance).
        It does NOT guarantee zero leakage. Formal empirical studies should use
        verified SEC EDGAR filing timestamps via `filing_store.py`.

    Rules:
        - Quarterly filings (Form 10-Q): statutory reporting deadline is 40-45 days.
          Default heuristic fallback lag = 45 calendar days.
        - Annual filings (Form 10-K): statutory reporting deadline is 60-90 days.
          Default heuristic fallback lag = 90 calendar days.
    """
    if not curr_date or data.empty:
        return data
    if filing_lag_days is None:
        filing_lag_days = 90 if freq.lower().startswith("a") else 45

    cutoff = pd.Timestamp(curr_date)
    col_dates = pd.to_datetime(data.columns, errors="coerce")
    avail_dates = col_dates + pd.Timedelta(days=filing_lag_days)
    mask = avail_dates <= cutoff
    return data.loc[:, mask]


def filter_financials_by_date(
    data: pd.DataFrame,
    curr_date: str,
    filing_lag_days: Optional[int] = None,
    freq: str = "quarterly",
) -> pd.DataFrame:
    """Convenience alias for approximate_availability_filter."""
    return approximate_availability_filter(
        data=data, curr_date=curr_date, filing_lag_days=filing_lag_days, freq=freq
    )


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
    ):
        data = load_ohlcv(symbol, curr_date)
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")

        df[indicator]  # trigger stockstats to calculate the indicator
        matching_rows = df[df["Date"].str.startswith(curr_date_str)]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: Not a trading day (weekend or holiday)"
