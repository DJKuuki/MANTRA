import logging
import pandas as pd
from .alpha_vantage_common import _make_api_request

logger = logging.getLogger(__name__)


def _filter_reports_by_date(result, curr_date: str):
    """Filter annualReports/quarterlyReports to enforce Point-in-Time availability.

    Prioritizes reportedDate / filingDate over fiscalDateEnding.
    If reportedDate is unavailable, applies statutory reporting latency (45 days for quarterly,
    90 days for annual) as a heuristic fallback. Pure fiscalDateEnding <= curr_date creates
    look-ahead bias because financial statements are not public on the quarter end date.
    """
    if not curr_date or not isinstance(result, dict):
        return result
    cutoff = pd.Timestamp(curr_date)
    for key in ("annualReports", "quarterlyReports"):
        if key in result:
            fallback_lag = 90 if key == "annualReports" else 45
            filtered = []
            for r in result[key]:
                avail_str = r.get("reportedDate") or r.get("filingDate") or r.get("filing_date")
                if avail_str:
                    try:
                        if pd.Timestamp(avail_str) <= cutoff:
                            filtered.append(r)
                    except Exception:
                        pass
                else:
                    fiscal_str = r.get("fiscalDateEnding", "")
                    if fiscal_str:
                        try:
                            avail_dt = pd.Timestamp(fiscal_str) + pd.Timedelta(days=fallback_lag)
                            if avail_dt <= cutoff:
                                filtered.append(r)
                        except Exception:
                            pass
            result[key] = filtered
    return result


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol using Alpha Vantage.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (not used for Alpha Vantage)

    Returns:
        str: Company overview data including financial ratios and key metrics
    """
    from .backtest_cache import get_backtest_cache
    _cache = get_backtest_cache()
    if _cache.is_active():
        cached = _cache.get_av_fundamentals(ticker, curr_date or "")
        if cached is not None:
            return cached
    params = {
        "symbol": ticker,
    }

    return _make_api_request("OVERVIEW", params)


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve balance sheet data for a given ticker symbol using Alpha Vantage."""
    from .backtest_cache import get_backtest_cache
    _cache = get_backtest_cache()
    if _cache.is_active():
        cached = _cache.get_av_balance_sheet(ticker, curr_date)
        if cached is not None:
            return cached
    result = _make_api_request("BALANCE_SHEET", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve cash flow statement data for a given ticker symbol using Alpha Vantage."""
    from .backtest_cache import get_backtest_cache
    _cache = get_backtest_cache()
    if _cache.is_active():
        cached = _cache.get_av_cashflow(ticker, curr_date)
        if cached is not None:
            return cached
    result = _make_api_request("CASH_FLOW", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve income statement data for a given ticker symbol using Alpha Vantage."""
    from .backtest_cache import get_backtest_cache
    _cache = get_backtest_cache()
    if _cache.is_active():
        cached = _cache.get_av_income_statement(ticker, curr_date)
        if cached is not None:
            return cached
    result = _make_api_request("INCOME_STATEMENT", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)

