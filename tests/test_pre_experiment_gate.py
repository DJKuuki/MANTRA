"""Pre-Experiment Gate Integration Tests.

Validates all gate criteria before enabling empirical encoder experiments:
1. No silent toy fallback in FOMCBenchmark()
2. Strict fail-fast on missing required dataset fields
3. Timezone-aware timestamp parsing and true UTC datetime comparison
4. Rejection of naive timestamps without explicit source timezone
5. Full integration test for YFinance annual statements (90-day lag routing in BacktestDataCache)
6. Global uniqueness of sample_id
7. Strict validation of task_label in {-1, 0, 1}
8. PIT provenance tagging and exact-only subset filtering
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from tradingagents.dataflows.backtest_cache import BacktestDataCache
from tradingagents.temporal_leakage import (
    DatasetValidationError,
    FOMCBenchmark,
    ToyFOMCBenchmark,
    TemporalSample,
    load_fomc_dataset,
    parse_iso_utc,
    validate_dataset,
    validate_temporal_sample,
)


def test_gate_1_no_silent_toy_fallback():
    """Gate 1: Instantiating FOMCBenchmark() without explicit samples must raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        FOMCBenchmark()
    err_msg = str(exc_info.value)
    assert "FOMCBenchmark requires an explicit research dataset" in err_msg
    assert "ToyFOMCBenchmark()" in err_msg

    # ToyFOMCBenchmark must continue to work for unit testing
    toy_bench = ToyFOMCBenchmark()
    assert len(toy_bench.samples) == 14
    assert toy_bench.dataset_validation_status == "validated"


def test_gate_1_required_fields_rejection(tmp_path):
    """Gate 1: Loader must fail-fast when required fields are missing, empty, or NaN."""
    base_record = {
        "sample_id": "s-001",
        "text": "The Committee raised interest rates by 25 basis points.",
        "event_time": "2022-03-16T14:00:00-04:00",
        "available_time": "2022-03-16T14:00:00-04:00",
        "task_label": 1,
        "source": "Federal Reserve",
        "annotation_source": "test_corpus",
    }

    # Test missing each required field in turn
    for missing_field in ["sample_id", "text", "event_time", "available_time", "task_label"]:
        bad_record = dict(base_record)
        del bad_record[missing_field]
        p = tmp_path / f"missing_{missing_field}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump([bad_record], f)

        with pytest.raises(DatasetValidationError) as exc_info:
            load_fomc_dataset(p)
        assert f"missing REQUIRED field '{missing_field}'" in str(exc_info.value)

    # Test empty string for text
    bad_record = dict(base_record, text="   ")
    p = tmp_path / "empty_text.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump([bad_record], f)
    with pytest.raises(DatasetValidationError):
        load_fomc_dataset(p)


def test_gate_1_duplicate_sample_id_rejection():
    """Gate 1: Duplicate sample_id across records must raise DatasetValidationError."""
    s1 = TemporalSample(
        sample_id="dup-001",
        text="Inflation remains elevated.",
        event_time="2022-01-26T14:00:00-05:00",
        available_time="2022-01-26T14:00:00-05:00",
        task_label=1,
        source="Federal Reserve",
        annotation_source="test",
    )
    s2 = TemporalSample(
        sample_id="dup-001",  # duplicate ID!
        text="The labor market is strong.",
        event_time="2022-03-16T14:00:00-04:00",
        available_time="2022-03-16T14:00:00-04:00",
        task_label=1,
        source="Federal Reserve",
        annotation_source="test",
    )
    with pytest.raises(DatasetValidationError) as exc_info:
        validate_dataset([s1, s2])
    assert "Duplicate sample_id 'dup-001'" in str(exc_info.value)


def test_gate_1_invalid_task_label_rejection():
    """Gate 1: task_label outside {-1, 0, 1} must raise DatasetValidationError."""
    for invalid_label in [2, -2, 99, 1.5]:
        sample = TemporalSample(
            sample_id="test-label",
            text="Valid text statement.",
            event_time="2022-01-26T14:00:00-05:00",
            available_time="2022-01-26T14:00:00-05:00",
            task_label=invalid_label,
            source="Federal Reserve",
            annotation_source="test",
        )
        with pytest.raises(DatasetValidationError) as exc_info:
            validate_temporal_sample(sample)
        assert "invalid task_label" in str(exc_info.value)


def test_gate_1_naive_timestamp_rejection(tmp_path):
    """Gate 1: Naive ISO timestamp (missing timezone offset) must be rejected by default."""
    naive_record = {
        "sample_id": "naive-001",
        "text": "Meeting held at 2 PM.",
        "event_time": "2022-03-16T14:00:00",  # No timezone!
        "available_time": "2022-03-16T14:00:00",
        "task_label": 0,
        "source": "Federal Reserve",
        "annotation_source": "test",
    }
    p = tmp_path / "naive.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump([naive_record], f)

    with pytest.raises(DatasetValidationError) as exc_info:
        load_fomc_dataset(p)
    assert "naive (missing timezone offset)" in str(exc_info.value)

    # If source_timezone is explicitly provided, it converts correctly
    loaded = load_fomc_dataset(p, source_timezone="America/New_York")
    assert len(loaded) == 1
    # 2022-03-16 is in EDT (-04:00), so 14:00 EDT is 18:00 UTC
    dt = parse_iso_utc(loaded[0].available_time, default_timezone="America/New_York")
    assert dt.hour == 18


def test_gate_1_timezone_aware_availability_comparison():
    """Gate 1: Compare availability across different UTC offsets.

    Example:
    Sample A is released at 2022-01-01T14:00:00-05:00 (EST).
    In UTC, this is 2022-01-01T19:00:00+00:00.
    At simulation time 2022-01-01T18:00:00Z (18:00 UTC):
    - A naive string comparison: "14:00" <= "18:00" -> True (WRONG! Leaks future data!)
    - A proper timezone-aware comparison: 19:00 UTC <= 18:00 UTC -> False (CORRECT!)
    """
    sample = TemporalSample(
        sample_id="est-sample",
        text="Fed statement text.",
        event_time="2022-01-01T14:00:00-05:00",
        available_time="2022-01-01T14:00:00-05:00",  # 19:00 UTC
        task_label=1,
        source="Federal Reserve",
        annotation_source="test",
    )

    # Simulation time: 18:00 UTC (1 hour BEFORE actual availability)
    sim_before = "2022-01-01T18:00:00Z"
    assert not sample.is_available_as_of(sim_before), (
        "14:00-05:00 (19:00 UTC) must NOT be available at 18:00 UTC!"
    )

    # Simulation time: 19:30 UTC (30 min AFTER actual availability)
    sim_after = "2022-01-01T19:30:00Z"
    assert sample.is_available_as_of(sim_after), (
        "14:00-05:00 (19:00 UTC) MUST be available at 19:30 UTC!"
    )


def test_gate_2_annual_yfinance_integration():
    """Gate 2: Integration test verifying 90-day statutory lag on annual statements in BacktestDataCache.

    Ensure that get_yf_balance_sheet(ticker="AAPL", freq="annual", curr_date=...)
    withholds the annual statement on Day 50 and reveals it on Day 95.
    """
    cache = BacktestDataCache()
    cache._active = True
    cache._ticker = "AAPL"

    # Fiscal year ended 2023-12-31
    fiscal_date = "2023-12-31"
    df_annual = pd.DataFrame(
        [[100000]],
        index=["TotalAssets"],
        columns=[fiscal_date],
    )
    cache._store["yf_balance_a"] = df_annual

    # Day 50 post-fiscal end: 2024-02-19 (50 days after 2023-12-31)
    # Under quarterly (45d), this would have leaked!
    # Under annual (90d), it MUST NOT be available!
    res_day50 = cache.get_yf_balance_sheet(ticker="AAPL", freq="annual", curr_date="2024-02-19")
    assert "No balance sheet data found" in res_day50, (
        f"Annual statement should be withheld on day 50, but got: {res_day50}"
    )

    # Day 95 post-fiscal end: 2024-04-04 (95 days after 2023-12-31)
    # Under annual (90d), this MUST now be available!
    res_day95 = cache.get_yf_balance_sheet(ticker="AAPL", freq="annual", curr_date="2024-04-04")
    assert "TotalAssets" in res_day95
    assert "2023-12-31" in res_day95


def test_gate_2_pit_provenance_subset_filtering():
    """Gate 2: Benchmark must differentiate exact PIT records from heuristic approximations."""
    exact_sample = TemporalSample(
        sample_id="exact-001",
        text="Official release statement.",
        event_time="2022-03-16T14:00:00-04:00",
        available_time="2022-03-16T14:00:00-04:00",
        task_label=1,
        source="Federal Reserve",
        annotation_source="official",
        availability_source="FED_OFFICIAL_RELEASE",
        availability_quality="exact",
    )
    heuristic_sample = TemporalSample(
        sample_id="heuristic-001",
        text="Estimated news report.",
        event_time="2022-03-16T14:00:00-04:00",
        available_time="2022-03-16T16:00:00-04:00",
        task_label=0,
        source="External News",
        annotation_source="scraped",
        availability_source="HEURISTIC_ESTIMATE",
        availability_quality="heuristic",
    )

    bench = FOMCBenchmark(samples=[exact_sample, heuristic_sample])
    assert len(bench.samples) == 2

    # Filter to exact only
    exact_subset = bench.get_pit_subset(exact_only=True)
    assert len(exact_subset) == 1
    assert exact_subset[0].sample_id == "exact-001"

    # All subset
    all_subset = bench.get_pit_subset(exact_only=False)
    assert len(all_subset) == 2
