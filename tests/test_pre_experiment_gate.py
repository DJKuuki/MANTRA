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
    parse_task_label,
    validate_dataset,
    validate_temporal_sample,
    load_experiment_config,
    validate_experiment_config,
    validate_benchmark_against_config,
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


def test_gate_task_label_float_rejection(tmp_path):
    """Test A: Floats (1.5, 1.0, -0.5), float strings, and non-integer strings are strictly rejected."""
    # Unit level checks
    for bad_val in [1.5, -0.5, 1.0, "1.0", "-0.5", "hawkish", "1.5", None]:
        with pytest.raises(DatasetValidationError):
            parse_task_label(bad_val)

    # Valid values check
    assert parse_task_label(1) == 1
    assert parse_task_label(0) == 0
    assert parse_task_label(-1) == -1
    assert parse_task_label("1") == 1
    assert parse_task_label("0") == 0
    assert parse_task_label("-1") == -1

    # JSON loading rejection
    json_path = tmp_path / "float_label.json"
    record = {
        "sample_id": "float-01",
        "text": "Rates held constant.",
        "event_time": "2022-03-16T14:00:00-04:00",
        "available_time": "2022-03-16T14:00:00-04:00",
        "task_label": 1.5,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([record], f)

    with pytest.raises(DatasetValidationError) as exc:
        load_fomc_dataset(json_path)
    assert "task_label cannot be float" in str(exc.value)

    # CSV loading rejection
    csv_path = tmp_path / "float_label.csv"
    pd.DataFrame([{
        "sample_id": "float-02",
        "text": "Rates raised.",
        "event_time": "2022-03-16T14:00:00-04:00",
        "available_time": "2022-03-16T14:00:00-04:00",
        "task_label": 1.5,
    }]).to_csv(csv_path, index=False)

    with pytest.raises(DatasetValidationError) as exc:
        load_fomc_dataset(csv_path)
    assert "task_label" in str(exc.value)


def test_gate_task_label_boolean_rejection(tmp_path):
    """Test B: Booleans (bool, np.bool_) are strictly rejected (cannot masquerade as int 1/0)."""
    for bool_val in [True, False]:
        with pytest.raises(DatasetValidationError) as exc:
            parse_task_label(bool_val)
        assert "task_label cannot be boolean" in str(exc.value)

    json_path = tmp_path / "bool_label.json"
    record = {
        "sample_id": "bool-01",
        "text": "Statement text.",
        "event_time": "2022-03-16T14:00:00-04:00",
        "available_time": "2022-03-16T14:00:00-04:00",
        "task_label": True,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([record], f)

    with pytest.raises(DatasetValidationError) as exc:
        load_fomc_dataset(json_path)
    assert "task_label cannot be boolean" in str(exc.value)


def test_gate_external_missing_provenance_defaults(tmp_path):
    """Test C: External datasets lacking PIT provenance default to UNVERIFIED and unknown (NEVER exact)."""
    record = {
        "sample_id": "unverified-01",
        "text": "Third party text statement.",
        "event_time": "2022-03-16T14:00:00-04:00",
        "available_time": "2022-03-16T14:00:00-04:00",
        "task_label": 0,
    }
    json_path = tmp_path / "unverified.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([record], f)

    samples = load_fomc_dataset(json_path)
    assert samples[0].availability_source == "UNVERIFIED"
    assert samples[0].availability_quality == "unknown"

    # CSV equivalent
    csv_path = tmp_path / "unverified.csv"
    pd.DataFrame([record]).to_csv(csv_path, index=False)
    csv_samples = load_fomc_dataset(csv_path)
    assert csv_samples[0].availability_source == "UNVERIFIED"
    assert csv_samples[0].availability_quality == "unknown"


def test_gate_provenance_quality_validation():
    """Test: Only exact, heuristic, and unknown are permissible availability_quality values."""
    base_sample = TemporalSample(
        sample_id="qual-01",
        text="Valid statement.",
        event_time="2022-03-16T14:00:00-04:00",
        available_time="2022-03-16T14:00:00-04:00",
        task_label=1,
        source="Federal Reserve",
        annotation_source="test",
        availability_source="TEST_SRC",
        availability_quality="invalid_quality_str",
    )
    with pytest.raises(DatasetValidationError) as exc:
        validate_temporal_sample(base_sample)
    assert "invalid availability_quality" in str(exc.value)

    # Valid values must pass
    for q in ["exact", "heuristic", "unknown"]:
        base_sample.availability_quality = q
        validate_temporal_sample(base_sample)


def test_gate_is_formal_research_ready_contract():
    """Test: validated != verified. Benchmark requires audit flags and exact samples to be formal ready."""
    toy = ToyFOMCBenchmark()
    # Toy has dataset_validation_status == 'validated', but is NOT formal research ready
    assert toy.dataset_validation_status == "validated"
    assert not toy.is_formal_research_ready()

    sample = TemporalSample(
        sample_id="exact-ready-01",
        text="Audited text.",
        event_time="2022-03-16T14:00:00-04:00",
        available_time="2022-03-16T14:00:00-04:00",
        task_label=1,
        source="Federal Reserve",
        annotation_source="test",
        availability_source="FED_OFFICIAL_RELEASE",
        availability_quality="exact",
    )
    # Not verified flags
    bench = FOMCBenchmark(
        samples=[sample],
        source_verified=False,
        annotation_verified=False,
        pit_verified=False,
    )
    assert not bench.is_formal_research_ready()

    # Verified flags and exact quality
    bench_ready = FOMCBenchmark(
        samples=[sample],
        source_verified=True,
        annotation_verified=True,
        pit_verified=True,
    )
    assert bench_ready.is_formal_research_ready()

    # If any sample is heuristic or unknown, it fails formal research ready
    sample_heur = TemporalSample(
        sample_id="heur-02",
        text="Audited text 2.",
        event_time="2022-03-16T14:00:00-04:00",
        available_time="2022-03-16T14:00:00-04:00",
        task_label=0,
        source="Federal Reserve",
        annotation_source="test",
        availability_source="ESTIMATE",
        availability_quality="heuristic",
    )
    bench_mixed = FOMCBenchmark(
        samples=[sample, sample_heur],
        source_verified=True,
        annotation_verified=True,
        pit_verified=True,
    )
    assert not bench_mixed.is_formal_research_ready()


def test_gate_experiment_config_runtime_contract(tmp_path):
    """Test D & Contract: Formal configuration rejects unverified benchmarks, toy benchmarks, and non-exact qualities."""
    formal_cfg = load_experiment_config("configs/fomc_formal_experiment.yaml")
    ci_cfg = load_experiment_config("configs/fomc_ci.yaml")

    assert formal_cfg["experiment_name"] == "fomc_formal_experiment"
    assert ci_cfg["experiment_name"] == "fomc_ci_smoke_test"

    # Formal config strictly requires exact only
    assert formal_cfg["pit"]["allowed_availability_qualities"] == ["exact"]

    toy_bench = ToyFOMCBenchmark()
    # Reject ToyFOMCBenchmark under formal config
    with pytest.raises(DatasetValidationError) as exc:
        validate_benchmark_against_config(toy_bench, formal_cfg)
    assert "ToyFOMCBenchmark cannot be used for formal experiment" in str(exc.value)

    # Benchmark with unknown quality sample rejected under formal config
    unknown_sample = TemporalSample(
        sample_id="unk-01",
        text="Text.",
        event_time="2022-03-16T14:00:00-04:00",
        available_time="2022-03-16T14:00:00-04:00",
        task_label=1,
        source="Federal Reserve",
        annotation_source="test",
        availability_source="UNVERIFIED",
        availability_quality="unknown",
    )
    unverified_bench = FOMCBenchmark(
        samples=[unknown_sample],
        source_verified=True,
        annotation_verified=True,
        pit_verified=True,
    )
    with pytest.raises(DatasetValidationError) as exc:
        validate_benchmark_against_config(unverified_bench, formal_cfg)
    assert "Benchmark fails formal research readiness check" in str(exc.value)


def test_gate_from_file_utc_canonicalization(tmp_path):
    """Test E: from_file(source_timezone=...) canonicalizes naive timestamps to UTC without validation error."""
    record = {
        "sample_id": "canonical-01",
        "text": "Statement with naive timestamp.",
        "event_time": "2022-03-16T14:00:00",
        "available_time": "2022-03-16T14:00:00",
        "task_label": 1,
    }
    p = tmp_path / "naive_timestamps.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump([record], f)

    # Ingesting with source_timezone canonicalizes to UTC ISO string
    bench = FOMCBenchmark.from_file(p, source_timezone="America/New_York")
    sample = bench.samples[0]
    assert sample.available_time == "2022-03-16T18:00:00+00:00"
    assert sample.event_time == "2022-03-16T18:00:00+00:00"


def test_gate_utc_split_boundary_comparison():
    """Test G: Available time in UTC lands in correct split (2018-12-31T23:30:00-05:00 is 2019-01-01T04:30:00Z -> dev)."""
    sample = TemporalSample(
        sample_id="boundary-sample",
        text="New Year statement.",
        event_time="2018-12-31T23:30:00-05:00",
        available_time="2018-12-31T23:30:00-05:00",  # In UTC: 2019-01-01T04:30:00+00:00
        task_label=0,
        source="Federal Reserve",
        annotation_source="test",
        availability_source="FED",
        availability_quality="exact",
    )
    bench = FOMCBenchmark(samples=[sample])

    train_split = bench.get_split("train")
    dev_split = bench.get_split("dev")
    test_split = bench.get_split("test")

    # In absolute UTC time, 2019-01-01T04:30:00Z is after 2018-12-31T23:59:59Z, so it MUST NOT be in train
    assert len(train_split) == 0
    # It must be in dev (between 2019-01-01 and 2019-12-31)
    assert len(dev_split) == 1
    assert dev_split[0].sample_id == "boundary-sample"
    assert len(test_split) == 0


def test_gate_literature_registry_integrity():
    """Test H: Literature registry integrity test (verified entries must have canonical_url and DOI or arXiv)."""
    registry_path = Path("docs/research/literature_registry.json")
    assert registry_path.exists(), f"Registry file not found at {registry_path}"

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    assert isinstance(registry, list)
    verified_count = 0

    for item in registry:
        if item.get("verified") is True:
            verified_count += 1
            assert item.get("title"), "Verified paper missing title"
            assert isinstance(item.get("authors"), list) and len(item["authors"]) > 0, f"Paper '{item['title']}' missing authors"
            assert isinstance(item.get("year"), int), f"Paper '{item['title']}' missing year"

            canonical = item.get("canonical_url")
            assert canonical and canonical.startswith("http"), f"Paper '{item['title']}' missing canonical_url"

            has_id = bool(item.get("doi") or item.get("arxiv_id"))
            assert has_id, f"Verified paper '{item['title']}' missing both doi and arxiv_id"

            if "Trillion Dollar Words" in item["title"]:
                assert item["code_url"] == "https://github.com/gtfintechlab/fomc-hawkish-dovish", (
                    f"Unexpected code_url for Trillion Dollar Words: {item.get('code_url')}"
                )

    assert verified_count >= 9, f"Expected at least 9 verified literature entries, found {verified_count}"
