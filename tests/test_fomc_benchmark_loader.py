"""Unit tests for formal FOMC benchmark file loading and format parsing."""

import json
from pathlib import Path
import pandas as pd
import pytest
from tradingagents.temporal_leakage.fomc_benchmark import (
    FOMCBenchmark,
    ToyFOMCBenchmark,
    load_fomc_dataset,
)


def test_toy_benchmark_properties():
    bench = ToyFOMCBenchmark()
    assert "TOY" in bench.__doc__
    assert len(bench.samples) == 14

    train = bench.get_split("train")
    dev = bench.get_split("dev")
    test = bench.get_split("test")

    assert len(train) > 0
    assert len(dev) > 0
    assert len(test) > 0


def test_load_fomc_dataset_from_csv(tmp_path):
    csv_file = tmp_path / "fomc_test.csv"
    data = [
        {
            "sample_id": "fomc-001",
            "text": "The Committee raised the target rate by 25 basis points.",
            "document_type": "statement",
            "event_time": "2022-03-16T14:00:00",
            "available_time": "2022-03-16T14:00:00",
            "task_label": 1,
            "meeting_id": "2022-03",
            "source": "FRB",
            "annotation_source": "Trillion Dollar Words",
            "next_action": 1,
            "spy_1d_return": 0.02,
        },
        {
            "sample_id": "fomc-002",
            "text": "The Committee decided to lower rates to zero.",
            "document_type": "statement",
            "event_time": "2020-03-15T17:00:00",
            "available_time": "2020-03-15T17:00:00",
            "task_label": -1,
            "meeting_id": "2020-03",
            "source": "FRB",
            "annotation_source": "Trillion Dollar Words",
            "next_action": 0,
            "spy_1d_return": -0.10,
        },
    ]
    pd.DataFrame(data).to_csv(csv_file, index=False)

    bench = FOMCBenchmark.from_file(csv_file)
    assert len(bench.samples) == 2
    assert bench.samples[0].sample_id == "fomc-001"
    assert bench.samples[0].task_label == 1
    assert bench.samples[0].future_macro_labels["next_action"] == 1
    assert bench.samples[0].market_outcomes["spy_1d_return"] == 0.02


def test_load_fomc_dataset_from_jsonl(tmp_path):
    jsonl_file = tmp_path / "fomc_test.jsonl"
    record = {
        "sample_id": "jsonl-001",
        "text": "Economic activity is expanding at a moderate pace.",
        "document_type": "statement",
        "event_time": "2019-05-01T14:00:00",
        "available_time": "2019-05-01T14:00:00",
        "task_label": 0,
        "future_macro_labels": {"next_action": -1},
        "market_outcomes": {"spy_5d_return": 0.01},
    }
    with open(jsonl_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    bench = FOMCBenchmark.from_file(jsonl_file)
    assert len(bench.samples) == 1
    assert bench.samples[0].sample_id == "jsonl-001"
    assert bench.samples[0].task_label == 0
