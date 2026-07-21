#!/usr/bin/env python3
"""Fail fast when a replay diverges from the recorded main outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def close(actual: float, expected: float, tolerance: float = 1e-5) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"expected {expected}, observed {actual}")


def main() -> None:
    news_root = ROOT / "outputs/news_ticker_centered_parkinson_axis_tuned"
    selection = json.loads((news_root / "selection.json").read_text(encoding="utf-8"))
    assert selection["activation_selected_layer"] == 20
    close(float(selection["activation_selected_alpha"]), 1.0)
    close(float(selection["bge_selected_alpha"]), 10.0)

    news = pd.read_csv(news_root / "test_metrics.csv").set_index("method")
    close(float(news.loc["qwen25_activation", "test_spearman"]), 0.15000084)
    close(float(news.loc["bge_m3_embedding", "test_spearman"]), 0.09050058)

    sec = json.loads(
        (
            ROOT / "outputs/sec8k_market_adjusted_parkinson_forecast/metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert sec["activation_layer"] == 27
    assert sec["panel_counts"] == {"train": 399, "val": 625, "test": 453}
    print("MAIN REPLAY SANITY PASS")


if __name__ == "__main__":
    main()
