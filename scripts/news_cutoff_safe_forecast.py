#!/usr/bin/env python3
"""Forecast comparison for the frozen cutoff-safe Llama 2 News score."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from multimodel_volatility_forecast import (
    add_lag_features,
    clean_working,
    date_block_bootstrap,
    fit_linear_variant,
    har_features,
    prediction_metrics,
    qlike,
    select_midas,
)
from ohlc_volatility_forecast_comparison import add_estimator_features


ROOT = Path(__file__).resolve().parent.parent
AXIS_ROOT = ROOT / "outputs" / "news_cutoff_safe_llama2"
OUT = ROOT / "outputs" / "news_cutoff_safe_llama2_forecast"
NEWS_ROOT = ROOT / "outputs" / "fintexts_news_axis_calendar_corrected"
NEWS_CACHE = ROOT / "outputs" / "fintexts_news_axis_e2e" / "hf_cache"
SCORES = AXIS_ROOT / "all_split_scores.parquet"
SELECTION = AXIS_ROOT / "selection.json"
MAX_LAG = 80


def build_panel() -> pd.DataFrame:
    snapshot = next(
        (NEWS_CACHE / "datasets--EXAONE-BI--FinTexTS" / "snapshots").iterdir()
    )
    paths = sorted((snapshot / "data").glob("train-*.parquet"))
    raw = pd.concat(
        [
            pd.read_parquet(
                path, columns=["date", "ticker", "open", "high", "low", "close"]
            )
            for path in paths
        ],
        ignore_index=True,
    )
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    audit = json.loads((NEWS_ROOT / "data_audit.json").read_text())
    closure_dates = pd.to_datetime(audit["carried_market_closure_dates"])
    raw = raw.loc[~raw["date"].isin(closure_dates)].copy()

    estimator = "parkinson"
    panel = pd.concat(
        [
            add_estimator_features(group, estimator, "end_of_day", MAX_LAG)
            for _, group in raw.groupby("ticker", sort=False)
        ],
        ignore_index=True,
    )
    har = [f"{estimator}_har_d", f"{estimator}_har_w", f"{estimator}_har_m"]
    for column in har:
        panel[f"market_{column}"] = panel.groupby("date")[column].transform("mean")
    panel = add_lag_features(panel, estimator)

    scores = pd.read_parquet(SCORES)
    scores["date"] = pd.to_datetime(scores["date"]).dt.tz_localize(None)
    panel = panel.merge(
        scores[["ticker", "date", "split", "activation_score", "embedding_score"]],
        on=["ticker", "date"],
        how="inner",
        validate="one_to_one",
    )
    # Shared MIDAS helpers require this compatibility column.
    panel["axis_score"] = panel["activation_score"]
    return panel


def model_specs(panel: pd.DataFrame, horizon: int) -> dict[str, list[str]]:
    estimator = "parkinson"
    ar = [f"{estimator}_log_lag_{lag}" for lag in range(1, 6)]
    k, theta, feature = select_midas(panel, estimator, horizon)
    return {
        "AR(5)": ar,
        "HAR": har_features("news", estimator, False),
        "HAR-X": har_features("news", estimator, True),
        f"MIDAS(k={k},theta={theta:g})": [feature],
    }


def main() -> None:
    if not SELECTION.exists() or not SCORES.exists():
        raise FileNotFoundError("run cutoff-safe select/extract-test/evaluate first")
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    for horizon in (1, 5):
        target = f"parkinson_target_h{horizon}"
        for forecast_model, base_features in model_specs(panel, horizon).items():
            required = [*base_features, "embedding_score", "activation_score"]
            working = clean_working(panel, target, required)
            test = working["split"].eq("test")
            variants = {
                "baseline": base_features,
                "bge": [*base_features, "embedding_score"],
                "llama2_activation": [*base_features, "activation_score"],
            }
            predictions = {
                variant: fit_linear_variant(working, target, features)[0]
                for variant, features in variants.items()
            }
            actual = working.loc[test, target].to_numpy(float)
            base_loss = qlike(actual, predictions["baseline"])
            for variant, prediction in predictions.items():
                metrics = prediction_metrics(actual, prediction)
                loss = qlike(actual, prediction)
                inference = (
                    {}
                    if variant == "baseline"
                    else date_block_bootstrap(
                        working.loc[test, "date"],
                        base_loss - loss,
                        block_length=horizon,
                        repetitions=2000,
                    )
                )
                rows.append(
                    {
                        "horizon": horizon,
                        "forecast_model": forecast_model,
                        "variant": variant,
                        "n_fit": int((~test).sum()),
                        "n_test": int(test.sum()),
                        "qlike": metrics["qlike"],
                        "qlike_improvement_vs_baseline_pct": float(
                            100 * (base_loss.mean() - loss.mean()) / base_loss.mean()
                        ),
                        **{key: value for key, value in metrics.items() if key != "qlike"},
                        **inference,
                    }
                )
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "horizon": horizon,
                            "forecast_model": forecast_model,
                            "variant": variant,
                            "ticker": working.loc[test, "ticker"].to_numpy(),
                            "date": working.loc[test, "date"].to_numpy(),
                            "actual_var": actual,
                            "prediction": prediction,
                            "qlike": loss,
                        }
                    )
                )
            print(f"forecast H{horizon} {forecast_model} complete", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "forecast_results.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        OUT / "test_predictions.parquet", index=False
    )
    selection = json.loads(SELECTION.read_text())
    (OUT / "metadata.json").write_text(
        json.dumps(
            {
                "axis_selection": selection,
                "forecast_fit": "panel OLS on train+validation with ticker fixed effects",
                "comparison": ["baseline", "bge", "llama2_activation"],
                "target": "Parkinson variance",
                "test_split": "2023 exploratory cutoff-safe replication",
                "gpu_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        results[
            [
                "horizon",
                "forecast_model",
                "variant",
                "qlike",
                "qlike_improvement_vs_baseline_pct",
                "raw_r2",
                "spearman",
                "qlike_gain_one_sided_p",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
