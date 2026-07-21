#!/usr/bin/env python3
"""Forecast comparison using validation-tuned BGE and activation axes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from multimodel_volatility_forecast import (
    clean_working,
    date_block_bootstrap,
    fit_linear_variant,
    har_features,
    prediction_metrics,
    prepare_panel,
    qlike,
    select_midas,
)


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "news_ticker_centered_parkinson_forecast_tuned"
SCORES = (
    ROOT
    / "outputs"
    / "news_ticker_centered_parkinson_axis_tuned"
    / "all_split_scores.parquet"
)
SELECTION = (
    ROOT
    / "outputs"
    / "news_ticker_centered_parkinson_axis_tuned"
    / "selection.json"
)


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
    OUT.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel("news", "parkinson").drop(columns=["axis_score"])
    scores = pd.read_parquet(SCORES)
    scores["date"] = pd.to_datetime(scores["date"]).dt.tz_localize(None)
    panel = panel.merge(
        scores[["ticker", "date", "activation_score", "embedding_score"]],
        on=["ticker", "date"],
        how="inner",
        validate="one_to_one",
    )
    # Compatibility column required by the shared MIDAS selection helper.
    panel["axis_score"] = panel["activation_score"]

    rows = []
    prediction_frames = []
    for horizon in [1, 5]:
        target = f"parkinson_target_h{horizon}"
        for forecast_model, base_features in model_specs(panel, horizon).items():
            required = [*base_features, "embedding_score", "activation_score"]
            working = clean_working(panel, target, required)
            test = working["split"].eq("test")
            variants = {
                "baseline": base_features,
                "bge": [*base_features, "embedding_score"],
                "activation": [*base_features, "activation_score"],
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
                        "log_mse": metrics["log_mse"],
                        "log_r2": metrics["log_r2"],
                        "raw_r2": metrics["raw_r2"],
                        "spearman": metrics["spearman"],
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

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "forecast_results.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        OUT / "test_predictions.parquet", index=False
    )
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    (OUT / "metadata.json").write_text(
        json.dumps(
            {
                "axis_selection": selection,
                "forecast_fit": "panel OLS on train+validation with ticker fixed effects",
                "comparison": ["baseline", "bge", "activation"],
                "test_split": "2023",
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
                "log_r2",
                "spearman",
                "qlike_gain_one_sided_p",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
