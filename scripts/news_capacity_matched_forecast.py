#!/usr/bin/env python3
"""Forecast comparison including the capacity-matched Qwen3 8B embedding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import news_cutoff_safe_forecast as original
from multimodel_volatility_forecast import (
    clean_working,
    date_block_bootstrap,
    fit_linear_variant,
    prediction_metrics,
    qlike,
)


ROOT = Path(__file__).resolve().parent.parent
CONTROL = ROOT / "outputs" / "news_qwen3_embedding_8b_control"
UNCENTERED = ROOT / "outputs" / "news_uncentered_linear_control"
OUT = ROOT / "outputs" / "news_capacity_matched_forecast"


def build_panel() -> pd.DataFrame:
    panel = original.build_panel()
    panel = panel.drop(
        columns=["split", "embedding_score", "activation_score"], errors="raise"
    )
    uncentered = pd.read_parquet(UNCENTERED / "all_split_scores.parquet")
    uncentered["date"] = pd.to_datetime(uncentered["date"]).dt.tz_localize(None)
    panel = panel.merge(
        uncentered[["ticker", "date", "split", "embedding_score", "activation_score"]],
        on=["ticker", "date"],
        how="inner",
        validate="one_to_one",
    )
    scores = pd.read_parquet(CONTROL / "all_split_qwen3_scores.parquet")
    scores["date"] = pd.to_datetime(scores["date"]).dt.tz_localize(None)
    panel = panel.merge(
        scores[["ticker", "date", "qwen3_embedding_score"]],
        on=["ticker", "date"],
        how="inner",
        validate="one_to_one",
    )
    return panel


def main() -> None:
    if not (CONTROL / "all_split_qwen3_scores.parquet").exists():
        raise FileNotFoundError("run the Qwen3 capacity-control evaluation first")
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    rows: list[dict] = []
    direct_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    for horizon in (1, 5):
        target = f"parkinson_target_h{horizon}"
        for forecast_model, base_features in original.model_specs(panel, horizon).items():
            required = [
                *base_features,
                "embedding_score",
                "qwen3_embedding_score",
                "activation_score",
            ]
            working = clean_working(panel, target, required)
            test = working["split"].eq("test")
            variants = {
                "baseline": base_features,
                "bge_m3_embedding": [*base_features, "embedding_score"],
                "qwen3_embedding_8b": [*base_features, "qwen3_embedding_score"],
                "llama2_activation": [*base_features, "activation_score"],
            }
            predictions = {
                variant: fit_linear_variant(working, target, features)[0]
                for variant, features in variants.items()
            }
            actual = working.loc[test, target].to_numpy(float)
            losses = {variant: qlike(actual, prediction) for variant, prediction in predictions.items()}
            base_mean = float(losses["baseline"].mean())
            for variant, prediction in predictions.items():
                metrics = prediction_metrics(actual, prediction)
                inference = (
                    {}
                    if variant == "baseline"
                    else date_block_bootstrap(
                        working.loc[test, "date"],
                        losses["baseline"] - losses[variant],
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
                            100 * (base_mean - losses[variant].mean()) / base_mean
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
                            "qlike": losses[variant],
                        }
                    )
                )
            qwen_mean = float(losses["qwen3_embedding_8b"].mean())
            direct_inference = date_block_bootstrap(
                working.loc[test, "date"],
                losses["qwen3_embedding_8b"] - losses["llama2_activation"],
                block_length=horizon,
                repetitions=2000,
            )
            direct_rows.append(
                {
                    "horizon": horizon,
                    "forecast_model": forecast_model,
                    "comparison": "llama2_activation_vs_qwen3_embedding_8b",
                    "activation_qlike": float(losses["llama2_activation"].mean()),
                    "qwen3_qlike": qwen_mean,
                    "activation_qlike_reduction_vs_qwen3_pct": float(
                        100
                        * (qwen_mean - losses["llama2_activation"].mean())
                        / qwen_mean
                    ),
                    "mean_qlike_gain": float(
                        (losses["qwen3_embedding_8b"] - losses["llama2_activation"]).mean()
                    ),
                    **direct_inference,
                }
            )
            print(f"capacity forecast H{horizon} {forecast_model} complete", flush=True)

    results = pd.DataFrame(rows)
    direct = pd.DataFrame(direct_rows)
    results.to_csv(OUT / "forecast_results.csv", index=False)
    direct.to_csv(OUT / "activation_vs_qwen3_forecast_bootstrap.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        OUT / "test_predictions.parquet", index=False
    )
    selection = json.loads((CONTROL / "selection.json").read_text())
    (OUT / "metadata.json").write_text(
        json.dumps(
            {
                "large_embedding_selection": selection,
                "forecast_fit": "panel OLS on train+validation with ticker fixed effects",
                "variants": list(results["variant"].unique()),
                "target": "Parkinson variance",
                "test_split": "2023 capacity control; Qwen3 is not cutoff-safe",
                "direct_comparison": "paired date-block QLIKE bootstrap, 2000 repetitions",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(results.to_string(index=False), flush=True)
    print(direct.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
