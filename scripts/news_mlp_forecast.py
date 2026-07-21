#!/usr/bin/env python3
"""Forecast comparison of Ridge and small-MLP impact scores."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import news_capacity_matched_forecast as capacity
import news_cutoff_safe_forecast as original
from multimodel_volatility_forecast import (
    clean_working,
    date_block_bootstrap,
    fit_linear_variant,
    prediction_metrics,
    qlike,
)


ROOT = Path(__file__).resolve().parent.parent
MLP = ROOT / "outputs" / "news_uncentered_mlp_control"
OUT = ROOT / "outputs" / "news_uncentered_mlp_forecast"


def build_panel() -> pd.DataFrame:
    panel = capacity.build_panel()
    scores = pd.read_parquet(MLP / "all_model_scores.parquet")
    scores["date"] = pd.to_datetime(scores["date"]).dt.tz_localize(None)
    columns = [
        "bge_m3_embedding_mlp_score",
        "qwen3_embedding_8b_mlp_score",
        "llama2_activation_mlp_score",
    ]
    return panel.merge(
        scores[["ticker", "date", *columns]],
        on=["ticker", "date"],
        how="inner",
        validate="one_to_one",
    )


def main() -> None:
    if not (MLP / "all_model_scores.parquet").exists():
        raise FileNotFoundError("run the uncentered MLP evaluation first")
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    results: list[dict] = []
    direct: list[dict] = []
    pair_names = {
        "bge_m3": ("bge_m3_ridge", "bge_m3_mlp"),
        "qwen3_embedding_8b": ("qwen3_ridge", "qwen3_mlp"),
        "llama2_activation": ("llama2_ridge", "llama2_mlp"),
    }
    for horizon in (1, 5):
        target = f"parkinson_target_h{horizon}"
        for forecast_model, base_features in original.model_specs(panel, horizon).items():
            variants = {
                "baseline": base_features,
                "bge_m3_ridge": [*base_features, "embedding_score"],
                "bge_m3_mlp": [*base_features, "bge_m3_embedding_mlp_score"],
                "qwen3_ridge": [*base_features, "qwen3_embedding_score"],
                "qwen3_mlp": [*base_features, "qwen3_embedding_8b_mlp_score"],
                "llama2_ridge": [*base_features, "activation_score"],
                "llama2_mlp": [*base_features, "llama2_activation_mlp_score"],
            }
            required = list(dict.fromkeys(feature for values in variants.values() for feature in values))
            working = clean_working(panel, target, required)
            test = working["split"].eq("test")
            predictions = {
                name: fit_linear_variant(working, target, features)[0]
                for name, features in variants.items()
            }
            actual = working.loc[test, target].to_numpy(float)
            losses = {name: qlike(actual, prediction) for name, prediction in predictions.items()}
            baseline = float(losses["baseline"].mean())
            for name, prediction in predictions.items():
                metrics = prediction_metrics(actual, prediction)
                inference = (
                    {}
                    if name == "baseline"
                    else date_block_bootstrap(
                        working.loc[test, "date"],
                        losses["baseline"] - losses[name],
                        block_length=horizon,
                        repetitions=2000,
                    )
                )
                results.append(
                    {
                        "horizon": horizon,
                        "forecast_model": forecast_model,
                        "variant": name,
                        "n_fit": int((~test).sum()),
                        "n_test": int(test.sum()),
                        "qlike": metrics["qlike"],
                        "qlike_improvement_vs_baseline_pct": float(
                            100 * (baseline - losses[name].mean()) / baseline
                        ),
                        **{key: value for key, value in metrics.items() if key != "qlike"},
                        **inference,
                    }
                )
            for representation, (ridge_name, mlp_name) in pair_names.items():
                inference = date_block_bootstrap(
                    working.loc[test, "date"],
                    losses[ridge_name] - losses[mlp_name],
                    block_length=horizon,
                    repetitions=2000,
                )
                direct.append(
                    {
                        "horizon": horizon,
                        "forecast_model": forecast_model,
                        "comparison": f"{representation}_mlp_vs_ridge",
                        "mlp_qlike_reduction_vs_ridge_pct": float(
                            100
                            * (losses[ridge_name].mean() - losses[mlp_name].mean())
                            / losses[ridge_name].mean()
                        ),
                        **inference,
                    }
                )
            inference = date_block_bootstrap(
                working.loc[test, "date"],
                losses["qwen3_mlp"] - losses["llama2_mlp"],
                block_length=horizon,
                repetitions=2000,
            )
            direct.append(
                {
                    "horizon": horizon,
                    "forecast_model": forecast_model,
                    "comparison": "llama2_activation_mlp_vs_qwen3_embedding_8b_mlp",
                    "mlp_qlike_reduction_vs_ridge_pct": float(
                        100
                        * (losses["qwen3_mlp"].mean() - losses["llama2_mlp"].mean())
                        / losses["qwen3_mlp"].mean()
                    ),
                    **inference,
                }
            )
            print(f"MLP forecast H{horizon} {forecast_model} complete", flush=True)
    result_frame = pd.DataFrame(results)
    direct_frame = pd.DataFrame(direct)
    result_frame.to_csv(OUT / "forecast_results.csv", index=False)
    direct_frame.to_csv(OUT / "paired_qlike_bootstrap.csv", index=False)
    (OUT / "metadata.json").write_text(
        json.dumps(
            {
                "centering": "none",
                "test": "2023 exploratory",
                "forecast_fit": "train+validation panel OLS with ticker fixed effects",
                "paired_inference": "date-block bootstrap, 2000 repetitions",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(result_frame.to_string(index=False), flush=True)
    print(direct_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
