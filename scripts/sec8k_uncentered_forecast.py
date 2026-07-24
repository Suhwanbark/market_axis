#!/usr/bin/env python3
"""8-K event-session volatility forecasts with all Ridge and MLP scores."""

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
SOURCE = ROOT / "outputs" / "sec8k_uncentered"
MLP = ROOT / "outputs" / "sec8k_uncentered_mlp"
OUT = ROOT / "outputs" / "sec8k_uncentered_forecast"
MAX_LAG = 80
SCORE_COLUMNS = [
    "bge_ridge_score",
    "bge_mlp_score",
    "qwen3_ridge_score",
    "qwen3_mlp_score",
    "llama2_ridge_score",
    "llama2_mlp_score",
]


def event_scores() -> pd.DataFrame:
    scores = pd.read_parquet(MLP / "all_model_scores.parquet")
    scores["event_session"] = pd.to_datetime(scores["event_session"]).dt.tz_localize(None)
    # A ticker can have multiple selected exhibits on the same reaction session.
    # The forecast receives the largest predicted impact available before that open.
    aggregation = {column: "max" for column in SCORE_COLUMNS}
    aggregation["split"] = "first"
    events = (
        scores.groupby(["ticker", "event_session"], as_index=False)
        .agg(aggregation)
        .rename(columns={"event_session": "date"})
    )
    split_counts = scores.groupby(["ticker", "event_session"])["split"].nunique()
    if split_counts.max() != 1:
        raise RuntimeError("a ticker/event session crosses data splits")
    return events


def build_panel() -> pd.DataFrame:
    raw = pd.read_parquet(SOURCE / "adjusted_ohlcv.parquet")
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    tickers = set(event_scores()["ticker"])
    firm = raw.loc[raw["ticker"].isin(tickers)].copy()
    panel = pd.concat(
        [
            add_estimator_features(group, "parkinson", "session_open", MAX_LAG)
            for _, group in firm.groupby("ticker", sort=False)
        ],
        ignore_index=True,
    )
    spy = raw.loc[raw["ticker"].eq("SPY")].copy()
    spy = add_estimator_features(spy, "parkinson", "session_open")
    har = ["parkinson_har_d", "parkinson_har_w", "parkinson_har_m"]
    spy = spy[["date", *har]].rename(
        columns={column: f"market_{column}" for column in har}
    )
    panel = panel.merge(spy, on="date", how="left", validate="many_to_one")
    panel = add_lag_features(panel, "parkinson")
    panel = panel.merge(event_scores(), on=["ticker", "date"], how="inner", validate="one_to_one")
    panel["axis_score"] = panel["llama2_ridge_score"]
    panel.to_parquet(OUT / "event_forecast_panel.parquet", index=False)
    return panel


def model_specs(panel: pd.DataFrame, horizon: int) -> dict[str, list[str]]:
    ar = [f"parkinson_log_lag_{lag}" for lag in range(1, 6)]
    k, theta, feature = select_midas(panel, "parkinson", horizon)
    return {
        "AR(5)": ar,
        "HAR": har_features("sec8k", "parkinson", False),
        "HAR-X": har_features("sec8k", "parkinson", True),
        f"MIDAS(k={k},theta={theta:g})": [feature],
    }


def main() -> None:
    if not (MLP / "all_model_scores.parquet").exists():
        raise FileNotFoundError("run the 8-K MLP evaluation first")
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    results = []
    paired = []
    prediction_frames = []
    pair_names = {
        "bge_mlp_vs_ridge": ("bge_ridge", "bge_mlp"),
        "qwen3_mlp_vs_ridge": ("qwen3_ridge", "qwen3_mlp"),
        "llama2_mlp_vs_ridge": ("llama2_ridge", "llama2_mlp"),
        "llama2_ridge_vs_qwen3_ridge": ("qwen3_ridge", "llama2_ridge"),
        "llama2_mlp_vs_qwen3_mlp": ("qwen3_mlp", "llama2_mlp"),
    }
    for horizon in (1, 5):
        target = f"parkinson_target_h{horizon}"
        for forecast_model, base_features in model_specs(panel, horizon).items():
            variants = {"baseline": base_features}
            for score in SCORE_COLUMNS:
                variants[score.removesuffix("_score")] = [*base_features, score]
            required = list(
                dict.fromkeys(feature for features in variants.values() for feature in features)
            )
            working = clean_working(panel, target, required)
            fit = working["split"].isin(["train", "val"])
            test = working["split"].eq("test")
            predictions = {
                name: fit_linear_variant(working, target, features)[0]
                for name, features in variants.items()
            }
            actual = working.loc[test, target].to_numpy(float)
            losses = {name: qlike(actual, prediction) for name, prediction in predictions.items()}
            base_loss = losses["baseline"]
            for name, prediction in predictions.items():
                metrics = prediction_metrics(actual, prediction)
                inference = (
                    {}
                    if name == "baseline"
                    else date_block_bootstrap(
                        working.loc[test, "date"],
                        base_loss - losses[name],
                        block_length=horizon,
                        repetitions=2000,
                    )
                )
                results.append(
                    {
                        "horizon": horizon,
                        "forecast_model": forecast_model,
                        "variant": name,
                        "n_fit": int(fit.sum()),
                        "n_test": int(test.sum()),
                        "qlike": metrics["qlike"],
                        "qlike_improvement_vs_baseline_pct": float(
                            100 * (base_loss.mean() - losses[name].mean()) / base_loss.mean()
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
                            "variant": name,
                            "ticker": working.loc[test, "ticker"].to_numpy(),
                            "date": working.loc[test, "date"].to_numpy(),
                            "actual_var": actual,
                            "prediction": prediction,
                            "qlike": losses[name],
                        }
                    )
                )
            for comparison, (reference, candidate) in pair_names.items():
                inference = date_block_bootstrap(
                    working.loc[test, "date"],
                    losses[reference] - losses[candidate],
                    block_length=horizon,
                    repetitions=2000,
                )
                paired.append(
                    {
                        "horizon": horizon,
                        "forecast_model": forecast_model,
                        "comparison": comparison,
                        "candidate_qlike_reduction_vs_reference_pct": float(
                            100
                            * (losses[reference].mean() - losses[candidate].mean())
                            / losses[reference].mean()
                        ),
                        **inference,
                    }
                )
            print(
                f"8-K forecast H{horizon} {forecast_model}: "
                f"n_fit={fit.sum()} n_test={test.sum()}",
                flush=True,
            )
    result_frame = pd.DataFrame(results)
    paired_frame = pd.DataFrame(paired)
    result_frame.to_csv(OUT / "forecast_results.csv", index=False)
    paired_frame.to_csv(OUT / "paired_qlike_bootstrap.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        OUT / "test_predictions.parquet", index=False
    )
    (OUT / "metadata.json").write_text(
        json.dumps(
            {
                "forecast_origin": "reaction-session open",
                "history": "strictly before event session",
                "targets": "event session h1 or event through next four sessions h5",
                "scores": SCORE_COLUMNS,
                "duplicate_document_rule": "maximum predicted score per ticker/event session and score type",
                "fit": "train+validation panel OLS with ticker fixed effects",
                "test": "2025 event sessions",
                "market_HAR_X": "SPY",
                "centering": "no ticker centering in impact labels or scores",
                "inference": "date-block bootstrap; 2000 repetitions",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        result_frame[
            [
                "horizon",
                "forecast_model",
                "variant",
                "n_test",
                "qlike",
                "qlike_improvement_vs_baseline_pct",
                "qlike_gain_one_sided_p",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(paired_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
