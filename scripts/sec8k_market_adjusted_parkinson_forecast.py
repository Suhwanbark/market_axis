#!/usr/bin/env python3
"""Forecast Parkinson variance with market-adjusted 8-K text scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
from sec8k_simple_label_family_benchmark import load_frame
from sec8k_simple_gk_label_benchmark import fit_score, normalize_rows


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "sec8k_market_adjusted_parkinson_forecast"
LABEL = "parkinson_market_adjusted_expansion"
ACTIVATION_LAYER = 27
RIDGE_ALPHA = 10.0


def model_specs(panel: pd.DataFrame, horizon: int) -> dict[str, list[str]]:
    estimator = "parkinson"
    ar = [f"{estimator}_log_lag_{lag}" for lag in range(1, 6)]
    k, theta, feature = select_midas(panel, estimator, horizon)
    return {
        "AR(5)": ar,
        "HAR": har_features("sec8k", estimator, False),
        "HAR-X": har_features("sec8k", estimator, True),
        f"MIDAS(k={k},theta={theta:g})": [feature],
    }


def build_event_scores() -> tuple[pd.DataFrame, dict]:
    frame, activations, embeddings = load_frame()
    train = frame["split"].eq("train").to_numpy()
    validation = frame["split"].eq("val").to_numpy()
    test = frame["split"].eq("test").to_numpy()
    label = frame[LABEL].to_numpy(float)

    embedding_score, _ = fit_score(embeddings, label, train)
    activation_x = normalize_rows(
        activations[
            frame["activation_row"].to_numpy(int), ACTIVATION_LAYER - 1
        ].astype(np.float32)
    )
    activation_score, _ = fit_score(activation_x, label, train)
    frame = frame[["ticker", "event_date", "split", LABEL]].copy()
    frame["embedding_score"] = embedding_score
    frame["activation_score"] = activation_score

    diagnostics = {
        "label": LABEL,
        "activation_layer": ACTIVATION_LAYER,
        "ridge_alpha": RIDGE_ALPHA,
        "document_level": {
            "embedding_validation_spearman": float(
                spearmanr(embedding_score[validation], label[validation]).statistic
            ),
            "activation_validation_spearman": float(
                spearmanr(activation_score[validation], label[validation]).statistic
            ),
            "embedding_test_spearman": float(
                spearmanr(embedding_score[test], label[test]).statistic
            ),
            "activation_test_spearman": float(
                spearmanr(activation_score[test], label[test]).statistic
            ),
        },
    }

    # Match the established 8-K event pipeline when multiple documents map to
    # the same ticker and reaction session.
    events = (
        frame.groupby(["ticker", "event_date", "split"], as_index=False)
        .agg(
            embedding_score=("embedding_score", "max"),
            activation_score=("activation_score", "max"),
            document_count=(LABEL, "size"),
        )
        .rename(columns={"event_date": "date"})
    )
    return events, diagnostics


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    event_scores, metadata = build_event_scores()
    event_scores.to_parquet(OUT / "event_scores.parquet", index=False)

    panel = prepare_panel("sec8k", "parkinson").drop(columns=["axis_score"])
    panel = panel.merge(
        event_scores[
            ["ticker", "date", "embedding_score", "activation_score"]
        ],
        on=["ticker", "date"],
        how="inner",
        validate="one_to_one",
    )
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
                "embedding": [*base_features, "embedding_score"],
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
                        "raw_mse": metrics["raw_mse"],
                        "raw_r2": metrics["raw_r2"],
                        "log_mse": metrics["log_mse"],
                        "log_r2": metrics["log_r2"],
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
    metadata.update(
        {
            "forecast_fit": "panel OLS on train+validation with ticker fixed effects",
            "event_score_aggregation": "maximum document score per ticker/reaction session",
            "comparison": ["baseline", "embedding", "activation"],
            "test_split": "2025",
            "panel_counts": panel.groupby("split").size().to_dict(),
            "gpu_used": False,
        }
    )
    (OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        results[
            [
                "horizon",
                "forecast_model",
                "variant",
                "n_test",
                "qlike",
                "qlike_improvement_vs_baseline_pct",
                "raw_mse",
                "raw_r2",
                "qlike_gain_one_sided_p",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
