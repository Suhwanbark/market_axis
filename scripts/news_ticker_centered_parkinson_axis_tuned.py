#!/usr/bin/env python3
"""Validation-tuned BGE and Qwen activation axes for normalized Parkinson impact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge

from simple_impact_label_benchmark import ACTIVATIONS, load_documents, normalize_rows


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "news_ticker_centered_parkinson_axis_tuned"
DIRECT_ROOT = ROOT / "outputs" / "impact_ratio_representation_benchmark"
LABEL = "parkinson_market_adjusted_expansion"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    fn = pearsonr if kind == "pearson" else spearmanr
    return float(fn(x, y).statistic)


def ticker_center_matrix(
    values: np.ndarray, frame: pd.DataFrame, train: np.ndarray
) -> np.ndarray:
    centered = values.copy()
    train_frame = pd.DataFrame(values[train])
    train_frame["ticker"] = frame.loc[train, "ticker"].to_numpy()
    means = train_frame.groupby("ticker").mean()
    for ticker, indices in frame.groupby("ticker").groups.items():
        if ticker in means.index:
            centered[np.asarray(list(indices), dtype=int)] -= means.loc[ticker].to_numpy(
                np.float32
            )
    return centered


def ticker_center_vector(
    values: np.ndarray, frame: pd.DataFrame, train: np.ndarray
) -> np.ndarray:
    train_means = (
        pd.Series(values[train], index=frame.index[train])
        .groupby(frame.loc[train, "ticker"])
        .mean()
    )
    fallback = float(np.mean(values[train]))
    return values - frame["ticker"].map(train_means).fillna(fallback).to_numpy(float)


def fit(
    values: np.ndarray, target: np.ndarray, train: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-5)
    model.fit(values[train], target[train])
    score = model.predict(values).astype(np.float32)
    axis = model.coef_.astype(np.float32)
    axis /= max(float(np.linalg.norm(axis)), 1e-8)
    return score, axis


def direct_test_score(frame: pd.DataFrame) -> np.ndarray:
    documents = pd.read_parquet(
        DIRECT_ROOT / "documents.parquet",
        columns=["domain", "ticker", "event_date", "benchmark_row"],
    )
    documents = documents.loc[documents["domain"].eq("news")].copy()
    documents["event_date"] = pd.to_datetime(documents["event_date"]).dt.tz_localize(None)
    direct = pd.read_parquet(
        DIRECT_ROOT / "direct_scores.parquet",
        columns=["benchmark_row", "direct_score"],
    )
    keys = frame[["ticker", "date"]].copy()
    keys["row"] = np.arange(len(keys))
    aligned = keys.merge(
        documents[["ticker", "event_date", "benchmark_row"]],
        left_on=["ticker", "date"],
        right_on=["ticker", "event_date"],
        validate="one_to_one",
    ).merge(direct, on="benchmark_row", validate="one_to_one")
    result = np.full(len(frame), np.nan, dtype=np.float32)
    result[aligned["row"].to_numpy(int)] = aligned["direct_score"].to_numpy(np.float32)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame, embeddings = load_documents()
    train = frame["split"].eq("train").to_numpy()
    validation = frame["split"].eq("val").to_numpy()
    test = frame["split"].eq("test").to_numpy()
    raw_target = frame[LABEL].to_numpy(float)
    target = ticker_center_vector(raw_target, frame, train)

    records: list[dict] = []
    centered_embedding = ticker_center_matrix(embeddings, frame, train)
    bge_candidates: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for alpha in ALPHAS:
        score, axis = fit(centered_embedding, target, train, alpha)
        bge_candidates[alpha] = (score, axis)
        records.append(
            {
                "representation": "bge_m3_embedding",
                "layer": np.nan,
                "alpha": alpha,
                "train_spearman": correlation(score[train], target[train], "spearman"),
                "validation_spearman": correlation(
                    score[validation], target[validation], "spearman"
                ),
                "validation_pearson": correlation(
                    score[validation], target[validation], "pearson"
                ),
            }
        )
    bge_rows = pd.DataFrame(records)
    bge_best = bge_rows.sort_values(
        ["validation_spearman", "alpha"], ascending=[False, True]
    ).iloc[0]
    bge_alpha = float(bge_best["alpha"])
    bge_score, bge_axis = bge_candidates[bge_alpha]
    del centered_embedding, bge_candidates

    activations = np.load(ACTIVATIONS, mmap_mode="r")
    array_rows = frame["array_row"].to_numpy(int)
    activation_records: list[dict] = []
    for layer in range(1, 29):
        values = normalize_rows(
            activations[array_rows, layer - 1].astype(np.float32)
        )
        values = ticker_center_matrix(values, frame, train)
        for alpha in ALPHAS:
            score, _ = fit(values, target, train, alpha)
            activation_records.append(
                {
                    "representation": "qwen25_activation",
                    "layer": layer,
                    "alpha": alpha,
                    "train_spearman": correlation(
                        score[train], target[train], "spearman"
                    ),
                    "validation_spearman": correlation(
                        score[validation], target[validation], "spearman"
                    ),
                    "validation_pearson": correlation(
                        score[validation], target[validation], "pearson"
                    ),
                }
            )
        pd.DataFrame(activation_records).to_csv(
            OUT / "activation_validation_sweep.partial.csv", index=False
        )
        print(f"completed activation layer {layer}/28", flush=True)
        del values

    activation_sweep = pd.DataFrame(activation_records)
    activation_best = activation_sweep.sort_values(
        ["validation_spearman", "layer", "alpha"],
        ascending=[False, True, True],
    ).iloc[0]
    selected_layer = int(activation_best["layer"])
    activation_alpha = float(activation_best["alpha"])
    selected_values = normalize_rows(
        activations[array_rows, selected_layer - 1].astype(np.float32)
    )
    selected_values = ticker_center_matrix(selected_values, frame, train)
    activation_score, activation_axis = fit(
        selected_values, target, train, activation_alpha
    )

    direct_score = direct_test_score(frame)
    metrics = []
    for method, score, layer, alpha in [
        ("direct_llm", direct_score, np.nan, np.nan),
        ("bge_m3_embedding", bge_score, np.nan, bge_alpha),
        ("qwen25_activation", activation_score, selected_layer, activation_alpha),
    ]:
        valid_test = test & np.isfinite(score)
        metrics.append(
            {
                "method": method,
                "selected_layer": layer,
                "selected_alpha": alpha,
                "n_test": int(valid_test.sum()),
                "test_spearman": correlation(
                    score[valid_test], target[valid_test], "spearman"
                ),
                "test_pearson": correlation(
                    score[valid_test], target[valid_test], "pearson"
                ),
            }
        )

    validation_sweep = pd.concat(
        [bge_rows, activation_sweep], ignore_index=True
    )
    validation_sweep.to_csv(OUT / "validation_sweep.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "test_metrics.csv", index=False)
    score_frame = frame[["ticker", "date", "split"]].copy()
    score_frame["centered_impact_label"] = target
    score_frame["activation_score"] = activation_score
    score_frame["embedding_score"] = bge_score
    score_frame["direct_score"] = direct_score
    score_frame.to_parquet(OUT / "all_split_scores.parquet", index=False)
    np.save(OUT / "selected_activation_axis.npy", activation_axis)
    np.save(OUT / "selected_embedding_axis.npy", bge_axis)
    (OUT / "selection.json").write_text(
        json.dumps(
            {
                "label": LABEL,
                "centering": "subtract train-ticker activation/embedding centroid and train-ticker label mean",
                "alpha_grid": list(ALPHAS),
                "bge_selected_alpha": bge_alpha,
                "bge_validation_spearman": float(bge_best["validation_spearman"]),
                "activation_selected_layer": selected_layer,
                "activation_selected_alpha": activation_alpha,
                "activation_validation_spearman": float(
                    activation_best["validation_spearman"]
                ),
                "axis_fit_split": "train only",
                "selection_split": "validation only",
                "test_evaluated_after_selection": True,
                "counts": frame.groupby("split").size().to_dict(),
                "gpu_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "activation_validation_sweep.partial.csv").unlink(missing_ok=True)
    print((OUT / "selection.json").read_text(encoding="utf-8"), flush=True)
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
