#!/usr/bin/env python3
"""CPU-only 8-K benchmark using the strict market-adjusted Garman-Klass expansion label."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "sec8k_simple_gk_label_benchmark"
DOCS = ROOT / "outputs" / "joint_impact_axis_v1" / "filing_documents.parquet"
ACTIVATIONS = ROOT / "outputs" / "joint_impact_axis_v1" / "activations" / "qwen25" / "filing_layers.npy"
RATIO_DOCS = ROOT / "outputs" / "impact_ratio_representation_benchmark" / "documents.parquet"
EMBEDDINGS = ROOT / "outputs" / "impact_ratio_representation_benchmark" / "embedding_vectors.npy"
PRICES = ROOT / "outputs" / "sec8k_eventtime_impact" / "prices_with_volume.parquet"
OPENS = ROOT / "outputs" / "ohlc_volatility_forecast_comparison" / "sec8k_adjusted_open.parquet"
ALPHA = 10.0
EPS = 1e-12
SEED = 20260720


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(pearsonr(x, y).statistic)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def fit_score(x: np.ndarray, y: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = Ridge(alpha=ALPHA, solver="lsqr", tol=1e-5)
    model.fit(x[train], y[train])
    score = model.predict(x).astype(np.float32)
    axis = model.coef_.astype(np.float32)
    axis /= max(float(np.linalg.norm(axis)), 1e-8)
    return score, axis


def build_label() -> pd.DataFrame:
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "close", "high", "low"])
    opens = pd.read_parquet(OPENS, columns=["date", "ticker", "open"])
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    opens["date"] = pd.to_datetime(opens["date"]).dt.tz_localize(None)
    prices = prices.merge(opens, on=["ticker", "date"], validate="one_to_one")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    valid = (prices[["open", "high", "low", "close"]] > 0).all(axis=1)
    prices = prices.loc[valid].copy()
    log_hl = np.log(prices["high"] / prices["low"])
    log_co = np.log(prices["close"] / prices["open"])
    prices["gk"] = (
        0.5 * log_hl.pow(2) - (2.0 * math.log(2.0) - 1.0) * log_co.pow(2)
    ).clip(lower=0.0)

    group = prices.groupby("ticker", sort=False)["gk"]
    prices["firm_pre20"] = group.transform(
        lambda x: x.shift(1).rolling(20, min_periods=20).mean()
    )
    prices["firm_post5"] = pd.concat(
        [group.shift(-offset) for offset in range(0, 5)], axis=1
    ).mean(axis=1, skipna=False)

    daily = prices.groupby("date")["gk"].mean().sort_index()
    market = pd.DataFrame(index=daily.index)
    market["market_pre20"] = daily.shift(1).rolling(20, min_periods=20).mean()
    market["market_post5"] = pd.concat(
        [daily.shift(-offset) for offset in range(0, 5)], axis=1
    ).mean(axis=1, skipna=False)
    prices = prices.merge(market.reset_index(), on="date", validate="many_to_one")
    prices["label"] = 0.5 * (
        np.log(prices["firm_post5"].clip(lower=EPS))
        - np.log(prices["firm_pre20"].clip(lower=EPS))
        - np.log(prices["market_post5"].clip(lower=EPS))
        + np.log(prices["market_pre20"].clip(lower=EPS))
    )
    return prices[["ticker", "date", "label"]]


def load_frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    docs = pd.read_parquet(DOCS)
    docs["activation_row"] = np.arange(len(docs))
    docs["event_date"] = pd.to_datetime(docs["event_date"]).dt.tz_localize(None)
    ratio = pd.read_parquet(RATIO_DOCS)
    ratio = ratio.loc[ratio["domain"].eq("filing"), ["domain_item_id", "benchmark_row"]]
    frame = docs.merge(ratio, on="domain_item_id", validate="one_to_one")
    frame = frame.merge(
        build_label(), left_on=["ticker", "event_date"], right_on=["ticker", "date"], validate="many_to_one"
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["label"]).reset_index(drop=True)
    acts = np.load(ACTIVATIONS, mmap_mode="r")
    embeddings = np.load(EMBEDDINGS, mmap_mode="r")
    emb = normalize_rows(np.asarray(embeddings[frame["benchmark_row"].to_numpy(int)], dtype=np.float32))
    return frame, acts, emb


def within_ticker(frame: pd.DataFrame, score: np.ndarray, method: str) -> float:
    work = frame[["ticker", "label"]].copy()
    work["score"] = score
    values = []
    for _, part in work.groupby("ticker"):
        if len(part) >= 3:
            values.append(spearman(part["score"].to_numpy(), part["label"].to_numpy()))
    return float(np.nanmean(values))


def block_bootstrap(test: pd.DataFrame, ours: np.ndarray, baseline: np.ndarray, kind: str, block: str) -> dict:
    y = test["label"].to_numpy(float)
    corr = spearman if kind == "spearman" else pearson
    observed = corr(ours, y) - corr(baseline, y)
    groups = [part.index.to_numpy() for _, part in test.groupby(block)]
    rng = np.random.default_rng(SEED + (0 if block == "event_date" else 1) + (10 if kind == "pearson" else 0))
    draws = []
    for _ in range(2000):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append(corr(ours[idx], y[idx]) - corr(baseline[idx], y[idx]))
    draws = np.asarray(draws)
    return {
        "correlation": kind,
        "block": block,
        "difference": observed,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "one_sided_p_ours_not_better": float(np.mean(draws <= 0)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame, activations, embeddings = load_frame()
    y = frame["label"].to_numpy(float)
    train = frame["split"].eq("train").to_numpy()
    validation = frame["split"].eq("val").to_numpy()
    test_mask = frame["split"].eq("test").to_numpy()

    embedding_score, embedding_axis = fit_score(embeddings, y, train)
    rows = []
    scores = {}
    axes = {}
    activation_rows = frame["activation_row"].to_numpy(int)
    for layer in range(1, 29):
        x = normalize_rows(activations[activation_rows, layer - 1].astype(np.float32))
        score, axis = fit_score(x, y, train)
        scores[layer] = score
        axes[layer] = axis
        rows.append(
            {
                "layer": layer,
                "train_spearman": spearman(score[train], y[train]),
                "validation_spearman": spearman(score[validation], y[validation]),
                "validation_pearson": pearson(score[validation], y[validation]),
            }
        )
    sweep = pd.DataFrame(rows)
    selected_layer = int(
        sweep.sort_values(["validation_spearman", "layer"], ascending=[False, True]).iloc[0]["layer"]
    )
    test = frame.loc[test_mask].copy().reset_index(drop=True)
    y_test = test["label"].to_numpy(float)
    y_ratio = np.exp(y_test)
    selected = scores[selected_layer][test_mask]
    l24 = scores[24][test_mask]
    embedding = embedding_score[test_mask]
    metrics = []
    for name, score in [
        (f"activation_l{selected_layer}", selected),
        ("activation_l24", l24),
        ("bge_m3_embedding", embedding),
    ]:
        metrics.append(
            {
                "method": name,
                "test_spearman_log": spearman(score, y_test),
                "test_pearson_log": pearson(score, y_test),
                "test_pearson_ratio": pearson(score, y_ratio),
                "within_ticker_spearman": within_ticker(test, score, name),
            }
        )
    metrics = pd.DataFrame(metrics)
    bootstrap = pd.DataFrame(
        [
            block_bootstrap(test, selected, embedding, kind, block)
            for kind in ["spearman", "pearson"]
            for block in ["event_date", "ticker"]
        ]
    )
    sweep.to_csv(OUT / "validation_layer_sweep.csv", index=False)
    metrics.to_csv(OUT / "test_metrics.csv", index=False)
    bootstrap.to_csv(OUT / "paired_bootstrap.csv", index=False)
    np.save(OUT / "selected_activation_axis.npy", axes[selected_layer])
    np.save(OUT / "embedding_axis.npy", embedding_axis)
    summary = {
        "gpu_used": False,
        "label": "strict market-adjusted Garman-Klass Post5/Pre20 volatility expansion",
        "selected_layer": selected_layer,
        "counts": frame.groupby("split").size().to_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(metrics.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
