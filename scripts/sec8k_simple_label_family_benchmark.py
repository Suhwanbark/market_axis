#!/usr/bin/env python3
"""CPU-only 8-K robustness across standard volatility estimators."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sec8k_simple_gk_label_benchmark import (
    ACTIVATIONS,
    DOCS,
    EMBEDDINGS,
    OPENS,
    PRICES,
    RATIO_DOCS,
    fit_score,
    normalize_rows,
    pearson,
    spearman,
)


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "sec8k_simple_label_family_benchmark"
LABELS = [
    "parkinson_market_adjusted_expansion",
    "garman_klass_market_adjusted_expansion",
    "rogers_satchell_market_adjusted_expansion",
    "ohlc_mean_market_adjusted_expansion",
    "close_market_adjusted_expansion",
]
EPS = 1e-12


def build_labels() -> pd.DataFrame:
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
    prices["parkinson"] = log_hl.pow(2) / (4.0 * math.log(2.0))
    prices["garman_klass"] = (
        0.5 * log_hl.pow(2) - (2.0 * math.log(2.0) - 1.0) * log_co.pow(2)
    ).clip(lower=0.0)
    prices["rogers_satchell"] = (
        np.log(prices["high"] / prices["close"])
        * np.log(prices["high"] / prices["open"])
        + np.log(prices["low"] / prices["close"])
        * np.log(prices["low"] / prices["open"])
    ).clip(lower=0.0)
    prices["ohlc_mean"] = prices[["parkinson", "garman_klass", "rogers_satchell"]].mean(axis=1)
    prices["return"] = prices.groupby("ticker", sort=False)["close"].transform(lambda x: np.log(x).diff())
    prices["close"] = prices["return"].pow(2)

    result = prices[["ticker", "date"]].copy()
    for estimator in ["parkinson", "garman_klass", "rogers_satchell", "ohlc_mean", "close"]:
        group = prices.groupby("ticker", sort=False)[estimator]
        firm_pre = group.transform(lambda x: x.shift(1).rolling(20, min_periods=20).mean())
        firm_post = pd.concat(
            [group.shift(-offset) for offset in range(0, 5)], axis=1
        ).mean(axis=1, skipna=False)
        daily = prices.groupby("date")[estimator].mean().sort_index()
        market_pre = daily.shift(1).rolling(20, min_periods=20).mean()
        market_post = pd.concat(
            [daily.shift(-offset) for offset in range(0, 5)], axis=1
        ).mean(axis=1, skipna=False)
        market = pd.DataFrame(
            {"market_pre": market_pre, "market_post": market_post}
        ).reset_index()
        aligned = prices[["date"]].merge(market, on="date", how="left", validate="many_to_one")
        result[f"{estimator}_market_adjusted_expansion"] = 0.5 * (
            np.log(firm_post.clip(lower=EPS))
            - np.log(firm_pre.clip(lower=EPS))
            - np.log(aligned["market_post"].clip(lower=EPS))
            + np.log(aligned["market_pre"].clip(lower=EPS))
        )
    return result[["ticker", "date", *LABELS]]


def load_frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    docs = pd.read_parquet(DOCS)
    docs["activation_row"] = np.arange(len(docs))
    docs["event_date"] = pd.to_datetime(docs["event_date"]).dt.tz_localize(None)
    ratio = pd.read_parquet(RATIO_DOCS)
    ratio = ratio.loc[ratio["domain"].eq("filing"), ["domain_item_id", "benchmark_row"]]
    frame = docs.merge(ratio, on="domain_item_id", validate="one_to_one")
    frame = frame.merge(
        build_labels(), left_on=["ticker", "event_date"], right_on=["ticker", "date"], validate="many_to_one"
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=LABELS).reset_index(drop=True)
    activations = np.load(ACTIVATIONS, mmap_mode="r")
    vectors = np.load(EMBEDDINGS, mmap_mode="r")
    embeddings = normalize_rows(np.asarray(vectors[frame["benchmark_row"].to_numpy(int)], dtype=np.float32))
    return frame, activations, embeddings


def ticker_fe(frame: pd.DataFrame, score: np.ndarray, label: str, correlation: str) -> float:
    work = frame[["ticker", label]].copy()
    work["score"] = score
    x = work["score"] - work.groupby("ticker")["score"].transform("mean")
    y = work[label] - work.groupby("ticker")[label].transform("mean")
    return pearson(x.to_numpy(), y.to_numpy()) if correlation == "pearson" else spearman(x.to_numpy(), y.to_numpy())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame, activations, embeddings = load_frame()
    train = frame["split"].eq("train").to_numpy()
    validation = frame["split"].eq("val").to_numpy()
    test_mask = frame["split"].eq("test").to_numpy()
    activation_rows = frame["activation_row"].to_numpy(int)

    embedding_scores = {}
    for label in LABELS:
        embedding_scores[label], _ = fit_score(embeddings, frame[label].to_numpy(float), train)
    scores = {}
    validation_rows = []
    for layer in range(1, 29):
        x = normalize_rows(activations[activation_rows, layer - 1].astype(np.float32))
        for label in LABELS:
            score, _ = fit_score(x, frame[label].to_numpy(float), train)
            scores[(label, layer)] = score
            validation_rows.append(
                {
                    "label": label,
                    "layer": layer,
                    "activation_validation_spearman": spearman(
                        score[validation], frame.loc[validation, label].to_numpy()
                    ),
                }
            )
    sweep = pd.DataFrame(validation_rows)
    rows = []
    for label in LABELS:
        part = sweep.loc[sweep["label"].eq(label)]
        best_layer = int(part.sort_values(["activation_validation_spearman", "layer"], ascending=[False, True]).iloc[0]["layer"])
        test = frame.loc[test_mask].reset_index(drop=True)
        y = test[label].to_numpy(float)
        for method, score in [
            (f"activation_l{best_layer}", scores[(label, best_layer)][test_mask]),
            ("bge_m3_embedding", embedding_scores[label][test_mask]),
        ]:
            rows.append(
                {
                    "label": label,
                    "validation_selected_layer": best_layer,
                    "method": method,
                    "test_spearman": spearman(score, y),
                    "test_pearson": pearson(score, y),
                    "ticker_fe_spearman": ticker_fe(test, score, label, "spearman"),
                    "ticker_fe_pearson": ticker_fe(test, score, label, "pearson"),
                }
            )
    results = pd.DataFrame(rows)
    sweep.to_csv(OUT / "validation_layer_sweep.csv", index=False)
    results.to_csv(OUT / "exploratory_test_metrics.csv", index=False)
    summary = {"gpu_used": False, "counts": frame.groupby("split").size().to_dict()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
