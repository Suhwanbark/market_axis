#!/usr/bin/env python3
"""CPU-only comparison of simple, economically interpretable news-impact labels."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "simple_impact_label_benchmark_pre_event"
ACTIVATIONS = ROOT / "outputs" / "fintexts_news_axis_e2e" / "news_layers1-28.npy"
ROW_MAP = ROOT / "outputs" / "layer_axis_robustness" / "qwen25_current_news" / "strategy_scores.parquet"
NEWS_RECORDS = ROOT / "outputs" / "fintexts_news_axis_calendar_corrected" / "news_records.parquet"
NEWS_AUDIT = ROOT / "outputs" / "fintexts_news_axis_calendar_corrected" / "data_audit.json"
HF_CACHE = ROOT / "outputs" / "fintexts_news_axis_e2e" / "hf_cache"
RATIO_ROOT = ROOT / "outputs" / "impact_ratio_representation_benchmark"
RIDGE_ALPHA = 10.0
EPS = 1e-12
SEED = 20260720

MAIN_LABELS = [
    "parkinson_market_adjusted_expansion",
    "garman_klass_market_adjusted_expansion",
    "rogers_satchell_market_adjusted_expansion",
    "ohlc_mean_market_adjusted_expansion",
    "close_market_adjusted_expansion",
]
DIAGNOSTIC_LABELS = [
    "parkinson_firm_only_expansion",
    "scaled_abs_abnormal_return_1d",
]


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def fit_score(values: np.ndarray, target: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", tol=1e-5)
    model.fit(values[train], target[train])
    score = model.predict(values).astype(np.float32)
    axis = model.coef_.astype(np.float32)
    axis /= max(float(np.linalg.norm(axis)), 1e-8)
    return score, axis


def load_prices() -> pd.DataFrame:
    snapshots = HF_CACHE / "datasets--EXAONE-BI--FinTexTS" / "snapshots"
    snapshot = next(snapshots.iterdir())
    paths = sorted((snapshot / "data").glob("train-*.parquet"))
    columns = ["date", "ticker", "open", "high", "low", "close"]
    prices = pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    audit = json.loads(NEWS_AUDIT.read_text(encoding="utf-8"))
    closures = set(pd.to_datetime(audit["carried_market_closure_dates"]))
    prices = prices.loc[~prices["date"].isin(closures)].copy()
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
    prices["close_var"] = prices["return"].pow(2)
    return prices


def add_windows(prices: pd.DataFrame, estimator: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = prices.groupby("ticker", sort=False)[estimator]
    pre = group.transform(lambda x: x.shift(1).rolling(20, min_periods=20).mean())
    post = pd.concat([group.shift(-offset) for offset in range(1, 6)], axis=1).mean(axis=1, skipna=False)
    firm = prices[["ticker", "date"]].copy()
    firm[f"{estimator}_firm_pre20"] = pre
    firm[f"{estimator}_firm_post5"] = post

    daily = prices.groupby("date", as_index=True)[estimator].mean().sort_index()
    market = pd.DataFrame(index=daily.index)
    market[f"{estimator}_market_pre20"] = daily.shift(1).rolling(20, min_periods=20).mean()
    market[f"{estimator}_market_post5"] = pd.concat(
        [daily.shift(-offset) for offset in range(1, 6)], axis=1
    ).mean(axis=1, skipna=False)
    return firm, market.reset_index()


def build_labels() -> pd.DataFrame:
    prices = load_prices()
    frame = prices[["ticker", "date"]].copy()
    estimators = ["parkinson", "garman_klass", "rogers_satchell", "ohlc_mean", "close_var"]
    for estimator in estimators:
        firm, market = add_windows(prices, estimator)
        frame = frame.merge(firm, on=["ticker", "date"], validate="one_to_one")
        frame = frame.merge(market, on="date", validate="many_to_one")
        firm_log_expansion = 0.5 * (
            np.log(frame[f"{estimator}_firm_post5"].clip(lower=EPS))
            - np.log(frame[f"{estimator}_firm_pre20"].clip(lower=EPS))
        )
        market_log_expansion = 0.5 * (
            np.log(frame[f"{estimator}_market_post5"].clip(lower=EPS))
            - np.log(frame[f"{estimator}_market_pre20"].clip(lower=EPS))
        )
        public_name = "close" if estimator == "close_var" else estimator
        frame[f"{public_name}_market_adjusted_expansion"] = firm_log_expansion - market_log_expansion
        if estimator == "parkinson":
            frame["parkinson_firm_only_expansion"] = firm_log_expansion

    market_return = prices.groupby("date")["return"].mean().rename("market_return")
    prices = prices.join(market_return, on="date")
    group = prices.groupby("ticker", sort=False)
    next_abnormal = group["return"].shift(-1) - group["market_return"].shift(-1)
    pre_close_var = group["close_var"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=20).mean()
    )
    abnormal = prices[["ticker", "date"]].copy()
    abnormal["scaled_abs_abnormal_return_1d"] = (
        np.log(next_abnormal.abs().clip(lower=EPS))
        - 0.5 * np.log(pre_close_var.clip(lower=EPS))
    )
    frame = frame.merge(abnormal, on=["ticker", "date"], validate="one_to_one")
    return frame[["ticker", "date", *MAIN_LABELS, *DIAGNOSTIC_LABELS]]


def load_documents() -> tuple[pd.DataFrame, np.ndarray]:
    records = pd.read_parquet(NEWS_RECORDS, columns=["ticker", "date", "split", "text_hash"])
    records["date"] = pd.to_datetime(records["date"]).dt.tz_localize(None)
    rows = pd.read_parquet(ROW_MAP, columns=["array_row", "ticker", "date"])
    rows["date"] = pd.to_datetime(rows["date"]).dt.tz_localize(None)
    documents = pd.read_parquet(RATIO_ROOT / "documents.parquet")
    documents = documents.loc[documents["domain"].eq("news"), ["ticker", "event_date", "benchmark_row"]]
    documents["event_date"] = pd.to_datetime(documents["event_date"]).dt.tz_localize(None)
    frame = records.merge(rows, on=["ticker", "date"], validate="one_to_one")
    frame = frame.merge(
        documents,
        left_on=["ticker", "date"],
        right_on=["ticker", "event_date"],
        validate="one_to_one",
    )
    frame = frame.merge(build_labels(), on=["ticker", "date"], validate="one_to_one")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[*MAIN_LABELS, *DIAGNOSTIC_LABELS])
    vectors = np.load(RATIO_ROOT / "embedding_vectors.npy", mmap_mode="r")
    embeddings = np.asarray(vectors[frame["benchmark_row"].to_numpy(int)], dtype=np.float32)
    embeddings = normalize_rows(embeddings)
    return frame.reset_index(drop=True), embeddings


def within_ticker_rho(frame: pd.DataFrame, score: np.ndarray, label: str) -> float:
    work = frame[["ticker", label]].copy()
    work["score"] = score
    values = [
        safe_spearman(part["score"].to_numpy(), part[label].to_numpy())
        for _, part in work.groupby("ticker")
        if len(part) >= 5
    ]
    return float(np.nanmean(values))


def paired_bootstrap(frame: pd.DataFrame, ours: np.ndarray, baseline: np.ndarray, label: str, block: str) -> dict:
    target = frame[label].to_numpy(float)
    observed = safe_spearman(ours, target) - safe_spearman(baseline, target)
    groups = [part.index.to_numpy() for _, part in frame.groupby(block)]
    rng = np.random.default_rng(SEED + (0 if block == "date" else 1))
    draws = []
    for _ in range(2000):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append(
            safe_spearman(ours[indices], target[indices])
            - safe_spearman(baseline[indices], target[indices])
        )
    draws = np.asarray(draws)
    return {
        "block": block,
        "difference": observed,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "one_sided_p_ours_not_better": float(np.mean(draws <= 0)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame, embeddings = load_documents()
    labels = [*MAIN_LABELS, *DIAGNOSTIC_LABELS]
    train = frame["split"].eq("train").to_numpy()
    validation = frame["split"].eq("val").to_numpy()
    test_mask = frame["split"].eq("test").to_numpy()

    embedding_scores = {}
    embedding_axes = {}
    for label in labels:
        score, axis = fit_score(embeddings, frame[label].to_numpy(float), train)
        embedding_scores[label] = score
        embedding_axes[label] = axis

    activations = np.load(ACTIVATIONS, mmap_mode="r")
    array_rows = frame["array_row"].to_numpy(int)
    layer_records = []
    activation_scores = {}
    activation_axes = {}
    for layer in range(1, 29):
        values = normalize_rows(activations[array_rows, layer - 1].astype(np.float32))
        for label in labels:
            score, axis = fit_score(values, frame[label].to_numpy(float), train)
            activation_scores[(label, layer)] = score
            activation_axes[(label, layer)] = axis
            layer_records.append(
                {
                    "label": label,
                    "layer": layer,
                    "train_rho": safe_spearman(score[train], frame.loc[train, label].to_numpy()),
                    "validation_rho": safe_spearman(score[validation], frame.loc[validation, label].to_numpy()),
                }
            )
        del values

    sweep = pd.DataFrame(layer_records)
    selection_rows = []
    for label in labels:
        part = sweep.loc[sweep["label"].eq(label)].sort_values(
            ["validation_rho", "layer"], ascending=[False, True]
        )
        best = part.iloc[0]
        embedding_val = safe_spearman(
            embedding_scores[label][validation], frame.loc[validation, label].to_numpy()
        )
        selection_rows.append(
            {
                "label": label,
                "eligible_main": label in MAIN_LABELS,
                "best_layer": int(best["layer"]),
                "activation_validation_rho": float(best["validation_rho"]),
                "embedding_validation_rho": embedding_val,
                "validation_difference": float(best["validation_rho"] - embedding_val),
                "l24_validation_rho": float(
                    part.loc[part["layer"].eq(24), "validation_rho"].iloc[0]
                ),
            }
        )
    selection = pd.DataFrame(selection_rows)
    selected = selection.loc[selection["eligible_main"]].sort_values(
        ["activation_validation_rho", "label"], ascending=[False, True]
    ).iloc[0]
    selected_label = str(selected["label"])
    selected_layer = int(selected["best_layer"])

    test = frame.loc[test_mask].copy().reset_index(drop=True)
    y_test = test[selected_label].to_numpy(float)
    selected_score = activation_scores[(selected_label, selected_layer)][test_mask]
    l24_score = activation_scores[(selected_label, 24)][test_mask]
    embedding_score = embedding_scores[selected_label][test_mask]
    metrics = pd.DataFrame(
        [
            {
                "method": f"activation_l{selected_layer}",
                "test_rho": safe_spearman(selected_score, y_test),
                "within_ticker_rho": within_ticker_rho(test, selected_score, selected_label),
            },
            {
                "method": "activation_l24",
                "test_rho": safe_spearman(l24_score, y_test),
                "within_ticker_rho": within_ticker_rho(test, l24_score, selected_label),
            },
            {
                "method": "bge_m3_embedding",
                "test_rho": safe_spearman(embedding_score, y_test),
                "within_ticker_rho": within_ticker_rho(test, embedding_score, selected_label),
            },
        ]
    )
    bootstrap = pd.DataFrame(
        [paired_bootstrap(test, selected_score, embedding_score, selected_label, block) for block in ["date", "ticker"]]
    )

    broad = selection.sort_values(
        ["activation_validation_rho", "label"], ascending=[False, True]
    ).iloc[0]
    broad_label = str(broad["label"])
    broad_layer = int(broad["best_layer"])
    broad_y = test[broad_label].to_numpy(float)
    broad_activation = activation_scores[(broad_label, broad_layer)][test_mask]
    broad_l24 = activation_scores[(broad_label, 24)][test_mask]
    broad_embedding = embedding_scores[broad_label][test_mask]
    broad_metrics = pd.DataFrame(
        [
            {
                "method": f"activation_l{broad_layer}",
                "test_rho": safe_spearman(broad_activation, broad_y),
                "within_ticker_rho": within_ticker_rho(test, broad_activation, broad_label),
            },
            {
                "method": "activation_l24",
                "test_rho": safe_spearman(broad_l24, broad_y),
                "within_ticker_rho": within_ticker_rho(test, broad_l24, broad_label),
            },
            {
                "method": "bge_m3_embedding",
                "test_rho": safe_spearman(broad_embedding, broad_y),
                "within_ticker_rho": within_ticker_rho(test, broad_embedding, broad_label),
            },
        ]
    )
    broad_bootstrap = pd.DataFrame(
        [
            paired_bootstrap(test, broad_activation, broad_embedding, broad_label, block)
            for block in ["date", "ticker"]
        ]
    )

    exploratory_rows = []
    for selection_row in selection.itertuples(index=False):
        label = str(selection_row.label)
        layer = int(selection_row.best_layer)
        target_values = test[label].to_numpy(float)
        activation_values = activation_scores[(label, layer)][test_mask]
        embedding_values = embedding_scores[label][test_mask]
        for method, values in [
            (f"activation_l{layer}", activation_values),
            ("bge_m3_embedding", embedding_values),
        ]:
            residual_frame = test[["ticker", label]].copy()
            residual_frame["score"] = values
            residual_score = residual_frame["score"] - residual_frame.groupby("ticker")["score"].transform("mean")
            residual_target = residual_frame[label] - residual_frame.groupby("ticker")[label].transform("mean")
            exploratory_rows.append(
                {
                    "label": label,
                    "eligible_main": bool(selection_row.eligible_main),
                    "validation_selected_layer": layer,
                    "method": method,
                    "test_spearman": safe_spearman(values, target_values),
                    "test_pearson": float(pearsonr(values, target_values).statistic),
                    "ticker_fe_spearman": safe_spearman(
                        residual_score.to_numpy(), residual_target.to_numpy()
                    ),
                    "ticker_fe_pearson": float(
                        pearsonr(residual_score, residual_target).statistic
                    ),
                }
            )
    exploratory = pd.DataFrame(exploratory_rows)

    sweep.to_csv(OUT / "validation_layer_sweep.csv", index=False)
    selection.to_csv(OUT / "validation_label_selection.csv", index=False)
    metrics.to_csv(OUT / "selected_label_test_metrics.csv", index=False)
    bootstrap.to_csv(OUT / "selected_label_bootstrap.csv", index=False)
    broad_metrics.to_csv(OUT / "broad_impact_test_metrics.csv", index=False)
    broad_bootstrap.to_csv(OUT / "broad_impact_bootstrap.csv", index=False)
    exploratory.to_csv(OUT / "exploratory_all_label_test_metrics.csv", index=False)
    np.save(OUT / "selected_activation_axis.npy", activation_axes[(selected_label, selected_layer)])
    np.save(OUT / "selected_embedding_axis.npy", embedding_axes[selected_label])
    output_scores = test[["ticker", "date", selected_label]].copy()
    output_scores[f"activation_l{selected_layer}"] = selected_score
    output_scores["activation_l24"] = l24_score
    output_scores["embedding_score"] = embedding_score
    output_scores.to_parquet(OUT / "selected_label_test_scores.parquet", index=False)
    summary = {
        "gpu_used": False,
        "selection_rule": "highest activation validation Spearman among the five market-adjusted expansion labels; embedding advantage is not used for selection",
        "selected_label": selected_label,
        "selected_layer": selected_layer,
        "broad_impact_validation_winner": broad_label,
        "broad_impact_layer": broad_layer,
        "counts": frame.groupby("split").size().to_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nVALIDATION LABEL COMPARISON")
    print(selection.sort_values("activation_validation_rho", ascending=False).to_string(index=False))
    print("\nFROZEN TEST")
    print(metrics.to_string(index=False))
    print("\nPAIRED BOOTSTRAP")
    print(bootstrap.to_string(index=False))
    print("\nBROAD IMPACT VALIDATION WINNER: FROZEN TEST")
    print(broad_metrics.to_string(index=False))
    print(broad_bootstrap.to_string(index=False))
    print("\nEXPLORATORY ALL-LABEL TEST AUDIT")
    print(exploratory.to_string(index=False))


if __name__ == "__main__":
    main()
