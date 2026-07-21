#!/usr/bin/env python3
"""Compare fixed activation scores across daily OHLC volatility estimators."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "ohlc_volatility_forecast_comparison"
NEWS_ROOT = ROOT / "outputs" / "fintexts_news_axis_calendar_corrected"
NEWS_CACHE = ROOT / "outputs" / "fintexts_news_axis_e2e" / "hf_cache"
SEC_PANEL = ROOT / "outputs" / "sec8k_harx_activation_forecast" / "daily_panel.parquet"
SEC_PRICES = ROOT / "outputs" / "sec8k_eventtime_impact" / "prices_with_volume.parquet"
SEC_OPEN_CACHE = OUT / "sec8k_adjusted_open.parquet"
EPS = 1e-10
SEED = 20260717
ESTIMATORS = ["close", "parkinson", "garman_klass", "rogers_satchell"]
HORIZONS = [1, 5]


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def variance_series(frame: pd.DataFrame, estimator: str) -> pd.Series:
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)

    if estimator == "close":
        return close.pct_change(fill_method=None).pow(2)

    valid_range = high.gt(0) & low.gt(0) & high.gt(low)
    log_hl = np.log(high / low)
    if estimator == "parkinson":
        value = log_hl.pow(2) / (4.0 * math.log(2.0))
        return value.where(valid_range & value.gt(EPS))

    open_ = frame["open"].astype(float)
    valid_ohlc = valid_range & open_.gt(0) & close.gt(0)
    if estimator == "garman_klass":
        log_co = np.log(close / open_)
        value = 0.5 * log_hl.pow(2) - (2.0 * math.log(2.0) - 1.0) * log_co.pow(2)
    elif estimator == "rogers_satchell":
        value = np.log(high / close) * np.log(high / open_)
        value += np.log(low / close) * np.log(low / open_)
    else:
        raise ValueError(f"unknown estimator: {estimator}")
    return value.where(valid_ohlc & value.gt(EPS))


def forward_mean(values: pd.Series, start: int, horizon: int) -> pd.Series:
    return pd.concat(
        [values.shift(-offset) for offset in range(start, start + horizon)], axis=1
    ).mean(axis=1, skipna=False)


def add_estimator_features(
    frame: pd.DataFrame,
    estimator: str,
    forecast_origin: str,
    lag_count: int = 0,
) -> pd.DataFrame:
    ordered = frame.sort_values("date").copy()
    daily = variance_series(ordered, estimator)
    returns = ordered["close"].astype(float).pct_change(fill_method=None)

    if forecast_origin == "end_of_day":
        history = daily
        target_start = 1
        control_return = returns
        momentum = ordered["close"] / ordered["close"].shift(20) - 1.0
        if "volume" in ordered:
            control_volume = np.log(ordered["volume"].clip(lower=1.0))
        else:
            control_volume = pd.Series(np.nan, index=ordered.index)
    elif forecast_origin == "session_open":
        history = daily.shift(1)
        target_start = 0
        control_return = returns.shift(1)
        momentum = ordered["close"].shift(1) / ordered["close"].shift(21) - 1.0
        if "volume" in ordered:
            control_volume = np.log(ordered["volume"].shift(1).clip(lower=1.0))
        else:
            control_volume = pd.Series(np.nan, index=ordered.index)
    else:
        raise ValueError(f"unknown forecast origin: {forecast_origin}")

    weekly_min = 3 if forecast_origin == "end_of_day" else 5
    monthly_min = 10 if forecast_origin == "end_of_day" else 22
    prefix = estimator
    ordered[f"{prefix}_har_d"] = np.log(history.clip(lower=EPS))
    ordered[f"{prefix}_har_w"] = np.log(
        history.rolling(5, min_periods=weekly_min).mean().clip(lower=EPS)
    )
    ordered[f"{prefix}_har_m"] = np.log(
        history.rolling(22, min_periods=monthly_min).mean().clip(lower=EPS)
    )
    ordered[f"{prefix}_target_h1"] = forward_mean(daily, target_start, 1)
    ordered[f"{prefix}_target_h5"] = forward_mean(daily, target_start, 5)
    ordered["x_abs_return"] = control_return.abs()
    ordered["x_negative_return"] = np.minimum(control_return, 0.0)
    ordered["x_momentum20"] = momentum
    ordered["x_log_volume"] = control_volume
    if lag_count:
        lagged = pd.concat(
            {
                f"{prefix}_var_lag_{lag}": history.shift(lag - 1)
                for lag in range(1, lag_count + 1)
            },
            axis=1,
        )
        ordered = pd.concat([ordered, lagged], axis=1)
    return ordered


def build_news_panel(estimator: str, lag_count: int = 0) -> pd.DataFrame:
    snapshot = next((NEWS_CACHE / "datasets--EXAONE-BI--FinTexTS" / "snapshots").iterdir())
    paths = sorted((snapshot / "data").glob("train-*.parquet"))
    raw = pd.concat(
        [pd.read_parquet(path, columns=["date", "ticker", "open", "high", "low", "close"]) for path in paths],
        ignore_index=True,
    )
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    audit = json.loads((NEWS_ROOT / "data_audit.json").read_text())
    closure_dates = pd.to_datetime(audit["carried_market_closure_dates"])
    raw = raw.loc[~raw["date"].isin(closure_dates)].copy()

    panel = pd.concat(
        [
            add_estimator_features(group, estimator, "end_of_day", lag_count)
            for _, group in raw.groupby("ticker", sort=False)
        ],
        ignore_index=True,
    )
    har = [f"{estimator}_har_d", f"{estimator}_har_w", f"{estimator}_har_m"]
    for column in har:
        panel[f"market_{column}"] = panel.groupby("date")[column].transform("mean")

    scores = pd.read_parquet(NEWS_ROOT / "corrected_selected_scores.parquet")
    scores = scores[["ticker", "date", "split", "axis_score"]].copy()
    scores["date"] = pd.to_datetime(scores["date"])
    return panel.merge(scores, on=["ticker", "date"], how="inner", validate="one_to_one")


def build_sec_panel(estimator: str, lag_count: int = 0) -> pd.DataFrame:
    source = pd.read_parquet(SEC_PANEL)
    source["date"] = pd.to_datetime(source["date"])
    open_prices = load_sec_open_prices()
    source = source.merge(
        open_prices, on=["ticker", "date"], how="left", validate="one_to_one"
    )
    panel = pd.concat(
        [
            add_estimator_features(group, estimator, "session_open", lag_count)
            for _, group in source.groupby("ticker", sort=False)
        ],
        ignore_index=True,
    )

    prices = pd.read_parquet(SEC_PRICES)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.merge(
        open_prices, on=["ticker", "date"], how="left", validate="one_to_one"
    )
    spy = prices.loc[prices["ticker"].eq("SPY")].copy()
    spy = add_estimator_features(spy, estimator, "session_open")
    har = [f"{estimator}_har_d", f"{estimator}_har_w", f"{estimator}_har_m"]
    spy = spy[["date", *har]].rename(columns={column: f"market_{column}" for column in har})
    panel = panel.merge(spy, on="date", how="left", validate="many_to_one")
    panel = panel.loc[panel["event_any"].eq(1)].copy()
    panel["axis_score"] = panel["activation_ridge"]
    return panel


def load_sec_open_prices() -> pd.DataFrame:
    if SEC_OPEN_CACHE.exists():
        cached = pd.read_parquet(SEC_OPEN_CACHE)
        cached["date"] = pd.to_datetime(cached["date"])
        return cached

    import yfinance as yf

    cached_hlc = pd.read_parquet(SEC_PRICES, columns=["date", "ticker", "close"])
    cached_hlc["date"] = pd.to_datetime(cached_hlc["date"])
    tickers = sorted(cached_hlc["ticker"].unique())
    start_date = str(cached_hlc["date"].min().date())
    end_date = str((cached_hlc["date"].max() + pd.Timedelta(days=1)).date())
    frames = []
    for start in range(0, len(tickers), 30):
        batch = tickers[start : start + 30]
        data = yf.download(
            batch,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
        )
        if not isinstance(data.columns, pd.MultiIndex):
            data.columns = pd.MultiIndex.from_product([data.columns, batch])
        for ticker in batch:
            if ("Open", ticker) not in data.columns or ("Close", ticker) not in data.columns:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(data.index).tz_localize(None),
                        "ticker": ticker,
                        "download_open": data[("Open", ticker)].to_numpy(float),
                        "download_close": data[("Close", ticker)].to_numpy(float),
                    }
                ).dropna()
            )
        print(
            f"downloaded SEC open prices: {min(start + len(batch), len(tickers))}/{len(tickers)}",
            flush=True,
        )
    downloaded = pd.concat(frames, ignore_index=True)
    aligned = cached_hlc.merge(
        downloaded, on=["ticker", "date"], how="left", validate="one_to_one"
    )
    aligned["open"] = (
        aligned["download_open"]
        * aligned["close"]
        / aligned["download_close"].replace(0.0, np.nan)
    )
    result = aligned[["ticker", "date", "open"]].dropna().copy()
    result.to_parquet(SEC_OPEN_CACHE, index=False)
    return result


def model_pipeline(features: list[str]) -> Pipeline:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    prep = ColumnTransformer(
        [("numeric", numeric, features), ("ticker", one_hot_encoder(), ["ticker"])]
    )
    return Pipeline([("prep", prep), ("ols", LinearRegression())])


def qlike(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    ratio = np.maximum(actual, EPS) / np.maximum(predicted, EPS)
    return ratio - np.log(ratio) - 1.0


def date_block_bootstrap(
    dates: pd.Series, improvement: np.ndarray, block_length: int, repetitions: int = 2000
) -> dict[str, float]:
    daily = (
        pd.DataFrame({"date": pd.to_datetime(dates).to_numpy(), "value": improvement})
        .groupby("date", as_index=False)["value"]
        .mean()
        .sort_values("date")
    )
    values = daily["value"].to_numpy(float)
    rng = np.random.default_rng(SEED + block_length)
    draws = np.empty(repetitions, dtype=float)
    starts_max = max(len(values) - block_length + 1, 1)
    blocks_needed = int(np.ceil(len(values) / block_length))
    for index in range(repetitions):
        starts = rng.integers(0, starts_max, size=blocks_needed)
        sampled = np.concatenate(
            [values[start : start + block_length] for start in starts]
        )[: len(values)]
        draws[index] = sampled.mean()
    return {
        "qlike_gain_ci_low": float(np.quantile(draws, 0.025)),
        "qlike_gain_ci_high": float(np.quantile(draws, 0.975)),
        "qlike_gain_one_sided_p": float((1 + np.sum(draws <= 0)) / (repetitions + 1)),
        "test_dates": int(len(values)),
    }


def fit_pair(
    panel: pd.DataFrame,
    domain: str,
    estimator: str,
    horizon: int,
    baseline: str,
) -> tuple[dict, pd.DataFrame]:
    har = [f"{estimator}_har_d", f"{estimator}_har_w", f"{estimator}_har_m"]
    if domain == "news":
        extra = [
            "x_abs_return",
            "x_negative_return",
            "x_momentum20",
            *[f"market_{column}" for column in har],
        ]
    else:
        extra = [
            "x_abs_return",
            "x_negative_return",
            "x_momentum20",
            "x_log_volume",
            *[f"market_{column}" for column in har],
        ]
    features = har if baseline == "HAR" else har + extra
    target = f"{estimator}_target_h{horizon}"
    working = panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[target, "axis_score", *features]
    ).copy()
    working["target_log"] = np.log(working[target] + EPS)
    fit = working["split"].isin(["train", "val"])
    test = working["split"].eq("test")
    if not fit.any() or not test.any():
        raise RuntimeError(f"empty split for {domain} {estimator} h{horizon} {baseline}")

    outputs = []
    loading = np.nan
    for variant, model_features in [
        ("baseline", features),
        ("ours", features + ["axis_score"]),
    ]:
        model = model_pipeline(model_features)
        model.fit(working.loc[fit], working.loc[fit, "target_log"])
        prediction_log = model.predict(working.loc[test])
        prediction_var = np.exp(prediction_log)
        outputs.append((variant, prediction_log, prediction_var))
        if variant == "ours":
            loading = float(model.named_steps["ols"].coef_[model_features.index("axis_score")])

    actual = working.loc[test, target].to_numpy(float)
    actual_log = working.loc[test, "target_log"].to_numpy(float)
    _, base_log, base_var = outputs[0]
    _, ours_log, ours_var = outputs[1]
    base_loss = qlike(actual, base_var)
    ours_loss = qlike(actual, ours_var)
    inference = date_block_bootstrap(
        working.loc[test, "date"], base_loss - ours_loss, block_length=horizon
    )
    result = {
        "domain": domain,
        "estimator": estimator,
        "horizon": horizon,
        "baseline": baseline,
        "n_fit": int(fit.sum()),
        "n_test": int(test.sum()),
        "baseline_qlike": float(base_loss.mean()),
        "ours_qlike": float(ours_loss.mean()),
        "qlike_improvement_pct": float(100 * (1 - ours_loss.mean() / base_loss.mean())),
        "baseline_log_mse": float(np.mean((actual_log - base_log) ** 2)),
        "ours_log_mse": float(np.mean((actual_log - ours_log) ** 2)),
        "baseline_log_r2": float(r2_score(actual_log, base_log)),
        "ours_log_r2": float(r2_score(actual_log, ours_log)),
        "baseline_raw_r2": float(r2_score(actual, base_var)),
        "ours_raw_r2": float(r2_score(actual, ours_var)),
        "baseline_raw_rmse": float(np.sqrt(np.mean((actual - base_var) ** 2))),
        "ours_raw_rmse": float(np.sqrt(np.mean((actual - ours_var) ** 2))),
        "baseline_spearman": float(spearmanr(actual, base_var).statistic),
        "ours_spearman": float(spearmanr(actual, ours_var).statistic),
        "axis_loading_logvar_per_1sd": loading,
        **inference,
    }
    predictions = pd.DataFrame(
        {
            "domain": domain,
            "estimator": estimator,
            "horizon": horizon,
            "baseline": baseline,
            "ticker": working.loc[test, "ticker"].to_numpy(),
            "date": working.loc[test, "date"].to_numpy(),
            "actual_var": actual,
            "baseline_prediction": base_var,
            "ours_prediction": ours_var,
            "baseline_qlike": base_loss,
            "ours_qlike": ours_loss,
        }
    )
    return result, predictions


def make_report(results: pd.DataFrame) -> str:
    labels = {
        "close": "Close-to-close",
        "parkinson": "Parkinson",
        "garman_klass": "Garman-Klass",
        "rogers_satchell": "Rogers-Satchell",
    }
    lines = [
        "# OHLC Volatility Forecast Comparison",
        "",
        "The News Ridge L24 and 8-K Ridge L28 axes are frozen. Each volatility estimator has its own HAR/HAR-X regression.",
        "",
    ]
    for domain in ["news", "sec8k"]:
        lines.extend(
            [
                f"## {domain}",
                "",
                "| Estimator | H | Baseline | N | QLIKE baseline -> ours | Improvement | Log-R2 baseline -> ours | Raw R2 baseline -> ours | Spearman baseline -> ours | p |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        subset = results.loc[results["domain"].eq(domain)]
        for row in subset.itertuples(index=False):
            lines.append(
                f"| {labels[row.estimator]} | {row.horizon} | {row.baseline} | {row.n_test:,} "
                f"| {row.baseline_qlike:.4f} -> {row.ours_qlike:.4f} "
                f"| {row.qlike_improvement_pct:.2f}% "
                f"| {row.baseline_log_r2:.3f} -> {row.ours_log_r2:.3f} "
                f"| {row.baseline_raw_r2:.3f} -> {row.ours_raw_r2:.3f} "
                f"| {row.baseline_spearman:.3f} -> {row.ours_spearman:.3f} "
                f"| {row.qlike_gain_one_sided_p:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    prediction_frames = []
    for domain in ["news", "sec8k"]:
        for estimator in ESTIMATORS:
            print(f"building {domain} {estimator}", flush=True)
            panel = build_news_panel(estimator) if domain == "news" else build_sec_panel(estimator)
            for horizon in HORIZONS:
                for baseline in ["HAR", "HAR-X"]:
                    print(f"fitting {domain} {estimator} h{horizon} {baseline}", flush=True)
                    result, predictions = fit_pair(
                        panel, domain, estimator, horizon, baseline
                    )
                    rows.append(result)
                    prediction_frames.append(predictions)

    results = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    results.to_csv(OUT / "forecast_results.csv", index=False)
    predictions.to_parquet(OUT / "test_predictions.parquet", index=False)
    (OUT / "report.md").write_text(make_report(results), encoding="utf-8")
    metadata = {
        "news_axis": "Qwen2.5-7B continuous Ridge alpha=10, layer 24, frozen on 2022 validation",
        "sec8k_axis": "Qwen2.5-7B continuous Ridge alpha=10, layer 28, frozen on 2024 validation",
        "estimators": ESTIMATORS,
        "news_clock": "news date end-of-day; target starts next actual trading session",
        "sec8k_clock": "pre-09:30 ET filing maps to same session; later filing maps to next session",
        "fit": "separate panel OLS with ticker fixed effects for every estimator/horizon/baseline",
        "disk_free_gib_after": float(
            __import__("shutil").disk_usage(ROOT).free / 1024**3
        ),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
