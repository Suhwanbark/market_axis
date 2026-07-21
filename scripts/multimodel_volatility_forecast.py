#!/usr/bin/env python3
"""Evaluate a frozen activation score across volatility forecast families.

The data split, event clock, ticker fixed effects, volatility estimators, and
axis scores are inherited from ``ohlc_volatility_forecast_comparison.py``.
Every baseline is paired with an otherwise identical model that receives one
additional input: the frozen event-level activation score.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
from dataclasses import dataclass
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

from ohlc_volatility_forecast_comparison import (
    EPS,
    ESTIMATORS,
    HORIZONS,
    build_news_panel,
    build_sec_panel,
    date_block_bootstrap,
    qlike,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "outputs" / "multimodel_fixed_axis_forecast"
MAX_LAG = 80
AR_LAGS = 5
LSTM_LAGS = 7
MIDAS_K = (30, 50, 80)
MIDAS_THETA2 = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
SEED = 20260717


def assert_disk_floor(path: Path, floor_gib: float) -> None:
    free_gib = shutil.disk_usage(path).free / 1024**3
    if free_gib < floor_gib:
        raise RuntimeError(
            f"disk floor reached: {free_gib:.2f} GiB free < {floor_gib:.2f} GiB"
        )


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def model_pipeline(features: list[str]) -> Pipeline:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    prep = ColumnTransformer(
        [("numeric", numeric, features), ("ticker", one_hot_encoder(), ["ticker"])]
    )
    return Pipeline([("prep", prep), ("model", LinearRegression())])


def beta_weights(k: int, theta2: float) -> np.ndarray:
    """Normalized one-parameter Beta-polynomial weights used by MIDAS."""
    positions = np.arange(1, k + 1, dtype=float)
    weights = np.power(1.0 - positions / (k + 1.0), theta2 - 1.0)
    return weights / weights.sum()


def add_lag_features(panel: pd.DataFrame, estimator: str) -> pd.DataFrame:
    result = panel.copy()
    raw_columns = [f"{estimator}_var_lag_{lag}" for lag in range(1, MAX_LAG + 1)]
    raw = result[raw_columns].to_numpy(float)
    log_values = np.log(np.clip(raw, EPS, None))
    log_values[~np.isfinite(raw) | (raw <= 0)] = np.nan
    log_columns = [f"{estimator}_log_lag_{lag}" for lag in range(1, MAX_LAG + 1)]
    additions = {
        column: log_values[:, index] for index, column in enumerate(log_columns)
    }
    for k in MIDAS_K:
        for theta2 in MIDAS_THETA2:
            name = midas_feature_name(estimator, k, theta2)
            weighted = raw[:, :k] @ beta_weights(k, theta2)
            invalid = ~np.isfinite(raw[:, :k]).all(axis=1) | (raw[:, :k] <= 0).any(axis=1)
            weighted[invalid] = np.nan
            additions[name] = np.log(np.clip(weighted, EPS, None))
    return pd.concat([result, pd.DataFrame(additions, index=result.index)], axis=1)


def midas_feature_name(estimator: str, k: int, theta2: float) -> str:
    theta = str(theta2).replace(".", "_")
    return f"{estimator}_midas_k{k}_theta{theta}"


def har_features(domain: str, estimator: str, with_x: bool) -> list[str]:
    har = [f"{estimator}_har_d", f"{estimator}_har_w", f"{estimator}_har_m"]
    if not with_x:
        return har
    extra = ["x_abs_return", "x_negative_return", "x_momentum20"]
    if domain == "sec8k":
        extra.append("x_log_volume")
    extra.extend(f"market_{column}" for column in har)
    return har + extra


def prepare_panel(domain: str, estimator: str) -> pd.DataFrame:
    if domain == "news":
        panel = build_news_panel(estimator, lag_count=MAX_LAG)
    elif domain == "sec8k":
        panel = build_sec_panel(estimator, lag_count=MAX_LAG)
    else:
        raise ValueError(domain)
    panel = add_lag_features(panel, estimator)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def prediction_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.maximum(np.asarray(actual, dtype=float), EPS)
    prediction = np.maximum(np.asarray(prediction, dtype=float), EPS)
    actual_log = np.log(actual)
    prediction_log = np.log(prediction)
    return {
        "qlike": float(np.mean(qlike(actual, prediction))),
        "log_mse": float(np.mean((actual_log - prediction_log) ** 2)),
        "log_r2": float(r2_score(actual_log, prediction_log)),
        "raw_mse": float(np.mean((actual - prediction) ** 2)),
        "raw_r2": float(r2_score(actual, prediction)),
        "spearman": float(spearmanr(actual, prediction).statistic),
    }


def paired_result(
    *,
    domain: str,
    estimator: str,
    horizon: int,
    forecast_model: str,
    actual: np.ndarray,
    baseline_prediction: np.ndarray,
    ours_prediction: np.ndarray,
    dates: pd.Series,
    n_fit: int,
    extra: dict | None = None,
) -> dict:
    baseline = prediction_metrics(actual, baseline_prediction)
    ours = prediction_metrics(actual, ours_prediction)
    base_loss = qlike(actual, baseline_prediction)
    ours_loss = qlike(actual, ours_prediction)
    inference = date_block_bootstrap(
        dates, base_loss - ours_loss, block_length=horizon, repetitions=2000
    )
    row = {
        "domain": domain,
        "estimator": estimator,
        "horizon": horizon,
        "forecast_model": forecast_model,
        "n_fit": int(n_fit),
        "n_test": int(len(actual)),
        "baseline_qlike": baseline["qlike"],
        "ours_qlike": ours["qlike"],
        "qlike_improvement_pct": 100.0
        * (baseline["qlike"] - ours["qlike"])
        / max(abs(baseline["qlike"]), EPS),
        **{f"baseline_{key}": value for key, value in baseline.items() if key != "qlike"},
        **{f"ours_{key}": value for key, value in ours.items() if key != "qlike"},
        **inference,
    }
    if extra:
        row.update(extra)
    return row


def clean_working(
    panel: pd.DataFrame, target: str, features: list[str]
) -> pd.DataFrame:
    required = list(
        dict.fromkeys(["ticker", "date", "split", "axis_score", target, *features])
    )
    return (
        panel[required]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[target, "axis_score", *features])
        .copy()
    )


def fit_linear_variant(
    working: pd.DataFrame, target: str, features: list[str]
) -> tuple[np.ndarray, Pipeline]:
    fit = working["split"].isin(["train", "val"])
    test = working["split"].eq("test")
    model = model_pipeline(features)
    model.fit(working.loc[fit], np.log(working.loc[fit, target] + EPS))
    prediction_log = np.clip(model.predict(working.loc[test]), -30.0, 5.0)
    return np.exp(prediction_log), model


def select_midas(
    panel: pd.DataFrame, estimator: str, horizon: int
) -> tuple[int, float, str, float]:
    target = f"{estimator}_target_h{horizon}"
    candidates = []
    for k in MIDAS_K:
        for theta2 in MIDAS_THETA2:
            feature = midas_feature_name(estimator, k, theta2)
            working = clean_working(panel, target, [feature])
            train = working["split"].eq("train")
            val = working["split"].eq("val")
            model = model_pipeline([feature])
            model.fit(working.loc[train], np.log(working.loc[train, target] + EPS))
            prediction = np.exp(np.clip(model.predict(working.loc[val]), -30.0, 5.0))
            loss = float(np.mean(qlike(working.loc[val, target].to_numpy(), prediction)))
            candidates.append((loss, k, theta2, feature))
    return min(candidates, key=lambda row: row[0])[1:]


def evaluate_linear_models(
    panel: pd.DataFrame, domain: str, estimator: str
) -> tuple[list[dict], list[pd.DataFrame]]:
    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    ar = [f"{estimator}_log_lag_{lag}" for lag in range(1, AR_LAGS + 1)]
    for horizon in HORIZONS:
        target = f"{estimator}_target_h{horizon}"
        midas_k, midas_theta2, midas_feature = select_midas(panel, estimator, horizon)
        specs = {
            "AR(5)": (ar, {}),
            "HAR": (har_features(domain, estimator, False), {}),
            "HAR-X": (har_features(domain, estimator, True), {}),
            "MIDAS": (
                [midas_feature],
                {"midas_k": midas_k, "midas_theta2": midas_theta2},
            ),
        }
        for model_name, (features, extra) in specs.items():
            working = clean_working(panel, target, features)
            fit = working["split"].isin(["train", "val"])
            test = working["split"].eq("test")
            baseline_prediction, _ = fit_linear_variant(working, target, features)
            ours_prediction, ours_model = fit_linear_variant(
                working, target, [*features, "axis_score"]
            )
            actual = working.loc[test, target].to_numpy(float)
            axis_index = len(features)
            axis_loading = float(ours_model.named_steps["model"].coef_[axis_index])
            row = paired_result(
                domain=domain,
                estimator=estimator,
                horizon=horizon,
                forecast_model=model_name,
                actual=actual,
                baseline_prediction=baseline_prediction,
                ours_prediction=ours_prediction,
                dates=working.loc[test, "date"],
                n_fit=int(fit.sum()),
                extra={"axis_loading_logvar_per_1sd": axis_loading, **extra},
            )
            rows.append(row)
            predictions.append(
                pd.DataFrame(
                    {
                        "domain": domain,
                        "estimator": estimator,
                        "horizon": horizon,
                        "forecast_model": model_name,
                        "ticker": working.loc[test, "ticker"].to_numpy(),
                        "date": working.loc[test, "date"].to_numpy(),
                        "actual_var": actual,
                        "baseline_prediction": baseline_prediction,
                        "ours_prediction": ours_prediction,
                    }
                )
            )
            print(
                f"{domain} {estimator} h{horizon} {model_name}: "
                f"{row['baseline_qlike']:.4f} -> {row['ours_qlike']:.4f} "
                f"({row['qlike_improvement_pct']:+.2f}%)",
                flush=True,
            )
    return rows, predictions


@dataclass
class SequenceData:
    sequence: np.ndarray
    ticker: np.ndarray
    score: np.ndarray
    target: np.ndarray
    date: np.ndarray
    split: np.ndarray
    ticker_count: int


def make_sequence_data(
    panel: pd.DataFrame, estimator: str, horizon: int
) -> SequenceData:
    sequence_columns = [
        f"{estimator}_log_lag_{lag}" for lag in range(LSTM_LAGS, 0, -1)
    ]
    target = f"{estimator}_target_h{horizon}"
    working = clean_working(panel, target, sequence_columns)
    ticker_values = sorted(working.loc[working["split"].eq("train"), "ticker"].unique())
    ticker_map = {ticker: index + 1 for index, ticker in enumerate(ticker_values)}
    ticker = working["ticker"].map(ticker_map).fillna(0).to_numpy(np.int64)
    return SequenceData(
        sequence=working[sequence_columns].to_numpy(np.float32)[:, :, None],
        ticker=ticker,
        score=working["axis_score"].to_numpy(np.float32)[:, None],
        target=working[target].to_numpy(np.float32),
        date=working["date"].to_numpy(),
        split=working["split"].to_numpy(str),
        ticker_count=len(ticker_map),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fit_lstm_variant(
    data: SequenceData,
    *,
    use_score: bool,
    seed: int,
    device_name: str,
    max_epochs: int,
) -> tuple[np.ndarray, int, float]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(seed)
    device = torch.device(device_name)
    train = data.split == "train"
    val = data.split == "val"
    test = data.split == "test"
    sequence_mean = data.sequence[train].mean()
    sequence_std = max(float(data.sequence[train].std()), 1e-6)
    sequence = (data.sequence - sequence_mean) / sequence_std
    score_mean = data.score[train].mean(axis=0, keepdims=True)
    score_std = np.maximum(data.score[train].std(axis=0, keepdims=True), 1e-6)
    score = (data.score - score_mean) / score_std
    if not use_score:
        score = np.zeros_like(score)
    log_target = np.log(data.target + EPS).astype(np.float32)[:, None]
    target_mean = log_target[train].mean(axis=0, keepdims=True)
    target_std = np.maximum(log_target[train].std(axis=0, keepdims=True), 1e-6)
    standardized_target = (log_target - target_mean) / target_std

    class ForecastLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=16,
                num_layers=3,
                dropout=0.1,
                batch_first=True,
            )
            self.ticker = nn.Embedding(data.ticker_count + 1, 4)
            self.head = nn.Sequential(
                nn.Linear(16 + 4 + 1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x, ticker_index, axis_score):
            _, (hidden, _) = self.lstm(x)
            joined = torch.cat(
                [hidden[-1], self.ticker(ticker_index), axis_score], dim=1
            )
            return self.head(joined)

    def dataset(mask: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(sequence[mask]),
            torch.from_numpy(data.ticker[mask]),
            torch.from_numpy(score[mask]),
            torch.from_numpy(standardized_target[mask]),
        )

    pin_memory = device.type == "cuda"

    def loader(mask: np.ndarray, shuffle: bool, loader_seed: int) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(loader_seed)
        return DataLoader(
            dataset(mask),
            batch_size=2048,
            shuffle=shuffle,
            pin_memory=pin_memory,
            generator=generator,
        )

    def new_model() -> nn.Module:
        set_seed(seed)
        return ForecastLSTM().to(device)

    model = new_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_function = nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    best_qlike = math.inf
    stale = 0
    train_loader = loader(train, True, seed)
    val_loader = loader(val, False, seed)
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, ticker_index, axis_score, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                x.to(device, non_blocking=True),
                ticker_index.to(device, non_blocking=True),
                axis_score.to(device, non_blocking=True),
            )
            loss = loss_function(prediction, y.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
        model.eval()
        values = []
        with torch.inference_mode():
            for x, ticker_index, axis_score, _ in val_loader:
                values.append(
                    model(
                        x.to(device, non_blocking=True),
                        ticker_index.to(device, non_blocking=True),
                        axis_score.to(device, non_blocking=True),
                    ).cpu().numpy()
                )
        standardized = np.concatenate(values, axis=0)
        prediction = np.exp(standardized * target_std + target_mean)[:, 0]
        validation_qlike = float(np.mean(qlike(data.target[val], prediction)))
        if validation_qlike < best_qlike - 1e-5:
            best_qlike = validation_qlike
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 7:
            break

    # Refit on all pre-test observations for the validation-selected epoch.
    fit = train | val
    model = new_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    fit_loader = loader(fit, True, seed + 1)
    for _ in range(best_epoch):
        model.train()
        for x, ticker_index, axis_score, y in fit_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                x.to(device, non_blocking=True),
                ticker_index.to(device, non_blocking=True),
                axis_score.to(device, non_blocking=True),
            )
            loss = loss_function(prediction, y.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()

    model.eval()
    test_loader = loader(test, False, seed)
    values = []
    with torch.inference_mode():
        for x, ticker_index, axis_score, _ in test_loader:
            values.append(
                model(
                    x.to(device, non_blocking=True),
                    ticker_index.to(device, non_blocking=True),
                    axis_score.to(device, non_blocking=True),
                ).cpu().numpy()
            )
    standardized = np.concatenate(values, axis=0)
    prediction = np.exp(standardized * target_std + target_mean)[:, 0]
    return np.maximum(prediction, EPS), best_epoch, best_qlike


def evaluate_lstm(
    panel: pd.DataFrame,
    domain: str,
    estimator: str,
    device: str,
    seeds: list[int],
    max_epochs: int,
) -> tuple[list[dict], list[pd.DataFrame]]:
    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        data = make_sequence_data(panel, estimator, horizon)
        test = data.split == "test"
        fit = np.isin(data.split, ["train", "val"])
        baseline_seeds = []
        ours_seeds = []
        seed_rows = []
        for seed in seeds:
            baseline, baseline_epoch, baseline_val = fit_lstm_variant(
                data,
                use_score=False,
                seed=seed,
                device_name=device,
                max_epochs=max_epochs,
            )
            ours, ours_epoch, ours_val = fit_lstm_variant(
                data,
                use_score=True,
                seed=seed,
                device_name=device,
                max_epochs=max_epochs,
            )
            baseline_seeds.append(baseline)
            ours_seeds.append(ours)
            seed_rows.append(
                {
                    "seed": seed,
                    "baseline_epoch": baseline_epoch,
                    "ours_epoch": ours_epoch,
                    "baseline_validation_qlike": baseline_val,
                    "ours_validation_qlike": ours_val,
                    "baseline_test_qlike": float(
                        np.mean(qlike(data.target[test], baseline))
                    ),
                    "ours_test_qlike": float(np.mean(qlike(data.target[test], ours))),
                }
            )
            print(
                f"{domain} {estimator} h{horizon} LSTM seed={seed}: "
                f"{seed_rows[-1]['baseline_test_qlike']:.4f} -> "
                f"{seed_rows[-1]['ours_test_qlike']:.4f}",
                flush=True,
            )
        baseline_prediction = np.mean(np.stack(baseline_seeds), axis=0)
        ours_prediction = np.mean(np.stack(ours_seeds), axis=0)
        row = paired_result(
            domain=domain,
            estimator=estimator,
            horizon=horizon,
            forecast_model="LSTM",
            actual=data.target[test],
            baseline_prediction=baseline_prediction,
            ours_prediction=ours_prediction,
            dates=pd.Series(data.date[test]),
            n_fit=int(fit.sum()),
            extra={
                "lstm_seeds": json.dumps(seeds),
                "seed_improved_count": int(
                    sum(
                        item["ours_test_qlike"] < item["baseline_test_qlike"]
                        for item in seed_rows
                    )
                ),
                "seed_count": len(seeds),
                "seed_diagnostics": json.dumps(seed_rows),
            },
        )
        rows.append(row)
        predictions.append(
            pd.DataFrame(
                {
                    "domain": domain,
                    "estimator": estimator,
                    "horizon": horizon,
                    "forecast_model": "LSTM",
                    "date": data.date[test],
                    "actual_var": data.target[test],
                    "baseline_prediction": baseline_prediction,
                    "ours_prediction": ours_prediction,
                }
            )
        )
        print(
            f"{domain} {estimator} h{horizon} LSTM ensemble: "
            f"{row['baseline_qlike']:.4f} -> {row['ours_qlike']:.4f} "
            f"({row['qlike_improvement_pct']:+.2f}%)",
            flush=True,
        )
    return rows, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["news", "sec8k", "all"], default="all")
    parser.add_argument("--estimators", nargs="+", choices=ESTIMATORS, default=ESTIMATORS)
    parser.add_argument(
        "--models", nargs="+", choices=["linear", "lstm"], default=["linear", "lstm"]
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--disk-floor-gib", type=float, default=20.0)
    args = parser.parse_args()

    assert_disk_floor(ROOT, args.disk_floor_gib)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    domains = ["news", "sec8k"] if args.domain == "all" else [args.domain]
    rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    for domain in domains:
        for estimator in args.estimators:
            assert_disk_floor(ROOT, args.disk_floor_gib)
            print(f"building {domain} {estimator}", flush=True)
            panel = prepare_panel(domain, estimator)
            if "linear" in args.models:
                new_rows, new_predictions = evaluate_linear_models(
                    panel, domain, estimator
                )
                rows.extend(new_rows)
                prediction_frames.extend(new_predictions)
            if "lstm" in args.models:
                new_rows, new_predictions = evaluate_lstm(
                    panel,
                    domain,
                    estimator,
                    args.device,
                    args.seeds,
                    args.max_epochs,
                )
                rows.extend(new_rows)
                prediction_frames.extend(new_predictions)
            pd.DataFrame(rows).to_csv(
                args.result_dir / f"{args.run_name}_results.partial.csv", index=False
            )
            assert_disk_floor(ROOT, args.disk_floor_gib)

    results = pd.DataFrame(rows)
    results.to_csv(args.result_dir / f"{args.run_name}_results.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_parquet(
            args.result_dir / f"{args.run_name}_predictions.parquet", index=False
        )
    manifest = {
        "run_name": args.run_name,
        "domains": domains,
        "estimators": args.estimators,
        "models": args.models,
        "device": args.device,
        "seeds": args.seeds,
        "max_epochs": args.max_epochs,
        "disk_floor_gib": args.disk_floor_gib,
        "news_axis": "frozen Qwen2.5-7B Ridge L24",
        "sec8k_axis": "frozen Qwen2.5-7B Ridge L28",
        "AR": "five most recent daily log-variance estimates",
        "HAR": "daily, five-day, and 22-day log-variance components",
        "MIDAS": "validation-selected Beta-polynomial lag over k in {30,50,80}",
        "LSTM": "three layers, seven-day sequence, five-seed ensemble, ticker embedding",
    }
    (args.result_dir / f"{args.run_name}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"rows": len(results), "run_name": args.run_name}), flush=True)


if __name__ == "__main__":
    main()
