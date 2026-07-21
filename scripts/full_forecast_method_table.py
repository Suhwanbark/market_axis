#!/usr/bin/env python3
"""Full News/8-K forecast table across models and four document scores.

Rows cover AR(5), HAR and HAR-X with OLS/Lasso/Ridge, MIDAS, and a five-seed
LSTM.  Every baseline is paired on identical test rows with BGE, Qwen3-8B
embedding, Llama2 activation, and a direct likelihood rating from that same
Llama2 checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import multimodel_volatility_forecast as multi
import news_capacity_matched_forecast as news_source


ROOT = Path(__file__).resolve().parent.parent
DIRECT = ROOT / "outputs" / "direct_llama2_rating" / "all_scores.parquet"
EMBEDDING_LAYERS = ROOT / "outputs" / "embedding_intermediate_layer_control"
SEC_PANEL = ROOT / "outputs" / "sec8k_uncentered_forecast" / "event_forecast_panel.parquet"
OUT = ROOT / "outputs" / "full_forecast_method_table"
SCORES = ("bge", "qwen", "llama", "direct_lm")
SCORE_COLUMNS = {name: f"score_{name}" for name in SCORES}
LASSO_ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
LSTM_SEEDS = (11, 22, 33, 44, 55)


def one_hot_encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=True)


def pipeline(features: list[str], regression: str, alpha: float | None) -> Pipeline:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    prep = ColumnTransformer(
        [("numeric", numeric, features), ("ticker", one_hot_encoder(), ["ticker"])]
    )
    if regression == "ols":
        model = LinearRegression()
    elif regression == "lasso":
        model = Lasso(alpha=float(alpha), max_iter=20_000, tol=1e-5, selection="cyclic")
    elif regression == "ridge":
        model = Ridge(alpha=float(alpha), solver="auto", tol=1e-5)
    else:
        raise ValueError(regression)
    return Pipeline([("prep", prep), ("model", model)])


def fit_predict(
    working: pd.DataFrame,
    target: str,
    features: list[str],
    regression: str,
) -> tuple[np.ndarray, float | None, float | None]:
    train = working["split"].eq("train")
    val = working["split"].eq("val")
    fit = train | val
    test = working["split"].eq("test")
    grid: tuple[float, ...]
    if regression == "lasso":
        grid = LASSO_ALPHAS
    elif regression == "ridge":
        grid = RIDGE_ALPHAS
    else:
        grid = ()
    selected: float | None = None
    validation_qlike: float | None = None
    if grid:
        candidates = []
        for alpha in grid:
            model = pipeline(features, regression, alpha)
            model.fit(working.loc[train], np.log(working.loc[train, target] + multi.EPS))
            prediction = np.exp(np.clip(model.predict(working.loc[val]), -30.0, 5.0))
            candidates.append(
                (float(np.mean(multi.qlike(working.loc[val, target], prediction))), alpha)
            )
        validation_qlike, selected = min(candidates, key=lambda row: (row[0], row[1]))
    model = pipeline(features, regression, selected)
    model.fit(working.loc[fit], np.log(working.loc[fit, target] + multi.EPS))
    prediction = np.exp(np.clip(model.predict(working.loc[test]), -30.0, 5.0))
    return np.maximum(prediction, multi.EPS), selected, validation_qlike


def news_panel() -> pd.DataFrame:
    panel = news_source.build_panel().rename(
        columns={
            "embedding_score": "score_bge",
            "qwen3_embedding_score": "score_qwen",
            "activation_score": "score_llama",
        }
    )
    direct = pd.read_parquet(DIRECT)
    direct = direct.loc[direct["domain"].eq("news")].copy()
    direct["date"] = pd.to_datetime(direct["date"]).dt.tz_localize(None)
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None)
    panel = panel.merge(
        direct[["ticker", "date", "direct_lm_score"]].rename(
            columns={"direct_lm_score": "score_direct_lm"}
        ),
        on=["ticker", "date"],
        how="inner",
        validate="one_to_one",
    )
    # Replace the legacy final-layer embedding controls with layers selected
    # strictly on validation impact correlation.  Test representations were
    # not extracted until these selections had been frozen.
    panel = panel.drop(columns=["score_bge", "score_qwen"])
    for model, source_column, output_column in (
        ("bge", "bge_best_score", "score_bge"),
        ("qwen3", "qwen3_best_score", "score_qwen"),
    ):
        scores = pd.read_parquet(EMBEDDING_LAYERS / f"news_{model}_all_scores.parquet")
        scores["date"] = pd.to_datetime(scores["date"]).dt.tz_localize(None)
        panel = panel.merge(
            scores[["ticker", "date", source_column]].rename(
                columns={source_column: output_column}
            ),
            on=["ticker", "date"],
            how="inner",
            validate="one_to_one",
        )
    panel["axis_score"] = panel["score_llama"]
    return panel


def sec_panel() -> pd.DataFrame:
    panel = pd.read_parquet(SEC_PANEL).rename(
        columns={
            "bge_ridge_score": "score_bge",
            "qwen3_ridge_score": "score_qwen",
            "llama2_ridge_score": "score_llama",
        }
    )
    direct = pd.read_parquet(DIRECT)
    direct = direct.loc[direct["domain"].eq("sec8k")].copy()
    direct["date"] = pd.to_datetime(direct["date"]).dt.tz_localize(None)
    split_counts = direct.groupby(["ticker", "date"])["split"].nunique()
    if split_counts.max() != 1:
        raise RuntimeError("8-K direct-rating event crosses splits")
    direct = (
        direct.groupby(["ticker", "date"], as_index=False)["direct_lm_score"]
        .max()
        .rename(columns={"direct_lm_score": "score_direct_lm"})
    )
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None)
    panel = panel.merge(direct, on=["ticker", "date"], how="inner", validate="one_to_one")
    panel = panel.drop(columns=["score_bge", "score_qwen"])
    for model, source_column, output_column in (
        ("bge", "bge_best_score", "score_bge"),
        ("qwen3", "qwen3_best_score", "score_qwen"),
    ):
        scores = pd.read_parquet(EMBEDDING_LAYERS / f"sec8k_{model}_all_scores.parquet")
        scores["event_session"] = pd.to_datetime(scores["event_session"]).dt.tz_localize(None)
        split_counts = scores.groupby(["ticker", "event_session"])["split"].nunique()
        if split_counts.max() != 1:
            raise RuntimeError(f"8-K {model} layer score event crosses splits")
        scores = (
            scores.groupby(["ticker", "event_session"], as_index=False)[source_column]
            .max()
            .rename(columns={"event_session": "date", source_column: output_column})
        )
        panel = panel.merge(scores, on=["ticker", "date"], how="inner", validate="one_to_one")
    panel["axis_score"] = panel["score_llama"]
    return panel


def model_specs(panel: pd.DataFrame, domain: str, horizon: int):
    ar = [f"parkinson_log_lag_{lag}" for lag in range(1, 6)]
    har = multi.har_features("sec8k" if domain == "sec8k" else "news", "parkinson", False)
    har_x = multi.har_features("sec8k" if domain == "sec8k" else "news", "parkinson", True)
    k, theta, feature = multi.select_midas(panel, "parkinson", horizon)
    return [
        ("AR(5)", "ols", ar, {}),
        ("HAR", "ols", har, {}),
        ("HAR", "lasso", har, {}),
        ("HAR", "ridge", har, {}),
        ("HAR-X", "ols", har_x, {}),
        ("HAR-X", "lasso", har_x, {}),
        ("HAR-X", "ridge", har_x, {}),
        ("MIDAS", "ols", [feature], {"midas_k": k, "midas_theta": theta}),
    ]


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.maximum(np.asarray(prediction, dtype=float), multi.EPS)
    return {
        "qlike": float(np.mean(multi.qlike(actual, prediction))),
        "raw_r2": float(r2_score(actual, prediction)),
        "raw_mse": float(np.mean((actual - prediction) ** 2)),
    }


def result_rows(
    *,
    domain: str,
    horizon: int,
    family: str,
    regression: str,
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    n_fit: int,
    n_test: int,
    alphas: dict[str, float | None],
    validation_qlike: dict[str, float | None],
    extra: dict,
) -> list[dict]:
    base = metrics(actual, predictions["baseline"])
    rows = []
    for variant, prediction in predictions.items():
        observed = metrics(actual, prediction)
        rows.append(
            {
                "domain": domain,
                "horizon": horizon,
                "forecast_family": family,
                "regression": regression,
                "variant": variant,
                "n_fit": n_fit,
                "n_test": n_test,
                **observed,
                "qlike_reduction_vs_baseline_pct": float(
                    100 * (base["qlike"] - observed["qlike"]) / base["qlike"]
                ),
                "raw_r2_change_vs_baseline": observed["raw_r2"] - base["raw_r2"],
                "raw_mse_reduction_vs_baseline_pct": float(
                    100 * (base["raw_mse"] - observed["raw_mse"]) / base["raw_mse"]
                ),
                "selected_alpha": alphas.get(variant),
                "validation_qlike_for_alpha": validation_qlike.get(variant),
                **extra,
            }
        )
    return rows


def evaluate_classical(
    panel: pd.DataFrame, domain: str, horizons: tuple[int, ...]
) -> tuple[list[dict], list[pd.DataFrame]]:
    rows = []
    prediction_frames = []
    for horizon in horizons:
        target = f"parkinson_target_h{horizon}"
        for family, regression, base_features, extra in model_specs(panel, domain, horizon):
            required = [*base_features, *SCORE_COLUMNS.values()]
            working = multi.clean_working(panel, target, required)
            test = working["split"].eq("test")
            fit = working["split"].isin(["train", "val"])
            variants = {"baseline": base_features}
            variants.update(
                {name: [*base_features, column] for name, column in SCORE_COLUMNS.items()}
            )
            predictions = {}
            alphas = {}
            validation = {}
            for name, features in variants.items():
                prediction, alpha, val_loss = fit_predict(
                    working, target, features, regression
                )
                predictions[name] = prediction
                alphas[name] = alpha
                validation[name] = val_loss
            actual = working.loc[test, target].to_numpy(float)
            rows.extend(
                result_rows(
                    domain=domain,
                    horizon=horizon,
                    family=family,
                    regression=regression,
                    actual=actual,
                    predictions=predictions,
                    n_fit=int(fit.sum()),
                    n_test=int(test.sum()),
                    alphas=alphas,
                    validation_qlike=validation,
                    extra=extra,
                )
            )
            for name, prediction in predictions.items():
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "domain": domain,
                            "horizon": horizon,
                            "forecast_family": family,
                            "regression": regression,
                            "variant": name,
                            "ticker": working.loc[test, "ticker"].astype(str).to_numpy(),
                            "date": working.loc[test, "date"].to_numpy(),
                            "actual_var": actual,
                            "prediction": prediction,
                        }
                    )
                )
            print(
                f"full forecast {domain} H{horizon} {family}/{regression} complete",
                flush=True,
            )
    return rows, prediction_frames


def sequence_with_score(panel: pd.DataFrame, score_column: str, horizon: int):
    working = panel.copy()
    working["axis_score"] = working[score_column]
    return multi.make_sequence_data(working, "parkinson", horizon)


def ensemble_lstm(data, use_score: bool, device: str, max_epochs: int):
    predictions = []
    diagnostics = []
    for seed in LSTM_SEEDS:
        prediction, epoch, val_qlike = multi.fit_lstm_variant(
            data,
            use_score=use_score,
            seed=seed,
            device_name=device,
            max_epochs=max_epochs,
        )
        predictions.append(prediction)
        diagnostics.append(
            {"seed": seed, "selected_epoch": epoch, "validation_qlike": val_qlike}
        )
    return np.mean(np.stack(predictions), axis=0), diagnostics


def evaluate_lstm(
    panel: pd.DataFrame,
    domain: str,
    device: str,
    max_epochs: int,
    horizons: tuple[int, ...],
) -> tuple[list[dict], list[pd.DataFrame]]:
    rows = []
    prediction_frames = []
    for horizon in horizons:
        baseline_data = sequence_with_score(panel, "score_llama", horizon)
        baseline, base_diagnostics = ensemble_lstm(
            baseline_data, False, device, max_epochs
        )
        predictions = {"baseline": baseline}
        all_diagnostics = {"baseline": base_diagnostics}
        for name, column in SCORE_COLUMNS.items():
            data = sequence_with_score(panel, column, horizon)
            if not (
                np.array_equal(data.date, baseline_data.date)
                and np.allclose(data.target, baseline_data.target)
                and np.array_equal(data.split, baseline_data.split)
            ):
                raise RuntimeError(f"LSTM rows differ for {domain}/{name}/H{horizon}")
            prediction, diagnostics = ensemble_lstm(data, True, device, max_epochs)
            predictions[name] = prediction
            all_diagnostics[name] = diagnostics
            print(f"full forecast {domain} H{horizon} LSTM +{name} complete", flush=True)
        test = baseline_data.split == "test"
        fit = np.isin(baseline_data.split, ["train", "val"])
        rows.extend(
            result_rows(
                domain=domain,
                horizon=horizon,
                family="LSTM",
                regression="lstm",
                actual=baseline_data.target[test],
                predictions=predictions,
                n_fit=int(fit.sum()),
                n_test=int(test.sum()),
                alphas={},
                validation_qlike={},
                extra={"lstm_diagnostics": json.dumps(all_diagnostics)},
            )
        )
        for name, prediction in predictions.items():
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "domain": domain,
                        "horizon": horizon,
                        "forecast_family": "LSTM",
                        "regression": "lstm",
                        "variant": name,
                        "ticker": np.asarray(baseline_data.ticker[test]).astype(str),
                        "date": baseline_data.date[test],
                        "actual_var": baseline_data.target[test],
                        "prediction": prediction,
                    }
                )
            )
    return rows, prediction_frames


def write_tables(results: pd.DataFrame) -> None:
    results.to_csv(OUT / "full_results_long.csv", index=False)
    for (domain, horizon), part in results.groupby(["domain", "horizon"]):
        part.to_csv(OUT / f"table_{domain}_h{horizon}.csv", index=False)
    compact = results.copy()
    compact["model"] = compact["forecast_family"] + "/" + compact["regression"]
    compact = compact[
        [
            "domain",
            "horizon",
            "model",
            "variant",
            "qlike",
            "qlike_reduction_vs_baseline_pct",
            "raw_r2",
            "raw_r2_change_vs_baseline",
            "raw_mse",
            "raw_mse_reduction_vs_baseline_pct",
        ]
    ]
    compact.to_csv(OUT / "comparison_table.csv", index=False)


def worker_paths(stem: str) -> tuple[Path, Path]:
    path = Path(stem)
    return Path(f"{path}.results.csv"), Path(f"{path}.predictions.parquet")


def write_manifest(lstm_max_epochs: int) -> None:
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "scores": {
                    "bge": "BGE-M3 validation-selected layer Ridge impact score",
                    "qwen": "Qwen3-Embedding-8B validation-selected layer Ridge impact score",
                    "llama": "Llama2-7B validation-selected activation Ridge impact score",
                    "direct_lm": "same Llama2-7B base constrained direct 1--9 likelihood score",
                },
                "centering": "none in labels and document scores",
                "forecast_models": [
                    "AR(5) OLS",
                    "HAR OLS/Lasso/Ridge",
                    "HAR-X OLS/Lasso/Ridge",
                    "MIDAS OLS",
                    "LSTM five-seed ensemble",
                ],
                "alpha_selection": "train fit, validation QLIKE selection, train+validation refit",
                "lasso_grid": LASSO_ALPHAS,
                "ridge_grid": RIDGE_ALPHAS,
                "lstm_seeds": LSTM_SEEDS,
                "lstm_max_epochs": lstm_max_epochs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def merge_workers(stems: list[str], lstm_max_epochs: int) -> None:
    result_parts = []
    prediction_parts = []
    for stem in stems:
        result_path, prediction_path = worker_paths(stem)
        if not result_path.exists():
            raise FileNotFoundError(f"incomplete forecast worker metrics: {stem}")
        result_parts.append(pd.read_csv(result_path))
        if prediction_path.exists():
            prediction_parts.append(pd.read_parquet(prediction_path))
    results = pd.concat(result_parts, ignore_index=True)
    key = ["domain", "horizon", "forecast_family", "regression", "variant"]
    if results.duplicated(key).any():
        raise RuntimeError("duplicate forecast rows across worker outputs")
    domain_horizon_pairs = results[["domain", "horizon"]].drop_duplicates()
    expected_rows = len(domain_horizon_pairs) * (
        8 * (1 + len(SCORES)) + (1 + len(SCORES))
    )
    if len(results) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} forecast rows, found {len(results)}")
    results = results.sort_values(key).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    write_tables(results)
    if prediction_parts:
        predictions = pd.concat(prediction_parts, ignore_index=True)
        predictions["ticker"] = predictions["ticker"].astype(str)
        predictions.to_parquet(OUT / "test_predictions.parquet", index=False)
    write_manifest(lstm_max_epochs)
    print(f"merged {len(results)} full forecast rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lstm-max-epochs", type=int, default=50)
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument(
        "--domains", nargs="+", choices=("news", "sec8k"), default=["news", "sec8k"]
    )
    parser.add_argument("--horizons", nargs="+", type=int, choices=(1, 5), default=[1, 5])
    parser.add_argument("--worker-stem")
    parser.add_argument("--merge-worker-stems", nargs="+")
    args = parser.parse_args()
    if not DIRECT.exists():
        raise FileNotFoundError("complete direct_llama2_rating.py first")
    if args.merge_worker_stems:
        merge_workers(args.merge_worker_stems, args.lstm_max_epochs)
        return
    if args.skip_lstm and args.skip_classical:
        raise ValueError("cannot skip both classical and LSTM forecasts")
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_predictions = []
    panel_builders = {"news": news_panel, "sec8k": sec_panel}
    horizons = tuple(dict.fromkeys(args.horizons))
    for domain in dict.fromkeys(args.domains):
        panel = panel_builders[domain]()
        if not args.skip_classical:
            rows, predictions = evaluate_classical(panel, domain, horizons)
            all_rows.extend(rows)
            all_predictions.extend(predictions)
        if not args.skip_lstm:
            rows, predictions = evaluate_lstm(
                panel, domain, args.device, args.lstm_max_epochs, horizons
            )
            all_rows.extend(rows)
            all_predictions.extend(predictions)
        if not args.worker_stem:
            partial = pd.DataFrame(all_rows)
            partial.to_csv(OUT / "full_results.partial.csv", index=False)
    results = pd.DataFrame(all_rows)
    predictions = pd.concat(all_predictions, ignore_index=True)
    if args.worker_stem:
        result_path, prediction_path = worker_paths(args.worker_stem)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(result_path, index=False)
        predictions.to_parquet(prediction_path, index=False)
        print(f"wrote forecast worker output {args.worker_stem}", flush=True)
        return
    write_tables(results)
    predictions.to_parquet(OUT / "test_predictions.parquet", index=False)
    write_manifest(args.lstm_max_epochs)
    print(f"wrote {len(results)} full forecast rows", flush=True)


if __name__ == "__main__":
    main()
