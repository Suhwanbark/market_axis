#!/usr/bin/env python3
"""Fair July comparison of frozen axes, exact embeddings, and direct ratings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
FEATURES = (
    ROOT / "outputs" / "retrieval_conditioned_feature_lock" / "features.parquet"
)
OUTCOMES = ROOT / "outputs" / "retrieval_conditioned_downstream"
AXIS = ROOT / "outputs" / "prefixless_article_axis_test" / "extractions"
BASELINES = ROOT / "outputs" / "exact_article_baselines"
OUTPUT = BASELINES / "evaluation"
MODELS = ("qwen25_7b", "qwen3_4b")
EMBEDDINGS = ("qwen3_embedding_06b", "qwen3_embedding_8b")
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0)
SEED = 20260725


def load_sharded_array(
    root: Path,
    terminal_status: str,
    array_name: str,
    expected_rows: int = 9_376,
) -> tuple[pd.DataFrame, np.ndarray]:
    pieces: list[tuple[pd.DataFrame, np.ndarray]] = []
    directories = sorted(
        path
        for path in root.glob("shard_*_of_*")
        if path.is_dir()
    )
    if len(directories) != 2:
        raise RuntimeError(f"expected two shards under {root}")
    for directory in directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("status") != terminal_status:
            raise RuntimeError(f"incomplete shard: {directory}")
        metadata = pd.read_parquet(
            directory / "metadata.parquet",
            columns=["global_row", "observation_id"],
        )
        values = np.asarray(
            np.load(directory / f"{array_name}.npy", mmap_mode="r")
        )
        if len(metadata) != len(values):
            raise RuntimeError(f"array row mismatch: {directory}")
        pieces.append((metadata, values))
    metadata = pd.concat(
        [piece[0] for piece in pieces], ignore_index=True
    )
    values = np.concatenate([piece[1] for piece in pieces], axis=0)
    order = np.argsort(metadata.global_row.to_numpy(int))
    metadata = metadata.iloc[order].reset_index(drop=True)
    values = values[order]
    if (
        len(metadata) != expected_rows
        or not metadata.observation_id.is_unique
        or metadata.global_row.tolist() != list(range(expected_rows))
    ):
        raise RuntimeError(f"identity failure under {root}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite {array_name} under {root}")
    return metadata, values


def assert_same_identity(
    expected: pd.DataFrame, observed: pd.DataFrame, name: str
) -> None:
    if (
        expected.observation_id.astype(str).tolist()
        != observed.observation_id.astype(str).tolist()
    ):
        raise RuntimeError(f"identity mismatch: {name}")


def fit_transform_package(x: np.ndarray) -> dict[str, np.ndarray]:
    median = np.nanmedian(x, axis=0)
    clean = np.where(np.isfinite(x), x, median)
    mean = clean.mean(axis=0)
    scale = clean.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return {"median": median, "mean": mean, "scale": scale}


def transform(x: np.ndarray, package: dict[str, np.ndarray]) -> np.ndarray:
    clean = np.where(np.isfinite(x), x, package["median"])
    return ((clean - package["mean"]) / package["scale"]).astype(
        np.float32
    )


def fit_select_predict(
    x: np.ndarray,
    target_log: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    package = fit_transform_package(x[train])
    x_train = transform(x[train], package)
    x_validation = transform(x[validation], package)
    rows: list[dict[str, float]] = []
    best_score = -np.inf
    best_alpha = ALPHAS[0]
    for alpha in ALPHAS:
        model = Ridge(
            alpha=alpha, solver="lsqr", tol=1e-5
        ).fit(x_train, target_log[train])
        prediction = model.predict(x_validation)
        score = float(r2_score(target_log[validation], prediction))
        rows.append({"alpha": alpha, "validation_r2_log": score})
        if (score, -alpha) > (best_score, -best_alpha):
            best_score, best_alpha = score, alpha
    fit = train | validation
    final_package = fit_transform_package(x[fit])
    x_fit = transform(x[fit], final_package)
    x_test = transform(x[test], final_package)
    model = Ridge(
        alpha=best_alpha, solver="lsqr", tol=1e-5
    ).fit(x_fit, target_log[fit])
    return model.predict(x_test), float(best_alpha), rows


def block_bootstrap_rho(
    frame: pd.DataFrame,
    score: str,
    target: str,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    block = frame.event_session.astype(str).to_numpy()
    blocks = np.unique(block)
    groups = {
        value: np.flatnonzero(block == value) for value in blocks
    }
    x = frame[score].to_numpy(float)
    y = frame[target].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        index = np.concatenate([groups[value] for value in sampled])
        rho = spearmanr(x[index], y[index]).statistic
        if np.isfinite(rho):
            values.append(float(rho))
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def within_date_permutation(
    frame: pd.DataFrame,
    score: str,
    target: str,
    observed: float,
    repeats: int,
    seed: int,
) -> float:
    block = frame.event_session.astype(str).to_numpy()
    groups = [
        np.flatnonzero(block == value) for value in np.unique(block)
    ]
    x = frame[score].to_numpy(float)
    y = frame[target].to_numpy(float)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(repeats):
        shuffled = x.copy()
        for index in groups:
            shuffled[index] = rng.permutation(shuffled[index])
        rho = spearmanr(shuffled, y).statistic
        exceed += int(np.isfinite(rho) and rho >= observed)
    return float((exceed + 1) / (repeats + 1))


def paired_rho_difference(
    frame: pd.DataFrame,
    candidate: str,
    baseline: str,
    target: str,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    block = frame.event_session.astype(str).to_numpy()
    blocks = np.unique(block)
    groups = {
        value: np.flatnonzero(block == value) for value in blocks
    }
    candidate_values = frame[candidate].to_numpy(float)
    baseline_values = frame[baseline].to_numpy(float)
    truth = frame[target].to_numpy(float)
    observed = float(
        spearmanr(candidate_values, truth).statistic
        - spearmanr(baseline_values, truth).statistic
    )
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        index = np.concatenate([groups[value] for value in sampled])
        delta = (
            spearmanr(candidate_values[index], truth[index]).statistic
            - spearmanr(baseline_values[index], truth[index]).statistic
        )
        if np.isfinite(delta):
            values.append(float(delta))
    array = np.asarray(values, float)
    return {
        "observed_delta_spearman": observed,
        "date_block_ci_low": float(np.quantile(array, 0.025)),
        "date_block_ci_high": float(np.quantile(array, 0.975)),
        "bootstrap_probability_axis_not_better": float(
            np.mean(array <= 0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2_000)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    terminal = OUTPUT / "manifest.json"
    if terminal.exists():
        status = json.loads(terminal.read_text()).get("status")
        if status == "EXACT_ARTICLE_BASELINE_EVALUATION_COMPLETE":
            print(f"already complete: {terminal}")
            return
        raise RuntimeError(f"non-terminal manifest exists: {terminal}")

    base = pd.read_parquet(
        FEATURES,
        columns=[
            "observation_id",
            "market_ticker",
            "event_session",
            "downstream_split",
        ],
    ).reset_index(drop=True)
    base["global_row"] = np.arange(len(base), dtype=np.int64)
    axis_arrays: dict[str, dict[str, np.ndarray]] = {}
    direct_arrays: dict[str, np.ndarray] = {}
    embedding_arrays: dict[str, np.ndarray] = {}
    for model in MODELS:
        axis_arrays[model] = {}
        for score in ("last_projection", "mean_projection"):
            identity, values = load_sharded_array(
                AXIS / model,
                "PREFIXLESS_ARTICLE_AXIS_SHARD_COMPLETE",
                score,
            )
            assert_same_identity(base, identity, f"axis {model} {score}")
            axis_arrays[model][score] = np.asarray(
                values, np.float32
            ).reshape(-1, 1)
        identity, values = load_sharded_array(
            BASELINES / "direct_rating" / model,
            "EXACT_ARTICLE_DIRECT_RATING_SHARD_COMPLETE",
            "expected_rating",
        )
        assert_same_identity(base, identity, f"direct {model}")
        direct_arrays[model] = np.asarray(values, np.float32).reshape(-1, 1)
    for name in EMBEDDINGS:
        identity, values = load_sharded_array(
            BASELINES / "embeddings" / name / "prefixless",
            "EXACT_ARTICLE_EMBEDDING_SHARD_COMPLETE",
            "embeddings",
        )
        assert_same_identity(base, identity, f"embedding {name}")
        embedding_arrays[name] = np.asarray(values, np.float32)

    outcomes = pd.concat(
        [
            pd.read_parquet(OUTCOMES / "development_outcomes.parquet"),
            pd.read_parquet(OUTCOMES / "july_confirmation_outcomes.parquet"),
        ],
        ignore_index=True,
    )
    data = base.merge(
        outcomes.drop(columns=["market_ticker", "downstream_split"]),
        on=["observation_id", "event_session"],
        how="inner",
        validate="one_to_one",
    ).sort_values("global_row")
    valid = (
        np.isfinite(data.abs_market_adjusted_return_h1)
        & np.isfinite(data.prevol20)
        & np.isfinite(data.pre_abs_market_adjusted_return_h1)
    )
    data = data.loc[valid].reset_index(drop=True)
    global_rows = data.global_row.to_numpy(int)
    train = data.downstream_split.eq("development_train").to_numpy()
    validation = data.downstream_split.eq("development_test").to_numpy()
    july = data.downstream_split.eq("july_confirmation").to_numpy()
    if july.sum() != 383:
        raise RuntimeError(f"July row mismatch: {july.sum()}")
    target = data.abs_market_adjusted_return_h1.to_numpy(float)
    target_log = np.log1p(10_000.0 * target)
    price = data[
        ["prevol20", "pre_abs_market_adjusted_return_h1"]
    ].to_numpy(np.float32)
    axis = {
        key: {
            score: values[global_rows]
            for score, values in model_values.items()
        }
        for key, model_values in axis_arrays.items()
    }
    direct = {
        key: value[global_rows] for key, value in direct_arrays.items()
    }
    embedding = {
        key: value[global_rows]
        for key, value in embedding_arrays.items()
    }
    embedding_pca: dict[str, np.ndarray] = {}
    for index, (name, values) in enumerate(embedding.items()):
        components = min(64, int(train.sum()) - 1, values.shape[1])
        pca = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=SEED + index,
        )
        pca.fit(values[train])
        embedding_pca[name] = pca.transform(values).astype(np.float32)

    configurations: dict[str, np.ndarray] = {
        "price_only": price,
        "axis_last_qwen25_ridge": axis["qwen25_7b"][
            "last_projection"
        ],
        "axis_last_qwen3_ridge": axis["qwen3_4b"][
            "last_projection"
        ],
        "axis_last_both_ridge": np.concatenate(
            [
                axis["qwen25_7b"]["last_projection"],
                axis["qwen3_4b"]["last_projection"],
            ],
            axis=1,
        ),
        "axis_mean_qwen25_ridge": axis["qwen25_7b"][
            "mean_projection"
        ],
        "axis_mean_qwen3_ridge": axis["qwen3_4b"][
            "mean_projection"
        ],
        "axis_mean_both_ridge": np.concatenate(
            [
                axis["qwen25_7b"]["mean_projection"],
                axis["qwen3_4b"]["mean_projection"],
            ],
            axis=1,
        ),
        "price_plus_axis_last_qwen25": np.concatenate(
            [price, axis["qwen25_7b"]["last_projection"]], axis=1
        ),
        "price_plus_axis_last_qwen3": np.concatenate(
            [price, axis["qwen3_4b"]["last_projection"]], axis=1
        ),
        "price_plus_axis_last_both": np.concatenate(
            [
                price,
                axis["qwen25_7b"]["last_projection"],
                axis["qwen3_4b"]["last_projection"],
            ],
            axis=1,
        ),
        "price_plus_axis_mean_qwen25": np.concatenate(
            [price, axis["qwen25_7b"]["mean_projection"]], axis=1
        ),
        "price_plus_axis_mean_qwen3": np.concatenate(
            [price, axis["qwen3_4b"]["mean_projection"]], axis=1
        ),
        "price_plus_axis_mean_both": np.concatenate(
            [
                price,
                axis["qwen25_7b"]["mean_projection"],
                axis["qwen3_4b"]["mean_projection"],
            ],
            axis=1,
        ),
        "direct_qwen25_ridge": direct["qwen25_7b"],
        "direct_qwen3_ridge": direct["qwen3_4b"],
        "direct_both_ridge": np.concatenate(
            [direct["qwen25_7b"], direct["qwen3_4b"]], axis=1
        ),
        "price_plus_direct_both": np.concatenate(
            [price, direct["qwen25_7b"], direct["qwen3_4b"]], axis=1
        ),
        "embedding_06b_ridge": embedding_pca["qwen3_embedding_06b"],
        "embedding_8b_ridge": embedding_pca["qwen3_embedding_8b"],
        "price_plus_embedding_06b": np.concatenate(
            [price, embedding_pca["qwen3_embedding_06b"]], axis=1
        ),
        "price_plus_embedding_8b": np.concatenate(
            [price, embedding_pca["qwen3_embedding_8b"]], axis=1
        ),
        "price_embedding_8b_plus_axis_both": np.concatenate(
            [
                price,
                embedding_pca["qwen3_embedding_8b"],
                axis["qwen25_7b"]["last_projection"],
                axis["qwen3_4b"]["last_projection"],
            ],
            axis=1,
        ),
        "price_embedding_8b_plus_axis_mean_both": np.concatenate(
            [
                price,
                embedding_pca["qwen3_embedding_8b"],
                axis["qwen25_7b"]["mean_projection"],
                axis["qwen3_4b"]["mean_projection"],
            ],
            axis=1,
        ),
    }
    threshold = float(
        json.loads(
            (OUTCOMES / "downstream_selection.json").read_text()
        )["importance_threshold_frozen_from_development"]
    )
    truth = target[july]
    large = truth >= threshold
    predictions = data.loc[
        july,
        [
            "observation_id",
            "market_ticker",
            "event_session",
            "abs_market_adjusted_return_h1",
        ],
    ].copy()
    predictions["large_reaction"] = large
    selection_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for order, (name, values) in enumerate(configurations.items()):
        prediction, alpha, sweep = fit_select_predict(
            values, target_log, train, validation, july
        )
        predictions[f"prediction_{name}"] = prediction
        for row in sweep:
            selection_rows.append({"feature_set": name, **row})
        top_n = max(1, int(np.ceil(0.10 * len(prediction))))
        top = np.argsort(prediction)[-top_n:]
        result_rows.append(
            {
                "feature_set": name,
                "feature_count": values.shape[1],
                "selected_alpha": alpha,
                "r2_log": float(
                    r2_score(np.log1p(10_000.0 * truth), prediction)
                ),
                "spearman": float(
                    spearmanr(truth, prediction).statistic
                ),
                "large_reaction_auc": float(
                    roc_auc_score(large, prediction)
                ),
                "top_decile_precision": float(large[top].mean()),
                "top_decile_mean_return_lift": float(
                    truth[top].mean() / truth.mean()
                ),
                "configuration_order": order,
            }
        )

    raw_scores = predictions[
        [
            "observation_id",
            "market_ticker",
            "event_session",
            "abs_market_adjusted_return_h1",
        ]
    ].copy()
    raw_scores["axis_last_qwen25"] = axis["qwen25_7b"][
        "last_projection"
    ][july, 0]
    raw_scores["axis_last_qwen3"] = axis["qwen3_4b"][
        "last_projection"
    ][july, 0]
    raw_scores["axis_mean_qwen25"] = axis["qwen25_7b"][
        "mean_projection"
    ][july, 0]
    raw_scores["axis_mean_qwen3"] = axis["qwen3_4b"][
        "mean_projection"
    ][july, 0]
    raw_scores["direct_qwen25"] = direct["qwen25_7b"][july, 0]
    raw_scores["direct_qwen3"] = direct["qwen3_4b"][july, 0]
    univariate_rows = []
    for index, score in enumerate(
        (
            "axis_last_qwen25",
            "axis_last_qwen3",
            "axis_mean_qwen25",
            "axis_mean_qwen3",
            "direct_qwen25",
            "direct_qwen3",
        )
    ):
        rho = float(
            spearmanr(
                raw_scores[score],
                raw_scores.abs_market_adjusted_return_h1,
            ).statistic
        )
        low, high = block_bootstrap_rho(
            raw_scores,
            score,
            "abs_market_adjusted_return_h1",
            args.repeats,
            SEED + index,
        )
        p_value = within_date_permutation(
            raw_scores,
            score,
            "abs_market_adjusted_return_h1",
            rho,
            args.repeats,
            SEED + 100 + index,
        )
        univariate_rows.append(
            {
                "score": score,
                "rows": len(raw_scores),
                "spearman": rho,
                "date_block_ci_low": low,
                "date_block_ci_high": high,
                "within_date_permutation_p_one_sided": p_value,
            }
        )

    comparison_pairs = (
        ("axis_last_qwen25", "direct_qwen25"),
        ("axis_last_qwen3", "direct_qwen3"),
        ("axis_mean_qwen25", "direct_qwen25"),
        ("axis_mean_qwen3", "direct_qwen3"),
        (
            "prediction_axis_last_qwen25_ridge",
            "prediction_embedding_8b_ridge",
        ),
        (
            "prediction_axis_last_qwen3_ridge",
            "prediction_embedding_8b_ridge",
        ),
        (
            "prediction_axis_mean_qwen25_ridge",
            "prediction_embedding_8b_ridge",
        ),
        (
            "prediction_axis_mean_qwen3_ridge",
            "prediction_embedding_8b_ridge",
        ),
        (
            "prediction_price_plus_axis_last_both",
            "prediction_price_plus_embedding_8b",
        ),
        (
            "prediction_price_plus_axis_mean_both",
            "prediction_price_plus_embedding_8b",
        ),
    )
    comparison_frame = raw_scores.merge(
        predictions.drop(
            columns=[
                "market_ticker",
                "abs_market_adjusted_return_h1",
                "large_reaction",
            ]
        ),
        on=["observation_id", "event_session"],
        how="inner",
        validate="one_to_one",
    )
    gain_rows = []
    for index, (candidate, baseline) in enumerate(comparison_pairs):
        gain_rows.append(
            {
                "axis_candidate": candidate,
                "baseline": baseline,
                **paired_rho_difference(
                    comparison_frame,
                    candidate,
                    baseline,
                    "abs_market_adjusted_return_h1",
                    args.repeats,
                    SEED + 200 + index,
                ),
            }
        )

    pd.DataFrame(selection_rows).to_csv(
        OUTPUT / "development_alpha_selection.csv", index=False
    )
    pd.DataFrame(result_rows).to_csv(
        OUTPUT / "july_predictive_comparison.csv", index=False
    )
    pd.DataFrame(univariate_rows).to_csv(
        OUTPUT / "july_zero_shot_comparison.csv", index=False
    )
    pd.DataFrame(gain_rows).to_csv(
        OUTPUT / "july_axis_baseline_gain.csv", index=False
    )
    predictions.to_parquet(
        OUTPUT / "july_predictions.parquet", index=False
    )
    terminal.write_text(
        json.dumps(
            {
                "status": "EXACT_ARTICLE_BASELINE_EVALUATION_COMPLETE",
                "interpretation": (
                    "retrospective fair-text comparison; exact prefixless "
                    "article for axis/embedding, fixed suffix question only "
                    "for direct LM"
                ),
                "rows": int(july.sum()),
                "event_dates": int(
                    data.loc[july, "event_session"].nunique()
                ),
                "development_train_rows": int(train.sum()),
                "development_validation_rows": int(validation.sum()),
                "market_labels_used_by_axis": False,
                "market_labels_used_by_direct_rating": False,
                "market_labels_used_by_embedding_forward": False,
                "market_labels_used_by_ridge_heads": True,
                "embedding_preprocessing": (
                    "64-component randomized PCA fit on eligible "
                    "development_train articles only"
                ),
                "bootstrap_repeats": args.repeats,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("EXACT_ARTICLE_BASELINE_EVALUATION_COMPLETE")


if __name__ == "__main__":
    main()
