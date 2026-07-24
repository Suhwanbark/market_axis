#!/usr/bin/env python3
"""Find a ticker-conditioned surprise readout without assuming linearity.

The controlled reports and company-memory splits were frozen by the completed
retrieval-conditioned study.  This retrospective extension:

1. keeps strong-memory rows for which the model correctly parses the report;
2. defines continuous ticker surprise from the firm's pre-report conditional
   probability of the realized outcome;
3. removes relation x realized-outcome mean activation updates using train
   firms only;
4. compares linear, absolute-delta, and pair-ranking MLP readouts using
   validation firms only; and
5. applies the frozen winner to real news without market labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
MEMORY = ROOT / "outputs" / "retrieval_conditioned_memory"
ANALYSIS = ROOT / "outputs" / "retrieval_conditioned_analysis"
OUTPUT = ROOT / "outputs" / "ticker_surprise_nonlinear"
MODELS = ("qwen25_7b", "qwen3_4b")
SEED = 20260724
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
MLP_CONFIGS = (
    {"hidden": 32, "weight_decay": 1e-3, "pair_weight": 1.0},
    {"hidden": 64, "weight_decay": 1e-3, "pair_weight": 1.0},
    {"hidden": 32, "weight_decay": 1e-2, "pair_weight": 2.0},
)
MLP_SEEDS = (41, 42, 43)
EPS = 1e-8


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_keys(table: pd.DataFrame) -> np.ndarray:
    return (
        table.relation.astype(str)
        + "|"
        + table.outcome_letter.astype(str)
    ).to_numpy(str)


def continuous_target(table: pd.DataFrame) -> np.ndarray:
    probabilities = table[
        ["p_favorable", "p_continuity", "p_adverse"]
    ].to_numpy(float)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), EPS)
    outcome = table.outcome_index.to_numpy(int)
    return -np.log(
        np.clip(probabilities[np.arange(len(table)), outcome], EPS, 1.0)
    )


def fit_group_scalar_normalizer(
    raw_target: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = np.asarray(sorted(np.unique(groups)), dtype="U64")
    global_mean = float(raw_target[train].mean())
    global_scale = max(float(raw_target[train].std()), 1e-6)
    means, scales = [], []
    normalized = np.empty(len(raw_target), dtype=np.float32)
    for key in keys:
        fit = train & (groups == key)
        mean = (
            float(raw_target[fit].mean()) if fit.any() else global_mean
        )
        scale = (
            max(float(raw_target[fit].std()), 1e-6)
            if fit.sum() > 1
            else global_scale
        )
        means.append(mean)
        scales.append(scale)
        selected = groups == key
        normalized[selected] = (raw_target[selected] - mean) / scale
    return (
        normalized,
        keys,
        np.asarray(means, np.float32),
        np.asarray(scales, np.float32),
    )


def fit_group_vector_centers(
    values: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(sorted(np.unique(groups)), dtype="U64")
    fallback = values[train].mean(axis=0, dtype=np.float64).astype(np.float32)
    centers = []
    for key in keys:
        selected = train & (groups == key)
        centers.append(
            values[selected].mean(axis=0, dtype=np.float64).astype(np.float32)
            if selected.any()
            else fallback
        )
    return keys, np.stack(centers).astype(np.float32)


def apply_group_centers(
    values: np.ndarray,
    groups: np.ndarray,
    keys: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    lookup = {str(key): index for index, key in enumerate(keys.tolist())}
    fallback = centers.mean(axis=0)
    result = np.empty_like(values, dtype=np.float32)
    for key in np.unique(groups):
        selected = groups == key
        index = lookup.get(str(key))
        result[selected] = values[selected] - (
            centers[index] if index is not None else fallback
        )
    return result


def standardize(
    values: np.ndarray, train: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values[train].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values[train].std(axis=0, dtype=np.float64).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return ((values - mean) / scale).astype(np.float32), mean, scale


def pair_concordance(
    target: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
) -> float:
    weighted_sum = total_weight = 0.0
    for key in np.unique(groups[mask]):
        selected = mask & (groups == key)
        if selected.sum() < 3:
            continue
        tau = kendalltau(
            target[selected], prediction[selected], variant="b"
        ).statistic
        if not np.isfinite(tau):
            continue
        weight = selected.sum() * (selected.sum() - 1) / 2
        weighted_sum += weight * (tau + 1.0) / 2.0
        total_weight += weight
    return (
        float(weighted_sum / total_weight)
        if total_weight > 0
        else math.nan
    )


def within_group_spearman(
    target: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
) -> float:
    weighted_sum = total_weight = 0.0
    for key in np.unique(groups[mask]):
        selected = mask & (groups == key)
        if (
            selected.sum() < 3
            or np.std(target[selected]) < EPS
            or np.std(prediction[selected]) < EPS
        ):
            continue
        value = spearmanr(
            target[selected], prediction[selected]
        ).statistic
        if not np.isfinite(value):
            continue
        weighted_sum += selected.sum() * value
        total_weight += selected.sum()
    return (
        float(weighted_sum / total_weight)
        if total_weight > 0
        else math.nan
    )


def group_weighted_conflict_auc(
    table: pd.DataFrame,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> float:
    diagnostic = mask & (
        table.is_confirm.to_numpy(bool)
        | table.is_contradict.to_numpy(bool)
    )
    groups = group_keys(table)
    labels = table.violation_label.to_numpy(int)
    weighted_sum = total_weight = 0.0
    for key in np.unique(groups[diagnostic]):
        selected = diagnostic & (groups == key)
        if selected.sum() < 2 or len(np.unique(labels[selected])) != 2:
            continue
        positive = labels[selected].sum()
        negative = selected.sum() - positive
        weight = positive * negative
        weighted_sum += weight * roc_auc_score(
            labels[selected], prediction[selected]
        )
        total_weight += weight
    return (
        float(weighted_sum / total_weight)
        if total_weight > 0
        else math.nan
    )


def metrics(
    table: pd.DataFrame,
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    groups = group_keys(table)
    return {
        "rows": int(mask.sum()),
        "firms": int(table.loc[mask, "cik"].nunique()),
        "global_spearman": float(
            spearmanr(target[mask], prediction[mask]).statistic
        ),
        "within_group_spearman": within_group_spearman(
            target, prediction, groups, mask
        ),
        "pair_concordance": pair_concordance(
            target, prediction, groups, mask
        ),
        "conflict_auc_same_outcome": group_weighted_conflict_auc(
            table, prediction, mask
        ),
        "mae": float(np.mean(np.abs(target[mask] - prediction[mask]))),
    }


def controlled_values(
    model: str,
    table: pd.DataFrame,
    positions: np.ndarray,
    layer: int,
    representation: str,
) -> np.ndarray:
    extraction = MEMORY / model
    pre_hidden = np.load(
        extraction / "expectations" / "pre_hidden.npy", mmap_mode="r"
    )
    post_hidden = np.load(
        extraction / "controlled" / "controlled_hidden.npy", mmap_mode="r"
    )
    i0 = table.expectation_row_v0.to_numpy(int)
    i1 = table.expectation_row_v1.to_numpy(int)
    # Reading the complete selected layer is much faster on GPFS than tens of
    # thousands of small fancy-index reads.
    pre_layer = np.asarray(pre_hidden[:, layer], dtype=np.float32)
    post_layer = np.asarray(post_hidden[:, layer], dtype=np.float32)
    pre = 0.5 * (pre_layer[i0] + pre_layer[i1])
    post = post_layer[positions]
    if representation == "delta":
        return post - pre
    if representation == "post":
        return post
    raise ValueError(representation)


def ridge_candidates(
    values: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    table: pd.DataFrame,
    method: str,
) -> tuple[list[dict[str, object]], dict[float, tuple[np.ndarray, dict[str, Any]]]]:
    transformed, mean, scale = standardize(values, train)
    if method == "absolute_delta":
        transformed = np.abs(transformed)
    rows = []
    packages: dict[float, tuple[np.ndarray, dict[str, Any]]] = {}
    for alpha in RIDGE_ALPHAS:
        estimator = Ridge(alpha=alpha, solver="lsqr", tol=1e-5)
        estimator.fit(transformed[train], target[train])
        prediction = estimator.predict(transformed)
        row = {
            "method": method,
            "alpha": alpha,
            **{
                f"validation_{key}": value
                for key, value in metrics(
                    table, target, prediction, validation
                ).items()
            },
        }
        rows.append(row)
        packages[alpha] = (
            np.asarray(prediction, float),
            {
                "method": method,
                "feature_mean": mean,
                "feature_scale": scale,
                "coef": np.asarray(estimator.coef_, np.float32),
                "intercept": np.asarray(
                    [estimator.intercept_], np.float32
                ),
            },
        )
    return rows, packages


def build_pair_indices(
    target: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    seed: int,
    maximum: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lower, upper = [], []
    train_positions = np.flatnonzero(train)
    local_lookup = {
        global_index: local_index
        for local_index, global_index in enumerate(train_positions.tolist())
    }
    for key in np.unique(groups[train]):
        index = np.flatnonzero(train & (groups == key))
        if len(index) < 4:
            continue
        ordered = index[np.argsort(target[index])]
        half = max(1, len(ordered) // 2)
        low_pool, high_pool = ordered[:half], ordered[-half:]
        draws = min(5_000, max(200, len(index) * 10))
        low = rng.choice(low_pool, draws, replace=True)
        high = rng.choice(high_pool, draws, replace=True)
        keep = target[high] > target[low] + 1e-6
        lower.extend(local_lookup[int(item)] for item in low[keep])
        upper.extend(local_lookup[int(item)] for item in high[keep])
    if len(lower) > maximum:
        chosen = rng.choice(len(lower), maximum, replace=False)
        lower = np.asarray(lower, int)[chosen]
        upper = np.asarray(upper, int)[chosen]
    return np.asarray(lower, int), np.asarray(upper, int)


def train_pair_mlp(
    transformed: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    table: pd.DataFrame,
    config: dict[str, float],
    seed: int,
    device: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch_device = torch.device(f"cuda:{device}")
    train_positions = np.flatnonzero(train)
    lower, upper = build_pair_indices(
        target, groups, train, seed=SEED + seed
    )
    x_train = torch.as_tensor(
        transformed[train], dtype=torch.float32, device=torch_device
    )
    y_train = torch.as_tensor(
        target[train], dtype=torch.float32, device=torch_device
    )
    lower_t = torch.as_tensor(lower, dtype=torch.long, device=torch_device)
    upper_t = torch.as_tensor(upper, dtype=torch.long, device=torch_device)
    model = torch.nn.Sequential(
        torch.nn.Linear(
            transformed.shape[1], int(config["hidden"])
        ),
        torch.nn.GELU(),
        torch.nn.Linear(int(config["hidden"]), 1),
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=float(config["weight_decay"]),
    )
    best_state = None
    best_score = -math.inf
    best_epoch = 0
    patience = 10
    checks_without_improvement = 0
    validation_positions = np.flatnonzero(validation)
    for epoch in range(301):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        score = model(x_train).squeeze(-1)
        regression = functional.smooth_l1_loss(score, y_train)
        ranking = functional.softplus(
            -(score[upper_t] - score[lower_t])
        ).mean()
        loss = regression + float(config["pair_weight"]) * ranking
        loss.backward()
        optimizer.step()
        if epoch % 10 != 0:
            continue
        model.eval()
        with torch.inference_mode():
            validation_prediction = (
                model(
                    torch.as_tensor(
                        transformed[validation],
                        dtype=torch.float32,
                        device=torch_device,
                    )
                )
                .squeeze(-1)
                .cpu()
                .numpy()
            )
        full_validation_prediction = np.zeros(len(target), dtype=float)
        full_validation_prediction[validation_positions] = (
            validation_prediction
        )
        score_value = pair_concordance(
            target,
            full_validation_prediction,
            groups,
            validation,
        )
        if np.isfinite(score_value) and score_value > best_score + 1e-5:
            best_score = float(score_value)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
        if checks_without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("MLP did not produce a finite validation score")
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(transformed), 1024):
            batch = torch.as_tensor(
                transformed[start : start + 1024],
                dtype=torch.float32,
                device=torch_device,
            )
            predictions.append(
                model(batch).squeeze(-1).cpu().numpy()
            )
    state = {
        "w1": best_state["0.weight"].numpy().T.astype(np.float32),
        "b1": best_state["0.bias"].numpy().astype(np.float32),
        "w2": best_state["2.weight"].numpy().reshape(-1).astype(np.float32),
        "b2": best_state["2.bias"].numpy().reshape(-1).astype(np.float32),
    }
    return (
        np.concatenate(predictions).astype(float),
        state,
        {
            "best_epoch": best_epoch,
            "best_validation_pair_concordance": best_score,
            "train_pairs": int(len(lower)),
            "train_rows": int(len(train_positions)),
        },
    )


def mlp_candidates(
    values: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    table: pd.DataFrame,
    device: int,
    representation: str,
) -> tuple[
    list[dict[str, object]],
    dict[int, tuple[np.ndarray, dict[str, Any]]],
]:
    transformed, mean, scale = standardize(values, train)
    rows = []
    packages = {}
    for config_index, config in enumerate(MLP_CONFIGS):
        seed_predictions, states, diagnostics = [], [], []
        for seed in MLP_SEEDS:
            prediction, state, diagnostic = train_pair_mlp(
                transformed,
                target,
                groups,
                train,
                validation,
                table,
                config,
                seed,
                device,
            )
            seed_predictions.append(prediction)
            states.append(state)
            diagnostics.append(diagnostic)
        prediction = np.mean(seed_predictions, axis=0)
        method = (
            "mlp_pair_delta"
            if representation == "delta"
            else "mlp_pair_post"
        )
        rows.append(
            {
                "method": method,
                "config_index": config_index,
                "hidden": int(config["hidden"]),
                "weight_decay": float(config["weight_decay"]),
                "pair_weight": float(config["pair_weight"]),
                **{
                    f"validation_{key}": value
                    for key, value in metrics(
                        table, target, prediction, validation
                    ).items()
                },
                "mean_best_epoch": float(
                    np.mean([item["best_epoch"] for item in diagnostics])
                ),
                "train_pairs": int(diagnostics[0]["train_pairs"]),
            }
        )
        packages[config_index] = (
            prediction,
            {
                "method": method,
                "feature_mean": mean,
                "feature_scale": scale,
                "states": states,
                "config": config,
            },
        )
    return rows, packages


def select_row(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(
        [
            "validation_pair_concordance",
            "validation_within_group_spearman",
            "validation_global_spearman",
            "layer",
            "configuration_order",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).iloc[0]


def apply_package(
    values: np.ndarray,
    groups: np.ndarray,
    package: dict[str, Any],
) -> np.ndarray:
    centered = apply_group_centers(
        values,
        groups,
        package["group_keys"],
        package["group_centers"],
    )
    transformed = (
        centered - package["feature_mean"]
    ) / package["feature_scale"]
    method = package["method"]
    if method == "absolute_delta":
        transformed = np.abs(transformed)
    if method in ("linear_delta", "linear_post", "absolute_delta"):
        return transformed @ package["coef"] + float(
            package["intercept"][0]
        )
    if method.startswith("mlp_pair"):
        predictions = []
        for state in package["states"]:
            hidden = transformed @ state["w1"] + state["b1"]
            hidden = (
                0.5
                * hidden
                * (1.0 + erf(hidden / math.sqrt(2.0)))
            )
            predictions.append(
                hidden @ state["w2"] + float(state["b2"][0])
            )
        return np.mean(predictions, axis=0)
    raise ValueError(method)


def save_package(path: Path, package: dict[str, Any]) -> None:
    payload: dict[str, np.ndarray] = {
        "group_keys": package["group_keys"],
        "group_centers": package["group_centers"],
        "feature_mean": package["feature_mean"],
        "feature_scale": package["feature_scale"],
        "target_group_keys": package["target_group_keys"],
        "target_group_means": package["target_group_means"],
        "target_group_scales": package["target_group_scales"],
    }
    method = package["method"]
    if method.startswith("mlp_pair"):
        for index, state in enumerate(package["states"]):
            for name, value in state.items():
                payload[f"seed{index}_{name}"] = value
    else:
        payload["coef"] = package["coef"]
        payload["intercept"] = package["intercept"]
    np.savez_compressed(path, **payload)


def real_news_scores(
    model: str,
    layer: int,
    package: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    relation = pd.read_parquet(
        ANALYSIS / model / "news_relation_features.parquet"
    )
    relation = relation.loc[relation.analysis_keep.astype(bool)].copy()
    pre_hidden = np.load(
        MEMORY / model / "expectations" / "pre_hidden.npy",
        mmap_mode="r",
    )
    news_hidden = np.load(
        MEMORY / model / "news" / "news_hidden.npy",
        mmap_mode="r",
    )
    pre_layer = np.asarray(pre_hidden[:, layer], dtype=np.float32)
    news_layer = np.asarray(news_hidden[:, layer], dtype=np.float32)
    i0 = relation.expectation_row_v0.to_numpy(int)
    i1 = relation.expectation_row_v1.to_numpy(int)
    news_row = relation.news_array_row.to_numpy(int)
    pre = 0.5 * (pre_layer[i0] + pre_layer[i1])
    delta = news_layer[news_row] - pre
    letter = relation.news_direction_index.map(
        {0: "A", 1: "B", 2: "C"}
    )
    groups = (
        relation.relation.astype(str) + "|" + letter.astype(str)
    ).to_numpy(str)
    score = apply_package(delta, groups, package)
    relation["ticker_surprise_score"] = score
    relation["ticker_surprise_positive"] = np.maximum(score, 0.0)
    relation["ticker_surprise_gated"] = (
        relation.ticker_surprise_positive
        * relation.directional_memory_strength.to_numpy(float)
    )
    article = (
        relation.groupby("observation_id")
        .agg(
            ticker_surprise_max=("ticker_surprise_score", "max"),
            ticker_surprise_positive_max=(
                "ticker_surprise_positive",
                "max",
            ),
            ticker_surprise_gated_max=(
                "ticker_surprise_gated",
                "max",
            ),
            ticker_surprise_mean=("ticker_surprise_score", "mean"),
        )
        .reset_index()
    )
    keep_columns = [
        "news_relation_id",
        "observation_id",
        "cik",
        "market_ticker",
        "recall_split",
        "downstream_split",
        "relation",
        "news_direction_index",
        "directional_memory_strength",
        "ticker_surprise_score",
        "ticker_surprise_positive",
        "ticker_surprise_gated",
    ]
    return relation[keep_columns], article


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    model = args.model
    out = OUTPUT / model
    out.mkdir(parents=True, exist_ok=True)
    terminal = out / "manifest.json"
    if terminal.exists():
        status = json.loads(terminal.read_text()).get("status")
        if status == "TICKER_SURPRISE_NONLINEAR_COMPLETE":
            print(f"already complete: {terminal}")
            return
        raise RuntimeError(f"non-terminal manifest exists: {terminal}")

    table_all = pd.read_parquet(
        ANALYSIS / model / "controlled_metadata.parquet"
    ).reset_index(drop=True)
    eligible_mask = (
        table_all.strong_memory_benchmark.astype(bool)
        & table_all.post_correct.astype(bool)
    )
    positions = np.flatnonzero(eligible_mask.to_numpy())
    table = table_all.loc[eligible_mask].reset_index(drop=True)
    table["source_row"] = positions
    groups = group_keys(table)
    split = table.recall_split.astype(str).to_numpy()
    train = split == "train"
    validation = split == "validation"
    test = split == "test"
    raw_target = continuous_target(table)
    (
        target,
        target_group_keys,
        target_group_means,
        target_group_scales,
    ) = fit_group_scalar_normalizer(raw_target, groups, train)

    prior_selection = json.loads(
        (ANALYSIS / model / "selection.json").read_text()
    )
    layers = [int(item) for item in prior_selection["geometry_candidate_layers"]]
    sweep_rows: list[dict[str, object]] = []
    package_cache: dict[tuple[object, ...], tuple[np.ndarray, dict[str, Any]]] = {}
    order = 0
    for layer in layers:
        delta = controlled_values(
            model, table, positions, layer, "delta"
        )
        group_center_keys, group_centers = fit_group_vector_centers(
            delta, groups, train
        )
        centered_delta = apply_group_centers(
            delta, groups, group_center_keys, group_centers
        )
        for method, values in (
            ("linear_delta", centered_delta),
            ("absolute_delta", centered_delta),
        ):
            rows, packages = ridge_candidates(
                values,
                target,
                train,
                validation,
                table,
                method,
            )
            for row in rows:
                row.update(
                    {
                        "model": model,
                        "layer": layer,
                        "representation": "delta",
                        "configuration_order": order,
                    }
                )
                order += 1
                sweep_rows.append(row)
                prediction, package = packages[float(row["alpha"])]
                package.update(
                    {
                        "layer": layer,
                        "group_keys": group_center_keys,
                        "group_centers": group_centers,
                    }
                )
                package_cache[
                    (method, layer, float(row["alpha"]))
                ] = (prediction, package)

        rows, packages = mlp_candidates(
            centered_delta,
            target,
            groups,
            train,
            validation,
            table,
            args.device,
            "delta",
        )
        for row in rows:
            row.update(
                {
                    "model": model,
                    "layer": layer,
                    "representation": "delta",
                    "configuration_order": order,
                }
            )
            order += 1
            sweep_rows.append(row)
            prediction, package = packages[int(row["config_index"])]
            package.update(
                {
                    "layer": layer,
                    "group_keys": group_center_keys,
                    "group_centers": group_centers,
                }
            )
            package_cache[
                ("mlp_pair_delta", layer, int(row["config_index"]))
            ] = (prediction, package)

        # Post-only MLP is a nuisance diagnostic. It receives the same
        # relation x outcome centering and validation procedure.
        post = controlled_values(
            model, table, positions, layer, "post"
        )
        post_center_keys, post_centers = fit_group_vector_centers(
            post, groups, train
        )
        centered_post = apply_group_centers(
            post, groups, post_center_keys, post_centers
        )
        post_rows, post_packages = mlp_candidates(
            centered_post,
            target,
            groups,
            train,
            validation,
            table,
            args.device,
            "post",
        )
        # Keep one fixed-capacity post-only family to limit search breadth.
        for row in post_rows:
            if int(row["config_index"]) != 0:
                continue
            row.update(
                {
                    "model": model,
                    "layer": layer,
                    "representation": "post",
                    "configuration_order": order,
                }
            )
            order += 1
            sweep_rows.append(row)
            prediction, package = post_packages[0]
            package.update(
                {
                    "layer": layer,
                    "group_keys": post_center_keys,
                    "group_centers": post_centers,
                }
            )
            package_cache[("mlp_pair_post", layer, 0)] = (
                prediction,
                package,
            )
        print(f"LAYER_COMPLETE model={model} layer={layer}", flush=True)

    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(out / "validation_sweep.csv", index=False)
    # Post-only is a nuisance diagnostic and is never eligible as the ticker
    # surprise representation applied to real pre/post deltas.
    best = select_row(sweep.loc[sweep.representation.eq("delta")])
    method = str(best.method)
    layer = int(best.layer)
    if method.startswith("mlp_pair"):
        key = (method, layer, int(best.config_index))
    else:
        key = (method, layer, float(best.alpha))
    prediction, selected_package = package_cache[key]
    selected_package.update(
        {
            "method": method,
            "target_group_keys": target_group_keys,
            "target_group_means": target_group_means,
            "target_group_scales": target_group_scales,
        }
    )
    selection_payload = {
        "status": "TICKER_SURPRISE_NONLINEAR_SELECTION_FROZEN",
        "model": model,
        "construct": (
            "ticker-conditioned continuous surprise relative to the model's "
            "strong pre-report company memory"
        ),
        "target": (
            "within relation x realized-outcome standardized "
            "-log P_model(outcome | ticker, relation, known outcome)"
        ),
        "eligible_rule": (
            "strong_memory_benchmark and correct controlled-report parsing; "
            "all A/B/C outcomes retained"
        ),
        "candidate_layers": layers,
        "selected_method": method,
        "selected_layer": layer,
        "selected_configuration": {
            key: (
                None
                if pd.isna(best.get(key, np.nan))
                else (
                    int(best[key])
                    if key in ("config_index", "hidden")
                    else float(best[key])
                )
            )
            for key in (
                "alpha",
                "config_index",
                "hidden",
                "weight_decay",
                "pair_weight",
            )
        },
        "validation_pair_concordance": float(
            best.validation_pair_concordance
        ),
        "validation_within_group_spearman": float(
            best.validation_within_group_spearman
        ),
        "fit_split": "recall_train_firms",
        "selection_split": "recall_validation_firms",
        "test_used_for_selection": False,
        "market_labels_used": False,
        "retrospective_extension_after_original_controlled_test_opened": True,
        "prior_controlled_selection_sha256": sha256(
            ANALYSIS / model / "selection.json"
        ),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "selection.json", selection_payload)

    suite_rows = []
    scored = table[
        [
            "controlled_id",
            "source_row",
            "cik",
            "company_name",
            "market_ticker",
            "recall_split",
            "relation",
            "outcome_letter",
            "is_confirm",
            "is_contradict",
            "violation_label",
            "directional_memory_strength",
        ]
    ].copy()
    scored["raw_firm_outcome_surprisal"] = raw_target
    scored["target_ticker_surprise_z"] = target
    scored["predicted_ticker_surprise"] = prediction
    scored.to_parquet(out / "controlled_scores.parquet", index=False)
    for suite, mask in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        suite_rows.append(
            {
                "model": model,
                "method": method,
                "layer": layer,
                "suite": suite,
                **metrics(table, target, prediction, mask),
            }
        )
    pd.DataFrame(suite_rows).to_csv(
        out / "selected_suite_metrics.csv", index=False
    )
    save_package(out / "selected_package.npz", selected_package)
    write_json(
        out / "selected_package.json",
        {
            "method": method,
            "layer": layer,
            "ensemble_seeds": list(MLP_SEEDS)
            if method.startswith("mlp_pair")
            else [],
            "activation_group_centering": "relation x realized outcome",
            "target_group_standardization": "relation x realized outcome",
            "market_labels_used": False,
        },
    )

    relation_scores, article_scores = real_news_scores(
        model, layer, selected_package
    )
    relation_scores.to_parquet(
        out / "real_news_relation_scores.parquet", index=False
    )
    article_scores.to_parquet(
        out / "real_news_article_scores.parquet", index=False
    )
    write_json(
        terminal,
        {
            "status": "TICKER_SURPRISE_NONLINEAR_COMPLETE",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "controlled_rows": len(table),
            "controlled_train_rows": int(train.sum()),
            "controlled_validation_rows": int(validation.sum()),
            "controlled_test_rows": int(test.sum()),
            "real_news_relation_rows": len(relation_scores),
            "real_news_article_rows": len(article_scores),
            "selected_method": method,
            "selected_layer": layer,
            "market_labels_used": False,
            "finite_controlled_scores": bool(
                np.isfinite(prediction).all()
            ),
            "finite_real_scores": bool(
                np.isfinite(
                    relation_scores.ticker_surprise_score.to_numpy(float)
                ).all()
            ),
        },
    )
    print(
        f"TICKER_SURPRISE_NONLINEAR_COMPLETE model={model}",
        flush=True,
    )


if __name__ == "__main__":
    main()
