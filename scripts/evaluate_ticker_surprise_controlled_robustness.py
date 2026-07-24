#!/usr/bin/env python3
"""Held-out robustness checks for the frozen nonlinear ticker-surprise readout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ticker_surprise_nonlinear_finder as finder  # noqa: E402


SOURCE = ROOT / "outputs" / "ticker_surprise_nonlinear"
ANALYSIS = ROOT / "outputs" / "retrieval_conditioned_analysis"
MODELS = ("qwen25_7b", "qwen3_4b")


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sampled_indices(
    test_table: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    firms = test_table.cik.astype(str).unique()
    blocks = {
        firm: np.flatnonzero(
            test_table.cik.astype(str).to_numpy() == firm
        )
        for firm in firms
    }
    sampled = rng.choice(firms, size=len(firms), replace=True)
    return np.concatenate([blocks[firm] for firm in sampled])


def metric_pair(
    target: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, float]:
    mask = np.ones(len(target), dtype=bool)
    return (
        finder.pair_concordance(target, prediction, groups, mask),
        finder.within_group_spearman(target, prediction, groups, mask),
    )


def evaluate_model(
    model: str, repeats: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_dir = SOURCE / model
    manifest = json.loads((model_dir / "manifest.json").read_text())
    if manifest.get("status") != "TICKER_SURPRISE_NONLINEAR_COMPLETE":
        raise RuntimeError(f"incomplete nonlinear finder: {model}")

    table_all = pd.read_parquet(
        ANALYSIS / model / "controlled_metadata.parquet"
    ).reset_index(drop=True)
    eligible = (
        table_all.strong_memory_benchmark.astype(bool)
        & table_all.post_correct.astype(bool)
    )
    positions = np.flatnonzero(eligible.to_numpy())
    table = table_all.loc[eligible].reset_index(drop=True)
    groups = finder.group_keys(table)
    split = table.recall_split.astype(str).to_numpy()
    train = split == "train"
    test = split == "test"
    raw_target = finder.continuous_target(table)
    target, _, _, _ = finder.fit_group_scalar_normalizer(
        raw_target, groups, train
    )

    sweep = pd.read_csv(model_dir / "validation_sweep.csv")
    linear = (
        sweep.loc[sweep.method.eq("linear_delta")]
        .sort_values(
            [
                "validation_pair_concordance",
                "validation_within_group_spearman",
                "configuration_order",
            ],
            ascending=[False, False, True],
        )
        .iloc[0]
    )
    layer = int(linear.layer)
    alpha = float(linear.alpha)
    delta = finder.controlled_values(
        model, table, positions, layer, "delta"
    )
    center_keys, centers = finder.fit_group_vector_centers(
        delta, groups, train
    )
    centered = finder.apply_group_centers(
        delta, groups, center_keys, centers
    )
    standardized, _, _ = finder.standardize(centered, train)
    estimator = Ridge(alpha=alpha, solver="lsqr", tol=1e-5)
    estimator.fit(standardized[train], target[train])
    linear_prediction = estimator.predict(standardized)

    selected = pd.read_parquet(model_dir / "controlled_scores.parquet")
    if selected.controlled_id.tolist() != table.controlled_id.tolist():
        raise RuntimeError(f"controlled row identity mismatch: {model}")
    mlp_prediction = selected.predicted_ticker_surprise.to_numpy(float)

    test_table = table.loc[test].reset_index(drop=True)
    test_target = target[test]
    test_groups = groups[test]
    test_mlp = mlp_prediction[test]
    test_linear = np.asarray(linear_prediction[test], float)
    mlp_pair, mlp_rho = metric_pair(
        test_target, test_mlp, test_groups
    )
    linear_pair, linear_rho = metric_pair(
        test_target, test_linear, test_groups
    )

    rng = np.random.default_rng(seed)
    boot = np.empty((repeats, 4), dtype=float)
    for draw in range(repeats):
        index = sampled_indices(test_table, rng)
        boot[draw, :2] = metric_pair(
            test_target[index], test_mlp[index], test_groups[index]
        )
        boot[draw, 2:] = metric_pair(
            test_target[index], test_linear[index], test_groups[index]
        )

    permutation = np.empty((repeats, 2), dtype=float)
    group_positions = [
        np.flatnonzero(test_groups == key)
        for key in np.unique(test_groups)
    ]
    for draw in range(repeats):
        permuted = test_mlp.copy()
        for index in group_positions:
            permuted[index] = rng.permutation(permuted[index])
        permutation[draw] = metric_pair(
            test_target, permuted, test_groups
        )

    metric_rows = []
    for metric_index, (metric_name, mlp_point, linear_point) in enumerate(
        (
            ("pair_concordance", mlp_pair, linear_pair),
            ("within_group_spearman", mlp_rho, linear_rho),
        )
    ):
        mlp_draw = boot[:, metric_index]
        linear_draw = boot[:, metric_index + 2]
        difference = mlp_draw - linear_draw
        null = permutation[:, metric_index]
        metric_rows.append(
            {
                "model": model,
                "metric": metric_name,
                "test_rows": int(test.sum()),
                "test_firms": int(test_table.cik.nunique()),
                "mlp_point": mlp_point,
                "mlp_firm_boot_ci_low": np.quantile(mlp_draw, 0.025),
                "mlp_firm_boot_ci_high": np.quantile(mlp_draw, 0.975),
                "linear_point": linear_point,
                "mlp_minus_linear": mlp_point - linear_point,
                "difference_firm_boot_ci_low": np.quantile(
                    difference, 0.025
                ),
                "difference_firm_boot_ci_high": np.quantile(
                    difference, 0.975
                ),
                "bootstrap_probability_mlp_gt_linear": np.mean(
                    difference > 0
                ),
                "stratified_permutation_p_one_sided": (
                    1 + np.sum(null >= mlp_point)
                )
                / (repeats + 1),
                "permutation_repeats": repeats,
                "linear_layer": layer,
                "linear_alpha": alpha,
            }
        )

    predictions = table[
        [
            "controlled_id",
            "cik",
            "recall_split",
            "relation",
            "outcome_letter",
        ]
    ].copy()
    predictions["target_ticker_surprise_z"] = target
    predictions["mlp_prediction"] = mlp_prediction
    predictions["linear_prediction"] = linear_prediction

    bootstrap_rows = pd.DataFrame(
        {
            "model": model,
            "draw": np.arange(repeats),
            "mlp_pair_concordance": boot[:, 0],
            "mlp_within_group_spearman": boot[:, 1],
            "linear_pair_concordance": boot[:, 2],
            "linear_within_group_spearman": boot[:, 3],
        }
    )
    return pd.DataFrame(metric_rows), predictions, bootstrap_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2_000)
    args = parser.parse_args()
    output = SOURCE / "controlled_robustness"
    output.mkdir(parents=True, exist_ok=True)
    terminal = output / "manifest.json"
    if terminal.exists():
        status = json.loads(terminal.read_text()).get("status")
        if status == "TICKER_SURPRISE_CONTROLLED_ROBUSTNESS_COMPLETE":
            print(f"already complete: {terminal}")
            return
        raise RuntimeError(f"non-terminal manifest exists: {terminal}")

    metric_frames = []
    for model_index, model in enumerate(MODELS):
        metrics, predictions, bootstrap = evaluate_model(
            model, args.repeats, 17_071 + model_index
        )
        metric_frames.append(metrics)
        predictions.to_parquet(
            output / f"{model}_predictions.parquet", index=False
        )
        bootstrap.to_parquet(
            output / f"{model}_firm_bootstrap.parquet", index=False
        )
        print(f"CONTROLLED_ROBUSTNESS_MODEL_COMPLETE model={model}")
    pd.concat(metric_frames, ignore_index=True).to_csv(
        output / "test_robustness.csv", index=False
    )
    write_json(
        terminal,
        {
            "status": "TICKER_SURPRISE_CONTROLLED_ROBUSTNESS_COMPLETE",
            "models": list(MODELS),
            "repeats": args.repeats,
            "gpu_used": False,
            "test_used_for_model_selection": False,
        },
    )
    print("TICKER_SURPRISE_CONTROLLED_ROBUSTNESS_COMPLETE")


if __name__ == "__main__":
    main()
