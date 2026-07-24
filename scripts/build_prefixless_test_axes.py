#!/usr/bin/env python3
"""Reproduce controlled-selected linear axes in raw activation coordinates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ticker_surprise_nonlinear_finder as finder  # noqa: E402


ANALYSIS = ROOT / "outputs" / "retrieval_conditioned_analysis"
SOURCE = ROOT / "outputs" / "ticker_surprise_nonlinear"
OUTPUT = ROOT / "outputs" / "prefixless_article_axis_test" / "axes"
MODELS = ("qwen25_7b", "qwen3_4b")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    terminal = OUTPUT / "manifest.json"
    if terminal.exists():
        status = json.loads(terminal.read_text()).get("status")
        if status == "PREFIXLESS_TEST_AXES_FROZEN":
            print(f"already complete: {terminal}")
            return
        raise RuntimeError(f"non-terminal manifest exists: {terminal}")

    payload: dict[str, object] = {
        "status": "PREFIXLESS_TEST_AXES_FROZEN",
        "market_labels_used": False,
        "test_articles_used": False,
        "models": {},
    }
    for model in MODELS:
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
        validation = split == "validation"
        raw_target = finder.continuous_target(table)
        target, _, _, _ = finder.fit_group_scalar_normalizer(
            raw_target, groups, train
        )

        sweep = pd.read_csv(SOURCE / model / "validation_sweep.csv")
        selected = (
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
        layer = int(selected.layer)
        alpha = float(selected.alpha)
        delta = finder.controlled_values(
            model, table, positions, layer, "delta"
        )
        center_keys, centers = finder.fit_group_vector_centers(
            delta, groups, train
        )
        centered = finder.apply_group_centers(
            delta, groups, center_keys, centers
        )
        standardized, mean, scale = finder.standardize(centered, train)
        estimator = Ridge(alpha=alpha, solver="lsqr", tol=1e-5)
        estimator.fit(standardized[train], target[train])
        raw_coefficient = np.asarray(estimator.coef_, np.float32) / scale
        unit_axis = raw_coefficient / np.linalg.norm(raw_coefficient)
        np.savez_compressed(
            OUTPUT / f"{model}.npz",
            unit_axis=unit_axis.astype(np.float32),
            raw_coefficient=raw_coefficient.astype(np.float32),
            standardized_coefficient=np.asarray(
                estimator.coef_, np.float32
            ),
            feature_mean=mean,
            feature_scale=scale,
            group_keys=center_keys,
            group_centers=centers,
            intercept=np.asarray([estimator.intercept_], np.float32),
        )
        payload["models"][model] = {
            "layer": layer,
            "alpha": alpha,
            "hidden": len(unit_axis),
            "train_rows": int(train.sum()),
            "validation_rows": int(validation.sum()),
            "validation_pair_concordance": float(
                selected.validation_pair_concordance
            ),
            "selection": (
                "best linear_delta layer/alpha by controlled validation "
                "pair concordance then within-group Spearman"
            ),
        }
        print(
            f"PREFIXLESS_AXIS_MODEL_COMPLETE model={model} layer={layer}",
            flush=True,
        )
    terminal.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print("PREFIXLESS_TEST_AXES_FROZEN")


if __name__ == "__main__":
    main()
