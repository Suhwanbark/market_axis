#!/usr/bin/env python3
"""Post-hoc 8-K sensitivity: remove train-ticker label and representation means."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import news_cutoff_safe_llama2 as protocol
import sec8k_uncentered_axis as axis


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "sec8k_ticker_effect_sensitivity"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = axis.part("train")
    val = axis.part("val")
    test = axis.part("test")
    tickers = sorted(train["ticker"].unique())
    train_tickers = train["ticker"].to_numpy(str)
    val_tickers = val["ticker"].to_numpy(str)
    test_tickers = test["ticker"].to_numpy(str)
    label_means = protocol.label_centering(
        train[axis.LABEL].to_numpy(float), train_tickers, tickers
    )
    y_train = protocol.center_labels(
        train[axis.LABEL].to_numpy(float), train_tickers, tickers, label_means
    )
    y_val = protocol.center_labels(
        val[axis.LABEL].to_numpy(float), val_tickers, tickers, label_means
    )
    y_test = protocol.center_labels(
        test[axis.LABEL].to_numpy(float), test_tickers, tickers, label_means
    )
    selected = {}
    test_scores = {}
    sweeps = []
    for name in ("bge", "qwen3"):
        x_train = protocol.normalize_rows(np.load(axis.rep_path(name, "train"), mmap_mode="r"))
        x_val = protocol.normalize_rows(np.load(axis.rep_path(name, "val"), mmap_mode="r"))
        x_test = protocol.normalize_rows(np.load(axis.rep_path(name, "test"), mmap_mode="r"))
        means = protocol.ticker_means_matrix(x_train, train_tickers, tickers)
        x_train = protocol.center_matrix(x_train, train_tickers, tickers, means)
        x_val = protocol.center_matrix(x_val, val_tickers, tickers, means)
        x_test = protocol.center_matrix(x_test, test_tickers, tickers, means)
        rows = protocol.tune_alphas(x_train, y_train, x_val, y_val, 6, "cpu")
        sweeps.extend({"representation": name, "layer": np.nan, **row} for row in rows)
        best = min(rows, key=lambda row: (-row["validation_spearman"], row["alpha"]))
        model = protocol.fit_final(x_train, y_train, float(best["alpha"]))
        test_scores[name] = model.predict(x_test).astype(np.float32)
        selected[name] = best
        print(f"centered 8-K {name}: {best}", flush=True)

    llama_train = protocol.load_llama_layers_to_ram(axis.rep_path("llama2", "train"))
    llama_val = protocol.load_llama_layers_to_ram(axis.rep_path("llama2", "val"))
    for layer in range(1, llama_train.shape[0] + 1):
        x_train = protocol.normalize_rows(llama_train[layer - 1])
        x_val = protocol.normalize_rows(llama_val[layer - 1])
        means = protocol.ticker_means_matrix(x_train, train_tickers, tickers)
        x_train = protocol.center_matrix(x_train, train_tickers, tickers, means)
        x_val = protocol.center_matrix(x_val, val_tickers, tickers, means)
        rows = protocol.tune_alphas(x_train, y_train, x_val, y_val, 6, "cpu")
        sweeps.extend(
            {"representation": "llama2", "layer": layer, **row} for row in rows
        )
        print(f"centered 8-K Llama layer {layer}/32", flush=True)
    candidates = [row for row in sweeps if row["representation"] == "llama2"]
    best = min(
        candidates,
        key=lambda row: (-row["validation_spearman"], row["layer"], row["alpha"]),
    )
    layer = int(best["layer"])
    x_train = protocol.normalize_rows(llama_train[layer - 1])
    means = protocol.ticker_means_matrix(x_train, train_tickers, tickers)
    x_train = protocol.center_matrix(x_train, train_tickers, tickers, means)
    x_test = protocol.normalize_rows(
        protocol.load_selected_llama_layer_to_ram(axis.rep_path("llama2", "test"), layer - 1)
    )
    x_test = protocol.center_matrix(x_test, test_tickers, tickers, means)
    model = protocol.fit_final(x_train, y_train, float(best["alpha"]))
    test_scores["llama2"] = model.predict(x_test).astype(np.float32)
    selected["llama2"] = best

    metrics = pd.DataFrame(
        [
            {
                "representation": name,
                "selected_layer": selected[name].get("layer", np.nan),
                "selected_alpha": selected[name]["alpha"],
                "validation_spearman": selected[name]["validation_spearman"],
                "test_spearman": protocol.safe_spearman(score, y_test),
                "test_pearson": protocol.safe_pearson(score, y_test),
            }
            for name, score in test_scores.items()
        ]
    )
    metrics.to_csv(OUT / "centered_test_metrics.csv", index=False)
    pd.DataFrame(sweeps).to_csv(OUT / "centered_validation_sweep.csv", index=False)
    uncentered = pd.read_csv(ROOT / "outputs" / "sec8k_uncentered" / "ridge_test_metrics.csv")
    comparison = metrics.merge(
        uncentered[["method", "test_spearman"]].rename(
            columns={"method": "representation", "test_spearman": "uncentered_test_spearman"}
        ),
        on="representation",
        validate="one_to_one",
    )
    comparison["centered_minus_uncentered"] = (
        comparison["test_spearman"] - comparison["uncentered_test_spearman"]
    )
    comparison.to_csv(OUT / "centered_vs_uncentered.csv", index=False)
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "sensitivity_only": True,
                "centering": "subtract train-ticker representation centroid and raw-label mean from train/val/test",
                "selection": "centered 2024 validation Spearman",
                "test": "2025",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
