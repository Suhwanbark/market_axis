#!/usr/bin/env python3
"""Re-fit the News BGE and Llama axes without any ticker centering."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import news_cutoff_safe_llama2 as protocol


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "outputs" / "news_cutoff_safe_llama2"
OUT = ROOT / "outputs" / "news_uncentered_linear_control"


def fit_state(x_train: np.ndarray, y_train: np.ndarray, alpha: float, path: Path):
    model = protocol.fit_final(x_train, y_train, alpha)
    np.savez_compressed(
        path,
        coef=model.coef_.astype(np.float32),
        intercept=np.float64(model.intercept_),
    )
    return model


def select(workers: int) -> None:
    if (OUT / "selection.json").exists():
        raise RuntimeError("uncentered selection already exists")
    OUT.mkdir(parents=True, exist_ok=True)
    documents = pd.read_parquet(SOURCE / "documents.parquet")
    train = protocol.split_documents(documents, "train")
    validation = protocol.split_documents(documents, "val")
    y_train = train[protocol.LABEL].to_numpy(float)
    y_val = validation[protocol.LABEL].to_numpy(float)
    sweep: list[dict] = []

    bge_train = protocol.normalize_rows(
        np.load(SOURCE / "representations" / "bge_train.npy", mmap_mode="r")
    )
    bge_val = protocol.normalize_rows(
        np.load(SOURCE / "representations" / "bge_val.npy", mmap_mode="r")
    )
    bge_rows = protocol.tune_alphas(
        bge_train, y_train, bge_val, y_val, workers, "cuda"
    )
    sweep.extend({"representation": "bge_m3_embedding", "layer": np.nan, **row} for row in bge_rows)
    bge_best = sorted(
        bge_rows, key=lambda row: (-row["validation_spearman"], row["alpha"])
    )[0]
    bge_model = fit_state(
        bge_train, y_train, float(bge_best["alpha"]), OUT / "bge_state.npz"
    )

    llama_train = protocol.load_llama_layers_to_ram(
        SOURCE / "representations" / "llama2_train.npy"
    )
    llama_val = protocol.load_llama_layers_to_ram(
        SOURCE / "representations" / "llama2_val.npy"
    )
    for layer in range(1, llama_train.shape[0] + 1):
        x_train = protocol.normalize_rows(llama_train[layer - 1])
        x_val = protocol.normalize_rows(llama_val[layer - 1])
        rows = protocol.tune_alphas(x_train, y_train, x_val, y_val, workers, "cuda")
        sweep.extend(
            {"representation": "llama2_activation", "layer": layer, **row}
            for row in rows
        )
        print(f"uncentered selection: Llama layer {layer}/{llama_train.shape[0]}", flush=True)
    llama_rows = [row for row in sweep if row["representation"] == "llama2_activation"]
    llama_best = sorted(
        llama_rows,
        key=lambda row: (-row["validation_spearman"], row["layer"], row["alpha"]),
    )[0]
    layer = int(llama_best["layer"])
    llama_x_train = protocol.normalize_rows(llama_train[layer - 1])
    llama_x_val = protocol.normalize_rows(llama_val[layer - 1])
    llama_model = fit_state(
        llama_x_train,
        y_train,
        float(llama_best["alpha"]),
        OUT / "llama2_state.npz",
    )
    pretest = pd.concat([train, validation], ignore_index=True)[
        ["doc_row", "ticker", "date", "split", protocol.LABEL]
    ].copy()
    pretest["impact_label"] = np.concatenate([y_train, y_val])
    pretest["embedding_score"] = np.concatenate(
        [bge_model.predict(bge_train), bge_model.predict(bge_val)]
    ).astype(np.float32)
    pretest["activation_score"] = np.concatenate(
        [llama_model.predict(llama_x_train), llama_model.predict(llama_x_val)]
    ).astype(np.float32)
    pretest.to_parquet(OUT / "pretest_scores.parquet", index=False)
    pd.DataFrame(sweep).to_csv(OUT / "validation_sweep.csv", index=False)
    selection = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "News uncentered linear control",
        "centering": "none; no ticker means subtracted from labels or representations",
        "label": protocol.LABEL,
        "axis_fit_split": "2019-2021 train only",
        "selection_split": "2022 validation only",
        "test_split": "2023 exploratory; representations already existed from prior protocol",
        "alpha_grid": list(protocol.ALPHAS),
        "bge": {
            "selected_alpha": float(bge_best["alpha"]),
            "validation_spearman": float(bge_best["validation_spearman"]),
        },
        "llama2": {
            "selected_layer": layer,
            "selected_alpha": float(llama_best["alpha"]),
            "validation_spearman": float(llama_best["validation_spearman"]),
        },
    }
    protocol.write_json(OUT / "selection.json", selection)
    print(json.dumps(selection, indent=2), flush=True)


def apply(values: np.ndarray, path: Path) -> np.ndarray:
    state = np.load(path)
    return (values @ state["coef"] + float(state["intercept"])).astype(np.float32)


def evaluate(repetitions: int, workers: int) -> None:
    if (OUT / "test_metrics.csv").exists():
        raise RuntimeError("uncentered test already evaluated")
    selection = json.loads((OUT / "selection.json").read_text())
    documents = pd.read_parquet(SOURCE / "documents.parquet")
    test = protocol.split_documents(documents, "test")
    target = test[protocol.LABEL].to_numpy(float)
    bge = protocol.normalize_rows(
        np.load(SOURCE / "representations" / "bge_test.npy", mmap_mode="r")
    )
    bge_score = apply(bge, OUT / "bge_state.npz")
    layer = int(selection["llama2"]["selected_layer"])
    llama = protocol.normalize_rows(
        protocol.load_selected_llama_layer_to_ram(
            SOURCE / "representations" / "llama2_test.npy", layer - 1
        )
    )
    llama_score = apply(llama, OUT / "llama2_state.npz")
    metrics = pd.DataFrame(
        [
            {
                "method": "bge_m3_embedding",
                "selected_layer": np.nan,
                "selected_alpha": selection["bge"]["selected_alpha"],
                "n_test": len(test),
                "test_spearman": protocol.safe_spearman(bge_score, target),
                "test_pearson": protocol.safe_pearson(bge_score, target),
            },
            {
                "method": "llama2_activation",
                "selected_layer": layer,
                "selected_alpha": selection["llama2"]["selected_alpha"],
                "n_test": len(test),
                "test_spearman": protocol.safe_spearman(llama_score, target),
                "test_pearson": protocol.safe_pearson(llama_score, target),
            },
        ]
    )
    metrics.to_csv(OUT / "test_metrics.csv", index=False)
    frame = test[["date", "ticker"]].reset_index(drop=True)
    bootstrap = pd.DataFrame(
        [
            protocol.parallel_bootstrap(
                frame, llama_score, bge_score, target, block, repetitions, workers
            )
            for block in ("date", "ticker")
        ]
    )
    bootstrap.to_csv(OUT / "activation_vs_bge_bootstrap.csv", index=False)
    scores = test[["doc_row", "ticker", "date", "split", protocol.LABEL]].copy()
    scores["impact_label"] = target
    scores["embedding_score"] = bge_score
    scores["activation_score"] = llama_score
    all_scores = pd.concat(
        [pd.read_parquet(OUT / "pretest_scores.parquet"), scores], ignore_index=True
    ).sort_values("doc_row")
    all_scores.to_parquet(OUT / "all_split_scores.parquet", index=False)
    print(metrics.to_string(index=False), flush=True)
    print(bootstrap.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--workers", type=int, default=6)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    evaluate_parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    if args.command == "select":
        select(args.workers)
    else:
        evaluate(args.bootstrap_repetitions, args.workers)


if __name__ == "__main__":
    main()
