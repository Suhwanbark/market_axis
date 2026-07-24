#!/usr/bin/env python3
"""Matched small-MLP heads for uncentered News representations."""

from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import news_cutoff_safe_llama2 as protocol


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "outputs" / "news_cutoff_safe_llama2"
UNCENTERED = ROOT / "outputs" / "news_uncentered_linear_control"
QWEN = ROOT / "outputs" / "news_qwen3_embedding_8b_control"
OUT = ROOT / "outputs" / "news_uncentered_mlp_control"
REPRESENTATIONS = ("bge_m3_embedding", "qwen3_embedding_8b", "llama2_activation")
SEEDS = (20260721, 20260722, 20260723, 20260724, 20260725)
HIDDEN = 64
DROPOUT = 0.1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 512
MAX_EPOCHS = 200
PATIENCE = 15


def selected_llama_layer() -> int:
    selection = json.loads((UNCENTERED / "selection.json").read_text())
    return int(selection["llama2"]["selected_layer"])


def llama_cache(split: str) -> Path:
    return OUT / "representations" / f"llama2_selected_{split}.npy"


def load_representation(name: str, split: str) -> np.ndarray:
    if name == "bge_m3_embedding":
        path = SOURCE / "representations" / f"bge_{split}.npy"
        return protocol.normalize_rows(np.load(path, mmap_mode="r"))
    if name == "qwen3_embedding_8b":
        path = QWEN / "representations" / f"qwen3_embedding_8b_{split}.npy"
        return protocol.normalize_rows(np.load(path, mmap_mode="r"))
    path = llama_cache(split)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        layer = selected_llama_layer()
        values = protocol.load_selected_llama_layer_to_ram(
            SOURCE / "representations" / f"llama2_{split}.npy", layer - 1
        )
        np.save(path, values.astype(np.float16))
        del values
    return protocol.normalize_rows(np.load(path, mmap_mode="r"))


def make_model(input_dim: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN),
        nn.GELU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN, 1),
    )


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(
    name: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    y_mean: float,
    y_std: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch

    seed_everything(seed)
    model = make_model(x_train.shape[1]).to("cuda:0")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    train_x = torch.as_tensor(x_train, dtype=torch.float32, device="cuda:0")
    val_x = torch.as_tensor(x_val, dtype=torch.float32, device="cuda:0")
    train_y = torch.as_tensor(
        (y_train - y_mean) / y_std, dtype=torch.float32, device="cuda:0"
    )
    criterion = torch.nn.MSELoss()
    best_rho = -np.inf
    best_epoch = -1
    best_state = None
    stale = 0
    generator = torch.Generator(device="cuda:0").manual_seed(seed)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator, device="cuda:0")
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_x[rows]).squeeze(1)
            loss = criterion(prediction, train_y[rows])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_prediction = (
                model(val_x).squeeze(1).float().cpu().numpy() * y_std + y_mean
            )
        rho = protocol.safe_spearman(val_prediction, y_val)
        if rho > best_rho + 1e-5:
            best_rho = rho
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(f"no finite validation checkpoint for {name} seed {seed}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        train_prediction = model(train_x).squeeze(1).float().cpu().numpy() * y_std + y_mean
        val_prediction = model(val_x).squeeze(1).float().cpu().numpy() * y_std + y_mean
    model_path = OUT / "models" / f"{name}_seed{seed}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in best_state.items()},
            "input_dim": x_train.shape[1],
            "y_mean": y_mean,
            "y_std": y_std,
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_spearman": best_rho,
        },
        model_path,
    )
    del model, optimizer, train_x, val_x, train_y
    torch.cuda.empty_cache()
    return train_prediction, val_prediction, {
        "representation": name,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_spearman": best_rho,
    }


def train() -> None:
    if (OUT / "selection.json").exists():
        raise RuntimeError("MLP selection is already frozen")
    documents = pd.read_parquet(SOURCE / "documents.parquet")
    train_frame = protocol.split_documents(documents, "train")
    val_frame = protocol.split_documents(documents, "val")
    y_train = train_frame[protocol.LABEL].to_numpy(float)
    y_val = val_frame[protocol.LABEL].to_numpy(float)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    pretest = pd.concat([train_frame, val_frame], ignore_index=True)[
        ["doc_row", "ticker", "date", "split", protocol.LABEL]
    ].copy()
    seed_rows: list[dict] = []
    representation_rows: list[dict] = []
    for name in REPRESENTATIONS:
        x_train = load_representation(name, "train")
        x_val = load_representation(name, "val")
        train_predictions = []
        val_predictions = []
        for seed in SEEDS:
            train_prediction, val_prediction, row = train_one(
                name, seed, x_train, y_train, x_val, y_val, y_mean, y_std
            )
            train_predictions.append(train_prediction)
            val_predictions.append(val_prediction)
            seed_rows.append(row)
            print(
                f"MLP {name} seed {seed}: epoch={row['best_epoch']} "
                f"val_rho={row['validation_spearman']:.6f}",
                flush=True,
            )
        train_score = np.mean(train_predictions, axis=0)
        val_score = np.mean(val_predictions, axis=0)
        column = f"{name}_mlp_score"
        pretest[column] = np.concatenate([train_score, val_score]).astype(np.float32)
        representation_rows.append(
            {
                "representation": name,
                "ensemble_validation_spearman": protocol.safe_spearman(val_score, y_val),
                "ensemble_validation_pearson": protocol.safe_pearson(val_score, y_val),
            }
        )
        del x_train, x_val, train_predictions, val_predictions
    OUT.mkdir(parents=True, exist_ok=True)
    pretest.to_parquet(OUT / "pretest_scores.parquet", index=False)
    pd.DataFrame(seed_rows).to_csv(OUT / "seed_validation_metrics.csv", index=False)
    pd.DataFrame(representation_rows).to_csv(
        OUT / "validation_metrics.csv", index=False
    )
    selection = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "centering": "none; no ticker means subtracted",
        "target_scaling": {"train_global_mean": y_mean, "train_global_std": y_std},
        "architecture": f"Linear(input,{HIDDEN})-GELU-Dropout({DROPOUT})-Linear({HIDDEN},1)",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping": f"validation Spearman, patience {PATIENCE}",
        "seeds": list(SEEDS),
        "ensemble": "arithmetic mean of five fixed-seed predictions",
        "llama_layer": selected_llama_layer(),
        "representations": representation_rows,
        "test_loaded_during_training": False,
    }
    protocol.write_json(OUT / "selection.json", selection)
    print(json.dumps(selection, indent=2), flush=True)


def ensemble_predict(name: str, values: np.ndarray) -> np.ndarray:
    import torch

    x = torch.as_tensor(values, dtype=torch.float32, device="cuda:0")
    predictions = []
    for seed in SEEDS:
        checkpoint = torch.load(
            OUT / "models" / f"{name}_seed{seed}.pt",
            map_location="cpu",
            weights_only=True,
        )
        model = make_model(int(checkpoint["input_dim"])).to("cuda:0")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.inference_mode():
            prediction = (
                model(x).squeeze(1).float().cpu().numpy() * float(checkpoint["y_std"])
                + float(checkpoint["y_mean"])
            )
        predictions.append(prediction)
        del model
    del x
    torch.cuda.empty_cache()
    return np.mean(predictions, axis=0).astype(np.float32)


def evaluate(repetitions: int, workers: int) -> None:
    if not (OUT / "selection.json").exists():
        raise RuntimeError("train and freeze MLP heads first")
    if (OUT / "test_metrics.csv").exists():
        raise RuntimeError("MLP test has already been evaluated")
    documents = pd.read_parquet(SOURCE / "documents.parquet")
    test = protocol.split_documents(documents, "test")
    target = test[protocol.LABEL].to_numpy(float)
    existing = pd.read_parquet(QWEN / "all_model_scores.parquet").sort_values("doc_row")
    existing_test = existing.loc[existing["split"].eq("test")].reset_index(drop=True)
    score_columns = {
        "bge_m3_embedding": "embedding_score",
        "qwen3_embedding_8b": "qwen3_embedding_score",
        "llama2_activation": "activation_score",
    }
    test_scores = test[["doc_row", "ticker", "date", "split", protocol.LABEL]].copy()
    metrics: list[dict] = []
    comparisons: list[dict] = []
    frame = test[["date", "ticker"]].reset_index(drop=True)
    for name in REPRESENTATIONS:
        values = load_representation(name, "test")
        mlp_score = ensemble_predict(name, values)
        ridge_score = existing_test[score_columns[name]].to_numpy(float)
        test_scores[f"{name}_mlp_score"] = mlp_score
        for head, score in (("ridge", ridge_score), ("mlp", mlp_score)):
            metrics.append(
                {
                    "representation": name,
                    "head": head,
                    "test_spearman": protocol.safe_spearman(score, target),
                    "test_pearson": protocol.safe_pearson(score, target),
                }
            )
        for block in ("date", "ticker"):
            comparisons.append(
                {
                    "comparison": f"{name}_mlp_minus_ridge",
                    **protocol.parallel_bootstrap(
                        frame, mlp_score, ridge_score, target, block, repetitions, workers
                    ),
                }
            )
    activation = test_scores["llama2_activation_mlp_score"].to_numpy(float)
    qwen = test_scores["qwen3_embedding_8b_mlp_score"].to_numpy(float)
    for block in ("date", "ticker"):
        comparisons.append(
            {
                "comparison": "llama2_activation_mlp_minus_qwen3_embedding_8b_mlp",
                **protocol.parallel_bootstrap(
                    frame, activation, qwen, target, block, repetitions, workers
                ),
            }
        )
    metrics_frame = pd.DataFrame(metrics)
    comparison_frame = pd.DataFrame(comparisons)
    metrics_frame.to_csv(OUT / "test_metrics.csv", index=False)
    comparison_frame.to_csv(OUT / "test_correlation_bootstrap.csv", index=False)
    all_mlp = pd.concat(
        [pd.read_parquet(OUT / "pretest_scores.parquet"), test_scores],
        ignore_index=True,
    ).sort_values("doc_row")
    all_scores = existing.merge(
        all_mlp[
            ["doc_row", *[f"{name}_mlp_score" for name in REPRESENTATIONS]]
        ],
        on="doc_row",
        how="inner",
        validate="one_to_one",
    )
    all_scores.to_parquet(OUT / "all_model_scores.parquet", index=False)
    print(metrics_frame.to_string(index=False), flush=True)
    print(comparison_frame.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train")
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    evaluate_parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    if args.command == "train":
        train()
    else:
        evaluate(args.bootstrap_repetitions, args.workers)


if __name__ == "__main__":
    main()
