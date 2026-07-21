#!/usr/bin/env python3
"""Matched small-MLP heads for the uncentered 8-K representations."""

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
import sec8k_uncentered_axis as axis


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "outputs" / "sec8k_uncentered"
OUT = ROOT / "outputs" / "sec8k_uncentered_mlp"
NAMES = ("bge", "qwen3", "llama2")
SEEDS = (20260721, 20260722, 20260723, 20260724, 20260725)
HIDDEN = 64
DROPOUT = 0.1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 512
MAX_EPOCHS = 200
PATIENCE = 15


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


def llama_layer() -> int:
    selection = json.loads((SOURCE / "selection.json").read_text())
    return int(selection["models"]["llama2"]["selected_layer"])


def load_representation(name: str, split: str) -> np.ndarray:
    if name != "llama2":
        return protocol.normalize_rows(np.load(axis.rep_path(name, split), mmap_mode="r"))
    cache = OUT / "representations" / f"llama2_layer{llama_layer()}_{split}.npy"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        values = protocol.load_selected_llama_layer_to_ram(
            axis.rep_path("llama2", split), llama_layer() - 1
        )
        np.save(cache, values.astype(np.float16))
    return protocol.normalize_rows(np.load(cache, mmap_mode="r"))


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
    generator = torch.Generator(device="cuda:0").manual_seed(seed)
    best_rho = -np.inf
    best_epoch = -1
    best_state = None
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator, device="cuda:0")
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(train_x[rows]).squeeze(1), train_y[rows])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = model(val_x).squeeze(1).float().cpu().numpy() * y_std + y_mean
        rho = protocol.safe_spearman(prediction, y_val)
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
        raise RuntimeError(f"no checkpoint for {name} seed {seed}")
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
        raise RuntimeError("8-K MLP selection already frozen")
    train_frame = axis.part("train")
    val_frame = axis.part("val")
    y_train = train_frame[axis.LABEL].to_numpy(float)
    y_val = val_frame[axis.LABEL].to_numpy(float)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    pretest = pd.concat([train_frame, val_frame], ignore_index=True)[
        ["doc_row", "split_row", "ticker", "file_date", "event_session", "split", axis.LABEL]
    ].copy()
    seed_rows = []
    aggregate_rows = []
    for name in NAMES:
        x_train = load_representation(name, "train")
        x_val = load_representation(name, "val")
        train_predictions = []
        val_predictions = []
        for seed in SEEDS:
            train_score, val_score, row = train_one(
                name, seed, x_train, y_train, x_val, y_val, y_mean, y_std
            )
            train_predictions.append(train_score)
            val_predictions.append(val_score)
            seed_rows.append(row)
            print(
                f"MLP {name} seed={seed}: epoch={row['best_epoch']} "
                f"val_rho={row['validation_spearman']:.6f}",
                flush=True,
            )
        train_ensemble = np.mean(train_predictions, axis=0)
        val_ensemble = np.mean(val_predictions, axis=0)
        pretest[f"{name}_mlp_score"] = np.concatenate(
            [train_ensemble, val_ensemble]
        ).astype(np.float32)
        aggregate_rows.append(
            {
                "representation": name,
                "validation_spearman": protocol.safe_spearman(val_ensemble, y_val),
                "validation_pearson": protocol.safe_pearson(val_ensemble, y_val),
            }
        )
        del x_train, x_val
    OUT.mkdir(parents=True, exist_ok=True)
    pretest["impact_label"] = pretest[axis.LABEL]
    pretest.to_parquet(OUT / "pretest_scores.parquet", index=False)
    pd.DataFrame(seed_rows).to_csv(OUT / "seed_validation_metrics.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(OUT / "validation_metrics.csv", index=False)
    protocol.write_json(
        OUT / "selection.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "architecture": f"Linear(input,{HIDDEN})-GELU-Dropout({DROPOUT})-Linear({HIDDEN},1)",
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping": f"2024 validation Spearman; patience {PATIENCE}",
            "seeds": list(SEEDS),
            "ensemble": "arithmetic mean of five fixed seeds",
            "target_scaling": {"train_global_mean": y_mean, "train_global_std": y_std},
            "centering": "none; no ticker mean subtraction",
            "llama_layer": llama_layer(),
            "validation_metrics": aggregate_rows,
            "test_used_for_training_or_early_stopping": False,
        },
    )
    print(pd.DataFrame(aggregate_rows).to_string(index=False), flush=True)


def ensemble_predict(name: str, values: np.ndarray) -> np.ndarray:
    import torch

    tensor = torch.as_tensor(values, dtype=torch.float32, device="cuda:0")
    predictions = []
    for seed in SEEDS:
        state = torch.load(
            OUT / "models" / f"{name}_seed{seed}.pt",
            map_location="cpu",
            weights_only=True,
        )
        model = make_model(int(state["input_dim"])).to("cuda:0")
        model.load_state_dict(state["state_dict"])
        model.eval()
        with torch.inference_mode():
            prediction = model(tensor).squeeze(1).float().cpu().numpy()
        predictions.append(prediction * float(state["y_std"]) + float(state["y_mean"]))
        del model
    del tensor
    torch.cuda.empty_cache()
    return np.mean(predictions, axis=0).astype(np.float32)


def evaluate(repetitions: int, workers: int) -> None:
    if not (OUT / "selection.json").exists():
        raise RuntimeError("train the 8-K MLP first")
    if (OUT / "test_metrics.csv").exists():
        raise RuntimeError("8-K MLP test already evaluated")
    test = axis.part("test")
    target = test[axis.LABEL].to_numpy(float)
    ridge = pd.read_parquet(SOURCE / "ridge_all_scores.parquet")
    ridge_test = ridge.loc[ridge["split"].eq("test")].sort_values("split_row").reset_index(drop=True)
    if not np.array_equal(ridge_test["doc_row"].to_numpy(), test["doc_row"].to_numpy()):
        raise RuntimeError("Ridge/test rows do not align")
    scores: dict[str, np.ndarray] = {}
    rows = []
    for name in NAMES:
        scores[name] = ensemble_predict(name, load_representation(name, "test"))
        rows.extend(
            [
                {
                    "representation": name,
                    "head": "ridge",
                    "test_spearman": protocol.safe_spearman(
                        ridge_test[f"{name}_ridge_score"], target
                    ),
                    "test_pearson": protocol.safe_pearson(
                        ridge_test[f"{name}_ridge_score"], target
                    ),
                },
                {
                    "representation": name,
                    "head": "mlp",
                    "test_spearman": protocol.safe_spearman(scores[name], target),
                    "test_pearson": protocol.safe_pearson(scores[name], target),
                },
            ]
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "test_metrics.csv", index=False)
    frame = test[["event_session", "ticker"]].rename(columns={"event_session": "date"})
    comparisons = []
    for name in NAMES:
        ridge_score = ridge_test[f"{name}_ridge_score"].to_numpy(float)
        for block in ("date", "ticker"):
            comparisons.append(
                {
                    "comparison": f"{name}_mlp_minus_ridge",
                    **protocol.parallel_bootstrap(
                        frame, scores[name], ridge_score, target, block, repetitions, workers
                    ),
                }
            )
    for block in ("date", "ticker"):
        comparisons.append(
            {
                "comparison": "llama2_mlp_minus_qwen3_mlp",
                **protocol.parallel_bootstrap(
                    frame, scores["llama2"], scores["qwen3"], target, block, repetitions, workers
                ),
            }
        )
    pd.DataFrame(comparisons).to_csv(OUT / "correlation_bootstrap.csv", index=False)
    test_scores = test[
        ["doc_row", "split_row", "ticker", "file_date", "event_session", "split", axis.LABEL]
    ].copy()
    for name, score in scores.items():
        test_scores[f"{name}_mlp_score"] = score
    test_scores["impact_label"] = target
    mlp_scores = pd.concat(
        [pd.read_parquet(OUT / "pretest_scores.parquet"), test_scores], ignore_index=True
    ).sort_values("doc_row")
    mlp_scores.to_parquet(OUT / "all_scores.parquet", index=False)
    all_scores = ridge.merge(
        mlp_scores[["doc_row", *[f"{name}_mlp_score" for name in NAMES]]],
        on="doc_row",
        how="inner",
        validate="one_to_one",
    )
    all_scores.to_parquet(OUT / "all_model_scores.parquet", index=False)
    print(metrics.to_string(index=False), flush=True)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


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
