#!/usr/bin/env python3
"""Refit impact axes on post-event volatility levels without firm normalization.

This is an explicitly diagnostic task: unlike the main expansion label, these
targets do not subtract the firm's pre-event volatility.  A high correlation
can therefore reflect persistent firm/ticker volatility rather than document
impact.  All Ridge fits use train only; alpha and the Llama layer are selected
on validation before test scores are evaluated.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import news_cutoff_safe_llama2 as protocol


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "no_firm_vol_normalization_refit"
EPS = 1e-12
TARGETS = ("firm_post_level", "market_adjusted_post_level")


def data(domain: str) -> pd.DataFrame:
    if domain == "news":
        documents = pd.read_parquet(
            ROOT / "outputs" / "news_cutoff_safe_llama2" / "documents.parquet"
        )
        components = protocol.normalized_parkinson_labels()[
            [
                "ticker",
                "date",
                "firm_pre20",
                "firm_post5",
                "market_pre20",
                "market_post5",
            ]
        ]
        documents = documents.merge(
            components, on=["ticker", "date"], how="left", validate="one_to_one"
        )
    else:
        documents = pd.read_parquet(
            ROOT / "outputs" / "sec8k_uncentered" / "documents.parquet"
        )
    required = ["firm_post5", "market_post5"]
    if documents[required].isna().any().any():
        raise RuntimeError(f"missing volatility components in {domain}")
    documents["firm_post_level"] = np.log(documents["firm_post5"].clip(lower=EPS))
    documents["market_adjusted_post_level"] = 0.5 * (
        np.log(documents["firm_post5"].clip(lower=EPS))
        - np.log(documents["market_post5"].clip(lower=EPS))
    )
    return documents.sort_values("doc_row").reset_index(drop=True)


def representation_path(domain: str, representation: str, split: str) -> Path:
    if domain == "news":
        if representation == "bge":
            return (
                ROOT
                / "outputs"
                / "news_cutoff_safe_llama2"
                / "representations"
                / f"bge_{split}.npy"
            )
        if representation == "qwen3":
            return (
                ROOT
                / "outputs"
                / "news_qwen3_embedding_8b_control"
                / "representations"
                / f"qwen3_embedding_8b_{split}.npy"
            )
        return (
            ROOT
            / "outputs"
            / "news_cutoff_safe_llama2"
            / "representations"
            / f"llama2_{split}.npy"
        )
    return (
        ROOT
        / "outputs"
        / "sec8k_uncentered"
        / "representations"
        / f"{representation}_{split}.npy"
    )


def split_frame(documents: pd.DataFrame, split: str) -> pd.DataFrame:
    return protocol.split_documents(documents, split)


def ridge_multi_target(
    x_train: np.ndarray,
    x_val: np.ndarray,
    targets_train: dict[str, np.ndarray],
    targets_val: dict[str, np.ndarray],
    device: str,
) -> tuple[list[dict], dict[str, tuple[np.ndarray, float, float]]]:
    """Tune all targets with one EVD of the shared representation matrix."""
    import torch

    train = torch.as_tensor(
        np.ascontiguousarray(x_train), dtype=torch.float32, device=device
    )
    validation = torch.as_tensor(
        np.ascontiguousarray(x_val), dtype=torch.float32, device=device
    )
    x_mean = train.mean(dim=0)
    centered_x = train - x_mean
    use_dual = train.shape[0] < train.shape[1]
    if use_dual:
        kernel = centered_x @ centered_x.T
        eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    else:
        gram = centered_x.T @ centered_x
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues.clamp_min_(0.0)
    rows = []
    states = {}
    for target_name in TARGETS:
        y_train = targets_train[target_name]
        y_val = targets_val[target_name]
        target = torch.as_tensor(y_train, dtype=torch.float32, device=device)
        y_mean = target.mean()
        centered_y = target - y_mean
        if use_dual:
            projected = eigenvectors.T @ centered_y
        else:
            rhs = centered_x.T @ centered_y
            projected = eigenvectors.T @ rhs
        candidates = []
        candidate_states = {}
        for alpha in protocol.ALPHAS:
            if use_dual:
                dual = eigenvectors @ (projected / (eigenvalues + alpha))
                coefficient = centered_x.T @ dual
            else:
                coefficient = eigenvectors @ (projected / (eigenvalues + alpha))
            intercept = y_mean - x_mean @ coefficient
            train_score = (train @ coefficient + intercept).cpu().numpy()
            val_score = (validation @ coefficient + intercept).cpu().numpy()
            row = {
                "target": target_name,
                "alpha": float(alpha),
                "train_spearman": protocol.safe_spearman(train_score, y_train),
                "validation_spearman": protocol.safe_spearman(val_score, y_val),
                "validation_pearson": protocol.safe_pearson(val_score, y_val),
            }
            candidates.append(row)
            candidate_states[float(alpha)] = (
                coefficient.cpu().numpy().astype(np.float32),
                float(intercept.cpu()),
            )
        best = min(
            candidates,
            key=lambda row: (-row["validation_spearman"], row["alpha"]),
        )
        coefficient, intercept = candidate_states[float(best["alpha"])]
        states[target_name] = (coefficient, intercept, float(best["alpha"]))
        rows.extend(candidates)
        del target, centered_y, projected
        if not use_dual:
            del rhs
    del train, validation, x_mean, centered_x, eigenvalues, eigenvectors
    if use_dual:
        del kernel
    else:
        del gram
    torch.cuda.empty_cache()
    return rows, states


def save_state(
    domain: str,
    target: str,
    representation: str,
    coefficient: np.ndarray,
    intercept: float,
    alpha: float,
    layer: int | None,
) -> None:
    np.savez_compressed(
        OUT / f"{domain}_{target}_{representation}_state.npz",
        coef=coefficient,
        intercept=np.float64(intercept),
        alpha=np.float64(alpha),
        layer=np.int32(-1 if layer is None else layer),
    )


def run(domain: str, device: str) -> None:
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    documents = data(domain)
    frames = {split: split_frame(documents, split) for split in ("train", "val", "test")}
    targets = {
        split: {name: frames[split][name].to_numpy(float) for name in TARGETS}
        for split in frames
    }
    sweeps = []
    selections = {}
    scores = {
        split: frames[split][
            [
                column
                for column in (
                    "doc_row",
                    "split_row",
                    "ticker",
                    "date",
                    "file_date",
                    "event_session",
                    "split",
                )
                if column in frames[split].columns
            ]
        ].copy()
        for split in frames
    }
    for split in frames:
        for target_name in TARGETS:
            scores[split][target_name] = targets[split][target_name]

    for representation in ("bge", "qwen3"):
        x_train = protocol.normalize_rows(
            np.load(representation_path(domain, representation, "train"), mmap_mode="r")
        )
        x_val = protocol.normalize_rows(
            np.load(representation_path(domain, representation, "val"), mmap_mode="r")
        )
        rows, states = ridge_multi_target(
            x_train, x_val, targets["train"], targets["val"], device
        )
        sweeps.extend(
            {"representation": representation, "layer": np.nan, **row}
            for row in rows
        )
        x_test = protocol.normalize_rows(
            np.load(representation_path(domain, representation, "test"), mmap_mode="r")
        )
        for target_name, (coefficient, intercept, alpha) in states.items():
            save_state(
                domain,
                target_name,
                representation,
                coefficient,
                intercept,
                alpha,
                None,
            )
            for split, values in (("train", x_train), ("val", x_val), ("test", x_test)):
                scores[split][f"{target_name}_{representation}_score"] = (
                    values @ coefficient + intercept
                ).astype(np.float32)
            best = min(
                [row for row in rows if row["target"] == target_name],
                key=lambda row: (-row["validation_spearman"], row["alpha"]),
            )
            selections[f"{target_name}/{representation}"] = {
                "selected_layer": None,
                "selected_alpha": alpha,
                "validation_spearman": best["validation_spearman"],
                "validation_pearson": best["validation_pearson"],
            }
        del x_train, x_val, x_test
        print(f"refit {domain}: {representation} complete", flush=True)

    llama_train = protocol.load_llama_layers_to_ram(
        representation_path(domain, "llama2", "train")
    )
    llama_val = protocol.load_llama_layers_to_ram(
        representation_path(domain, "llama2", "val")
    )
    llama_layer_states = {name: {} for name in TARGETS}
    for layer_index in range(llama_train.shape[0]):
        x_train = protocol.normalize_rows(llama_train[layer_index])
        x_val = protocol.normalize_rows(llama_val[layer_index])
        rows, states = ridge_multi_target(
            x_train, x_val, targets["train"], targets["val"], device
        )
        layer = layer_index + 1
        sweeps.extend(
            {"representation": "llama2", "layer": layer, **row} for row in rows
        )
        for target_name, state in states.items():
            best = min(
                [row for row in rows if row["target"] == target_name],
                key=lambda row: (-row["validation_spearman"], row["alpha"]),
            )
            llama_layer_states[target_name][layer] = (state, best)
        print(f"refit {domain}: llama layer {layer}/{llama_train.shape[0]}", flush=True)
        del x_train, x_val

    llama_test_mapped = np.load(
        representation_path(domain, "llama2", "test"), mmap_mode="r"
    )
    for target_name in TARGETS:
        layer, (state, best) = min(
            llama_layer_states[target_name].items(),
            key=lambda item: (
                -item[1][1]["validation_spearman"],
                item[0],
                item[1][1]["alpha"],
            ),
        )
        coefficient, intercept, alpha = state
        save_state(
            domain, target_name, "llama2", coefficient, intercept, alpha, layer
        )
        train_values = protocol.normalize_rows(llama_train[layer - 1])
        val_values = protocol.normalize_rows(llama_val[layer - 1])
        # Existing Llama arrays are document-major on disk.
        test_values = protocol.normalize_rows(
            np.asarray(llama_test_mapped[:, layer - 1, :], dtype=np.float32)
        )
        for split, values in (
            ("train", train_values),
            ("val", val_values),
            ("test", test_values),
        ):
            scores[split][f"{target_name}_llama2_score"] = (
                values @ coefficient + intercept
            ).astype(np.float32)
        selections[f"{target_name}/llama2"] = {
            "selected_layer": layer,
            "selected_alpha": alpha,
            "validation_spearman": best["validation_spearman"],
            "validation_pearson": best["validation_pearson"],
        }

    metrics = []
    for target_name in TARGETS:
        for representation in ("bge", "qwen3", "llama2"):
            column = f"{target_name}_{representation}_score"
            y = scores["test"][target_name].to_numpy(float)
            prediction = scores["test"][column].to_numpy(float)
            selection = selections[f"{target_name}/{representation}"]
            metrics.append(
                {
                    "domain": domain,
                    "target": target_name,
                    "representation": representation,
                    **selection,
                    "n_test": len(y),
                    "test_spearman": protocol.safe_spearman(prediction, y),
                    "test_pearson": protocol.safe_pearson(prediction, y),
                }
            )
    pd.DataFrame(sweeps).to_csv(OUT / f"{domain}_validation_sweep.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / f"{domain}_test_metrics.csv", index=False)
    pd.concat(scores.values(), ignore_index=True).sort_values("doc_row").to_parquet(
        OUT / f"{domain}_all_scores.parquet", index=False
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "targets": {
            "firm_post_level": "log firm post5 Parkinson variance; no firm pre20 or market subtraction",
            "market_adjusted_post_level": "0.5 * (log firm post5 - log market post5); no firm pre20 subtraction",
        },
        "centering": "none; no ticker mean subtraction",
        "axis_fit": "train-only Ridge",
        "selection": "validation Spearman over alpha and all 32 Llama layers",
        "alpha_grid": list(protocol.ALPHAS),
        "test_access": "after validation choices were computed within this run",
        "interpretation_warning": "future volatility level can reward persistent firm/ticker volatility rather than document impact",
        "selections": selections,
    }
    protocol.write_json(OUT / f"{domain}_manifest.json", manifest)
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    del llama_train, llama_val, llama_test_mapped
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("news", "sec8k"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.domain, args.device)


if __name__ == "__main__":
    main()
