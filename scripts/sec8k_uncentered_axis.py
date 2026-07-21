#!/usr/bin/env python3
"""Uncentered 8-K market-impact axes for BGE, Qwen-8B, and Llama-7B.

All encoders are frozen.  Ridge hyperparameters and the Llama activation layer
are selected on 2024 validation data after fitting directions on 2022--2023.
The 2025 representations are blocked until selection has been frozen.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import news_cutoff_safe_llama2 as protocol
from news_large_embedding_control import last_token_pool


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "sec8k_uncentered"
DOCS = OUT / "documents.parquet"
REP_DIR = OUT / "representations"
LABEL = "parkinson_market_adjusted_expansion"
REPRESENTATIONS = ("bge", "qwen3", "llama2")
MODEL_INFO = {
    "bge": {
        "id": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "parameters": "0.57B",
    },
    "qwen3": {
        "id": "Qwen/Qwen3-Embedding-8B",
        "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        "parameters": "8B",
    },
    "llama2": {
        "id": "meta-llama/Llama-2-7b-hf",
        "revision": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d",
        "parameters": "7B",
    },
}


def part(split: str) -> pd.DataFrame:
    documents = pd.read_parquet(DOCS)
    return protocol.split_documents(documents, split)


def rep_path(name: str, split: str) -> Path:
    return REP_DIR / f"{name}_{split}.npy"


def checkpoint_path(name: str, split: str) -> Path:
    return REP_DIR / f"{name}_{split}.checkpoint.json"


def done_path(name: str, split: str) -> Path:
    return REP_DIR / f"{name}_{split}.done.json"


def load_or_create(name: str, split: str, shape: tuple[int, ...]) -> tuple[np.memmap, int]:
    path = rep_path(name, split)
    checkpoint = checkpoint_path(name, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not checkpoint.exists():
            raise RuntimeError(f"partial representation lacks checkpoint: {path}")
        output = np.load(path, mmap_mode="r+")
        if output.shape != shape or output.dtype != np.float16:
            raise RuntimeError(f"representation mismatch: {output.shape}, {output.dtype}")
        return output, int(json.loads(checkpoint.read_text())["next_order_index"])
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape), 0


def masked_mean_layers(hidden_states, attention_mask):
    import torch

    mask = attention_mask.bool().clone()
    # Llama tokenizer inserts BOS in position zero; exclude it from document pooling.
    mask[:, 0] = False
    denominator = mask.sum(dim=1, keepdim=True).clamp_min(1)
    layers = []
    for state in hidden_states[1:]:
        layers.append((state * mask.unsqueeze(-1)).sum(dim=1) / denominator)
    return torch.stack(layers, dim=1)


def extract(
    name: str,
    split: str,
    model_path: str,
    batch_size: int,
    max_length: int,
) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if split == "test" and not (OUT / "selection.json").exists():
        raise RuntimeError("refusing test extraction before validation selection freeze")
    if done_path(name, split).exists():
        print(f"{name} {split}: already complete", flush=True)
        return
    frame = part(split)
    order = np.argsort(frame["text_chars"].to_numpy(), kind="stable")
    info = MODEL_INFO[name]
    tokenizer_kwargs = {
        "revision": info["revision"],
        "local_files_only": True,
        "use_fast": True,
    }
    if name == "qwen3":
        tokenizer_kwargs["padding_side"] = "left"
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if name == "bge" else torch.bfloat16
    model = AutoModel.from_pretrained(
        model_path,
        revision=info["revision"],
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" if name != "bge" else None,
    ).to("cuda:0").eval()
    model.config.use_cache = False
    hidden = int(model.config.hidden_size)
    shape = (
        (len(frame), int(model.config.num_hidden_layers), hidden)
        if name == "llama2"
        else (len(frame), hidden)
    )
    output, order_start = load_or_create(name, split, shape)
    active_batch = batch_size
    torch.set_float32_matmul_precision("high")
    with torch.inference_mode():
        while order_start < len(order):
            positions = order[order_start : order_start + active_batch]
            encoded = tokenizer(
                frame.iloc[positions]["text"].tolist(),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            try:
                result = model(
                    **encoded,
                    output_hidden_states=name == "llama2",
                    use_cache=False,
                    return_dict=True,
                )
                if name == "bge":
                    pooled = torch.nn.functional.normalize(
                        result.last_hidden_state[:, 0].float(), p=2, dim=1
                    )
                elif name == "qwen3":
                    pooled = torch.nn.functional.normalize(
                        last_token_pool(result.last_hidden_state, encoded["attention_mask"]).float(),
                        p=2,
                        dim=1,
                    )
                else:
                    pooled = masked_mean_layers(result.hidden_states, encoded["attention_mask"]).float()
            except torch.cuda.OutOfMemoryError:
                del encoded
                torch.cuda.empty_cache()
                if active_batch == 1:
                    raise
                active_batch = max(1, active_batch // 2)
                print(f"{name} {split}: CUDA OOM, retry batch={active_batch}", flush=True)
                continue
            output[positions] = pooled.cpu().numpy().astype(np.float16)
            next_index = order_start + len(positions)
            if next_index % 128 < active_batch or next_index == len(order):
                output.flush()
                protocol.write_json(
                    checkpoint_path(name, split),
                    {
                        "next_order_index": next_index,
                        "rows": len(frame),
                        "active_batch_size": active_batch,
                    },
                )
            if next_index % max(100, active_batch) < active_batch or next_index == len(order):
                print(f"{name} {split}: {next_index}/{len(order)}", flush=True)
            del result, pooled, encoded
            order_start = next_index
    output.flush()
    pooling = {
        "bge": "CLS then row L2 normalization",
        "qwen3": "official last-token pooling then row L2 normalization",
        "llama2": "non-BOS document-token mean for every transformer layer",
    }[name]
    protocol.write_json(
        done_path(name, split),
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "representation": name,
            "model_id": info["id"],
            "model_path": model_path,
            "revision": info["revision"],
            "parameters": info["parameters"],
            "split": split,
            "rows": len(frame),
            "shape": shape,
            "forward_dtype": str(dtype),
            "storage_dtype": "float16",
            "pooling": pooling,
            "input": "same cleaned/truncated raw 8-K text for all representations",
            "max_length": max_length,
            "requested_batch_size": batch_size,
            "final_batch_size": active_batch,
        },
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def fit_and_save(x: np.ndarray, y: np.ndarray, alpha: float, path: Path):
    model = protocol.fit_final(x, y, alpha)
    np.savez_compressed(path, coef=model.coef_.astype(np.float32), intercept=np.float64(model.intercept_))
    return model


def tune_dual_cuda(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> list[dict]:
    """Exact Ridge sweep in sample space, efficient here because n << d."""
    import torch

    train = torch.as_tensor(np.ascontiguousarray(x_train), dtype=torch.float32, device="cuda:0")
    validation = torch.as_tensor(
        np.ascontiguousarray(x_val), dtype=torch.float32, device="cuda:0"
    )
    target = torch.as_tensor(y_train, dtype=torch.float32, device="cuda:0")
    x_mean = train.mean(dim=0)
    y_mean = target.mean()
    centered_x = train - x_mean
    centered_y = target - y_mean
    kernel = centered_x @ centered_x.T
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    eigenvalues = eigenvalues.clamp_min_(0.0)
    projected = eigenvectors.T @ centered_y
    rows: list[dict] = []
    for alpha in protocol.ALPHAS:
        dual = eigenvectors @ (projected / (eigenvalues + alpha))
        coefficient = centered_x.T @ dual
        intercept = y_mean - x_mean @ coefficient
        train_score = (train @ coefficient + intercept).cpu().numpy()
        val_score = (validation @ coefficient + intercept).cpu().numpy()
        rows.append(
            {
                "alpha": alpha,
                "train_spearman": protocol.safe_spearman(train_score, y_train),
                "validation_spearman": protocol.safe_spearman(val_score, y_val),
                "validation_pearson": protocol.safe_pearson(val_score, y_val),
            }
        )
    del train, validation, target, centered_x, centered_y, kernel, eigenvalues, eigenvectors
    torch.cuda.empty_cache()
    return rows


def select(workers: int) -> None:
    if (OUT / "selection.json").exists():
        raise RuntimeError("8-K selection already frozen")
    for name in REPRESENTATIONS:
        for split in ("train", "val"):
            if not done_path(name, split).exists():
                raise RuntimeError(f"missing completed {name} {split}")
        if rep_path(name, "test").exists():
            raise RuntimeError(f"{name} test representation exists before selection freeze")
    train = part("train")
    validation = part("val")
    y_train = train[LABEL].to_numpy(float)
    y_val = validation[LABEL].to_numpy(float)
    sweeps: list[dict] = []
    models = {}
    pretest = pd.concat([train, validation], ignore_index=True)[
        ["doc_row", "split_row", "ticker", "file_date", "event_session", "split", LABEL]
    ].copy()
    for name in ("bge", "qwen3"):
        x_train = protocol.normalize_rows(np.load(rep_path(name, "train"), mmap_mode="r"))
        x_val = protocol.normalize_rows(np.load(rep_path(name, "val"), mmap_mode="r"))
        rows = tune_dual_cuda(x_train, y_train, x_val, y_val)
        sweeps.extend({"representation": name, "layer": np.nan, **row} for row in rows)
        best = min(rows, key=lambda row: (-row["validation_spearman"], row["alpha"]))
        model = fit_and_save(x_train, y_train, float(best["alpha"]), OUT / f"{name}_ridge_state.npz")
        pretest[f"{name}_ridge_score"] = np.concatenate(
            [model.predict(x_train), model.predict(x_val)]
        ).astype(np.float32)
        models[name] = {
            "selected_alpha": float(best["alpha"]),
            "validation_spearman": float(best["validation_spearman"]),
            "validation_pearson": float(best["validation_pearson"]),
        }
        del x_train, x_val
        print(f"selection {name}: {models[name]}", flush=True)

    llama_train = protocol.load_llama_layers_to_ram(rep_path("llama2", "train"))
    llama_val = protocol.load_llama_layers_to_ram(rep_path("llama2", "val"))
    for layer in range(1, llama_train.shape[0] + 1):
        x_train = protocol.normalize_rows(llama_train[layer - 1])
        x_val = protocol.normalize_rows(llama_val[layer - 1])
        rows = tune_dual_cuda(x_train, y_train, x_val, y_val)
        sweeps.extend({"representation": "llama2", "layer": layer, **row} for row in rows)
        print(f"selection llama2: layer {layer}/{llama_train.shape[0]}", flush=True)
    llama_rows = [row for row in sweeps if row["representation"] == "llama2"]
    best = min(
        llama_rows,
        key=lambda row: (-row["validation_spearman"], row["layer"], row["alpha"]),
    )
    layer = int(best["layer"])
    x_train = protocol.normalize_rows(llama_train[layer - 1])
    x_val = protocol.normalize_rows(llama_val[layer - 1])
    model = fit_and_save(
        x_train, y_train, float(best["alpha"]), OUT / "llama2_ridge_state.npz"
    )
    pretest["llama2_ridge_score"] = np.concatenate(
        [model.predict(x_train), model.predict(x_val)]
    ).astype(np.float32)
    models["llama2"] = {
        "selected_layer": layer,
        "selected_alpha": float(best["alpha"]),
        "validation_spearman": float(best["validation_spearman"]),
        "validation_pearson": float(best["validation_pearson"]),
    }
    pretest["impact_label"] = pretest[LABEL]
    pretest.to_parquet(OUT / "ridge_pretest_scores.parquet", index=False)
    pd.DataFrame(sweeps).to_csv(OUT / "ridge_validation_sweep.csv", index=False)
    selection = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "axis_fit_split": "2022-2023 train only",
        "selection_split": "2024 validation only",
        "test_split": "2025 exploratory",
        "test_representations_present_at_freeze": False,
        "alpha_grid": list(protocol.ALPHAS),
        "centering": "none; no ticker means subtracted from representations or labels",
        "representation_normalization": "row L2 only",
        "label": LABEL,
        "models": models,
        "model_info": MODEL_INFO,
        "qwen_cutoff_safe": False,
        "qwen_warning": "capacity control only; Qwen3-Embedding-8B is not cutoff-safe for 2025",
    }
    protocol.write_json(OUT / "selection.json", selection)
    print(json.dumps(selection, indent=2), flush=True)


def apply_state(x: np.ndarray, path: Path) -> np.ndarray:
    state = np.load(path)
    return (x @ state["coef"] + float(state["intercept"])).astype(np.float32)


def evaluate(repetitions: int, workers: int) -> None:
    if not (OUT / "selection.json").exists():
        raise RuntimeError("selection is not frozen")
    for name in REPRESENTATIONS:
        if not done_path(name, "test").exists():
            raise RuntimeError(f"missing completed {name} test")
    if (OUT / "ridge_test_metrics.csv").exists():
        raise RuntimeError("8-K Ridge test already evaluated")
    selection = json.loads((OUT / "selection.json").read_text())
    test = part("test")
    target = test[LABEL].to_numpy(float)
    scores: dict[str, np.ndarray] = {}
    for name in ("bge", "qwen3"):
        values = protocol.normalize_rows(np.load(rep_path(name, "test"), mmap_mode="r"))
        scores[name] = apply_state(values, OUT / f"{name}_ridge_state.npz")
    layer = int(selection["models"]["llama2"]["selected_layer"])
    values = protocol.normalize_rows(
        protocol.load_selected_llama_layer_to_ram(rep_path("llama2", "test"), layer - 1)
    )
    scores["llama2"] = apply_state(values, OUT / "llama2_ridge_state.npz")
    metrics = pd.DataFrame(
        [
            {
                "method": name,
                "parameters": MODEL_INFO[name]["parameters"],
                "selected_layer": layer if name == "llama2" else np.nan,
                "selected_alpha": selection["models"][name]["selected_alpha"],
                "n_test": len(test),
                "test_spearman": protocol.safe_spearman(score, target),
                "test_pearson": protocol.safe_pearson(score, target),
            }
            for name, score in scores.items()
        ]
    )
    metrics.to_csv(OUT / "ridge_test_metrics.csv", index=False)
    frame = test[["event_session", "ticker"]].rename(columns={"event_session": "date"})
    comparisons = []
    for baseline in ("bge", "qwen3"):
        for block in ("date", "ticker"):
            comparisons.append(
                {
                    "comparison": f"llama2_minus_{baseline}",
                    **protocol.parallel_bootstrap(
                        frame,
                        scores["llama2"],
                        scores[baseline],
                        target,
                        block,
                        repetitions,
                        workers,
                    ),
                }
            )
    pd.DataFrame(comparisons).to_csv(OUT / "ridge_correlation_bootstrap.csv", index=False)
    test_scores = test[
        ["doc_row", "split_row", "ticker", "file_date", "event_session", "split", LABEL]
    ].copy()
    for name, score in scores.items():
        test_scores[f"{name}_ridge_score"] = score
    test_scores["impact_label"] = target
    all_scores = pd.concat(
        [pd.read_parquet(OUT / "ridge_pretest_scores.parquet"), test_scores], ignore_index=True
    ).sort_values("doc_row")
    all_scores.to_parquet(OUT / "ridge_all_scores.parquet", index=False)
    protocol.write_json(
        OUT / "ridge_evaluation_manifest.json",
        {
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "test_rows": len(test),
            "bootstrap_repetitions": repetitions,
            "bootstrap_blocks": ["event_session", "ticker"],
            "centering": "none",
        },
    )
    print(metrics.to_string(index=False), flush=True)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    extract_parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    extract_parser.add_argument("--model-path", required=True)
    extract_parser.add_argument("--batch-size", type=int, default=8)
    extract_parser.add_argument("--max-length", type=int, default=2048)
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    evaluate_parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.representation, args.split, args.model_path, args.batch_size, args.max_length)
    elif args.command == "select":
        select(args.workers)
    else:
        evaluate(args.bootstrap_repetitions, args.workers)


if __name__ == "__main__":
    main()
