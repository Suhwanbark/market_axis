#!/usr/bin/env python3
"""Layer-wise embedding controls for BGE-M3 and Qwen3-Embedding-8B.

The experiment mirrors the uncentered Llama activation protocol.  For every
transformer block, it pools the same token used by the encoder's official
final embedding (BGE CLS; Qwen last token), L2-normalizes the vector, and fits
a train-only Ridge impact axis.  Layer and alpha choices use validation impact
Spearman only.  Test representations are blocked until that selection is
frozen.

Large layer arrays are resumable scratch artifacts under /tmp.  Compact axes,
scores, selections, and metrics are written to the repository output folder.
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import news_cutoff_safe_llama2 as protocol


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "embedding_intermediate_layer_control"
TMP = Path("/tmp/market_axis_embedding_layers")
NEWS_DOCS = ROOT / "outputs" / "news_cutoff_safe_llama2" / "documents.parquet"
SEC_DOCS = ROOT / "outputs" / "sec8k_uncentered" / "documents.parquet"
LABEL = protocol.LABEL
MODEL_INFO = {
    "bge": {
        "id": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "parameters": "0.57B",
        "pooling": "CLS",
        "layers": 24,
        "hidden": 1024,
    },
    "qwen3": {
        "id": "Qwen/Qwen3-Embedding-8B",
        "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        "parameters": "8B",
        "pooling": "last token",
        "layers": 36,
        "hidden": 4096,
    },
}


def frame(domain: str, split: str) -> pd.DataFrame:
    path = NEWS_DOCS if domain == "news" else SEC_DOCS
    documents = pd.read_parquet(path)
    return protocol.split_documents(documents, split)


def text_column(domain: str) -> str:
    return "news_text" if domain == "news" else "text"


def layer_path(domain: str, model: str, split: str) -> Path:
    return TMP / f"{domain}_{model}_{split}_layers.npy"


def checkpoint_path(domain: str, model: str, split: str) -> Path:
    return TMP / f"{domain}_{model}_{split}.checkpoint.json"


def done_path(domain: str, model: str, split: str) -> Path:
    return TMP / f"{domain}_{model}_{split}.done.json"


def selection_path(domain: str, model: str) -> Path:
    return OUT / f"{domain}_{model}_selection.json"


def state_path(domain: str, model: str) -> Path:
    return OUT / f"{domain}_{model}_layer_states.npz"


def get_blocks(model, name: str):
    if name == "qwen3":
        if hasattr(model, "layers"):
            return list(model.layers)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
    elif name == "bge":
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return list(model.encoder.layer)
        if hasattr(model, "roberta"):
            return list(model.roberta.encoder.layer)
    raise RuntimeError(f"cannot locate transformer blocks for {type(model).__name__}")


def open_layers(
    domain: str, model: str, split: str, shape: tuple[int, int, int]
) -> tuple[np.memmap, int]:
    TMP.mkdir(parents=True, exist_ok=True)
    path = layer_path(domain, model, split)
    checkpoint = checkpoint_path(domain, model, split)
    if path.exists() and checkpoint.exists():
        values = np.load(path, mmap_mode="r+")
        if values.shape != shape or values.dtype != np.float16:
            raise RuntimeError(f"scratch layer shape mismatch: {path} {values.shape}")
        return values, int(json.loads(checkpoint.read_text())["next_order_index"])
    # A failure before the first checkpoint leaves no valid scratch progress.
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    return values, 0


def extract(
    *,
    domain: str,
    name: str,
    split: str,
    model_path: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if split == "test" and not selection_path(domain, name).exists():
        raise RuntimeError("test extraction is blocked until layer selection is frozen")
    if done_path(domain, name, split).exists():
        print(f"layers {domain}/{name}/{split}: already complete", flush=True)
        return
    info = MODEL_INFO[name]
    data = frame(domain, split)
    order = np.argsort(data["text_chars"].to_numpy(), kind="stable")
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
    dtype = torch.bfloat16 if name == "qwen3" else torch.float16
    model_kwargs = {
        "revision": info["revision"],
        "local_files_only": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if name == "qwen3":
        model_kwargs["attn_implementation"] = "sdpa"
    model = AutoModel.from_pretrained(model_path, **model_kwargs).to(device).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    blocks = get_blocks(model, name)
    if len(blocks) != int(info["layers"]):
        raise RuntimeError(f"expected {info['layers']} blocks, found {len(blocks)}")
    hidden = int(model.config.hidden_size)
    output, order_start = open_layers(
        domain, name, split, (len(blocks), len(data), hidden)
    )
    captured = []

    def capture_last_or_cls(_module, _inputs, block_output):
        state = block_output[0] if isinstance(block_output, tuple) else block_output
        token_index = -1 if name == "qwen3" else 0
        # A float copy keeps only one token per layer instead of retaining all
        # sequence-wide hidden states until the end of the forward pass.
        captured.append(state[:, token_index, :].detach().float())

    handles = [block.register_forward_hook(capture_last_or_cls) for block in blocks]
    active_batch = batch_size
    torch.set_float32_matmul_precision("high")
    try:
        with torch.inference_mode():
            while order_start < len(order):
                positions = order[order_start : order_start + active_batch]
                encoded = tokenizer(
                    data.iloc[positions][text_column(domain)].tolist(),
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                captured.clear()
                try:
                    result = model(**encoded, use_cache=False, return_dict=True)
                    if len(captured) != len(blocks):
                        raise RuntimeError(
                            f"captured {len(captured)} layers, expected {len(blocks)}"
                        )
                    # The final model norm can live outside the last block.  Use
                    # last_hidden_state for the last layer so it exactly matches
                    # the official final embedding baseline.
                    token_index = -1 if name == "qwen3" else 0
                    captured[-1] = result.last_hidden_state[:, token_index, :].float()
                    pooled = torch.nn.functional.normalize(
                        torch.stack(captured, dim=1), p=2, dim=2
                    )
                except torch.cuda.OutOfMemoryError:
                    captured.clear()
                    del encoded
                    torch.cuda.empty_cache()
                    if active_batch == 1:
                        raise
                    active_batch = max(1, active_batch // 2)
                    print(
                        f"layers {domain}/{name}/{split}: OOM, batch={active_batch}",
                        flush=True,
                    )
                    continue
                values = pooled.cpu().numpy().astype(np.float16).transpose(1, 0, 2)
                output[:, positions, :] = values
                next_index = order_start + len(positions)
                checkpoint_every = 512 if domain == "news" else 128
                if next_index % checkpoint_every < active_batch or next_index == len(order):
                    output.flush()
                    protocol.write_json(
                        checkpoint_path(domain, name, split),
                        {
                            "next_order_index": next_index,
                            "rows": len(data),
                            "active_batch_size": active_batch,
                        },
                    )
                if next_index % max(200, active_batch) < active_batch or next_index == len(order):
                    print(
                        f"layers {domain}/{name}/{split}: {next_index}/{len(order)}",
                        flush=True,
                    )
                del result, pooled, values, encoded
                captured.clear()
                order_start = next_index
    finally:
        for handle in handles:
            handle.remove()
    output.flush()
    protocol.write_json(
        done_path(domain, name, split),
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "domain": domain,
            "split": split,
            "model": name,
            "model_id": info["id"],
            "revision": info["revision"],
            "parameters": info["parameters"],
            "rows": len(data),
            "layers": len(blocks),
            "hidden_size": hidden,
            "pooling_per_layer": info["pooling"],
            "normalization": "row L2 at every layer",
            "centering": "none",
            "max_length": max_length,
            "forward_dtype": str(dtype),
            "storage_dtype": "float16",
            "requested_batch_size": batch_size,
            "final_batch_size": active_batch,
            "scratch_path": str(layer_path(domain, name, split)),
        },
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def tune_and_fit_layer_cuda(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: str,
) -> tuple[list[dict], np.ndarray, float, float]:
    import torch

    train = torch.as_tensor(np.ascontiguousarray(x_train), dtype=torch.float32, device=device)
    validation = torch.as_tensor(
        np.ascontiguousarray(x_val), dtype=torch.float32, device=device
    )
    target = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    x_mean = train.mean(dim=0)
    y_mean = target.mean()
    centered_x = train - x_mean
    centered_y = target - y_mean
    gram = centered_x.T @ centered_x
    rhs = centered_x.T @ centered_y
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues.clamp_min_(0.0)
    projected = eigenvectors.T @ rhs
    rows = []
    coefficients = {}
    intercepts = {}
    for alpha in protocol.ALPHAS:
        coefficient = eigenvectors @ (projected / (eigenvalues + alpha))
        intercept = y_mean - x_mean @ coefficient
        train_score = (train @ coefficient + intercept).cpu().numpy()
        val_score = (validation @ coefficient + intercept).cpu().numpy()
        rows.append(
            {
                "alpha": float(alpha),
                "train_spearman": protocol.safe_spearman(train_score, y_train),
                "validation_spearman": protocol.safe_spearman(val_score, y_val),
                "validation_pearson": protocol.safe_pearson(val_score, y_val),
            }
        )
        coefficients[float(alpha)] = coefficient.cpu().numpy().astype(np.float32)
        intercepts[float(alpha)] = float(intercept.cpu())
    best = min(rows, key=lambda row: (-row["validation_spearman"], row["alpha"]))
    best_alpha = float(best["alpha"])
    result = coefficients[best_alpha], intercepts[best_alpha], best_alpha
    del (
        train,
        validation,
        target,
        centered_x,
        centered_y,
        gram,
        rhs,
        eigenvalues,
        eigenvectors,
        projected,
    )
    torch.cuda.empty_cache()
    return rows, *result


def id_columns(domain: str) -> list[str]:
    if domain == "news":
        return ["doc_row", "split_row", "ticker", "date", "split", LABEL]
    return [
        "doc_row",
        "split_row",
        "ticker",
        "file_date",
        "event_session",
        "split",
        LABEL,
    ]


def select(domain: str, name: str, device: str) -> None:
    if selection_path(domain, name).exists():
        print(f"selection {domain}/{name}: already frozen", flush=True)
        return
    for split in ("train", "val"):
        if not done_path(domain, name, split).exists():
            raise RuntimeError(f"missing train/validation layers: {domain}/{name}/{split}")
    if layer_path(domain, name, "test").exists():
        raise RuntimeError("test layer file exists before validation selection freeze")
    OUT.mkdir(parents=True, exist_ok=True)
    train_frame = frame(domain, "train")
    val_frame = frame(domain, "val")
    y_train = train_frame[LABEL].to_numpy(float)
    y_val = val_frame[LABEL].to_numpy(float)
    train_layers = np.load(layer_path(domain, name, "train"), mmap_mode="r")
    val_layers = np.load(layer_path(domain, name, "val"), mmap_mode="r")
    if train_layers.shape[0] != val_layers.shape[0]:
        raise RuntimeError("train/validation layer count mismatch")
    coefficients = np.empty(
        (train_layers.shape[0], train_layers.shape[2]), dtype=np.float32
    )
    intercepts = np.empty(train_layers.shape[0], dtype=np.float64)
    selected_alphas = np.empty(train_layers.shape[0], dtype=np.float64)
    sweep = []
    for layer_index in range(train_layers.shape[0]):
        x_train = protocol.normalize_rows(train_layers[layer_index])
        x_val = protocol.normalize_rows(val_layers[layer_index])
        rows, coefficient, intercept, alpha = tune_and_fit_layer_cuda(
            x_train, y_train, x_val, y_val, device
        )
        coefficients[layer_index] = coefficient
        intercepts[layer_index] = intercept
        selected_alphas[layer_index] = alpha
        sweep.extend(
            {"layer": layer_index + 1, **row}
            for row in rows
        )
        print(
            f"selection {domain}/{name}: layer {layer_index + 1}/{train_layers.shape[0]}",
            flush=True,
        )
        del x_train, x_val
    sweep_frame = pd.DataFrame(sweep)
    per_layer = (
        sweep_frame.sort_values(
            ["layer", "validation_spearman", "alpha"],
            ascending=[True, False, True],
        )
        .groupby("layer", as_index=False)
        .first()
    )
    best_any = per_layer.sort_values(
        ["validation_spearman", "layer", "alpha"], ascending=[False, True, True]
    ).iloc[0]
    intermediate = per_layer.loc[per_layer["layer"].lt(train_layers.shape[0])]
    best_intermediate = intermediate.sort_values(
        ["validation_spearman", "layer", "alpha"], ascending=[False, True, True]
    ).iloc[0]
    np.savez_compressed(
        state_path(domain, name),
        coefficients=coefficients,
        intercepts=intercepts,
        selected_alphas=selected_alphas,
    )
    sweep_frame.to_csv(OUT / f"{domain}_{name}_validation_sweep.csv", index=False)
    per_layer.to_csv(OUT / f"{domain}_{name}_validation_by_layer.csv", index=False)
    selection = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "model": name,
        "model_info": MODEL_INFO[name],
        "label": LABEL,
        "centering": "none; no ticker means subtracted",
        "representation": f"{MODEL_INFO[name]['pooling']} from every transformer block, row L2",
        "axis_fit_split": "2019-2021 train" if domain == "news" else "2022-2023 train",
        "selection_split": "2022 validation" if domain == "news" else "2024 validation",
        "test_split": "2023 test" if domain == "news" else "2025 test",
        "alpha_grid": list(protocol.ALPHAS),
        "test_representation_present_at_freeze": False,
        "selected_intermediate_layer": int(best_intermediate["layer"]),
        "selected_intermediate_alpha": float(best_intermediate["alpha"]),
        "selected_intermediate_validation_spearman": float(
            best_intermediate["validation_spearman"]
        ),
        "selected_any_layer": int(best_any["layer"]),
        "selected_any_alpha": float(best_any["alpha"]),
        "selected_any_validation_spearman": float(best_any["validation_spearman"]),
        "final_layer": int(train_layers.shape[0]),
        "final_layer_alpha": float(per_layer.iloc[-1]["alpha"]),
        "final_layer_validation_spearman": float(
            per_layer.iloc[-1]["validation_spearman"]
        ),
    }
    protocol.write_json(selection_path(domain, name), selection)
    # ``best`` is the primary control requested for forecasting: no layer is
    # chosen by hand, and the final layer competes with every intermediate
    # layer.  ``mid`` and ``final`` are retained as transparent diagnostics.
    selected_layers = {
        "best": int(best_any["layer"]),
        "mid": int(best_intermediate["layer"]),
        "final": int(train_layers.shape[0]),
    }
    pretest_parts = []
    for split_name, split_frame, layers in (
        ("train", train_frame, train_layers),
        ("val", val_frame, val_layers),
    ):
        part = split_frame[id_columns(domain)].copy()
        for variant, layer in selected_layers.items():
            layer_index = layer - 1
            values = protocol.normalize_rows(layers[layer_index])
            part[f"{name}_{variant}_score"] = (
                values @ coefficients[layer_index] + intercepts[layer_index]
            ).astype(np.float32)
        pretest_parts.append(part)
    pd.concat(pretest_parts, ignore_index=True).to_parquet(
        OUT / f"{domain}_{name}_pretest_scores.parquet", index=False
    )
    print(json.dumps(selection, indent=2), flush=True)


def evaluate(domain: str, name: str) -> None:
    if not selection_path(domain, name).exists():
        raise RuntimeError("selection is not frozen")
    if not done_path(domain, name, "test").exists():
        raise RuntimeError("test layer extraction is incomplete")
    selection = json.loads(selection_path(domain, name).read_text())
    test_frame = frame(domain, "test")
    target = test_frame[LABEL].to_numpy(float)
    layers = np.load(layer_path(domain, name, "test"), mmap_mode="r")
    state = np.load(state_path(domain, name))
    coefficients = state["coefficients"]
    intercepts = state["intercepts"]
    rows = []
    selected_scores = {}
    mid_layer = int(selection["selected_intermediate_layer"])
    final_layer = int(selection["final_layer"])
    any_layer = int(selection["selected_any_layer"])
    for layer_index in range(layers.shape[0]):
        values = protocol.normalize_rows(layers[layer_index])
        score = (values @ coefficients[layer_index] + intercepts[layer_index]).astype(
            np.float32
        )
        layer = layer_index + 1
        rows.append(
            {
                "domain": domain,
                "model": name,
                "layer": layer,
                "selected_alpha": float(state["selected_alphas"][layer_index]),
                "is_selected_intermediate": layer == mid_layer,
                "is_selected_any": layer == any_layer,
                "is_final": layer == final_layer,
                "n_test": len(test_frame),
                "test_spearman": protocol.safe_spearman(score, target),
                "test_pearson": protocol.safe_pearson(score, target),
            }
        )
        if layer == mid_layer:
            selected_scores["mid"] = score
        if layer == final_layer:
            selected_scores["final"] = score
        if layer == any_layer:
            selected_scores["best"] = score
        print(f"evaluation {domain}/{name}: layer {layer}/{layers.shape[0]}", flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / f"{domain}_{name}_layerwise_test_metrics.csv", index=False)
    test_scores = test_frame[id_columns(domain)].copy()
    for variant, score in selected_scores.items():
        test_scores[f"{name}_{variant}_score"] = score
    all_scores = pd.concat(
        [
            pd.read_parquet(OUT / f"{domain}_{name}_pretest_scores.parquet"),
            test_scores,
        ],
        ignore_index=True,
    ).sort_values("doc_row")
    all_scores.to_parquet(OUT / f"{domain}_{name}_all_scores.parquet", index=False)
    print(
        metrics.loc[
            metrics["is_selected_any"]
            | metrics["is_selected_intermediate"]
            | metrics["is_final"]
        ].to_string(index=False),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("--domain", choices=("news", "sec8k"), required=True)
    extract_parser.add_argument("--model", choices=tuple(MODEL_INFO), required=True)
    extract_parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    extract_parser.add_argument("--model-path", required=True)
    extract_parser.add_argument("--batch-size", type=int, required=True)
    extract_parser.add_argument("--max-length", type=int, required=True)
    extract_parser.add_argument("--device", default="cuda:0")
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--domain", choices=("news", "sec8k"), required=True)
    select_parser.add_argument("--model", choices=tuple(MODEL_INFO), required=True)
    select_parser.add_argument("--device", default="cuda:0")
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--domain", choices=("news", "sec8k"), required=True)
    evaluate_parser.add_argument("--model", choices=tuple(MODEL_INFO), required=True)
    args = parser.parse_args()
    if args.command == "extract":
        extract(
            domain=args.domain,
            name=args.model,
            split=args.split,
            model_path=args.model_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
    elif args.command == "select":
        select(args.domain, args.model, args.device)
    else:
        evaluate(args.domain, args.model)


if __name__ == "__main__":
    main()
