#!/usr/bin/env python3
"""Capacity-matched Qwen3-Embedding-8B control for the News market axis.

This is deliberately separate from the cutoff-safe Llama 2 result.  Qwen3-
Embedding-8B is a post-2023 model, so this experiment controls model capacity
but cannot establish temporal cutoff safety.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import news_cutoff_safe_llama2 as protocol


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "outputs" / "news_cutoff_safe_llama2"
UNCENTERED = ROOT / "outputs" / "news_uncentered_linear_control"
OUT = ROOT / "outputs" / "news_qwen3_embedding_8b_control"
MODEL_ID = "Qwen/Qwen3-Embedding-8B"
MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
REPRESENTATION = "qwen3_embedding_8b"


def representation_path(split: str) -> Path:
    return OUT / "representations" / f"{REPRESENTATION}_{split}.npy"


def checkpoint_path(split: str) -> Path:
    return OUT / "representations" / f"{REPRESENTATION}_{split}.checkpoint.json"


def done_path(split: str) -> Path:
    return OUT / "representations" / f"{REPRESENTATION}_{split}.done.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def last_token_pool(last_hidden_state, attention_mask):
    import torch

    if bool(torch.all(attention_mask[:, -1] == 1)):
        return last_hidden_state[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    rows = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[rows, lengths]


def load_or_create(split: str, shape: tuple[int, int]) -> tuple[np.memmap, int]:
    path = representation_path(split)
    checkpoint = checkpoint_path(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not checkpoint.exists():
            raise RuntimeError(f"partial array has no checkpoint: {path}")
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.float16:
            raise RuntimeError(f"partial array mismatch: {array.shape} {array.dtype}")
        start = int(json.loads(checkpoint.read_text())["next_order_index"])
        return array, start
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape), 0


def extract(
    split: str,
    model_path: str,
    revision: str,
    batch_size: int,
    max_length: int,
) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if split == "test" and not (OUT / "selection.json").exists():
        raise RuntimeError("refusing Qwen3 test extraction before validation selection")
    if done_path(split).exists():
        print(f"qwen3 {split}: already complete", flush=True)
        return
    documents = pd.read_parquet(SOURCE / "documents.parquet")
    part = protocol.split_documents(documents, split)
    order = np.argsort(part["text_chars"].to_numpy(), kind="stable")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=revision or None,
        local_files_only=True,
        use_fast=True,
        padding_side="left",
    )
    model = AutoModel.from_pretrained(
        model_path,
        revision=revision or None,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda:0").eval()
    model.config.use_cache = False
    hidden = int(model.config.hidden_size)
    output, order_start = load_or_create(split, (len(part), hidden))
    active_batch_size = batch_size
    torch.set_float32_matmul_precision("high")

    with torch.inference_mode():
        while order_start < len(order):
            positions = order[order_start : order_start + active_batch_size]
            texts = part.iloc[positions]["news_text"].tolist()
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to("cuda:0")
            try:
                result = model(**encoded, use_cache=False, return_dict=True)
                pooled = torch.nn.functional.normalize(
                    last_token_pool(result.last_hidden_state, encoded["attention_mask"]).float(),
                    p=2,
                    dim=1,
                )
            except torch.cuda.OutOfMemoryError:
                del encoded
                torch.cuda.empty_cache()
                if active_batch_size == 1:
                    raise
                active_batch_size = max(1, active_batch_size // 2)
                print(
                    f"qwen3 {split}: CUDA OOM; retrying batch {active_batch_size}",
                    flush=True,
                )
                continue
            output[positions] = pooled.cpu().numpy().astype(np.float16)
            next_index = order_start + len(positions)
            checkpoint_due = (
                next_index % 512 < active_batch_size or next_index == len(part)
            )
            if checkpoint_due:
                output.flush()
                protocol.write_json(
                    checkpoint_path(split),
                    {
                        "next_order_index": next_index,
                        "rows": len(part),
                        "requested_batch_size": batch_size,
                        "active_batch_size": active_batch_size,
                    },
                )
            if next_index % max(200, batch_size) < active_batch_size or next_index == len(part):
                print(f"qwen3 {split}: {next_index}/{len(part)}", flush=True)
            del result, pooled, encoded
            order_start = next_index

    output.flush()
    protocol.write_json(
        done_path(split),
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "model_id": MODEL_ID,
            "model_path": model_path,
            "revision": revision,
            "split": split,
            "rows": len(part),
            "parameters": "8B",
            "hidden_size": hidden,
            "dtype": "float16 storage / bfloat16 forward",
            "pooling": "official last-token pooling then row L2 normalization",
            "padding_side": "left",
            "max_length": max_length,
            "requested_batch_size": batch_size,
            "final_batch_size": active_batch_size,
            "input": "raw news text, matching the BGE embedding baseline",
        },
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def protocol_data():
    documents = pd.read_parquet(SOURCE / "documents.parquet")
    train = protocol.split_documents(documents, "train")
    validation = protocol.split_documents(documents, "val")
    y_train = train[protocol.LABEL].to_numpy(float)
    y_val = validation[protocol.LABEL].to_numpy(float)
    return documents, train, validation, y_train, y_val


def select(workers: int) -> None:
    if (OUT / "selection.json").exists():
        raise RuntimeError("large-embedding selection is already frozen")
    for split in ("train", "val"):
        if not done_path(split).exists():
            raise RuntimeError(f"missing completed Qwen3 {split} embedding")
    if representation_path("test").exists():
        raise RuntimeError("test representation exists before selection freeze")
    _, train, validation, y_train, y_val = protocol_data()
    x_train = protocol.normalize_rows(np.load(representation_path("train"), mmap_mode="r"))
    x_val = protocol.normalize_rows(np.load(representation_path("val"), mmap_mode="r"))
    sweep = protocol.tune_alphas(x_train, y_train, x_val, y_val, workers, "cuda")
    best = sorted(sweep, key=lambda row: (-row["validation_spearman"], row["alpha"]))[0]
    model = protocol.fit_final(x_train, y_train, float(best["alpha"]))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "qwen3_state.npz",
        coef=model.coef_.astype(np.float32),
        intercept=np.float64(model.intercept_),
    )
    pretest = pd.concat([train, validation], ignore_index=True)[
        ["doc_row", "ticker", "date", "split", protocol.LABEL]
    ].copy()
    pretest["impact_label"] = np.concatenate([y_train, y_val])
    pretest["qwen3_embedding_score"] = np.concatenate(
        [model.predict(x_train), model.predict(x_val)]
    ).astype(np.float32)
    pretest.to_parquet(OUT / "pretest_scores.parquet", index=False)
    pd.DataFrame(sweep).to_csv(OUT / "validation_sweep.csv", index=False)
    manifest = json.loads(done_path("train").read_text())
    selection = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "capacity-matched large-embedding control",
        "cutoff_safe": False,
        "cutoff_warning": "Qwen3-Embedding-8B was released after the 2023 test period",
        "frozen_before_test": True,
        "test_representation_present_at_freeze": False,
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "parameters": manifest["parameters"],
        "embedding_dimension": manifest["hidden_size"],
        "pooling": manifest["pooling"],
        "input": manifest["input"],
        "axis_fit_split": "2019-2021 train only",
        "selection_split": "2022 validation only",
        "alpha_grid": list(protocol.ALPHAS),
        "selected_alpha": float(best["alpha"]),
        "validation_spearman": float(best["validation_spearman"]),
        "validation_pearson": float(best["validation_pearson"]),
        "centering": "none; no ticker means are subtracted from labels or representations",
    }
    protocol.write_json(OUT / "selection.json", selection)
    print(json.dumps(selection, indent=2), flush=True)


def evaluate(repetitions: int, workers: int) -> None:
    selection_path = OUT / "selection.json"
    if not selection_path.exists() or not done_path("test").exists():
        raise RuntimeError("freeze selection and complete Qwen3 test extraction first")
    if (OUT / "test_metrics.csv").exists():
        raise RuntimeError("capacity-control test has already been evaluated")
    selection = json.loads(selection_path.read_text())
    documents = pd.read_parquet(SOURCE / "documents.parquet")
    test = protocol.split_documents(documents, "test")
    target = test[protocol.LABEL].to_numpy(float)
    qwen = protocol.normalize_rows(np.load(representation_path("test"), mmap_mode="r"))
    state = np.load(OUT / "qwen3_state.npz")
    qwen_score = (qwen @ state["coef"] + float(state["intercept"])).astype(np.float32)
    existing = pd.read_parquet(UNCENTERED / "all_split_scores.parquet")
    existing_test = existing.loc[existing["split"].eq("test")].sort_values("split_row") if "split_row" in existing else existing.loc[existing["split"].eq("test")].sort_values("doc_row")
    existing_test = existing_test.reset_index(drop=True)
    if not np.array_equal(existing_test["doc_row"].to_numpy(), test["doc_row"].to_numpy()):
        raise RuntimeError("existing Llama/BGE test scores do not align with Qwen3 rows")
    llama_score = existing_test["activation_score"].to_numpy(float)
    bge_score = existing_test["embedding_score"].to_numpy(float)
    metrics = pd.DataFrame(
        [
            {
                "method": "bge_m3_embedding",
                "parameters": "0.57B",
                "test_spearman": protocol.safe_spearman(bge_score, target),
                "test_pearson": protocol.safe_pearson(bge_score, target),
            },
            {
                "method": "qwen3_embedding_8b",
                "parameters": "8B",
                "test_spearman": protocol.safe_spearman(qwen_score, target),
                "test_pearson": protocol.safe_pearson(qwen_score, target),
            },
            {
                "method": "llama2_activation",
                "parameters": "7B",
                "test_spearman": protocol.safe_spearman(llama_score, target),
                "test_pearson": protocol.safe_pearson(llama_score, target),
            },
        ]
    )
    metrics.to_csv(OUT / "test_metrics.csv", index=False)
    frame = test[["date", "ticker"]].reset_index(drop=True)
    direct = pd.DataFrame(
        [
            {
                "comparison": "llama2_activation_minus_qwen3_embedding_8b",
                **protocol.parallel_bootstrap(
                    frame, llama_score, qwen_score, target, block, repetitions, workers
                ),
            }
            for block in ("date", "ticker")
        ]
    )
    direct.to_csv(OUT / "activation_vs_qwen3_bootstrap.csv", index=False)
    test_scores = test[["doc_row", "ticker", "date", "split", protocol.LABEL]].copy()
    test_scores["impact_label"] = target
    test_scores["qwen3_embedding_score"] = qwen_score
    qwen_scores = pd.concat(
        [pd.read_parquet(OUT / "pretest_scores.parquet"), test_scores], ignore_index=True
    ).sort_values("doc_row")
    qwen_scores.to_parquet(OUT / "all_split_qwen3_scores.parquet", index=False)
    all_scores = existing.merge(
        qwen_scores[["doc_row", "qwen3_embedding_score"]],
        on="doc_row",
        how="inner",
        validate="one_to_one",
    )
    if len(all_scores) != len(existing):
        raise RuntimeError("Qwen3 score merge dropped rows")
    all_scores.to_parquet(OUT / "all_model_scores.parquet", index=False)
    protocol.write_json(
        OUT / "evaluation_manifest.json",
        {
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "selection_sha256": sha256(selection_path),
            "test_rows": len(test),
            "bootstrap_repetitions": repetitions,
            "bootstrap_workers": workers,
            "interpretation": "capacity control only; Qwen3 is not cutoff-safe for 2023",
        },
    )
    print(metrics.to_string(index=False), flush=True)
    print(direct.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    extract_parser.add_argument("--model-path", required=True)
    extract_parser.add_argument("--revision", default=MODEL_REVISION)
    extract_parser.add_argument("--batch-size", type=int, default=16)
    extract_parser.add_argument("--max-length", type=int, default=512)
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    evaluate_parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "extract":
        extract(args.split, args.model_path, args.revision, args.batch_size, args.max_length)
    elif args.command == "select":
        select(args.workers)
    elif args.command == "evaluate":
        evaluate(args.bootstrap_repetitions, args.workers)


if __name__ == "__main__":
    main()
