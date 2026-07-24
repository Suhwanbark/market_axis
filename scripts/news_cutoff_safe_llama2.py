#!/usr/bin/env python3
"""Cutoff-safe News market-impact axis with Llama 2 and BGE-M3.

The protocol is intentionally staged:

1. ``prepare`` builds the normalized Parkinson label and freezes document rows.
2. ``extract`` creates one representation array for one split and model.
3. ``select`` uses train/validation only and writes a frozen selection.
4. ``evaluate`` is the only command allowed to open the 2023 test arrays.

Llama 2 and BGE parameters remain frozen.  Only the supervised Ridge direction
is fitted on 2019--2021 News labels.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "news_cutoff_safe_llama2"
SOURCE_RECORDS = ROOT / "outputs" / "fintexts_news_axis_e2e" / "news_records.parquet"
SOURCE_AUDIT = ROOT / "outputs" / "fintexts_news_axis_e2e" / "data_audit.json"
HF_CACHE = ROOT / "outputs" / "fintexts_news_axis_e2e" / "hf_cache"
LABEL = "parkinson_market_adjusted_expansion"
SPLITS = ("train", "val", "test")
REPRESENTATIONS = ("llama2", "bge")
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
EPS = 1e-12
SEED = 20260721
PREFIX = "Financial news:\n"
SUFFIX = "\n"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=str, sort_keys=True), encoding="utf-8"
    )


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def assert_disk_floor(minimum_gib: float) -> None:
    available = shutil.disk_usage(ROOT).free / 1024**3
    if available < minimum_gib:
        raise RuntimeError(
            f"disk floor reached: {available:.1f} GiB < {minimum_gib:.1f} GiB"
        )


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan")
    return float(pearsonr(x[valid], y[valid]).statistic)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def load_llama_layers_to_ram(path: Path) -> np.ndarray:
    """Stage a document-major activation file as contiguous layer-major RAM."""
    mapped = np.load(path, mmap_mode="r")
    if mapped.ndim != 3:
        raise RuntimeError(f"expected a 3D Llama activation array, found {mapped.shape}")
    print(f"selection: staging {path.name} from GPFS into RAM", flush=True)
    document_major = np.array(mapped, copy=True, order="C")
    del mapped
    layer_major = np.empty(
        (document_major.shape[1], document_major.shape[0], document_major.shape[2]),
        dtype=document_major.dtype,
    )
    for layer in range(document_major.shape[1]):
        np.copyto(layer_major[layer], document_major[:, layer, :])
    del document_major
    print(f"selection: staged {path.name} as {layer_major.shape}", flush=True)
    return layer_major


def load_selected_llama_layer_to_ram(path: Path, layer_index: int) -> np.ndarray:
    """Read a document-major activation file sequentially, then select in RAM."""
    mapped = np.load(path, mmap_mode="r")
    if mapped.ndim != 3 or not 0 <= layer_index < mapped.shape[1]:
        raise RuntimeError(f"invalid layer {layer_index} for activation shape {mapped.shape}")
    print(f"evaluation: staging {path.name} sequentially into RAM", flush=True)
    document_major = np.array(mapped, copy=True, order="C")
    del mapped
    selected = np.array(document_major[:, layer_index, :], copy=True, order="C")
    del document_major
    return selected


def load_raw_prices() -> pd.DataFrame:
    snapshots = HF_CACHE / "datasets--EXAONE-BI--FinTexTS" / "snapshots"
    snapshot_paths = sorted(path for path in snapshots.iterdir() if path.is_dir())
    if len(snapshot_paths) != 1:
        raise RuntimeError(f"expected one pinned FinTexTS snapshot, found {snapshot_paths}")
    paths = sorted((snapshot_paths[0] / "data").glob("train-*.parquet"))
    if len(paths) != 4:
        raise RuntimeError(f"expected four FinTexTS parquet shards, found {paths}")
    prices = pd.concat(
        [
            pd.read_parquet(path, columns=["date", "ticker", "high", "low"])
            for path in paths
        ],
        ignore_index=True,
    )
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    closures = set(pd.to_datetime(audit["carried_market_closure_dates"]))
    prices = prices.loc[~prices["date"].isin(closures)].copy()
    valid = prices["high"].gt(0) & prices["low"].gt(0) & prices["high"].ge(prices["low"])
    prices = prices.loc[valid].sort_values(["ticker", "date"]).reset_index(drop=True)
    prices["parkinson"] = (
        np.log(prices["high"] / prices["low"]).pow(2) / (4.0 * math.log(2.0))
    )
    return prices


def normalized_parkinson_labels() -> pd.DataFrame:
    prices = load_raw_prices()
    group = prices.groupby("ticker", sort=False)["parkinson"]
    # News dated t: history ends at t-1; reaction begins at the next real session t+1.
    prices["firm_pre20"] = group.transform(
        lambda value: value.shift(1).rolling(20, min_periods=20).mean()
    )
    # Preserve the historical representation-benchmark eligibility filter.
    # That benchmark formed its pre-20 window through the dated row before the
    # normalized label was introduced.  It removes 39 zero-range train rows and
    # defines the canonical 50,296-document alignment used by the main table.
    # The value is *only* an alignment filter; the label above remains strictly
    # pre-event and ends at t-1.
    prices["alignment_pre20"] = group.transform(
        lambda value: value.rolling(20, min_periods=20).mean()
    )
    prices["firm_post5"] = pd.concat(
        [group.shift(-offset) for offset in range(1, 6)], axis=1
    ).mean(axis=1, skipna=False)

    market_daily = prices.groupby("date", sort=True)["parkinson"].mean()
    market = pd.DataFrame({"date": market_daily.index})
    market["market_pre20"] = (
        market_daily.shift(1).rolling(20, min_periods=20).mean().to_numpy()
    )
    market["market_post5"] = pd.concat(
        [market_daily.shift(-offset) for offset in range(1, 6)], axis=1
    ).mean(axis=1, skipna=False).to_numpy()

    labels = prices[
        ["ticker", "date", "firm_pre20", "firm_post5", "alignment_pre20"]
    ].merge(
        market, on="date", validate="many_to_one"
    )
    labels[LABEL] = 0.5 * (
        np.log(labels["firm_post5"].clip(lower=EPS))
        - np.log(labels["firm_pre20"].clip(lower=EPS))
        - np.log(labels["market_post5"].clip(lower=EPS))
        + np.log(labels["market_pre20"].clip(lower=EPS))
    )
    return labels


def prepare() -> None:
    from simple_impact_label_benchmark import (
        DIAGNOSTIC_LABELS,
        MAIN_LABELS,
        build_labels,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    if not SOURCE_RECORDS.exists() or not SOURCE_AUDIT.exists():
        raise FileNotFoundError(
            "run fintexts_news_axis_pipeline.py --stage prepare before this command"
        )
    records = pd.read_parquet(
        SOURCE_RECORDS,
        columns=["item_id", "ticker", "date", "split", "news_text", "text_hash"],
    )
    records["date"] = pd.to_datetime(records["date"]).dt.tz_localize(None)
    # Reuse the exact historical eligibility mask.  The main analysis aligned
    # all normalized OHLC and abnormal-return diagnostic labels before fitting
    # the Parkinson axis, which removes 39 train rows.  Keeping this alignment
    # makes the cutoff-safe comparison differ only in the text model.
    labels = build_labels().replace([np.inf, -np.inf], np.nan)
    eligibility = [*MAIN_LABELS, *DIAGNOSTIC_LABELS]
    labels = labels.dropna(subset=eligibility)
    documents = records.merge(
        labels[["ticker", "date", LABEL]],
        on=["ticker", "date"],
        validate="one_to_one",
    )
    documents = documents.sort_values("item_id", kind="stable").reset_index(drop=True)
    documents["doc_row"] = np.arange(len(documents), dtype=np.int64)
    documents["split_row"] = documents.groupby("split", sort=False).cumcount().astype(np.int64)
    documents["text_chars"] = documents["news_text"].str.len().astype(np.int32)
    documents.to_parquet(OUT / "documents.parquet", index=False)

    counts = documents.groupby("split").size().to_dict()
    expected = {"train": 25767, "val": 12051, "test": 12478}
    if counts != expected:
        raise RuntimeError(f"unexpected cutoff-safe sample counts: {counts} != {expected}")
    source_audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    write_json(
        OUT / "data_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "EXAONE-BI/FinTexTS targetCompany_category1-3",
            "source_snapshot": "6a4ce04ac3f2e8a4ae15a5826e6980b50560eb29",
            "source_records_sha256": sha256(SOURCE_RECORDS),
            "counts": counts,
            "split": "2019-2021 train / 2022 validation / 2023 exploratory test",
            "label": LABEL,
            "label_definition": (
                "0.5 * [log(firm post5/pre20 Parkinson variance) - "
                "log(equal-weight market post5/pre20 Parkinson variance)]"
            ),
            "news_timing": "pre t-20..t-1, post next actual sessions t+1..t+5",
            "input": "Financial news prefix; mean pooling over body tokens only",
            "max_tokens": 512,
            "closure_dates_removed": source_audit["carried_market_closure_dates"],
            "test_policy": "test representations and outcomes stay unopened until select freezes validation choices",
        },
    )
    print(json.dumps({"counts": counts, "output": str(OUT / "documents.parquet")}, indent=2))


def representation_path(representation: str, split: str) -> Path:
    return OUT / "representations" / f"{representation}_{split}.npy"


def checkpoint_path(representation: str, split: str) -> Path:
    return OUT / "representations" / f"{representation}_{split}.checkpoint.json"


def done_path(representation: str, split: str) -> Path:
    return OUT / "representations" / f"{representation}_{split}.done.json"


def body_encoded(tokenizer, texts: list[str], max_length: int):
    prefix_ids = tokenizer(PREFIX, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(SUFFIX, add_special_tokens=False).input_ids
    budget = max_length - len(prefix_ids) - len(suffix_ids)
    text_batches = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=budget,
        padding=False,
    ).input_ids
    encoded: list[list[int]] = []
    spans: list[tuple[int, int]] = []
    for text_ids in text_batches:
        ids = prefix_ids + text_ids + suffix_ids
        encoded.append(ids)
        spans.append((len(prefix_ids), len(prefix_ids) + len(text_ids)))
    pad_id = tokenizer.pad_token_id
    width = max(len(ids) for ids in encoded)
    input_ids = np.full((len(encoded), width), pad_id, dtype=np.int64)
    attention = np.zeros((len(encoded), width), dtype=np.int64)
    for index, ids in enumerate(encoded):
        input_ids[index, : len(ids)] = ids
        attention[index, : len(ids)] = 1
    return input_ids, attention, spans


def load_or_create_array(
    path: Path, shape: tuple[int, ...], checkpoint: Path
) -> tuple[np.memmap, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and checkpoint.exists():
        array = np.lib.format.open_memmap(path, mode="r+")
        if tuple(array.shape) != shape:
            raise RuntimeError(f"shape mismatch for resume: {array.shape} != {shape}")
        start = int(json.loads(checkpoint.read_text())["next_order_index"])
        return array, start
    array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    return array, 0


def extract_llama(
    split: str,
    model_path: str,
    model_revision: str,
    batch_size: int,
    max_length: int,
    disk_floor_gib: float,
) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if split == "test" and not (OUT / "selection.json").exists():
        raise RuntimeError("refusing to extract test before validation selection is frozen")
    assert_disk_floor(disk_floor_gib)
    documents = pd.read_parquet(OUT / "documents.parquet")
    part = documents.loc[documents["split"].eq(split)].sort_values("split_row")
    order = np.argsort(part["text_chars"].to_numpy(), kind="stable")
    if done_path("llama2", split).exists():
        print(f"llama2 {split}: already complete")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=model_revision or None, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_path,
        revision=model_revision or None,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda:0").eval()
    layers = int(model.config.num_hidden_layers)
    hidden = int(model.config.hidden_size)
    output, start = load_or_create_array(
        representation_path("llama2", split), (len(part), layers, hidden), checkpoint_path("llama2", split)
    )

    active_batch_size = batch_size
    order_start = start
    with torch.inference_mode():
        while order_start < len(order):
            positions = order[order_start : order_start + active_batch_size]
            batch = part.iloc[positions]
            ids, mask, spans = body_encoded(tokenizer, batch["news_text"].tolist(), max_length)
            input_ids = torch.as_tensor(ids, device="cuda:0")
            attention = torch.as_tensor(mask, device="cuda:0")
            try:
                result = model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            except torch.cuda.OutOfMemoryError:
                del input_ids, attention
                torch.cuda.empty_cache()
                if active_batch_size == 1:
                    raise
                active_batch_size = max(1, active_batch_size // 2)
                print(
                    f"llama2 {split}: CUDA OOM; retrying with batch {active_batch_size}",
                    flush=True,
                )
                continue
            pooled_layers = []
            for layer in range(1, layers + 1):
                state = result.hidden_states[layer]
                pooled = torch.stack(
                    [state[row, begin:end].mean(dim=0) for row, (begin, end) in enumerate(spans)]
                )
                pooled_layers.append(pooled)
            values = torch.stack(pooled_layers, dim=1).float().cpu().numpy().astype(np.float16)
            output[positions] = values
            next_index = order_start + len(positions)
            checkpoint_due = (
                next_index % 512 < active_batch_size or next_index == len(part)
            )
            if checkpoint_due:
                output.flush()
                write_json(
                    checkpoint_path("llama2", split),
                    {
                        "next_order_index": next_index,
                        "rows": len(part),
                        "requested_batch_size": batch_size,
                        "active_batch_size": active_batch_size,
                    },
                )
            if next_index % max(200, batch_size) < batch_size or next_index == len(part):
                print(f"llama2 {split}: {next_index}/{len(part)}", flush=True)
            del result, pooled_layers, values, input_ids, attention
            if checkpoint_due:
                assert_disk_floor(disk_floor_gib)
            order_start = next_index

    output.flush()

    write_json(
        done_path("llama2", split),
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "model": model_path,
            "revision": model_revision,
            "split": split,
            "rows": len(part),
            "layers": layers,
            "hidden_size": hidden,
            "dtype": "float16 storage / bfloat16 forward",
            "pooling": "body-token mean per layer",
            "max_length": max_length,
            "requested_batch_size": batch_size,
            "final_batch_size": active_batch_size,
        },
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def extract_bge(
    split: str,
    model_path: str,
    model_revision: str,
    batch_size: int,
    max_length: int,
    disk_floor_gib: float,
) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if split == "test" and not (OUT / "selection.json").exists():
        raise RuntimeError("refusing to extract test before validation selection is frozen")
    assert_disk_floor(disk_floor_gib)
    documents = pd.read_parquet(OUT / "documents.parquet")
    part = documents.loc[documents["split"].eq(split)].sort_values("split_row")
    order = np.argsort(part["text_chars"].to_numpy(), kind="stable")
    if done_path("bge", split).exists():
        print(f"bge {split}: already complete")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=model_revision or None, local_files_only=True, use_fast=True
    )
    model = AutoModel.from_pretrained(
        model_path,
        revision=model_revision or None,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda:0").eval()
    hidden = int(model.config.hidden_size)
    output, start = load_or_create_array(
        representation_path("bge", split), (len(part), hidden), checkpoint_path("bge", split)
    )

    with torch.inference_mode():
        for order_start in range(start, len(order), batch_size):
            positions = order[order_start : order_start + batch_size]
            texts = part.iloc[positions]["news_text"].tolist()
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            result = model(**encoded, return_dict=True)
            pooled = torch.nn.functional.normalize(result.last_hidden_state[:, 0].float(), dim=1)
            output[positions] = pooled.cpu().numpy().astype(np.float16)
            output.flush()
            next_index = order_start + len(positions)
            write_json(
                checkpoint_path("bge", split),
                {"next_order_index": next_index, "rows": len(part), "batch_size": batch_size},
            )
            if next_index % max(500, batch_size) < batch_size or next_index == len(part):
                print(f"bge {split}: {next_index}/{len(part)}", flush=True)
            del result, pooled, encoded
            assert_disk_floor(disk_floor_gib)

    write_json(
        done_path("bge", split),
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "model": model_path,
            "revision": model_revision,
            "split": split,
            "rows": len(part),
            "hidden_size": hidden,
            "dtype": "float16",
            "pooling": "CLS then row L2 normalization",
            "max_length": max_length,
            "batch_size": batch_size,
        },
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def split_documents(documents: pd.DataFrame, split: str) -> pd.DataFrame:
    part = documents.loc[documents["split"].eq(split)].sort_values("split_row").copy()
    expected = np.arange(len(part))
    if not np.array_equal(part["split_row"].to_numpy(), expected):
        raise RuntimeError(f"non-contiguous split rows for {split}")
    return part.reset_index(drop=True)


def ticker_means_matrix(
    train_values: np.ndarray, train_tickers: np.ndarray, tickers: list[str]
) -> np.ndarray:
    global_mean = train_values.mean(axis=0, keepdims=True)
    means = np.repeat(global_mean, len(tickers), axis=0).astype(np.float32)
    for index, ticker in enumerate(tickers):
        mask = train_tickers == ticker
        if mask.any():
            means[index] = train_values[mask].mean(axis=0)
    return means


def center_matrix(values: np.ndarray, row_tickers: np.ndarray, tickers: list[str], means: np.ndarray) -> np.ndarray:
    lookup = {ticker: index for index, ticker in enumerate(tickers)}
    fallback = means.mean(axis=0)
    centered = np.asarray(values, dtype=np.float32).copy()
    for ticker in np.unique(row_tickers):
        mask = row_tickers == ticker
        centered[mask] -= means[lookup[ticker]] if ticker in lookup else fallback
    return centered


def label_centering(
    train_y: np.ndarray, train_tickers: np.ndarray, tickers: list[str]
) -> np.ndarray:
    fallback = float(train_y.mean())
    result = np.full(len(tickers), fallback, dtype=np.float64)
    for index, ticker in enumerate(tickers):
        mask = train_tickers == ticker
        if mask.any():
            result[index] = float(train_y[mask].mean())
    return result


def center_labels(values: np.ndarray, row_tickers: np.ndarray, tickers: list[str], means: np.ndarray) -> np.ndarray:
    lookup = {ticker: index for index, ticker in enumerate(tickers)}
    fallback = float(means.mean())
    return np.asarray(values, dtype=float) - np.asarray(
        [means[lookup[ticker]] if ticker in lookup else fallback for ticker in row_tickers]
    )


def ridge_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float,
) -> dict:
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-5)
    model.fit(x_train, y_train)
    return {
        "alpha": alpha,
        "train_spearman": safe_spearman(model.predict(x_train), y_train),
        "validation_spearman": safe_spearman(model.predict(x_val), y_val),
        "validation_pearson": safe_pearson(model.predict(x_val), y_val),
    }


def tune_alphas(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    workers: int,
    device: str,
) -> list[dict]:
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Ridge requested but CUDA is unavailable")
        train_tensor = torch.as_tensor(
            np.ascontiguousarray(x_train), dtype=torch.float32, device="cuda:0"
        )
        val_tensor = torch.as_tensor(
            np.ascontiguousarray(x_val), dtype=torch.float32, device="cuda:0"
        )
        train_target = torch.as_tensor(y_train, dtype=torch.float32, device="cuda:0")
        val_target = torch.as_tensor(y_val, dtype=torch.float32, device="cuda:0")
        x_mean = train_tensor.mean(dim=0)
        y_mean = train_target.mean()
        centered_x = train_tensor - x_mean
        centered_y = train_target - y_mean
        gram = centered_x.T @ centered_x
        rhs = centered_x.T @ centered_y
        # One eigendecomposition yields exact primal Ridge solutions for the
        # entire alpha grid, avoiding six repeated CPU LSQR fits per layer.
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.clamp_min_(0.0)
        projected_rhs = eigenvectors.T @ rhs
        rows = []
        for alpha in ALPHAS:
            coefficient = eigenvectors @ (projected_rhs / (eigenvalues + alpha))
            intercept = y_mean - x_mean @ coefficient
            train_score = train_tensor @ coefficient + intercept
            val_score = val_tensor @ coefficient + intercept
            rows.append(
                {
                    "alpha": alpha,
                    "train_spearman": safe_spearman(
                        train_score.cpu().numpy(), y_train
                    ),
                    "validation_spearman": safe_spearman(
                        val_score.cpu().numpy(), y_val
                    ),
                    "validation_pearson": safe_pearson(
                        val_score.cpu().numpy(), y_val
                    ),
                }
            )
        del (
            train_tensor,
            val_tensor,
            train_target,
            val_target,
            centered_x,
            centered_y,
            gram,
            rhs,
            eigenvalues,
            eigenvectors,
            projected_rhs,
        )
        torch.cuda.empty_cache()
        return rows

    from threadpoolctl import threadpool_limits

    threads_per_fit = max(1, (os.cpu_count() or 1) // min(workers, len(ALPHAS)))
    with threadpool_limits(limits=threads_per_fit):
        with ThreadPoolExecutor(max_workers=min(workers, len(ALPHAS))) as executor:
            futures = [
                executor.submit(ridge_candidate, x_train, y_train, x_val, y_val, alpha)
                for alpha in ALPHAS
            ]
            return [future.result() for future in futures]


def fit_final(x: np.ndarray, y: np.ndarray, alpha: float) -> Ridge:
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-5)
    model.fit(x, y)
    return model


def select(workers: int, ridge_device: str) -> None:
    if (OUT / "selection.json").exists():
        raise RuntimeError("selection is already frozen; remove outputs explicitly to rerun")
    for representation in REPRESENTATIONS:
        for split in ("train", "val"):
            if not done_path(representation, split).exists():
                raise RuntimeError(f"missing completed {representation} {split} representation")
    documents = pd.read_parquet(OUT / "documents.parquet")
    train = split_documents(documents, "train")
    validation = split_documents(documents, "val")
    tickers = sorted(train["ticker"].unique().tolist())
    train_tickers = train["ticker"].to_numpy(str)
    val_tickers = validation["ticker"].to_numpy(str)
    label_means = label_centering(train[LABEL].to_numpy(float), train_tickers, tickers)
    y_train = center_labels(train[LABEL].to_numpy(float), train_tickers, tickers, label_means)
    y_val = center_labels(validation[LABEL].to_numpy(float), val_tickers, tickers, label_means)
    partial_path = OUT / "validation_sweep.partial.csv"
    if partial_path.exists():
        records = pd.read_csv(partial_path).to_dict("records")
        print(f"selection: resuming {len(records)} saved sweep rows", flush=True)
    else:
        records: list[dict] = []

    effective_ridge_device = "cuda" if ridge_device == "auto" else ridge_device
    ridge_benchmark_seconds: dict[str, float] = {}

    # BGE-M3 alpha selection.
    bge_train = normalize_rows(np.load(representation_path("bge", "train"), mmap_mode="r"))
    bge_val = normalize_rows(np.load(representation_path("bge", "val"), mmap_mode="r"))
    bge_means = ticker_means_matrix(bge_train, train_tickers, tickers)
    bge_train = center_matrix(bge_train, train_tickers, tickers, bge_means)
    bge_val = center_matrix(bge_val, val_tickers, tickers, bge_means)
    bge_rows = [row for row in records if row["representation"] == "bge_m3_embedding"]
    if len(bge_rows) != len(ALPHAS):
        records = [row for row in records if row["representation"] != "bge_m3_embedding"]
        for row in tune_alphas(
            bge_train, y_train, bge_val, y_val, workers, effective_ridge_device
        ):
            records.append({"representation": "bge_m3_embedding", "layer": np.nan, **row})
        bge_rows = [
            row for row in records if row["representation"] == "bge_m3_embedding"
        ]
    bge_best = sorted(bge_rows, key=lambda row: (-row["validation_spearman"], row["alpha"]))[0]
    bge_model = fit_final(bge_train, y_train, float(bge_best["alpha"]))
    np.savez_compressed(
        OUT / "bge_state.npz",
        coef=bge_model.coef_.astype(np.float32),
        intercept=np.float64(bge_model.intercept_),
        ticker_means=bge_means,
    )
    pretest = pd.concat([train, validation], ignore_index=True)[
        ["doc_row", "ticker", "date", "split", LABEL]
    ].copy()
    pretest["embedding_score"] = np.concatenate(
        [bge_model.predict(bge_train), bge_model.predict(bge_val)]
    ).astype(np.float32)
    del bge_train, bge_val, bge_means

    # Llama layer and alpha selection.
    train_map = np.load(representation_path("llama2", "train"), mmap_mode="r")
    val_map = np.load(representation_path("llama2", "val"), mmap_mode="r")
    if train_map.shape[1:] != val_map.shape[1:]:
        raise RuntimeError("train/validation Llama shapes disagree")
    num_layers = int(train_map.shape[1])
    del train_map, val_map
    llama_train_array = load_llama_layers_to_ram(representation_path("llama2", "train"))
    llama_val_array = load_llama_layers_to_ram(representation_path("llama2", "val"))
    completed_layers = {
        int(layer)
        for layer, count in (
            pd.DataFrame(records)
            .loc[lambda frame: frame["representation"].eq("llama2_activation")]
            .groupby("layer")
            .size()
            .items()
        )
        if count == len(ALPHAS)
    }
    if ridge_device == "auto":
        benchmark_layer = next(
            (layer for layer in range(1, num_layers + 1) if layer not in completed_layers),
            1,
        )
        x_train = normalize_rows(llama_train_array[benchmark_layer - 1])
        x_val = normalize_rows(llama_val_array[benchmark_layer - 1])
        means = ticker_means_matrix(x_train, train_tickers, tickers)
        x_train = center_matrix(x_train, train_tickers, tickers, means)
        x_val = center_matrix(x_val, val_tickers, tickers, means)
        benchmark_rows: dict[str, list[dict]] = {}
        for device in ("cuda", "cpu"):
            started = time.perf_counter()
            benchmark_rows[device] = tune_alphas(
                x_train, y_train, x_val, y_val, workers, device
            )
            ridge_benchmark_seconds[device] = time.perf_counter() - started
            print(
                f"selection: {device} full-layer benchmark "
                f"{ridge_benchmark_seconds[device]:.3f}s",
                flush=True,
            )
        effective_ridge_device = min(
            ridge_benchmark_seconds, key=ridge_benchmark_seconds.get
        )
        print(
            f"selection: using {effective_ridge_device} for remaining layers",
            flush=True,
        )
        records = [
            row
            for row in records
            if not (
                row["representation"] == "llama2_activation"
                and int(row["layer"]) == benchmark_layer
            )
        ]
        records.extend(
            {
                "representation": "llama2_activation",
                "layer": benchmark_layer,
                **row,
            }
            for row in benchmark_rows[effective_ridge_device]
        )
        completed_layers.add(benchmark_layer)
        pd.DataFrame(records).to_csv(partial_path, index=False)
        del x_train, x_val, means, benchmark_rows

    for layer in range(1, num_layers + 1):
        if layer in completed_layers:
            print(f"selection: reusing Llama layer {layer}/{num_layers}", flush=True)
            continue
        records = [
            row
            for row in records
            if not (
                row["representation"] == "llama2_activation"
                and int(row["layer"]) == layer
            )
        ]
        x_train = normalize_rows(llama_train_array[layer - 1])
        x_val = normalize_rows(llama_val_array[layer - 1])
        means = ticker_means_matrix(x_train, train_tickers, tickers)
        x_train = center_matrix(x_train, train_tickers, tickers, means)
        x_val = center_matrix(x_val, val_tickers, tickers, means)
        for row in tune_alphas(
            x_train, y_train, x_val, y_val, workers, effective_ridge_device
        ):
            records.append({"representation": "llama2_activation", "layer": layer, **row})
        print(f"selection: completed Llama layer {layer}/{num_layers}", flush=True)
        del x_train, x_val, means
        pd.DataFrame(records).to_csv(partial_path, index=False)

    llama_rows = [row for row in records if row["representation"] == "llama2_activation"]
    llama_best = sorted(
        llama_rows,
        key=lambda row: (-row["validation_spearman"], row["layer"], row["alpha"]),
    )[0]
    layer = int(llama_best["layer"])
    x_train = normalize_rows(llama_train_array[layer - 1])
    x_val = normalize_rows(llama_val_array[layer - 1])
    llama_means = ticker_means_matrix(x_train, train_tickers, tickers)
    x_train = center_matrix(x_train, train_tickers, tickers, llama_means)
    x_val = center_matrix(x_val, val_tickers, tickers, llama_means)
    llama_model = fit_final(x_train, y_train, float(llama_best["alpha"]))
    np.savez_compressed(
        OUT / "llama2_state.npz",
        coef=llama_model.coef_.astype(np.float32),
        intercept=np.float64(llama_model.intercept_),
        ticker_means=llama_means,
    )
    pretest["activation_score"] = np.concatenate(
        [llama_model.predict(x_train), llama_model.predict(x_val)]
    ).astype(np.float32)
    pretest["centered_impact_label"] = np.concatenate([y_train, y_val])
    pretest.to_parquet(OUT / "pretest_scores.parquet", index=False)
    pd.DataFrame(records).to_csv(OUT / "validation_sweep.csv", index=False)
    partial_path.unlink(missing_ok=True)

    llama_manifest = json.loads(done_path("llama2", "train").read_text())
    bge_manifest = json.loads(done_path("bge", "train").read_text())
    selection = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_before_test": True,
        "label": LABEL,
        "centering": "subtract train-ticker representation centroid and train-ticker label mean",
        "alpha_grid": list(ALPHAS),
        "axis_fit_split": "train only",
        "selection_split": "2022 validation only",
        "ridge_grid_device": effective_ridge_device,
        "ridge_benchmark_seconds": ridge_benchmark_seconds,
        "test_representation_present_at_freeze": any(
            representation_path(rep, "test").exists() for rep in REPRESENTATIONS
        ),
        "tickers": tickers,
        "label_ticker_means": label_means.tolist(),
        "bge": {
            "model": bge_manifest["model"],
            "revision": bge_manifest["revision"],
            "selected_alpha": float(bge_best["alpha"]),
            "validation_spearman": float(bge_best["validation_spearman"]),
        },
        "llama2": {
            "model": llama_manifest["model"],
            "revision": llama_manifest["revision"],
            "pretraining_cutoff": "September 2022 (base model; no chat tuning)",
            "selected_layer": layer,
            "selected_alpha": float(llama_best["alpha"]),
            "validation_spearman": float(llama_best["validation_spearman"]),
            "num_layers": num_layers,
        },
    }
    write_json(OUT / "selection.json", selection)
    print(json.dumps(selection, indent=2), flush=True)


def apply_state(
    values: np.ndarray,
    tickers_for_rows: np.ndarray,
    ticker_names: list[str],
    state_path: Path,
) -> np.ndarray:
    state = np.load(state_path)
    centered = center_matrix(values, tickers_for_rows, ticker_names, state["ticker_means"])
    return (centered @ state["coef"] + float(state["intercept"])).astype(np.float32)


def block_bootstrap_draws(
    groups: list[np.ndarray],
    activation: np.ndarray,
    embedding: np.ndarray,
    target: np.ndarray,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for draw in range(repetitions):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        draws[draw] = safe_spearman(activation[indices], target[indices]) - safe_spearman(
            embedding[indices], target[indices]
        )
    return draws


def parallel_bootstrap(
    frame: pd.DataFrame,
    activation: np.ndarray,
    embedding: np.ndarray,
    target: np.ndarray,
    block: str,
    repetitions: int,
    workers: int,
) -> dict:
    groups = [part.index.to_numpy() for _, part in frame.reset_index(drop=True).groupby(block)]
    workers = max(1, min(workers, repetitions))
    counts = [repetitions // workers + (index < repetitions % workers) for index in range(workers)]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                block_bootstrap_draws,
                groups,
                activation,
                embedding,
                target,
                count,
                SEED + index + (10000 if block == "ticker" else 0),
            )
            for index, count in enumerate(counts)
            if count
        ]
        draws = np.concatenate([future.result() for future in futures])
    difference = safe_spearman(activation, target) - safe_spearman(embedding, target)
    return {
        "block": block,
        "repetitions": repetitions,
        "difference": difference,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "one_sided_p_activation_not_better": float(np.mean(draws <= 0)),
    }


def evaluate(bootstrap_repetitions: int, workers: int) -> None:
    selection_path = OUT / "selection.json"
    if not selection_path.exists():
        raise RuntimeError("validation selection must be frozen before test evaluation")
    if (OUT / "test_metrics.csv").exists():
        raise RuntimeError("test has already been evaluated; refusing an implicit rerun")
    for representation in REPRESENTATIONS:
        if not done_path(representation, "test").exists():
            raise RuntimeError(f"missing completed {representation} test representation")
    selection = json.loads(selection_path.read_text())
    documents = pd.read_parquet(OUT / "documents.parquet")
    test = split_documents(documents, "test")
    tickers = list(selection["tickers"])
    row_tickers = test["ticker"].to_numpy(str)
    target = center_labels(
        test[LABEL].to_numpy(float),
        row_tickers,
        tickers,
        np.asarray(selection["label_ticker_means"], dtype=float),
    )

    bge = normalize_rows(np.load(representation_path("bge", "test"), mmap_mode="r"))
    embedding_score = apply_state(bge, row_tickers, tickers, OUT / "bge_state.npz")
    layer = int(selection["llama2"]["selected_layer"])
    activation = normalize_rows(
        load_selected_llama_layer_to_ram(
            representation_path("llama2", "test"), layer - 1
        )
    )
    activation_score = apply_state(
        activation, row_tickers, tickers, OUT / "llama2_state.npz"
    )

    metrics = pd.DataFrame(
        [
            {
                "method": "bge_m3_embedding",
                "selected_layer": np.nan,
                "selected_alpha": float(selection["bge"]["selected_alpha"]),
                "n_test": len(test),
                "test_spearman": safe_spearman(embedding_score, target),
                "test_pearson": safe_pearson(embedding_score, target),
            },
            {
                "method": "llama2_activation",
                "selected_layer": layer,
                "selected_alpha": float(selection["llama2"]["selected_alpha"]),
                "n_test": len(test),
                "test_spearman": safe_spearman(activation_score, target),
                "test_pearson": safe_pearson(activation_score, target),
            },
        ]
    )
    metrics.to_csv(OUT / "test_metrics.csv", index=False)

    scores = test[["doc_row", "ticker", "date", "split", LABEL]].copy()
    scores["centered_impact_label"] = target
    scores["activation_score"] = activation_score
    scores["embedding_score"] = embedding_score
    pretest = pd.read_parquet(OUT / "pretest_scores.parquet")
    all_scores = pd.concat([pretest, scores], ignore_index=True).sort_values("doc_row")
    all_scores.to_parquet(OUT / "all_split_scores.parquet", index=False)

    test_for_bootstrap = scores[["date", "ticker"]].reset_index(drop=True)
    bootstrap = pd.DataFrame(
        [
            parallel_bootstrap(
                test_for_bootstrap,
                activation_score,
                embedding_score,
                target,
                block,
                bootstrap_repetitions,
                workers,
            )
            for block in ("date", "ticker")
        ]
    )
    bootstrap.to_csv(OUT / "activation_vs_bge_bootstrap.csv", index=False)
    write_json(
        OUT / "evaluation_manifest.json",
        {
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "selection_sha256": sha256(selection_path),
            "test_rows": len(test),
            "bootstrap_repetitions": bootstrap_repetitions,
            "bootstrap_workers": workers,
            "warning": "2023 test was reused in prior Qwen exploration; this run removes cutoff overlap but is not a fresh confirmatory holdout",
        },
    )
    print(metrics.to_string(index=False), flush=True)
    print(bootstrap.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")

    extract = subparsers.add_parser("extract")
    extract.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    extract.add_argument("--split", choices=SPLITS, required=True)
    extract.add_argument("--model-path", required=True)
    extract.add_argument("--model-revision", default="")
    extract.add_argument("--batch-size", type=int, required=True)
    extract.add_argument("--max-length", type=int, default=512)
    extract.add_argument("--disk-floor-gib", type=float, default=40.0)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    select_parser.add_argument(
        "--ridge-device", choices=("auto", "cpu", "cuda"), default="auto"
    )

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    evaluate_parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "extract":
        if args.representation == "llama2":
            extract_llama(
                args.split,
                args.model_path,
                args.model_revision,
                args.batch_size,
                args.max_length,
                args.disk_floor_gib,
            )
        else:
            extract_bge(
                args.split,
                args.model_path,
                args.model_revision,
                args.batch_size,
                args.max_length,
                args.disk_floor_gib,
            )
    elif args.command == "select":
        select(args.workers, args.ridge_device)
    elif args.command == "evaluate":
        evaluate(args.bootstrap_repetitions, args.workers)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
