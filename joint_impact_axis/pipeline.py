#!/usr/bin/env python3
"""Joint news/8-K activation-axis pipeline with a frozen validation protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import t as t_dist
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.utils.extmath import randomized_svd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "joint_impact_axis_v1"
NEWS_SOURCE = ROOT / "outputs" / "fintexts_news_axis_e2e"
FILING_RECORDS = ROOT / "outputs" / "sec8k_activation_market" / "records.parquet"
FILING_PANEL = ROOT / "outputs" / "sec8k_eventtime_impact" / "exact_event_panel.parquet"
FILING_PRICES = (
    ROOT / "outputs" / "sec8k_eventtime_impact" / "prices_with_volume.parquet"
)

MODEL_SPECS = {
    "qwen25": {
        "path": Path(os.environ.get("QWEN25_MODEL", ROOT / "models/Qwen2.5-7B-Instruct")),
        "layers": 28,
        "hidden_size": 3584,
    },
    "qwen35": {
        "path": Path(os.environ.get("QWEN35_MODEL", ROOT / "models/Qwen3.5-4B")),
        "layers": 32,
        "hidden_size": 2560,
    },
    "llama31": {
        "path": Path(os.environ.get("LLAMA31_MODEL", ROOT / "models/Meta-Llama-3.1-8B-Instruct")),
        "layers": 32,
        "hidden_size": 4096,
    },
}

DOMAINS = ("news", "filing")
VERSIONS = ("news_only", "filing_only", "joint")
LABELS = ("z_absar1", "z_rv5", "z_range5", "impact_composite")
METHODS = (
    "mean_contrast",
    "logistic",
    "ridge",
    "within_ticker_ridge",
    "matched_delta_pc1",
)
EPS = 1e-10
SEED = 20260716
DEFAULT_DISK_FLOOR_GIB = 25.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def assert_disk_floor(path: Path, floor_gib: float) -> None:
    free = shutil.disk_usage(path).free
    floor = int(floor_gib * 1024**3)
    if free < floor:
        raise RuntimeError(
            f"disk floor reached: {free / 1024**3:.2f} GiB free < {floor_gib:.2f} GiB"
        )


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def normalize_vector(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-8)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.nanstd(x[valid]) == 0 or np.nanstd(y[valid]) == 0:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def cluster_meat(scores: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, int]:
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.zeros((len(unique), scores.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, scores)
    return sums.T @ sums, len(unique)


def two_way_cluster_beta(
    y: np.ndarray,
    x: np.ndarray,
    ticker: np.ndarray,
    date: np.ndarray,
) -> dict:
    x = (x - x.mean()) / max(float(x.std(ddof=0)), 1e-12)
    y = (y - y.mean()) / max(float(y.std(ddof=0)), 1e-12)
    design = np.column_stack([np.ones(len(x)), x])
    bread = np.linalg.pinv(design.T @ design)
    beta = bread @ design.T @ y
    residual = y - design @ beta
    scores = design * residual[:, None]
    meat_ticker, ticker_groups = cluster_meat(scores, ticker.astype(str))
    meat_date, date_groups = cluster_meat(scores, date.astype(str))
    intersection = np.char.add(
        np.char.add(ticker.astype(str), "|"), date.astype(str)
    )
    meat_intersection, intersection_groups = cluster_meat(scores, intersection)
    observations, parameters = design.shape

    def correction(groups: int) -> float:
        return groups / max(groups - 1, 1) * (observations - 1) / max(
            observations - parameters, 1
        )

    meat = (
        correction(ticker_groups) * meat_ticker
        + correction(date_groups) * meat_date
        - correction(intersection_groups) * meat_intersection
    )
    covariance = bread @ meat @ bread
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    statistic = float(beta[1] / max(standard_error, 1e-12))
    degrees = max(min(ticker_groups, date_groups) - 1, 1)
    pvalue = float(2 * (1 - t_dist.cdf(abs(statistic), df=degrees)))
    return {
        "standardized_beta": float(beta[1]),
        "two_way_cluster_se": standard_error,
        "two_way_cluster_t": statistic,
        "two_way_cluster_pvalue": pvalue,
        "ticker_clusters": ticker_groups,
        "date_clusters": date_groups,
    }


def group_demean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"value": values, "group": groups})
    return (frame["value"] - frame.groupby("group")["value"].transform("mean")).to_numpy()


def within_ticker_spearman(
    score: np.ndarray, label: np.ndarray, ticker: np.ndarray
) -> float:
    return safe_spearman(group_demean(score, ticker), group_demean(label, ticker))


def two_way_demean(
    values: np.ndarray, ticker: np.ndarray, date: np.ndarray, iterations: int = 20
) -> np.ndarray:
    residual = np.asarray(values, dtype=float).copy()
    residual -= np.nanmean(residual)
    for _ in range(iterations):
        residual = group_demean(residual, ticker)
        residual = group_demean(residual, date)
    return residual


def train_standardize(frame: pd.DataFrame, source: str, target: str) -> None:
    frame[target] = np.nan
    for domain in DOMAINS:
        train = frame["domain"].eq(domain) & frame["split"].eq("train")
        mean = float(frame.loc[train, source].mean())
        std = float(frame.loc[train, source].std(ddof=0))
        mask = frame["domain"].eq(domain)
        frame.loc[mask, target] = (frame.loc[mask, source] - mean) / max(std, 1e-8)


def filing_range5(panel: pd.DataFrame, prices: pd.DataFrame) -> np.ndarray:
    price_map = {
        ticker: part.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        for ticker, part in prices.groupby("ticker")
    }
    output = np.full(len(panel), np.nan, dtype=float)
    for row_index, row in enumerate(panel.itertuples(index=False)):
        if pd.isna(row.event_session):
            continue
        stock = price_map.get(row.ticker)
        if stock is None:
            continue
        dates = stock["date"].to_numpy(dtype="datetime64[ns]")
        position = int(
            np.searchsorted(dates, np.datetime64(pd.Timestamp(row.event_session)), side="left")
        )
        if position + 5 > len(stock):
            continue
        high = stock.loc[position : position + 4, "high"].to_numpy(float)
        low = stock.loc[position : position + 4, "low"].to_numpy(float)
        daily = np.log(high / low) ** 2 / (4.0 * math.log(2.0))
        output[row_index] = float(np.nanmean(daily))
    return output


def filing_cross_sectional_absar(panel: pd.DataFrame, prices: pd.DataFrame) -> np.ndarray:
    market = prices.loc[~prices["ticker"].eq("SPY"), ["ticker", "date", "close"]].copy()
    market["date"] = pd.to_datetime(market["date"])
    market = market.sort_values(["ticker", "date"])
    market["return"] = market.groupby("ticker")["close"].pct_change(fill_method=None)
    market_return = market.groupby("date")["return"].mean()
    event_dates = pd.to_datetime(panel["event_session"])
    benchmark = event_dates.map(market_return)
    abnormal = panel["event_return"].astype(float) - benchmark.to_numpy(float)
    return np.log(np.abs(abnormal) + 1e-6).to_numpy(float)


def prepare(out_dir: Path, disk_floor_gib: float) -> None:
    assert_disk_floor(out_dir.parent, disk_floor_gib)
    out_dir.mkdir(parents=True, exist_ok=True)

    news = pd.read_parquet(NEWS_SOURCE / "news_records.parquet").copy()
    news_frame = pd.DataFrame(
        {
            "domain": "news",
            "domain_item_id": news["item_id"].astype(np.int64),
            "ticker": news["ticker"].astype(str),
            "event_date": pd.to_datetime(news["date"]),
            "split": news["split"].astype(str),
            "text": news["news_text"].astype(str),
            "text_len_chars": news["text_len"].astype(int),
            "raw_absar1": news["log_abs_abret1"].astype(float),
            "raw_rv5": np.log(news["target_var_h5"].astype(float) + EPS),
            "raw_range5": np.log(news["target_range_h5"].astype(float) + EPS),
        }
    )

    filing_text = pd.read_parquet(FILING_RECORDS)[
        ["item_id", "accession", "target_text"]
    ].copy()
    filing = pd.read_parquet(FILING_PANEL).merge(
        filing_text, on=["item_id", "accession"], how="left", validate="one_to_one"
    )
    prices = pd.read_parquet(FILING_PRICES)
    filing_range = filing_range5(filing, prices)
    filing_absar = filing_cross_sectional_absar(filing, prices)
    filing_var = filing["rv5"].astype(float).pow(2) / 5.0
    filing_frame = pd.DataFrame(
        {
            "domain": "filing",
            "domain_item_id": filing["item_id"].astype(np.int64),
            "ticker": filing["ticker"].astype(str),
            "event_date": pd.to_datetime(filing["event_session"]),
            "split": filing["split"].astype(str),
            "text": filing["target_text"].fillna("").astype(str),
            "text_len_chars": filing["text_len"].astype(int),
            "raw_absar1": filing_absar,
            "raw_rv5": np.log(filing_var + EPS),
            "raw_range5": np.log(filing_range + EPS),
        }
    )

    documents = pd.concat([news_frame, filing_frame], ignore_index=True)
    documents = documents.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["event_date", "raw_absar1", "raw_rv5", "raw_range5", "text"]
    )
    documents = documents.loc[documents["text"].str.len().ge(80)].copy()
    documents = documents.sort_values(
        ["domain", "domain_item_id"], kind="stable"
    ).reset_index(drop=True)
    documents["global_id"] = np.arange(len(documents), dtype=np.int64)

    train_standardize(documents, "raw_absar1", "z_absar1")
    train_standardize(documents, "raw_rv5", "z_rv5")
    train_standardize(documents, "raw_range5", "z_range5")
    documents["impact_composite"] = documents[
        ["z_absar1", "z_rv5", "z_range5"]
    ].mean(axis=1)

    documents.to_parquet(out_dir / "documents.parquet", index=False)
    for domain in DOMAINS:
        subset = documents.loc[documents["domain"].eq(domain)].reset_index(drop=True)
        subset.to_parquet(out_dir / f"{domain}_documents.parquet", index=False)
        with (out_dir / f"{domain}_items.jsonl").open("w", encoding="utf-8") as handle:
            for row in subset.itertuples(index=False):
                handle.write(
                    json.dumps(
                        {
                            "row_id": int(row.Index) if hasattr(row, "Index") else None,
                            "global_id": int(row.global_id),
                            "domain_item_id": int(row.domain_item_id),
                            "text": row.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    audit = {
        "created_at": utc_now(),
        "protocol": "raw common labels; no ticker/date/length/item/earnings residualization",
        "labels": {
            "z_absar1": "train-domain standardized log absolute return minus same-date equal-weight market return",
            "z_rv5": "train-domain standardized log mean five-session close variance",
            "z_range5": "train-domain standardized log mean five-session Parkinson variance",
            "impact_composite": "equal mean of z_absar1, z_rv5, z_range5",
        },
        "news_timing": "text date t; response starts at t+1",
        "filing_timing": "SEC acceptance before 09:30 ET same session; otherwise next session",
        "counts": documents.groupby(["domain", "split"]).size().unstack(fill_value=0).to_dict("index"),
        "tickers": documents.groupby("domain")["ticker"].nunique().to_dict(),
        "date_ranges": {
            domain: {
                "min": str(part["event_date"].min().date()),
                "max": str(part["event_date"].max().date()),
            }
            for domain, part in documents.groupby("domain")
        },
        "input": "body text only; no ticker/date/item header",
        "source_text_limits": {
            "news": "FinTexTS target-company summary compacted to 3,000 characters upstream",
            "filing": "preferred 8-K/EX-99 disclosure compacted to 5,000 characters upstream",
        },
        "max_context_tokens": 2048,
        "disk_floor_gib": disk_floor_gib,
    }
    write_json(out_dir / "data_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def model_forward(model, input_ids, attention_mask):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
        "use_cache": False,
        "return_dict": True,
    }
    try:
        return model(**kwargs, logits_to_keep=1)
    except TypeError:
        try:
            return model(**kwargs, num_logits_to_keep=1)
        except TypeError:
            return model(**kwargs)


def chunk_token_ids(
    tokenizer, text: str, max_length: int, overlap: int
) -> list[tuple[list[int], int]]:
    ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    if len(ids) <= max_length:
        return [(ids, len(ids))]
    step = max_length - overlap
    chunks: list[tuple[list[int], int]] = []
    for start in range(0, len(ids), step):
        chunk = ids[start : start + max_length]
        if not chunk:
            break
        unique_weight = len(chunk) if start == 0 else max(len(chunk) - overlap, 1)
        chunks.append((chunk, unique_weight))
        if start + max_length >= len(ids):
            break
    return chunks


def extract_shard(
    out_dir: Path,
    model_key: str,
    domain: str,
    shard_id: int,
    num_shards: int,
    document_batch_size: int,
    chunk_batch_size: int,
    max_length: int,
    overlap: int,
    disk_floor_gib: float,
    limit: int | None,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODEL_SPECS[model_key]
    model_path = Path(spec["path"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    items = load_jsonl(out_dir / f"{domain}_items.jsonl")
    items = [item for index, item in enumerate(items) if index % num_shards == shard_id]
    if limit is not None:
        items = items[:limit]

    activation_dir = out_dir / "activations" / model_key
    activation_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{domain}_shard{shard_id}_of_{num_shards}"
    array_path = activation_dir / f"{stem}.npy"
    ids_path = activation_dir / f"{stem}_global_ids.npy"
    checkpoint_path = activation_dir / f"{stem}.checkpoint.json"
    done_path = activation_dir / f"{stem}.done.json"
    if done_path.exists() and array_path.exists() and ids_path.exists():
        print(f"{stem}: already complete", flush=True)
        return

    assert_disk_floor(out_dir, disk_floor_gib)
    if array_path.exists() and checkpoint_path.exists():
        output = np.lib.format.open_memmap(array_path, mode="r+")
        start_index = int(json.loads(checkpoint_path.read_text())["next_index"])
    else:
        output = np.lib.format.open_memmap(
            array_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(items), int(spec["layers"]), int(spec["hidden_size"])),
        )
        np.save(
            ids_path,
            np.asarray([int(item["global_id"]) for item in items], dtype=np.int64),
        )
        start_index = 0

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    ).eval()

    for doc_start in range(start_index, len(items), document_batch_size):
        assert_disk_floor(out_dir, disk_floor_gib)
        batch = items[doc_start : doc_start + document_batch_size]
        all_chunks: list[list[int]] = []
        owners: list[int] = []
        weights: list[int] = []
        for owner, item in enumerate(batch):
            chunks = chunk_token_ids(tokenizer, item["text"], max_length, overlap)
            for ids, weight in chunks:
                all_chunks.append(ids)
                owners.append(owner)
                weights.append(weight)

        accum = np.zeros(
            (len(batch), int(spec["layers"]), int(spec["hidden_size"])),
            dtype=np.float32,
        )
        total_weight = np.zeros(len(batch), dtype=np.float32)
        for chunk_start in range(0, len(all_chunks), chunk_batch_size):
            chunk_ids = all_chunks[chunk_start : chunk_start + chunk_batch_size]
            local_owners = owners[chunk_start : chunk_start + chunk_batch_size]
            local_weights = weights[chunk_start : chunk_start + chunk_batch_size]
            encoded = tokenizer.pad(
                {"input_ids": chunk_ids}, padding=True, return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to("cuda")
            attention_mask = encoded["attention_mask"].to("cuda")
            with torch.inference_mode():
                result = model_forward(model, input_ids, attention_mask)
            pooled_layers = []
            mask = attention_mask.unsqueeze(-1)
            denominator = mask.sum(dim=1).clamp_min(1)
            for hidden in result.hidden_states[1 : int(spec["layers"]) + 1]:
                pooled_layers.append((hidden * mask).sum(dim=1) / denominator)
            pooled = torch.stack(pooled_layers, dim=1).float().cpu().numpy()
            for local_index, owner in enumerate(local_owners):
                weight = float(local_weights[local_index])
                accum[owner] += pooled[local_index] * weight
                total_weight[owner] += weight
            del result, pooled, pooled_layers, input_ids, attention_mask, encoded

        accum /= np.maximum(total_weight[:, None, None], 1.0)
        doc_end = doc_start + len(batch)
        output[doc_start:doc_end] = accum.astype(np.float16)
        output.flush()
        write_json(
            checkpoint_path,
            {
                "next_index": doc_end,
                "total": len(items),
                "updated_at": utc_now(),
            },
        )
        if doc_end % 100 < document_batch_size or doc_end == len(items):
            print(f"{stem}: {doc_end}/{len(items)}", flush=True)

    write_json(
        done_path,
        {
            "completed_at": utc_now(),
            "items": len(items),
            "model": model_key,
            "domain": domain,
            "max_length": max_length,
            "overlap": overlap,
        },
    )


def merge_shards(
    out_dir: Path,
    model_key: str,
    domain: str,
    num_shards: int,
    disk_floor_gib: float,
    remove_shards: bool,
) -> None:
    spec = MODEL_SPECS[model_key]
    records = pd.read_parquet(out_dir / f"{domain}_documents.parquet")
    global_to_local = {
        int(global_id): index for index, global_id in enumerate(records["global_id"])
    }
    activation_dir = out_dir / "activations" / model_key
    destination = activation_dir / f"{domain}_layers.npy"
    manifest = activation_dir / f"{domain}_layers.manifest.json"
    if destination.exists() and manifest.exists():
        print(f"{destination}: already merged", flush=True)
        return
    assert_disk_floor(out_dir, disk_floor_gib)
    merged = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float16,
        shape=(len(records), int(spec["layers"]), int(spec["hidden_size"])),
    )
    for shard_id in range(num_shards):
        stem = f"{domain}_shard{shard_id}_of_{num_shards}"
        done = activation_dir / f"{stem}.done.json"
        if not done.exists():
            raise RuntimeError(f"incomplete shard: {done}")
        values = np.load(activation_dir / f"{stem}.npy", mmap_mode="r")
        global_ids = np.load(activation_dir / f"{stem}_global_ids.npy")
        local_ids = np.asarray([global_to_local[int(value)] for value in global_ids])
        merged[local_ids] = values
        merged.flush()
        assert_disk_floor(out_dir, disk_floor_gib)
        print(f"merged {stem}: {len(global_ids)}", flush=True)
    del merged
    write_json(
        manifest,
        {
            "created_at": utc_now(),
            "model": model_key,
            "domain": domain,
            "rows": len(records),
            "layers": int(spec["layers"]),
            "hidden_size": int(spec["hidden_size"]),
            "dtype": "float16",
            "max_context_tokens": 2048,
        },
    )
    if remove_shards:
        for shard_id in range(num_shards):
            stem = f"{domain}_shard{shard_id}_of_{num_shards}"
            for suffix in [
                ".npy",
                "_global_ids.npy",
                ".checkpoint.json",
            ]:
                (activation_dir / f"{stem}{suffix}").unlink(missing_ok=True)


def domain_weights(domains: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(domains), dtype=np.float32)
    for domain in np.unique(domains):
        mask = domains == domain
        weights[mask] = 0.5 / max(int(mask.sum()), 1)
    return weights * len(domains)


def matched_pairs(records: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    for domain in records["domain"].unique():
        domain_rows = np.where(records["domain"].to_numpy() == domain)[0]
        domain_labels = labels[domain_rows]
        low, high = np.nanquantile(domain_labels, [0.25, 0.75])
        high_rows = domain_rows[domain_labels >= high]
        low_rows = domain_rows[domain_labels <= low]
        ticker = records["ticker"].to_numpy()
        dates = pd.to_datetime(records["event_date"]).to_numpy(dtype="datetime64[D]").astype(np.int64)
        low_by_ticker = {
            value: low_rows[ticker[low_rows] == value]
            for value in np.unique(ticker[domain_rows])
        }
        for high_row in high_rows:
            candidates = low_by_ticker.get(ticker[high_row], np.empty(0, dtype=int))
            if not len(candidates):
                continue
            low_row = int(candidates[np.argmin(np.abs(dates[candidates] - dates[high_row]))])
            pairs.append((int(high_row), low_row))
    return np.asarray(pairs, dtype=np.int64)


def center_matrix_by_ticker(values: np.ndarray, records: pd.DataFrame) -> np.ndarray:
    centered = values.copy()
    keys = records["domain"].astype(str) + ":" + records["ticker"].astype(str)
    for key in keys.unique():
        mask = keys.eq(key).to_numpy()
        centered[mask] -= centered[mask].mean(axis=0, keepdims=True)
    return centered


def fit_axis(
    method: str,
    values: np.ndarray,
    records: pd.DataFrame,
    labels: np.ndarray,
) -> np.ndarray:
    domains = records["domain"].to_numpy()
    weights = domain_weights(domains) if len(np.unique(domains)) > 1 else None
    if method == "mean_contrast":
        high_means = []
        low_means = []
        for domain in np.unique(domains):
            mask = domains == domain
            low, high = np.nanquantile(labels[mask], [0.25, 0.75])
            high_means.append(values[mask & (labels >= high)].mean(axis=0))
            low_means.append(values[mask & (labels <= low)].mean(axis=0))
        axis = np.mean(high_means, axis=0) - np.mean(low_means, axis=0)
    elif method == "logistic":
        low_masks = []
        high_masks = []
        for domain in np.unique(domains):
            mask = domains == domain
            low, high = np.nanquantile(labels[mask], [0.25, 0.75])
            low_masks.append(mask & (labels <= low))
            high_masks.append(mask & (labels >= high))
        selected = np.logical_or(np.logical_or.reduce(low_masks), np.logical_or.reduce(high_masks))
        binary = np.logical_or.reduce(high_masks)[selected].astype(int)
        selected_weights = domain_weights(domains[selected]) if weights is not None else None
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-4,
            random_state=SEED,
        )
        model.fit(values[selected], binary, sample_weight=selected_weights)
        axis = model.coef_[0]
    elif method in {"ridge", "within_ticker_ridge"}:
        fit_values = values
        fit_labels = labels
        if method == "within_ticker_ridge":
            fit_values = center_matrix_by_ticker(values, records)
            keys = (records["domain"].astype(str) + ":" + records["ticker"].astype(str)).to_numpy()
            fit_labels = group_demean(labels, keys)
        model = Ridge(alpha=10.0, solver="lsqr", tol=1e-5)
        model.fit(fit_values, fit_labels, sample_weight=weights)
        axis = model.coef_
    elif method == "matched_delta_pc1":
        pairs = matched_pairs(records, labels)
        if len(pairs) < 2:
            raise RuntimeError("not enough matched pairs")
        deltas = values[pairs[:, 0]] - values[pairs[:, 1]]
        _, _, right = randomized_svd(
            deltas,
            n_components=1,
            n_iter=3,
            random_state=SEED,
        )
        axis = right[0]
        if float(axis @ deltas.mean(axis=0)) < 0:
            axis = -axis
    else:
        raise ValueError(method)
    axis = normalize_vector(axis)
    train_score = values @ axis
    correlations = []
    for domain in np.unique(domains):
        mask = domains == domain
        correlations.append(safe_spearman(train_score[mask], labels[mask]))
    if float(np.nanmean(correlations)) < 0:
        axis = -axis
    return axis


def load_domain_activations(out_dir: Path, model_key: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frames = []
    activations: dict[str, np.ndarray] = {}
    for domain in DOMAINS:
        frame = pd.read_parquet(out_dir / f"{domain}_documents.parquet").copy()
        frame["activation_row"] = np.arange(len(frame), dtype=np.int64)
        frames.append(frame)
        activations[domain] = np.load(
            out_dir / "activations" / model_key / f"{domain}_layers.npy",
            mmap_mode="r",
        )
    return pd.concat(frames, ignore_index=True), activations


def version_mask(records: pd.DataFrame, version: str, split: str) -> np.ndarray:
    mask = records["split"].eq(split).to_numpy()
    if version == "news_only":
        mask &= records["domain"].eq("news").to_numpy()
    elif version == "filing_only":
        mask &= records["domain"].eq("filing").to_numpy()
    elif version != "joint":
        raise ValueError(version)
    return mask


def layer_values(
    records: pd.DataFrame,
    activations: dict[str, np.ndarray],
    layer_index: int,
) -> np.ndarray:
    hidden = next(iter(activations.values())).shape[2]
    values = np.empty((len(records), hidden), dtype=np.float32)
    for domain in DOMAINS:
        mask = records["domain"].eq(domain).to_numpy()
        rows = records.loc[mask, "activation_row"].to_numpy(int)
        values[mask] = activations[domain][rows, layer_index].astype(np.float32)
    return normalize_rows(values)


def evaluate_axis(
    axis: np.ndarray,
    values: np.ndarray,
    records: pd.DataFrame,
    label: str,
    split: str,
) -> dict[str, float]:
    score = values @ axis
    output: dict[str, float] = {}
    for domain in DOMAINS:
        mask = records["split"].eq(split).to_numpy() & records["domain"].eq(domain).to_numpy()
        y = records.loc[mask, label].to_numpy(float)
        x = score[mask]
        tickers = records.loc[mask, "ticker"].to_numpy()
        output[f"{domain}_rho"] = safe_spearman(x, y)
        output[f"{domain}_within_ticker_rho"] = within_ticker_spearman(x, y, tickers)
    output["macro_rho"] = float(
        np.nanmean([output["news_rho"], output["filing_rho"]])
    )
    output["macro_within_ticker_rho"] = float(
        np.nanmean(
            [output["news_within_ticker_rho"], output["filing_within_ticker_rho"]]
        )
    )
    return output


def select(out_dir: Path, model_key: str, disk_floor_gib: float) -> None:
    model_dir = out_dir / "models" / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    if (model_dir / "test_results.csv").exists():
        raise RuntimeError("test already exists; refusing to change selection")
    records, activations = load_domain_activations(out_dir, model_key)
    spec = MODEL_SPECS[model_key]
    axes: list[np.ndarray] = []
    rows: list[dict] = []
    for layer_index in range(int(spec["layers"])):
        assert_disk_floor(out_dir, disk_floor_gib)
        values = layer_values(records, activations, layer_index)
        for label in LABELS:
            for version in VERSIONS:
                train_mask = version_mask(records, version, "train")
                train_records = records.loc[train_mask].reset_index(drop=True)
                train_labels = records.loc[train_mask, label].to_numpy(float)
                train_values = values[train_mask]
                for method in METHODS:
                    try:
                        axis = fit_axis(
                            method, train_values, train_records, train_labels
                        )
                    except Exception as error:
                        print(
                            f"skip {layer_index + 1} {label} {version} {method}: {error}",
                            flush=True,
                        )
                        continue
                    axis_index = len(axes)
                    axes.append(axis)
                    train_metrics = evaluate_axis(
                        axis, values, records, label, "train"
                    )
                    val_metrics = evaluate_axis(axis, values, records, label, "val")
                    rows.append(
                        {
                            "axis_index": axis_index,
                            "model": model_key,
                            "layer": layer_index + 1,
                            "label": label,
                            "version": version,
                            "method": method,
                            **{f"train_{key}": value for key, value in train_metrics.items()},
                            **{f"validation_{key}": value for key, value in val_metrics.items()},
                        }
                    )
        print(f"{model_key}: swept layer {layer_index + 1}/{spec['layers']}", flush=True)

    axes_array = np.stack(axes).astype(np.float32)
    np.save(model_dir / "candidate_axes.npy", axes_array)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(model_dir / "validation_sweep.csv", index=False)

    selected_rows = []
    for version in VERSIONS:
        candidates = sweep[sweep["version"].eq(version)].copy()
        if version == "news_only":
            candidates["selection_score"] = candidates["validation_news_rho"]
        elif version == "filing_only":
            candidates["selection_score"] = candidates["validation_filing_rho"]
        else:
            eligible = candidates[
                candidates["validation_news_rho"].gt(0)
                & candidates["validation_filing_rho"].gt(0)
            ]
            if len(eligible):
                candidates = eligible
            candidates["selection_score"] = candidates["validation_macro_rho"]
        selected_rows.append(
            candidates.sort_values(
                ["selection_score", "validation_macro_within_ticker_rho"],
                ascending=False,
            ).iloc[0]
        )
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected.to_csv(model_dir / "selected_axes.csv", index=False)
    joint = selected[selected["version"].eq("joint")].iloc[0]
    selected_axis = {
        row.version: axes_array[int(row.axis_index)]
        for row in selected.itertuples(index=False)
    }
    axis_cosines = {
        "news_vs_filing": float(selected_axis["news_only"] @ selected_axis["filing_only"]),
        "joint_vs_news": float(selected_axis["joint"] @ selected_axis["news_only"]),
        "joint_vs_filing": float(selected_axis["joint"] @ selected_axis["filing_only"]),
    }
    gate = bool(
        joint["validation_news_rho"] >= 0.05
        and joint["validation_filing_rho"] >= 0.05
        and joint["validation_macro_rho"] >= 0.075
    )
    frozen = {
        "created_at": utc_now(),
        "model": model_key,
        "selection_data": "domain-specific validation splits only",
        "joint_selection": "equal-domain macro Spearman; both domains must be positive",
        "qwen35_replication_gate": {
            "passed": gate,
            "criteria": "news rho >= .05, filing rho >= .05, macro rho >= .075",
        },
        "selected_axis_cosines": axis_cosines,
        "selected": selected.to_dict("records"),
        "test_opened": False,
    }
    write_json(model_dir / "frozen_axis_config.json", frozen)
    print(json.dumps(frozen, indent=2), flush=True)


def select_news_cpu(out_dir: Path, model_key: str, disk_floor_gib: float) -> None:
    model_dir = out_dir / "models" / model_key / "news_only_cpu"
    model_dir.mkdir(parents=True, exist_ok=True)
    if (model_dir / "test_results.csv").exists():
        raise RuntimeError("news test already exists; refusing to change selection")
    records = pd.read_parquet(out_dir / "news_documents.parquet").reset_index(drop=True)
    activations = np.load(
        out_dir / "activations" / model_key / "news_layers.npy", mmap_mode="r"
    )
    spec = MODEL_SPECS[model_key]
    train = records["split"].eq("train").to_numpy()
    val = records["split"].eq("val").to_numpy()
    train_records = records.loc[train].reset_index(drop=True)
    axes: list[np.ndarray] = []
    rows: list[dict] = []
    for layer_index in range(int(spec["layers"])):
        assert_disk_floor(out_dir, disk_floor_gib)
        values = normalize_rows(activations[:, layer_index].astype(np.float32))
        for label in LABELS:
            train_label = records.loc[train, label].to_numpy(float)
            val_label = records.loc[val, label].to_numpy(float)
            for method in METHODS:
                axis = fit_axis(
                    method,
                    values[train],
                    train_records,
                    train_label,
                )
                axis_index = len(axes)
                axes.append(axis)
                train_score = values[train] @ axis
                val_score = values[val] @ axis
                rows.append(
                    {
                        "axis_index": axis_index,
                        "model": model_key,
                        "layer": layer_index + 1,
                        "label": label,
                        "method": method,
                        "train_rho": safe_spearman(train_score, train_label),
                        "train_within_ticker_rho": within_ticker_spearman(
                            train_score,
                            train_label,
                            records.loc[train, "ticker"].to_numpy(),
                        ),
                        "validation_rho": safe_spearman(val_score, val_label),
                        "validation_within_ticker_rho": within_ticker_spearman(
                            val_score,
                            val_label,
                            records.loc[val, "ticker"].to_numpy(),
                        ),
                    }
                )
        print(
            f"{model_key} news-only CPU sweep layer {layer_index + 1}/{spec['layers']}",
            flush=True,
        )
    axes_array = np.stack(axes).astype(np.float32)
    np.save(model_dir / "candidate_axes.npy", axes_array)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(model_dir / "validation_sweep.csv", index=False)
    eligible = sweep[sweep["validation_rho"].gt(0)].copy()
    if not len(eligible):
        eligible = sweep.copy()
    selected = eligible.sort_values(
        ["validation_rho", "validation_within_ticker_rho"], ascending=False
    ).iloc[0]
    frozen = {
        "created_at": utc_now(),
        "model": model_key,
        "scope": "news_only_cpu_from_completed_2048_token_activations",
        "selection_data": "2022 validation only",
        "selection_rule": "highest positive raw validation Spearman; within-ticker rho tie-break",
        "selected": selected.to_dict(),
        "test_opened": False,
    }
    write_json(model_dir / "frozen_axis_config.json", frozen)
    print(json.dumps(frozen, indent=2), flush=True)


def open_news_test_cpu(out_dir: Path, model_key: str) -> None:
    model_dir = out_dir / "models" / model_key / "news_only_cpu"
    result_path = model_dir / "test_results.csv"
    if result_path.exists():
        raise RuntimeError("news test already opened")
    frozen = json.loads((model_dir / "frozen_axis_config.json").read_text())
    selected = frozen["selected"]
    records = pd.read_parquet(out_dir / "news_documents.parquet").reset_index(drop=True)
    activations = np.load(
        out_dir / "activations" / model_key / "news_layers.npy", mmap_mode="r"
    )
    axes = np.load(model_dir / "candidate_axes.npy", mmap_mode="r")
    values = normalize_rows(
        activations[:, int(selected["layer"]) - 1].astype(np.float32)
    )
    raw_score = values @ axes[int(selected["axis_index"])]
    score = calibrated_score(raw_score, records)
    test = records["split"].eq("test").to_numpy()
    x = score[test]
    ticker = records.loc[test, "ticker"].to_numpy()
    dates = records.loc[test, "event_date"].astype(str).to_numpy()
    rows = []
    for label in LABELS:
        y = records.loc[test, label].to_numpy(float)
        raw = spearmanr(x, y)
        rows.append(
            {
                "training_label": selected["label"],
                "method": selected["method"],
                "layer": int(selected["layer"]),
                "outcome": label,
                "n": int(test.sum()),
                "raw_spearman": float(raw.statistic),
                "raw_spearman_pvalue": float(raw.pvalue),
                "within_ticker_spearman": within_ticker_spearman(x, y, ticker),
                "ticker_date_spearman": safe_spearman(
                    two_way_demean(x, ticker, dates),
                    two_way_demean(y, ticker, dates),
                ),
                **two_way_cluster_beta(y, x, ticker, dates),
                **company_bootstrap_rho(x, y, ticker),
            }
        )
    pd.DataFrame(rows).to_csv(result_path, index=False)
    score_frame = records.drop(columns=["text"]).copy()
    score_frame["news_score"] = score
    score_frame.to_parquet(model_dir / "scores.parquet", index=False)
    sample = pd.DataFrame(
        {
            "score": x,
            "outcome": records.loc[test, selected["label"]].to_numpy(float),
        }
    )
    sample["decile"] = pd.qcut(sample["score"], 10, labels=False, duplicates="drop")
    sample.groupby("decile", as_index=False).agg(
        n=("score", "size"),
        mean_score=("score", "mean"),
        mean_outcome=("outcome", "mean"),
    ).to_csv(model_dir / "deciles.csv", index=False)
    frozen["test_opened"] = True
    frozen["test_opened_at"] = utc_now()
    write_json(model_dir / "frozen_axis_config.json", frozen)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def calibrated_score(
    score: np.ndarray, records: pd.DataFrame
) -> np.ndarray:
    result = np.full(len(score), np.nan, dtype=float)
    for domain in DOMAINS:
        train = records["domain"].eq(domain).to_numpy() & records["split"].eq("train").to_numpy()
        mask = records["domain"].eq(domain).to_numpy()
        mean = float(np.mean(score[train]))
        std = float(np.std(score[train], ddof=0))
        result[mask] = (score[mask] - mean) / max(std, 1e-8)
    return result


def company_bootstrap_rho(
    score: np.ndarray,
    label: np.ndarray,
    ticker: np.ndarray,
    repetitions: int = 500,
) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    companies = np.unique(ticker)
    by_company = {company: np.where(ticker == company)[0] for company in companies}
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(companies, size=len(companies), replace=True)
        indices = np.concatenate([by_company[company] for company in sampled])
        values.append(safe_spearman(score[indices], label[indices]))
    values = np.asarray(values, dtype=float)
    return {
        "company_bootstrap_ci_low": float(np.nanquantile(values, 0.025)),
        "company_bootstrap_ci_high": float(np.nanquantile(values, 0.975)),
        "company_bootstrap_positive_fraction": float(np.nanmean(values > 0)),
    }


def open_test(out_dir: Path, model_key: str) -> None:
    model_dir = out_dir / "models" / model_key
    result_path = model_dir / "test_results.csv"
    if result_path.exists():
        raise RuntimeError("test already opened")
    frozen = json.loads((model_dir / "frozen_axis_config.json").read_text())
    selected = pd.DataFrame(frozen["selected"])
    axes = np.load(model_dir / "candidate_axes.npy", mmap_mode="r")
    records, activations = load_domain_activations(out_dir, model_key)
    score_frame = records[
        ["global_id", "domain", "domain_item_id", "ticker", "event_date", "split", *LABELS]
    ].copy()
    rows: list[dict] = []
    deciles: list[dict] = []
    for selected_row in selected.itertuples(index=False):
        values = layer_values(records, activations, int(selected_row.layer) - 1)
        axis = axes[int(selected_row.axis_index)]
        raw_score = values @ axis
        score = calibrated_score(raw_score, records)
        score_name = f"{selected_row.version}_score"
        score_frame[score_name] = score
        for domain in DOMAINS:
            mask = records["domain"].eq(domain).to_numpy() & records["split"].eq("test").to_numpy()
            for label in LABELS:
                x = score[mask]
                y = records.loc[mask, label].to_numpy(float)
                ticker = records.loc[mask, "ticker"].to_numpy()
                dates = records.loc[mask, "event_date"].astype(str).to_numpy()
                raw = spearmanr(x, y)
                clustered = two_way_cluster_beta(y, x, ticker, dates)
                bootstrap = company_bootstrap_rho(x, y, ticker)
                rows.append(
                    {
                        "version": selected_row.version,
                        "training_label": selected_row.label,
                        "method": selected_row.method,
                        "layer": int(selected_row.layer),
                        "test_domain": domain,
                        "outcome": label,
                        "n": int(mask.sum()),
                        "raw_spearman": float(raw.statistic),
                        "raw_spearman_pvalue": float(raw.pvalue),
                        "within_ticker_spearman": within_ticker_spearman(x, y, ticker),
                        "ticker_date_spearman": safe_spearman(
                            two_way_demean(x, ticker, dates),
                            two_way_demean(y, ticker, dates),
                        ),
                        **clustered,
                        **bootstrap,
                    }
                )
            if selected_row.version == "joint":
                sample = pd.DataFrame(
                    {
                        "score": score[mask],
                        "outcome": records.loc[mask, selected_row.label].to_numpy(float),
                    }
                )
                sample["decile"] = pd.qcut(
                    sample["score"], 10, labels=False, duplicates="drop"
                )
                for decile, group in sample.groupby("decile"):
                    deciles.append(
                        {
                            "domain": domain,
                            "label": selected_row.label,
                            "decile": int(decile) + 1,
                            "n": len(group),
                            "mean_score": float(group["score"].mean()),
                            "mean_outcome": float(group["outcome"].mean()),
                        }
                    )
    score_frame.to_parquet(model_dir / "scores.parquet", index=False)
    pd.DataFrame(rows).to_csv(result_path, index=False)
    pd.DataFrame(deciles).to_csv(model_dir / "joint_deciles.csv", index=False)
    frozen["test_opened"] = True
    frozen["test_opened_at"] = utc_now()
    write_json(model_dir / "frozen_axis_config.json", frozen)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "prepare",
            "extract-shard",
            "merge",
            "select",
            "test",
            "select-news-cpu",
            "test-news-cpu",
        ],
    )
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--model", choices=MODEL_SPECS, default="qwen25")
    parser.add_argument("--domain", choices=DOMAINS, default="news")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--document-batch-size", type=int, default=4)
    parser.add_argument("--chunk-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--disk-floor-gib", type=float, default=DEFAULT_DISK_FLOOR_GIB)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--keep-shards", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare(args.out_dir, args.disk_floor_gib)
    elif args.stage == "extract-shard":
        extract_shard(
            args.out_dir,
            args.model,
            args.domain,
            args.shard_id,
            args.num_shards,
            args.document_batch_size,
            args.chunk_batch_size,
            args.max_length,
            args.overlap,
            args.disk_floor_gib,
            args.limit,
        )
    elif args.stage == "merge":
        merge_shards(
            args.out_dir,
            args.model,
            args.domain,
            args.num_shards,
            args.disk_floor_gib,
            remove_shards=not args.keep_shards,
        )
    elif args.stage == "select":
        select(args.out_dir, args.model, args.disk_floor_gib)
    elif args.stage == "test":
        open_test(args.out_dir, args.model)
    elif args.stage == "select-news-cpu":
        select_news_cpu(args.out_dir, args.model, args.disk_floor_gib)
    elif args.stage == "test-news-cpu":
        open_news_test_cpu(args.out_dir, args.model)


if __name__ == "__main__":
    main()
