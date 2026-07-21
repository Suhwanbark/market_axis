#!/usr/bin/env python3
"""End-to-end, time-split market-impact axes from FinTexTS company news."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.extmath import randomized_svd

OUT = Path("outputs/fintexts_news_axis_e2e")
MODEL_PATH = os.environ.get("QWEN25_MODEL", "Qwen/Qwen2.5-7B-Instruct")
REPO_ID = "EXAONE-BI/FinTexTS"
PARQUET_FILES = [
    "data/train-00000-of-00004.parquet",
    "data/train-00001-of-00004.parquet",
    "data/train-00002-of-00004.parquet",
    "data/train-00003-of-00004.parquet",
]
BASE_COLUMNS = ["date", "ticker", "open", "high", "low", "close"]
LAYERS = list(range(1, 29))
HIDDEN_SIZE = 3584
TEXT_COLUMNS = [
    "targetCompany_category1",
    "targetCompany_category2",
    "targetCompany_category3",
]
EVENT_PATTERNS = {
    "event_earnings": r"earnings|quarterly results|revenue|eps|guidance|fiscal quarter",
    "event_analyst": r"analyst|price target|upgrade|downgrade|rating|consensus estimate",
    "event_mna": r"acqui(?:re|red|sition)|merger|takeover|divest|stake in|definitive agreement",
    "event_legal": r"lawsuit|court|regulat|investigat|settlement|antitrust|department of justice",
    "event_product": r"launch|introduc|product|partnership|contract|approval|clinical trial|fda",
    "event_financing": r"dividend|buyback|repurchase|offering|debt|bond|capital return",
    "event_price_recap": r"stock (?:rose|fell|jumped|dropped|gained|declined)|shares (?:rose|fell|jumped|dropped|gained|declined|up|down)|pre-market|52.week",
}
EVENT_COLUMNS = list(EVENT_PATTERNS)
NUMERIC_CONTROLS = [
    "log_hist_var_d",
    "log_hist_var_w",
    "log_hist_var_m",
    "log_hist_range_d",
    "log_hist_range_w",
    "log_hist_range_m",
    "current_abs_return",
    "current_negative_return",
    "momentum20",
    "market_log_var_d",
    "market_log_var_w",
    "market_log_var_m",
    "text_len",
] + EVENT_COLUMNS
IMPACT_COMPONENTS = [
    "log_abs_abret1",
    "market_adjusted_log_rv5",
    "market_adjusted_log_range5",
]
METHODS = ["mean_contrast", "matched_delta_pc1", "ridge", "within_ticker_ridge"]
EPS = 1e-10
SEED = 20260716
OHLC_COLUMNS = ["open", "high", "low", "close"]


def compact_text(parts: list[str | None], max_chars: int) -> str:
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not isinstance(part, str):
            continue
        text = " ".join(part.split())
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return "\n".join(cleaned)[:max_chars]


def load_fintexts(out_dir: Path) -> pd.DataFrame:
    paths = [
        hf_hub_download(
            REPO_ID,
            filename=filename,
            repo_type="dataset",
            cache_dir=str(out_dir / "hf_cache"),
        )
        for filename in PARQUET_FILES
    ]
    frames = [
        pd.read_parquet(path, columns=BASE_COLUMNS + TEXT_COLUMNS) for path in paths
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def encode_items(tokenizer, items: list[dict], max_length: int):
    encoded: list[list[int]] = []
    spans: list[tuple[int, int]] = []
    lengths: list[int] = []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    for item in items:
        prefix_ids = tokenizer(item["prefix"], add_special_tokens=False).input_ids
        text_ids = tokenizer(item["text"], add_special_tokens=False).input_ids
        suffix_ids = tokenizer(item["suffix"], add_special_tokens=False).input_ids
        budget = max_length - len(prefix_ids) - len(suffix_ids)
        if budget < 16:
            prefix_ids = prefix_ids[: max(16, max_length // 2)]
            budget = max_length - len(prefix_ids) - len(suffix_ids)
        text_ids = text_ids[: max(8, budget)]
        ids = (prefix_ids + text_ids + suffix_ids)[:max_length]
        span_start = min(len(prefix_ids), len(ids) - 1)
        span_end = min(len(prefix_ids) + len(text_ids), len(ids))
        if span_end <= span_start:
            span_start = max(0, len(ids) - 2)
            span_end = len(ids)
        encoded.append(ids)
        spans.append((span_start, span_end))
        lengths.append(len(ids))

    max_len = max(lengths)
    input_ids = np.full((len(items), max_len), pad_id, dtype=np.int64)
    attention_mask = np.zeros((len(items), max_len), dtype=np.int64)
    for index, ids in enumerate(encoded):
        input_ids[index, : len(ids)] = ids
        attention_mask[index, : len(ids)] = 1
    return input_ids, attention_mask, spans, lengths


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def assign_split(date: pd.Timestamp) -> str:
    if date.year <= 2021:
        return "train"
    if date.year == 2022:
        return "val"
    return "test"


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def normalize_vector(value: np.ndarray) -> np.ndarray:
    return value / max(float(np.linalg.norm(value)), 1e-8)


def within_ticker_spearman(
    score: np.ndarray, labels: np.ndarray, tickers: np.ndarray
) -> float:
    frame = pd.DataFrame({"score": score, "label": labels, "ticker": tickers})
    frame["score"] -= frame.groupby("ticker")["score"].transform("mean")
    frame["label"] -= frame.groupby("ticker")["label"].transform("mean")
    return float(spearmanr(frame["score"], frame["label"]).statistic)


def future_mean(series: pd.Series, horizon: int) -> pd.Series:
    values = [series.shift(-offset) for offset in range(1, horizon + 1)]
    return pd.concat(values, axis=1).mean(axis=1)


def carried_market_closure_dates(
    raw: pd.DataFrame,
    minimum_tickers: int = 20,
    minimum_share: float = 0.95,
) -> pd.DataFrame:
    """Find synthetic holiday rows whose OHLC was copied from the prior session."""
    ordered = raw.sort_values(["ticker", "date"]).copy()
    previous = ordered.groupby("ticker", sort=False)[OHLC_COLUMNS].shift(1)
    comparable = ordered[OHLC_COLUMNS].notna().all(axis=1) & previous.notna().all(axis=1)
    unchanged = comparable.copy()
    for column in OHLC_COLUMNS:
        unchanged &= np.isclose(
            ordered[column].to_numpy(float),
            previous[column].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        )

    audit = pd.DataFrame(
        {
            "date": ordered["date"].to_numpy(),
            "comparable": comparable.to_numpy(dtype=np.int8),
            "unchanged": unchanged.to_numpy(dtype=np.int8),
        }
    )
    by_date = audit.groupby("date", as_index=False).agg(
        comparable_tickers=("comparable", "sum"),
        unchanged_tickers=("unchanged", "sum"),
    )
    by_date["unchanged_share"] = (
        by_date["unchanged_tickers"]
        / by_date["comparable_tickers"].clip(lower=1)
    )
    return by_date.loc[
        by_date["comparable_tickers"].ge(minimum_tickers)
        & by_date["unchanged_share"].ge(minimum_share)
    ].reset_index(drop=True)


def add_market_features(raw: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in raw.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        part = group.copy().reset_index(drop=True)
        ret = part["close"].pct_change(fill_method=None)
        squared = ret.pow(2)
        log_range = np.log(part["high"] / part["low"]).replace([np.inf, -np.inf], np.nan)
        parkinson = log_range.pow(2) / (4.0 * math.log(2.0))

        part["ret0"] = ret
        part["hist_var_d"] = squared
        part["hist_var_w"] = squared.rolling(5, min_periods=3).mean()
        part["hist_var_m"] = squared.rolling(22, min_periods=10).mean()
        part["hist_range_d"] = parkinson
        part["hist_range_w"] = parkinson.rolling(5, min_periods=3).mean()
        part["hist_range_m"] = parkinson.rolling(22, min_periods=10).mean()
        part["current_abs_return"] = ret.abs()
        part["current_negative_return"] = (-ret).clip(lower=0)
        part["momentum20"] = part["close"] / part["close"].shift(20) - 1.0
        part["fwd_ret1"] = ret.shift(-1)
        part["target_var_h1"] = squared.shift(-1)
        part["target_var_h5"] = future_mean(squared, 5)
        part["target_range_h1"] = parkinson.shift(-1)
        part["target_range_h5"] = future_mean(parkinson, 5)
        frames.append(part)

    panel = pd.concat(frames, ignore_index=True)
    panel["market_fwd_ret1"] = panel.groupby("date")["fwd_ret1"].transform("mean")
    panel["abret1"] = panel["fwd_ret1"] - panel["market_fwd_ret1"]
    panel["log_abs_abret1"] = np.log(panel["abret1"].abs() + 1e-6)
    panel["log_rv5"] = np.log(panel["target_var_h5"] + EPS)
    panel["log_range5"] = np.log(panel["target_range_h5"] + EPS)
    panel["market_adjusted_log_rv5"] = panel["log_rv5"] - panel.groupby("date")["log_rv5"].transform("mean")
    panel["market_adjusted_log_range5"] = panel["log_range5"] - panel.groupby("date")["log_range5"].transform("mean")
    for source, target in [
        ("hist_var_d", "log_hist_var_d"),
        ("hist_var_w", "log_hist_var_w"),
        ("hist_var_m", "log_hist_var_m"),
        ("hist_range_d", "log_hist_range_d"),
        ("hist_range_w", "log_hist_range_w"),
        ("hist_range_m", "log_hist_range_m"),
    ]:
        panel[target] = np.log(panel[source] + EPS)
    panel["market_log_var_d"] = panel.groupby("date")["log_hist_var_d"].transform("mean")
    panel["market_log_var_w"] = panel.groupby("date")["log_hist_var_w"].transform("mean")
    panel["market_log_var_m"] = panel.groupby("date")["log_hist_var_m"].transform("mean")
    panel["split"] = panel["date"].map(assign_split)
    return panel


def near_duplicate_flags(frame: pd.DataFrame, threshold: float = 0.92, days: int = 7) -> np.ndarray:
    duplicate = np.zeros(len(frame), dtype=bool)
    for _, indices in frame.sort_values(["ticker", "date"]).groupby("ticker").indices.items():
        ordered = np.asarray(indices)
        history: list[tuple[pd.Timestamp, set[str]]] = []
        for index in ordered:
            date = pd.Timestamp(frame.at[index, "date"])
            tokens = set(re.findall(r"[a-z0-9]+", frame.at[index, "normalized_text"]))
            history = [(d, t) for d, t in history if (date - d).days <= days]
            for _, prior in history:
                union = len(tokens | prior)
                similarity = len(tokens & prior) / max(union, 1)
                if similarity >= threshold:
                    duplicate[index] = True
                    break
            if not duplicate[index]:
                history.append((date, tokens))
    return duplicate


def nuisance_pipeline() -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    prep = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_CONTROLS),
            ("categorical", one_hot_encoder(), ["ticker", "event_month"]),
        ]
    )
    return Pipeline([("prep", prep), ("ridge", Ridge(alpha=10.0, solver="lsqr"))])


def prepare(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_fintexts(out_dir)
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    raw_rows = len(raw)
    closure_audit = carried_market_closure_dates(raw)
    closure_dates = set(pd.to_datetime(closure_audit["date"]))
    carried_rows = int(raw["date"].isin(closure_dates).sum())
    raw = raw.loc[~raw["date"].isin(closure_dates)].copy()
    raw["news_text"] = raw[TEXT_COLUMNS].apply(
        lambda row: compact_text(row.tolist(), max_chars=3000), axis=1
    )
    panel = add_market_features(raw)

    news = panel.loc[panel["news_text"].str.len().ge(80)].copy()
    news["normalized_text"] = (
        news["news_text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    news["text_hash"] = news["normalized_text"].map(
        lambda text: hashlib.sha256(text.encode()).hexdigest()
    )
    news["near_duplicate"] = near_duplicate_flags(news.reset_index(drop=True))
    near_duplicate_count = int(news["near_duplicate"].sum())
    news = news.loc[~news["near_duplicate"]].copy()
    news["text_len"] = news["news_text"].str.len()
    lowered = news["news_text"].str.lower()
    for column, pattern in EVENT_PATTERNS.items():
        news[column] = lowered.str.contains(pattern, regex=True).astype(int)
    # Month-of-year transfers across chronological splits; year-month dummies do not.
    news["event_month"] = news["date"].dt.month.astype(str).str.zfill(2)

    required = NUMERIC_CONTROLS + IMPACT_COMPONENTS + ["target_var_h1", "target_var_h5"]
    news = news.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    news = news.sort_values(["date", "ticker"]).reset_index(drop=True)
    news["item_id"] = np.arange(len(news), dtype=np.int64)

    train = news["split"].eq("train")
    for component in IMPACT_COMPONENTS:
        model = nuisance_pipeline()
        model.fit(news.loc[train], news.loc[train, component])
        residual = news[component].to_numpy(float) - model.predict(news)
        scale = float(np.std(residual[train], ddof=0))
        news[f"resid_{component}"] = residual / max(scale, 1e-8)
    residual_columns = [f"resid_{column}" for column in IMPACT_COMPONENTS]
    news["impact_composite"] = news[residual_columns].mean(axis=1)
    cutoff = float(news.loc[train, "impact_composite"].quantile(0.75))
    news["high_impact"] = news["impact_composite"].ge(cutoff).astype(int)

    record_columns = [
        "item_id",
        "ticker",
        "date",
        "split",
        "news_text",
        "text_hash",
        "text_len",
        "event_month",
        "impact_composite",
        "high_impact",
    ] + EVENT_COLUMNS + NUMERIC_CONTROLS[:12] + IMPACT_COMPONENTS + residual_columns + [
        "target_var_h1",
        "target_var_h5",
        "target_range_h1",
        "target_range_h5",
    ]
    record_columns = list(dict.fromkeys(record_columns))
    news[record_columns].to_parquet(out_dir / "news_records.parquet", index=False)

    daily_columns = [
        "ticker",
        "date",
        "split",
        "ret0",
        "target_var_h1",
        "target_var_h5",
        "target_range_h1",
        "target_range_h5",
        "log_hist_var_d",
        "log_hist_var_w",
        "log_hist_var_m",
        "log_hist_range_d",
        "log_hist_range_w",
        "log_hist_range_m",
        "current_abs_return",
        "current_negative_return",
        "momentum20",
        "market_log_var_d",
        "market_log_var_w",
        "market_log_var_m",
    ]
    panel[daily_columns].replace([np.inf, -np.inf], np.nan).dropna().to_parquet(
        out_dir / "daily_panel.parquet", index=False
    )

    with (out_dir / "news_items.jsonl").open("w") as handle:
        for row in news[["item_id", "ticker", "date", "split", "news_text"]].to_dict("records"):
            item = {
                "item_id": int(row["item_id"]),
                "ticker": row["ticker"],
                "date": str(pd.Timestamp(row["date"]).date()),
                "split": row["split"],
                "prefix": "Financial news:\n",
                "text": row["news_text"],
                "suffix": "\n",
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    hashes = news.groupby("text_hash")["split"].nunique()
    summary = {
        "source": "EXAONE-BI/FinTexTS targetCompany_category1-3",
        "document_unit": "ticker-day target-company news summary",
        "raw_rows": int(raw_rows),
        "carried_market_closure_dates_removed": int(len(closure_dates)),
        "carried_market_closure_rows_removed": carried_rows,
        "carried_market_closure_dates": [
            str(pd.Timestamp(date).date()) for date in sorted(closure_dates)
        ],
        "market_target_policy": (
            "remove cross-sectionally carried OHLC closure rows before computing "
            "returns; news on those non-trading dates is excluded from the primary sample"
        ),
        "rows_before_news_filter": int(len(panel)),
        "near_duplicates_removed": near_duplicate_count,
        "news_rows": int(len(news)),
        "news_by_split": news["split"].value_counts().to_dict(),
        "tickers": int(news["ticker"].nunique()),
        "exact_hashes_crossing_splits": int(hashes.gt(1).sum()),
        "ticker_date_duplicates": int(news.duplicated(["ticker", "date"], False).sum()),
        "date_min": str(news["date"].min().date()),
        "date_max": str(news["date"].max().date()),
        "protocol": "2019-2021 train, 2022 validation, 2023 untouched test",
        "prompt": "pure news text; no ticker/date header",
    }
    (out_dir / "data_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def load_items(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_shard(
    out_dir: Path,
    shard_id: int,
    num_shards: int,
    batch_size: int,
    max_length: int,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    all_items = load_items(out_dir / "news_items.jsonl")
    items = [item for item in all_items if int(item["item_id"]) % num_shards == shard_id]
    base = out_dir / f"news_all_layers_shard{shard_id}"
    array_path = base.with_suffix(".npy")
    ids_path = out_dir / f"news_all_layers_shard{shard_id}_ids.npy"
    checkpoint_path = out_dir / f"news_all_layers_shard{shard_id}.checkpoint.json"
    done_path = out_dir / f"news_all_layers_shard{shard_id}.done.json"
    if done_path.exists() and array_path.exists() and ids_path.exists():
        print(f"shard {shard_id}: already complete", flush=True)
        return

    start = 0
    if array_path.exists() and checkpoint_path.exists():
        output = np.lib.format.open_memmap(array_path, mode="r+")
        start = int(json.loads(checkpoint_path.read_text()).get("next_index", 0))
    else:
        output = np.lib.format.open_memmap(
            array_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(items), len(LAYERS), HIDDEN_SIZE),
        )
        np.save(ids_path, np.asarray([int(item["item_id"]) for item in items], dtype=np.int64))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    ).eval()

    for batch_start in range(start, len(items), batch_size):
        batch = items[batch_start : batch_start + batch_size]
        input_ids, attention_mask, spans, _ = encode_items(tokenizer, batch, max_length)
        ids_tensor = torch.tensor(input_ids, device="cuda")
        mask_tensor = torch.tensor(attention_mask, device="cuda")
        with torch.no_grad():
            result = model(
                input_ids=ids_tensor,
                attention_mask=mask_tensor,
                output_hidden_states=True,
                use_cache=False,
            )
        layer_rows = []
        for layer in LAYERS:
            hidden = result.hidden_states[layer]
            pooled = [hidden[index, start_:end].mean(dim=0) for index, (start_, end) in enumerate(spans)]
            layer_rows.append(torch.stack(pooled))
        activations = (
            torch.stack(layer_rows, dim=1).float().cpu().numpy().astype(np.float16)
        )
        batch_end = batch_start + len(batch)
        output[batch_start:batch_end] = activations
        output.flush()
        checkpoint_path.write_text(
            json.dumps({"next_index": batch_end, "total": len(items)}), encoding="utf-8"
        )
        del result, activations, ids_tensor, mask_tensor, layer_rows
        if batch_end % 200 < batch_size or batch_end == len(items):
            print(f"shard {shard_id}: {batch_end}/{len(items)}", flush=True)
    done_path.write_text(
        json.dumps({"items": len(items), "finished_at": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )


def merge_shards(out_dir: Path, num_shards: int, remove_shards: bool) -> None:
    records = pd.read_parquet(out_dir / "news_records.parquet", columns=["item_id"])
    destination = out_dir / "news_layers1-28.npy"
    merged = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float16,
        shape=(len(records), len(LAYERS), HIDDEN_SIZE),
    )
    for shard_id in range(num_shards):
        base = out_dir / f"news_all_layers_shard{shard_id}"
        array_path = base.with_suffix(".npy")
        ids_path = out_dir / f"news_all_layers_shard{shard_id}_ids.npy"
        done_path = out_dir / f"news_all_layers_shard{shard_id}.done.json"
        if not done_path.exists():
            raise RuntimeError(f"shard {shard_id} is not complete")
        values = np.load(array_path, mmap_mode="r")
        ids = np.load(ids_path)
        merged[ids] = values
        merged.flush()
        print(f"merged shard {shard_id}: {len(ids)} rows", flush=True)
    del merged
    if remove_shards:
        for shard_id in range(num_shards):
            for path in [
                out_dir / f"news_all_layers_shard{shard_id}.npy",
                out_dir / f"news_all_layers_shard{shard_id}_ids.npy",
                out_dir / f"news_all_layers_shard{shard_id}.checkpoint.json",
            ]:
                path.unlink(missing_ok=True)
    print(destination, flush=True)


def ticker_center(values: np.ndarray, tickers: np.ndarray) -> np.ndarray:
    centered = values.copy()
    for ticker in np.unique(tickers):
        mask = tickers == ticker
        centered[mask] -= centered[mask].mean(axis=0, keepdims=True)
    return centered


def matched_pair_indices(
    records: pd.DataFrame,
    labels: np.ndarray,
) -> np.ndarray:
    low_cut, high_cut = np.quantile(labels, [0.25, 0.75])
    high_rows = np.where(labels >= high_cut)[0]
    low_rows = np.where(labels <= low_cut)[0]
    tickers = records["ticker"].to_numpy()
    dates = pd.to_datetime(records["date"]).to_numpy(dtype="datetime64[D]").astype(np.int64)
    event_values = records[EVENT_COLUMNS].to_numpy(float)
    historical = records["log_hist_var_m"].to_numpy(float)
    low_by_ticker = {
        ticker: low_rows[tickers[low_rows] == ticker] for ticker in np.unique(tickers)
    }
    pairs: list[tuple[int, int]] = []
    for high in high_rows:
        candidates = low_by_ticker[tickers[high]]
        if not len(candidates):
            continue
        day_distance = np.abs(dates[candidates] - dates[high]) / 365.0
        event_distance = np.mean(np.abs(event_values[candidates] - event_values[high]), axis=1)
        market_distance = np.abs(historical[candidates] - historical[high])
        distance = day_distance + event_distance + 0.25 * market_distance
        low = int(candidates[np.argmin(distance)])
        pairs.append((int(high), low))
    return np.asarray(pairs, dtype=np.int64)


def matched_deltas(
    values: np.ndarray,
    records: pd.DataFrame,
    labels: np.ndarray,
    pair_indices: np.ndarray | None = None,
) -> np.ndarray:
    pairs = matched_pair_indices(records, labels) if pair_indices is None else pair_indices
    return (values[pairs[:, 0]] - values[pairs[:, 1]]).astype(np.float32)


def fit_axis(
    method: str,
    values: np.ndarray,
    records: pd.DataFrame,
    labels: np.ndarray,
    pair_indices: np.ndarray | None = None,
) -> np.ndarray:
    fit_values = values
    if method == "mean_contrast":
        low, high = np.quantile(labels, [0.25, 0.75])
        axis = values[labels >= high].mean(axis=0) - values[labels <= low].mean(axis=0)
    elif method == "matched_delta_pc1":
        deltas = matched_deltas(values, records, labels, pair_indices)
        sign = deltas.mean(axis=0)
        _, _, right = randomized_svd(
            deltas,
            n_components=1,
            n_iter=3,
            random_state=SEED,
        )
        axis = right[0]
        if float(axis @ sign) < 0:
            axis = -axis
    elif method in {"ridge", "within_ticker_ridge"}:
        if method == "within_ticker_ridge":
            fit_values = ticker_center(values, records["ticker"].to_numpy())
        model = Ridge(alpha=10.0, solver="lsqr", tol=1e-5)
        model.fit(fit_values, labels)
        axis = model.coef_
    else:
        raise ValueError(method)
    axis = normalize_vector(np.asarray(axis, dtype=np.float32))
    orientation_values = fit_values if method == "within_ticker_ridge" else values
    if within_ticker_spearman(
        orientation_values @ axis,
        labels,
        records["ticker"].to_numpy(),
    ) < 0:
        axis = -axis
    return axis


def select_axes(out_dir: Path) -> None:
    if (out_dir / "test_results.csv").exists():
        raise RuntimeError("test has already been evaluated; preserve the audit trail")
    records = pd.read_parquet(out_dir / "news_records.parquet")
    activations = np.load(out_dir / "news_layers1-28.npy", mmap_mode="r")
    train_mask = records["split"].eq("train").to_numpy()
    val_mask = records["split"].eq("val").to_numpy()
    train_records = records.loc[train_mask].reset_index(drop=True)
    train_labels = records.loc[train_mask, "impact_composite"].to_numpy(float)
    val_labels = records.loc[val_mask, "impact_composite"].to_numpy(float)
    matched_pairs = matched_pair_indices(train_records, train_labels)
    axes = np.empty((len(METHODS), len(LAYERS), HIDDEN_SIZE), dtype=np.float32)
    rows: list[dict] = []
    for layer_index, layer in enumerate(LAYERS):
        values = normalize_rows(activations[:, layer_index].astype(np.float32))
        for method_index, method in enumerate(METHODS):
            axis = fit_axis(
                method,
                values[train_mask],
                train_records,
                train_labels,
                matched_pairs if method == "matched_delta_pc1" else None,
            )
            axes[method_index, layer_index] = axis
            train_score = values[train_mask] @ axis
            val_score = values[val_mask] @ axis
            train_rho = within_ticker_spearman(
                train_score,
                train_labels,
                train_records["ticker"].to_numpy(),
            )
            val_rho = within_ticker_spearman(
                val_score,
                val_labels,
                records.loc[val_mask, "ticker"].to_numpy(),
            )
            rows.append(
                {
                    "method": method,
                    "layer": layer,
                    "train_raw_spearman": float(
                        spearmanr(train_score, train_labels).statistic
                    ),
                    "train_spearman": train_rho,
                    "validation_raw_spearman": float(
                        spearmanr(val_score, val_labels).statistic
                    ),
                    "validation_spearman": val_rho,
                }
            )
        print(f"selected candidates through layer {layer}/28", flush=True)
    np.save(out_dir / "axes_all_methods_layers1-28.npy", axes)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out_dir / "validation_layer_sweep.csv", index=False)

    selected = (
        sweep.sort_values(["method", "validation_spearman"], ascending=[True, False])
        .groupby("method", as_index=False)
        .head(1)
        .sort_values("validation_spearman", ascending=False)
        .reset_index(drop=True)
    )
    primary = selected.iloc[0]
    selection = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_data": "2022 validation only",
        "primary_method": str(primary["method"]),
        "primary_layer": int(primary["layer"]),
        "primary_validation_spearman": float(primary["validation_spearman"]),
        "selected_by_method": selected.to_dict("records"),
        "test_opened": False,
    }
    (out_dir / "selection_frozen.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    scores = records.copy()
    for row in selected.to_dict("records"):
        method = str(row["method"])
        layer = int(row["layer"])
        method_index = METHODS.index(method)
        values = normalize_rows(activations[:, layer - 1].astype(np.float32))
        scores[f"{method}_l{layer}"] = values @ axes[method_index, layer - 1]
    scores.to_parquet(out_dir / "selected_scores.parquet", index=False)
    selected.to_csv(out_dir / "selected_axes.csv", index=False)
    print(json.dumps(selection, indent=2), flush=True)


def open_test(out_dir: Path) -> None:
    result_path = out_dir / "test_results.csv"
    if result_path.exists():
        raise RuntimeError("untouched test has already been opened")
    selection_path = out_dir / "selection_frozen.json"
    selection = json.loads(selection_path.read_text())
    scores = pd.read_parquet(out_dir / "selected_scores.parquet")
    test = scores[scores["split"].eq("test")].copy()
    rows = []
    for selected in selection["selected_by_method"]:
        method = str(selected["method"])
        layer = int(selected["layer"])
        score = f"{method}_l{layer}"
        for outcome in ["impact_composite"] + IMPACT_COMPONENTS:
            result = spearmanr(test[score], test[outcome])
            within = within_ticker_spearman(
                test[score].to_numpy(float),
                test[outcome].to_numpy(float),
                test["ticker"].to_numpy(),
            )
            rows.append(
                {
                    "method": method,
                    "layer": layer,
                    "score": score,
                    "outcome": outcome,
                    "n_test": len(test),
                    "test_spearman": float(result.statistic),
                    "test_pvalue": float(result.pvalue),
                    "test_within_ticker_spearman": within,
                }
            )
    pd.DataFrame(rows).to_csv(result_path, index=False)
    selection["test_opened"] = True
    selection["test_opened_at"] = datetime.now(timezone.utc).isoformat()
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["prepare", "extract-shard", "merge", "select", "test"],
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--keep-shards", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare(args.out_dir)
    elif args.stage == "extract-shard":
        extract_shard(
            args.out_dir,
            args.shard_id,
            args.num_shards,
            args.batch_size,
            args.max_length,
        )
    elif args.stage == "merge":
        merge_shards(args.out_dir, args.num_shards, remove_shards=not args.keep_shards)
    elif args.stage == "select":
        select_axes(args.out_dir)
    elif args.stage == "test":
        open_test(args.out_dir)


if __name__ == "__main__":
    main()
