#!/usr/bin/env python3
"""Compare direct LLM, text embedding, and hidden activations on impact ratios.

The primary target is intentionally simple:

    log impact ratio = log(post-event 5-day volatility / pre-event 20-day volatility)

Both windows use the Parkinson daily variance estimator. The square root is taken
after averaging daily variances, so the ratio is expressed in volatility units.

The command is split into explicit stages so that validation choices are frozen
before the untouched test set is evaluated.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "impact_ratio_representation_benchmark"
NEWS_RECORDS = ROOT / "outputs" / "fintexts_news_axis_calendar_corrected" / "news_records.parquet"
NEWS_AUDIT = ROOT / "outputs" / "fintexts_news_axis_calendar_corrected" / "data_audit.json"
NEWS_HF_CACHE = ROOT / "outputs" / "fintexts_news_axis_e2e" / "hf_cache"
NEWS_ACTIVATIONS = ROOT / "outputs" / "fintexts_news_axis_e2e" / "news_layers1-28.npy"
FILING_DOCUMENTS = ROOT / "outputs" / "joint_impact_axis_v1" / "filing_documents.parquet"
FILING_PRICES = ROOT / "outputs" / "sec8k_eventtime_impact" / "prices_with_volume.parquet"
FILING_ACTIVATIONS = (
    ROOT / "outputs" / "joint_impact_axis_v1" / "activations" / "qwen25" / "filing_layers.npy"
)
QWEN25 = Path(os.environ.get("QWEN25_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
SHARED_HF_CACHE = ROOT / "shared_cache" / "huggingface"
EPS = 1e-12
RIDGE_ALPHA = 10.0
LAYERS = tuple(range(1, 29))
SEED = 20260720
DISK_FLOOR_GIB = 20.0


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def assert_disk_floor() -> None:
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    if free_gib < DISK_FLOOR_GIB:
        raise RuntimeError(
            f"disk floor reached: {free_gib:.2f} GiB < {DISK_FLOOR_GIB:.2f} GiB"
        )


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def parkinson_variance(frame: pd.DataFrame) -> pd.Series:
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    valid = high.gt(0) & low.gt(0) & high.ge(low)
    result = np.log(high / low).pow(2) / (4.0 * math.log(2.0))
    return result.where(valid)


def add_ratio_windows(
    frame: pd.DataFrame,
    post_start: int,
    include_event_in_history: bool,
) -> pd.DataFrame:
    ordered = frame.sort_values("date").drop_duplicates("date").copy()
    daily = parkinson_variance(ordered)
    history = daily if include_event_in_history else daily.shift(1)
    ordered["pre_var20"] = history.rolling(20, min_periods=20).mean()
    ordered["post_var5"] = pd.concat(
        [daily.shift(-offset) for offset in range(post_start, post_start + 5)],
        axis=1,
    ).mean(axis=1, skipna=False)
    return ordered[["date", "pre_var20", "post_var5"]]


def load_news_prices() -> pd.DataFrame:
    snapshots = NEWS_HF_CACHE / "datasets--EXAONE-BI--FinTexTS" / "snapshots"
    snapshot = next(snapshots.iterdir())
    paths = sorted((snapshot / "data").glob("train-*.parquet"))
    prices = pd.concat(
        [
            pd.read_parquet(path, columns=["date", "ticker", "high", "low"])
            for path in paths
        ],
        ignore_index=True,
    )
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    audit = json.loads(NEWS_AUDIT.read_text(encoding="utf-8"))
    closure_dates = pd.to_datetime(audit["carried_market_closure_dates"])
    return prices.loc[~prices["date"].isin(closure_dates)].copy()


def news_documents_with_ratio() -> pd.DataFrame:
    records = pd.read_parquet(
        NEWS_RECORDS,
        columns=["item_id", "ticker", "date", "split", "news_text", "text_hash"],
    ).copy()
    records["date"] = pd.to_datetime(records["date"])
    prices = load_news_prices()
    windows = pd.concat(
        [
            part.assign(ticker=ticker)
            for ticker, group in prices.groupby("ticker", sort=False)
            for part in [
                add_ratio_windows(
                    group,
                    post_start=1,
                    include_event_in_history=True,
                )
            ]
        ],
        ignore_index=True,
    )
    merged = records.merge(windows, on=["ticker", "date"], how="left", validate="one_to_one")
    return pd.DataFrame(
        {
            "domain": "news",
            "domain_item_id": merged["item_id"].astype(np.int64),
            "source_row": merged["item_id"].astype(np.int64),
            "ticker": merged["ticker"].astype(str),
            "event_date": merged["date"],
            "split": merged["split"].astype(str),
            "text": merged["news_text"].astype(str),
            "text_hash": merged["text_hash"].astype(str),
            "pre_var20": merged["pre_var20"].astype(float),
            "post_var5": merged["post_var5"].astype(float),
        }
    )


def filing_documents_with_ratio() -> pd.DataFrame:
    documents = pd.read_parquet(FILING_DOCUMENTS).copy().reset_index(drop=True)
    documents["source_row"] = np.arange(len(documents), dtype=np.int64)
    documents["event_date"] = pd.to_datetime(documents["event_date"])
    prices = pd.read_parquet(FILING_PRICES, columns=["date", "ticker", "high", "low"])
    prices["date"] = pd.to_datetime(prices["date"])
    windows = pd.concat(
        [
            part.assign(ticker=ticker)
            for ticker, group in prices.groupby("ticker", sort=False)
            if ticker != "SPY"
            for part in [
                add_ratio_windows(
                    group,
                    post_start=0,
                    include_event_in_history=False,
                )
            ]
        ],
        ignore_index=True,
    ).rename(columns={"date": "event_date"})
    merged = documents.merge(
        windows,
        on=["ticker", "event_date"],
        how="left",
        validate="many_to_one",
    )
    return pd.DataFrame(
        {
            "domain": "filing",
            "domain_item_id": merged["domain_item_id"].astype(np.int64),
            "source_row": merged["source_row"].astype(np.int64),
            "ticker": merged["ticker"].astype(str),
            "event_date": merged["event_date"],
            "split": merged["split"].astype(str),
            "text": merged["text"].astype(str),
            "text_hash": pd.util.hash_pandas_object(merged["text"], index=False).astype(str),
            "pre_var20": merged["pre_var20"].astype(float),
            "post_var5": merged["post_var5"].astype(float),
        }
    )


def prepare() -> None:
    assert_disk_floor()
    OUT.mkdir(parents=True, exist_ok=True)
    documents = pd.concat(
        [news_documents_with_ratio(), filing_documents_with_ratio()],
        ignore_index=True,
    )
    documents = documents.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["pre_var20", "post_var5", "text", "event_date"]
    )
    documents = documents.loc[
        documents["pre_var20"].gt(EPS)
        & documents["post_var5"].gt(EPS)
        & documents["text"].str.len().ge(40)
    ].copy()
    documents["pre_sigma20"] = np.sqrt(documents["pre_var20"])
    documents["post_sigma5"] = np.sqrt(documents["post_var5"])
    documents["impact_ratio"] = documents["post_sigma5"] / documents["pre_sigma20"]
    documents["log_impact_ratio"] = np.log(documents["impact_ratio"])
    documents = documents.sort_values(
        ["domain", "domain_item_id"], kind="stable"
    ).reset_index(drop=True)
    documents["benchmark_row"] = np.arange(len(documents), dtype=np.int64)
    documents.to_parquet(OUT / "documents.parquet", index=False)

    counts = (
        documents.groupby(["domain", "split"]).size().unstack(fill_value=0).to_dict("index")
    )
    quantiles = (
        documents.groupby(["domain", "split"])["impact_ratio"]
        .quantile([0.01, 0.1, 0.5, 0.9, 0.99])
        .rename("impact_ratio")
        .reset_index()
        .to_dict("records")
    )
    audit = {
        "target": "log(post_5d_volatility / pre_20d_volatility)",
        "display_target": "post_5d_volatility / pre_20d_volatility",
        "daily_variance": "Parkinson high-low estimator",
        "news_timing": "pre includes news date t; post uses next five trading days t+1..t+5",
        "filing_timing": "pre ends before mapped reaction session; post starts at reaction session",
        "counts": counts,
        "impact_ratio_quantiles": quantiles,
        "news_activation_source": str(NEWS_ACTIVATIONS),
        "filing_activation_source": str(FILING_ACTIVATIONS),
    }
    write_json(OUT / "data_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


def require_cuda_one_visible() -> None:
    visible = str(torch.cuda.device_count())
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one visible CUDA device (physical GPU 1), found {visible}"
        )


def domain_max_tokens(domain: str) -> int:
    return 512 if domain == "news" else 2048


def extract_embeddings(model_ref: str, batch_news: int, batch_filing: int) -> None:
    assert_disk_floor()
    require_cuda_one_visible()
    documents = pd.read_parquet(OUT / "documents.parquet")
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        cache_dir=SHARED_HF_CACHE,
    )
    model = AutoModel.from_pretrained(
        model_ref,
        cache_dir=SHARED_HF_CACHE,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    hidden_size = int(model.config.hidden_size)
    output = np.lib.format.open_memmap(
        OUT / "embedding_vectors.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(documents), hidden_size),
    )

    with torch.inference_mode():
        for domain, part in documents.groupby("domain", sort=False):
            row_ids = part.index.to_numpy(np.int64)
            texts = part["text"].tolist()
            batch_size = batch_news if domain == "news" else batch_filing
            max_length = domain_max_tokens(domain)
            for start in range(0, len(texts), batch_size):
                batch_text = texts[start : start + batch_size]
                encoded = tokenizer(
                    batch_text,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
                result = model(**encoded, return_dict=True)
                # BGE-M3 and other BERT-style embedding encoders use the CLS state.
                pooled = result.last_hidden_state[:, 0].float()
                pooled = torch.nn.functional.normalize(pooled, dim=1)
                rows = row_ids[start : start + len(batch_text)]
                output[rows] = pooled.cpu().numpy().astype(np.float16)
                if start % max(batch_size * 100, 1) == 0:
                    print(
                        f"embedding {domain}: {min(start + len(batch_text), len(texts))}/{len(texts)}",
                        flush=True,
                    )
                assert_disk_floor()
    output.flush()
    write_json(
        OUT / "embedding_manifest.json",
        {
            "model": model_ref,
            "rows": len(documents),
            "hidden_size": hidden_size,
            "dtype": "float16",
            "pooling": "CLS then L2 normalize",
            "max_tokens": {"news": 512, "filing": 2048},
            "cuda_visible_devices": "physical GPU 1 only",
        },
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


DIRECT_INSTRUCTION = """Assess the magnitude of the stock-price volatility that the document itself is likely to cause relative to the company's usual level. Judge magnitude, not whether the price will rise or fall. Use 1 for routine information with little expected effect and 9 for information likely to cause major repricing or uncertainty. Answer with exactly one digit from 1 to 9.\n\nDocument:\n{text}\n\nScore:"""


def direct_candidate_ids(tokenizer) -> list[int]:
    candidate_ids = []
    for digit in range(1, 10):
        encoded = tokenizer.encode(str(digit), add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"direct score digit {digit} is not a single token: {encoded}")
        candidate_ids.append(encoded[0])
    return candidate_ids


def extract_direct_scores(batch_news: int, batch_filing: int) -> None:
    assert_disk_floor()
    require_cuda_one_visible()
    documents = pd.read_parquet(OUT / "documents.parquet")
    tokenizer = AutoTokenizer.from_pretrained(QWEN25, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    candidate_ids = direct_candidate_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN25,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()

    partial_path = OUT / "direct_scores.partial.npz"
    if partial_path.exists():
        partial = np.load(partial_path)
        score = partial["score"].astype(np.float32)
        probabilities = partial["probabilities"].astype(np.float32)
        print(f"resuming {int(np.isfinite(score).sum())} direct scores", flush=True)
    else:
        score = np.full(len(documents), np.nan, dtype=np.float32)
        probabilities = np.full((len(documents), 9), np.nan, dtype=np.float32)
    candidate_tensor = torch.tensor(candidate_ids, device="cuda")
    values = torch.arange(1, 10, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        for domain, part in documents.groupby("domain", sort=False):
            # The direct method is not trained, so train rows are unnecessary.
            part = part.loc[
                part["split"].isin(["val", "test"])
                & np.isnan(score[part.index.to_numpy(np.int64)])
            ]
            row_ids = part.index.to_numpy(np.int64)
            texts = part["text"].tolist()
            batch_size = batch_news if domain == "news" else batch_filing
            max_doc_tokens = domain_max_tokens(domain)
            for start in range(0, len(texts), batch_size):
                raw_batch = texts[start : start + batch_size]
                truncated = tokenizer(
                    raw_batch,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_doc_tokens,
                )["input_ids"]
                prompts = []
                for token_ids in truncated:
                    text = tokenizer.decode(token_ids, skip_special_tokens=True)
                    user = DIRECT_INSTRUCTION.format(text=text)
                    prompts.append(
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": user}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )
                encoded = tokenizer(
                    prompts,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
                logits = model(
                    **encoded,
                    return_dict=True,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits[:, -1]
                digit_logits = logits.index_select(1, candidate_tensor).float()
                probs = torch.softmax(digit_logits, dim=1)
                expected = (probs * values).sum(dim=1)
                rows = row_ids[start : start + len(raw_batch)]
                score[rows] = expected.cpu().numpy()
                probabilities[rows] = probs.cpu().numpy()
                if start % max(batch_size * 100, 1) == 0:
                    print(
                        f"direct {domain}: {min(start + len(raw_batch), len(texts))}/{len(texts)}",
                        flush=True,
                    )
                    np.savez_compressed(
                        partial_path,
                        score=score,
                        probabilities=probabilities,
                    )

    output = documents[["benchmark_row", "domain", "split"]].copy()
    output["direct_score"] = score
    for index in range(9):
        output[f"prob_{index + 1}"] = probabilities[:, index]
    output.to_parquet(OUT / "direct_scores.parquet", index=False)
    partial_path.unlink(missing_ok=True)
    write_json(
        OUT / "direct_manifest.json",
        {
            "model": str(QWEN25),
            "prompt": DIRECT_INSTRUCTION,
            "score": "expected value of next-token probabilities restricted to digits 1..9",
            "splits": ["val", "test"],
            "max_document_tokens": {"news": 512, "filing": 2048},
            "cuda_visible_devices": "physical GPU 1 only",
        },
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def activation_source(domain: str) -> Path:
    return NEWS_ACTIVATIONS if domain == "news" else FILING_ACTIVATIONS


def fit_axis(x: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, float]:
    model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr")
    model.fit(x, label)
    axis = np.asarray(model.coef_, dtype=np.float32)
    axis /= max(float(np.linalg.norm(axis)), 1e-8)
    return axis, float(model.intercept_)


def fit() -> None:
    documents = pd.read_parquet(OUT / "documents.parquet")
    embedding = np.load(OUT / "embedding_vectors.npy", mmap_mode="r")
    selection_rows: list[dict] = []
    frozen: dict[str, dict] = {}

    for domain, part in documents.groupby("domain", sort=False):
        part_rows = part.index.to_numpy(np.int64)
        source_rows = part["source_row"].to_numpy(np.int64)
        train = part["split"].eq("train").to_numpy()
        validation = part["split"].eq("val").to_numpy()
        label = part["log_impact_ratio"].to_numpy(float)

        embedding_x = normalize_rows(embedding[part_rows])
        embedding_axis, _ = fit_axis(embedding_x[train], label[train])
        embedding_score = embedding_x @ embedding_axis
        embedding_val = safe_spearman(embedding_score[validation], label[validation])
        np.save(OUT / f"{domain}_embedding_axis.npy", embedding_axis)

        activations = np.load(activation_source(domain), mmap_mode="r")
        layer_axes = np.empty((len(LAYERS), activations.shape[2]), dtype=np.float32)
        layer_rows = []
        for layer in LAYERS:
            x = normalize_rows(activations[source_rows, layer - 1])
            axis, _ = fit_axis(x[train], label[train])
            score = x @ axis
            train_rho = safe_spearman(score[train], label[train])
            validation_rho = safe_spearman(score[validation], label[validation])
            layer_axes[layer - 1] = axis
            layer_rows.append(
                {
                    "domain": domain,
                    "representation": "activation",
                    "layer": layer,
                    "alpha": RIDGE_ALPHA,
                    "train_spearman": train_rho,
                    "validation_spearman": validation_rho,
                }
            )
            print(
                f"fit {domain} layer {layer}: validation rho={validation_rho:.4f}",
                flush=True,
            )
        layer_frame = pd.DataFrame(layer_rows)
        selection_rows.extend(layer_rows)
        selected = layer_frame.sort_values(
            ["validation_spearman", "layer"], ascending=[False, True]
        ).iloc[0]
        selected_layer = int(selected["layer"])
        np.save(OUT / f"{domain}_activation_axes.npy", layer_axes)
        selection_rows.append(
            {
                "domain": domain,
                "representation": "dedicated_embedding",
                "layer": np.nan,
                "alpha": RIDGE_ALPHA,
                "train_spearman": safe_spearman(embedding_score[train], label[train]),
                "validation_spearman": embedding_val,
            }
        )
        frozen[domain] = {
            "activation_layer": selected_layer,
            "activation_validation_spearman": float(selected["validation_spearman"]),
            "qwen_final_layer": 28,
            "embedding_validation_spearman": embedding_val,
            "ridge_alpha": RIDGE_ALPHA,
            "selection_target": "log_impact_ratio",
            "selection_split": "val",
        }

    pd.DataFrame(selection_rows).to_csv(OUT / "validation_selection.csv", index=False)
    write_json(
        OUT / "frozen_config.json",
        {
            "frozen_before_test": True,
            "domains": frozen,
            "test_selection_prohibited": True,
        },
    )
    print(json.dumps(frozen, indent=2), flush=True)


def decile_ids(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="first")
    return pd.qcut(ranks, 10, labels=False).to_numpy(np.int64) + 1


def within_ticker_rho(frame: pd.DataFrame, score_column: str) -> float:
    score = frame[score_column] - frame.groupby("ticker")[score_column].transform("mean")
    label = frame["log_impact_ratio"] - frame.groupby("ticker")[
        "log_impact_ratio"
    ].transform("mean")
    return safe_spearman(score.to_numpy(), label.to_numpy())


def method_metrics(frame: pd.DataFrame, method: str) -> tuple[dict, pd.DataFrame]:
    working = frame[["ticker", "event_date", "log_impact_ratio", "impact_ratio", method]].dropna()
    working = working.copy()
    working["decile"] = decile_ids(working[method].to_numpy())
    true_top = decile_ids(working["log_impact_ratio"].to_numpy()) == 10
    predicted_top = working["decile"].to_numpy() == 10
    deciles = (
        working.groupby("decile")
        .agg(
            n=(method, "size"),
            mean_log_impact=("log_impact_ratio", "mean"),
            median_impact_ratio=("impact_ratio", "median"),
        )
        .reset_index()
    )
    deciles["geometric_mean_impact_ratio"] = np.exp(deciles["mean_log_impact"])
    deciles["method"] = method
    metrics = {
        "method": method,
        "n_test": len(working),
        "test_spearman": safe_spearman(working[method], working["log_impact_ratio"]),
        "within_ticker_spearman": within_ticker_rho(working, method),
        "decile_mean_spearman": safe_spearman(
            deciles["decile"], deciles["mean_log_impact"]
        ),
        "adjacent_decile_increases": int(
            np.sum(np.diff(deciles["mean_log_impact"].to_numpy()) > 0)
        ),
        "top_bottom_log_spread": float(
            deciles.loc[deciles["decile"].eq(10), "mean_log_impact"].iloc[0]
            - deciles.loc[deciles["decile"].eq(1), "mean_log_impact"].iloc[0]
        ),
        "top_bottom_ratio_multiple": float(
            np.exp(
                deciles.loc[deciles["decile"].eq(10), "mean_log_impact"].iloc[0]
                - deciles.loc[deciles["decile"].eq(1), "mean_log_impact"].iloc[0]
            )
        ),
        "precision_at_10pct": float(np.mean(true_top[predicted_top])),
    }
    return metrics, deciles


def bootstrap_rho_difference(
    frame: pd.DataFrame,
    ours: str,
    baseline: str,
    repetitions: int,
) -> dict:
    rng = np.random.default_rng(SEED)
    dates = pd.Index(frame["event_date"].drop_duplicates())
    by_date = {date: np.flatnonzero(frame["event_date"].eq(date).to_numpy()) for date in dates}
    differences = []
    for _ in range(repetitions):
        sampled = rng.choice(dates.to_numpy(), size=len(dates), replace=True)
        rows = np.concatenate([by_date[pd.Timestamp(date)] for date in sampled])
        ours_rho = safe_spearman(
            frame.iloc[rows][ours], frame.iloc[rows]["log_impact_ratio"]
        )
        baseline_rho = safe_spearman(
            frame.iloc[rows][baseline], frame.iloc[rows]["log_impact_ratio"]
        )
        differences.append(ours_rho - baseline_rho)
    values = np.asarray(differences, dtype=float)
    return {
        "ours": ours,
        "baseline": baseline,
        "rho_difference": safe_spearman(
            frame[ours], frame["log_impact_ratio"]
        )
        - safe_spearman(frame[baseline], frame["log_impact_ratio"]),
        "ci_low": float(np.nanquantile(values, 0.025)),
        "ci_high": float(np.nanquantile(values, 0.975)),
        "one_sided_p_activation_not_better": float(np.nanmean(values <= 0)),
        "bootstrap_repetitions": repetitions,
    }


def score_test_domain(
    domain: str,
    documents: pd.DataFrame,
    embedding: np.ndarray,
    direct: pd.DataFrame,
    frozen: dict,
) -> pd.DataFrame:
    part = documents.loc[documents["domain"].eq(domain)].copy()
    part_rows = part.index.to_numpy(np.int64)
    source_rows = part["source_row"].to_numpy(np.int64)
    embedding_axis = np.load(OUT / f"{domain}_embedding_axis.npy")
    embedding_x = normalize_rows(embedding[part_rows])
    part["embedding_score"] = embedding_x @ embedding_axis

    activations = np.load(activation_source(domain), mmap_mode="r")
    layer_axes = np.load(OUT / f"{domain}_activation_axes.npy")
    selected_layer = int(frozen["activation_layer"])
    selected_x = normalize_rows(activations[source_rows, selected_layer - 1])
    part["activation_score"] = selected_x @ layer_axes[selected_layer - 1]
    final_x = normalize_rows(activations[source_rows, 27])
    part["qwen_final_score"] = final_x @ layer_axes[27]
    part = part.merge(
        direct[["benchmark_row", "direct_score"]],
        on="benchmark_row",
        how="left",
        validate="one_to_one",
    )
    return part.loc[part["split"].eq("test")].copy()


def make_figure(deciles: pd.DataFrame) -> None:
    labels = {
        "direct_score": "LLM direct",
        "embedding_score": "Dedicated embedding + Ridge",
        "qwen_final_score": "Qwen final layer + Ridge",
        "activation_score": "Hidden activation axis",
    }
    colors = {
        "direct_score": "#666666",
        "embedding_score": "#d97706",
        "qwen_final_score": "#2563eb",
        "activation_score": "#15803d",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=False)
    for axis, (domain, part) in zip(axes, deciles.groupby("domain", sort=False)):
        for method, curve in part.groupby("method", sort=False):
            axis.plot(
                curve["decile"],
                curve["geometric_mean_impact_ratio"],
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=colors[method],
                label=labels[method],
            )
        axis.axhline(1.0, color="#999999", linewidth=0.9, linestyle="--")
        axis.set_xticks(range(1, 11))
        axis.set_xlabel("Document score decile")
        axis.set_ylabel("Post-5d / Pre-20d volatility")
        axis.set_title("News" if domain == "news" else "8-K")
        axis.grid(alpha=0.18)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Does the document score rank normalized market impact?")
    fig.tight_layout(rect=[0, 0.13, 1, 0.94])
    fig.savefig(OUT / "representation_decile_comparison.png", dpi=220)
    plt.close(fig)


def evaluate(bootstrap_repetitions: int) -> None:
    frozen_path = OUT / "frozen_config.json"
    if not frozen_path.exists():
        raise RuntimeError("fit stage must freeze validation choices before test evaluation")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["domains"]
    documents = pd.read_parquet(OUT / "documents.parquet")
    embedding = np.load(OUT / "embedding_vectors.npy", mmap_mode="r")
    direct = pd.read_parquet(OUT / "direct_scores.parquet")
    all_scores = []
    metric_rows = []
    decile_rows = []
    bootstrap_rows = []
    methods = [
        "direct_score",
        "embedding_score",
        "qwen_final_score",
        "activation_score",
    ]
    for domain in ["news", "filing"]:
        test = score_test_domain(domain, documents, embedding, direct, frozen[domain])
        all_scores.append(test)
        for method in methods:
            metrics, deciles = method_metrics(test, method)
            metrics["domain"] = domain
            deciles["domain"] = domain
            metric_rows.append(metrics)
            decile_rows.append(deciles)
        for baseline in ["direct_score", "embedding_score", "qwen_final_score"]:
            result = bootstrap_rho_difference(
                test.dropna(subset=["activation_score", baseline]),
                "activation_score",
                baseline,
                bootstrap_repetitions,
            )
            result["domain"] = domain
            bootstrap_rows.append(result)

    scores = pd.concat(all_scores, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    deciles = pd.concat(decile_rows, ignore_index=True)
    bootstrap = pd.DataFrame(bootstrap_rows)
    scores.to_parquet(OUT / "test_scores.parquet", index=False)
    metrics.to_csv(OUT / "test_metrics.csv", index=False)
    deciles.to_csv(OUT / "test_deciles.csv", index=False)
    bootstrap.to_csv(OUT / "paired_date_bootstrap.csv", index=False)
    make_figure(deciles)

    report = [
        "# Impact-ratio representation benchmark",
        "",
        "Primary target: `log(Post-5d Parkinson volatility / Pre-20d Parkinson volatility)`.",
        "The figure displays the same target as an intuitive volatility multiple.",
        "All supervised representations use Ridge(alpha=10). Hidden layer selection uses validation only.",
        "",
        "## Frozen selection",
        "",
        "```json",
        json.dumps(frozen, indent=2),
        "```",
        "",
        "## Untouched test metrics",
        "",
        "```text",
        metrics.to_string(index=False),
        "```",
        "",
        "## Paired date-block bootstrap: activation minus baseline Spearman",
        "",
        "```text",
        bootstrap.to_string(index=False),
        "```",
        "",
        "![Decile comparison](representation_decile_comparison.png)",
    ]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)
    print(bootstrap.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")

    embedding = subparsers.add_parser("extract-embedding")
    embedding.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    embedding.add_argument("--batch-news", type=int, default=16)
    embedding.add_argument("--batch-filing", type=int, default=4)

    direct = subparsers.add_parser("extract-direct")
    direct.add_argument("--batch-news", type=int, default=8)
    direct.add_argument("--batch-filing", type=int, default=2)

    subparsers.add_parser("fit")
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--bootstrap-repetitions", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "extract-embedding":
        extract_embeddings(args.model, args.batch_news, args.batch_filing)
    elif args.command == "extract-direct":
        extract_direct_scores(args.batch_news, args.batch_filing)
    elif args.command == "fit":
        fit()
    elif args.command == "evaluate":
        evaluate(args.bootstrap_repetitions)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
