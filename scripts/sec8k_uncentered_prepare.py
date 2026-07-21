#!/usr/bin/env python3
"""Build a reproducible uncentered SEC 8-K document/event panel.

The public disclosure dump does not include EDGAR acceptance timestamps.  The
event clock is therefore deliberately conservative: the first ticker trading
session strictly after ``file_date``.  No ticker means are removed from either
the label or (later) the representations.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "sec8k_disclosure_dataset" / "8k_all_data_with_text.csv"
OUT = ROOT / "outputs" / "sec8k_uncentered"
PRICES = OUT / "adjusted_ohlcv.parquet"
LABEL = "parkinson_market_adjusted_expansion"
DATASET_ID = "Disclosures-SSRC/8k_disclosure_dataset"
DATASET_REVISION = "2370739fda5983b658556692b38a3b9bdbc5a0cb"
EPS = 1e-12


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


def parse_ticker(value: object) -> str | None:
    try:
        candidates = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        ticker = str(candidate).strip().upper()
        # Avoid warrants and punctuation that Yahoo handles inconsistently.
        if re.fullmatch(r"[A-Z]{1,5}", ticker):
            return ticker
    return None


def document_priority(file_type: object) -> int:
    kind = str(file_type).strip().upper().replace(" ", "")
    if kind in {"EX-99.1", "EX-99.01"}:
        return 0
    if kind.startswith("EX-99"):
        return 1
    if kind == "8-K":
        return 2
    return 3


def clean_documents() -> pd.DataFrame:
    columns = [
        "accession",
        "company_cik_trimmed",
        "company",
        "tickers",
        "root_form",
        "file_date",
        "file_type",
        "file_description",
        "items",
        "filing_document_url",
        "text",
    ]
    raw = pd.read_csv(SOURCE, usecols=columns, low_memory=False)
    raw["file_date"] = pd.to_datetime(raw["file_date"], errors="coerce").dt.tz_localize(None)
    raw = raw.loc[
        raw["root_form"].astype(str).str.upper().eq("8-K")
        & raw["file_date"].dt.year.between(2022, 2025)
        & raw["text"].notna()
    ].copy()
    raw["ticker"] = raw["tickers"].map(parse_ticker)
    raw["priority"] = raw["file_type"].map(document_priority)
    raw = raw.loc[raw["ticker"].notna() & raw["priority"].lt(3)].copy()
    raw["text"] = raw["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    raw["source_text_chars"] = raw["text"].str.len()
    raw = raw.loc[raw["source_text_chars"].ge(500)].copy()
    raw = (
        raw.sort_values(["accession", "priority", "source_text_chars"], ascending=[True, True, False])
        .drop_duplicates("accession", keep="first")
        .reset_index(drop=True)
    )
    raw["text"] = raw["text"].str.slice(0, 5000)
    raw["text_chars"] = raw["text"].str.len()
    year = raw["file_date"].dt.year
    raw["split"] = np.select(
        [year.le(2023), year.eq(2024), year.eq(2025)], ["train", "val", "test"], default=""
    )
    coverage = raw.groupby("ticker")["split"].agg(["size", "nunique"])
    keep = coverage.index[coverage["size"].ge(8) & coverage["nunique"].eq(3)]
    return raw.loc[raw["ticker"].isin(keep)].copy()


def unpack_yahoo(data: pd.DataFrame, tickers: list[str]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    if data.empty:
        return frames
    if not isinstance(data.columns, pd.MultiIndex):
        if len(tickers) != 1:
            return frames
        data.columns = pd.MultiIndex.from_product([[tickers[0]], data.columns])
    # group_by='ticker' normally gives (ticker, field).  Handle the inverse too.
    outer = set(map(str, data.columns.get_level_values(0)))
    ticker_first = bool(outer.intersection(tickers))
    for ticker in tickers:
        try:
            part = data[ticker] if ticker_first else data.xs(ticker, axis=1, level=1)
        except (KeyError, ValueError):
            continue
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if not set(needed).issubset(part.columns):
            continue
        frame = part[needed].copy()
        frame.columns = [column.lower() for column in frame.columns]
        frame["date"] = pd.to_datetime(frame.index).tz_localize(None)
        frame["ticker"] = ticker
        frame = frame.reset_index(drop=True)
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["open", "high", "low", "close"])
        if not frame.empty:
            frames.append(frame[["ticker", "date", "open", "high", "low", "close", "volume"]])
    return frames


def download_prices(tickers: list[str], batch_size: int) -> pd.DataFrame:
    import yfinance as yf

    PRICES.parent.mkdir(parents=True, exist_ok=True)
    if PRICES.exists():
        result = pd.read_parquet(PRICES)
        result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)
        print(f"prices: using cached {len(result):,} rows", flush=True)
        return result
    requested = sorted(set(tickers) | {"SPY"})
    frames: list[pd.DataFrame] = []
    found: set[str] = set()
    for start in range(0, len(requested), batch_size):
        batch = requested[start : start + batch_size]
        data = yf.download(
            batch,
            start="2022-01-01",
            end="2025-11-01",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
            timeout=30,
        )
        batch_frames = unpack_yahoo(data, batch)
        frames.extend(batch_frames)
        found.update(frame["ticker"].iloc[0] for frame in batch_frames)
        print(f"prices: {min(start + len(batch), len(requested))}/{len(requested)}", flush=True)
        time.sleep(0.25)
    missing = sorted(set(requested) - found)
    # A batch can be throttled selectively; make one explicit retry per missing symbol.
    for index, ticker in enumerate(missing, start=1):
        data = yf.download(
            ticker,
            start="2022-01-01",
            end="2025-11-01",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="ticker",
            timeout=30,
        )
        ticker_frames = unpack_yahoo(data, [ticker])
        frames.extend(ticker_frames)
        found.update(frame["ticker"].iloc[0] for frame in ticker_frames)
        print(f"prices retry: {index}/{len(missing)} {ticker}", flush=True)
        time.sleep(0.25)
    if not frames:
        raise RuntimeError("Yahoo returned no usable price series")
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(["ticker", "date"], keep="last")
    valid = (
        result["open"].gt(0)
        & result["high"].gt(0)
        & result["low"].gt(0)
        & result["close"].gt(0)
        & result["high"].ge(result["low"])
    )
    result = result.loc[valid].sort_values(["ticker", "date"]).reset_index(drop=True)
    result.to_parquet(PRICES, index=False)
    write_json(
        OUT / "price_manifest.json",
        {
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "provider": "Yahoo Finance through yfinance",
            "auto_adjust": True,
            "start_inclusive": "2022-01-01",
            "end_exclusive": "2025-11-01",
            "requested_tickers": requested,
            "found_tickers": sorted(found),
            "missing_tickers": sorted(set(requested) - found),
            "rows": len(result),
        },
    )
    return result


def add_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices["parkinson"] = (
        np.log(prices["high"] / prices["low"]).pow(2) / (4.0 * math.log(2.0))
    )
    firm = prices.groupby("ticker", sort=False)["parkinson"]
    prices["firm_pre20"] = firm.transform(
        lambda values: values.shift(1).rolling(20, min_periods=20).mean()
    )
    prices["firm_post5"] = pd.concat(
        [firm.shift(-offset) for offset in range(0, 5)], axis=1
    ).mean(axis=1, skipna=False)

    market_daily = prices.loc[~prices["ticker"].eq("SPY")].groupby("date")["parkinson"].mean()
    market = pd.DataFrame({"date": market_daily.index, "market_daily": market_daily.to_numpy()})
    market["market_pre20"] = (
        market["market_daily"].shift(1).rolling(20, min_periods=20).mean()
    )
    market["market_post5"] = pd.concat(
        [market["market_daily"].shift(-offset) for offset in range(0, 5)], axis=1
    ).mean(axis=1, skipna=False)
    return prices.merge(
        market[["date", "market_pre20", "market_post5"]],
        on="date",
        how="left",
        validate="many_to_one",
    )


def attach_event_sessions(documents: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for ticker, docs in documents.groupby("ticker", sort=False):
        dates = (
            prices.loc[prices["ticker"].eq(ticker), "date"]
            .drop_duplicates()
            .sort_values()
            .to_numpy(dtype="datetime64[ns]")
        )
        if not len(dates):
            continue
        filing = docs["file_date"].to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(dates, filing, side="right")
        valid = positions < len(dates)
        part = docs.loc[valid].copy()
        part["event_session"] = pd.to_datetime(dates[positions[valid]])
        pieces.append(part)
    if not pieces:
        raise RuntimeError("no documents could be mapped to a later trading session")
    return pd.concat(pieces, ignore_index=True)


def prepare(batch_size: int) -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUT.mkdir(parents=True, exist_ok=True)
    documents = clean_documents()
    candidate_counts = documents.groupby("split").size().to_dict()
    candidate_tickers = int(documents["ticker"].nunique())
    prices = download_prices(documents["ticker"].unique().tolist(), batch_size)
    featured = add_price_features(prices)
    documents = attach_event_sessions(documents, featured)
    label_fields = [
        "ticker",
        "date",
        "firm_pre20",
        "firm_post5",
        "market_pre20",
        "market_post5",
    ]
    labels = featured[label_fields].rename(columns={"date": "event_session"})
    documents = documents.merge(
        labels, on=["ticker", "event_session"], how="left", validate="many_to_one"
    )
    documents[LABEL] = 0.5 * (
        np.log(documents["firm_post5"].clip(lower=EPS))
        - np.log(documents["firm_pre20"].clip(lower=EPS))
        - np.log(documents["market_post5"].clip(lower=EPS))
        + np.log(documents["market_pre20"].clip(lower=EPS))
    )
    documents = documents.replace([np.inf, -np.inf], np.nan).dropna(subset=[LABEL]).copy()
    # Keep a stable cross-split universe after price and label eligibility too.
    coverage = documents.groupby("ticker")["split"].agg(["size", "nunique"])
    keep = coverage.index[coverage["size"].ge(8) & coverage["nunique"].eq(3)]
    documents = documents.loc[documents["ticker"].isin(keep)].copy()
    documents = documents.sort_values(["file_date", "ticker", "accession"]).reset_index(drop=True)
    documents["doc_row"] = np.arange(len(documents), dtype=np.int64)
    documents["split_row"] = documents.groupby("split").cumcount().astype(np.int64)
    documents["item_id"] = documents["accession"].astype(str)
    output_columns = [
        "doc_row",
        "split_row",
        "item_id",
        "accession",
        "ticker",
        "company_cik_trimmed",
        "company",
        "file_date",
        "event_session",
        "split",
        "file_type",
        "file_description",
        "items",
        "filing_document_url",
        "source_text_chars",
        "text_chars",
        "text",
        "firm_pre20",
        "firm_post5",
        "market_pre20",
        "market_post5",
        LABEL,
    ]
    documents[output_columns].to_parquet(OUT / "documents.parquet", index=False)
    counts = documents.groupby("split").size().reindex(["train", "val", "test"], fill_value=0)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_rows": 9444,
        "document_rule": "root 8-K; one document per accession; EX-99.1 > other EX-99 > 8-K",
        "text_rule": "whitespace compacted, source >=500 chars, truncated to first 5000 chars",
        "ticker_rule": "first plain 1-5 letter ticker; >=8 eligible documents and all three splits",
        "split_rule": {"train": "2022-2023", "val": "2024", "test": "2025"},
        "candidate_counts_before_prices": candidate_counts,
        "candidate_tickers_before_prices": candidate_tickers,
        "final_counts": {key: int(value) for key, value in counts.items()},
        "final_tickers": int(documents["ticker"].nunique()),
        "event_clock": "first ticker trading session strictly after file_date",
        "event_clock_limitation": "public CSV lacks EDGAR acceptance time; exact before/after-close timing is unavailable",
        "label": LABEL,
        "label_formula": "0.5 * (log firm post5/pre20 - log equal-weight market post5/pre20)",
        "pre_window": "20 ticker trading sessions ending before event_session",
        "post_window": "event_session through next four ticker trading sessions",
        "market": "daily equal-weight Parkinson variance across downloaded non-SPY ticker universe",
        "centering": "none; no ticker mean subtraction for labels or representations",
        "test_opened_during_preparation": True,
        "note": "splits and labels are deterministic preprocessing; representation/model selection remains train/validation only",
    }
    write_json(OUT / "data_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=30)
    args = parser.parse_args()
    prepare(args.batch_size)


if __name__ == "__main__":
    main()
