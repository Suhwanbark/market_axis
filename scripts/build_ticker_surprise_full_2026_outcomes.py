#!/usr/bin/env python3
"""Materialize retrospective outcomes for all frozen 2026 eligible articles."""

from __future__ import annotations

import json
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
FEATURES = (
    ROOT / "outputs" / "retrieval_conditioned_feature_lock" / "features.parquet"
)
FEATURE_LOCK = (
    ROOT
    / "outputs"
    / "retrieval_conditioned_feature_lock"
    / "feature_lock.json"
)
PRICES = ROOT / "data/yahoo_finance_data/raw/data/stock_prices.parquet"
AXIS_MANIFEST = (
    ROOT / "outputs" / "prefixless_article_axis_test" / "axes" / "manifest.json"
)
OUTPUT = ROOT / "outputs" / "ticker_surprise_full_2026"
OUTCOME_COLUMNS = (
    "abs_market_adjusted_return_h1",
    "abs_market_adjusted_return_h2",
    "market_adjusted_return",
    "pre_abs_market_adjusted_return_h1",
    "prevol20",
    "parkinson_expansion_h1",
    "abnormal_volume_h1",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_feature_lock() -> None:
    if not FEATURE_LOCK.exists() or not FEATURES.exists():
        raise RuntimeError("retrieval-conditioned feature lock is missing")
    lock = json.loads(FEATURE_LOCK.read_text())
    if lock.get("status") != "RETRIEVAL_CONDITIONED_FEATURE_LOCK_FROZEN":
        raise RuntimeError("feature lock status is not frozen")
    if sha256(FEATURES) != lock["features"]["sha256"]:
        raise RuntimeError("feature file changed after lock")


def load_price_values(
    tickers: list[str], start: str, end: str
) -> pd.DataFrame:
    table = pq.read_table(
        PRICES,
        columns=[
            "symbol",
            "report_date",
            "open",
            "close",
            "high",
            "low",
            "volume",
        ],
        filters=[
            ("symbol", "in", sorted(set(tickers) | {"SPY"})),
            ("report_date", ">=", start),
            ("report_date", "<=", end),
        ],
    ).to_pandas()
    table.rename(
        columns={"symbol": "market_ticker", "report_date": "date"},
        inplace=True,
    )
    table["date"] = pd.to_datetime(table["date"], errors="coerce")
    for column in ("open", "close", "high", "low", "volume"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    valid = (
        table["date"].notna()
        & table["open"].gt(0)
        & table["close"].gt(0)
        & table["high"].gt(0)
        & table["low"].gt(0)
        & table["high"].ge(table["low"])
    )
    return (
        table.loc[valid]
        .drop_duplicates(["market_ticker", "date"], keep="last")
        .sort_values(["market_ticker", "date"])
        .reset_index(drop=True)
    )


def make_outcomes(
    prices: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    spy = (
        prices.loc[prices["market_ticker"].eq("SPY")]
        .sort_values("date")
        .copy()
    )
    if spy.empty:
        raise RuntimeError("SPY price benchmark is missing")
    spy["spy_return"] = np.log(spy["close"] / spy["close"].shift(1))
    spy_return = spy.set_index("date")["spy_return"]
    rows: list[pd.DataFrame] = []
    for _, group in prices.loc[
        ~prices["market_ticker"].eq("SPY")
    ].groupby("market_ticker", sort=False):
        part = group.sort_values("date").copy()

        # The source has raw OHLC. Exclude split-like overnight gaps before
        # returns or rolling-volatility quantities are formed.
        overnight_ratio = part["open"] / part["close"].shift(1)
        split_factor = np.maximum(overnight_ratio, 1.0 / overnight_ratio)
        standard_factors = np.asarray([1.5, 2.0, 3.0, 4.0, 5.0, 10.0])
        factor_distance = np.abs(
            np.log(split_factor.to_numpy(dtype=float))[:, None]
            - np.log(standard_factors)[None, :]
        )
        nearest_factor = np.argmin(
            np.where(
                np.isfinite(factor_distance),
                factor_distance,
                np.inf,
            ),
            axis=1,
        )
        nearest_log_distance = factor_distance[
            np.arange(len(factor_distance)), nearest_factor
        ]
        factor_tolerance = np.where(
            standard_factors[nearest_factor] == 1.5,
            np.log(1.02),
            np.log(1.05),
        )
        intraday_move = np.abs(np.log(part["close"] / part["open"]))
        part["mechanical_split_flag"] = (
            np.abs(np.log(overnight_ratio)).gt(np.log(1.25))
            & (nearest_log_distance <= factor_tolerance)
            & intraday_move.lt(np.log(1.25))
        )
        raw_return = np.log(part["close"] / part["close"].shift(1))
        part["firm_return"] = raw_return.mask(
            part["mechanical_split_flag"]
        )
        part["spy_return"] = part["date"].map(spy_return)
        part["market_adjusted_return"] = (
            part["firm_return"] - part["spy_return"]
        )
        part["abs_market_adjusted_return_h1"] = part[
            "market_adjusted_return"
        ].abs()
        part["pre_abs_market_adjusted_return_h1"] = part[
            "market_adjusted_return"
        ].shift(1).abs()
        part["_pre_abs_market_adjusted_return_lag2"] = part[
            "market_adjusted_return"
        ].shift(2).abs()
        part["abs_market_adjusted_return_h2"] = (
            part["market_adjusted_return"]
            + part["market_adjusted_return"].shift(-1)
        ).abs()
        part["prevol20"] = (
            part["market_adjusted_return"]
            .shift(1)
            .rolling(20, min_periods=15)
            .std()
        )
        part["_prevol20_lag2"] = (
            part["market_adjusted_return"]
            .shift(2)
            .rolling(20, min_periods=15)
            .std()
        )
        part["parkinson"] = np.log(
            part["high"] / part["low"]
        ).pow(2) / (4.0 * math.log(2.0))
        part["pre_parkinson20"] = (
            part["parkinson"]
            .shift(1)
            .rolling(20, min_periods=15)
            .mean()
        )
        part["parkinson_expansion_h1"] = np.log(
            (part["parkinson"] + 1e-12)
            / (part["pre_parkinson20"] + 1e-12)
        )
        part["pre_volume20"] = (
            part["volume"]
            .shift(1)
            .rolling(20, min_periods=15)
            .median()
        )
        part["abnormal_volume_h1"] = np.log(
            (part["volume"] + 1.0) / (part["pre_volume20"] + 1.0)
        )
        rows.append(part)

    panel = pd.concat(rows, ignore_index=True)
    columns = [
        "market_ticker",
        "date",
        "abs_market_adjusted_return_h1",
        "pre_abs_market_adjusted_return_h1",
        "_pre_abs_market_adjusted_return_lag2",
        "abs_market_adjusted_return_h2",
        "market_adjusted_return",
        "mechanical_split_flag",
        "prevol20",
        "_prevol20_lag2",
        "parkinson_expansion_h1",
        "abnormal_volume_h1",
    ]
    outcomes = features[
        [
            "observation_id",
            "market_ticker",
            "event_session",
            "published_at",
            "downstream_split",
        ]
    ].merge(
        panel[columns],
        left_on=["market_ticker", "event_session"],
        right_on=["market_ticker", "date"],
        how="left",
        validate="many_to_one",
    )
    outcomes.drop(columns="date", inplace=True)

    # Intraday articles use lag 2 rather than same-session information for
    # pre-event controls. Outcomes themselves still refer to event_session.
    published = pd.to_datetime(
        outcomes["published_at"], utc=True, errors="coerce"
    )
    published_ny = published.dt.tz_convert("America/New_York")
    local_date = published_ny.dt.tz_localize(None).dt.normalize()
    local_minutes = published_ny.dt.hour * 60 + published_ny.dt.minute
    trading_dates = set(pd.to_datetime(spy["date"]).dt.normalize())
    intraday = (
        local_date.isin(trading_dates)
        & local_minutes.ge(9 * 60 + 30)
        & local_minutes.lt(16 * 60)
    )
    outcomes["publication_intraday_during_market"] = intraday.astype(bool)
    outcomes.loc[
        intraday, "pre_abs_market_adjusted_return_h1"
    ] = outcomes.loc[intraday, "_pre_abs_market_adjusted_return_lag2"]
    outcomes.loc[intraday, "prevol20"] = outcomes.loc[
        intraday, "_prevol20_lag2"
    ]
    outcomes.drop(
        columns=[
            "published_at",
            "_pre_abs_market_adjusted_return_lag2",
            "_prevol20_lag2",
        ],
        inplace=True,
    )
    return outcomes


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    terminal = OUTPUT / "outcome_manifest.json"
    if terminal.exists():
        status = json.loads(terminal.read_text()).get("status")
        if status == "TICKER_SURPRISE_FULL_2026_OUTCOMES_COMPLETE":
            print(f"already complete: {terminal}")
            return
        raise RuntimeError(f"non-terminal manifest exists: {terminal}")
    axis = json.loads(AXIS_MANIFEST.read_text())
    if (
        axis.get("status") != "PREFIXLESS_TEST_AXES_FROZEN"
        or axis.get("market_labels_used") is not False
        or axis.get("test_articles_used") is not False
    ):
        raise RuntimeError("frozen controlled axis audit failed")
    verify_feature_lock()
    features = pd.read_parquet(
        FEATURES,
        columns=[
            "observation_id",
            "market_ticker",
            "event_session",
            "published_at",
            "downstream_split",
        ],
    )
    event = pd.to_datetime(features.event_session, errors="coerce")
    if len(features) != 9_376 or not features.observation_id.is_unique:
        raise RuntimeError("full-2026 article identity failure")
    if not event.notna().all() or not event.dt.year.eq(2026).all():
        raise RuntimeError("eligible article cohort is not entirely in 2026")
    tickers = sorted(features.market_ticker.astype(str).unique())
    start = (event.min() - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    end = (event.max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    prices = load_price_values(tickers, start, end)
    outcomes = make_outcomes(prices, features)
    if (
        len(outcomes) != len(features)
        or outcomes.observation_id.tolist()
        != features.observation_id.tolist()
    ):
        raise RuntimeError("full-2026 outcome row alignment failure")
    outcomes.to_parquet(OUTPUT / "outcomes.parquet", index=False)
    finite_counts = {
        column: int(np.isfinite(outcomes[column]).sum())
        for column in OUTCOME_COLUMNS
    }
    terminal.write_text(
        json.dumps(
            {
                "status": "TICKER_SURPRISE_FULL_2026_OUTCOMES_COMPLETE",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "interpretation": (
                    "retrospective outcome materialization after controlled "
                    "axis and prefixless score freeze"
                ),
                "rows": len(outcomes),
                "tickers": int(outcomes.market_ticker.nunique()),
                "event_date_min": str(event.min().date()),
                "event_date_max": str(event.max().date()),
                "event_dates": int(event.nunique()),
                "price_start": start,
                "price_end": end,
                "finite_counts": finite_counts,
                "market_values_opened": True,
                "market_labels_used_to_fit_axis": False,
                "axis_manifest": str(AXIS_MANIFEST.relative_to(ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("TICKER_SURPRISE_FULL_2026_OUTCOMES_COMPLETE")


if __name__ == "__main__":
    main()
