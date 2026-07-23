#!/usr/bin/env python3
"""Collect the minimal FMP dataset for the cutoff-safe market-volatility study.

The collector is deliberately single-threaded, rate-limited, resumable, and
cache-first. API keys are sent in a request header and are never written to
disk. Each completed response is stored as compressed JSON with request
metadata and a SHA-256 digest of the uncompressed response body.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


BASE_URL = "https://financialmodelingprep.com/stable"
DEFAULT_START = date(2024, 1, 1)
DEFAULT_END = date(2026, 6, 30)
DEFAULT_LIMIT = 250

MARKET_NEWS_SYMBOLS = (
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLB",
    "XLU",
    "XLRE",
    "XLC",
    "TLT",
    "HYG",
    "UUP",
    "GLD",
    "USO",
    "VXX",
)

MACRO_EVENT_PATTERNS = (
    re.compile(r"\bCPI\b", re.IGNORECASE),
    re.compile(r"\bPCE Price Index\b", re.IGNORECASE),
    re.compile(r"\bNon[\s-]?farm Payrolls\b", re.IGNORECASE),
    re.compile(r"\bUnemployment Rate\b", re.IGNORECASE),
    re.compile(r"\bAverage Hourly Earnings\b", re.IGNORECASE),
    re.compile(r"\bRetail Sales\b", re.IGNORECASE),
    re.compile(
        r"\bISM (?:Manufacturing|Services|Non-Manufacturing).*PMI\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bFed Interest Rate Decision\b", re.IGNORECASE),
)


class RequestBudgetReached(RuntimeError):
    """Raised when a configured per-run request budget is exhausted."""


class RateLimitReached(RuntimeError):
    """Raised immediately on HTTP 429 to avoid adding more API load."""


@dataclass(frozen=True)
class MonthWindow:
    key: str
    start: date
    end: date


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def month_windows(start: date, end: date) -> Iterator[MonthWindow]:
    if end < start:
        raise ValueError("end date must not precede start date")

    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        if cursor.month == 12:
            next_month = date(cursor.year + 1, 1, 1)
        else:
            next_month = date(cursor.year, cursor.month + 1, 1)
        yield MonthWindow(
            key=cursor.strftime("%Y-%m"),
            start=max(start, cursor),
            end=min(end, next_month - timedelta(days=1)),
        )
        cursor = next_month


def day_windows(
    start: date,
    end: date,
    *,
    maximum_days: int,
) -> Iterator[tuple[date, date]]:
    if end < start:
        raise ValueError("end date must not precede start date")
    if maximum_days <= 0:
        raise ValueError("maximum_days must be positive")

    cursor = start
    while cursor <= end:
        window_end = min(end, cursor + timedelta(days=maximum_days - 1))
        yield cursor, window_end
        cursor = window_end + timedelta(days=1)


def is_target_macro_event(event: str) -> bool:
    if "GDP" in event.upper():
        normalized = event.lower()
        if "gdpnow" not in normalized and "nowcast" not in normalized:
            return True
    return any(pattern.search(event) for pattern in MACRO_EVENT_PATTERNS)


def load_api_key(env_path: Path) -> str:
    existing = os.getenv("FMP_API_KEY")
    if existing:
        return existing

    if not env_path.exists():
        raise FileNotFoundError(f"environment file not found: {env_path}")

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "FMP_API_KEY":
            value = value.strip().strip("'\"")
            if value:
                return value
    raise RuntimeError(f"FMP_API_KEY is missing from {env_path}")


def read_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "data" not in payload or "_meta" not in payload:
        raise ValueError(f"invalid cached response: {path}")
    return payload


def atomic_write_json(path: Path, payload: Any, *, compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".json.gz" if compress else ".json"
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=f"{suffix}.tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        if compress:
            with gzip.open(temporary_path, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        else:
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


class FMPCollector:
    def __init__(
        self,
        *,
        api_key: str,
        output_dir: Path,
        request_interval_seconds: float,
        jitter_seconds: float,
        max_requests_per_run: int,
        timeout_seconds: float,
        workers: int,
    ) -> None:
        self.api_key = api_key
        self.output_dir = output_dir
        self.request_interval_seconds = request_interval_seconds
        self.jitter_seconds = jitter_seconds
        self.max_requests_per_run = max_requests_per_run
        self.timeout_seconds = timeout_seconds
        self.workers = workers
        self.request_count = 0
        self.last_request_started: float | None = None
        self.request_log = output_dir / "request_log.jsonl"
        self.progress_path = output_dir / "progress.json"
        self._rate_lock = threading.Lock()
        self._io_lock = threading.Lock()

    def _write_progress(
        self,
        *,
        status: str,
        endpoint: str | None = None,
        output_path: Path | None = None,
        message: str | None = None,
    ) -> None:
        payload = {
            "status": status,
            "updated_at_utc": utc_now(),
            "requests_this_run": self.request_count,
            "endpoint": endpoint,
            "output_path": (
                str(output_path.relative_to(self.output_dir))
                if output_path is not None
                else None
            ),
            "message": message,
        }
        with self._io_lock:
            atomic_write_json(self.progress_path, payload, compress=False)

    def _append_request_log(self, record: dict[str, Any]) -> None:
        with self._io_lock:
            self.request_log.parent.mkdir(parents=True, exist_ok=True)
            with self.request_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _reserve_request_slot(self) -> int:
        with self._rate_lock:
            if (
                self.max_requests_per_run
                and self.request_count >= self.max_requests_per_run
            ):
                raise RequestBudgetReached(
                    f"per-run request budget reached: {self.max_requests_per_run}"
                )
            now = time.monotonic()
            minimum_delay = self.request_interval_seconds + random.uniform(
                0.0, self.jitter_seconds
            )
            if self.last_request_started is None:
                scheduled_start = now
            else:
                scheduled_start = max(
                    now,
                    self.last_request_started + minimum_delay,
                )
            self.last_request_started = scheduled_start
            self.request_count += 1
            request_number = self.request_count
        remaining = scheduled_start - now
        if remaining > 0:
            time.sleep(remaining)
        return request_number

    def fetch(
        self,
        endpoint: str,
        params: dict[str, str | int],
    ) -> tuple[Any, dict[str, Any]]:
        query = urllib.parse.urlencode(params)
        url = f"{BASE_URL}/{endpoint}?{query}" if query else f"{BASE_URL}/{endpoint}"
        retry_delays = (15.0, 45.0, 120.0, 300.0)

        for attempt in range(len(retry_delays) + 1):
            request_number = self._reserve_request_slot()
            started_at = utc_now()
            started_clock = time.monotonic()
            request = urllib.request.Request(
                url,
                headers={
                    "apikey": self.api_key,
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "User-Agent": "market-axis-research-collector/1.0",
                },
            )

            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    downloaded_body = response.read()
                    content_encoding = response.headers.get("Content-Encoding", "")
                    if content_encoding.lower() == "gzip":
                        body = gzip.decompress(downloaded_body)
                    else:
                        body = downloaded_body
                    data = json.loads(body.decode("utf-8"))
                    if isinstance(data, dict) and any(
                        key in data for key in ("Error Message", "error", "Error")
                    ):
                        raise RuntimeError(
                            f"FMP returned an error object for {endpoint}: {data}"
                        )
                    elapsed = round(time.monotonic() - started_clock, 3)
                    metadata = {
                        "endpoint": endpoint,
                        "params": params,
                        "request_number": request_number,
                        "retrieved_at_utc": started_at,
                        "http_status": response.status,
                        "downloaded_bytes": len(downloaded_body),
                        "uncompressed_bytes": len(body),
                        "content_encoding": content_encoding or None,
                        "response_sha256": hashlib.sha256(body).hexdigest(),
                        "response_count": len(data) if isinstance(data, list) else None,
                        "elapsed_seconds": elapsed,
                    }
                    self._append_request_log(metadata)
                    return data, metadata
            except urllib.error.HTTPError as error:
                elapsed = round(time.monotonic() - started_clock, 3)
                record = {
                    "endpoint": endpoint,
                    "params": params,
                    "request_number": request_number,
                    "retrieved_at_utc": started_at,
                    "http_status": error.code,
                    "elapsed_seconds": elapsed,
                    "attempt": attempt + 1,
                }
                self._append_request_log(record)
                if error.code == 429:
                    raise RateLimitReached(
                        "FMP returned HTTP 429; stopping without another request"
                    ) from error
                if error.code not in {500, 502, 503, 504} or attempt == len(
                    retry_delays
                ):
                    raise
            except (TimeoutError, urllib.error.URLError, json.JSONDecodeError):
                if attempt == len(retry_delays):
                    raise

            time.sleep(retry_delays[attempt])

        raise AssertionError("unreachable")

    def cached_or_fetch(
        self,
        *,
        output_path: Path,
        endpoint: str,
        params: dict[str, str | int],
    ) -> tuple[Any, bool]:
        if output_path.exists():
            return read_json_gz(output_path)["data"], False

        self._write_progress(
            status="running",
            endpoint=endpoint,
            output_path=output_path,
        )
        data, metadata = self.fetch(endpoint, params)
        atomic_write_json(
            output_path,
            {"_meta": metadata, "data": data},
            compress=True,
        )
        count = len(data) if isinstance(data, list) else "object"
        print(
            f"[{metadata['request_number']}] {endpoint} -> "
            f"{output_path.relative_to(self.output_dir)} "
            f"({count})",
            flush=True,
        )
        return data, True

    def pause(self, message: str) -> None:
        self._write_progress(status="paused", message=message)

    def fail(self, message: str) -> None:
        self._write_progress(status="failed", message=message)

    def complete(self) -> None:
        self._write_progress(status="complete", message="all selected stages complete")


def write_manifest(
    *,
    output_dir: Path,
    start: date,
    end: date,
    request_interval_seconds: float,
    limit: int,
    workers: int,
) -> None:
    manifest_path = output_dir / "manifest.json"
    request_policy = {
        "concurrency": workers,
        "minimum_interval_seconds": request_interval_seconds,
    }
    expected = {
        "schema_version": 1,
        "source": "Financial Modeling Prep stable API",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "news_symbols": list(MARKET_NEWS_SYMBOLS),
        "news_page_limit": limit,
        "news_query_window": "one calendar day; paginate until an empty response",
        "price_series": {
            "^GSPC": "5min in <=10-calendar-day request windows",
            "^VIX": "EOD full",
            "ESUSD": "1min on selected macro-event dates",
        },
        "request_policy": request_policy,
        "request_policy_history": [
            {**request_policy, "effective_at_utc": utc_now()}
        ],
        "created_at_utc": utc_now(),
    }
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("schema_version", "start_date", "end_date", "news_symbols"):
            if current.get(key) != expected.get(key):
                raise RuntimeError(
                    f"existing manifest conflicts on {key}: {manifest_path}"
                )
        previous_policy = current.get("request_policy")
        history = current.setdefault("request_policy_history", [])
        if previous_policy != request_policy:
            if previous_policy and not history:
                history.append(
                    {
                        **previous_policy,
                        "effective_at_utc": current.get("created_at_utc"),
                    }
                )
            history.append({**request_policy, "effective_at_utc": utc_now()})
        current["price_series"] = expected["price_series"]
        current["news_query_window"] = expected["news_query_window"]
        current["request_policy"] = request_policy
        atomic_write_json(manifest_path, current, compress=False)
        return
    atomic_write_json(manifest_path, expected, compress=False)


def run_parallel(
    items: list[Any],
    worker: Any,
    *,
    max_workers: int,
) -> None:
    if max_workers == 1:
        for item in items:
            worker(item)
        return

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    futures = [executor.submit(worker, item) for item in items]
    try:
        for future in concurrent.futures.as_completed(futures):
            future.result()
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def collect_calendar(
    collector: FMPCollector,
    *,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    calendar_dir = collector.output_dir / "calendar"
    all_us_events: list[dict[str, Any]] = []

    for window in month_windows(start, end):
        output_path = calendar_dir / f"{window.key}.json.gz"
        data, _ = collector.cached_or_fetch(
            output_path=output_path,
            endpoint="economic-calendar",
            params={
                "from": window.start.isoformat(),
                "to": window.end.isoformat(),
            },
        )
        if not isinstance(data, list):
            raise TypeError("economic-calendar response must be a list")
        all_us_events.extend(
            row for row in data if isinstance(row, dict) and row.get("country") == "US"
        )

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_us_events:
        event_name = str(row.get("event") or "")
        event_time = str(row.get("date") or "")
        if event_time and is_target_macro_event(event_name):
            clusters[event_time].append(row)

    derived_payload = {
        "_meta": {
            "created_at_utc": utc_now(),
            "timezone": "UTC",
            "selection": "US whitelist for CPI/PCE/jobs/GDP/retail/ISM/FOMC",
            "cluster_count": len(clusters),
            "event_date_count": len({timestamp[:10] for timestamp in clusters}),
        },
        "clusters": [
            {"event_time_utc": timestamp, "releases": clusters[timestamp]}
            for timestamp in sorted(clusters)
        ],
    }
    atomic_write_json(
        collector.output_dir / "derived" / "macro_event_clusters.json",
        derived_payload,
        compress=False,
    )
    return derived_payload["clusters"]


def collect_paginated_news(
    collector: FMPCollector,
    *,
    start: date,
    end: date,
    limit: int,
    feed_name: str,
    endpoint: str,
    extra_params: dict[str, str],
) -> None:
    feed_dir = collector.output_dir / "news" / feed_name

    def collect_day(news_date: date) -> None:
        page = 0
        while True:
            output_path = (
                feed_dir
                / f"{news_date.year:04d}"
                / f"{news_date.month:02d}"
                / f"{news_date.day:02d}"
                / f"page_{page:04d}.json.gz"
            )
            params: dict[str, str | int] = {
                **extra_params,
                "from": news_date.isoformat(),
                "to": news_date.isoformat(),
                "page": page,
                "limit": limit,
            }
            data, _ = collector.cached_or_fetch(
                output_path=output_path,
                endpoint=endpoint,
                params=params,
            )
            if not isinstance(data, list):
                raise TypeError(f"{endpoint} response must be a list")
            if not data:
                break
            page += 1
            if page > 100:
                raise RuntimeError(
                    f"pagination safety limit reached for {news_date.isoformat()}"
                )

    run_parallel(
        [
            start + timedelta(days=offset)
            for offset in range((end - start).days + 1)
        ],
        collect_day,
        max_workers=collector.workers,
    )


def collect_market_prices(
    collector: FMPCollector,
    *,
    start: date,
    end: date,
    macro_clusters: list[dict[str, Any]],
) -> None:
    gspc_dir = collector.output_dir / "prices" / "gspc_5min"
    gspc_windows = list(day_windows(start, end, maximum_days=10))

    def collect_gspc_window(window: tuple[date, date]) -> None:
        window_start, window_end = window
        output_path = (
            gspc_dir
            / f"{window_start.isoformat()}_{window_end.isoformat()}.json.gz"
        )
        data, _ = collector.cached_or_fetch(
            output_path=output_path,
            endpoint="historical-chart/5min",
            params={
                "symbol": "^GSPC",
                "from": window_start.isoformat(),
                "to": window_end.isoformat(),
            },
        )
        if not isinstance(data, list):
            raise TypeError("^GSPC 5-minute response must be a list")

    run_parallel(
        gspc_windows,
        collect_gspc_window,
        max_workers=collector.workers,
    )

    vix_path = (
        collector.output_dir
        / "prices"
        / "vix_eod"
        / f"{start.isoformat()}_{end.isoformat()}.json.gz"
    )
    vix_data, _ = collector.cached_or_fetch(
        output_path=vix_path,
        endpoint="historical-price-eod/full",
        params={
            "symbol": "^VIX",
            "from": start.isoformat(),
            "to": end.isoformat(),
        },
    )
    if not isinstance(vix_data, list):
        raise TypeError("^VIX EOD response must be a list")

    event_dates = sorted(
        {
            str(cluster["event_time_utc"])[:10]
            for cluster in macro_clusters
            if start.isoformat() <= str(cluster["event_time_utc"])[:10] <= end.isoformat()
        }
    )
    es_dir = collector.output_dir / "prices" / "esusd_1min"

    def collect_es_event(event_date: str) -> None:
        event_day = date.fromisoformat(event_date)
        output_path = (
            es_dir
            / f"{event_day.year:04d}"
            / f"{event_day.month:02d}"
            / f"{event_date}.json.gz"
        )
        data, _ = collector.cached_or_fetch(
            output_path=output_path,
            endpoint="historical-chart/1min",
            params={
                "symbol": "ESUSD",
                "from": event_date,
                "to": event_date,
            },
        )
        if not isinstance(data, list):
            raise TypeError("ESUSD 1-minute response must be a list")

    run_parallel(
        event_dates,
        collect_es_event,
        max_workers=collector.workers,
    )


def build_summary(output_dir: Path) -> dict[str, Any]:
    groups: dict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "rows": 0, "downloaded_bytes": 0}
    )
    for path in output_dir.rglob("*.json.gz"):
        payload = read_json_gz(path)
        relative = path.relative_to(output_dir)
        group = (
            "/".join(relative.parts[:2])
            if relative.parts[0] in {"news", "prices"}
            else relative.parts[0]
        )
        metadata = payload["_meta"]
        data = payload["data"]
        groups[group]["files"] += 1
        groups[group]["rows"] += len(data) if isinstance(data, list) else 0
        groups[group]["downloaded_bytes"] += int(
            metadata.get("downloaded_bytes") or 0
        )

    summary = {
        "created_at_utc": utc_now(),
        "groups": dict(sorted(groups.items())),
        "total_files": sum(group["files"] for group in groups.values()),
        "total_rows": sum(group["rows"] for group in groups.values()),
        "total_downloaded_bytes": sum(
            group["downloaded_bytes"] for group in groups.values()
        ),
    }
    atomic_write_json(output_dir / "summary.json", summary, compress=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=parse_iso_date, default=DEFAULT_START)
    parser.add_argument("--end-date", type=parse_iso_date, default=DEFAULT_END)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--request-interval-seconds", type=float, default=0.6)
    parser.add_argument("--jitter-seconds", type=float, default=0.0)
    parser.add_argument(
        "--max-requests-per-run",
        type=int,
        default=0,
        help="Zero means no per-run cap; HTTP 429 always stops immediately.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--stages",
        default="calendar,market,news",
        help="Comma-separated subset of calendar,market,news.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    unknown_stages = stages - {"calendar", "market", "news"}
    if unknown_stages:
        raise ValueError(f"unknown stages: {sorted(unknown_stages)}")
    if args.end_date < args.start_date:
        raise ValueError("--end-date must not precede --start-date")
    if args.request_interval_seconds < 0 or args.jitter_seconds < 0:
        raise ValueError("request intervals must be nonnegative")
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not 1 <= args.workers <= 4:
        raise ValueError("--workers must be between 1 and 4")

    output_dir = args.output_dir or Path(
        "data/fmp_raw"
    ) / f"{args.start_date.isoformat()}_{args.end_date.isoformat()}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        output_dir=output_dir,
        start=args.start_date,
        end=args.end_date,
        request_interval_seconds=args.request_interval_seconds,
        limit=args.limit,
        workers=args.workers,
    )

    api_key = load_api_key(args.env_file)
    collector = FMPCollector(
        api_key=api_key,
        output_dir=output_dir,
        request_interval_seconds=args.request_interval_seconds,
        jitter_seconds=args.jitter_seconds,
        max_requests_per_run=args.max_requests_per_run,
        timeout_seconds=args.timeout_seconds,
        workers=args.workers,
    )

    try:
        macro_clusters: list[dict[str, Any]]
        if "calendar" in stages or "market" in stages:
            macro_clusters = collect_calendar(
                collector,
                start=args.start_date,
                end=args.end_date,
            )
        else:
            derived_path = output_dir / "derived" / "macro_event_clusters.json"
            if not derived_path.exists():
                raise RuntimeError("market stage requires completed calendar data")
            macro_clusters = json.loads(
                derived_path.read_text(encoding="utf-8")
            )["clusters"]

        if "market" in stages:
            collect_market_prices(
                collector,
                start=args.start_date,
                end=args.end_date,
                macro_clusters=macro_clusters,
            )

        if "news" in stages:
            collect_paginated_news(
                collector,
                start=args.start_date,
                end=args.end_date,
                limit=args.limit,
                feed_name="general",
                endpoint="news/general-latest",
                extra_params={},
            )
            collect_paginated_news(
                collector,
                start=args.start_date,
                end=args.end_date,
                limit=args.limit,
                feed_name="market_etfs",
                endpoint="news/stock",
                extra_params={"symbols": ",".join(MARKET_NEWS_SYMBOLS)},
            )
    except (RequestBudgetReached, RateLimitReached) as error:
        collector.pause(str(error))
        build_summary(output_dir)
        print(f"PAUSED: {error}", file=sys.stderr, flush=True)
        return 75
    except KeyboardInterrupt:
        collector.pause("interrupted by operator; safe to resume")
        build_summary(output_dir)
        print("PAUSED: interrupted by operator; safe to resume", file=sys.stderr)
        return 130
    except Exception as error:
        collector.fail(f"{type(error).__name__}: {error}")
        build_summary(output_dir)
        raise

    collector.complete()
    summary = build_summary(output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
