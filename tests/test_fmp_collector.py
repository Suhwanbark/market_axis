from __future__ import annotations

import sys
from datetime import date

from scripts.collect_fmp_research_data import (
    day_windows,
    is_target_macro_event,
    month_windows,
    parse_args,
)


def test_month_windows_clip_boundaries() -> None:
    observed = list(month_windows(date(2024, 1, 15), date(2024, 3, 2)))
    assert [(window.key, window.start, window.end) for window in observed] == [
        ("2024-01", date(2024, 1, 15), date(2024, 1, 31)),
        ("2024-02", date(2024, 2, 1), date(2024, 2, 29)),
        ("2024-03", date(2024, 3, 1), date(2024, 3, 2)),
    ]


def test_macro_event_whitelist() -> None:
    assert is_target_macro_event("Core CPI MoM (Dec)")
    assert is_target_macro_event("Non Farm Payrolls (Dec)")
    assert is_target_macro_event("ISM Services PMI (Jun)")
    assert is_target_macro_event("GDP Growth Rate QoQ Adv (Q1)")
    assert is_target_macro_event("Fed Interest Rate Decision")


def test_macro_event_whitelist_excludes_nowcasts_and_noise() -> None:
    assert not is_target_macro_event("Atlanta Fed GDPNow (Q4)")
    assert not is_target_macro_event("FOMC Minutes")
    assert not is_target_macro_event("CFTC Corn speculative net positions")


def test_day_windows_use_inclusive_maximum_span() -> None:
    observed = list(
        day_windows(date(2024, 1, 1), date(2024, 1, 25), maximum_days=10)
    )
    assert observed == [
        (date(2024, 1, 1), date(2024, 1, 10)),
        (date(2024, 1, 11), date(2024, 1, 20)),
        (date(2024, 1, 21), date(2024, 1, 25)),
    ]


def test_default_request_policy_targets_100_requests_per_minute(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["collect_fmp_research_data.py"])
    args = parse_args()
    assert args.request_interval_seconds == 0.6
    assert args.jitter_seconds == 0.0
    assert args.workers == 2
