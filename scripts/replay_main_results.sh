#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/scripts:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

python scripts/news_ticker_centered_parkinson_axis_tuned.py
python scripts/news_ticker_centered_parkinson_forecast_tuned.py
python scripts/sec8k_simple_label_family_benchmark.py
python scripts/sec8k_market_adjusted_parkinson_forecast.py

python scripts/verify_main_results.py
