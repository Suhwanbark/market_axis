#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MODEL_PATH="${QWEN3_EMBEDDING_MODEL:-$ROOT/models/qwen3-embedding-8b}"
REVISION="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"

export PYTHONPATH="$ROOT/scripts:$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-64}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-64}"

cd "$ROOT"

"$PYTHON" scripts/news_large_embedding_control.py extract \
  --split train --model-path "$MODEL_PATH" --revision "$REVISION" --batch-size 16
"$PYTHON" scripts/news_large_embedding_control.py extract \
  --split val --model-path "$MODEL_PATH" --revision "$REVISION" --batch-size 16
"$PYTHON" scripts/news_large_embedding_control.py select --workers 6
"$PYTHON" scripts/news_large_embedding_control.py extract \
  --split test --model-path "$MODEL_PATH" --revision "$REVISION" --batch-size 16
"$PYTHON" scripts/news_large_embedding_control.py evaluate \
  --bootstrap-repetitions 2000 --workers 64
"$PYTHON" scripts/news_capacity_matched_forecast.py
