#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
LLAMA_PATH="${LLAMA2_MODEL:-$ROOT/models/llama2-7b-base}"
BGE_PATH="${BGE_MODEL:-$ROOT/models/bge-m3}"
LLAMA_REVISION="8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
BGE_REVISION="5617a9f61b028005a4858fdac845db406aefb181"

export PYTHONPATH="$ROOT/scripts:$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-64}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-64}"

cd "$ROOT"

"$PYTHON" scripts/news_cutoff_safe_llama2.py prepare

# Pre-test representations only.
"$PYTHON" scripts/news_cutoff_safe_llama2.py extract \
  --representation bge --split train --model-path "$BGE_PATH" \
  --model-revision "$BGE_REVISION" --batch-size 32
"$PYTHON" scripts/news_cutoff_safe_llama2.py extract \
  --representation bge --split val --model-path "$BGE_PATH" \
  --model-revision "$BGE_REVISION" --batch-size 64
"$PYTHON" scripts/news_cutoff_safe_llama2.py extract \
  --representation llama2 --split train --model-path "$LLAMA_PATH" \
  --model-revision "$LLAMA_REVISION" --batch-size 32
"$PYTHON" scripts/news_cutoff_safe_llama2.py extract \
  --representation llama2 --split val --model-path "$LLAMA_PATH" \
  --model-revision "$LLAMA_REVISION" --batch-size 32

# Freeze BGE alpha and Llama layer/alpha before test representations exist.
"$PYTHON" scripts/news_cutoff_safe_llama2.py select --workers 6 --ridge-device auto

# Open the 2023 test only after the selection file has been written.
"$PYTHON" scripts/news_cutoff_safe_llama2.py extract \
  --representation bge --split test --model-path "$BGE_PATH" \
  --model-revision "$BGE_REVISION" --batch-size 64
"$PYTHON" scripts/news_cutoff_safe_llama2.py extract \
  --representation llama2 --split test --model-path "$LLAMA_PATH" \
  --model-revision "$LLAMA_REVISION" --batch-size 32
"$PYTHON" scripts/news_cutoff_safe_llama2.py evaluate \
  --bootstrap-repetitions 2000 --workers 64
"$PYTHON" scripts/news_cutoff_safe_forecast.py
