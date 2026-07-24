#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
BGE_PATH="${BGE_MODEL:-/tmp/market_axis_bge_m3}"
QWEN_PATH="${QWEN_MODEL:-/tmp/market_axis_qwen3_embedding_8b}"
LLAMA_PATH="${LLAMA2_MODEL:-/tmp/market_axis_llama2_7b}"

export PYTHONPATH="$ROOT/scripts:$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-64}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-64}"

cd "$ROOT"
"$PYTHON" scripts/sec8k_uncentered_prepare.py

for representation in bge qwen3 llama2; do
  case "$representation" in
    bge) model_path="$BGE_PATH"; batch=16 ;;
    qwen3) model_path="$QWEN_PATH"; batch=8 ;;
    llama2) model_path="$LLAMA_PATH"; batch=8 ;;
  esac
  for split in train val; do
    "$PYTHON" scripts/sec8k_uncentered_axis.py extract \
      --representation "$representation" --split "$split" \
      --model-path "$model_path" --batch-size "$batch" --max-length 2048
  done
done

"$PYTHON" scripts/sec8k_uncentered_axis.py select --workers 6

for representation in bge qwen3 llama2; do
  case "$representation" in
    bge) model_path="$BGE_PATH"; batch=16 ;;
    qwen3) model_path="$QWEN_PATH"; batch=8 ;;
    llama2) model_path="$LLAMA_PATH"; batch=8 ;;
  esac
  "$PYTHON" scripts/sec8k_uncentered_axis.py extract \
    --representation "$representation" --split test \
    --model-path "$model_path" --batch-size "$batch" --max-length 2048
done

"$PYTHON" scripts/sec8k_uncentered_axis.py evaluate --bootstrap-repetitions 2000 --workers 64
"$PYTHON" scripts/sec8k_uncentered_mlp.py train
"$PYTHON" scripts/sec8k_uncentered_mlp.py evaluate --bootstrap-repetitions 2000 --workers 64
"$PYTHON" scripts/sec8k_uncentered_forecast.py
