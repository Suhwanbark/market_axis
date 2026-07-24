#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/felab/workspace/axis
cd "$ROOT"

LOG_DIR=outputs/exact_article_baselines/logs
mkdir -p "$LOG_DIR"

wait_for_files() {
  while true; do
    local ready=1
    for target in "$@"; do
      [[ -f "$target" ]] || ready=0
    done
    [[ "$ready" -eq 1 ]] && return 0
    sleep 20
  done
}

wait_for_files \
  outputs/exact_article_baselines/direct_rating/qwen25_7b/shard_00_of_02/manifest.json \
  outputs/exact_article_baselines/direct_rating/qwen25_7b/shard_01_of_02/manifest.json \
  outputs/exact_article_baselines/direct_rating/qwen3_4b/shard_00_of_02/manifest.json \
  outputs/exact_article_baselines/direct_rating/qwen3_4b/shard_01_of_02/manifest.json

bash scripts/bootstrap_tmp_runtime.sh 2>&1 | tee "$LOG_DIR/embedding_bootstrap.log"

launch_embedding() {
  local session_name=$1
  local physical_gpu=$2
  local model_name=$3
  local shard_index=$4
  local batch_size=$5
  local short_name=$6
  local manifest="outputs/exact_article_baselines/embeddings/${model_name}/prefixless/shard_0${shard_index}_of_02/manifest.json"
  if [[ -f "$manifest" ]]; then
    return 0
  fi
  if tmux has-session -t "$session_name" 2>/dev/null; then
    return 0
  fi
  tmux new-session -d -s "$session_name" \
    "bash -lc 'set -o pipefail; cd $ROOT && env CUDA_VISIBLE_DEVICES=$physical_gpu PYTHONUNBUFFERED=1 /tmp/axis_conda_sh/bin/python scripts/extract_exact_article_embeddings.py --name $model_name --variant prefixless --device 0 --shard-index $shard_index --num-shards 2 --batch-size $batch_size --max-length 2048 2>&1 | tee $LOG_DIR/${short_name}.log'"
}

launch_embedding exact_emb8_s0 0 qwen3_embedding_8b 0 2 emb8_s0
launch_embedding exact_emb8_s1 1 qwen3_embedding_8b 1 2 emb8_s1
launch_embedding exact_emb06_s0 2 qwen3_embedding_06b 0 8 emb06_s0
launch_embedding exact_emb06_s1 3 qwen3_embedding_06b 1 8 emb06_s1

wait_for_files \
  outputs/exact_article_baselines/embeddings/qwen3_embedding_8b/prefixless/shard_00_of_02/manifest.json \
  outputs/exact_article_baselines/embeddings/qwen3_embedding_8b/prefixless/shard_01_of_02/manifest.json \
  outputs/exact_article_baselines/embeddings/qwen3_embedding_06b/prefixless/shard_00_of_02/manifest.json \
  outputs/exact_article_baselines/embeddings/qwen3_embedding_06b/prefixless/shard_01_of_02/manifest.json

env PYTHONUNBUFFERED=1 /tmp/axis_conda_sh/bin/python \
  scripts/evaluate_exact_article_baselines.py \
  --repeats 2000 2>&1 | tee "$LOG_DIR/evaluation.log"

echo EXACT_ARTICLE_BASELINE_COORDINATOR_COMPLETE
