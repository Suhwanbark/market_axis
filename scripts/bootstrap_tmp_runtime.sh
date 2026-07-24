#!/usr/bin/env bash
set -euo pipefail

AXIS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSISTENT_ENV="/home/felab/.conda/envs/sh"
LOCAL_ENV="/tmp/axis_conda_sh"

copy_model_if_needed() {
  local source_dir="$1"
  local target_dir="$2"
  local model_name="$3"

  if [[ ! -d "$source_dir" ]]; then
    echo "ERROR: persistent source for ${model_name} is missing: ${source_dir}" >&2
    return 1
  fi

  local source_bytes target_bytes
  source_bytes="$(find -L "$source_dir" -type f -printf '%s\n' | awk '{s += $1} END {printf "%.0f", s}')"
  target_bytes="0"
  if [[ -d "$target_dir" ]]; then
    target_bytes="$(find -L "$target_dir" -type f -printf '%s\n' | awk '{s += $1} END {printf "%.0f", s}')"
  fi

  if [[ -f "$target_dir/config.json" ]] && \
     find "$target_dir" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q . && \
     [[ "$source_bytes" == "$target_bytes" ]]; then
    echo "READY model=${model_name} path=${target_dir} bytes=${target_bytes}"
    return 0
  fi

  echo "COPY model=${model_name} source=${source_dir} target=${target_dir}"
  mkdir -p "$target_dir"
  # -L is required for Hugging Face snapshot symlinks: /tmp must contain the
  # actual weight files rather than links back to the slow persistent cache.
  cp -auL "$source_dir"/. "$target_dir"/

  target_bytes="$(find -L "$target_dir" -type f -printf '%s\n' | awk '{s += $1} END {printf "%.0f", s}')"
  # An interrupted cp can leave a short file with a newer mtime than its
  # source.  In that case `cp -u` will keep skipping the corrupt target on
  # every rerun.  Repair the copy in place rather than requiring callers to
  # delete the resumable target directory.
  if [[ "$source_bytes" != "$target_bytes" ]]; then
    echo "REPAIR model=${model_name} source_bytes=${source_bytes} target_bytes=${target_bytes}"
    cp -aL "$source_dir"/. "$target_dir"/
    target_bytes="$(find -L "$target_dir" -type f -printf '%s\n' | awk '{s += $1} END {printf "%.0f", s}')"
  fi
  if [[ ! -f "$target_dir/config.json" ]] || \
     ! find "$target_dir" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q . || \
     [[ "$source_bytes" != "$target_bytes" ]]; then
    echo "ERROR: incomplete local model copy for ${model_name}: source=${source_bytes}, target=${target_bytes}" >&2
    return 1
  fi
  echo "READY model=${model_name} path=${target_dir} bytes=${target_bytes}"
}

copy_env_if_needed() {
  if [[ -x "$LOCAL_ENV/bin/python" ]] && \
     "$LOCAL_ENV/bin/python" -c 'import torch, transformers, vllm' >/dev/null 2>&1; then
    echo "READY conda=${LOCAL_ENV}"
    return 0
  fi

  if [[ ! -d "$PERSISTENT_ENV" ]]; then
    echo "ERROR: persistent conda environment is missing: ${PERSISTENT_ENV}" >&2
    return 1
  fi

  echo "COPY conda source=${PERSISTENT_ENV} target=${LOCAL_ENV}"
  mkdir -p "$LOCAL_ENV"
  # Preserve environment symlinks.  Following every link with `-L` makes an
  # otherwise valid copy fail on optional, broken CLI links unrelated to the
  # Python runtime.  A first interrupted copy can also leave a short regular
  # file with a newer mtime, so repair size mismatches explicitly.
  cp -au "$PERSISTENT_ENV"/. "$LOCAL_ENV"/
  while IFS= read -r -d '' source_file; do
    relative_path="${source_file#"$PERSISTENT_ENV"/}"
    target_file="$LOCAL_ENV/$relative_path"
    if [[ ! -f "$target_file" ]] || \
       [[ "$(stat -c '%s' "$source_file")" != "$(stat -c '%s' "$target_file")" ]]; then
      mkdir -p "$(dirname "$target_file")"
      cp -a "$source_file" "$target_file"
    fi
  done < <(find "$PERSISTENT_ENV" -type f -print0)
  "$LOCAL_ENV/bin/python" -c 'import torch, transformers, vllm; print("READY conda imports=torch,transformers,vllm")'
}

copy_model_if_needed \
  "/home/felab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28" \
  "/tmp/axis_qwen25_7b" \
  "Qwen2.5-7B-Instruct"

copy_model_if_needed \
  "/home/felab/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c" \
  "/tmp/axis_qwen3_4b" \
  "Qwen3-4B"

copy_model_if_needed \
  "$AXIS_ROOT/models/qwen3-embedding-0.6b" \
  "/tmp/market_axis_qwen3_embedding_06b" \
  "Qwen3-Embedding-0.6B"

copy_model_if_needed \
  "$AXIS_ROOT/models/qwen3-embedding-8b" \
  "/tmp/market_axis_qwen3_embedding_8b" \
  "Qwen3-Embedding-8B"

copy_env_if_needed

echo "RUNTIME_BOOTSTRAP_OK"
