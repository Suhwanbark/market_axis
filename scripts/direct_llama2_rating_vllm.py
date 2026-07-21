#!/usr/bin/env python3
"""Two-replica vLLM acceleration for the same-Llama direct rating.

Run ``prepare`` once, then launch ``worker --rank 0`` on GPU 0 and rank 1 on
GPU 1 concurrently.  Each worker owns a full 7B replica at the requested GPU
memory utilization and writes only to its own /tmp result arrays.  ``merge``
fills the canonical resumable arrays created by ``direct_llama2_rating.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import direct_llama2_rating as base
import news_cutoff_safe_llama2 as protocol


ROOT = Path(__file__).resolve().parent.parent
TMP = Path("/tmp/market_axis_direct_vllm")
TASKS = TMP / "tasks.parquet"
MODEL_PATH_DEFAULT = "/tmp/market_axis_llama2_7b"
SPLITS = ("train", "val", "test")
DOMAINS = ("news", "sec8k")


def canonical_start(domain: str, split: str) -> int:
    if base.done_path(domain, split).exists():
        return len(base.document_frame(domain, split))
    checkpoint = base.checkpoint_path(domain, split)
    if checkpoint.exists():
        return int(json.loads(checkpoint.read_text())["next_order_index"])
    return 0


def prepare(world_size: int) -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    # Stage the small frozen document tables locally; model weights already live in /tmp.
    shutil.copy2(base.NEWS_DOCS, TMP / "news_documents.parquet")
    shutil.copy2(base.SEC_DOCS, TMP / "sec8k_documents.parquet")
    rows = []
    task_id = 0
    for domain in DOMAINS:
        max_tokens = 512 if domain == "news" else 2048
        for split in SPLITS:
            frame = base.document_frame(domain, split)
            order = np.argsort(frame["text_chars"].to_numpy(), kind="stable")
            start = canonical_start(domain, split)
            for order_index in range(start, len(order)):
                position = int(order[order_index])
                chars = int(frame.iloc[position]["text_chars"])
                # Approximate token work for balanced assignment; cap at the protocol limit.
                cost = min(chars / 3.5, max_tokens) + 100
                rows.append(
                    {
                        "task_id": task_id,
                        "domain": domain,
                        "split": split,
                        "position": position,
                        "order_index": order_index,
                        "cost": cost,
                    }
                )
                task_id += 1
    tasks = pd.DataFrame(rows)
    if tasks.empty:
        print("no remaining direct-rating tasks", flush=True)
        tasks.to_parquet(TASKS, index=False)
        return
    # Longest-processing-time greedy partition gives near-identical token loads.
    loads = [0.0] * world_size
    assignments = np.empty(len(tasks), dtype=np.int16)
    for index in tasks.sort_values("cost", ascending=False).index:
        rank = int(np.argmin(loads))
        assignments[index] = rank
        loads[rank] += float(tasks.at[index, "cost"])
    tasks["rank"] = assignments
    tasks.to_parquet(TASKS, index=False)
    protocol.write_json(
        TMP / "task_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "world_size": world_size,
            "tasks": len(tasks),
            "estimated_loads": loads,
            "counts": {
                f"rank{rank}/{domain}/{split}": int(count)
                for (rank, domain, split), count in tasks.groupby(
                    ["rank", "domain", "split"]
                ).size().items()
            },
            "canonical_starts": {
                f"{domain}/{split}": canonical_start(domain, split)
                for domain in DOMAINS
                for split in SPLITS
            },
        },
    )
    print(tasks.groupby(["rank", "domain", "split"]).size(), flush=True)
    print("estimated loads", loads, flush=True)


def local_document_frame(domain: str, split: str) -> pd.DataFrame:
    path = TMP / f"{domain}_documents.parquet"
    frame = pd.read_parquet(path)
    part = protocol.split_documents(frame, split)
    text_column = "news_text" if domain == "news" else "text"
    return part[[text_column, "text_chars"]].rename(columns={text_column: "text"})


def result_paths(rank: int) -> tuple[Path, Path, Path]:
    return (
        TMP / f"rank{rank}_scores.npy",
        TMP / f"rank{rank}_probabilities.npy",
        TMP / f"rank{rank}_checkpoint.json",
    )


def open_result_arrays(rank: int, rows: int):
    score_path, probability_path, checkpoint_path = result_paths(rank)
    if score_path.exists() or probability_path.exists():
        if score_path.exists() and probability_path.exists() and checkpoint_path.exists():
            scores = np.load(score_path, mmap_mode="r+")
            probabilities = np.load(probability_path, mmap_mode="r+")
            if scores.shape != (rows,) or probabilities.shape != (rows, 9):
                raise RuntimeError(f"rank {rank} checkpoint shape mismatch")
            start = int(json.loads(checkpoint_path.read_text())["next_index"])
            return scores, probabilities, start
        # A failure before the first inference chunk may leave empty memmaps but
        # no checkpoint. Recreate those rank-local temporary arrays safely.
    scores = np.lib.format.open_memmap(score_path, mode="w+", dtype=np.float32, shape=(rows,))
    probabilities = np.lib.format.open_memmap(
        probability_path, mode="w+", dtype=np.float16, shape=(rows, 9)
    )
    return scores, probabilities, 0


def make_prompt_ids(tokenizer, frame: pd.DataFrame, positions: np.ndarray, max_tokens: int, boundary_id: int):
    raw = frame.iloc[positions]["text"].tolist()
    document_ids = tokenizer(
        raw,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )["input_ids"]
    prompts = []
    for ids in document_ids:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        encoded = tokenizer.encode(base.PROMPT.format(text=text), add_special_tokens=True)
        prompts.append({"prompt_token_ids": [*encoded, boundary_id]})
    return prompts


def extract_probabilities(outputs, digit_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.empty((len(outputs), 9), dtype=np.float32)
    for row, output in enumerate(outputs):
        sample = output.outputs[0]
        if not sample.logprobs or not sample.logprobs[0]:
            raise RuntimeError("vLLM did not return next-token log probabilities")
        values = sample.logprobs[0]
        for column, token_id in enumerate(digit_ids):
            if token_id not in values:
                raise RuntimeError(f"allowed digit {token_id} missing from vLLM logprobs")
            probabilities[row, column] = math.exp(float(values[token_id].logprob))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    scores = probabilities @ np.arange(1, 10, dtype=np.float32)
    return scores, probabilities


def worker(
    rank: int,
    model_path: str,
    gpu_memory_utilization: float,
    chunk_size: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
) -> None:
    from transformers import LlamaTokenizerFast
    from vllm import LLM, SamplingParams

    tasks = pd.read_parquet(TASKS)
    tasks = tasks.loc[tasks["rank"].eq(rank)].sort_values("task_id").reset_index(drop=True)
    scores, probabilities, start = open_result_arrays(rank, len(tasks))
    tokenizer = LlamaTokenizerFast(
        tokenizer_file=str(Path(model_path) / "tokenizer.json"), legacy=False
    )
    digit_ids = [tokenizer.convert_tokens_to_ids(str(value)) for value in range(1, 10)]
    boundary_id = tokenizer.convert_tokens_to_ids("▁")
    if len(set(digit_ids)) != 9 or any(value == tokenizer.unk_token_id for value in digit_ids):
        raise RuntimeError(f"invalid Llama digit ids: {digit_ids}")
    engine = LLM(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2560,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        # vLLM otherwise reports top log-probabilities from the raw full
        # vocabulary before applying ``allowed_token_ids``.  We need the
        # normalized distribution after restricting the vocabulary to 1..9.
        logprobs_mode="processed_logprobs",
        trust_remote_code=False,
        seed=20260721 + rank,
    )
    sampling = SamplingParams(
        # Greedy sampling is sufficient because only the complete processed
        # 1..9 distribution is consumed; it also avoids needless RNG work.
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_tokens=1,
        logprobs=9,
        allowed_token_ids=digit_ids,
        ignore_eos=True,
        detokenize=False,
        seed=20260721,
    )
    frame_cache = {
        (domain, split): local_document_frame(domain, split)
        for domain in DOMAINS
        for split in SPLITS
        if ((tasks["domain"] == domain) & (tasks["split"] == split)).any()
    }
    while start < len(tasks):
        end = min(start + chunk_size, len(tasks))
        chunk = tasks.iloc[start:end].reset_index(drop=True)
        # Batch tokenization by source table.  The old one-row tokenizer calls
        # left both GPUs idle for minutes before every inference chunk.
        prompts = [None] * len(chunk)
        for (domain, split), group in chunk.groupby(["domain", "split"], sort=False):
            local_indices = group.index.to_numpy(dtype=int)
            positions = group["position"].to_numpy(dtype=int)
            group_prompts = make_prompt_ids(
                tokenizer,
                frame_cache[(domain, split)],
                positions,
                512 if domain == "news" else 2048,
                boundary_id,
            )
            for local_index, prompt in zip(local_indices, group_prompts):
                prompts[local_index] = prompt
        if any(prompt is None for prompt in prompts):
            raise RuntimeError("failed to construct every prompt in the inference chunk")
        outputs = engine.generate(prompts, sampling, use_tqdm=True)
        chunk_scores, chunk_probabilities = extract_probabilities(outputs, digit_ids)
        scores[start:end] = chunk_scores
        probabilities[start:end] = chunk_probabilities.astype(np.float16)
        scores.flush()
        probabilities.flush()
        protocol.write_json(
            result_paths(rank)[2],
            {
                "rank": rank,
                "next_index": end,
                "rows": len(tasks),
                "gpu_memory_utilization": gpu_memory_utilization,
                "chunk_size": chunk_size,
            },
        )
        print(f"vLLM rank {rank}: {end}/{len(tasks)}", flush=True)
        start = end
    protocol.write_json(
        TMP / f"rank{rank}.done.json",
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "rank": rank,
            "rows": len(tasks),
            "model_path": model_path,
            "model_id": base.MODEL_ID,
            "revision": base.REVISION,
            "gpu_memory_utilization": gpu_memory_utilization,
            "engine": "vLLM single-GPU replica",
        },
    )
    del engine
    gc.collect()


def merge(world_size: int) -> None:
    tasks = pd.read_parquet(TASKS)
    for rank in range(world_size):
        if not (TMP / f"rank{rank}.done.json").exists():
            raise RuntimeError(f"vLLM rank {rank} is incomplete")
    for domain in DOMAINS:
        for split in SPLITS:
            frame = base.document_frame(domain, split)
            score, probabilities, _ = base.open_arrays(domain, split, len(frame))
            selected = tasks.loc[
                tasks["domain"].eq(domain) & tasks["split"].eq(split)
            ]
            for rank, part in selected.groupby("rank"):
                rank_tasks = tasks.loc[tasks["rank"].eq(rank)].sort_values("task_id").reset_index()
                task_to_local = pd.Series(rank_tasks.index.to_numpy(), index=rank_tasks["task_id"])
                local_rows = part["task_id"].map(task_to_local).to_numpy(int)
                rank_scores = np.load(result_paths(int(rank))[0], mmap_mode="r")
                rank_probabilities = np.load(result_paths(int(rank))[1], mmap_mode="r")
                positions = part["position"].to_numpy(int)
                score[positions] = rank_scores[local_rows]
                probabilities[positions] = rank_probabilities[local_rows]
            score.flush()
            probabilities.flush()
            if not np.isfinite(score).all():
                raise RuntimeError(f"non-finite merged score for {domain}/{split}")
            protocol.write_json(
                base.checkpoint_path(domain, split),
                {"next_order_index": len(frame), "rows": len(frame), "engine": "vLLM data parallel"},
            )
            protocol.write_json(
                base.done_path(domain, split),
                {
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "model_id": base.MODEL_ID,
                    "revision": base.REVISION,
                    "model_path": MODEL_PATH_DEFAULT,
                    "domain": domain,
                    "split": split,
                    "rows": len(frame),
                    "prompt": base.PROMPT,
                    "score": "expected value of vLLM logits restricted to digit tokens 1..9 after common boundary",
                    "same_checkpoint_as_activation": True,
                    "engine": "two concurrent single-GPU vLLM replicas",
                    "gpu_memory_utilization": 0.85,
                    "max_document_tokens": 512 if domain == "news" else 2048,
                },
            )
            print(f"merged {domain}/{split}: {len(frame)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--world-size", type=int, default=2)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--rank", type=int, required=True)
    worker_parser.add_argument("--model-path", default=MODEL_PATH_DEFAULT)
    worker_parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    worker_parser.add_argument("--chunk-size", type=int, default=4096)
    worker_parser.add_argument("--max-num-seqs", type=int, default=128)
    worker_parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--world-size", type=int, default=2)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.world_size)
    elif args.command == "worker":
        worker(
            args.rank,
            args.model_path,
            args.gpu_memory_utilization,
            args.chunk_size,
            args.max_num_seqs,
            args.max_num_batched_tokens,
        )
    else:
        merge(args.world_size)


if __name__ == "__main__":
    main()
