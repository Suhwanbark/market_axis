#!/usr/bin/env python3
"""Checkpointed, market-label-free extraction for retrieval-conditioned news.

Tasks
-----
expectations
    Pre-news A/B/C/D distributions and all-layer residual states for the real
    firm.  Unknown-company, popularity-matched shuffled-company, and exact
    subject-attention-block controls are stored as logits/divergences.
controlled
    All-layer states for matched favorable/continuity/adverse synthetic reports.
news
    All-layer states and A/B/C/D logits for real corporate-event articles.
profiles
    Two deterministic eight-relation free-recall profiles per observed firm.

No task imports, reads, or derives a market outcome.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import extract_financial_recall_revision as recall_runtime


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/retrieval_conditioned_memory"
DEFAULT_OUTPUT = ROOT / "outputs/retrieval_conditioned_memory"
MODEL_SPECS = recall_runtime.MODEL_SPECS
TASK_SOURCES = {
    "expectations": DATA / "expectations.parquet",
    "controlled": DATA / "controlled.parquet",
    "news": DATA / "news_relations.parquet",
    "profiles": DATA / "profiles.parquet",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def option_ids(tokenizer: Any, torch: Any, device: int) -> Any:
    ids: list[int] = []
    for letter in "ABCD":
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"option {letter!r} is not one token: {encoded}")
        ids.append(int(encoded[0]))
    return torch.tensor(ids, dtype=torch.long, device=f"cuda:{device}")


def open_arrays(
    output_dir: Path,
    specs: dict[str, tuple[np.dtype, tuple[int, ...]]],
    checkpoint: Path,
) -> tuple[dict[str, np.memmap], int]:
    paths = {name: output_dir / f"{name}.npy" for name in specs}
    exists = [path.exists() for path in paths.values()]
    if checkpoint.exists():
        if not all(exists):
            raise RuntimeError("checkpoint exists but an extraction array is missing")
        arrays = {
            name: np.lib.format.open_memmap(path, mode="r+")
            for name, path in paths.items()
        }
        for name, (_, shape) in specs.items():
            if tuple(arrays[name].shape) != shape:
                raise RuntimeError(f"resume shape mismatch for {name}")
        return arrays, int(json.loads(checkpoint.read_text())["next_row"])
    if any(exists):
        raise RuntimeError("partial arrays exist without a checkpoint; preserve and audit")
    arrays = {
        name: np.lib.format.open_memmap(
            paths[name], mode="w+", dtype=dtype, shape=shape
        )
        for name, (dtype, shape) in specs.items()
    }
    flush(arrays, checkpoint, 0, shape_rows(specs), 0)
    return arrays, 0


def shape_rows(specs: dict[str, tuple[np.dtype, tuple[int, ...]]]) -> int:
    rows = {shape[0] for _, shape in specs.values()}
    if len(rows) != 1:
        raise RuntimeError("array specs disagree on row count")
    return int(next(iter(rows)))


def flush(
    arrays: dict[str, np.memmap],
    checkpoint: Path,
    next_row: int,
    rows: int,
    active_batch: int,
) -> None:
    for array in arrays.values():
        array.flush()
    write_json(
        checkpoint,
        {
            "updated_at_utc": utc_now(),
            "next_row": int(next_row),
            "rows": int(rows),
            "active_batch_size": int(active_batch),
        },
    )


def prepare_task(task: str, output_dir: Path) -> tuple[pd.DataFrame, str]:
    source = TASK_SOURCES[task]
    frame = pd.read_parquet(source)
    source_hash = sha256(source)
    metadata = output_dir / "metadata.parquet"
    if metadata.exists():
        existing = pd.read_parquet(metadata)
        id_column = {
            "expectations": "expectation_id",
            "controlled": "controlled_id",
            "news": "news_relation_id",
            "profiles": "profile_id",
        }[task]
        if existing[id_column].tolist() != frame[id_column].tolist():
            raise RuntimeError(f"metadata row identity mismatch for {task}")
    else:
        shutil.copy2(source, metadata)
    return frame, source_hash


def forward_logits_only(
    model: Any,
    input_ids: Any,
    attention: Any,
    answer_ids: Any,
    torch: Any,
) -> tuple[Any, Any]:
    result = model(
        input_ids=input_ids,
        attention_mask=attention,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )
    rows = torch.arange(input_ids.shape[0], device=input_ids.device)
    last = attention.sum(dim=1) - 1
    logits = result.logits[rows, last].float()
    return logits.index_select(dim=1, index=answer_ids), logits


def array_manifest(arrays: dict[str, np.memmap]) -> dict[str, object]:
    return {
        name: {"shape": list(array.shape), "dtype": str(array.dtype)}
        for name, array in arrays.items()
    }


def task_manifest(
    *,
    task: str,
    model_name: str,
    source_hash: str,
    arrays: dict[str, np.memmap],
    output_dir: Path,
    max_length: int,
    final_batch: int,
    extra: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "task": task,
        "model_name": model_name,
        "model": MODEL_SPECS[model_name],
        "source_path": str(TASK_SOURCES[task].relative_to(ROOT)),
        "source_sha256": source_hash,
        "metadata_sha256": sha256(output_dir / "metadata.parquet"),
        "rows": int(next(iter(arrays.values())).shape[0]),
        "arrays": array_manifest(arrays),
        "max_length": max_length,
        "final_batch_size": final_batch,
        "market_labels_used": False,
    }
    if extra:
        payload.update(extra)
    write_json(output_dir / "manifest.json", payload)


def extract_expectations(
    *,
    model_name: str,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: int,
    batch_size: int,
    max_length: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    done = output_dir / "manifest.json"
    if done.exists() and json.loads(done.read_text()).get("status") == "complete":
        print(f"already complete: {done}", flush=True)
        return
    frame, source_hash = prepare_task("expectations", output_dir)
    spec = MODEL_SPECS[model_name]
    rows, layers, hidden = len(frame), spec["layers"], spec["hidden"]
    specs = {
        "pre_hidden": (np.float16, (rows, layers, hidden)),
        "pre_logits_abcd": (np.float32, (rows, 4)),
        "blocked_logits_abcd": (np.float32, (rows, 4)),
        "unknown_logits_abcd": (np.float32, (rows, 4)),
        "shuffled_logits_abcd": (np.float32, (rows, 4)),
        "subject_block_js": (np.float32, (rows,)),
        "unknown_js": (np.float32, (rows,)),
        "shuffled_js": (np.float32, (rows,)),
    }
    checkpoint = output_dir / "checkpoint.json"
    arrays, position = open_arrays(output_dir, specs, checkpoint)
    answers = option_ids(tokenizer, torch, device)
    active_batch = batch_size
    with torch.inference_mode():
        while position < rows:
            stop = min(rows, position + active_batch)
            batch = frame.iloc[position:stop]
            size = len(batch)
            company_prompts = batch["prompt"].astype(str).tolist()
            subjects = batch["company_name"].astype(str).tolist()
            prompts = company_prompts + company_prompts
            ids = attention = subject_mask = blocked_attention = None
            final = logits_abcd = logits = None
            control_ids = control_attention = control_options = control_logits = None
            try:
                ids, attention, subject_mask = recall_runtime.encode_prompts(
                    tokenizer,
                    prompts,
                    subjects + subjects,
                    model_name,
                    max_length,
                    torch,
                )
                ids = ids.to(f"cuda:{device}")
                attention = attention.to(f"cuda:{device}")
                assert subject_mask is not None
                subject_mask = subject_mask.to(f"cuda:{device}")
                blocked_attention = recall_runtime.make_last_to_subject_block_mask(
                    attention,
                    subject_mask,
                    torch.arange(size, 2 * size, device=ids.device),
                    next(model.parameters()).dtype,
                    torch,
                )
                final, _, logits_abcd, logits = recall_runtime.forward_states(
                    model=model,
                    input_ids=ids,
                    attention=attention,
                    subject_mask=None,
                    answer_ids=answers,
                    torch=torch,
                    model_attention=blocked_attention,
                )
                control_prompts = (
                    batch["unknown_prompt"].astype(str).tolist()
                    + batch["shuffled_prompt"].astype(str).tolist()
                )
                control_ids, control_attention, _ = recall_runtime.encode_prompts(
                    tokenizer, control_prompts, None, model_name, max_length, torch
                )
                control_ids = control_ids.to(f"cuda:{device}")
                control_attention = control_attention.to(f"cuda:{device}")
                control_options, control_logits = forward_logits_only(
                    model,
                    control_ids,
                    control_attention,
                    answers,
                    torch,
                )
            except torch.cuda.OutOfMemoryError:
                del (
                    ids,
                    attention,
                    subject_mask,
                    blocked_attention,
                    final,
                    logits_abcd,
                    logits,
                    control_ids,
                    control_attention,
                    control_options,
                    control_logits,
                )
                torch.cuda.empty_cache()
                if active_batch == 1:
                    raise
                active_batch = max(1, active_batch // 2)
                print(f"expectations OOM; retry batch={active_batch}", flush=True)
                continue
            slc = slice(position, stop)
            arrays["pre_hidden"][slc] = final[:size].cpu().numpy().astype(np.float16)
            arrays["pre_logits_abcd"][slc] = logits_abcd[:size].cpu().numpy()
            arrays["blocked_logits_abcd"][slc] = logits_abcd[size:].cpu().numpy()
            arrays["unknown_logits_abcd"][slc] = control_options[:size].cpu().numpy()
            arrays["shuffled_logits_abcd"][slc] = control_options[size:].cpu().numpy()
            arrays["subject_block_js"][slc] = recall_runtime.js_divergence_from_logits(
                logits[:size], logits[size:], torch
            ).cpu().numpy()
            arrays["unknown_js"][slc] = recall_runtime.js_divergence_from_logits(
                logits[:size], control_logits[:size], torch
            ).cpu().numpy()
            arrays["shuffled_js"][slc] = recall_runtime.js_divergence_from_logits(
                logits[:size], control_logits[size:], torch
            ).cpu().numpy()
            position = stop
            if position % 128 < active_batch or position == rows:
                flush(arrays, checkpoint, position, rows, active_batch)
                print(f"{model_name}/expectations {position}/{rows}", flush=True)
            del (
                ids,
                attention,
                subject_mask,
                blocked_attention,
                final,
                logits_abcd,
                logits,
                control_ids,
                control_attention,
                control_options,
                control_logits,
            )
    flush(arrays, checkpoint, rows, rows, active_batch)
    task_manifest(
        task="expectations",
        model_name=model_name,
        source_hash=source_hash,
        arrays=arrays,
        output_dir=output_dir,
        max_length=max_length,
        final_batch=active_batch,
        extra={
            "activation_position": "last pre-answer rendered-chat token",
            "controls": [
                "Unknown Corporation",
                "popularity-bin-matched shuffled company",
                "last-query attention blocked to company subject tokens at every layer",
                "disjoint expectation prompt paraphrase",
            ],
            "answer_semantics": {
                "A": "favorable",
                "B": "continuity/mixed",
                "C": "adverse",
                "D": "no firm-specific memory/unrelated/insufficient",
            },
        },
    )


def extract_state_task(
    *,
    task: str,
    model_name: str,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: int,
    batch_size: int,
    max_length: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    done = output_dir / "manifest.json"
    if done.exists() and json.loads(done.read_text()).get("status") == "complete":
        print(f"already complete: {done}", flush=True)
        return
    frame, source_hash = prepare_task(task, output_dir)
    spec = MODEL_SPECS[model_name]
    rows, layers, hidden = len(frame), spec["layers"], spec["hidden"]
    prefix = "controlled" if task == "controlled" else "news"
    specs = {
        f"{prefix}_hidden": (np.float16, (rows, layers, hidden)),
        f"{prefix}_logits_abcd": (np.float32, (rows, 4)),
    }
    checkpoint = output_dir / "checkpoint.json"
    arrays, position = open_arrays(output_dir, specs, checkpoint)
    answers = option_ids(tokenizer, torch, device)
    active_batch = batch_size
    with torch.inference_mode():
        while position < rows:
            stop = min(rows, position + active_batch)
            prompts = frame.iloc[position:stop]["prompt"].astype(str).tolist()
            ids = attention = final = logits_abcd = None
            try:
                ids, attention, _ = recall_runtime.encode_prompts(
                    tokenizer, prompts, None, model_name, max_length, torch
                )
                ids = ids.to(f"cuda:{device}")
                attention = attention.to(f"cuda:{device}")
                final, _, logits_abcd, _ = recall_runtime.forward_states(
                    model=model,
                    input_ids=ids,
                    attention=attention,
                    subject_mask=None,
                    answer_ids=answers,
                    torch=torch,
                )
            except torch.cuda.OutOfMemoryError:
                del ids, attention, final, logits_abcd
                torch.cuda.empty_cache()
                if active_batch == 1:
                    raise
                active_batch = max(1, active_batch // 2)
                print(f"{task} OOM; retry batch={active_batch}", flush=True)
                continue
            slc = slice(position, stop)
            arrays[f"{prefix}_hidden"][slc] = final.cpu().numpy().astype(np.float16)
            arrays[f"{prefix}_logits_abcd"][slc] = logits_abcd.cpu().numpy()
            position = stop
            if position % 128 < active_batch or position == rows:
                flush(arrays, checkpoint, position, rows, active_batch)
                print(f"{model_name}/{task} {position}/{rows}", flush=True)
            del ids, attention, final, logits_abcd
    flush(arrays, checkpoint, rows, rows, active_batch)
    task_manifest(
        task=task,
        model_name=model_name,
        source_hash=source_hash,
        arrays=arrays,
        output_dir=output_dir,
        max_length=max_length,
        final_batch=active_batch,
        extra={
            "activation_position": "last pre-answer rendered-chat token",
            "answer_semantics": {
                "A": "favorable",
                "B": "continuity/mixed",
                "C": "adverse",
                "D": "unrelated/insufficient",
            },
        },
    )


def extract_profiles(
    *,
    model_name: str,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: int,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    done = output_dir / "manifest.json"
    if done.exists() and json.loads(done.read_text()).get("status") == "complete":
        print(f"already complete: {done}", flush=True)
        return
    frame, source_hash = prepare_task("profiles", output_dir)
    rows = len(frame)
    specs = {
        "generated_ids": (np.int32, (rows, max_new_tokens)),
        "generated_length": (np.int16, (rows,)),
    }
    checkpoint = output_dir / "checkpoint.json"
    arrays, position = open_arrays(output_dir, specs, checkpoint)
    active_batch = batch_size
    with torch.inference_mode():
        while position < rows:
            stop = min(rows, position + active_batch)
            prompts = frame.iloc[position:stop]["prompt"].astype(str).tolist()
            try:
                ids, lengths, _ = recall_runtime.greedy_answers(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    model_name=model_name,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    torch=torch,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if active_batch == 1:
                    raise
                active_batch = max(1, active_batch // 2)
                print(f"profiles OOM; retry batch={active_batch}", flush=True)
                continue
            arrays["generated_ids"][position:stop] = ids
            arrays["generated_length"][position:stop] = lengths
            position = stop
            if position % 64 < active_batch or position == rows:
                flush(arrays, checkpoint, position, rows, active_batch)
                print(f"{model_name}/profiles {position}/{rows}", flush=True)
    flush(arrays, checkpoint, rows, rows, active_batch)
    texts: list[str] = []
    for ids, length in zip(arrays["generated_ids"], arrays["generated_length"]):
        texts.append(
            tokenizer.decode(ids[: int(length)].tolist(), skip_special_tokens=True).strip()
        )
    generations = frame.drop(columns="prompt").copy()
    generations["generated_text"] = texts
    generations["generated_length"] = np.asarray(arrays["generated_length"])
    generations.to_parquet(output_dir / "generations.parquet", index=False)
    task_manifest(
        task="profiles",
        model_name=model_name,
        source_hash=source_hash,
        arrays=arrays,
        output_dir=output_dir,
        max_length=max_length,
        final_batch=active_batch,
        extra={
            "max_new_tokens": max_new_tokens,
            "generation": "deterministic greedy; Qwen3 thinking disabled",
            "generations_path": str((output_dir / "generations.parquet").relative_to(ROOT)),
            "generations_sha256": sha256(output_dir / "generations.parquet"),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument(
        "--task",
        choices=("expectations", "controlled", "news", "profiles", "all"),
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--profile-max-new-tokens", type=int, default=320)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    for source in TASK_SOURCES.values():
        if not source.exists():
            raise RuntimeError(f"missing prompt source {source}")
    print(
        json.dumps(
            {
                "started_at_utc": utc_now(),
                "model": args.model,
                "device": args.device,
                "task": args.task,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "market_labels_used": False,
                "pid": os.getpid(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    torch, tokenizer, model = recall_runtime.load_runtime(args.model, args.device)
    model_dir = args.output_root / args.model
    tasks = (
        ("expectations", "controlled", "news", "profiles")
        if args.task == "all"
        else (args.task,)
    )
    for task in tasks:
        if task == "expectations":
            extract_expectations(
                model_name=args.model,
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                device=args.device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                output_dir=model_dir / task,
            )
        elif task in {"controlled", "news"}:
            extract_state_task(
                task=task,
                model_name=args.model,
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                device=args.device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                output_dir=model_dir / task,
            )
        else:
            extract_profiles(
                model_name=args.model,
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                device=args.device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                max_new_tokens=args.profile_max_new_tokens,
                output_dir=model_dir / task,
            )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"RETRIEVAL_EXTRACTION_COMPLETE model={args.model} task={args.task}", flush=True)


if __name__ == "__main__":
    main()
