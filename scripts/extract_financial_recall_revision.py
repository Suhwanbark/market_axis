#!/usr/bin/env python3
"""Checkpointed activation extraction for the financial recall/revision study.

The extractor intentionally never consumes a market outcome.  Recall examples
come from public company-name/ticker facts; revision examples are synthetic
matched-prior events.  Every long run is resumable at a row boundary.
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


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "financial_recall_revision"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "financial_recall_revision"
SYSTEM_PROMPT = (
    "You are a precise financial question-answering system. Follow the user's "
    "answer-format instruction exactly."
)
MODEL_SPECS = {
    "qwen25_7b": {"path": "/tmp/axis_qwen25_7b", "layers": 28, "hidden": 3584},
    "qwen3_4b": {"path": "/tmp/axis_qwen3_4b", "layers": 36, "hidden": 2560},
}


def normalize_ticker_answer(text: str) -> str:
    """Normalize a format-compliant greedy answer without mining explanations."""
    import re

    candidate = text.strip().upper().strip("`").strip()
    if candidate.startswith("$"):
        candidate = candidate[1:]
    candidate = candidate.rstrip(".")
    return candidate if re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z])?", candidate) else ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_chat(tokenizer: Any, prompt: str, model_name: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if model_name == "qwen3_4b":
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def encode_prompts(
    tokenizer: Any,
    prompts: list[str],
    subjects: list[str] | None,
    model_name: str,
    max_length: int,
    torch: Any,
) -> tuple[Any, Any, Any | None]:
    """Encode rendered chats and map subject character spans to token masks."""
    rendered = [render_chat(tokenizer, prompt, model_name) for prompt in prompts]
    encoded: list[list[int]] = []
    masks: list[list[bool]] = []
    for row, text in enumerate(rendered):
        tokenized = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=subjects is not None,
        )
        ids = list(tokenized["input_ids"])
        if len(ids) > max_length:
            raise RuntimeError(
                f"rendered prompt has {len(ids)} tokens, above frozen max_length={max_length}"
            )
        encoded.append(ids)
        if subjects is not None:
            subject = subjects[row]
            begin = text.find(subject)
            if begin < 0:
                raise RuntimeError(f"subject {subject!r} missing from rendered prompt")
            end = begin + len(subject)
            offsets = tokenized["offset_mapping"]
            mask = [right > begin and left < end for left, right in offsets]
            if not any(mask):
                raise RuntimeError(f"no subject tokens found for {subject!r}")
            masks.append(mask)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    width = max(map(len, encoded))
    input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(encoded), width), dtype=torch.long)
    subject_mask = (
        torch.zeros((len(encoded), width), dtype=torch.float32)
        if subjects is not None
        else None
    )
    for row, ids in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[row, : len(ids)] = 1
        if subject_mask is not None:
            subject_mask[row, : len(ids)] = torch.tensor(masks[row], dtype=torch.float32)
    return input_ids, attention, subject_mask


def encode_generation_prompts(
    tokenizer: Any,
    prompts: list[str],
    model_name: str,
    max_length: int,
    torch: Any,
) -> tuple[Any, Any]:
    """Left-pad decoder-only inputs so batched greedy generation is valid."""
    rendered = [render_chat(tokenizer, prompt, model_name) for prompt in prompts]
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = previous_side
    if encoded["input_ids"].shape[1] > max_length:
        raise RuntimeError(
            f"generation prompt exceeds frozen max_length={max_length}"
        )
    return encoded["input_ids"].to(dtype=torch.long), encoded["attention_mask"].to(
        dtype=torch.long
    )


def make_last_to_subject_block_mask(
    attention: Any,
    subject_mask: Any,
    blocked_rows: Any,
    dtype: Any,
    torch: Any,
) -> dict[str, Any]:
    """Build an additive causal mask and block subject keys for final queries.

    This is the finance analogue of the subject-to-last-token attention
    intervention used to distinguish successful recall, attentive errors, and
    inattentive errors in arXiv:2510.09033.  All other causal edges are kept.
    """
    batch, width = attention.shape
    query_index = torch.arange(width, device=attention.device).view(1, 1, width, 1)
    key_index = torch.arange(width, device=attention.device).view(1, 1, 1, width)
    key_is_real = attention.bool().view(batch, 1, 1, width)
    allowed = (key_index <= query_index) & key_is_real
    minimum = torch.finfo(dtype).min
    additive = torch.full(
        (batch, 1, width, width),
        minimum,
        dtype=dtype,
        device=attention.device,
    )
    additive.masked_fill_(allowed, 0.0)
    last_index = attention.sum(dim=1) - 1
    for row in blocked_rows.detach().cpu().tolist():
        subject_positions = subject_mask[row].bool()
        additive[row, 0, last_index[row], subject_positions] = minimum
    return {"full_attention": additive}


def greedy_answers(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    model_name: str,
    max_length: int,
    max_new_tokens: int,
    device: int,
    torch: Any,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ids, attention = encode_generation_prompts(
        tokenizer, prompts, model_name, max_length, torch
    )
    ids = ids.to(f"cuda:{device}")
    attention = attention.to(f"cuda:{device}")
    generated = model.generate(
        input_ids=ids,
        attention_mask=attention,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = generated[:, ids.shape[1] :]
    stored = np.full((len(prompts), max_new_tokens), -1, dtype=np.int32)
    lengths = np.zeros(len(prompts), dtype=np.int16)
    texts: list[str] = []
    for row, tokens in enumerate(new_ids.detach().cpu().tolist()):
        clean: list[int] = []
        for token in tokens:
            if token in {tokenizer.eos_token_id, tokenizer.pad_token_id}:
                break
            clean.append(int(token))
        lengths[row] = len(clean)
        stored[row, : len(clean)] = clean
        texts.append(tokenizer.decode(clean, skip_special_tokens=True).strip())
    return stored, lengths, texts


def forward_states(
    *,
    model: Any,
    input_ids: Any,
    attention: Any,
    subject_mask: Any | None,
    answer_ids: Any,
    torch: Any,
    model_attention: Any | None = None,
) -> tuple[Any, Any | None, Any, Any]:
    result = model(
        input_ids=input_ids,
        attention_mask=attention if model_attention is None else model_attention,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    row_index = torch.arange(input_ids.shape[0], device=input_ids.device)
    last_index = attention.sum(dim=1) - 1
    states = result.hidden_states[1:]
    final = torch.stack(
        [state[row_index, last_index].float() for state in states], dim=1
    )
    subject = None
    if subject_mask is not None:
        mask = subject_mask.unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        subject = torch.stack(
            [(state.float() * mask).sum(dim=1) / denominator for state in states],
            dim=1,
        )
    final_logits = result.logits[row_index, last_index].float()
    logits_ab = final_logits.index_select(dim=1, index=answer_ids)
    return final, subject, logits_ab, final_logits


def js_divergence_from_logits(left: Any, right: Any, torch: Any) -> Any:
    """Full-vocabulary Jensen-Shannon divergence, one value per row."""
    log_left = torch.log_softmax(left.float(), dim=-1)
    log_right = torch.log_softmax(right.float(), dim=-1)
    log_middle = torch.logaddexp(log_left, log_right) - math.log(2.0)
    left_kl = (log_left.exp() * (log_left - log_middle)).sum(dim=-1)
    right_kl = (log_right.exp() * (log_right - log_middle)).sum(dim=-1)
    return (0.5 * (left_kl + right_kl)).clamp_min(0.0)


def array_manifest(arrays: dict[str, np.memmap]) -> dict[str, dict[str, object]]:
    return {
        name: {"shape": list(array.shape), "dtype": str(array.dtype)}
        for name, array in arrays.items()
    }


def open_arrays(
    output_dir: Path,
    specs: dict[str, tuple[np.dtype, tuple[int, ...]]],
    checkpoint: Path,
) -> tuple[dict[str, np.memmap], int]:
    paths = {name: output_dir / f"{name}.npy" for name in specs}
    existing = [path.exists() for path in paths.values()]
    if checkpoint.exists():
        if not all(existing):
            raise RuntimeError("checkpoint exists but one or more representation arrays are missing")
        arrays = {
            name: np.lib.format.open_memmap(path, mode="r+")
            for name, path in paths.items()
        }
        for name, (_, shape) in specs.items():
            if tuple(arrays[name].shape) != shape:
                raise RuntimeError(f"resume shape mismatch for {name}")
        start = int(json.loads(checkpoint.read_text())["next_row"])
        return arrays, start
    if any(existing):
        raise RuntimeError(
            f"partial arrays exist without checkpoint in {output_dir}; preserve for audit and inspect"
        )
    arrays = {
        name: np.lib.format.open_memmap(
            paths[name], mode="w+", dtype=dtype, shape=shape
        )
        for name, (dtype, shape) in specs.items()
    }
    return arrays, 0


def flush_checkpoint(
    arrays: dict[str, np.memmap],
    checkpoint: Path,
    next_row: int,
    rows: int,
    active_batch_size: int,
) -> None:
    for array in arrays.values():
        array.flush()
    write_json(
        checkpoint,
        {
            "updated_at_utc": utc_now(),
            "next_row": next_row,
            "rows": rows,
            "active_batch_size": active_batch_size,
        },
    )


def ensure_metadata(source: Path, output_dir: Path) -> tuple[pd.DataFrame, str]:
    frame = pd.read_parquet(source)
    target = output_dir / "metadata.parquet"
    source_hash = sha256(source)
    if target.exists():
        if sha256(target) != source_hash:
            # Parquet rewriting is not byte-stable, so verify the row identity instead.
            existing = pd.read_parquet(target, columns=["sample_id"])
            if existing.sample_id.tolist() != frame.sample_id.tolist():
                raise RuntimeError(f"metadata identity mismatch in {output_dir}")
    else:
        shutil.copy2(source, target)
    return frame, source_hash


def extract_recall(
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
    source = DATA_DIR / "financial_recall.parquet"
    frame, source_hash = ensure_metadata(source, output_dir)
    spec = MODEL_SPECS[model_name]
    rows, layers, hidden = len(frame), spec["layers"], spec["hidden"]
    specs = {
        "recall_hidden": (np.float16, (rows, layers, hidden)),
        "recall_attention_blocked_hidden": (np.float16, (rows, layers, hidden)),
        "recall_subject_hidden": (np.float16, (rows, layers, hidden)),
        "recall_recognition_hidden": (np.float16, (rows, layers, hidden)),
        "recall_recognition_ablated_hidden": (np.float16, (rows, layers, hidden)),
        "recall_logits_ab": (np.float32, (rows, 2)),
        "recall_ablated_logits_ab": (np.float32, (rows, 2)),
        "recall_js_divergence": (np.float32, (rows,)),
        "recall_subject_replacement_js_divergence": (np.float32, (rows,)),
        "recall_recognition_js_divergence": (np.float32, (rows,)),
        "recall_generated_ids": (np.int32, (rows, max_new_tokens)),
        "recall_generated_length": (np.int16, (rows,)),
        "recall_correct": (np.uint8, (rows,)),
    }
    checkpoint = output_dir / "checkpoint.json"
    arrays, start = open_arrays(output_dir, specs, checkpoint)
    answer_ids = torch.tensor(
        [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in ("A", "B")],
        dtype=torch.long,
        device=f"cuda:{device}",
    )
    active_batch = batch_size
    position = start
    with torch.inference_mode():
        while position < rows:
            stop = min(rows, position + active_batch)
            batch = frame.iloc[position:stop]
            original = batch.prompt.astype(str).tolist()
            ablated = batch.subject_ablated_prompt.astype(str).tolist()
            prompts = original + original
            company_subjects = batch.company_name.astype(str).tolist()
            subjects = company_subjects + company_subjects
            ids, attention, subject_mask = encode_prompts(
                tokenizer, prompts, subjects, model_name, max_length, torch
            )
            ids = ids.to(f"cuda:{device}")
            attention = attention.to(f"cuda:{device}")
            assert subject_mask is not None
            subject_mask = subject_mask.to(f"cuda:{device}")
            size = len(batch)
            blocked_rows = torch.arange(size, 2 * size, device=ids.device)
            blocked_attention = make_last_to_subject_block_mask(
                attention,
                subject_mask,
                blocked_rows,
                next(model.parameters()).dtype,
                torch,
            )
            final = subject = logits = js = None
            replacement_ids = replacement_attention = replacement_logits = None
            replacement_js = None
            recognition_ids = recognition_attention = None
            recognition_hidden = recognition_logits_ab = recognition_logits = None
            recognition_js = None
            try:
                final, subject, _, logits = forward_states(
                    model=model,
                    input_ids=ids,
                    attention=attention,
                    subject_mask=subject_mask,
                    answer_ids=answer_ids,
                    torch=torch,
                    model_attention=blocked_attention,
                )
                js = js_divergence_from_logits(logits[:size], logits[size:], torch)
                replacement_ids, replacement_attention, _ = encode_prompts(
                    tokenizer, ablated, None, model_name, max_length, torch
                )
                replacement_ids = replacement_ids.to(f"cuda:{device}")
                replacement_attention = replacement_attention.to(f"cuda:{device}")
                _, _, _, replacement_logits = forward_states(
                    model=model,
                    input_ids=replacement_ids,
                    attention=replacement_attention,
                    subject_mask=None,
                    answer_ids=answer_ids,
                    torch=torch,
                )
                replacement_js = js_divergence_from_logits(
                    logits[:size], replacement_logits, torch
                )
                recognition_prompts = (
                    batch.recognition_prompt.astype(str).tolist()
                    + batch.recognition_subject_ablated_prompt.astype(str).tolist()
                )
                recognition_ids, recognition_attention, _ = encode_prompts(
                    tokenizer,
                    recognition_prompts,
                    None,
                    model_name,
                    max_length,
                    torch,
                )
                recognition_ids = recognition_ids.to(f"cuda:{device}")
                recognition_attention = recognition_attention.to(f"cuda:{device}")
                recognition_hidden, _, recognition_logits_ab, recognition_logits = (
                    forward_states(
                        model=model,
                        input_ids=recognition_ids,
                        attention=recognition_attention,
                        subject_mask=None,
                        answer_ids=answer_ids,
                        torch=torch,
                    )
                )
                recognition_js = js_divergence_from_logits(
                    recognition_logits[:size], recognition_logits[size:], torch
                )
                generated_ids, generated_length, generated_text = greedy_answers(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=original,
                    model_name=model_name,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    torch=torch,
                )
            except torch.cuda.OutOfMemoryError:
                del (
                    ids,
                    attention,
                    subject_mask,
                    final,
                    subject,
                    logits,
                    js,
                    replacement_ids,
                    replacement_attention,
                    replacement_logits,
                    replacement_js,
                    recognition_ids,
                    recognition_attention,
                    recognition_hidden,
                    recognition_logits_ab,
                    recognition_logits,
                    recognition_js,
                )
                torch.cuda.empty_cache()
                if active_batch == 1:
                    raise
                active_batch = max(1, active_batch // 2)
                print(f"recall OOM; retrying with batch_size={active_batch}", flush=True)
                continue
            assert subject is not None
            slc = slice(position, stop)
            arrays["recall_hidden"][slc] = final[:size].cpu().numpy().astype(np.float16)
            arrays["recall_attention_blocked_hidden"][slc] = (
                final[size:].cpu().numpy().astype(np.float16)
            )
            arrays["recall_subject_hidden"][slc] = subject[:size].cpu().numpy().astype(np.float16)
            arrays["recall_recognition_hidden"][slc] = (
                recognition_hidden[:size].cpu().numpy().astype(np.float16)
            )
            arrays["recall_recognition_ablated_hidden"][slc] = (
                recognition_hidden[size:].cpu().numpy().astype(np.float16)
            )
            arrays["recall_logits_ab"][slc] = recognition_logits_ab[:size].cpu().numpy()
            arrays["recall_ablated_logits_ab"][slc] = recognition_logits_ab[size:].cpu().numpy()
            arrays["recall_js_divergence"][slc] = js.cpu().numpy()
            arrays["recall_subject_replacement_js_divergence"][slc] = (
                replacement_js.cpu().numpy()
            )
            arrays["recall_recognition_js_divergence"][slc] = recognition_js.cpu().numpy()
            arrays["recall_generated_ids"][slc] = generated_ids
            arrays["recall_generated_length"][slc] = generated_length
            arrays["recall_correct"][slc] = np.asarray(
                [
                    normalize_ticker_answer(text) == ticker.upper()
                    for text, ticker in zip(generated_text, batch.ticker.astype(str))
                ],
                dtype=np.uint8,
            )
            position = stop
            if position % 128 < active_batch or position == rows:
                flush_checkpoint(arrays, checkpoint, position, rows, active_batch)
                print(f"{model_name}/recall: {position}/{rows}", flush=True)
            del (
                final,
                subject,
                logits,
                js,
                replacement_ids,
                replacement_attention,
                replacement_logits,
                replacement_js,
                recognition_hidden,
                recognition_logits_ab,
                recognition_logits,
                recognition_js,
                recognition_ids,
                recognition_attention,
                ids,
                attention,
                subject_mask,
                blocked_attention,
            )

    flush_checkpoint(arrays, checkpoint, rows, rows, active_batch)
    write_json(
        done,
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "task": "recall",
            "model_name": model_name,
            "model": spec,
            "source_path": str(source.relative_to(ROOT)),
            "source_sha256": source_hash,
            "rows": rows,
            "arrays": array_manifest(arrays),
            "activation_position": "last rendered-chat token immediately before answer",
            "subject_pooling": "mean of tokens overlapping the company-name character span",
            "primary_recall_label": (
                "normalized exact match of deterministic greedy free generation"
            ),
            "primary_behavioral_metric": (
                "full-vocabulary JS between original output and a run where the "
                "final query cannot attend to company subject tokens at any layer"
            ),
            "subject_replacement_control": (
                "full-vocabulary JS(original, Unknown Corporation replacement); "
                "stored separately because replacement effects may saturate"
            ),
            "recognition_control": (
                "matched two-choice prompt; logits and hidden states are stored separately"
            ),
            "max_new_tokens": max_new_tokens,
            "answer_token_ids": {"A": int(answer_ids[0]), "B": int(answer_ids[1])},
            "max_length": max_length,
            "final_batch_size": active_batch,
            "market_labels_used": False,
        },
    )


def extract_revision(
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
    source = DATA_DIR / "controlled_revision.parquet"
    frame, source_hash = ensure_metadata(source, output_dir)
    spec = MODEL_SPECS[model_name]
    rows, layers, hidden = len(frame), spec["layers"], spec["hidden"]
    specs = {
        "revision_pre_hidden": (np.float16, (rows, layers, hidden)),
        "revision_post_hidden": (np.float16, (rows, layers, hidden)),
        "revision_pre_logits_ab": (np.float32, (rows, 2)),
        "revision_post_logits_ab": (np.float32, (rows, 2)),
        "revision_js_divergence": (np.float32, (rows,)),
    }
    checkpoint = output_dir / "checkpoint.json"
    arrays, start = open_arrays(output_dir, specs, checkpoint)
    answer_ids = torch.tensor(
        [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in ("A", "B")],
        dtype=torch.long,
        device=f"cuda:{device}",
    )
    active_batch = batch_size
    position = start
    with torch.inference_mode():
        while position < rows:
            stop = min(rows, position + active_batch)
            batch = frame.iloc[position:stop]
            size = len(batch)
            prompts = batch.pre_prompt.astype(str).tolist() + batch.post_prompt.astype(str).tolist()
            ids, attention, _ = encode_prompts(
                tokenizer, prompts, None, model_name, max_length, torch
            )
            ids = ids.to(f"cuda:{device}")
            attention = attention.to(f"cuda:{device}")
            try:
                final, _, logits_ab, logits = forward_states(
                    model=model,
                    input_ids=ids,
                    attention=attention,
                    subject_mask=None,
                    answer_ids=answer_ids,
                    torch=torch,
                )
                js = js_divergence_from_logits(logits[:size], logits[size:], torch)
            except torch.cuda.OutOfMemoryError:
                del ids, attention
                torch.cuda.empty_cache()
                if active_batch == 1:
                    raise
                active_batch = max(1, active_batch // 2)
                print(f"revision OOM; retrying with batch_size={active_batch}", flush=True)
                continue
            slc = slice(position, stop)
            arrays["revision_pre_hidden"][slc] = final[:size].cpu().numpy().astype(np.float16)
            arrays["revision_post_hidden"][slc] = final[size:].cpu().numpy().astype(np.float16)
            arrays["revision_pre_logits_ab"][slc] = logits_ab[:size].cpu().numpy()
            arrays["revision_post_logits_ab"][slc] = logits_ab[size:].cpu().numpy()
            arrays["revision_js_divergence"][slc] = js.cpu().numpy()
            position = stop
            if position % 128 < active_batch or position == rows:
                flush_checkpoint(arrays, checkpoint, position, rows, active_batch)
                print(f"{model_name}/revision: {position}/{rows}", flush=True)
            del final, logits_ab, logits, js, ids, attention

    flush_checkpoint(arrays, checkpoint, rows, rows, active_batch)
    write_json(
        done,
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "task": "revision",
            "model_name": model_name,
            "model": spec,
            "source_path": str(source.relative_to(ROOT)),
            "source_sha256": source_hash,
            "rows": rows,
            "arrays": array_manifest(arrays),
            "activation_position": "last rendered-chat token immediately before answer",
            "behavioral_metric": "full-vocabulary JS(post, pre) plus normalized A/B logits",
            "answer_token_ids": {"A": int(answer_ids[0]), "B": int(answer_ids[1])},
            "max_length": max_length,
            "final_batch_size": active_batch,
            "market_labels_used": False,
        },
    )


def load_runtime(model_name: str, device: int) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODEL_SPECS[model_name]
    model_path = Path(spec["path"])
    if not model_path.exists():
        raise RuntimeError(
            f"missing {model_path}; run bash scripts/bootstrap_tmp_runtime.sh first"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for letter in ("A", "B"):
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{letter!r} is not one token: {ids}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        attn_implementation="sdpa",
    ).to(f"cuda:{device}").eval()
    if (
        int(model.config.num_hidden_layers) != spec["layers"]
        or int(model.config.hidden_size) != spec["hidden"]
    ):
        raise RuntimeError(f"model config mismatch for {model_name}")
    torch.set_grad_enabled(False)
    return torch, tokenizer, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--task", choices=("recall", "revision", "all"), default="all")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--recall-max-new-tokens", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    for source in (
        DATA_DIR / "financial_recall.parquet",
        DATA_DIR / "controlled_revision.parquet",
    ):
        if not source.exists():
            raise RuntimeError(
                f"missing {source}; run scripts/build_financial_recall_revision.py first"
            )
    print(
        json.dumps(
            {
                "started_at_utc": utc_now(),
                "model": args.model,
                "task": args.task,
                "device": args.device,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "pid": os.getpid(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    torch, tokenizer, model = load_runtime(args.model, args.device)
    model_dir = args.output_root / args.model
    if args.task in ("recall", "all"):
        extract_recall(
            model_name=args.model,
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.recall_max_new_tokens,
            output_dir=model_dir / "recall",
        )
    if args.task in ("revision", "all"):
        extract_revision(
            model_name=args.model,
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            output_dir=model_dir / "revision",
        )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"EXTRACTION_COMPLETE model={args.model} task={args.task}", flush=True)


if __name__ == "__main__":
    main()
