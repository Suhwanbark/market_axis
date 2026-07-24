#!/usr/bin/env python3
"""Forward prefixless real articles and project hidden states on a frozen axis."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import extract_financial_recall_revision as runtime
import extract_retrieval_conditioned_memory as shared


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "data" / "retrieval_conditioned_news" / "documents.parquet"
FEATURES = (
    ROOT
    / "outputs"
    / "retrieval_conditioned_feature_lock"
    / "features.parquet"
)
AXES = ROOT / "outputs" / "prefixless_article_axis_test" / "axes"
OUTPUT = ROOT / "outputs" / "prefixless_article_axis_test" / "extractions"
MODELS = ("qwen25_7b", "qwen3_4b")


def plain_article(title: str, body: str) -> str:
    """No company prefix, task instruction, options, or answer delimiter."""
    return str(title).strip() + "\n\n" + str(body).strip()


def pool_states(hidden: Any, attention: Any, torch: Any) -> tuple[Any, Any]:
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    last_index = attention.sum(dim=1) - 1
    last = hidden[rows, last_index].float()
    mask = attention.unsqueeze(-1).float()
    mean = (hidden.float() * mask).sum(dim=1) / mask.sum(
        dim=1
    ).clamp_min(1.0)
    return last, mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")

    axis_manifest = json.loads((AXES / "manifest.json").read_text())
    if axis_manifest.get("status") != "PREFIXLESS_TEST_AXES_FROZEN":
        raise RuntimeError("prefixless test axes are not frozen")
    model_info = axis_manifest["models"][args.model]
    layer = int(model_info["layer"])
    with np.load(AXES / f"{args.model}.npz") as archive:
        unit_axis_numpy = np.asarray(archive["unit_axis"], np.float32)

    output = (
        OUTPUT
        / args.model
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    output.mkdir(parents=True, exist_ok=True)
    terminal = output / "manifest.json"
    if terminal.exists():
        status = json.loads(terminal.read_text()).get("status")
        if status == "PREFIXLESS_ARTICLE_AXIS_SHARD_COMPLETE":
            print(f"already complete: {terminal}", flush=True)
            return
        raise RuntimeError(f"non-terminal manifest exists: {terminal}")

    identities = pd.read_parquet(
        FEATURES,
        columns=[
            "observation_id",
            "market_ticker",
            "downstream_split",
            "event_session",
        ],
    ).reset_index(drop=True)
    identities["global_row"] = np.arange(len(identities), dtype=np.int64)
    documents = pd.read_parquet(
        DOCUMENTS,
        columns=[
            "observation_id",
            "title",
            "analysis_body",
        ],
    )
    metadata = identities.merge(
        documents,
        on="observation_id",
        how="inner",
        validate="one_to_one",
    )
    metadata = metadata.loc[
        metadata.global_row.mod(args.num_shards).eq(args.shard_index)
    ].reset_index(drop=True)
    metadata["plain_article"] = [
        plain_article(title, body)
        for title, body in zip(metadata.title, metadata.analysis_body)
    ]
    metadata_path = output / "metadata.parquet"
    if metadata_path.exists():
        existing = pd.read_parquet(
            metadata_path, columns=["observation_id"]
        )
        if existing.observation_id.tolist() != metadata.observation_id.tolist():
            raise RuntimeError("shard metadata identity mismatch")
    else:
        metadata.to_parquet(metadata_path, index=False)

    hidden_size = len(unit_axis_numpy)
    specs = {
        "last_hidden": (
            np.float16,
            (len(metadata), hidden_size),
        ),
        "mean_hidden": (
            np.float16,
            (len(metadata), hidden_size),
        ),
        "last_projection": (np.float32, (len(metadata),)),
        "last_cosine": (np.float32, (len(metadata),)),
        "mean_projection": (np.float32, (len(metadata),)),
        "mean_cosine": (np.float32, (len(metadata),)),
        "retained_tokens": (np.int32, (len(metadata),)),
    }
    checkpoint = output / "checkpoint.json"
    arrays, position = shared.open_arrays(output, specs, checkpoint)
    torch, tokenizer, model = runtime.load_runtime(
        args.model, args.device
    )
    unit_axis = torch.tensor(
        unit_axis_numpy,
        dtype=torch.float32,
        device=f"cuda:{args.device}",
    )
    captured: list[Any] = []

    def capture_layer(_module: Any, _inputs: Any, result: Any) -> None:
        value = result[0] if isinstance(result, tuple) else result
        captured.append(value)

    hook = model.model.layers[layer].register_forward_hook(capture_layer)
    active_batch = args.batch_size
    with torch.inference_mode():
        while position < len(metadata):
            stop = min(len(metadata), position + active_batch)
            texts = metadata.iloc[position:stop].plain_article.astype(
                str
            ).tolist()
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(f"cuda:{args.device}")
            captured.clear()
            result = last = mean = None
            try:
                result = model.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                )
                if len(captured) != 1:
                    raise RuntimeError(
                        f"layer hook captured {len(captured)} outputs"
                    )
                last, mean = pool_states(
                    captured[0], encoded["attention_mask"], torch
                )
            except torch.cuda.OutOfMemoryError:
                del encoded, result, last, mean
                captured.clear()
                torch.cuda.empty_cache()
                if active_batch == 1:
                    raise
                active_batch = max(1, active_batch // 2)
                print(
                    f"OOM; retry batch={active_batch}", flush=True
                )
                continue
            slc = slice(position, stop)
            last_projection = last @ unit_axis
            mean_projection = mean @ unit_axis
            last_cosine = last_projection / last.norm(
                dim=1
            ).clamp_min(1e-8)
            mean_cosine = mean_projection / mean.norm(
                dim=1
            ).clamp_min(1e-8)
            arrays["last_hidden"][slc] = (
                last.cpu().numpy().astype(np.float16)
            )
            arrays["mean_hidden"][slc] = (
                mean.cpu().numpy().astype(np.float16)
            )
            arrays["last_projection"][slc] = (
                last_projection.cpu().numpy()
            )
            arrays["last_cosine"][slc] = last_cosine.cpu().numpy()
            arrays["mean_projection"][slc] = (
                mean_projection.cpu().numpy()
            )
            arrays["mean_cosine"][slc] = mean_cosine.cpu().numpy()
            arrays["retained_tokens"][slc] = (
                encoded["attention_mask"].sum(dim=1).cpu().numpy()
            )
            position = stop
            if position % 64 < active_batch or position == len(metadata):
                shared.flush(
                    arrays,
                    checkpoint,
                    position,
                    len(metadata),
                    active_batch,
                )
                print(
                    f"PREFIXLESS_PROGRESS model={args.model} "
                    f"shard={args.shard_index}/{args.num_shards} "
                    f"rows={position}/{len(metadata)}",
                    flush=True,
                )
            del encoded, result, last, mean
            captured.clear()
    hook.remove()
    shared.flush(
        arrays,
        checkpoint,
        len(metadata),
        len(metadata),
        active_batch,
    )
    finite = all(
        np.isfinite(np.asarray(arrays[name])).all()
        for name in (
            "last_hidden",
            "mean_hidden",
            "last_projection",
            "last_cosine",
            "mean_projection",
            "mean_cosine",
        )
    )
    if not finite:
        raise RuntimeError("non-finite prefixless extraction")
    shared.write_json(
        terminal,
        {
            "status": "PREFIXLESS_ARTICLE_AXIS_SHARD_COMPLETE",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "layer": layer,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "rows": len(metadata),
            "hidden": hidden_size,
            "input": (
                "title + two newlines + analysis_body; no company prefix, "
                "instruction, options, question, or answer delimiter"
            ),
            "max_length": args.max_length,
            "pooling": ["last article token", "mean retained article tokens"],
            "market_labels_used": False,
            "finite": finite,
        },
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"PREFIXLESS_ARTICLE_AXIS_SHARD_COMPLETE model={args.model} "
        f"shard={args.shard_index}",
        flush=True,
    )


if __name__ == "__main__":
    main()
