#!/usr/bin/env python3
"""Project company/ticker-prefixed real articles on frozen controlled axes."""

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
from extract_prefixless_article_axis_scores import pool_states


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "data" / "retrieval_conditioned_news" / "documents.parquet"
FEATURES = (
    ROOT / "outputs" / "retrieval_conditioned_feature_lock" / "features.parquet"
)
AXES = ROOT / "outputs" / "prefixless_article_axis_test" / "axes"
OUTPUT = (
    ROOT
    / "outputs"
    / "company_prefixed_article_axis_test"
    / "extractions"
)
MODELS = ("qwen25_7b", "qwen3_4b")


def prefixed_article(
    company: str, ticker: str, title: str, body: str
) -> str:
    return (
        f"Target company: {str(company).strip()} "
        f"({str(ticker).strip()})\n\n"
        f"{str(title).strip()}\n\n{str(body).strip()}"
    )


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
        raise RuntimeError("controlled axes are not frozen")
    layer = int(axis_manifest["models"][args.model]["layer"])
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
        if status == "COMPANY_PREFIXED_AXIS_SHARD_COMPLETE":
            print(f"already complete: {terminal}", flush=True)
            return
        raise RuntimeError(f"non-terminal manifest exists: {terminal}")

    identities = pd.read_parquet(
        FEATURES,
        columns=[
            "observation_id",
            "market_ticker",
            "company_name",
            "downstream_split",
            "event_session",
        ],
    ).reset_index(drop=True)
    identities["global_row"] = np.arange(len(identities), dtype=np.int64)
    documents = pd.read_parquet(
        DOCUMENTS,
        columns=["observation_id", "title", "analysis_body"],
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
    metadata["article_input"] = [
        prefixed_article(company, ticker, title, body)
        for company, ticker, title, body in zip(
            metadata.company_name,
            metadata.market_ticker,
            metadata.title,
            metadata.analysis_body,
        )
    ]
    metadata_path = output / "metadata.parquet"
    if metadata_path.exists():
        existing = pd.read_parquet(
            metadata_path, columns=["observation_id"]
        )
        if existing.observation_id.tolist() != metadata.observation_id.tolist():
            raise RuntimeError("company-prefix shard identity mismatch")
    else:
        metadata.to_parquet(metadata_path, index=False)

    specs = {
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
            texts = metadata.iloc[position:stop].article_input.astype(
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
                    f"COMPANY_PREFIX_OOM batch={active_batch}", flush=True
                )
                continue
            slc = slice(position, stop)
            last_projection = last @ unit_axis
            mean_projection = mean @ unit_axis
            arrays["last_projection"][slc] = (
                last_projection.cpu().numpy()
            )
            arrays["last_cosine"][slc] = (
                last_projection
                / last.norm(dim=1).clamp_min(1e-8)
            ).cpu().numpy()
            arrays["mean_projection"][slc] = (
                mean_projection.cpu().numpy()
            )
            arrays["mean_cosine"][slc] = (
                mean_projection
                / mean.norm(dim=1).clamp_min(1e-8)
            ).cpu().numpy()
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
                    f"COMPANY_PREFIX_PROGRESS model={args.model} "
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
            "last_projection",
            "last_cosine",
            "mean_projection",
            "mean_cosine",
        )
    )
    if not finite:
        raise RuntimeError("non-finite company-prefix axis score")
    shared.write_json(
        terminal,
        {
            "status": "COMPANY_PREFIXED_AXIS_SHARD_COMPLETE",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "layer": layer,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "rows": len(metadata),
            "input": (
                "Target company: company_name (market_ticker), then title + "
                "two newlines + analysis_body; no instruction, question, "
                "answer options, or delimiter"
            ),
            "market_labels_used": False,
            "max_length": args.max_length,
            "pooling": ["last retained token", "mean retained tokens"],
            "finite": finite,
        },
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"COMPANY_PREFIXED_AXIS_SHARD_COMPLETE "
        f"model={args.model} shard={args.shard_index}",
        flush=True,
    )


if __name__ == "__main__":
    main()
