#!/usr/bin/env python3
"""Extract dedicated embeddings for the exact article-axis input variants."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import extract_retrieval_conditioned_memory as shared


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "data" / "retrieval_conditioned_news" / "documents.parquet"
FEATURES = (
    ROOT / "outputs" / "retrieval_conditioned_feature_lock" / "features.parquet"
)
OUTPUT = ROOT / "outputs" / "exact_article_baselines" / "embeddings"
SPECS = {
    "qwen3_embedding_06b": {
        "path": "/tmp/market_axis_qwen3_embedding_06b",
        "hidden": 1024,
        "revision": "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418",
    },
    "qwen3_embedding_8b": {
        "path": "/tmp/market_axis_qwen3_embedding_8b",
        "hidden": 4096,
        "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
    },
}
VARIANTS = ("prefixless", "company_prefix")


def render(
    title: str,
    body: str,
    company: str,
    ticker: str,
    variant: str,
) -> str:
    article = str(title).strip() + "\n\n" + str(body).strip()
    if variant == "prefixless":
        return article
    if variant == "company_prefix":
        return (
            f"Target company: {str(company).strip()} "
            f"({str(ticker).strip()})\n\n{article}"
        )
    raise ValueError(variant)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=sorted(SPECS), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")

    import torch
    from transformers import AutoModel, AutoTokenizer

    spec = SPECS[args.name]
    model_path = Path(str(spec["path"]))
    if not model_path.exists():
        raise RuntimeError("run bootstrap_tmp_runtime.sh before extraction")
    output = (
        OUTPUT
        / args.name
        / args.variant
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    output.mkdir(parents=True, exist_ok=True)
    terminal = output / "manifest.json"
    if terminal.exists():
        status = json.loads(terminal.read_text()).get("status")
        if status == "EXACT_ARTICLE_EMBEDDING_SHARD_COMPLETE":
            print(f"already complete: {terminal}", flush=True)
            return
        raise RuntimeError(f"non-terminal manifest exists: {terminal}")

    identities = pd.read_parquet(
        FEATURES,
        columns=[
            "observation_id",
            "market_ticker",
            "company_name",
            "event_session",
            "downstream_split",
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
    metadata["embedding_text"] = [
        render(title, body, company, ticker, args.variant)
        for title, body, company, ticker in zip(
            metadata.title,
            metadata.analysis_body,
            metadata.company_name,
            metadata.market_ticker,
        )
    ]
    metadata_path = output / "metadata.parquet"
    if metadata_path.exists():
        existing = pd.read_parquet(
            metadata_path, columns=["observation_id"]
        )
        if existing.observation_id.tolist() != metadata.observation_id.tolist():
            raise RuntimeError("embedding shard identity mismatch")
    else:
        metadata.to_parquet(metadata_path, index=False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(f"cuda:{args.device}").eval()
    hidden = int(model.config.hidden_size)
    if hidden != int(spec["hidden"]):
        raise RuntimeError(f"hidden mismatch: {hidden} != {spec['hidden']}")
    specs = {
        "embeddings": (np.float16, (len(metadata), hidden)),
        "retained_tokens": (np.int32, (len(metadata),)),
    }
    checkpoint = output / "checkpoint.json"
    arrays, position = shared.open_arrays(output, specs, checkpoint)
    active_batch = args.batch_size
    with torch.inference_mode():
        while position < len(metadata):
            stop = min(len(metadata), position + active_batch)
            texts = metadata.iloc[position:stop].embedding_text.astype(
                str
            ).tolist()
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(f"cuda:{args.device}")
            result = pooled = None
            try:
                result = model(**encoded, return_dict=True)
                rows = torch.arange(
                    len(texts), device=encoded["input_ids"].device
                )
                last = encoded["attention_mask"].sum(dim=1) - 1
                pooled = result.last_hidden_state[rows, last].float()
                pooled = torch.nn.functional.normalize(pooled, dim=1)
            except torch.cuda.OutOfMemoryError:
                del encoded, result, pooled
                torch.cuda.empty_cache()
                if active_batch == 1:
                    raise
                active_batch = max(1, active_batch // 2)
                print(
                    f"EXACT_EMBEDDING_OOM batch={active_batch}", flush=True
                )
                continue
            slc = slice(position, stop)
            arrays["embeddings"][slc] = (
                pooled.cpu().numpy().astype(np.float16)
            )
            arrays["retained_tokens"][slc] = (
                encoded["attention_mask"].sum(dim=1).cpu().numpy()
            )
            position = stop
            if position % 128 < active_batch or position == len(metadata):
                shared.flush(
                    arrays,
                    checkpoint,
                    position,
                    len(metadata),
                    active_batch,
                )
                print(
                    f"EXACT_EMBEDDING_PROGRESS name={args.name} "
                    f"variant={args.variant} "
                    f"shard={args.shard_index}/{args.num_shards} "
                    f"rows={position}/{len(metadata)}",
                    flush=True,
                )
            del encoded, result, pooled

    shared.flush(
        arrays,
        checkpoint,
        len(metadata),
        len(metadata),
        active_batch,
    )
    finite = bool(
        np.isfinite(np.asarray(arrays["embeddings"])).all()
    )
    if not finite:
        raise RuntimeError("non-finite exact article embeddings")
    shared.write_json(
        terminal,
        {
            "status": "EXACT_ARTICLE_EMBEDDING_SHARD_COMPLETE",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.name,
            "model_spec": spec,
            "variant": args.variant,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "rows": len(metadata),
            "hidden": hidden,
            "input": (
                "title + two newlines + analysis_body"
                if args.variant == "prefixless"
                else (
                    "Target company: company_name (ticker), then title + "
                    "two newlines + analysis_body"
                )
            ),
            "instruction": None,
            "market_labels_used": False,
            "max_length": args.max_length,
            "pooling": "final-layer final retained token, L2 normalized",
            "finite": finite,
        },
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"EXACT_ARTICLE_EMBEDDING_SHARD_COMPLETE "
        f"name={args.name} variant={args.variant} "
        f"shard={args.shard_index}",
        flush=True,
    )


if __name__ == "__main__":
    main()
