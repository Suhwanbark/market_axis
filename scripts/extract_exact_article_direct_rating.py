#!/usr/bin/env python3
"""Extract label-free direct-LLM surprise ratings from exact cleaned articles.

The article content is identical to the prefixless axis input. A fixed rating
question is appended only because this is the explicit direct-LM baseline.
Article tokens are truncated before the question is appended, so the answer
instruction is always retained.
"""

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
    ROOT / "outputs" / "retrieval_conditioned_feature_lock" / "features.parquet"
)
OUTPUT = ROOT / "outputs" / "exact_article_baselines" / "direct_rating"
MODELS = ("qwen25_7b", "qwen3_4b")
QUESTION_TEMPLATE = """\
Considering only expectations already encoded in your parameters before this
article, how surprising is this news about {company_name} ({market_ticker})?

0 means fully expected and consistent with the remembered company profile.
9 means extremely unexpected and inconsistent with the remembered profile.

Return only one digit from 0 to 9."""


def article_text(title: str, body: str) -> str:
    return str(title).strip() + "\n\n" + str(body).strip()


def render_chat(tokenizer: Any, prompt: str, model_name: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the user's financial-news question directly. "
                "Follow the requested output format exactly."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if model_name == "qwen3_4b":
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def fit_article_to_prompt(
    tokenizer: Any,
    article: str,
    question: str,
    model_name: str,
    max_length: int,
) -> tuple[str, int, int]:
    """Keep the article prefix while reserving all question/chat tokens."""
    separator = "\n\n"
    empty_rendered = render_chat(
        tokenizer, separator + question, model_name
    )
    overhead = len(
        tokenizer.encode(empty_rendered, add_special_tokens=False)
    )
    budget = max(1, max_length - overhead)
    article_ids = tokenizer.encode(article, add_special_tokens=False)
    retained = min(len(article_ids), budget)
    while retained >= 1:
        clipped = tokenizer.decode(
            article_ids[:retained], skip_special_tokens=True
        ).strip()
        rendered = render_chat(
            tokenizer, clipped + separator + question, model_name
        )
        rendered_ids = tokenizer.encode(
            rendered, add_special_tokens=False
        )
        if len(rendered_ids) <= max_length:
            return rendered, retained, len(article_ids)
        retained -= max(1, len(rendered_ids) - max_length)
    raise RuntimeError("unable to reserve direct-rating question tokens")


def digit_ids(tokenizer: Any, torch: Any, device: int) -> Any:
    ids = []
    for digit in "0123456789":
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"digit {digit} is not one token: {encoded}")
        ids.append(int(encoded[0]))
    return torch.tensor(
        ids, dtype=torch.long, device=f"cuda:{device}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")

    output = (
        OUTPUT
        / args.model
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    output.mkdir(parents=True, exist_ok=True)
    terminal = output / "manifest.json"
    if terminal.exists():
        status = json.loads(terminal.read_text()).get("status")
        if status == "EXACT_ARTICLE_DIRECT_RATING_SHARD_COMPLETE":
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
    metadata["article"] = [
        article_text(title, body)
        for title, body in zip(metadata.title, metadata.analysis_body)
    ]
    metadata_path = output / "metadata.parquet"
    if metadata_path.exists():
        existing = pd.read_parquet(
            metadata_path, columns=["observation_id"]
        )
        if existing.observation_id.tolist() != metadata.observation_id.tolist():
            raise RuntimeError("direct-rating shard identity mismatch")
    else:
        metadata.to_parquet(metadata_path, index=False)

    specs = {
        "digit_logits": (np.float32, (len(metadata), 10)),
        "digit_probabilities": (np.float32, (len(metadata), 10)),
        "expected_rating": (np.float32, (len(metadata),)),
        "rating_entropy": (np.float32, (len(metadata),)),
        "argmax_rating": (np.int8, (len(metadata),)),
        "retained_article_tokens": (np.int32, (len(metadata),)),
        "original_article_tokens": (np.int32, (len(metadata),)),
        "rendered_tokens": (np.int32, (len(metadata),)),
    }
    checkpoint = output / "checkpoint.json"
    arrays, position = shared.open_arrays(output, specs, checkpoint)
    torch, tokenizer, model = runtime.load_runtime(
        args.model, args.device
    )
    answers = digit_ids(tokenizer, torch, args.device)
    rating_values = torch.arange(
        10, dtype=torch.float32, device=f"cuda:{args.device}"
    )
    active_batch = args.batch_size
    with torch.inference_mode():
        while position < len(metadata):
            stop = min(len(metadata), position + active_batch)
            rendered: list[str] = []
            retained: list[int] = []
            original: list[int] = []
            for row in metadata.iloc[position:stop].itertuples():
                question = QUESTION_TEMPLATE.format(
                    company_name=row.company_name,
                    market_ticker=row.market_ticker,
                )
                text, kept, total = fit_article_to_prompt(
                    tokenizer,
                    row.article,
                    question,
                    args.model,
                    args.max_length,
                )
                rendered.append(text)
                retained.append(kept)
                original.append(total)
            encoded = tokenizer(
                rendered,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            ).to(f"cuda:{args.device}")
            result = selected = probabilities = None
            try:
                result = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                )
                rows = torch.arange(
                    len(rendered), device=encoded["input_ids"].device
                )
                last = encoded["attention_mask"].sum(dim=1) - 1
                selected = result.logits[rows, last].float().index_select(
                    dim=1, index=answers
                )
                probabilities = torch.softmax(selected, dim=1)
            except torch.cuda.OutOfMemoryError:
                del encoded, result, selected, probabilities
                torch.cuda.empty_cache()
                if active_batch == 1:
                    raise
                active_batch = max(1, active_batch // 2)
                print(
                    f"DIRECT_RATING_OOM batch={active_batch}", flush=True
                )
                continue
            slc = slice(position, stop)
            arrays["digit_logits"][slc] = selected.cpu().numpy()
            arrays["digit_probabilities"][slc] = (
                probabilities.cpu().numpy()
            )
            arrays["expected_rating"][slc] = (
                probabilities @ rating_values
            ).cpu().numpy()
            arrays["rating_entropy"][slc] = (
                -(probabilities * probabilities.clamp_min(1e-12).log())
                .sum(dim=1)
                .cpu()
                .numpy()
            )
            arrays["argmax_rating"][slc] = (
                probabilities.argmax(dim=1).cpu().numpy()
            )
            arrays["retained_article_tokens"][slc] = retained
            arrays["original_article_tokens"][slc] = original
            arrays["rendered_tokens"][slc] = (
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
                    f"EXACT_DIRECT_PROGRESS model={args.model} "
                    f"shard={args.shard_index}/{args.num_shards} "
                    f"rows={position}/{len(metadata)}",
                    flush=True,
                )
            del encoded, result, selected, probabilities

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
            "digit_logits",
            "digit_probabilities",
            "expected_rating",
            "rating_entropy",
        )
    )
    if not finite:
        raise RuntimeError("non-finite exact direct-rating output")
    shared.write_json(
        terminal,
        {
            "status": "EXACT_ARTICLE_DIRECT_RATING_SHARD_COMPLETE",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "rows": len(metadata),
            "market_labels_used": False,
            "article_input": "title + two newlines + analysis_body",
            "question": QUESTION_TEMPLATE,
            "question_position": "after article; always retained",
            "max_length": args.max_length,
            "score": "expected next-token digit rating from 0 to 9",
            "finite": finite,
        },
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"EXACT_ARTICLE_DIRECT_RATING_SHARD_COMPLETE "
        f"model={args.model} shard={args.shard_index}",
        flush=True,
    )


if __name__ == "__main__":
    main()
