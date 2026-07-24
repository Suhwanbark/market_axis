#!/usr/bin/env python3
"""Zero-shot direct impact ratings from the same base Llama 2 used for activations.

The frozen base checkpoint receives the repository's pre-existing 1--9 impact
instruction as plain text.  Because this is not a chat-tuned model, the score
is a constrained likelihood readout: after a common word-boundary token, take
the expected value under logits restricted to the nine digit tokens.
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import news_cutoff_safe_llama2 as protocol


ROOT = Path(__file__).resolve().parent.parent
NEWS_DOCS = ROOT / "outputs" / "news_cutoff_safe_llama2" / "documents.parquet"
SEC_DOCS = ROOT / "outputs" / "sec8k_uncentered" / "documents.parquet"
OUT = ROOT / "outputs" / "direct_llama2_rating"
MODEL_ID = "meta-llama/Llama-2-7b-hf"
REVISION = "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
PROMPT = """Assess the magnitude of the stock-price volatility that the document itself is likely to cause relative to the company's usual level. Judge magnitude, not whether the price will rise or fall. Use 1 for routine information with little expected effect and 9 for information likely to cause major repricing or uncertainty. Answer with exactly one digit from 1 to 9.

Document:
{text}

Score:"""


def document_frame(domain: str, split: str) -> pd.DataFrame:
    path = NEWS_DOCS if domain == "news" else SEC_DOCS
    frame = pd.read_parquet(path)
    part = protocol.split_documents(frame, split)
    text_column = "news_text" if domain == "news" else "text"
    result = part[["doc_row", "split_row", "ticker", "split", text_column, "text_chars"]].copy()
    if domain == "news":
        result["date"] = pd.to_datetime(part["date"]).dt.tz_localize(None)
    else:
        result["date"] = pd.to_datetime(part["event_session"]).dt.tz_localize(None)
    return result.rename(columns={text_column: "text"})


def score_path(domain: str, split: str) -> Path:
    return OUT / "arrays" / f"{domain}_{split}_direct_score.npy"


def probability_path(domain: str, split: str) -> Path:
    return OUT / "arrays" / f"{domain}_{split}_digit_probabilities.npy"


def checkpoint_path(domain: str, split: str) -> Path:
    return OUT / "arrays" / f"{domain}_{split}.checkpoint.json"


def done_path(domain: str, split: str) -> Path:
    return OUT / "arrays" / f"{domain}_{split}.done.json"


def open_arrays(domain: str, split: str, rows: int):
    scores = score_path(domain, split)
    probabilities = probability_path(domain, split)
    checkpoint = checkpoint_path(domain, split)
    scores.parent.mkdir(parents=True, exist_ok=True)
    if scores.exists() or probabilities.exists():
        if not scores.exists() or not probabilities.exists() or not checkpoint.exists():
            raise RuntimeError(f"incomplete direct-rating checkpoint for {domain}/{split}")
        score = np.load(scores, mmap_mode="r+")
        probability = np.load(probabilities, mmap_mode="r+")
        if score.shape != (rows,) or probability.shape != (rows, 9):
            raise RuntimeError("direct-rating checkpoint shape mismatch")
        start = int(json.loads(checkpoint.read_text())["next_order_index"])
        return score, probability, start
    score = np.lib.format.open_memmap(scores, mode="w+", dtype=np.float32, shape=(rows,))
    probability = np.lib.format.open_memmap(
        probabilities, mode="w+", dtype=np.float16, shape=(rows, 9)
    )
    return score, probability, 0


def extract_domain(
    domain: str,
    splits: list[str],
    model_path: str,
    batch_size: int,
    max_doc_tokens: int,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, LlamaTokenizerFast

    tokenizer = LlamaTokenizerFast(
        tokenizer_file=str(Path(model_path) / "tokenizer.json"), legacy=False
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    digit_ids = [tokenizer.convert_tokens_to_ids(str(value)) for value in range(1, 10)]
    if len(set(digit_ids)) != 9 or any(value == tokenizer.unk_token_id for value in digit_ids):
        raise RuntimeError(f"digits are not distinct Llama tokens: {digit_ids}")
    boundary_id = tokenizer.convert_tokens_to_ids("▁")
    if boundary_id == tokenizer.unk_token_id:
        raise RuntimeError("missing Llama word-boundary token")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=REVISION,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda:0").eval()
    model.config.use_cache = False
    candidates = torch.tensor(digit_ids, dtype=torch.long, device="cuda:0")
    values = torch.arange(1, 10, dtype=torch.float32, device="cuda:0")
    active_batch = batch_size
    with torch.inference_mode():
        for split in splits:
            if done_path(domain, split).exists():
                print(f"direct Llama {domain}/{split}: already complete", flush=True)
                continue
            frame = document_frame(domain, split)
            order = np.argsort(frame["text_chars"].to_numpy(), kind="stable")
            score, probabilities, order_start = open_arrays(domain, split, len(frame))
            while order_start < len(order):
                positions = order[order_start : order_start + active_batch]
                raw = frame.iloc[positions]["text"].tolist()
                document_ids = tokenizer(
                    raw,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_doc_tokens,
                )["input_ids"]
                prompt_ids = []
                for ids in document_ids:
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    encoded = tokenizer.encode(
                        PROMPT.format(text=text), add_special_tokens=True
                    )
                    prompt_ids.append([*encoded, boundary_id])
                encoded = tokenizer.pad(
                    {"input_ids": prompt_ids}, padding=True, return_tensors="pt"
                ).to("cuda:0")
                try:
                    logits = model(
                        **encoded,
                        use_cache=False,
                        return_dict=True,
                        logits_to_keep=1,
                    ).logits[:, -1]
                    digit_logits = logits.index_select(1, candidates).float()
                    probs = torch.softmax(digit_logits, dim=1)
                    expected = (probs * values).sum(dim=1)
                except torch.cuda.OutOfMemoryError:
                    del encoded
                    torch.cuda.empty_cache()
                    if active_batch == 1:
                        raise
                    active_batch = max(1, active_batch // 2)
                    print(
                        f"direct Llama {domain}/{split}: OOM, retry batch={active_batch}",
                        flush=True,
                    )
                    continue
                score[positions] = expected.cpu().numpy().astype(np.float32)
                probabilities[positions] = probs.cpu().numpy().astype(np.float16)
                next_index = order_start + len(positions)
                if next_index % 512 < active_batch or next_index == len(order):
                    score.flush()
                    probabilities.flush()
                    protocol.write_json(
                        checkpoint_path(domain, split),
                        {
                            "next_order_index": next_index,
                            "rows": len(frame),
                            "active_batch_size": active_batch,
                        },
                    )
                if next_index % max(500, active_batch) < active_batch or next_index == len(order):
                    print(
                        f"direct Llama {domain}/{split}: {next_index}/{len(order)}",
                        flush=True,
                    )
                del encoded, logits, digit_logits, probs, expected
                order_start = next_index
            score.flush()
            probabilities.flush()
            protocol.write_json(
                done_path(domain, split),
                {
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "model_id": MODEL_ID,
                    "revision": REVISION,
                    "model_path": model_path,
                    "domain": domain,
                    "split": split,
                    "rows": len(frame),
                    "prompt": PROMPT,
                    "score": "expected value of logits restricted to digit tokens 1..9 after a common boundary token",
                    "base_model_warning": "plain-text constrained likelihood; checkpoint is not instruction tuned",
                    "max_document_tokens": max_doc_tokens,
                    "requested_batch_size": batch_size,
                    "final_batch_size": active_batch,
                    "forward_dtype": "bfloat16",
                },
            )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def finalize() -> None:
    rows = []
    for domain in ("news", "sec8k"):
        for split in ("train", "val", "test"):
            if not done_path(domain, split).exists():
                raise RuntimeError(f"missing direct score: {domain}/{split}")
            frame = document_frame(domain, split)
            score = np.load(score_path(domain, split))
            if len(score) != len(frame) or not np.isfinite(score).all():
                raise RuntimeError(f"invalid direct score: {domain}/{split}")
            part = frame[["doc_row", "split_row", "ticker", "date", "split"]].copy()
            part["domain"] = domain
            part["direct_lm_score"] = score
            rows.append(part)
    result = pd.concat(rows, ignore_index=True)
    result.to_parquet(OUT / "all_scores.parquet", index=False)
    protocol.write_json(
        OUT / "manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_id": MODEL_ID,
            "revision": REVISION,
            "same_checkpoint_as_activation": True,
            "domains": {
                f"{domain}/{split}": int(count)
                for (domain, split), count in result.groupby(["domain", "split"]).size().items()
            },
            "centering": "none",
            "label_access": "none; zero-shot score",
        },
    )
    print(result.groupby(["domain", "split"])["direct_lm_score"].agg(["size", "mean", "std", "min", "max"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("--domain", choices=("news", "sec8k"), required=True)
    extract_parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    extract_parser.add_argument("--model-path", required=True)
    extract_parser.add_argument("--batch-size", type=int, required=True)
    extract_parser.add_argument("--max-doc-tokens", type=int, required=True)
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "extract":
        extract_domain(
            args.domain, args.splits, args.model_path, args.batch_size, args.max_doc_tokens
        )
    else:
        finalize()


if __name__ == "__main__":
    main()
