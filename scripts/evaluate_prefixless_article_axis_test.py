#!/usr/bin/env python3
"""Test-only correlation of frozen axes on prefixless real-news forwards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
EXTRACTIONS = (
    ROOT / "outputs" / "prefixless_article_axis_test" / "extractions"
)
OUTCOMES = ROOT / "outputs" / "retrieval_conditioned_downstream"
OUTPUT = ROOT / "outputs" / "prefixless_article_axis_test" / "evaluation"
MODELS = ("qwen25_7b", "qwen3_4b")
SCORES = (
    "last_projection",
    "last_cosine",
    "mean_projection",
    "mean_cosine",
)
SEED = 20260725


def load_model(model: str) -> pd.DataFrame:
    pieces = []
    directories = sorted(
        path
        for path in (EXTRACTIONS / model).glob("shard_*_of_*")
        if path.is_dir()
    )
    if len(directories) != 2:
        raise RuntimeError(f"expected two shards for {model}: {directories}")
    for directory in directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("status") != "PREFIXLESS_ARTICLE_AXIS_SHARD_COMPLETE":
            raise RuntimeError(f"incomplete shard: {directory}")
        metadata = pd.read_parquet(
            directory / "metadata.parquet",
            columns=[
                "global_row",
                "observation_id",
                "market_ticker",
                "downstream_split",
                "event_session",
            ],
        )
        for score in SCORES:
            metadata[score] = np.load(
                directory / f"{score}.npy", mmap_mode="r"
            )
        metadata["retained_tokens"] = np.load(
            directory / "retained_tokens.npy", mmap_mode="r"
        )
        pieces.append(metadata)
    result = (
        pd.concat(pieces, ignore_index=True)
        .sort_values("global_row")
        .reset_index(drop=True)
    )
    if len(result) != 9_376 or not result.observation_id.is_unique:
        raise RuntimeError(f"row identity failure for {model}")
    numeric = result[list(SCORES) + ["retained_tokens"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"non-finite prefixless scores: {model}")
    return result


def block_bootstrap(
    frame: pd.DataFrame,
    score: str,
    target: str,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    blocks = sorted(frame.event_session.astype(str).unique())
    groups = {
        block: np.flatnonzero(
            frame.event_session.astype(str).to_numpy() == block
        )
        for block in blocks
    }
    x = frame[score].to_numpy(float)
    y = frame[target].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        index = np.concatenate([groups[block] for block in sampled])
        value = spearmanr(x[index], y[index]).statistic
        if np.isfinite(value):
            values.append(float(value))
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def stratified_permutation_p(
    frame: pd.DataFrame,
    score: str,
    target: str,
    observed: float,
    repeats: int,
    seed: int,
) -> float:
    block_values = frame.event_session.astype(str).to_numpy()
    groups = [
        np.flatnonzero(block_values == block)
        for block in np.unique(block_values)
    ]
    x = frame[score].to_numpy(float)
    y = frame[target].to_numpy(float)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(repeats):
        shuffled = x.copy()
        for index in groups:
            shuffled[index] = rng.permutation(shuffled[index])
        value = spearmanr(shuffled, y).statistic
        exceed += int(np.isfinite(value) and value >= observed)
    return float((exceed + 1) / (repeats + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5_000)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    terminal = OUTPUT / "manifest.json"
    if terminal.exists():
        status = json.loads(terminal.read_text()).get("status")
        if status == "PREFIXLESS_ARTICLE_AXIS_TEST_COMPLETE":
            print(f"already complete: {terminal}")
            return
        raise RuntimeError(f"non-terminal manifest exists: {terminal}")

    july_outcomes = pd.read_parquet(
        OUTCOMES / "july_confirmation_outcomes.parquet"
    )[
        [
            "observation_id",
            "abs_market_adjusted_return_h1",
            "market_adjusted_return",
        ]
    ]
    result_rows = []
    test_frames = []
    for model_index, model in enumerate(MODELS):
        extracted = load_model(model)
        test = extracted.loc[
            extracted.downstream_split.eq("july_confirmation")
        ].merge(
            july_outcomes,
            on="observation_id",
            how="inner",
            validate="one_to_one",
        )
        if len(test) != 383:
            raise RuntimeError(f"July row mismatch for {model}: {len(test)}")
        test["model"] = model
        test_frames.append(test)
        for score_index, score in enumerate(SCORES):
            rho = float(
                spearmanr(
                    test[score],
                    test.abs_market_adjusted_return_h1,
                ).statistic
            )
            low, high = block_bootstrap(
                test,
                score,
                "abs_market_adjusted_return_h1",
                args.repeats,
                SEED + 100 * model_index + score_index,
            )
            p_value = stratified_permutation_p(
                test,
                score,
                "abs_market_adjusted_return_h1",
                rho,
                args.repeats,
                SEED + 1000 + 100 * model_index + score_index,
            )
            result_rows.append(
                {
                    "model": model,
                    "score": score,
                    "primary": score == "last_projection",
                    "rows": len(test),
                    "event_dates": test.event_session.nunique(),
                    "spearman": rho,
                    "date_block_ci_low": low,
                    "date_block_ci_high": high,
                    "within_date_permutation_p_one_sided": p_value,
                    "retained_tokens_mean": float(
                        test.retained_tokens.mean()
                    ),
                    "retained_tokens_median": float(
                        test.retained_tokens.median()
                    ),
                    "max_length_fraction": float(
                        (test.retained_tokens == 2048).mean()
                    ),
                }
            )
    results = pd.DataFrame(result_rows)
    results.to_csv(OUTPUT / "july_prefixless_axis_results.csv", index=False)
    pd.concat(test_frames, ignore_index=True).to_parquet(
        OUTPUT / "july_prefixless_axis_scores.parquet", index=False
    )
    primary = results.loc[results.primary.astype(bool)]
    terminal.write_text(
        json.dumps(
            {
                "status": "PREFIXLESS_ARTICLE_AXIS_TEST_COMPLETE",
                "interpretation": (
                    "test-only zero-shot projection; no market-label fitting, "
                    "no prefix/question, retrospective July cohort"
                ),
                "primary_score": "last_projection",
                "rows": 383,
                "event_dates": 9,
                "repeats": args.repeats,
                "primary_results": primary.to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("PREFIXLESS_ARTICLE_AXIS_TEST_COMPLETE")


if __name__ == "__main__":
    main()
