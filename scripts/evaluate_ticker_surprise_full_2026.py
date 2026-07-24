#!/usr/bin/env python3
"""Evaluate frozen axis scores on all available eligible 2026 news articles."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
PREFIXLESS = ROOT / "outputs" / "prefixless_article_axis_test" / "extractions"
PREFIXED = (
    ROOT
    / "outputs"
    / "company_prefixed_article_axis_test"
    / "extractions"
)
DIRECT = ROOT / "outputs" / "exact_article_baselines" / "direct_rating"
FULL = ROOT / "outputs" / "ticker_surprise_full_2026"
MODELS = ("qwen25_7b", "qwen3_4b")
SCORES = (
    "last_projection",
    "last_cosine",
    "mean_projection",
    "mean_cosine",
)
OUTCOMES = (
    "abs_market_adjusted_return_h1",
    "abs_market_adjusted_return_h2",
    "parkinson_expansion_h1",
    "abnormal_volume_h1",
)
SEED = 20260725


def load_scores(
    root: Path, model: str, status: str
) -> pd.DataFrame:
    pieces = []
    directories = sorted(
        path
        for path in (root / model).glob("shard_*_of_*")
        if path.is_dir()
    )
    if len(directories) != 2:
        raise RuntimeError(f"expected two shards under {root / model}")
    for directory in directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("status") != status:
            raise RuntimeError(f"incomplete shard: {directory}")
        metadata = pd.read_parquet(
            directory / "metadata.parquet",
            columns=[
                "global_row",
                "observation_id",
                "market_ticker",
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
    if (
        len(result) != 9_376
        or not result.observation_id.is_unique
        or result.global_row.tolist() != list(range(9_376))
    ):
        raise RuntimeError(f"score row identity failure: {root / model}")
    if not np.isfinite(result[list(SCORES)].to_numpy(float)).all():
        raise RuntimeError(f"non-finite score: {root / model}")
    return result


def load_direct(model: str) -> pd.DataFrame:
    pieces = []
    directories = sorted(
        path
        for path in (DIRECT / model).glob("shard_*_of_*")
        if path.is_dir()
    )
    if len(directories) != 2:
        raise RuntimeError(f"expected direct-rating shards for {model}")
    for directory in directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest.get("status")
            != "EXACT_ARTICLE_DIRECT_RATING_SHARD_COMPLETE"
        ):
            raise RuntimeError(f"incomplete direct rating: {directory}")
        metadata = pd.read_parquet(
            directory / "metadata.parquet",
            columns=["global_row", "observation_id"],
        )
        metadata["direct_rating"] = np.load(
            directory / "expected_rating.npy", mmap_mode="r"
        )
        pieces.append(metadata)
    result = (
        pd.concat(pieces, ignore_index=True)
        .sort_values("global_row")
        .reset_index(drop=True)
    )
    if (
        len(result) != 9_376
        or not result.observation_id.is_unique
        or not np.isfinite(result.direct_rating).all()
    ):
        raise RuntimeError(f"direct-rating identity failure: {model}")
    return result


def standardized_ranks(values: np.ndarray) -> np.ndarray:
    ranked = rankdata(values, method="average").astype(np.float64)
    ranked -= ranked.mean()
    scale = ranked.std()
    if not np.isfinite(scale) or scale < 1e-12:
        raise RuntimeError("degenerate rank vector")
    return ranked / scale


def group_moments(
    x: np.ndarray, y: np.ndarray, blocks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    keys, inverse = np.unique(blocks, return_inverse=True)
    moments = np.zeros((len(keys), 6), dtype=np.float64)
    for index in range(len(keys)):
        selected = inverse == index
        xv, yv = x[selected], y[selected]
        moments[index] = (
            selected.sum(),
            xv.sum(),
            yv.sum(),
            np.square(xv).sum(),
            np.square(yv).sum(),
            (xv * yv).sum(),
        )
    return keys, moments


def correlation_from_moments(moment: np.ndarray) -> float:
    n, sx, sy, sxx, syy, sxy = moment
    covariance = sxy - sx * sy / n
    variance_x = sxx - sx * sx / n
    variance_y = syy - sy * sy / n
    denominator = np.sqrt(max(variance_x * variance_y, 0.0))
    return float(covariance / denominator) if denominator > 0 else np.nan


def date_block_distribution(
    x: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    repeats: int,
    seed: int,
) -> np.ndarray:
    x_rank = standardized_ranks(x)
    y_rank = standardized_ranks(y)
    keys, moments = group_moments(x_rank, y_rank, blocks)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = rng.integers(0, len(keys), size=len(keys))
        values[repeat] = correlation_from_moments(
            moments[sampled].sum(axis=0)
        )
    return values[np.isfinite(values)]


def within_date_rank_correlation(
    x: np.ndarray, y: np.ndarray, blocks: np.ndarray
) -> float:
    x_rank = standardized_ranks(x)
    y_rank = standardized_ranks(y)
    for block in np.unique(blocks):
        selected = blocks == block
        x_rank[selected] -= x_rank[selected].mean()
        y_rank[selected] -= y_rank[selected].mean()
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def within_date_permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    observed: float,
    repeats: int,
    seed: int,
) -> float:
    x_rank = standardized_ranks(x)
    y_rank = standardized_ranks(y)
    groups = [
        np.flatnonzero(blocks == block) for block in np.unique(blocks)
    ]
    for index in groups:
        x_rank[index] -= x_rank[index].mean()
        y_rank[index] -= y_rank[index].mean()
    denominator = np.sqrt(
        np.square(x_rank).sum() * np.square(y_rank).sum()
    )
    rng = np.random.default_rng(seed)
    exceed = 0
    shuffled = x_rank.copy()
    for _ in range(repeats):
        for index in groups:
            shuffled[index] = rng.permutation(x_rank[index])
        value = float((shuffled * y_rank).sum() / denominator)
        exceed += int(value >= observed)
    return float((exceed + 1) / (repeats + 1))


def bh_adjust(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def paired_block_delta(
    candidate: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    blocks: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    candidate_rank = standardized_ranks(candidate)
    baseline_rank = standardized_ranks(baseline)
    target_rank = standardized_ranks(target)
    keys, candidate_moments = group_moments(
        candidate_rank, target_rank, blocks
    )
    baseline_keys, baseline_moments = group_moments(
        baseline_rank, target_rank, blocks
    )
    if keys.tolist() != baseline_keys.tolist():
        raise RuntimeError("paired block mismatch")
    observed = float(
        spearmanr(candidate, target).statistic
        - spearmanr(baseline, target).statistic
    )
    candidate_rho = float(spearmanr(candidate, target).statistic)
    baseline_rho = float(spearmanr(baseline, target).statistic)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = rng.integers(0, len(keys), size=len(keys))
        values[repeat] = correlation_from_moments(
            candidate_moments[sampled].sum(axis=0)
        ) - correlation_from_moments(
            baseline_moments[sampled].sum(axis=0)
        )
    values = values[np.isfinite(values)]
    return {
        "candidate_spearman": candidate_rho,
        "baseline_spearman": baseline_rho,
        "observed_delta_spearman": observed,
        "date_block_ci_low": float(np.quantile(values, 0.025)),
        "date_block_ci_high": float(np.quantile(values, 0.975)),
        "bootstrap_probability_candidate_not_better": float(
            np.mean(values <= 0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2_000)
    parser.add_argument("--permutation-repeats", type=int, default=2_000)
    parser.add_argument("--overwrite-complete", action="store_true")
    args = parser.parse_args()
    FULL.mkdir(parents=True, exist_ok=True)
    terminal = FULL / "evaluation_manifest.json"
    if terminal.exists():
        status = json.loads(terminal.read_text()).get("status")
        if (
            status == "TICKER_SURPRISE_FULL_2026_EVALUATION_COMPLETE"
            and not args.overwrite_complete
        ):
            print(f"already complete: {terminal}")
            return
        if status != "TICKER_SURPRISE_FULL_2026_EVALUATION_COMPLETE":
            raise RuntimeError(f"non-terminal manifest exists: {terminal}")
    outcome_manifest = json.loads(
        (FULL / "outcome_manifest.json").read_text()
    )
    if (
        outcome_manifest.get("status")
        != "TICKER_SURPRISE_FULL_2026_OUTCOMES_COMPLETE"
    ):
        raise RuntimeError("full-2026 outcomes are not frozen")
    outcomes = pd.read_parquet(FULL / "outcomes.parquet")
    blocks_all = outcomes.event_session.astype(str).to_numpy()
    issuers_all = outcomes.market_ticker.astype(str).to_numpy()

    score_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for model in MODELS:
        score_frames[(model, "prefixless")] = load_scores(
            PREFIXLESS,
            model,
            "PREFIXLESS_ARTICLE_AXIS_SHARD_COMPLETE",
        )
        score_frames[(model, "company_prefix")] = load_scores(
            PREFIXED,
            model,
            "COMPANY_PREFIXED_AXIS_SHARD_COMPLETE",
        )
    direct_frames = {model: load_direct(model) for model in MODELS}
    reference_ids = outcomes.observation_id.astype(str).tolist()
    for key, frame in score_frames.items():
        if frame.observation_id.astype(str).tolist() != reference_ids:
            raise RuntimeError(f"full-2026 identity mismatch: {key}")
    for model, frame in direct_frames.items():
        if frame.observation_id.astype(str).tolist() != reference_ids:
            raise RuntimeError(f"direct identity mismatch: {model}")

    rows = []
    order = 0
    for model in MODELS:
        for variant in ("prefixless", "company_prefix"):
            frame = score_frames[(model, variant)]
            for score in SCORES:
                for outcome in OUTCOMES:
                    valid = (
                        np.isfinite(frame[score])
                        & np.isfinite(outcomes[outcome])
                    ).to_numpy()
                    x = frame.loc[valid, score].to_numpy(float)
                    y = outcomes.loc[valid, outcome].to_numpy(float)
                    blocks = blocks_all[valid]
                    issuers = issuers_all[valid]
                    rho = float(spearmanr(x, y).statistic)
                    distribution = date_block_distribution(
                        x,
                        y,
                        blocks,
                        args.repeats,
                        SEED + order,
                    )
                    issuer_distribution = date_block_distribution(
                        x,
                        y,
                        issuers,
                        args.repeats,
                        SEED + 100_000 + order,
                    )
                    within = within_date_rank_correlation(x, y, blocks)
                    primary = (
                        score == "last_projection"
                        and outcome == "abs_market_adjusted_return_h1"
                    )
                    permutation_p = (
                        within_date_permutation_p(
                            x,
                            y,
                            blocks,
                            within,
                            args.permutation_repeats,
                            SEED + 10_000 + order,
                        )
                        if primary
                        else np.nan
                    )
                    rows.append(
                        {
                            "model": model,
                            "variant": variant,
                            "score": score,
                            "outcome": outcome,
                            "primary": primary,
                            "rows": int(valid.sum()),
                            "event_dates": int(np.unique(blocks).size),
                            "spearman": rho,
                            "date_block_ci_low": float(
                                np.quantile(distribution, 0.025)
                            ),
                            "date_block_ci_high": float(
                                np.quantile(distribution, 0.975)
                            ),
                            "date_block_probability_nonpositive": float(
                                np.mean(distribution <= 0)
                            ),
                            "date_block_p_one_sided": float(
                                (
                                    np.count_nonzero(distribution <= 0) + 1
                                )
                                / (len(distribution) + 1)
                            ),
                            "issuer_block_ci_low": float(
                                np.quantile(
                                    issuer_distribution, 0.025
                                )
                            ),
                            "issuer_block_ci_high": float(
                                np.quantile(
                                    issuer_distribution, 0.975
                                )
                            ),
                            "issuer_block_probability_nonpositive": float(
                                np.mean(issuer_distribution <= 0)
                            ),
                            "issuer_block_p_one_sided": float(
                                (
                                    np.count_nonzero(
                                        issuer_distribution <= 0
                                    )
                                    + 1
                                )
                                / (len(issuer_distribution) + 1)
                            ),
                            "within_date_rank_correlation": within,
                            "within_date_permutation_p_one_sided": permutation_p,
                            "configuration_order": order,
                        }
                    )
                    order += 1
    results = pd.DataFrame(rows)
    results["date_block_bh_q_all_64"] = bh_adjust(
        results.date_block_p_one_sided
    )
    results["issuer_block_bh_q_all_64"] = bh_adjust(
        results.issuer_block_p_one_sided
    )
    results["primary_model_bh_q_within_variant"] = np.nan
    results[
        "primary_model_issuer_bh_q_within_variant"
    ] = np.nan
    for variant in ("prefixless", "company_prefix"):
        selected = results.primary & results.variant.eq(variant)
        results.loc[
            selected, "primary_model_bh_q_within_variant"
        ] = bh_adjust(results.loc[selected, "date_block_p_one_sided"])
        results.loc[
            selected, "primary_model_issuer_bh_q_within_variant"
        ] = bh_adjust(results.loc[selected, "issuer_block_p_one_sided"])
    if len(results) != 64 or int(results.primary.sum()) != 4:
        raise RuntimeError("unexpected full-2026 correlation table shape")
    results.to_csv(FULL / "full_2026_axis_correlations.csv", index=False)

    stability_rows = []
    for model in MODELS:
        prefixless = score_frames[(model, "prefixless")]
        prefixed = score_frames[(model, "company_prefix")]
        token_delta = (
            prefixed.retained_tokens.to_numpy(float)
            - prefixless.retained_tokens.to_numpy(float)
        )
        for score in SCORES:
            before = prefixless[score].to_numpy(float)
            after = prefixed[score].to_numpy(float)
            scale = float(before.std())
            stability_rows.append(
                {
                    "model": model,
                    "score": score,
                    "rows": len(before),
                    "score_spearman": float(
                        spearmanr(before, after).statistic
                    ),
                    "score_pearson": float(
                        np.corrcoef(before, after)[0, 1]
                    ),
                    "mean_absolute_score_change_in_prefixless_sd": float(
                        np.mean(np.abs(after - before)) / scale
                    ),
                    "prefixless_rows_at_2048_tokens": int(
                        prefixless.retained_tokens.eq(2048).sum()
                    ),
                    "company_prefix_rows_at_2048_tokens": int(
                        prefixed.retained_tokens.eq(2048).sum()
                    ),
                    "median_total_token_increase": float(
                        np.median(token_delta)
                    ),
                }
            )
    stability = pd.DataFrame(stability_rows)
    if len(stability) != 8:
        raise RuntimeError("unexpected input-stability table shape")
    stability.to_csv(
        FULL / "full_2026_input_variant_stability.csv", index=False
    )

    prefix_gain_rows = []
    prefix_order = 0
    for model in MODELS:
        prefixless = score_frames[(model, "prefixless")]
        prefixed = score_frames[(model, "company_prefix")]
        for score in ("last_projection", "mean_projection"):
            for outcome in OUTCOMES:
                valid = np.isfinite(outcomes[outcome]).to_numpy()
                candidate = prefixed.loc[valid, score].to_numpy(float)
                baseline = prefixless.loc[valid, score].to_numpy(float)
                target = outcomes.loc[valid, outcome].to_numpy(float)
                date_result = paired_block_delta(
                    candidate,
                    baseline,
                    target,
                    blocks_all[valid],
                    args.repeats,
                    SEED + 180_000 + prefix_order,
                )
                issuer_result = paired_block_delta(
                    candidate,
                    baseline,
                    target,
                    issuers_all[valid],
                    args.repeats,
                    SEED + 190_000 + prefix_order,
                )
                prefix_gain_rows.append(
                    {
                        "model": model,
                        "score": score,
                        "outcome": outcome,
                        "rows": int(valid.sum()),
                        **date_result,
                        "issuer_block_ci_low": issuer_result[
                            "date_block_ci_low"
                        ],
                        "issuer_block_ci_high": issuer_result[
                            "date_block_ci_high"
                        ],
                        (
                            "issuer_bootstrap_probability_"
                            "prefix_not_better"
                        ): issuer_result[
                            "bootstrap_probability_candidate_not_better"
                        ],
                    }
                )
                prefix_order += 1
    prefix_gains = pd.DataFrame(prefix_gain_rows)
    if len(prefix_gains) != 16:
        raise RuntimeError("unexpected prefix-gain table shape")
    prefix_gains.to_csv(
        FULL / "full_2026_prefix_gain_all_outcomes.csv", index=False
    )

    direct_rows = []
    direct_order = 0
    for model in MODELS:
        direct = direct_frames[model]
        for outcome in OUTCOMES:
            valid = (
                np.isfinite(direct.direct_rating)
                & np.isfinite(outcomes[outcome])
            ).to_numpy()
            x = direct.loc[valid, "direct_rating"].to_numpy(float)
            y = outcomes.loc[valid, outcome].to_numpy(float)
            blocks = blocks_all[valid]
            issuers = issuers_all[valid]
            date_distribution = date_block_distribution(
                x,
                y,
                blocks,
                args.repeats,
                SEED + 200_000 + direct_order,
            )
            issuer_distribution = date_block_distribution(
                x,
                y,
                issuers,
                args.repeats,
                SEED + 210_000 + direct_order,
            )
            direct_rows.append(
                {
                    "model": model,
                    "outcome": outcome,
                    "rows": int(valid.sum()),
                    "spearman": float(spearmanr(x, y).statistic),
                    "date_block_ci_low": float(
                        np.quantile(date_distribution, 0.025)
                    ),
                    "date_block_ci_high": float(
                        np.quantile(date_distribution, 0.975)
                    ),
                    "date_block_p_one_sided": float(
                        (
                            np.count_nonzero(date_distribution <= 0) + 1
                        )
                        / (len(date_distribution) + 1)
                    ),
                    "issuer_block_ci_low": float(
                        np.quantile(issuer_distribution, 0.025)
                    ),
                    "issuer_block_ci_high": float(
                        np.quantile(issuer_distribution, 0.975)
                    ),
                    "issuer_block_p_one_sided": float(
                        (
                            np.count_nonzero(issuer_distribution <= 0) + 1
                        )
                        / (len(issuer_distribution) + 1)
                    ),
                }
            )
            direct_order += 1
    direct_results = pd.DataFrame(direct_rows)
    if len(direct_results) != 8:
        raise RuntimeError("unexpected Direct-LM correlation table shape")
    direct_results["date_block_bh_q_all_8"] = bh_adjust(
        direct_results.date_block_p_one_sided
    )
    direct_results["issuer_block_bh_q_all_8"] = bh_adjust(
        direct_results.issuer_block_p_one_sided
    )
    direct_results.to_csv(
        FULL / "full_2026_direct_lm_correlations.csv", index=False
    )

    axis_direct_rows = []
    comparison_order = 0
    for model in MODELS:
        direct = direct_frames[model]
        for variant in ("prefixless", "company_prefix"):
            frame = score_frames[(model, variant)]
            for score in ("last_projection", "mean_projection"):
                for outcome in OUTCOMES:
                    valid = np.isfinite(outcomes[outcome]).to_numpy()
                    candidate = frame.loc[valid, score].to_numpy(float)
                    baseline = direct.loc[
                        valid, "direct_rating"
                    ].to_numpy(float)
                    target = outcomes.loc[valid, outcome].to_numpy(float)
                    date_result = paired_block_delta(
                        candidate,
                        baseline,
                        target,
                        blocks_all[valid],
                        args.repeats,
                        SEED + 220_000 + comparison_order,
                    )
                    issuer_result = paired_block_delta(
                        candidate,
                        baseline,
                        target,
                        issuers_all[valid],
                        args.repeats,
                        SEED + 230_000 + comparison_order,
                    )
                    axis_direct_rows.append(
                        {
                            "model": model,
                            "variant": variant,
                            "score": score,
                            "outcome": outcome,
                            "rows": int(valid.sum()),
                            **date_result,
                            "issuer_block_ci_low": issuer_result[
                                "date_block_ci_low"
                            ],
                            "issuer_block_ci_high": issuer_result[
                                "date_block_ci_high"
                            ],
                            (
                                "issuer_bootstrap_probability_"
                                "axis_not_better"
                            ): issuer_result[
                                "bootstrap_probability_candidate_not_better"
                            ],
                        }
                    )
                    comparison_order += 1
    axis_direct = pd.DataFrame(axis_direct_rows)
    if len(axis_direct) != 32:
        raise RuntimeError("unexpected axis-versus-direct table shape")
    axis_direct.to_csv(
        FULL / "full_2026_axis_vs_direct_all_outcomes.csv", index=False
    )

    gain_rows = []
    for model_index, model in enumerate(MODELS):
        prefixless = score_frames[(model, "prefixless")]
        prefixed = score_frames[(model, "company_prefix")]
        direct = direct_frames[model]
        outcome = "abs_market_adjusted_return_h1"
        valid = np.isfinite(outcomes[outcome]).to_numpy()
        y = outcomes.loc[valid, outcome].to_numpy(float)
        blocks = blocks_all[valid]
        issuers = issuers_all[valid]
        comparisons = (
            (
                "company_prefix_last_projection",
                prefixed.loc[valid, "last_projection"].to_numpy(float),
                "prefixless_last_projection",
                prefixless.loc[valid, "last_projection"].to_numpy(float),
            ),
            (
                "prefixless_last_projection",
                prefixless.loc[valid, "last_projection"].to_numpy(float),
                "direct_lm_rating",
                direct.loc[valid, "direct_rating"].to_numpy(float),
            ),
            (
                "company_prefix_last_projection",
                prefixed.loc[valid, "last_projection"].to_numpy(float),
                "direct_lm_rating",
                direct.loc[valid, "direct_rating"].to_numpy(float),
            ),
        )
        for comparison_index, (
            candidate_name,
            candidate,
            baseline_name,
            baseline,
        ) in enumerate(comparisons):
            date_result = paired_block_delta(
                candidate,
                baseline,
                y,
                blocks,
                args.repeats,
                SEED
                + 20_000
                + 100 * model_index
                + comparison_index,
            )
            issuer_result = paired_block_delta(
                candidate,
                baseline,
                y,
                issuers,
                args.repeats,
                SEED
                + 120_000
                + 100 * model_index
                + comparison_index,
            )
            gain_rows.append(
                {
                    "model": model,
                    "outcome": outcome,
                    "candidate": candidate_name,
                    "baseline": baseline_name,
                    "rows": int(valid.sum()),
                    **date_result,
                    "issuer_block_ci_low": issuer_result[
                        "date_block_ci_low"
                    ],
                    "issuer_block_ci_high": issuer_result[
                        "date_block_ci_high"
                    ],
                    "issuer_bootstrap_probability_candidate_not_better": (
                        issuer_result[
                            "bootstrap_probability_candidate_not_better"
                        ]
                    ),
                }
            )
    gains = pd.DataFrame(gain_rows)
    if len(gains) != 6:
        raise RuntimeError("unexpected primary-gain table shape")
    gains.to_csv(FULL / "full_2026_primary_gain.csv", index=False)
    terminal.write_text(
        json.dumps(
            {
                "status": "TICKER_SURPRISE_FULL_2026_EVALUATION_COMPLETE",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "interpretation": (
                    "retrospective all-eligible-2026 evaluation; controlled "
                    "axes remain frozen and use no market labels"
                ),
                "rows": len(outcomes),
                "event_date_min": str(
                    pd.to_datetime(outcomes.event_session).min().date()
                ),
                "event_date_max": str(
                    pd.to_datetime(outcomes.event_session).max().date()
                ),
                "models": list(MODELS),
                "variants": ["prefixless", "company_prefix"],
                "primary_score": "last_projection",
                "primary_outcome": "abs_market_adjusted_return_h1",
                "date_block_repeats": args.repeats,
                "issuer_block_repeats": args.repeats,
                "within_date_permutation_repeats": (
                    args.permutation_repeats
                ),
                "post_open_audit_extension": (
                    "Direct-LM comparison across all four outcomes; "
                    "does not alter primary score, outcome, or decision"
                ),
                "axis_market_labels_used": False,
                "market_outcomes_used_for_evaluation": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("TICKER_SURPRISE_FULL_2026_EVALUATION_COMPLETE")


if __name__ == "__main__":
    main()
