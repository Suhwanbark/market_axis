#!/usr/bin/env python3
"""Paired date-block incremental tests for exact-article baseline predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "exact_article_baselines" / "evaluation"
COMPARISONS = (
    (
        "price_embedding_8b_plus_axis_both",
        "price_plus_embedding_8b",
        "incremental last-token axes beyond price+embedding-8B",
    ),
    (
        "price_embedding_8b_plus_axis_mean_both",
        "price_plus_embedding_8b",
        "incremental mean-pooled axes beyond price+embedding-8B",
    ),
    (
        "price_plus_axis_last_both",
        "price_only",
        "incremental last-token axes beyond price controls",
    ),
    (
        "price_plus_axis_mean_both",
        "price_only",
        "incremental mean-pooled axes beyond price controls",
    ),
    (
        "price_plus_direct_both",
        "price_only",
        "incremental direct ratings beyond price controls",
    ),
)


def paired_bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    baseline: str,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    blocks = frame.event_session.astype(str).to_numpy()
    keys = np.unique(blocks)
    groups = {key: np.flatnonzero(blocks == key) for key in keys}
    truth = frame.abs_market_adjusted_return_h1.to_numpy(float)
    truth_log = np.log1p(10_000.0 * truth)
    label = frame.large_reaction.to_numpy(bool)
    candidate_values = frame[f"prediction_{candidate}"].to_numpy(float)
    baseline_values = frame[f"prediction_{baseline}"].to_numpy(float)
    observed_r2 = float(
        r2_score(truth_log, candidate_values)
        - r2_score(truth_log, baseline_values)
    )
    observed_auc = float(
        roc_auc_score(label, candidate_values)
        - roc_auc_score(label, baseline_values)
    )
    rng = np.random.default_rng(seed)
    delta_r2, delta_auc = [], []
    for _ in range(repeats):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        index = np.concatenate([groups[key] for key in sampled])
        if np.var(truth_log[index]) > 1e-12:
            delta_r2.append(
                r2_score(truth_log[index], candidate_values[index])
                - r2_score(truth_log[index], baseline_values[index])
            )
        if np.unique(label[index]).size == 2:
            delta_auc.append(
                roc_auc_score(label[index], candidate_values[index])
                - roc_auc_score(label[index], baseline_values[index])
            )
    r2 = np.asarray(delta_r2, float)
    auc = np.asarray(delta_auc, float)
    return {
        "observed_delta_r2_log": observed_r2,
        "delta_r2_ci_low": float(np.quantile(r2, 0.025)),
        "delta_r2_ci_high": float(np.quantile(r2, 0.975)),
        "bootstrap_probability_delta_r2_nonpositive": float(
            np.mean(r2 <= 0)
        ),
        "observed_delta_auc": observed_auc,
        "delta_auc_ci_low": float(np.quantile(auc, 0.025)),
        "delta_auc_ci_high": float(np.quantile(auc, 0.975)),
        "bootstrap_probability_delta_auc_nonpositive": float(
            np.mean(auc <= 0)
        ),
        "valid_r2_draws": len(r2),
        "valid_auc_draws": len(auc),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5_000)
    args = parser.parse_args()
    source_manifest = json.loads((SOURCE / "manifest.json").read_text())
    if (
        source_manifest.get("status")
        != "EXACT_ARTICLE_BASELINE_EVALUATION_COMPLETE"
    ):
        raise RuntimeError("exact baseline evaluation is not complete")
    output = SOURCE / "incremental_gain_manifest.json"
    if output.exists():
        status = json.loads(output.read_text()).get("status")
        if status == "EXACT_ARTICLE_INCREMENTAL_GAIN_COMPLETE":
            print(f"already complete: {output}")
            return
        raise RuntimeError(f"non-terminal manifest exists: {output}")
    predictions = pd.read_parquet(SOURCE / "july_predictions.parquet")
    rows = []
    for index, (candidate, baseline, interpretation) in enumerate(
        COMPARISONS
    ):
        rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "interpretation": interpretation,
                **paired_bootstrap(
                    predictions,
                    candidate,
                    baseline,
                    args.repeats,
                    20260725 + index,
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        SOURCE / "july_incremental_gain.csv", index=False
    )
    output.write_text(
        json.dumps(
            {
                "status": "EXACT_ARTICLE_INCREMENTAL_GAIN_COMPLETE",
                "rows": len(predictions),
                "event_dates": int(predictions.event_session.nunique()),
                "bootstrap_repeats": args.repeats,
                "comparisons": [
                    {"candidate": row[0], "baseline": row[1]}
                    for row in COMPARISONS
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("EXACT_ARTICLE_INCREMENTAL_GAIN_COMPLETE")


if __name__ == "__main__":
    main()
