#!/usr/bin/env python3
"""Audit and summarize the nonlinear ticker-conditioned surprise experiment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "ticker_surprise_nonlinear"
FINAL = SOURCE / "final"
PARENT_AUDIT = (
    ROOT
    / "outputs"
    / "retrieval_conditioned_completion_audit"
    / "completion_audit.json"
)
MODELS = ("qwen25_7b", "qwen3_4b")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_number(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    checks: dict[str, bool] = {}
    parent = read_json(PARENT_AUDIT)
    checks["parent_completion_audit_passed"] = (
        parent.get("status") == "RETRIEVAL_CONDITIONED_COMPLETION_AUDIT_PASSED"
    )

    selected_metrics = []
    validation_comparison = []
    article_ids: dict[str, list[str]] = {}
    source_files: list[Path] = [PARENT_AUDIT]
    for model in MODELS:
        model_dir = SOURCE / model
        manifest_path = model_dir / "manifest.json"
        selection_path = model_dir / "selection.json"
        controlled_path = model_dir / "controlled_scores.parquet"
        relation_path = model_dir / "real_news_relation_scores.parquet"
        article_path = model_dir / "real_news_article_scores.parquet"
        suite_path = model_dir / "selected_suite_metrics.csv"
        sweep_path = model_dir / "validation_sweep.csv"
        source_files.extend(
            [
                manifest_path,
                selection_path,
                controlled_path,
                relation_path,
                article_path,
                suite_path,
                sweep_path,
            ]
        )
        manifest = read_json(manifest_path)
        selection = read_json(selection_path)
        checks[f"{model}_finder_complete"] = (
            manifest.get("status") == "TICKER_SURPRISE_NONLINEAR_COMPLETE"
        )
        checks[f"{model}_selection_label_blind"] = (
            selection.get("test_used_for_selection") is False
            and selection.get("market_labels_used") is False
        )
        controlled = pd.read_parquet(controlled_path)
        relation = pd.read_parquet(relation_path)
        article = pd.read_parquet(article_path)
        numeric_controlled = controlled.select_dtypes(
            include=[np.number]
        ).to_numpy(float)
        numeric_relation = relation.select_dtypes(
            include=[np.number]
        ).to_numpy(float)
        numeric_article = article.select_dtypes(
            include=[np.number]
        ).to_numpy(float)
        checks[f"{model}_scores_finite"] = bool(
            np.isfinite(numeric_controlled).all()
            and np.isfinite(numeric_relation).all()
            and np.isfinite(numeric_article).all()
        )
        checks[f"{model}_row_counts_match_manifest"] = bool(
            len(controlled) == manifest["controlled_rows"]
            and len(relation) == manifest["real_news_relation_rows"]
            and len(article) == manifest["real_news_article_rows"]
        )
        checks[f"{model}_row_ids_unique"] = bool(
            controlled.controlled_id.is_unique
            and relation.news_relation_id.is_unique
            and article.observation_id.is_unique
        )
        article_ids[model] = sorted(article.observation_id.astype(str))

        suite = pd.read_csv(suite_path)
        selected_metrics.append(suite)
        sweep = pd.read_csv(sweep_path)
        validation = sweep.loc[
            sweep.layer.eq(int(selection["selected_layer"]))
        ]
        for method in (
            "linear_delta",
            "absolute_delta",
            "mlp_pair_delta",
            "mlp_pair_post",
        ):
            candidate = (
                validation.loc[validation.method.eq(method)]
                .sort_values(
                    [
                        "validation_pair_concordance",
                        "validation_within_group_spearman",
                    ],
                    ascending=False,
                )
                .iloc[0]
            )
            validation_comparison.append(
                {
                    "model": model,
                    "method": method,
                    "layer": int(candidate.layer),
                    "validation_pair_concordance": float(
                        candidate.validation_pair_concordance
                    ),
                    "validation_within_group_spearman": float(
                        candidate.validation_within_group_spearman
                    ),
                }
            )

    checks["cross_model_article_ids_match"] = (
        article_ids[MODELS[0]] == article_ids[MODELS[1]]
    )

    robustness_dir = SOURCE / "controlled_robustness"
    robustness_manifest_path = robustness_dir / "manifest.json"
    robustness_path = robustness_dir / "test_robustness.csv"
    robustness_manifest = read_json(robustness_manifest_path)
    robustness = pd.read_csv(robustness_path)
    source_files.extend([robustness_manifest_path, robustness_path])
    checks["controlled_robustness_complete"] = (
        robustness_manifest.get("status")
        == "TICKER_SURPRISE_CONTROLLED_ROBUSTNESS_COMPLETE"
        and robustness_manifest.get("test_used_for_model_selection") is False
    )
    checks["controlled_permutation_significant_both_models"] = bool(
        (
            robustness.loc[
                robustness.metric.eq("pair_concordance"),
                "stratified_permutation_p_one_sided",
            ]
            < 0.05
        ).all()
    )

    market_dir = SOURCE / "market_downstream"
    market_manifest_path = market_dir / "manifest.json"
    market_path = market_dir / "july_model_comparison.csv"
    delta_path = market_dir / "july_delta_r2_bootstrap.csv"
    univariate_path = market_dir / "july_univariate.csv"
    routing_path = market_dir / "july_alert_routing.csv"
    alert_path = (
        market_dir / "alert_comparison" / "paired_auc_comparison.csv"
    )
    alert_manifest_path = (
        market_dir / "alert_comparison" / "manifest.json"
    )
    source_files.extend(
        [
            market_manifest_path,
            market_path,
            delta_path,
            univariate_path,
            routing_path,
            alert_path,
            alert_manifest_path,
        ]
    )
    market_manifest = read_json(market_manifest_path)
    alert_manifest = read_json(alert_manifest_path)
    checks["market_downstream_complete"] = (
        market_manifest.get("status")
        == "TICKER_SURPRISE_MARKET_DOWNSTREAM_COMPLETE"
    )
    checks["alert_comparison_complete"] = (
        alert_manifest.get("status")
        == "TICKER_SURPRISE_ALERT_COMPARISON_COMPLETE"
    )
    market = pd.read_csv(market_path)
    delta = pd.read_csv(delta_path)
    univariate = pd.read_csv(univariate_path)
    routing = pd.read_csv(routing_path)
    alert = pd.read_csv(alert_path)

    q3_univariate = univariate.loc[
        univariate.score.eq("q3_ticker_surprise_positive_max")
    ].iloc[0]
    q3_route = routing.loc[
        routing.score.eq("q3_ticker_surprise_positive_max")
        & routing.queue_fraction.eq(0.1)
    ].iloc[0]
    primary_delta = delta.loc[
        delta.candidate.eq("all_memory_plus_ticker_surprise")
    ].iloc[0]
    checks["q3_zero_shot_alert_auc_ci_above_chance"] = bool(
        q3_route.date_block_auc_ci_low > 0.5
    )
    checks["incremental_market_r2_ci_above_zero"] = bool(
        primary_delta.ci_lower > 0
    )

    validation_table = pd.DataFrame(validation_comparison)
    validation_table.to_csv(
        FINAL / "validation_method_comparison.csv", index=False
    )
    all_suite = pd.concat(selected_metrics, ignore_index=True)
    all_suite.to_csv(
        FINAL / "controlled_selected_metrics.csv", index=False
    )

    test_rows = all_suite.loc[all_suite.suite.eq("test")].set_index("model")
    robust_pair = robustness.loc[
        robustness.metric.eq("pair_concordance")
    ].set_index("model")
    q25 = test_rows.loc["qwen25_7b"]
    q3 = test_rows.loc["qwen3_4b"]
    report = f"""# Nonlinear ticker-conditioned surprise: final report

## Bottom line

The controlled experiment supports a **ticker-conditioned parametric-surprise
readout**, not a universal or causal surprise axis. A relation/outcome-conditioned
MLP selected without controlled-test or market labels generalized to held-out firms
in both Qwen2.5-7B and Qwen3-4B. The market downstream result is narrower: the
Qwen3 score is useful as a zero-shot large-reaction alert ranker, while Qwen2.5 does
not replicate that market association and adding all ticker-surprise features to the
already strong supervised market model does not yield a reliable incremental R² gain.

## What was measured

For a real company/ticker and relation, the model first supplied an A/B/C outcome
distribution. For each controlled realized outcome the target was
`-log P_model(outcome | ticker, relation)`, standardized within relation × realized
outcome using training firms only. This makes the target firm-specific while holding
outcome semantics fixed. Hidden-state updates were also centered within those strata.

The candidate readouts were linear Ridge, absolute-value Ridge, and a one-hidden-layer
GELU MLP trained with Huber regression plus within-stratum pair-ranking loss.
Layers and hyperparameters were selected on validation firms; test firms and all
market outcomes were excluded from selection.

## Controlled held-out results

| Model | Test firms | Within-stratum Spearman | Pair concordance | Same-outcome conflict AUC |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | {int(q25.firms)} | {format_number(q25.within_group_spearman)} | {format_number(q25.pair_concordance)} | {format_number(q25.conflict_auc_same_outcome)} |
| Qwen3-4B | {int(q3.firms)} | {format_number(q3.within_group_spearman)} | {format_number(q3.pair_concordance)} | {format_number(q3.conflict_auc_same_outcome)} |

The firm-block 95% intervals for MLP pair concordance are
[{format_number(robust_pair.loc['qwen25_7b'].mlp_firm_boot_ci_low)},
{format_number(robust_pair.loc['qwen25_7b'].mlp_firm_boot_ci_high)}] for
Qwen2.5 and
[{format_number(robust_pair.loc['qwen3_4b'].mlp_firm_boot_ci_low)},
{format_number(robust_pair.loc['qwen3_4b'].mlp_firm_boot_ci_high)}] for Qwen3.
Stratified permutation p-values are
{format_number(robust_pair.loc['qwen25_7b'].stratified_permutation_p_one_sided, 4)}
and {format_number(robust_pair.loc['qwen3_4b'].stratified_permutation_p_one_sided, 4)}.

The MLP beat the best linear delta candidate on controlled validation in both models.
The paired held-out differences and their firm-block intervals are preserved in
`controlled_robustness/test_robustness.csv`. However, post-only MLP validation
performance was almost equal to delta MLP performance. Therefore the experiment does
not establish that pre/post differencing is uniquely necessary or that the score is a
pure belief-update mechanism.

## Real-news downstream

The frozen readout was applied to 9,376 eligible real-news articles. On the 383-row
July cohort, the Qwen3 positive article-max score had:

- Spearman with one-session absolute market-adjusted return:
  {format_number(q3_univariate.spearman)}, date-block 95% CI
  [{format_number(q3_univariate.date_block_ci_low)},
  {format_number(q3_univariate.date_block_ci_high)}].
- Large-reaction AUC: {format_number(q3_route.auc)}, date-block 95% CI
  [{format_number(q3_route.date_block_auc_ci_low)},
  {format_number(q3_route.date_block_auc_ci_high)}].
- Top-decile alert precision: {format_number(q3_route.queue_precision)} versus
  base rate {format_number(q3_route.base_rate)}; mean absolute-return lift:
  {format_number(q3_route.mean_absolute_return_lift)}×.

Against the same-model parent linear activation score, paired date-block ΔAUC was
{format_number(alert.loc[alert.baseline.eq('q3_activation_memory_violation'), 'delta_auc'].iloc[0])},
95% CI
[{format_number(alert.loc[alert.baseline.eq('q3_activation_memory_violation'), 'date_block_delta_ci_low'].iloc[0])},
{format_number(alert.loc[alert.baseline.eq('q3_activation_memory_violation'), 'date_block_delta_ci_high'].iloc[0])}].

Adding ticker-surprise features to the full supervised memory model changed July log-R²
by only {format_number(primary_delta.observed_delta_r2_log, 4)}, with 95% interval
[{format_number(primary_delta.ci_lower, 4)},
{format_number(primary_delta.ci_upper, 4)}]. This is not reliable incremental
forecasting evidence. The useful downstream interpretation is therefore a cheap,
market-label-free triage/ranking signal in Qwen3, not an improvement over the full
market-trained forecasting stack.

## Claim boundary

- Supported: a nonlinear ticker-conditioned parametric-surprise readout that
  generalizes across held-out firms in two models.
- Promising but model-specific: zero-shot large-market-reaction alert ranking in
  Qwen3.
- Not supported: a universal linear axis, delta-specific mechanism, causal control,
  cross-model market replication, or reliable incremental R² over the full model.
- Evaluation status: retrospective. The readout itself never used market labels, but
  the parent study had already opened the July outcome cohort.
"""
    (FINAL / "FINAL_REPORT.md").write_text(report)

    key_results = {
        "controlled_test": {
            model: {
                "within_group_spearman": float(
                    test_rows.loc[model].within_group_spearman
                ),
                "pair_concordance": float(
                    test_rows.loc[model].pair_concordance
                ),
                "same_outcome_conflict_auc": float(
                    test_rows.loc[model].conflict_auc_same_outcome
                ),
            }
            for model in MODELS
        },
        "q3_market_alert": {
            "spearman": float(q3_univariate.spearman),
            "auc": float(q3_route.auc),
            "auc_ci": [
                float(q3_route.date_block_auc_ci_low),
                float(q3_route.date_block_auc_ci_high),
            ],
            "top_decile_precision": float(q3_route.queue_precision),
            "base_rate": float(q3_route.base_rate),
            "mean_absolute_return_lift": float(
                q3_route.mean_absolute_return_lift
            ),
        },
        "incremental_r2": {
            "point": float(primary_delta.observed_delta_r2_log),
            "ci": [
                float(primary_delta.ci_lower),
                float(primary_delta.ci_upper),
            ],
        },
    }
    (FINAL / "key_results.json").write_text(
        json.dumps(key_results, indent=2, sort_keys=True) + "\n"
    )

    audit = {
        "status": (
            "TICKER_SURPRISE_NONLINEAR_COMPLETION_AUDIT_PASSED"
            if all(checks.values())
            or (
                all(
                    value
                    for key, value in checks.items()
                    if key != "incremental_market_r2_ci_above_zero"
                )
                and not checks["incremental_market_r2_ci_above_zero"]
            )
            else "TICKER_SURPRISE_NONLINEAR_COMPLETION_AUDIT_FAILED"
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "expected_negative_result": {
            "incremental_market_r2_ci_above_zero": False
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in source_files
        },
        "claim": (
            "ticker-conditioned parametric-surprise readout; Qwen3-only "
            "retrospective zero-shot market-alert utility"
        ),
    }
    (FINAL / "completion_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    if audit["status"] != "TICKER_SURPRISE_NONLINEAR_COMPLETION_AUDIT_PASSED":
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"completion audit failed: {failed}")
    print("TICKER_SURPRISE_NONLINEAR_COMPLETION_AUDIT_PASSED")


if __name__ == "__main__":
    main()
