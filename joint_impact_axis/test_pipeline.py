from __future__ import annotations

import numpy as np
import pandas as pd

from joint_impact_axis.pipeline import (
    METHODS,
    fit_axis,
    safe_spearman,
    two_way_demean,
)
from joint_impact_axis.forecast import choose_and_fit


def synthetic_records(n_per_domain: int = 80) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    latent = np.concatenate(
        [rng.normal(size=n_per_domain), rng.normal(size=n_per_domain)]
    )
    domain = np.repeat(["news", "filing"], n_per_domain)
    values = rng.normal(scale=0.2, size=(2 * n_per_domain, 24))
    values[:, 3] += latent
    records = pd.DataFrame(
        {
            "domain": domain,
            "ticker": np.tile(np.repeat(["A", "B", "C", "D"], n_per_domain // 4), 2),
            "event_date": pd.date_range("2020-01-01", periods=2 * n_per_domain),
        }
    )
    return records, values.astype(np.float32), latent


def test_all_axis_methods_recover_shared_signal() -> None:
    records, values, labels = synthetic_records()
    for method in METHODS:
        axis = fit_axis(method, values, records, labels)
        assert safe_spearman(values @ axis, labels) > 0.7


def test_two_way_demean_removes_group_means() -> None:
    ticker = np.array(["A", "A", "B", "B"] * 4)
    date = np.repeat(np.array(["d1", "d2", "d3", "d4"]), 4)
    values = (ticker == "A").astype(float) * 3 + np.repeat(np.arange(4), 4)
    residual = two_way_demean(values, ticker, date)
    frame = pd.DataFrame({"value": residual, "ticker": ticker, "date": date})
    assert np.max(np.abs(frame.groupby("ticker")["value"].mean())) < 1e-8
    assert np.max(np.abs(frame.groupby("date")["value"].mean())) < 1e-8


def test_forecast_selection_runs_on_time_splits() -> None:
    rng = np.random.default_rng(11)
    n = 360
    feature = rng.normal(size=n)
    split = np.repeat(["train", "val", "test"], [180, 90, 90])
    frame = pd.DataFrame(
        {
            "log_hist_var_d": feature,
            "target": np.exp(0.4 * feature + rng.normal(scale=0.1, size=n)),
            "ticker": np.tile(["A", "B", "C"], n // 3),
            "domain": "news",
            "split": split,
            "date": pd.date_range("2020-01-01", periods=n),
        }
    )
    info, prediction = choose_and_fit(frame, ["log_hist_var_d"], "target")
    assert info["n_test"] == 90
    assert prediction.shape == (90,)
    assert np.all(prediction > 0)

    frozen, frozen_prediction = choose_and_fit(
        frame,
        ["log_hist_var_d"],
        "target",
        frozen_spec=(info["estimator"], info["alpha"]),
    )
    assert frozen["estimator"] == info["estimator"]
    assert frozen["alpha"] == info["alpha"]
    assert frozen["selection_source"] == "frozen_from_baseline"
    assert np.allclose(frozen_prediction, prediction)
