from __future__ import annotations

import numpy as np

from scripts.news_cutoff_safe_llama2 import (
    center_labels,
    center_matrix,
    label_centering,
    normalize_rows,
    ticker_means_matrix,
    tune_alphas,
)


def test_train_ticker_centering_applies_to_future_rows() -> None:
    train = np.asarray(
        [[1.0, 3.0], [3.0, 5.0], [10.0, 20.0], [14.0, 24.0]], dtype=np.float32
    )
    train_tickers = np.asarray(["A", "A", "B", "B"])
    tickers = ["A", "B"]
    means = ticker_means_matrix(train, train_tickers, tickers)
    future = np.asarray([[4.0, 6.0], [13.0, 25.0]], dtype=np.float32)
    observed = center_matrix(future, np.asarray(["A", "B"]), tickers, means)
    np.testing.assert_allclose(observed, [[2.0, 2.0], [1.0, 3.0]])


def test_label_centering_never_uses_validation_means() -> None:
    train_y = np.asarray([1.0, 3.0, 10.0, 14.0])
    train_tickers = np.asarray(["A", "A", "B", "B"])
    tickers = ["A", "B"]
    means = label_centering(train_y, train_tickers, tickers)
    observed = center_labels(
        np.asarray([100.0, 200.0]), np.asarray(["A", "B"]), tickers, means
    )
    np.testing.assert_allclose(observed, [98.0, 188.0])


def test_row_normalization_has_unit_norm() -> None:
    values = normalize_rows(np.asarray([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32))
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), [1.0, 1.0])


def test_cuda_ridge_grid_matches_cpu_when_available() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(256, 24)).astype(np.float32)
    x_val = rng.normal(size=(96, 24)).astype(np.float32)
    coefficient = rng.normal(size=24)
    y_train = x_train @ coefficient + rng.normal(scale=0.2, size=len(x_train))
    y_val = x_val @ coefficient + rng.normal(scale=0.2, size=len(x_val))
    cpu = tune_alphas(x_train, y_train, x_val, y_val, workers=6, device="cpu")
    cuda = tune_alphas(x_train, y_train, x_val, y_val, workers=6, device="cuda")
    np.testing.assert_allclose(
        [row["validation_spearman"] for row in cuda],
        [row["validation_spearman"] for row in cpu],
        atol=1e-5,
    )


if __name__ == "__main__":
    test_train_ticker_centering_applies_to_future_rows()
    test_label_centering_never_uses_validation_means()
    test_row_normalization_has_unit_norm()
    test_cuda_ridge_grid_matches_cpu_when_available()
    print("CUTOFF-SAFE PIPELINE TESTS PASS")
