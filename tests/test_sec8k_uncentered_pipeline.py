from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge

from scripts.news_cutoff_safe_llama2 import safe_spearman
from scripts.sec8k_uncentered_axis import masked_mean_layers, tune_dual_cuda
from scripts.sec8k_uncentered_prepare import document_priority, parse_ticker


def test_sec_ticker_parser_uses_first_plain_equity_symbol() -> None:
    assert parse_ticker("['QBTS-WT', 'QBTS']") == "QBTS"
    assert parse_ticker("['VSEEW']") == "VSEEW"
    assert parse_ticker("not a list") is None


def test_sec_preferred_document_priority() -> None:
    assert document_priority("EX-99.1") < document_priority("EX-99.2")
    assert document_priority("EX-99.2") < document_priority("8-K")


def test_llama_pooling_excludes_bos_and_padding() -> None:
    states = (
        torch.tensor([[[100.0], [2.0], [4.0], [999.0]], [[100.0], [3.0], [5.0], [7.0]]]),
        torch.tensor([[[200.0], [4.0], [8.0], [999.0]], [[200.0], [6.0], [10.0], [14.0]]]),
    )
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    observed = masked_mean_layers(states, mask)
    np.testing.assert_allclose(observed.numpy().squeeze(-1), [[6.0], [10.0]])


def test_dual_cuda_ridge_sweep_matches_sklearn() -> None:
    if not torch.cuda.is_available():
        return
    rng = np.random.default_rng(31)
    x_train = rng.normal(size=(80, 160)).astype(np.float32)
    x_val = rng.normal(size=(40, 160)).astype(np.float32)
    y_train = rng.normal(size=80)
    y_val = rng.normal(size=40)
    observed = tune_dual_cuda(x_train, y_train, x_val, y_val)
    expected = []
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-7).fit(x_train, y_train)
        expected.append(safe_spearman(model.predict(x_val), y_val))
    np.testing.assert_allclose(
        [row["validation_spearman"] for row in observed], expected, atol=1e-5
    )
