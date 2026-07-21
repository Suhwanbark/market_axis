from __future__ import annotations

import math

import numpy as np


def parkinson_variance(high: float, low: float) -> float:
    return math.log(high / low) ** 2 / (4.0 * math.log(2.0))


def impact_label(
    firm_post: float,
    firm_pre: float,
    market_post: float,
    market_pre: float,
) -> float:
    return math.log(firm_post / firm_pre) - math.log(market_post / market_pre)


def qlike(actual_variance: float, predicted_variance: float) -> float:
    ratio = actual_variance / predicted_variance
    return ratio - math.log(ratio) - 1.0


def test_parkinson_is_nonnegative() -> None:
    assert parkinson_variance(105.0, 100.0) > 0
    assert parkinson_variance(100.0, 100.0) == 0


def test_market_adjusted_expansion_example() -> None:
    observed = impact_label(2.0, 1.0, 1.2, 1.0)
    assert np.isclose(observed, math.log(2.0 / 1.2))


def test_qlike_is_zero_at_exact_prediction() -> None:
    assert np.isclose(qlike(0.001, 0.001), 0.0)
    assert qlike(0.002, 0.001) > 0


if __name__ == "__main__":
    test_parkinson_is_nonnegative()
    test_market_adjusted_expansion_example()
    test_qlike_is_zero_at_exact_prediction()
    print("CORE FORMULA TESTS PASS")
