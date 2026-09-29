import itertools

import numpy as np
import pytest

from mio_scorecard import fit_integer_scorecard


def _bruteforce_objective(
    x: np.ndarray,
    y: np.ndarray,
    coefficient_bound: int,
    l0_penalty: float,
    margin: float,
) -> float:
    best = float("inf")
    values = range(-coefficient_bound, coefficient_bound + 1)
    for params in itertools.product(values, repeat=x.shape[1] + 1):
        intercept = params[0]
        coefficients = np.asarray(params[1:], dtype=int)
        scores = intercept + x @ coefficients
        violations = np.mean(y * scores < margin)
        selected = np.count_nonzero(coefficients)
        best = min(best, float(violations + l0_penalty * selected))
    return best


def test_milp_matches_bruteforce_on_tiny_instance() -> None:
    x = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1], [-1, 0], [0, -1]],
        dtype=float,
    )
    y = np.array([-1, 1, -1, 1, -1, -1], dtype=int)
    penalty = 0.05

    model = fit_integer_scorecard(x, y, coefficient_bound=2, l0_penalty=penalty)
    brute = _bruteforce_objective(x, y, 2, penalty, 1.0)

    assert np.isclose(model.objective, brute, atol=1e-9)


def test_decoded_objective_matches_model_fields() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 3))
    y = np.where(1.5 * x[:, 0] - x[:, 1] >= 0.0, 1, -1)
    penalty = 0.03

    model = fit_integer_scorecard(x, y, coefficient_bound=3, l0_penalty=penalty)
    expected = model.margin_violations / len(y) + penalty * model.selected_features

    assert np.isclose(model.objective, expected, atol=1e-9)
    assert model.selected_features == np.count_nonzero(model.coefficients)


def test_predictions_are_binary_labels() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(24, 2))
    y = np.where(x[:, 0] + 0.5 * x[:, 1] >= 0.0, 1, -1)

    model = fit_integer_scorecard(x, y, coefficient_bound=2)
    assert set(np.unique(model.predict(x))).issubset({-1, 1})


def test_invalid_label_encoding_is_rejected() -> None:
    x = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError, match=r"encoded as -1 and \+1"):
        fit_integer_scorecard(x, y)
