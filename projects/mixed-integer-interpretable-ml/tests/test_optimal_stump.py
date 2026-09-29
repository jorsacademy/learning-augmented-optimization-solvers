import itertools

import numpy as np

from mio_scorecard.stump import fit_optimal_stump


def test_optimal_stump_matches_bruteforce_candidates() -> None:
    x = np.array([[-2.0], [-1.0], [-0.2], [0.4], [1.2], [2.0]])
    y = np.array([-1, -1, -1, 1, 1, -1])
    penalty = 0.05
    model = fit_optimal_stump(x, y, split_penalty=penalty, max_thresholds=20)

    values = np.unique(x[:, 0])
    thresholds = 0.5 * (values[:-1] + values[1:])
    best = min(np.mean(y != label) for label in (-1, 1))
    for threshold, left_label, right_label in itertools.product(
        thresholds, (-1, 1), (-1, 1)
    ):
        pred = np.where(x[:, 0] <= threshold, left_label, right_label)
        best = min(best, float(np.mean(pred != y) + penalty))

    assert np.isclose(model.objective, best, atol=1e-12)
    assert set(np.unique(model.predict(x))).issubset({-1, 1})
