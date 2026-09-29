"""Depth-1 optimal classification tree learned with a one-hot MILP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class OptimalStump:
    feature: int | None
    threshold: float | None
    left_label: int
    right_label: int
    objective: float
    errors: int

    def predict(self, x: ArrayLike) -> NDArray[np.int64]:
        matrix = np.asarray(x, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("x must be two-dimensional")
        if self.feature is None:
            return np.full(matrix.shape[0], self.left_label, dtype=np.int64)
        assert self.threshold is not None
        return np.where(
            matrix[:, self.feature] <= self.threshold,
            self.left_label,
            self.right_label,
        ).astype(np.int64)


@dataclass(frozen=True)
class _Candidate:
    feature: int | None
    threshold: float | None
    left_label: int
    right_label: int
    errors: int
    complexity: int


def _candidates(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    *,
    max_thresholds: int,
) -> list[_Candidate]:
    candidates = [
        _Candidate(None, None, -1, -1, int(np.sum(y != -1)), 0),
        _Candidate(None, None, 1, 1, int(np.sum(y != 1)), 0),
    ]
    for feature in range(x.shape[1]):
        values = np.unique(x[:, feature])
        if values.size < 2:
            continue
        mids = 0.5 * (values[:-1] + values[1:])
        if mids.size > max_thresholds:
            positions = np.linspace(0, mids.size - 1, max_thresholds).round().astype(int)
            mids = np.unique(mids[positions])
        for threshold in mids:
            left = x[:, feature] <= threshold
            for left_label in (-1, 1):
                for right_label in (-1, 1):
                    pred = np.where(left, left_label, right_label)
                    candidates.append(
                        _Candidate(
                            feature,
                            float(threshold),
                            left_label,
                            right_label,
                            int(np.sum(pred != y)),
                            1,
                        )
                    )
    return candidates


def fit_optimal_stump(
    x: ArrayLike,
    y: ArrayLike,
    *,
    split_penalty: float = 0.01,
    max_thresholds: int = 32,
) -> OptimalStump:
    """Fit a globally optimal depth-1 classification tree over candidate thresholds.

    Binary variables select exactly one constant or split configuration. Misclassification
    counts are exact for every precomputed configuration; the split penalty provides the
    same accuracy-vs-complexity role used in optimal-tree formulations.
    """

    matrix = np.asarray(x, dtype=float)
    labels = np.asarray(y, dtype=np.int64)
    if matrix.ndim != 2 or labels.ndim != 1 or matrix.shape[0] != labels.size:
        raise ValueError("x/y shapes are invalid")
    if not np.all(np.isin(labels, [-1, 1])):
        raise ValueError("labels must be -1/+1")
    if split_penalty < 0.0:
        raise ValueError("split_penalty must be non-negative")
    if max_thresholds < 1:
        raise ValueError("max_thresholds must be positive")

    configs = _candidates(matrix, labels, max_thresholds=max_thresholds)
    objective = np.asarray(
        [candidate.errors / labels.size + split_penalty * candidate.complexity for candidate in configs],
        dtype=float,
    )
    result = milp(
        c=objective,
        integrality=np.ones(len(configs), dtype=int),
        bounds=Bounds(np.zeros(len(configs)), np.ones(len(configs))),
        constraints=LinearConstraint(
            np.ones((1, len(configs))),
            lb=np.array([1.0]),
            ub=np.array([1.0]),
        ),
        options={"disp": False},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"optimal stump MILP failed: {result.message}")

    chosen = int(np.argmax(result.x))
    candidate = configs[chosen]
    return OptimalStump(
        feature=candidate.feature,
        threshold=candidate.threshold,
        left_label=candidate.left_label,
        right_label=candidate.right_label,
        objective=float(result.fun),
        errors=candidate.errors,
    )
