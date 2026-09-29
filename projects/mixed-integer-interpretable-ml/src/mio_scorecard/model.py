"""Sparse integer scorecards learned with an exact mixed-integer formulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class IntegerScorecard:
    intercept: int
    coefficients: NDArray[np.int64]
    objective: float
    margin_violations: int
    selected_features: int
    status: int
    message: str

    def scores(self, x: ArrayLike) -> NDArray[np.float64]:
        matrix = np.asarray(x, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.coefficients.size:
            raise ValueError("x must be a 2D matrix with the fitted feature dimension")
        return self.intercept + matrix @ self.coefficients.astype(float)

    def predict(self, x: ArrayLike) -> NDArray[np.int64]:
        return np.where(self.scores(x) >= 0.0, 1, -1).astype(np.int64)


def _validate_data(x: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    matrix = np.asarray(x, dtype=float)
    labels = np.asarray(y, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("x must be a non-empty 2D matrix")
    if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
        raise ValueError("y must be one-dimensional with one label per row")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("x must be finite")
    if not np.all(np.isin(labels, [-1, 1])):
        raise ValueError("labels must be encoded as -1 and +1")
    return matrix, labels


def fit_integer_scorecard(
    x: ArrayLike,
    y: ArrayLike,
    *,
    coefficient_bound: int = 3,
    l0_penalty: float = 0.02,
    margin: float = 1.0,
) -> IntegerScorecard:
    """Fit a sparse bounded-integer linear classifier by mixed-integer optimization.

    The objective is

        mean(margin_violation_i) + l0_penalty * number_of_selected_features.

    Each sample has a binary violation variable. Feature-selection binaries enforce
    beta_j = 0 whenever feature j is not selected.
    """

    matrix, labels = _validate_data(x, y)
    if coefficient_bound < 1:
        raise ValueError("coefficient_bound must be at least one")
    if not np.isfinite(l0_penalty) or l0_penalty < 0.0:
        raise ValueError("l0_penalty must be finite and non-negative")
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be finite and positive")

    n_samples, n_features = matrix.shape

    intercept_index = 0
    beta_start = 1
    violation_start = beta_start + n_features
    selected_start = violation_start + n_samples
    total_variables = selected_start + n_features

    objective = np.zeros(total_variables, dtype=float)
    objective[violation_start:selected_start] = 1.0 / n_samples
    objective[selected_start:] = l0_penalty

    lower = np.zeros(total_variables, dtype=float)
    upper = np.zeros(total_variables, dtype=float)

    lower[intercept_index] = -coefficient_bound
    upper[intercept_index] = coefficient_bound
    lower[beta_start:violation_start] = -coefficient_bound
    upper[beta_start:violation_start] = coefficient_bound
    lower[violation_start:selected_start] = 0.0
    upper[violation_start:selected_start] = 1.0
    lower[selected_start:] = 0.0
    upper[selected_start:] = 1.0

    integrality = np.ones(total_variables, dtype=int)

    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []

    for i in range(n_samples):
        row = np.zeros(total_variables, dtype=float)
        row[intercept_index] = labels[i]
        row[beta_start:violation_start] = labels[i] * matrix[i]

        score_abs_bound = coefficient_bound * (1.0 + float(np.sum(np.abs(matrix[i]))))
        big_m = margin + score_abs_bound
        row[violation_start + i] = big_m

        rows.append(row)
        lb.append(margin)
        ub.append(np.inf)

    for j in range(n_features):
        row_pos = np.zeros(total_variables, dtype=float)
        row_pos[beta_start + j] = 1.0
        row_pos[selected_start + j] = -coefficient_bound
        rows.append(row_pos)
        lb.append(-np.inf)
        ub.append(0.0)

        row_neg = np.zeros(total_variables, dtype=float)
        row_neg[beta_start + j] = -1.0
        row_neg[selected_start + j] = -coefficient_bound
        rows.append(row_neg)
        lb.append(-np.inf)
        ub.append(0.0)

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(np.stack(rows), np.asarray(lb), np.asarray(ub)),
        options={"disp": False},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"scorecard MILP failed: {result.message}")

    intercept = int(np.rint(result.x[intercept_index]))
    coefficients = np.rint(result.x[beta_start:violation_start]).astype(np.int64)
    scores = intercept + matrix @ coefficients.astype(float)
    margin_violations = int(np.sum(labels * scores < margin - 1e-8))
    selected_features = int(np.count_nonzero(coefficients))

    exact_objective = margin_violations / n_samples + l0_penalty * selected_features
    if not np.isclose(exact_objective, result.fun, atol=1e-7):
        raise RuntimeError(
            "decoded integer scorecard objective disagrees with MILP objective: "
            f"decoded={exact_objective}, solver={result.fun}"
        )

    return IntegerScorecard(
        intercept=intercept,
        coefficients=coefficients,
        objective=float(result.fun),
        margin_violations=margin_violations,
        selected_features=selected_features,
        status=int(result.status),
        message=str(result.message),
    )
