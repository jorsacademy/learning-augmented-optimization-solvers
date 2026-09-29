"""Synthetic sparsity/accuracy benchmark for integer scorecards."""

from __future__ import annotations

import json

import numpy as np

from .model import fit_integer_scorecard


def generate_data(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    latent = 2.2 * x[:, 0] - 1.6 * x[:, 1] + 0.9 * x[:, 2] + rng.normal(0.0, 1.0, n)
    y = np.where(latent >= 0.0, 1, -1).astype(np.int64)
    return x, y


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def run_experiment(seed: int = 13) -> dict[str, object]:
    x_train, y_train = generate_data(seed, 220)
    x_test, y_test = generate_data(seed + 1000, 3000)

    results: dict[str, object] = {}
    for penalty in (0.0, 0.01, 0.03, 0.06):
        model = fit_integer_scorecard(
            x_train,
            y_train,
            coefficient_bound=4,
            l0_penalty=penalty,
        )
        results[f"l0_{penalty:g}"] = {
            "intercept": model.intercept,
            "coefficients": model.coefficients.tolist(),
            "selected_features": model.selected_features,
            "training_margin_violations": model.margin_violations,
            "objective": model.objective,
            "train_accuracy": accuracy(y_train, model.predict(x_train)),
            "test_accuracy": accuracy(y_test, model.predict(x_test)),
        }
    return results


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
