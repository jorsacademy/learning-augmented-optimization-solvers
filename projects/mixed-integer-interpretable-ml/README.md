# Mixed-Integer Interpretable Machine Learning

A compact research project showing the **reverse direction** of learning-augmented optimization: instead of using ML to control a solver, mixed-integer optimization is used to construct an interpretable machine-learning model.

The initial model is a sparse integer scorecard classifier with bounded coefficients, explicit 0–1 margin violations, and an L0 feature-selection penalty.

## Research question

> Can we learn a small, directly inspectable integer scoring rule by optimizing classification errors and sparsity in one mixed-integer model?

The learned classifier is

```text
score(x) = beta_0 + sum_j beta_j x_j
prediction = +1 if score(x) >= 0 else -1,
```

with small integer coefficients.

## Mixed-integer formulation

For each training sample `i`, binary variable `z_i` records whether the required classification margin may be violated. Binary variable `s_j` records whether feature `j` is selected.

```text
minimize    (1/n) sum_i z_i + lambda sum_j s_j

subject to  y_i (beta_0 + x_i^T beta) + M_i z_i >= margin   for all i
            -B s_j <= beta_j <= B s_j                       for all j
            beta_0, beta_j integer
            z_i, s_j binary.
```

The implementation computes a sample-specific valid big-M bound from the coefficient box and feature magnitudes.

The L0 term is represented exactly: `s_j=0` forces `beta_j=0`; a nonzero integer coefficient requires `s_j=1`.

## Exactness check

The unit test includes a tiny two-feature instance where every permitted integer intercept/coefficient combination is enumerated. The best brute-force objective is compared directly with the SciPy/HiGHS MILP solution.

That test is intended to verify the formulation, not to claim scalability to large interpretable-ML problems.

## Benchmark

The synthetic benchmark fits scorecards across several L0 penalties and reports:

- integer intercept and coefficients;
- selected-feature count;
- training margin violations;
- training accuracy;
- held-out accuracy;
- exact MILP objective.

Run:

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python -m mio_scorecard.experiment
```

## Relationship to OCT and RiskSLIM

This project is motivated by the broader literature on using modern mixed-integer optimization to fit interpretable models.

- **Optimal Classification Trees (OCT)** use MIO to optimize an entire decision tree rather than greedily selecting splits.
- **RiskSLIM** learns sparse risk scores with small integer coefficients using a substantially richer optimization model, including logistic-loss calibration.

This v0.1 is **not** an implementation of OCT or RiskSLIM. It isolates a simpler exact scorecard MILP so the discrete loss, integer coefficients, sparsity indicators, big-M logic, and optimality check are transparent.

## Scope boundary

Current limitations:

- binary classification only;
- bounded integer coefficients;
- linear scorecards only;
- margin-violation loss rather than logistic loss;
- synthetic benchmark;
- no monotonicity/fairness/domain-specific scorecard constraints yet.

Natural extensions include scorecard feature groups, monotonicity, asymmetric misclassification costs, fairness constraints, piecewise-linear calibration loss, and an OCT project as a separate formulation.

## References

- Bertsimas, D. and Dunn, J. (2017), "Optimal Classification Trees," *Machine Learning* 106:1039–1082.
- Ustun, B. and Rudin, C., "Learning Optimized Risk Scores," *Journal of Machine Learning Research* 20.

## License

This project inherits the umbrella repository's non-commercial/source-available licensing terms.
