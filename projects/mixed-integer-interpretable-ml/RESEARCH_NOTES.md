# Research Notes

## Why this project belongs here

Most projects in this umbrella use machine learning to make decisions *inside an optimizer*. This project intentionally reverses that direction: the MIP solver constructs the machine-learning model itself.

It is kept here for now because it shares the same solver-engineering and mixed-integer modeling audience. If the portfolio later gains OCT, RiskSLIM-style calibration, rule lists, and other optimization-built ML models, those projects can be separated into a dedicated interpretable-ML-with-optimization umbrella.

## Objective semantics

The scorecard objective counts **margin violations**, not only sign mistakes. A correctly signed observation with score magnitude below the required margin still incurs a violation. The brute-force validation uses the same definition.

## Big-M safety

For coefficient bound `B`, the absolute score of sample `x_i` is bounded by

```text
B * (1 + sum_j |x_ij|),
```

including the intercept. The implementation therefore uses this bound plus the required margin as a valid sample-specific deactivation constant.
