# Ecole + SCIP Learning Environments

A bridge from the portfolio's transparent learning-to-branch implementations to a real SCIP control environment.

The project separates two layers:

1. a CI-safe imitation-learning core that consumes candidate feature vectors and learns an expert branching preference;
2. an optional Ecole adapter using `ecole.environment.Branching` with `NodeBipartite` observations and SCIP-backed rewards.

This avoids pretending a synthetic classifier is a solver integration while keeping the repository testable on runners without SCIP/Ecole binaries.

A real experiment should collect strong-branching expert actions, train on NodeBipartite observations, and report nodes, wall time, primal/dual bounds and generalization across instance families. Ecole remains the authoritative environment; learned actions must never redefine feasibility or optimality.
