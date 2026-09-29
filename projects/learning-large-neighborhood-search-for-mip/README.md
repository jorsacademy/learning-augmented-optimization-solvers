# Learning Large Neighborhood Search for MIP

> **Status:** planned research scaffold. The project is registered in the portfolio, but no benchmark result or paper reproduction is claimed yet.

## Research question

Can a learned policy choose effective destroy/fix neighborhoods for general mixed-integer programs, while an exact MIP solver remains responsible for re-optimizing each residual subproblem and validating every incumbent?

This project extends the monorepo from one-shot primal guidance such as Neural Diving to **iterative learned neighborhood selection**.

## Planned hybrid loop

```text
MILP instance
    |
    v
initial feasible incumbent
    |
    v
state / incumbent / variable features
    |
    v
learned neighborhood policy
    |
    +--> choose variables to unfix / destroy
    |
    v
residual MIP neighborhood
    |
    v
exact solver re-optimization
    |
    +--> improved feasible incumbent? -- yes --> repeat
    |                               \
    +-------------------------------- no --> fallback / next neighborhood
```

The learned policy controls **where to search**. It does not certify feasibility or optimality.

## Initial benchmark scope

Start with a controlled MILP family already represented in the monorepo, such as binary packing, before moving to a broader SCIP/MIPLIB-style evaluation.

Planned baselines:

1. random fixed-size neighborhoods;
2. incumbent-based variable fixing;
3. simple score-based destroy policies;
4. a local-branching-style classical neighborhood;
5. learned variable/neighborhood scoring;
6. Neural Diving as a one-shot primal-guidance reference.

## Learning targets

Candidate formulations include:

- supervised imitation of an expensive lookahead neighborhood expert;
- learning variable destroy scores from realized improvement;
- contextual bandit selection among neighborhood templates;
- reinforcement learning over repeated LNS iterations.

The first implementation should prefer the simplest target that permits controlled comparison and exact auditing.

## Evaluation contract

Report at least:

- incumbent objective versus wall time;
- primal integral or an equivalent anytime-quality metric;
- improvement per neighborhood solve;
- neighborhood solve time;
- final optimality gap under a fixed budget;
- feasibility rate of accepted incumbents;
- number and size of neighborhoods explored;
- generalization across instance size and distribution.

Matched classical LNS controls are mandatory. A learned policy should not be credited for improvements caused only by larger neighborhoods or more solver time.

## Safety / exactness boundary

- Every accepted incumbent must be independently feasibility-checked.
- Fixings define a search neighborhood; they are not asserted as globally correct variable assignments.
- The residual subproblem is solved by an exact mathematical-programming solver or a solver with an explicit termination/gap status.
- If the learned policy proposes an unusable neighborhood, fall back to a classical neighborhood.
- Global optimality is claimed only when the underlying exact solver establishes it; LNS alone is a primal heuristic.

## Relationship to neighboring projects

- `neural-diving-mip-solution-prediction`: one-shot learned partial assignment / primal guidance.
- `gnn-guided-generalized-assignment-variable-fixing-pytorch`: learned fixing decisions.
- `learning-to-select-primal-heuristics`: chooses among complete constructive heuristics.
- `learning-to-search-bnb-nodes`: controls branch-and-bound tree traversal.
- Neural LNS projects in routing/scheduling repositories: problem-specific LNS; this project targets general MILP neighborhoods.

## Planned milestones

1. Build deterministic LNS infrastructure around a small exact MILP benchmark.
2. Add random, score-based, and local-branching-style baselines.
3. Generate lookahead labels from candidate-neighborhood improvement.
4. Train a learned destroy/neighborhood policy.
5. Add iterative inference with strict solver-time matching.
6. Extend to SCIP/PySCIPOpt and a curated heterogeneous MILP benchmark.

## Research lineage

The project is motivated by Sonnerat et al., **Learning a Large Neighborhood Search Algorithm for Mixed Integer Programs** (2021), together with the broader Neural Diving / learning-augmented MIP literature. It is literature-informed and should not be described as a paper reproduction unless that claim is later justified explicitly.
