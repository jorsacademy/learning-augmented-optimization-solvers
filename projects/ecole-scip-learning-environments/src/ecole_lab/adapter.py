"""Optional Ecole branching environment adapter."""

from __future__ import annotations


def make_branching_env():
    try:
        import ecole
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the optional 'ecole' dependency with a compatible SCIP build") from exc

    return ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        information_function={
            "nb_nodes": ecole.reward.NNodes(),
            "time": ecole.reward.SolvingTime(),
        },
    )
