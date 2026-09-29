import numpy as np

from ecole_lab.imitation import BranchingDataset, BranchingImitator


def test_branching_imitator_prefers_expert_like_candidate() -> None:
    rng=np.random.default_rng(4)
    x=np.vstack([rng.normal(-1,0.2,size=(80,3)),rng.normal(1,0.2,size=(80,3))])
    y=np.array([0]*80+[1]*80)
    model=BranchingImitator.fit(BranchingDataset.from_arrays(x,y))
    candidates=np.array([[-0.8,-1.0,-0.9],[0.9,1.0,0.8]])
    assert model.choose(candidates) == 1
