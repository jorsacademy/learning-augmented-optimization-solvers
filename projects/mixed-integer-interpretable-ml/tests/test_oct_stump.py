import numpy as np

from mio_scorecard.oct_stump import fit_optimal_stump


def test_optimal_stump_finds_perfect_threshold() -> None:
    x=np.array([[-2.0,0],[-1.0,1],[-0.2,0],[0.4,1],[1.0,0],[2.0,1]])
    y=np.array([-1,-1,-1,1,1,1])
    model=fit_optimal_stump(x,y)
    assert model.feature == 0
    assert model.training_errors == 0
    assert np.array_equal(model.predict(x),y)
