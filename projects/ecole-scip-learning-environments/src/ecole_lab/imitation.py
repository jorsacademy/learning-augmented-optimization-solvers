"""CI-safe imitation-learning core for SCIP branching observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class BranchingDataset:
    features: NDArray[np.float64]
    labels: NDArray[np.int64]

    @classmethod
    def from_arrays(cls, features: ArrayLike, labels: ArrayLike) -> BranchingDataset:
        x=np.asarray(features,dtype=float)
        y=np.asarray(labels,dtype=np.int64)
        if x.ndim != 2 or y.shape != (x.shape[0],):
            raise ValueError("invalid branching dataset shapes")
        if np.unique(y).size < 2:
            raise ValueError("at least two candidate classes are required")
        return cls(x,y)


@dataclass
class BranchingImitator:
    model: LogisticRegression

    @classmethod
    def fit(cls,data:BranchingDataset) -> BranchingImitator:
        model=LogisticRegression(max_iter=1000).fit(data.features,data.labels)
        return cls(model)

    def score(self,features:ArrayLike) -> NDArray[np.float64]:
        x=np.asarray(features,dtype=float)
        return self.model.predict_proba(x)[:,1]

    def choose(self,features:ArrayLike) -> int:
        return int(np.argmax(self.score(features)))
