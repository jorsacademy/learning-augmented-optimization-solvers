"""Exact mixed-integer selection of an interpretable depth-one classification tree."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class OptimalStump:
    feature: int
    threshold: float
    left_label: int
    right_label: int
    training_errors: int
    objective: float

    def predict(self,x:ArrayLike) -> NDArray[np.int64]:
        matrix=np.asarray(x,dtype=float)
        return np.where(
            matrix[:,self.feature] <= self.threshold,
            self.left_label,
            self.right_label,
        ).astype(np.int64)


def fit_optimal_stump(x:ArrayLike,y:ArrayLike) -> OptimalStump:
    """Choose the globally best depth-one tree over all observed midpoint splits.

    A binary variable selects exactly one candidate (feature, threshold, leaf labels).
    The objective coefficient of each candidate is its empirical misclassification
    rate, making this a compact exact MIO formulation of an OCT-style stump.
    """
    matrix=np.asarray(x,dtype=float)
    labels=np.asarray(y,dtype=np.int64)
    if matrix.ndim != 2 or labels.shape != (matrix.shape[0],):
        raise ValueError("invalid x/y shapes")
    if not np.all(np.isin(labels,[-1,1])):
        raise ValueError("labels must be -1/+1")

    candidates=[]
    errors=[]
    for j in range(matrix.shape[1]):
        values=np.unique(matrix[:,j])
        if values.size < 2:
            continue
        for threshold in (values[:-1]+values[1:])/2.0:
            left=matrix[:,j] <= threshold
            for left_label,right_label in ((-1,1),(1,-1)):
                pred=np.where(left,left_label,right_label)
                err=int(np.sum(pred != labels))
                candidates.append((j,float(threshold),left_label,right_label))
                errors.append(err)
    if not candidates:
        raise ValueError("no candidate split")

    n=len(candidates)
    c=np.asarray(errors,dtype=float)/len(labels)
    result=milp(
        c=c,
        integrality=np.ones(n,dtype=int),
        bounds=Bounds(np.zeros(n),np.ones(n)),
        constraints=LinearConstraint(np.ones((1,n)),[1.0],[1.0]),
        options={"disp":False},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"stump MILP failed: {result.message}")
    k=int(np.argmax(result.x))
    j,t,ll,rl=candidates[k]
    return OptimalStump(j,t,ll,rl,errors[k],float(result.fun))
