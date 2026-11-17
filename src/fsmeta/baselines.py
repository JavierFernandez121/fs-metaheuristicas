"""Metodos clasicos de comparacion.   Responsable: Persona B

Todos devuelven un `Result` igual que las metaheuristicas y usan la MISMA funcion
objetivo para elegir su mejor configuracion, asi la comparacion es directa.

    - all:     todas las variables (referencia trivial)
    - sfs:     seleccion secuencial hacia delante (voraz), con la funcion objetivo comun
    - mi:      filtro por informacion mutua; se elige el k (numero de variables) con mejor nota
    - lasso:   regresion logistica con penalizacion L1; se elige la C con mejor nota
"""
from __future__ import annotations

import numpy as np
import sklearn
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .fitness import BudgetExhausted, FitnessFunction
from .metaheuristics.base import Result


def all_features(fitness: FitnessFunction, rng: np.random.Generator) -> Result:
    fitness(np.ones(fitness.n_features, dtype=bool))
    return Result.from_fitness(fitness)


def sfs(fitness: FitnessFunction, rng: np.random.Generator, max_features: int | None = None) -> Result:
    """Seleccion hacia delante: anade en cada paso la variable que mas mejora f.

    Para cuando ninguna variable mejora, al llegar a max_features o al agotar el presupuesto."""
    n = fitness.n_features
    max_features = max_features or n
    mask = np.zeros(n, dtype=bool)
    current = np.inf
    try:
        while mask.sum() < max_features:
            best_j, best_f = None, current
            for j in np.flatnonzero(~mask):
                cand = mask.copy()
                cand[j] = True
                f = fitness(cand)
                if f < best_f:
                    best_j, best_f = j, f
            if best_j is None:
                break
            mask[best_j] = True
            current = best_f
    except BudgetExhausted:
        pass
    return Result.from_fitness(fitness)


def mutual_info(fitness: FitnessFunction, rng: np.random.Generator,
                ks: tuple[int, ...] = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300)) -> Result:
    scores = mutual_info_classif(fitness.X, fitness.y, random_state=int(rng.integers(2**31)))
    order = np.argsort(scores)[::-1]
    try:
        for k in ks:
            if k > fitness.n_features:
                break
            mask = np.zeros(fitness.n_features, dtype=bool)
            mask[order[:k]] = True
            fitness(mask)
    except BudgetExhausted:
        pass
    return Result.from_fitness(fitness)


def _l1_logreg(C: float) -> LogisticRegression:
    """Regresion logistica con penalizacion L1 (la sintaxis cambio en scikit-learn 1.8)."""
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C, max_iter=5000)
    return LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=5000)


def lasso(fitness: FitnessFunction, rng: np.random.Generator,
          Cs: np.ndarray = np.logspace(-3, 1, 20)) -> Result:
    Xs = StandardScaler().fit_transform(fitness.X)
    try:
        for C in Cs:
            model = _l1_logreg(C)
            model.fit(Xs, fitness.y)
            mask = np.any(model.coef_ != 0, axis=0)
            if mask.any():
                fitness(mask)
    except BudgetExhausted:
        pass
    return Result.from_fitness(fitness)


BASELINES = {
    "all": all_features,
    "sfs": sfs,
    "mi": mutual_info,
    "lasso": lasso,
}
