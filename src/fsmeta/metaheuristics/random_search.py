"""Busqueda aleatoria: linea base minima. Si un metodo no le gana a esto, no aporta nada.

Tambien sirve como ejemplo completo de como cumplir la interfaz comun (ver base.py).
"""
from __future__ import annotations

import numpy as np

from ..fitness import BudgetExhausted, FitnessFunction
from .base import Result, random_mask


def run(fitness: FitnessFunction, rng: np.random.Generator, p: float | None = None) -> Result:
    """Evalua mascaras aleatorias hasta agotar el presupuesto.

    p: probabilidad de incluir cada variable. Si es None se sortea en cada iteracion
       (asi se exploran tanto subconjuntos pequenos como grandes).
    """
    if fitness.budget is None:
        raise ValueError("La busqueda aleatoria necesita un presupuesto (budget)")
    n = fitness.n_features
    try:
        while True:
            prob = rng.uniform(0.02, 0.6) if p is None else p
            fitness(random_mask(n, rng, prob))
    except BudgetExhausted:
        pass
    return Result.from_fitness(fitness)
