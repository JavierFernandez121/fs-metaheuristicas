"""Busqueda aleatoria: linea base minima. Si un metodo no le gana a esto, no aporta nada.

Tambien sirve como ejemplo completo de como cumplir la interfaz comun (ver base.py).

Las mascaras repetidas salen de la cache y no gastan presupuesto. Si hay menos combinaciones
posibles que presupuesto (pocas variables), el bucle no terminaria nunca; por eso se usa el
mismo guardian que en las metaheuristicas (_guard.StallGuard): corta tras `patience`
evaluaciones seguidas sin ningun entrenamiento nuevo.
"""
from __future__ import annotations

import numpy as np

from ..fitness import BudgetExhausted, FitnessFunction
from ._guard import StallGuard
from .base import Result, random_mask


def run(fitness: FitnessFunction, rng: np.random.Generator, p: float | None = None,
        patience: int = 3000) -> Result:
    """Evalua mascaras aleatorias hasta agotar el presupuesto.

    p: probabilidad de incluir cada variable. Si es None se sortea en cada iteracion
       (asi se exploran tanto subconjuntos pequenos como grandes).
    patience: evaluaciones seguidas sin entrenamiento nuevo antes de parar.
    """
    if fitness.budget is None:
        raise ValueError("La busqueda aleatoria necesita un presupuesto (budget)")
    n = fitness.n_features
    guard = StallGuard(fitness, patience=patience)
    try:
        while True:
            prob = rng.uniform(0.02, 0.6) if p is None else p
            fitness(random_mask(n, rng, prob))
            guard.tick()
    except BudgetExhausted:
        pass
    return Result.from_fitness(fitness)
