
"""Busqueda tabu para seleccion de caracteristicas. Responsable: Persona A.

Se muestrean vecinos obtenidos invirtiendo un bit.
Se permite un movimiento tabu si mejora el mejor global anterior.
Tras patience iteraciones sin mejora se diversifica la busqueda.
"""
from __future__ import annotations

import numpy as np

from ..fitness import BudgetExhausted, FitnessFunction
from ._guard import StallGuard
from .base import Result, random_mask


def run(
    fitness: FitnessFunction,
    rng: np.random.Generator,
    n_neighbors: int = 20,
    tenure: int | None = None,
    patience: int = 30,
    perturb_frac: float = 0.05,
    p_init: float | None = None,
) -> Result:
    n = fitness.n_features
    tenure = max(5, int(round(np.sqrt(n)))) if tenure is None else tenure

    if n_neighbors < 1:
        raise ValueError("n_neighbors debe ser >= 1")
    if tenure < 1:
        raise ValueError("tenure debe ser >= 1")
    if patience < 1:
        raise ValueError("patience debe ser >= 1")
    if not 0 < perturb_frac <= 1:
        raise ValueError("perturb_frac debe estar en (0, 1]")

    guard = StallGuard(fitness)
    tabu_until = np.zeros(n, dtype=int)
    it = 0
    last_improve = 0

    try:
        x = random_mask(
            n, rng, rng.uniform(0.02, 0.5) if p_init is None else p_init
        )
        fitness(x)
        guard.tick()

        while True:
            it += 1
            global_best_before = fitness.best_f
            best_j = None
            best_fy = np.inf

            for j_raw in rng.choice(
                n, size=min(n_neighbors, n), replace=False
            ):
                j = int(j_raw)
                y = x.copy()
                y[j] = ~y[j]

                if not y.any():
                    continue

                # Guardar el mejor global antes de evaluar el candidato.
                best_before_candidate = fitness.best_f
                fy = fitness(y)
                guard.tick()

                is_tabu = tabu_until[j] > it
                aspiration = fy < best_before_candidate

                if is_tabu and not aspiration:
                    continue

                if fy < best_fy:
                    best_j = j
                    best_fy = fy

            # Detectar mejoras globales producidas por cualquier vecino.
            if fitness.best_f < global_best_before:
                last_improve = it

            if best_j is not None:
                x[best_j] = ~x[best_j]
                tabu_until[best_j] = it + tenure

            # Diversificación si no mejora el mejor global.
            if it - last_improve >= patience:
                n_flip = min(
                    n, max(1, int(round(perturb_frac * n)))
                )
                idx = rng.choice(n, size=n_flip, replace=False)
                x[idx] = ~x[idx]

                if not x.any():
                    x[rng.integers(n)] = True

                fitness(x)
                guard.tick()
                tabu_until[:] = 0
                last_improve = it

    except BudgetExhausted:
        pass

    return Result.from_fitness(fitness)
