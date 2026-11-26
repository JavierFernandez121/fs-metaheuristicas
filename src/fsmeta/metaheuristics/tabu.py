"""Busqueda tabu para seleccion de caracteristicas.   Responsable: Persona A

Movimiento = invertir un bit. Con n=500 evaluar los 500 vecinos gastaria el presupuesto en
unas pocas iteraciones, asi que cada iteracion se muestrean `n_neighbors` vecinos al azar.
Se elige el mejor vecino no tabu (aunque empeore); el bit movido queda tabu durante `tenure`
iteraciones; aspiracion: un movimiento tabu se permite si mejora el mejor global. Si pasan
`patience` iteraciones sin mejorar el mejor global se diversifica perturbando un
`perturb_frac` de los bits y vaciando la lista tabu.

Pseudocodigo
------------
    x <- mascara aleatoria; evaluar
    mientras quede presupuesto:
        vecinos <- n_neighbors bits j al azar (x con el bit j invertido); evaluar cada uno
        descartar los tabu que no superen al mejor global (aspiracion)
        mover al mejor restante; j queda tabu hasta it + tenure
        si it - ultima_mejora > patience:  perturbar x; vaciar lista tabu
    devolver Result.from_fitness(fitness)

Parametros: n_neighbors=20, tenure=None (-> max(5, round(sqrt(n)))), patience=30,
perturb_frac=0.05, p_init=None
Complejidad por iteracion: O(n_neighbors * coste_evaluacion).
"""
from __future__ import annotations

import numpy as np

from ..fitness import BudgetExhausted, FitnessFunction
from ._guard import StallGuard
from .base import Result, random_mask


def run(fitness: FitnessFunction, rng: np.random.Generator, n_neighbors: int = 20,
        tenure: int | None = None, patience: int = 30, perturb_frac: float = 0.05,
        p_init: float | None = None) -> Result:
    n = fitness.n_features
    tenure = max(5, int(round(np.sqrt(n)))) if tenure is None else tenure
    guard = StallGuard(fitness)
    tabu_until = np.zeros(n, dtype=int)
    it = last_improve = 0
    try:
        x = random_mask(n, rng, rng.uniform(0.02, 0.5) if p_init is None else p_init)
        fitness(x)
        guard.tick()
        while True:
            it += 1
            best_before = fitness.best_f  # antes de evaluar: fitness() actualiza best_f al evaluar
            best_j, best_fy = None, np.inf
            for j in rng.choice(n, size=min(n_neighbors, n), replace=False):
                y = x.copy()
                y[j] = ~y[j]
                if not y.any():
                    continue
                fy = fitness(y)
                guard.tick()
                if tabu_until[j] > it and fy >= best_before:  # tabu sin aspiracion
                    continue
                if fy < best_fy:
                    best_j, best_fy = int(j), fy
            if fitness.best_f < best_before:  # algun vecino mejoro el mejor global
                last_improve = it
            if best_j is not None:
                x[best_j] = ~x[best_j]
                tabu_until[best_j] = it + tenure
            if it - last_improve > patience:  # diversificacion
                idx = rng.choice(n, size=max(2, int(perturb_frac * n)), replace=False)
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
