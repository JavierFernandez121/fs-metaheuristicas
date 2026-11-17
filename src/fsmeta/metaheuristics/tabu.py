"""Busqueda tabu (Glover, 1986).   Responsable: Persona A

Idea: en cada paso se mueve al mejor vecino aunque sea peor, pero prohibe durante
`tenure` iteraciones volver a tocar las variables cambiadas recientemente (lista tabu).
Excepcion (criterio de aspiracion): un movimiento tabu se permite si mejora la mejor
solucion global.

Pseudocodigo
------------
    x <- mascara aleatoria; tabu[j] <- 0 para todo j
    mientras quede presupuesto:
        candidatos <- `n_neighbors` bits j elegidos al azar (no todos: n puede ser 500)
        para cada j: y_j <- x con el bit j invertido; evaluar y_j
        elegir el mejor y_j tal que (tabu[j] <= iter) o (f(y_j) < mejor_global)
        x <- y_j; tabu[j] <- iter + tenure
    devolver Result.from_fitness(fitness)

Parametros a justificar en la memoria
-------------------------------------
    n_neighbors=20 (muestreo de la vecindad), tenure=~sqrt(n)
"""
from __future__ import annotations

import numpy as np

from ..fitness import BudgetExhausted, FitnessFunction  # noqa: F401
from .base import Result, random_mask  # noqa: F401


def run(fitness: FitnessFunction, rng: np.random.Generator, n_neighbors: int = 20,
        tenure: int | None = None) -> Result:
    # TODO (Persona A): implementar siguiendo el pseudocodigo de arriba.
    raise NotImplementedError("Busqueda tabu pendiente de implementar")
