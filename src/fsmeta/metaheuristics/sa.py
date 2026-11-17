"""Recocido simulado (Kirkpatrick et al., 1983).   Responsable: Persona A

Idea: busqueda local que acepta a veces soluciones peores, con una probabilidad que baja
a medida que la "temperatura" se enfria. Asi puede escapar de optimos locales.

Pseudocodigo
------------
    x <- mascara aleatoria; fx <- fitness(x); T <- T0
    mientras quede presupuesto:
        y <- flip(x, rng, k)            # vecino: invertir k bits (k=1..k_max)
        fy <- fitness(y); delta <- fy - fx
        si delta < 0 o rand() < exp(-delta / T):
            x, fx <- y, fy
        cada `iters_per_temp` iteraciones: T <- cooling * T
    devolver Result.from_fitness(fitness)

Parametros a justificar en la memoria
-------------------------------------
    T0: elegir para que al inicio se acepte ~80 % de los empeoramientos tipicos
        (p. ej. probar 50 vecinos, medir el delta medio d y T0 = -d / ln(0.8)),
    cooling=0.95, iters_per_temp=20, k_max=3
"""
from __future__ import annotations

import numpy as np

from ..fitness import BudgetExhausted, FitnessFunction  # noqa: F401
from .base import Result, flip, random_mask  # noqa: F401


def run(fitness: FitnessFunction, rng: np.random.Generator, T0: float | None = None,
        cooling: float = 0.95, iters_per_temp: int = 20, k_max: int = 3) -> Result:
    # TODO (Persona A): implementar siguiendo el pseudocodigo de arriba.
    raise NotImplementedError("Recocido simulado pendiente de implementar")
