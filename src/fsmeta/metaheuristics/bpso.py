"""PSO binario (Kennedy y Eberhart, 1997).   Responsable: Persona A

Idea: un enjambre de particulas; cada una recuerda su mejor posicion (pbest) y conoce
la mejor del enjambre (gbest). La velocidad real se convierte en probabilidad de que
cada bit valga 1 mediante una funcion sigmoide.

Pseudocodigo
------------
    x_i <- mascaras aleatorias; v_i <- 0 (vectores reales de longitud n)
    pbest_i <- x_i; gbest <- mejor de todas
    mientras quede presupuesto:
        para cada particula i:
            r1, r2 <- uniformes(0,1) de longitud n
            v_i <- w*v_i + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)
            v_i <- recortar(v_i, -v_max, v_max)
            x_i[j] <- 1 si rand() < sigmoide(v_i[j]) si no 0
            evaluar x_i; actualizar pbest_i y gbest
    devolver Result.from_fitness(fitness)

Parametros a justificar en la memoria
-------------------------------------
    n_particles=30, w=0.7 (inercia; puede bajar linealmente de 0.9 a 0.4),
    c1=c2=1.5, v_max=4.0
"""
from __future__ import annotations

import numpy as np

from ..fitness import BudgetExhausted, FitnessFunction  # noqa: F401
from .base import Result, random_mask  # noqa: F401


def run(fitness: FitnessFunction, rng: np.random.Generator, n_particles: int = 30,
        w: float = 0.7, c1: float = 1.5, c2: float = 1.5, v_max: float = 4.0) -> Result:
    # TODO (Persona A): implementar siguiendo el pseudocodigo de arriba.
    raise NotImplementedError("PSO binario pendiente de implementar")
