"""Protección contra bucles infinitos.

En fitness.py las soluciones repetidas salen de la caché y NO gastan presupuesto. Si un
método converge y solo vuelve a proponer máscaras ya vistas, `n_evals` no avanza y el
bucle no terminaría nunca. Este guardián corta la búsqueda tras `patience` evaluaciones
seguidas sin ningún entrenamiento nuevo.
"""
from ..fitness import BudgetExhausted


class StallGuard:
    def __init__(self, fitness, patience: int = 3000):
        self.fitness = fitness
        self.patience = patience
        self._last = fitness.n_evals
        self._stalled = 0

    def tick(self):
        """Llamar una vez por cada solución evaluada."""
        if self.fitness.n_evals != self._last:
            self._last, self._stalled = self.fitness.n_evals, 0
        else:
            self._stalled += 1
            if self._stalled > self.patience:
                raise BudgetExhausted


def progress(fitness, default_budget: int = 3000) -> float:
    """Fracción de presupuesto gastada (0..1), para calendarios de inercia/temperatura."""
    budget = fitness.budget or default_budget
    return min(1.0, fitness.n_evals / budget)
