"""Funcion objetivo comun (enfoque wrapper) para TODOS los metodos.

Una solucion es una mascara binaria x de longitud n (1 = la variable se usa).

    Monoobjetivo:  f(x) = alpha * err_CV(x) + (1 - alpha) * |x| / n      (minimizar)
    Biobjetivo:    F(x) = (err_CV(x), |x|)                              (minimizar ambos)

err_CV(x) = error de validacion cruzada estratificada (k particiones) de un clasificador
entrenado SOLO con las columnas elegidas y SOLO con los datos de entrenamiento.

Reglas para que la comparacion sea justa:
    - Todos los metodos usan esta clase y el mismo presupuesto (`budget`).
    - `n_evals` cuenta entrenamientos reales; las soluciones repetidas salen de la cache
      y no gastan presupuesto.
    - Al agotar el presupuesto se lanza `BudgetExhausted`; el metodo debe capturarla y
      devolver su mejor solucion.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class BudgetExhausted(Exception):
    """Se ha alcanzado el numero maximo de evaluaciones permitido."""


def make_classifier(name: str = "knn"):
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    raise ValueError(f"Clasificador desconocido: {name!r}")


class FitnessFunction:
    def __init__(self, X, y, classifier: str = "knn", cv: int = 5, alpha: float = 0.99,
                 budget: int | None = None, cv_seed: int = 0):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.n_features = self.X.shape[1]
        self.model = make_classifier(classifier)
        self.cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=cv_seed)
        self.alpha = alpha
        self.budget = budget
        self.n_evals = 0
        self._cache: dict[bytes, float] = {}
        # Mejor solucion vista y curva de convergencia [(n_evals, mejor_f), ...]
        self.best_f = np.inf
        self.best_mask: np.ndarray | None = None
        self.history: list[tuple[int, float]] = []

    # ------------------------------------------------------------------ error CV
    def cv_error(self, mask) -> float:
        """Error de validacion cruzada de la mascara (con cache y presupuesto)."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_features,):
            raise ValueError(f"La mascara debe tener longitud {self.n_features}")
        if not mask.any():
            return 1.0  # sin variables: peor error posible
        key = np.packbits(mask).tobytes()
        if key in self._cache:
            return self._cache[key]
        if self.budget is not None and self.n_evals >= self.budget:
            raise BudgetExhausted
        acc = cross_val_score(clone(self.model), self.X[:, mask], self.y, cv=self.cv).mean()
        self.n_evals += 1
        err = 1.0 - float(acc)
        self._cache[key] = err
        return err

    # ------------------------------------------------------------ monoobjetivo
    def __call__(self, mask) -> float:
        mask = np.asarray(mask, dtype=bool)
        err = self.cv_error(mask)
        f = self.alpha * err + (1 - self.alpha) * mask.sum() / self.n_features
        if f < self.best_f:
            self.best_f = f
            self.best_mask = mask.copy()
            self.history.append((self.n_evals, f))
        return f

    # -------------------------------------------------------------- biobjetivo
    def multi(self, mask) -> tuple[float, int]:
        """Devuelve (error CV, numero de variables), ambos a minimizar."""
        mask = np.asarray(mask, dtype=bool)
        return self.cv_error(mask), int(mask.sum())

    @property
    def remaining(self) -> float:
        return np.inf if self.budget is None else self.budget - self.n_evals


def test_accuracy(mask, X_train, y_train, X_test, y_test, classifier: str = "knn") -> float:
    """Entrena con TODO el train usando la mascara y mide la precision en test.

    Solo se llama al final, nunca durante la busqueda."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    model = make_classifier(classifier).fit(X_train[:, mask], y_train)
    return float(model.score(X_test[:, mask], y_test))
