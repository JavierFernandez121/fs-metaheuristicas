"""Pruebas de los 5 metodos de Persona A.  Ejecutar:  python -m tests.test_persona_a"""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # para importar fsmeta
from sklearn.datasets import load_breast_cancer, make_classification
from fsmeta.fitness import FitnessFunction
from fsmeta.metaheuristics import METHODS as _ALL_METHODS

# Solo los metodos de Persona A (el registro del repo incluye tambien 'random' y otros)
METHODS = {k: _ALL_METHODS[k] for k in ("ga", "bpso", "sa", "tabu", "nsga2")}
from fsmeta.metaheuristics.nsga2 import dominates, fast_non_dominated_sort, crowding_distance


def _fit(X, y, budget):
    return FitnessFunction(X, y, budget=budget)


def test_nsga2_helpers():
    rng = np.random.default_rng(0)
    F = rng.random((60, 2))
    rank, fronts = fast_non_dominated_sort(F)
    brute = [i for i in range(60) if not any(dominates(F[j], F[i]) for j in range(60))]
    assert sorted(fronts[0]) == sorted(brute)
    assert np.isinf(crowding_distance(F[fronts[0]])).sum() >= 2


def test_all_methods_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    for name, run in METHODS.items():
        fit = _fit(X, y, 300)
        t0 = time.time()
        r = run(fit, np.random.default_rng(1))
        assert r.n_evals <= 300 and r.best_mask.dtype == bool and r.best_mask.any(), name
        assert np.isfinite(r.best_f) and len(r.history) > 0, name
        fs = [h[1] for h in r.history]
        assert all(a >= b for a, b in zip(fs, fs[1:])), name   # historial monotono
        print(f"{name:6s} f={r.best_f:.4f} vars={r.best_mask.sum():2d} evals={r.n_evals} {time.time()-t0:.1f}s")


def test_nsga2_pareto_valid():
    X, y = load_breast_cancer(return_X_y=True)
    r = METHODS["nsga2"](_fit(X, y, 400), np.random.default_rng(2))
    assert r.pareto and all(len(t) == 3 for t in r.pareto)
    P = np.array([[e, k] for e, k, _ in r.pareto])
    for i in range(len(P)):
        for j in range(len(P)):
            assert i == j or not dominates(P[j], P[i])
    print("pareto:", [(round(e, 3), k) for e, k, _ in r.pareto])


def test_reproducible():
    X, y = load_breast_cancer(return_X_y=True)
    a = METHODS["ga"](_fit(X, y, 150), np.random.default_rng(5))
    b = METHODS["ga"](_fit(X, y, 150), np.random.default_rng(5))
    assert (a.best_mask == b.best_mask).all() and a.best_f == b.best_f


def test_terminates_when_cache_exhausts_search_space():
    """Con 6 variables solo hay 63 mascaras: el presupuesto (500) nunca se gasta del todo.
    Sin el guardian los metodos se quedarian en bucle infinito."""
    X, y = make_classification(n_samples=120, n_features=6, n_informative=3, random_state=0)
    for name, run in METHODS.items():
        t0 = time.time()
        r = run(_fit(X, y, 500), np.random.default_rng(0))
        assert time.time() - t0 < 60 and r.n_evals <= 63, (name, r.n_evals)
    print("OK guardian (espacio pequeno)")


def test_finds_informative_features():
    X, y = make_classification(n_samples=500, n_features=60, n_informative=5, n_redundant=0,
                               shuffle=False, class_sep=2.0, random_state=0)
    for name, run in METHODS.items():
        r = run(_fit(X, y, 800), np.random.default_rng(3))
        good = int(r.best_mask[:5].sum())
        print(f"{name:6s} utiles={good}/5 vars={r.best_mask.sum():2d} f={r.best_f:.4f}")


if __name__ == "__main__":
    for t in [test_nsga2_helpers, test_all_methods_breast_cancer, test_nsga2_pareto_valid,
              test_reproducible, test_terminates_when_cache_exhausts_search_space,
              test_finds_informative_features]:
        t(); print("OK", t.__name__)
