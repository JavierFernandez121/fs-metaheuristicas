"""Pruebas rapidas (segundos) con el dataset pequeno. Ejecutar: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fsmeta.baselines import BASELINES  # noqa: E402
from fsmeta.data import make_problem  # noqa: E402
from fsmeta.fitness import BudgetExhausted, FitnessFunction  # noqa: E402
from fsmeta.metaheuristics import METHODS  # noqa: E402
from fsmeta.stats import hypervolume_2d, pareto_front  # noqa: E402


@pytest.fixture(scope="module")
def problem():
    return make_problem("breast_cancer")


def test_split_is_fixed(problem):
    again = make_problem("breast_cancer")
    assert np.array_equal(problem.y_test, again.y_test)
    assert problem.n_features == 30


def test_fitness_cache_and_budget(problem):
    f = FitnessFunction(problem.X_train, problem.y_train, budget=2)
    m = np.zeros(30, dtype=bool)
    m[:5] = True
    a = f(m)
    b = f(m)                       # cache: no gasta presupuesto
    assert a == b and f.n_evals == 1
    f(~m)
    with pytest.raises(BudgetExhausted):
        f(np.ones(30, dtype=bool))
    assert f.multi(np.zeros(30, dtype=bool)) == (1.0, 0)


@pytest.mark.parametrize("name", list(METHODS) + list(BASELINES))
def test_methods_respect_interface(problem, name):
    """Cada metodo implementado devuelve una mascara valida sin pasarse del presupuesto."""
    budget = 40
    f = FitnessFunction(problem.X_train, problem.y_train, budget=budget)
    run = {**METHODS, **BASELINES}[name]
    try:
        res = run(f, np.random.default_rng(0))
    except NotImplementedError:
        pytest.skip(f"{name} aun no implementado")
    assert res.best_mask.dtype == bool and res.best_mask.shape == (30,)
    assert res.best_mask.any()
    assert res.n_evals <= budget


def test_pareto_and_hypervolume():
    pts = np.array([[0.1, 0.5], [0.2, 0.2], [0.3, 0.3], [0.4, 0.1]])
    front = pareto_front(pts)
    assert len(front) == 3                      # (0.3, 0.3) esta dominado por (0.2, 0.2)
    assert hypervolume_2d(pts, ref=(1.0, 1.0)) == pytest.approx(0.9 * 0.5 + 0.8 * 0.3 + 0.6 * 0.1)
