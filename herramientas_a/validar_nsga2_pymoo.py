"""Validacion de NUESTRO NSGA-II frente al de pymoo (misma funcion objetivo, mismo presupuesto).

Requisito:  pip install pymoo
Uso (desde la raiz del repo):
    python herramientas_a/validar_nsga2_pymoo.py                       # 5 semillas, presupuesto 3000
    python herramientas_a/validar_nsga2_pymoo.py --seeds 3 --budget 1000
    python herramientas_a/validar_nsga2_pymoo.py --sin-pymoo            # solo nuestro NSGA-II

Como se compara (igual para los dos): cada metodo usa su PROPIA FitnessFunction con el mismo
presupuesto de entrenamientos reales; el frente de cada uno es el conjunto no dominado de TODAS
las mascaras que evaluo (se lee de la cache de la funcion objetivo). Metrica: hipervolumen 2D con
objetivos (error CV, n_vars/n) y punto de referencia (1, 1); mayor = mejor.
Solo se usa el conjunto de entrenamiento.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_train():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    return Xtr, ytr


def nondominated(P):
    """P: (m, 2) a minimizar. Devuelve los puntos no dominados, sin duplicados, ordenados por k."""
    P = np.unique(np.asarray(P, dtype=float), axis=0)
    le = (P[:, None, :] <= P[None, :, :]).all(axis=2)
    lt = (P[:, None, :] < P[None, :, :]).any(axis=2)
    dominated = (le & lt).any(axis=0)
    F = P[~dominated]
    return F[np.argsort(F[:, 1])]


def evaluated_front(fit):
    """Frente de Pareto de todas las mascaras evaluadas (lee la cache de fitness)."""
    n = fit.n_features
    pts = [(err, int(np.unpackbits(np.frombuffer(key, dtype=np.uint8))[:n].sum()))
           for key, err in fit._cache.items()]
    return nondominated(pts)


def hypervolume(F, n):
    """Hipervolumen 2D normalizado: objetivos (err, k/n), referencia (1, 1)."""
    P = np.column_stack([F[:, 0], F[:, 1] / n])
    P = P[np.argsort(P[:, 0])]
    hv, best_y = 0.0, 1.0
    for a, b in P:
        if b < best_y:
            hv += (1.0 - a) * (best_y - b)
            best_y = b
    return hv


def run_ours(Xtr, ytr, budget, seed, pop_size):
    from fsmeta.fitness import FitnessFunction
    from fsmeta.metaheuristics import METHODS
    fit = FitnessFunction(Xtr, ytr, budget=budget)
    r = METHODS["nsga2"](fit, np.random.default_rng(seed), pop_size=pop_size)
    final = np.array([[e, k] for e, k, _ in r.pareto]) if r.pareto else np.empty((0, 2))
    return evaluated_front(fit), final, fit.n_evals


def run_pymoo(Xtr, ytr, budget, seed, pop_size):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.ux import UniformCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.operators.sampling.rnd import BinaryRandomSampling
    from pymoo.optimize import minimize
    from fsmeta.fitness import BudgetExhausted, FitnessFunction

    fit = FitnessFunction(Xtr, ytr, budget=budget)
    n = fit.n_features

    class FSProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=n, n_obj=2, xl=0, xu=1, vtype=bool)

        def _evaluate(self, x, out, *args, **kwargs):
            err, k = fit.multi(np.asarray(x, dtype=bool))   # mascara vacia -> (1.0, 0), no entrena
            out["F"] = [err, k]

    algo = NSGA2(pop_size=pop_size, sampling=BinaryRandomSampling(),
                 crossover=UniformCrossover(prob=0.9), mutation=BitflipMutation(prob=1.0, prob_var=1.0 / n),
                 eliminate_duplicates=True)
    try:
        minimize(FSProblem(), algo, termination=("n_gen", 10 ** 6), seed=seed, verbose=False)
    except BudgetExhausted:
        pass                                              # presupuesto agotado: es la parada normal
    return evaluated_front(fit), fit.n_evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--sin-pymoo", action="store_true")
    a = ap.parse_args()
    if not a.sin_pymoo:
        try:
            import pymoo
            print("pymoo", pymoo.__version__)
        except ImportError:
            sys.exit("Falta pymoo:  pip install pymoo   (o usa --sin-pymoo)")

    Xtr, ytr = load_train()
    n = Xtr.shape[1]
    print(f"breast_cancer train: {Xtr.shape} | presupuesto {a.budget} | poblacion {a.pop} | {a.seeds} semillas\n")
    print(f"{'semilla':>7} | {'HV nuestro':>10} {'puntos':>6} | {'HV pymoo':>9} {'puntos':>6} | {'ratio':>6} | {'HV frente final run()':>21}")
    hv_o, hv_p, hv_f = [], [], []
    for s in range(a.seeds):
        t0 = time.time()
        Fo, Ff, _ = run_ours(Xtr, ytr, a.budget, s, a.pop)
        ho = hypervolume(Fo, n)
        hf = hypervolume(Ff, n) if len(Ff) else float("nan")
        hv_o.append(ho)
        hv_f.append(hf)
        if a.sin_pymoo:
            print(f"{s:>7} | {ho:10.4f} {len(Fo):6d} | {'-':>9} {'-':>6} | {'-':>6} | {hf:21.4f}   ({time.time()-t0:.0f}s)")
            continue
        Fp, _ = run_pymoo(Xtr, ytr, a.budget, s, a.pop)
        hp = hypervolume(Fp, n)
        hv_p.append(hp)
        print(f"{s:>7} | {ho:10.4f} {len(Fo):6d} | {hp:9.4f} {len(Fp):6d} | {ho/hp:6.3f} | {hf:21.4f}   ({time.time()-t0:.0f}s)")
    print(f"\nHV medio nuestro (todas las mascaras evaluadas): {np.mean(hv_o):.4f} +- {np.std(hv_o):.4f}")
    print(f"HV medio del frente final que devuelve run()   : {np.nanmean(hv_f):.4f} +- {np.nanstd(hv_f):.4f}"
          "   <- el que usan los experimentos de la memoria")
    if hv_p:
        print(f"HV medio pymoo  : {np.mean(hv_p):.4f} +- {np.std(hv_p):.4f}")
        print(f"ratio medio     : {np.mean(hv_o) / np.mean(hv_p):.3f}   (~1 = equivalentes; <0.95 = revisar nsga2.py)")
        print("Nota: pymoo inicializa cada bit con p=0.5 y el nuestro con densidad variable en [0.05, 0.5];"
              " pequenas diferencias son normales.")


if __name__ == "__main__":
    main()
