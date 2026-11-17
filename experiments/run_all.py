"""Lanza todos los experimentos y guarda una fila de CSV por ejecucion.

Uso:
    python experiments/run_all.py                                  # todo lo de config.py
    python experiments/run_all.py --datasets breast_cancer --methods random mi --seeds 0 1 2 --budget 200

Salidas:
    results/<dataset>.csv                  una fila por (metodo, semilla)
    results/histories/<dataset>_<metodo>_<semilla>.json   curva de convergencia (+ frente de Pareto)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import config  # noqa: E402
from fsmeta.baselines import BASELINES  # noqa: E402
from fsmeta.data import make_problem  # noqa: E402
from fsmeta.fitness import FitnessFunction, test_accuracy  # noqa: E402
from fsmeta.metaheuristics import METHODS  # noqa: E402

ALL_METHODS = {**METHODS, **BASELINES}
RESULTS = ROOT / "results"


def run_one(problem, method, seed, budget):
    fitness = FitnessFunction(problem.X_train, problem.y_train, classifier=config.CLASSIFIER,
                              cv=config.CV_FOLDS, alpha=config.ALPHA, budget=budget)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    res = ALL_METHODS[method](fitness, rng, **config.PARAMS.get(method, {}))
    elapsed = time.perf_counter() - t0
    mask = res.best_mask
    row = {
        "dataset": problem.name, "method": method, "seed": seed,
        "cv_error": fitness.cv_error(mask),          # sale de la cache, no gasta presupuesto
        "f": res.best_f,
        "test_acc": test_accuracy(mask, problem.X_train, problem.y_train,
                                  problem.X_test, problem.y_test, config.CLASSIFIER),
        "n_selected": int(mask.sum()), "n_features": problem.n_features,
        "n_evals": res.n_evals, "time_s": round(elapsed, 3),
        "selected": " ".join(map(str, np.flatnonzero(mask))),
    }
    extra = {"history": res.history}
    if res.pareto is not None:
        extra["pareto"] = [
            {"cv_error": e, "n_selected": k,
             "test_acc": test_accuracy(m, problem.X_train, problem.y_train, problem.X_test,
                                       problem.y_test, config.CLASSIFIER)}
            for e, k, m in res.pareto
        ]
    return row, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=config.DATASETS)
    ap.add_argument("--methods", nargs="+", default=config.METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    ap.add_argument("--budget", type=int, default=config.BUDGET)
    args = ap.parse_args()

    (RESULTS / "histories").mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        problem = make_problem(ds)
        print(f"== {ds}: {problem.X_train.shape[0]} train, {problem.X_test.shape[0]} test, "
              f"{problem.n_features} variables")
        rows = []
        for method in args.methods:
            seeds = args.seeds[:1] if method in config.DETERMINISTIC else args.seeds
            for seed in seeds:
                try:
                    row, extra = run_one(problem, method, seed, args.budget)
                except NotImplementedError as e:
                    print(f"   [omitido] {method}: {e}")
                    break
                rows.append(row)
                (RESULTS / "histories" / f"{ds}_{method}_{seed}.json").write_text(json.dumps(extra))
                print(f"   {method:7s} seed={seed:2d}  cv_err={row['cv_error']:.4f}  "
                      f"test_acc={row['test_acc']:.4f}  k={row['n_selected']:4d}  "
                      f"evals={row['n_evals']:5d}  {row['time_s']:.1f}s")
        if rows:
            out = RESULTS / f"{ds}.csv"
            new = pd.DataFrame(rows)
            if out.exists():  # sustituye las filas (metodo, semilla) que se han vuelto a lanzar
                old = pd.read_csv(out)
                key = set(zip(new.method, new.seed))
                old = old[[(m, s) not in key for m, s in zip(old.method, old.seed)]]
                new = pd.concat([old, new], ignore_index=True)
            new.to_csv(out, index=False)
            print(f"   -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
