"""Ajuste de parametros de los 5 metodos de Persona A (variando un parametro cada vez).

IMPORTANTE: usa SOLO el conjunto de entrenamiento (70 %); el test no se toca aqui.

Uso (desde la raiz del repo):
    python herramientas_a/ajuste_parametros.py --methods bpso --seeds 3 --budget 1000   # prueba rapida
    python herramientas_a/ajuste_parametros.py                                            # completo
Opciones: --methods ga bpso sa tabu nsga2 | --seeds 5 | --budget 3000 | --jobs -1 | --dataset breast_cancer|sintetico
Salida: tabla por metodo en pantalla + CSV con todas las ejecuciones en herramientas_a/.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_train(name):
    from sklearn.datasets import load_breast_cancer, make_classification
    from sklearn.model_selection import train_test_split
    if name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
    else:  # sintetico tipo Madelon: pocas variables utiles entre muchas de ruido
        X, y = make_classification(n_samples=600, n_features=200, n_informative=5, n_redundant=15,
                                   n_clusters_per_class=2, class_sep=1.0, flip_y=0.01,
                                   shuffle=True, random_state=0)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    return Xtr, ytr


def grids(n):
    """Para cada metodo: lista de (etiqueta, parametros). 'base' = valores actuales."""
    return {
        "ga": [("base", {"pop_size": 30}),
               ("pop_size=20", {"pop_size": 20}), ("pop_size=50", {"pop_size": 50}),
               ("crossover=0.7", {"pop_size": 30, "crossover_rate": 0.7}),
               ("mut=0.5/n", {"pop_size": 30, "mutation_rate": 0.5 / n}),
               ("mut=2/n", {"pop_size": 30, "mutation_rate": 2.0 / n}),
               ("torneo=2", {"pop_size": 30, "tournament_k": 2}),
               ("torneo=5", {"pop_size": 30, "tournament_k": 5}),
               ("p_init=None", {"pop_size": 30, "p_init": None})],
        "bpso": [("base", {"n_particles": 30}),
                 ("w=0.4", {"n_particles": 30, "w": 0.4}), ("w=0.9", {"n_particles": 30, "w": 0.9}),
                 ("c1=c2=1.0", {"n_particles": 30, "c1": 1.0, "c2": 1.0}),
                 ("c1=c2=2.0", {"n_particles": 30, "c1": 2.0, "c2": 2.0}),
                 ("v_max=2", {"n_particles": 30, "v_max": 2.0}), ("v_max=6", {"n_particles": 30, "v_max": 6.0}),
                 ("n_particles=20", {"n_particles": 20}), ("n_particles=50", {"n_particles": 50})],
        "sa": [("base", {}),
               ("cooling=0.90", {"cooling": 0.90}), ("cooling=0.99", {"cooling": 0.99}),
               ("iters_per_temp=10", {"iters_per_temp": 10}), ("iters_per_temp=50", {"iters_per_temp": 50}),
               ("k_max=1", {"k_max": 1}), ("k_max=5", {"k_max": 5}),
               ("T0=0.02", {"T0": 0.02})],
        "tabu": [("base", {"n_neighbors": 20}),
                 ("n_neighbors=10", {"n_neighbors": 10}), ("n_neighbors=40", {"n_neighbors": 40}),
                 ("tenure=5", {"n_neighbors": 20, "tenure": 5}), ("tenure=15", {"n_neighbors": 20, "tenure": 15}),
                 ("tenure=30", {"n_neighbors": 20, "tenure": 30}),
                 ("patience=15", {"n_neighbors": 20, "patience": 15}),
                 ("patience=60", {"n_neighbors": 20, "patience": 60}),
                 ("perturb=0.10", {"n_neighbors": 20, "perturb_frac": 0.10})],
        "nsga2": [("base", {"pop_size": 40}),
                  ("pop_size=20", {"pop_size": 20}), ("pop_size=80", {"pop_size": 80}),
                  ("crossover=0.7", {"pop_size": 40, "crossover_rate": 0.7}),
                  ("mut=0.5/n", {"pop_size": 40, "mutation_rate": 0.5 / n}),
                  ("mut=2/n", {"pop_size": 40, "mutation_rate": 2.0 / n})],
    }


def run_one(method, label, params, seed, Xtr, ytr, budget):
    sys.path.insert(0, str(ROOT / "src"))   # los procesos hijos tambien necesitan la ruta
    from fsmeta.fitness import FitnessFunction
    from fsmeta.metaheuristics import METHODS
    fit = FitnessFunction(Xtr, ytr, budget=budget)
    t0 = time.time()
    r = METHODS[method](fit, np.random.default_rng(seed), **params)
    return {"method": method, "config": label, "seed": seed, "best_f": r.best_f,
            "err_cv": fit.cv_error(r.best_mask), "n_vars": int(r.best_mask.sum()),
            "n_evals": r.n_evals, "secs": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["ga", "bpso", "sa", "tabu", "nsga2"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--jobs", type=int, default=-1, help="procesos en paralelo (-1 = todos los nucleos)")
    ap.add_argument("--dataset", default="breast_cancer", choices=["breast_cancer", "sintetico"])
    a = ap.parse_args()

    import pandas as pd
    from joblib import Parallel, delayed
    Xtr, ytr = load_train(a.dataset)
    G = grids(Xtr.shape[1])
    jobs = [(m, lab, p, s) for m in a.methods for lab, p in G[m] for s in range(a.seeds)]
    print(f"{a.dataset}: {Xtr.shape[0]} muestras x {Xtr.shape[1]} variables | {len(jobs)} ejecuciones "
          f"(presupuesto {a.budget}, {a.seeds} semillas)", flush=True)
    t0 = time.time()
    rows = Parallel(n_jobs=a.jobs, verbose=5)(delayed(run_one)(m, lab, p, s, Xtr, ytr, a.budget)
                                              for m, lab, p, s in jobs)
    df = pd.DataFrame(rows)
    out = ROOT / "herramientas_a" / f"resultados_ajuste_{a.dataset}_{'-'.join(a.methods)}.csv"
    df.to_csv(out, index=False)
    print(f"\nTiempo total: {time.time() - t0:.0f}s | CSV: {out}\n")

    for m in a.methods:
        g = df[df.method == m].groupby("config", sort=False).agg(
            f_media=("best_f", "mean"), f_std=("best_f", "std"), err_cv=("err_cv", "mean"),
            n_vars=("n_vars", "mean"), segundos=("secs", "mean"))
        g["f_ee"] = g.f_std / np.sqrt(a.seeds)       # error estandar de la media
        base = g.loc["base", "f_media"]
        g["mejora_vs_base"] = base - g.f_media         # positivo = mejor que base
        print(f"=== {m} (menor f = mejor; ordenado) ===")
        print(g.sort_values("f_media")[["f_media", "f_ee", "err_cv", "n_vars", "mejora_vs_base", "segundos"]]
              .round(4).to_string(), "\n")
    print("Criterio: cambia un parametro solo si 'mejora_vs_base' supera ~2 veces 'f_ee'; si no, es ruido.")


if __name__ == "__main__":
    main()
