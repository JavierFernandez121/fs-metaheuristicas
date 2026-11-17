"""Configuracion de los experimentos. Cambiar aqui, no dentro del codigo."""

DATASETS = ["breast_cancer", "madelon"]        # anadir "arcene" si da tiempo
METHODS = ["random", "ga", "bpso", "sa", "tabu", "nsga2", "all", "sfs", "mi", "lasso"]
SEEDS = list(range(10))                         # subir a 30 si los tiempos lo permiten
BUDGET = 3000                                   # evaluaciones de la funcion objetivo por ejecucion
CLASSIFIER = "knn"                              # "knn" o "logreg"
CV_FOLDS = 5
ALPHA = 0.99                                    # peso del error frente al n.o de variables

# Parametros de cada metodo (se pasan como **kwargs a su run)
PARAMS = {
    "ga": {"pop_size": 30},
    "bpso": {"n_particles": 30},
    "sa": {},
    "tabu": {"n_neighbors": 20},
    "nsga2": {"pop_size": 40},
}

# Metodos deterministas: basta con una semilla
DETERMINISTIC = {"all", "sfs", "lasso"}
