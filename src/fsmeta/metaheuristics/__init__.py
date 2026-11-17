"""Registro de metodos de busqueda. Todos cumplen la interfaz de base.py."""
from . import bpso, ga, nsga2, random_search, sa, tabu

METHODS = {
    "random": random_search.run,
    "ga": ga.run,
    "bpso": bpso.run,
    "sa": sa.run,
    "tabu": tabu.run,
    "nsga2": nsga2.run,
}
