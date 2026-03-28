"""Repair operators — @njit dispatch by index."""
from numba import njit
import numpy as np

from src.alns.repair_ops import (
    greedy_insertion,
    regret_k_insertion,
    satellite_aware_insertion,
    selective_drop_insertion,
    blinks_insertion,
)


@njit(cache=True)
def repair_dispatch(idx, sol, removed, customers, dist_matrix,
                    vehicles, seed, config):
    """Dispatch repair operator by index."""
    if idx == 0:
        greedy_insertion(sol, removed, customers, dist_matrix, vehicles, seed, config)
    elif idx == 1:
        regret_k_insertion(sol, removed, customers, dist_matrix, vehicles, seed, config)
    elif idx == 2:
        satellite_aware_insertion(sol, removed, customers, dist_matrix, vehicles, seed, config)
    elif idx == 3:
        selective_drop_insertion(sol, removed, customers, dist_matrix, vehicles, seed, config)
    elif idx == 4:
        blinks_insertion(sol, removed, customers, dist_matrix, vehicles, seed, config)
