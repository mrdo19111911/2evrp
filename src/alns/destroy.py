"""Destroy operators — @njit dispatch by index."""
from numba import njit
import numpy as np

from src.alns.destroy_basic import (
    random_removal, worst_cost_removal, shaw_removal, zone_removal,
)
from src.alns.destroy_route import (
    route_removal, satellite_removal, route_split_removal, cluster_removal,
)
from src.alns.destroy_advanced import (
    time_pressure_removal, cascade_worst_removal,
    inject_unserved, sisr_removal, history_removal,
)


@njit(cache=True)
def destroy_dispatch(idx, sol, customers, dist_matrix, seed, config, history_freq):
    """Dispatch destroy operator by index. Returns i32[:] removed customers."""
    if idx == 0:
        return random_removal(sol, customers, dist_matrix, seed, config)
    elif idx == 1:
        return worst_cost_removal(sol, customers, dist_matrix, seed, config)
    elif idx == 2:
        return shaw_removal(sol, customers, dist_matrix, seed, config)
    elif idx == 3:
        return route_removal(sol, customers, dist_matrix, seed, config)
    elif idx == 4:
        return satellite_removal(sol, customers, dist_matrix, seed, config)
    elif idx == 5:
        return zone_removal(sol, customers, dist_matrix, seed, config)
    elif idx == 6:
        return time_pressure_removal(sol, customers, dist_matrix, seed, config)
    elif idx == 7:
        return route_split_removal(sol, customers, dist_matrix, seed, config)
    elif idx == 8:
        return cascade_worst_removal(sol, customers, dist_matrix, seed, config)
    elif idx == 9:
        return inject_unserved(sol, customers, dist_matrix, seed, config)
    elif idx == 10:
        return sisr_removal(sol, customers, dist_matrix, seed, config)
    elif idx == 11:
        return cluster_removal(sol, customers, dist_matrix, seed, config)
    elif idx == 12:
        return history_removal(sol, customers, dist_matrix, seed, config, history_freq)
    return np.empty(0, dtype=np.int32)
