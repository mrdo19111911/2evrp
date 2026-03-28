"""Cross-layer operators — dispatch by index. All ops are @njit."""
import numpy as np
from numba import njit

from src.alns.crosslayer_ops import (
    swap_assignment,
    relocate_satellite,
    create_new_satellite,
    remove_satellite_node,
)
from src.alns.crosslayer_rebalance import echelon_rebalance


@njit(cache=True)
def cross_dispatch(idx, sol, customers, dist_matrix, vehicles, seed, config):
    """Dispatch cross-layer operator by index. Returns bool."""
    if idx == 0:
        return swap_assignment(sol, customers, dist_matrix, vehicles, seed, config)
    elif idx == 1:
        return relocate_satellite(sol, customers, dist_matrix, vehicles, seed, config)
    elif idx == 2:
        return create_new_satellite(sol, customers, dist_matrix, vehicles, seed, config)
    elif idx == 3:
        return remove_satellite_node(sol, customers, dist_matrix, vehicles, seed, config)
    elif idx == 4:
        return echelon_rebalance(sol, customers, dist_matrix, vehicles, seed, config)
    return False
