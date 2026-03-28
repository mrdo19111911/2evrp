"""Helper functions for cross-layer operators. All @njit. i64 types."""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, COL_DEMAND,
    SAT_CUST, SAT_BIKE, VEH_TRUCK, VEH_BIKE,
    SOL_BIKE_STOPS, SOL_BIKE_ACTIONS, SOL_BIKE_LENGTHS,
    SOL_META, META_N_TRUCKS, META_N_BIKES, META_N_SATELLITES,
    SOL_SATELLITES,
)
from src.solution.route_ops import remove_stop
from src.solution._helpers import get_route_arrays


@njit(cache=True)
def replace_reload_node(sol, old_node, new_node):
    """Replace RELOAD stops at old_node with new_node in all bike routes."""
    bike_stops = sol[SOL_BIKE_STOPS]
    bike_actions = sol[SOL_BIKE_ACTIONS]
    bike_lengths = sol[SOL_BIKE_LENGTHS]
    n_bikes = sol[SOL_META][META_N_BIKES]
    for b in range(n_bikes):
        L = int(bike_lengths[b])
        for i in range(L):
            if bike_stops[b, i] == old_node and bike_actions[b, i] == ACT_RELOAD:
                bike_stops[b, i] = new_node


@njit(cache=True)
def estimate_transfer_g(sol, bike_id, reload_pos, customers):
    """Estimate grams transferred after reload_pos. Returns i64."""
    bike_stops = sol[SOL_BIKE_STOPS]
    bike_actions = sol[SOL_BIKE_ACTIONS]
    bike_lengths = sol[SOL_BIKE_LENGTHS]
    L = int(bike_lengths[bike_id])
    total = np.int64(0)
    for i in range(reload_pos + 1, L):
        if bike_actions[bike_id, i] == ACT_DELIVER:
            c = int(bike_stops[bike_id, i])
            total += customers[c, COL_DEMAND]
    if total > 0:
        return total
    return np.int64(60000)  # 60kg default


@njit(cache=True)
def _remove_reload_stops(sol, vtype, vid, node, dist_matrix, customers):
    """Remove RELOAD stops for node in a single route (backward scan)."""
    stops, actions, lengths = get_route_arrays(sol, vtype)
    i = int(lengths[vid]) - 1
    while i >= 0:
        if stops[vid, i] == node and actions[vid, i] == ACT_RELOAD:
            remove_stop(sol, vtype, vid, i, dist_matrix, customers)
        i -= 1


@njit(cache=True)
def remove_all_reload_at_node(sol, node, dist_matrix, customers):
    """Remove all RELOAD stops at node from all routes."""
    meta = sol[SOL_META]
    for b in range(meta[META_N_BIKES]):
        _remove_reload_stops(sol, VEH_BIKE, b, node, dist_matrix, customers)
    for t in range(meta[META_N_TRUCKS]):
        _remove_reload_stops(sol, VEH_TRUCK, t, node, dist_matrix, customers)


@njit(cache=True)
def score_satellite_candidate(sol, cand, old_sat, dist_matrix):
    """Score candidate for satellite relocation (lower is better). i64 meters."""
    sats = sol[SOL_SATELLITES]
    n_sats = sol[SOL_META][META_N_SATELLITES]
    n_bikes = sol[SOL_META][META_N_BIKES]
    bike_stops = sol[SOL_BIKE_STOPS]
    bike_actions = sol[SOL_BIKE_ACTIONS]
    bike_lengths = sol[SOL_BIKE_LENGTHS]
    # Collect unique bike_ids for old_sat using boolean seen array
    seen_bikes = np.zeros(n_bikes, dtype=np.bool_)
    for i in range(n_sats):
        if sats[i, SAT_CUST] == old_sat:
            bid = int(sats[i, SAT_BIKE])
            if bid < n_bikes:
                seen_bikes[bid] = True
    score = np.int64(0)
    for bid in range(n_bikes):
        if not seen_bikes[bid]:
            continue
        L = int(bike_lengths[bid])
        for i in range(L):
            if bike_actions[bid, i] == ACT_DELIVER:
                c = int(bike_stops[bid, i])
                score += dist_matrix[cand + 1, c + 1]
    return score
