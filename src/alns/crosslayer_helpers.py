"""Helper functions for cross-layer operators."""
import numpy as np

from src.data.constants import ACT_DELIVER, ACT_RELOAD, SAT_CUST, SAT_BIKE, VEH_TRUCK, VEH_BIKE
from src.solution.route_ops import remove_stop


def replace_reload_node(sol, old_node, new_node):
    """Replace RELOAD stops at old_node with new_node in all bike routes."""
    for b in range(sol["n_bikes"]):
        L = int(sol["bike_lengths"][b])
        for i in range(L):
            if (sol["bike_stops"][b, i] == old_node
                    and sol["bike_actions"][b, i] == ACT_RELOAD):
                sol["bike_stops"][b, i] = new_node


def estimate_transfer_kg(sol, bike_id, reload_pos, customers):
    """Estimate kg transferred: sum demands of DELIVER stops after reload_pos."""
    from src.data.constants import ACT_DELIVER, COL_DEMAND
    L = int(sol["bike_lengths"][bike_id])
    total = 0.0
    for i in range(reload_pos + 1, L):
        if sol["bike_actions"][bike_id, i] == ACT_DELIVER:
            c = int(sol["bike_stops"][bike_id, i])
            total += float(customers[c, COL_DEMAND])
    return total if total > 0 else 60.0


def remove_all_reload_at_node(sol, node, dist_matrix, customers):
    """Remove all RELOAD stops at node from all bike and truck routes."""
    for b in range(sol["n_bikes"]):
        _remove_reload_stops(sol, VEH_BIKE, b, node, dist_matrix, customers)
    for t in range(sol["n_trucks"]):
        _remove_reload_stops(sol, VEH_TRUCK, t, node, dist_matrix, customers)


def _remove_reload_stops(sol, vtype, vid, node, dist_matrix, customers):
    """Remove RELOAD stops for node in a single route (backward scan)."""
    lengths = sol["truck_lengths"] if vtype == VEH_TRUCK else sol["bike_lengths"]
    stops = sol["truck_stops"] if vtype == VEH_TRUCK else sol["bike_stops"]
    actions = sol["truck_actions"] if vtype == VEH_TRUCK else sol["bike_actions"]
    i = int(lengths[vid]) - 1
    while i >= 0:
        if (stops[vid, i] == node and actions[vid, i] == ACT_RELOAD):
            remove_stop(sol, vtype, vid, i, dist_matrix, customers)
        i -= 1


def score_satellite_candidate(sol, cand, old_sat, satellites, dist_matrix):
    """Score a candidate node for satellite relocation (lower is better).
    Sum distance from candidate to all bikes using the old satellite."""
    mask = satellites[:, SAT_CUST] == old_sat
    bike_ids = np.unique(satellites[mask, SAT_BIKE].astype(int))
    score = 0.0
    for bid in bike_ids:
        if bid >= sol["n_bikes"]:
            continue
        L = int(sol["bike_lengths"][bid])
        for i in range(L):
            if sol["bike_actions"][bid, i] == ACT_DELIVER:
                c = int(sol["bike_stops"][bid, i])
                score += dist_matrix[cand + 1, c + 1]
    return score
