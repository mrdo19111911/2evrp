"""Helper functions for cross-layer operators."""
import numpy as np

from src.data.constants import (
    ACT_DELIVER, ACT_PICKUP, DM_DIST,
    ORD_LOC, ORD_QTY, ORD_UNIT_W,
    VEH_TYPE, VTYPE_BIKE, TR_HUB, TR_BIKE,
)
from src.solution.query import get_route_customers_only
from src.solution.route_ops import remove_stop


def replace_pickup_node(sol, old_loc, new_loc):
    """Replace PICKUP stops at old_loc with new_loc in all routes."""
    for v in range(sol["n_vehicles"]):
        L = sol["lengths"][v]
        for i in range(L):
            if (sol["stops"][v, i] == old_loc
                    and sol["actions"][v, i] == ACT_PICKUP):
                sol["stops"][v, i] = new_loc


def estimate_transfer_kg(sol, vehicle_id, pickup_pos, orders):
    """Estimate kg needed: sum demands of DELIVER stops after pickup_pos."""
    from src.solution.query import _find_customer_for_loc
    L = sol["lengths"][vehicle_id]
    total = 0.0
    for i in range(pickup_pos + 1, L):
        if sol["actions"][vehicle_id, i] == ACT_DELIVER:
            c = _find_customer_for_loc(sol, vehicle_id, i)
            if c >= 0:
                total += orders[c, ORD_QTY] * orders[c, ORD_UNIT_W]
    return total if total > 0 else sol["loads_kg"][vehicle_id]


def remove_all_pickup_at_loc(sol, loc, dist_matrix, vehicles, orders):
    """Remove all PICKUP stops at loc from all routes."""
    for v in range(sol["n_vehicles"]):
        _remove_pickup_stops(sol, v, loc, dist_matrix, vehicles, orders)


def _remove_pickup_stops(sol, vid, loc, dist_matrix, vehicles, orders):
    """Remove all PICKUP stops for loc in a single route (backward scan)."""
    i = sol["lengths"][vid] - 1
    while i >= 0:
        if (sol["stops"][vid, i] == loc
                and sol["actions"][vid, i] == ACT_PICKUP):
            remove_stop(sol, vid, i, dist_matrix, vehicles, orders)
        i -= 1


def score_transfer_candidate(sol, cand_loc, old_hub, transfers, dist_matrix,
                             orders=None):
    """Score a candidate hub for transfer relocation (lower is better)."""
    dm = dist_matrix[DM_DIST]
    mask = transfers[:, TR_HUB] == old_hub
    bike_ids = np.unique(transfers[mask, TR_BIKE].astype(int))
    score = 0.0
    for bid in bike_ids:
        if bid >= sol["n_vehicles"]:
            continue
        bike_custs = get_route_customers_only(sol, int(bid), orders)
        if len(bike_custs) > 0:
            cust_locs = np.array([int(sol["stops"][bid, p])
                                  for p in range(sol["lengths"][bid])
                                  if sol["actions"][bid, p] == ACT_DELIVER])
            if len(cust_locs) > 0:
                score += dm[cand_loc, cust_locs].sum()
    return score
