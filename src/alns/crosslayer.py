"""Cross-layer operators -- change assignment between truck/bike."""
import numpy as np

from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, VEH_TRUCK, VEH_BIKE,
    COL_X, COL_Y, COL_DEMAND, SAT_CUST,
)
from src.data.cost import TRUCK_CAPACITY, BIKE_CAPACITY
from src.solution.query import (
    get_assigned_customers, get_customer_info, get_route_customers_only,
)
from src.solution.route_ops import remove_stop, insert_stop
from src.solution.delta import find_best_insertion_all_routes, best_insertion_pos
from src.solution.satellite_ops import (
    remove_satellites_for_customer, remove_satellites_for_bike,
    add_satellite_event,
)
from src.solution.solution_ops import clear_route
from src.alns.crosslayer_helpers import (
    replace_reload_node, estimate_transfer_kg,
    remove_all_reload_at_node, score_satellite_candidate,
)


def swap_assignment(sol, customers, restricted, dist_matrix, satellites, rng, params):
    """Move customer truck<->bike. Returns True if successful."""
    N = len(customers)
    assigned = get_assigned_customers(sol, N)
    flexible = []
    for c_raw in assigned:
        c = int(c_raw)
        if customers[c, COL_DEMAND] > BIKE_CAPACITY:
            continue
        if restricted[c] == 1:
            continue
        vt = int(sol["cust_vtype"][c])
        if vt == VEH_TRUCK:
            flexible.append(c)

    if not flexible:
        return False

    c = int(rng.choice(flexible))
    vtype, vid, pos = get_customer_info(sol, c)

    if vtype == VEH_TRUCK:
        return _truck_to_bike(sol, c, vid, pos, dist_matrix, customers)
    else:
        return _bike_to_truck(sol, c, vid, pos, dist_matrix, customers)


def _truck_to_bike(sol, c, vid, pos, dist_matrix, customers):
    """Move customer from truck to bike. Returns True if successful."""
    remove_stop(sol, VEH_TRUCK, vid, pos, dist_matrix, customers)
    best_vid, best_pos, delta = find_best_insertion_all_routes(
        sol, VEH_BIKE, c, dist_matrix, customers, BIKE_CAPACITY)
    if best_vid >= 0:
        insert_stop(sol, VEH_BIKE, best_vid, best_pos, c, ACT_DELIVER,
                     dist_matrix, customers)
        return True
    insert_stop(sol, VEH_TRUCK, vid, min(pos, sol["truck_lengths"][vid]),
                 c, ACT_DELIVER, dist_matrix, customers)
    return False


def _bike_to_truck(sol, c, vid, pos, dist_matrix, customers):
    """Move customer from bike to truck. Returns True if successful."""
    remove_stop(sol, VEH_BIKE, vid, pos, dist_matrix, customers)
    remove_satellites_for_customer(sol, c)
    best_vid, best_pos, delta = find_best_insertion_all_routes(
        sol, VEH_TRUCK, c, dist_matrix, customers, TRUCK_CAPACITY)
    if best_vid >= 0:
        insert_stop(sol, VEH_TRUCK, best_vid, best_pos, c, ACT_DELIVER,
                     dist_matrix, customers)
        return True
    insert_stop(sol, VEH_BIKE, vid, min(pos, sol["bike_lengths"][vid]),
                 c, ACT_DELIVER, dist_matrix, customers)
    return False


def relocate_satellite(sol, customers, restricted, dist_matrix, satellites, rng, params):
    """Move satellite to better customer node. Returns True if successful."""
    sats = sol["satellites"]
    if len(sats) == 0:
        return False

    N = len(customers)
    unique_sats = np.unique(sats[:, SAT_CUST].astype(int))
    old_sat = int(rng.choice(unique_sats))

    assigned = get_assigned_customers(sol, N)
    dists = dist_matrix[old_sat + 1, assigned + 1]
    nearby = assigned[np.argsort(dists)[:10]]

    best_new, best_score = None, np.inf
    for cand in nearby:
        cand = int(cand)
        if cand == old_sat:
            continue
        score = score_satellite_candidate(sol, cand, old_sat, sats, dist_matrix)
        if score < best_score:
            best_score = score
            best_new = cand

    if best_new is None:
        return False

    mask = sats[:, SAT_CUST] == old_sat
    sol["satellites"][mask, SAT_CUST] = best_new
    replace_reload_node(sol, old_sat, best_new)
    return True


def create_new_satellite(sol, customers, restricted, dist_matrix, satellites, rng, params):
    """Create satellite at center of long bike route. Returns True if successful."""
    bike_dists = sol["bike_distances"]
    if bike_dists.max() == 0:
        return False

    worst_bike = int(np.argmax(bike_dists))
    bike_custs = get_route_customers_only(sol, VEH_BIKE, worst_bike)
    if len(bike_custs) == 0:
        return False

    N = len(customers)
    cx = customers[bike_custs, COL_X].mean()
    cy = customers[bike_custs, COL_Y].mean()
    assigned = get_assigned_customers(sol, N)
    dists_to_center = np.sqrt(
        (customers[assigned, COL_X] - cx) ** 2 +
        (customers[assigned, COL_Y] - cy) ** 2
    )
    new_sat = int(assigned[np.argmin(dists_to_center)])

    best_truck, best_delta = None, np.inf
    for t in range(sol["n_trucks"]):
        pos, delta = best_insertion_pos(sol, VEH_TRUCK, t, new_sat, dist_matrix)
        if delta < best_delta:
            best_delta = delta
            best_truck = (t, pos)

    if best_truck is None:
        return False

    t, pos = best_truck
    insert_stop(sol, VEH_TRUCK, t, pos, new_sat, ACT_RELOAD,
                 dist_matrix, customers)

    bike_pos, _ = best_insertion_pos(sol, VEH_BIKE, worst_bike, new_sat,
                                      dist_matrix)
    insert_stop(sol, VEH_BIKE, worst_bike, bike_pos, new_sat, ACT_RELOAD,
                 dist_matrix, customers)

    transfer_kg = estimate_transfer_kg(sol, worst_bike, bike_pos, customers)
    planned_time = 0.0
    add_satellite_event(sol, new_sat, worst_bike, t, transfer_kg, planned_time)
    return True


def remove_satellite_node(sol, customers, restricted, dist_matrix, satellites, rng, params):
    """Remove satellite completely. Returns True if successful."""
    sats = sol["satellites"]
    if len(sats) == 0:
        return False

    unique_sats = np.unique(sats[:, SAT_CUST].astype(int))
    target_sat = int(rng.choice(unique_sats))

    remove_satellites_for_customer(sol, target_sat)
    remove_all_reload_at_node(sol, target_sat, dist_matrix, customers)
    return True


CROSS_OPS = [
    swap_assignment,
    relocate_satellite,
    create_new_satellite,
    remove_satellite_node,
]
