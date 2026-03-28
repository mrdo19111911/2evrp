"""Local search improvement operators."""
from src.data.constants import ACT_DELIVER, VEH_TRUCK, VEH_BIKE, COL_DEMAND
from src.data.cost import TRUCK_CAPACITY, BIKE_CAPACITY
from src.solution.route_ops import reverse_segment
from src.solution.solution_ops import move_stop
from src.solution.delta import removal_cost_delta, best_insertion_pos
from src.alns.ls_helpers import (
    _get_lengths, _get_stops, _get_actions, _get_loads,
    _two_opt_delta, _or_opt_delta, _do_or_opt_move,
)


def two_opt(sol, vtype, vid, dist_matrix):
    """Reverse segment in route. Returns True if improved."""
    L = _get_lengths(sol, vtype)[vid]
    if L < 3:
        return False

    improved = False
    for i in range(L - 1):
        for j in range(i + 2, L):
            delta = _two_opt_delta(sol, vtype, vid, i, j, dist_matrix)
            if delta < -1e-6:
                reverse_segment(sol, vtype, vid, i, j, dist_matrix)
                improved = True
    return improved


def or_opt(sol, vtype, vid, dist_matrix):
    """Move segment (1-3 stops) within route. Returns True if improved."""
    L = _get_lengths(sol, vtype)[vid]
    improved = False

    for seg_len in [1, 2, 3]:
        for i in range(L - seg_len + 1):
            for j in range(L + 1):
                if j >= i and j <= i + seg_len:
                    continue
                delta = _or_opt_delta(sol, vtype, vid, i, seg_len, j, dist_matrix)
                if delta < -1e-6:
                    _do_or_opt_move(sol, vtype, vid, i, seg_len, j, dist_matrix)
                    improved = True
                    break
    return improved


def relocate_inter_route(sol, vtype, dist_matrix, customers):
    """Move customer between routes of same vtype. Returns True if improved."""
    n_vehicles = sol["n_trucks"] if vtype == VEH_TRUCK else sol["n_bikes"]
    if n_vehicles < 2:
        return False

    capacity = TRUCK_CAPACITY if vtype == VEH_TRUCK else BIKE_CAPACITY
    lengths = _get_lengths(sol, vtype)
    stops = _get_stops(sol, vtype)
    actions = _get_actions(sol, vtype)
    loads = _get_loads(sol, vtype)
    improved = False

    for va in range(n_vehicles):
        La = lengths[va]
        for pos_a in range(La):
            if actions[va, pos_a] != ACT_DELIVER:
                continue
            c = int(stops[va, pos_a])
            demand = customers[c, COL_DEMAND]
            rem_delta = removal_cost_delta(sol, vtype, va, pos_a, dist_matrix)

            for vb in range(n_vehicles):
                if vb == va:
                    continue
                if loads[vb] + demand > capacity:
                    continue
                ins_pos, ins_delta = best_insertion_pos(sol, vtype, vb, c,
                                                        dist_matrix)
                if rem_delta + ins_delta < -1e-6:
                    move_stop(sol, vtype, va, pos_a, vtype, vb, ins_pos,
                              dist_matrix, customers)
                    improved = True
                    break
    return improved


def run_local_search(sol, dist_matrix, customers):
    """Run all LS operators on all routes."""
    improved = False
    for t in range(sol["n_trucks"]):
        if two_opt(sol, VEH_TRUCK, t, dist_matrix):
            improved = True
        if or_opt(sol, VEH_TRUCK, t, dist_matrix):
            improved = True
    for b in range(sol["n_bikes"]):
        if two_opt(sol, VEH_BIKE, b, dist_matrix):
            improved = True
        if or_opt(sol, VEH_BIKE, b, dist_matrix):
            improved = True
    if relocate_inter_route(sol, VEH_TRUCK, dist_matrix, customers):
        improved = True
    if relocate_inter_route(sol, VEH_BIKE, dist_matrix, customers):
        improved = True
    return improved
