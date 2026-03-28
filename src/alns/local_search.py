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


def exchange_inter_route(sol, vtype, dist_matrix, customers, restricted):
    """Swap 1 customer between 2 routes of same vtype. Returns True if improved."""
    from src.solution.solution_ops import swap_stops_between
    from src.solution.check import can_swap_customers

    n_vehicles = sol["n_trucks"] if vtype == VEH_TRUCK else sol["n_bikes"]
    if n_vehicles < 2:
        return False

    lengths = _get_lengths(sol, vtype)
    stops = _get_stops(sol, vtype)
    actions = _get_actions(sol, vtype)
    improved = False

    for va in range(n_vehicles):
        La = int(lengths[va])
        for pa in range(La):
            if int(actions[va, pa]) != ACT_DELIVER:
                continue
            ca = int(stops[va, pa])

            for vb in range(va + 1, n_vehicles):
                Lb = int(lengths[vb])
                for pb in range(Lb):
                    if int(actions[vb, pb]) != ACT_DELIVER:
                        continue
                    cb = int(stops[vb, pb])

                    if not can_swap_customers(sol, ca, cb, customers, restricted):
                        continue

                    # Delta: removal savings + insertion costs at each other's position
                    rem_a = removal_cost_delta(sol, vtype, va, pa, dist_matrix)
                    rem_b = removal_cost_delta(sol, vtype, vb, pb, dist_matrix)
                    ins_b_at_a = _swap_insertion_delta(sol, vtype, va, pa, cb, dist_matrix)
                    ins_a_at_b = _swap_insertion_delta(sol, vtype, vb, pb, ca, dist_matrix)
                    delta = rem_a + rem_b + ins_b_at_a + ins_a_at_b

                    if delta < -1e-6:
                        swap_stops_between(sol, vtype, va, pa, vtype, vb, pb,
                                           dist_matrix, customers)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            break
    return improved


def _swap_insertion_delta(sol, vtype, vid, pos, new_cust, dist_matrix):
    """Cost of having new_cust at position pos instead of current customer.
    = d(prev, new) + d(new, next) - d(prev, next)"""
    if vtype == VEH_TRUCK:
        stops, lengths = sol["truck_stops"], sol["truck_lengths"]
    else:
        stops, lengths = sol["bike_stops"], sol["bike_lengths"]
    L = int(lengths[vid])
    cn = new_cust + 1  # dist_matrix index

    prev_dm = 0 if pos == 0 else int(stops[vid, pos - 1]) + 1
    next_dm = 0 if pos >= L - 1 else int(stops[vid, pos + 1]) + 1

    return (dist_matrix[prev_dm, cn] + dist_matrix[cn, next_dm]
            - dist_matrix[prev_dm, next_dm])


def run_local_search(sol, dist_matrix, customers, restricted=None):
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
    if restricted is not None:
        if exchange_inter_route(sol, VEH_TRUCK, dist_matrix, customers, restricted):
            improved = True
        if exchange_inter_route(sol, VEH_BIKE, dist_matrix, customers, restricted):
            improved = True
    return improved
