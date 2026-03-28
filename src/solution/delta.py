"""Delta cost evaluation. O(1) per operation -- speed-critical for ALNS."""
import numpy as np

from src.data.constants import (
    VEH_TRUCK, VEH_BIKE, COL_DEMAND, COL_TW_CLOSE,
    ACT_DELIVER, VCOL_SPEED,
)
from src.data.cost import TRUCK_SPEED_URBAN, BIKE_SPEED_URBAN, SERVICE_BASE, SERVICE_PER_100KG, DAY_LENGTH
from src.solution._helpers import (
    get_route_arrays as _get_route_arrays,
    get_loads as _get_loads,
)


def insertion_cost_delta(sol, vtype, vid, pos, customer, dist_matrix):
    """O(1) delta distance if inserting customer at pos."""
    stops, _, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    cn = customer + 1  # dist_matrix index

    if L == 0:
        return dist_matrix[0, cn] + dist_matrix[cn, 0]

    if pos == 0:
        old_first = stops[vid, 0] + 1
        return dist_matrix[0, cn] + dist_matrix[cn, old_first] - dist_matrix[0, old_first]

    if pos == L:
        old_last = stops[vid, L - 1] + 1
        return dist_matrix[old_last, cn] + dist_matrix[cn, 0] - dist_matrix[old_last, 0]

    a = stops[vid, pos - 1] + 1
    b = stops[vid, pos] + 1
    return dist_matrix[a, cn] + dist_matrix[cn, b] - dist_matrix[a, b]


def removal_cost_delta(sol, vtype, vid, pos, dist_matrix):
    """O(1) delta distance if removing stop at pos."""
    stops, _, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    cn = stops[vid, pos] + 1

    if L == 1:
        return -(dist_matrix[0, cn] + dist_matrix[cn, 0])

    if pos == 0:
        next_node = stops[vid, 1] + 1
        return dist_matrix[0, next_node] - dist_matrix[0, cn] - dist_matrix[cn, next_node]

    if pos == L - 1:
        prev_node = stops[vid, L - 2] + 1
        return dist_matrix[prev_node, 0] - dist_matrix[prev_node, cn] - dist_matrix[cn, 0]

    prev_node = stops[vid, pos - 1] + 1
    next_node = stops[vid, pos + 1] + 1
    return (dist_matrix[prev_node, next_node]
            - dist_matrix[prev_node, cn] - dist_matrix[cn, next_node])


def cascade_removal_value(sol, vtype, vid, pos, dist_matrix, customers):
    """Value of removing customer at pos: distance savings + TW cascade savings.
    Higher value = more valuable to remove. O(L)."""
    stops, actions, lengths = _get_route_arrays(sol, vtype)
    L = int(lengths[vid])
    speed = TRUCK_SPEED_URBAN if vtype == VEH_TRUCK else BIKE_SPEED_URBAN

    if L <= 1:
        return -removal_cost_delta(sol, vtype, vid, pos, dist_matrix)

    dist_savings = -removal_cost_delta(sol, vtype, vid, pos, dist_matrix)

    # Time saved by removing this stop
    cn = int(stops[vid, pos]) + 1
    demand = float(customers[int(stops[vid, pos]), COL_DEMAND])
    service = SERVICE_BASE + SERVICE_PER_100KG * (demand / 100.0)

    prev_dm = 0 if pos == 0 else int(stops[vid, pos - 1]) + 1
    next_dm = 0 if pos == L - 1 else int(stops[vid, pos + 1]) + 1

    old_travel = (dist_matrix[prev_dm, cn] + dist_matrix[cn, next_dm]) / speed * 60.0
    new_travel = dist_matrix[prev_dm, next_dm] / speed * 60.0
    time_saved = old_travel - new_travel + service

    # Count downstream DELIVER stops that benefit from time_saved
    n_downstream = sum(1 for i in range(pos + 1, L)
                       if int(actions[vid, i]) == ACT_DELIVER)

    # Unified TW rate for both cascade and overtime (same concept)
    from src.data.cost import PENALTY_LATE
    tw_rate = PENALTY_LATE

    # Cascade value: each downstream stop's TW improves by time_saved minutes
    cascade_value = n_downstream * time_saved * tw_rate * 0.1

    # Overtime = depot TW violation, same rate
    overtime_value = time_saved * tw_rate if time_saved > 0 else 0.0

    return dist_savings + cascade_value + overtime_value


def best_insertion_pos(sol, vtype, vid, customer, dist_matrix):
    """Best position in 1 route. O(route_len). Returns (pos, delta)."""
    stops, _, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    best_pos = 0
    best_delta = np.inf

    for pos in range(L + 1):
        delta = insertion_cost_delta(sol, vtype, vid, pos, customer, dist_matrix)
        if delta < best_delta:
            best_delta = delta
            best_pos = pos

    return best_pos, best_delta


def _estimate_arrival_at_pos(sol, vtype, vid, pos, dist_matrix, customers):
    """Estimate arrival time at position pos in route. O(pos) scan."""
    stops, actions, lengths = _get_route_arrays(sol, vtype)
    L = int(lengths[vid])
    speed = TRUCK_SPEED_URBAN if vtype == VEH_TRUCK else BIKE_SPEED_URBAN

    clock = 0.0
    prev_dm = 0  # depot
    target = min(pos, L)
    for i in range(target):
        c = int(stops[vid, i])
        clock += dist_matrix[prev_dm, c + 1] / speed * 60.0
        tw_open = float(customers[c, 3])  # COL_TW_OPEN
        if clock < tw_open:
            clock = tw_open
        demand = float(customers[c, COL_DEMAND])
        clock += SERVICE_BASE + SERVICE_PER_100KG * (demand / 100.0)
        prev_dm = c + 1
    return clock, prev_dm


def _estimate_route_return_time(sol, vtype, vid, dist_matrix, customers):
    """Estimate total route time including return to depot."""
    stops, _, lengths = _get_route_arrays(sol, vtype)
    L = int(lengths[vid])
    if L == 0:
        return 0.0
    clock, prev_dm = _estimate_arrival_at_pos(sol, vtype, vid, L, dist_matrix, customers)
    speed = TRUCK_SPEED_URBAN if vtype == VEH_TRUCK else BIKE_SPEED_URBAN
    clock += dist_matrix[prev_dm, 0] / speed * 60.0
    return clock


def _check_tw_at_insertion(sol, vtype, vid, pos, customer, dist_matrix, customers):
    """Quick check: would inserting customer at pos violate its TW?"""
    clock, prev_dm = _estimate_arrival_at_pos(sol, vtype, vid, pos, dist_matrix, customers)
    cn = customer + 1
    arrival = clock + dist_matrix[prev_dm, cn] / (TRUCK_SPEED_URBAN if vtype == VEH_TRUCK else BIKE_SPEED_URBAN) * 60.0
    tw_close = float(customers[customer, COL_TW_CLOSE])
    return arrival <= tw_close


def find_best_insertion_all_routes(sol, vtype, customer, dist_matrix, customers,
                                   vehicle_capacity):
    """Best (vid, pos, delta) across all routes of vtype.
    Checks capacity AND time window feasibility.
    Returns (-1,-1,inf) if none feasible."""
    demand = customers[customer, COL_DEMAND]
    n_vehicles = sol["n_trucks"] if vtype == VEH_TRUCK else sol["n_bikes"]
    loads = _get_loads(sol, vtype)

    best_vid, best_pos, best_delta = -1, -1, np.inf

    lengths = sol["truck_lengths"] if vtype == VEH_TRUCK else sol["bike_lengths"]

    for vid in range(n_vehicles):
        if loads[vid] + demand > vehicle_capacity:
            continue
        # Hard cap: route return time must be under DAY_LENGTH
        rt = _estimate_route_return_time(sol, vtype, vid, dist_matrix, customers)
        if rt > DAY_LENGTH:
            continue
        pos, delta = best_insertion_pos(sol, vtype, vid, customer, dist_matrix)
        if not _check_tw_at_insertion(sol, vtype, vid, pos, customer, dist_matrix, customers):
            continue
        if delta < best_delta:
            best_vid, best_pos, best_delta = vid, pos, delta

    return best_vid, best_pos, best_delta
