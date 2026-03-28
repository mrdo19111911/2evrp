"""Route feasibility checks for initial solution builder.

Single Responsibility: check constraints for ONE trip/route.
Each function is pure: input data -> output verdict.
"""
import numpy as np

from ..data.constants import COL_DEMAND, COL_TW_CLOSE, ACT_DELIVER, ACT_RELOAD
from ..data.cost import SERVICE_BASE, SERVICE_PER_100KG, DAY_LENGTH


def calc_service_time(demand):
    """Service time in minutes for a delivery."""
    return SERVICE_BASE + SERVICE_PER_100KG * (demand / 100.0)


def check_trip_capacity(stops, actions, customers, capacity):
    """Check cumulative load respects capacity.

    Load starts at 0, increases at DELIVER, resets to 0 at RELOAD.
    Infeasible if ANY stop exceeds capacity.
    Returns (feasible, max_load).
    """
    load = 0.0
    max_load = 0.0
    for i in range(len(stops)):
        if actions[i] == ACT_RELOAD:
            load = 0.0
        elif actions[i] == ACT_DELIVER:
            load += float(customers[stops[i], COL_DEMAND])
        max_load = max(max_load, load)
    return max_load <= capacity, max_load


def check_trip_time_window(stops, actions, customers, dist_matrix, speed,
                           start_node=0):
    """Check TW at DELIVER stops and total time <= DAY_LENGTH.

    Returns (feasible, n_tw_violations, total_time).
    """
    if len(stops) == 0:
        return True, 0, 0.0
    clock = 0.0
    n_violations = 0
    prev = start_node
    for i in range(len(stops)):
        d = dist_matrix[prev, stops[i] + 1]
        clock += d / speed * 60.0
        if actions[i] == ACT_DELIVER:
            tw_close = float(customers[stops[i], COL_TW_CLOSE])
            if clock > tw_close:
                n_violations += 1
            clock += calc_service_time(float(customers[stops[i], COL_DEMAND]))
        elif actions[i] == ACT_RELOAD:
            clock += calc_service_time(0.0)  # reload service time
        prev = stops[i] + 1
    clock += dist_matrix[prev, start_node] / speed * 60.0  # return
    feasible = n_violations == 0 and clock <= DAY_LENGTH
    return feasible, n_violations, clock


def check_trip_distance(stops, dist_matrix, start_node=0):
    """Total travel distance: start -> stops -> start. Returns distance (km)."""
    if len(stops) == 0:
        return 0.0
    total = dist_matrix[start_node, stops[0] + 1]
    for i in range(len(stops) - 1):
        total += dist_matrix[stops[i] + 1, stops[i + 1] + 1]
    total += dist_matrix[stops[-1] + 1, start_node]
    return float(total)


def check_trip_feasibility(stops, actions, customers, dist_matrix,
                           capacity, speed, start_node=0):
    """Combined capacity + TW check for one trip.

    Returns (feasible, max_load, n_tw_violations, total_time, distance).
    """
    cap_ok, max_load = check_trip_capacity(stops, actions, customers, capacity)
    tw_ok, n_tw, total_time = check_trip_time_window(
        stops, actions, customers, dist_matrix, speed, start_node)
    distance = check_trip_distance(stops, dist_matrix, start_node)
    return cap_ok and tw_ok, max_load, n_tw, total_time, distance
