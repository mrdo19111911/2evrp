"""Route feasibility checks for initial solution builder.

Single Responsibility: check constraints for ONE trip/route.
Each function is pure: input data -> output verdict.
All units: meters (i64), seconds (i64), grams (i64).
"""
import numpy as np
from numba import njit

from ..data.constants import (
    COL_DEMAND, COL_TW_OPEN, COL_TW_CLOSE, COL_SERVICE,
    ACT_DELIVER, ACT_RELOAD, travel_time_s,
)
from ..data.cost import DAY_LENGTH, RELOAD_SERVICE_TIME


@njit(cache=True)
def check_trip_capacity(stops, actions, customers, capacity):
    """Check cumulative load (grams) respects capacity. Returns (feasible, max_load)."""
    load = np.int64(0)
    max_load = np.int64(0)
    for i in range(len(stops)):
        if actions[i] == ACT_RELOAD:
            load = np.int64(0)
        elif actions[i] == ACT_DELIVER:
            load += customers[stops[i], COL_DEMAND]
        if load > max_load:
            max_load = load
    return max_load <= capacity, max_load


@njit(cache=True)
def check_trip_time_window(stops, actions, customers, dist_matrix,
                           speed_us, start_node=0):
    """Check TW at DELIVER stops. Returns (feasible, n_tw_violations, total_time_s)."""
    n = len(stops)
    if n == 0:
        return True, np.int32(0), np.int64(0)
    clock = np.int64(0)
    n_violations = np.int32(0)
    prev = np.int32(start_node)
    for i in range(n):
        d = dist_matrix[prev, stops[i] + 1]
        clock += travel_time_s(d, speed_us)
        if actions[i] == ACT_DELIVER:
            tw_close = customers[stops[i], COL_TW_CLOSE]
            if clock > tw_close:
                n_violations += 1
            clock += customers[stops[i], COL_SERVICE]
        elif actions[i] == ACT_RELOAD:
            clock += RELOAD_SERVICE_TIME
        prev = stops[i] + 1
    clock += travel_time_s(dist_matrix[prev, start_node], speed_us)
    feasible = n_violations == 0 and clock <= DAY_LENGTH
    return feasible, n_violations, clock


@njit(cache=True)
def check_trip_distance(stops, dist_matrix, start_node=0):
    """Total travel distance in meters: start -> stops -> start."""
    n = len(stops)
    if n == 0:
        return np.int64(0)
    total = dist_matrix[start_node, stops[0] + 1]
    for i in range(n - 1):
        total += dist_matrix[stops[i] + 1, stops[i + 1] + 1]
    total += dist_matrix[stops[n - 1] + 1, start_node]
    return total


@njit(cache=True)
def check_trip_feasibility(stops, actions, customers, dist_matrix,
                           capacity, speed_us, start_node=0):
    """Combined capacity + TW check. Returns (ok, max_load, n_tw, time_s, dist_m)."""
    cap_ok, max_load = check_trip_capacity(stops, actions, customers, capacity)
    tw_ok, n_tw, total_time = check_trip_time_window(
        stops, actions, customers, dist_matrix, speed_us, start_node)
    distance = check_trip_distance(stops, dist_matrix, start_node)
    return cap_ok and tw_ok, max_load, n_tw, total_time, distance
