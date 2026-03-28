"""Time synchronization between truck and bike routes.

All units: meters (i64), seconds (i64), grams (i64).
"""
import numpy as np

from ..data.constants import (
    ACT_DELIVER, ACT_RELOAD, COL_DEMAND, COL_SERVICE,
    SAT_CUST, SAT_BIKE, SAT_TRUCK, SAT_KG, SAT_TIME,
    travel_time_s,
)
from ..data.cost import (
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M, SYNC_DELTA_T,
)


def estimate_all_arrival_times(routes, dist_matrix, speed_us,
                               customers=None):
    """Estimate arrival time (seconds) at each stop. Returns list of ndarray."""
    if not routes:
        return []

    result = []
    for route in routes:
        stops = route["stops"]
        n = len(stops)
        times = np.zeros(n, dtype=np.int64)

        current_time = 0
        for i in range(n):
            if i == 0:
                d = int(dist_matrix[0, stops[0] + 1])
            else:
                d = int(dist_matrix[stops[i - 1] + 1, stops[i] + 1])
            current_time += travel_time_s(d, speed_us)
            times[i] = current_time
            if customers is not None:
                current_time += int(customers[int(stops[i]), COL_SERVICE])
            else:
                current_time += 600  # 10 min default
        result.append(times)
    return result


def match_reload_events(truck_sol, bike_sol, truck_times, bike_times,
                        customers):
    """Pair truck-bike at satellites. Returns satellites i64 ndarray."""
    events = []

    truck_reload_map = {}
    for t_idx, route in enumerate(truck_sol):
        for pos in range(len(route["stops"])):
            if route["actions"][pos] == ACT_RELOAD:
                node = int(route["stops"][pos])
                t_time = int(truck_times[t_idx][pos])
                if node not in truck_reload_map:
                    truck_reload_map[node] = []
                truck_reload_map[node].append((t_idx, pos, t_time))

    for b_idx, route in enumerate(bike_sol):
        sat_node = route.get("satellite_node", -1)
        for pos in range(len(route["stops"])):
            if route["actions"][pos] == ACT_RELOAD:
                node = int(route["stops"][pos])
                b_time = int(bike_times[b_idx][pos])

                lookup = node if node in truck_reload_map else sat_node
                if lookup not in truck_reload_map:
                    continue
                if not truck_reload_map[lookup]:
                    continue

                candidates = truck_reload_map[lookup]
                match = min(candidates,
                            key=lambda m: abs(m[2] - b_time))
                t_idx = match[0]
                t_time = match[2]

                transfer_g = 0
                for k in range(pos + 1, len(route["stops"])):
                    if route["actions"][k] == ACT_RELOAD:
                        break
                    if route["actions"][k] == ACT_DELIVER:
                        transfer_g += int(
                            customers[route["stops"][k], COL_DEMAND])

                planned_time = max(t_time, b_time)
                events.append([node, b_idx, t_idx, transfer_g,
                               planned_time])

    if not events:
        return np.zeros((0, 5), dtype=np.int64)
    return np.array(events, dtype=np.int64)


def adjust_sync_times(satellites, truck_times, bike_times, delta_t):
    """Clamp satellite planned times to non-negative."""
    if satellites.shape[0] == 0:
        return satellites
    adjusted = satellites.copy()
    adjusted[:, SAT_TIME] = np.maximum(adjusted[:, SAT_TIME], 0)
    return adjusted


def synchronize_times(truck_sol, bike_sol, customers, dist_matrix, vehicles):
    """Estimate times + create satellite events. Returns satellites i64."""
    truck_times = estimate_all_arrival_times(
        truck_sol, dist_matrix, TRUCK_SPEED_US_PER_M, customers)
    bike_times = estimate_all_arrival_times(
        bike_sol, dist_matrix, BIKE_SPEED_US_PER_M, customers)
    satellites = match_reload_events(truck_sol, bike_sol, truck_times,
                                     bike_times, customers)
    satellites = adjust_sync_times(satellites, truck_times, bike_times,
                                   SYNC_DELTA_T)
    return satellites
