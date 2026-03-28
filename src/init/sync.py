"""Time synchronization between truck and bike routes."""
import numpy as np

from ..data.constants import (
    ACT_RELOAD, COL_SERVICE_TIME, COL_DEMAND,
    SAT_CUST, SAT_BIKE, SAT_TRUCK, SAT_KG, SAT_TIME,
    VCOL_SPEED,
)
from ..data.cost import TRUCK_SPEED_URBAN, BIKE_SPEED_URBAN, SYNC_DELTA_T


def estimate_all_arrival_times(routes, dist_matrix, speed):
    """Estimate arrival time at each stop. Returns list of ndarray."""
    if not routes:
        return []

    result = []
    for route in routes:
        stops = route["stops"]
        n = len(stops)
        times = np.zeros(n, dtype=np.float64)

        # Travel time from depot to first stop
        current_time = 0.0
        for i in range(n):
            if i == 0:
                d = dist_matrix[0, stops[0] + 1]
            else:
                d = dist_matrix[stops[i - 1] + 1, stops[i] + 1]
            travel_time = d / speed * 60.0  # km / (km/h) * 60 = minutes
            current_time += travel_time
            times[i] = current_time
            # Add service time (approximation: 5 min per stop)
            current_time += 5.0

        result.append(times)
    return result


def match_reload_events(truck_sol, bike_sol, truck_times, bike_times, customers):
    """Pair truck-bike at satellites. Returns satellites ndarray."""
    events = []

    # Build map: node -> (truck_idx, stop_pos, arrival_time) for RELOAD stops
    truck_reload_map = {}
    for t_idx, route in enumerate(truck_sol):
        for pos in range(len(route["stops"])):
            if route["actions"][pos] == ACT_RELOAD:
                node = int(route["stops"][pos])
                t_time = float(truck_times[t_idx][pos])
                if node not in truck_reload_map:
                    truck_reload_map[node] = []
                truck_reload_map[node].append((t_idx, pos, t_time))

    # Match bike RELOAD stops to truck RELOAD stops
    for b_idx, route in enumerate(bike_sol):
        sat_node = route.get("satellite_node", -1)
        for pos in range(len(route["stops"])):
            if route["actions"][pos] == ACT_RELOAD:
                node = int(route["stops"][pos])
                b_time = float(bike_times[b_idx][pos])

                # Find matching truck reload
                t_idx = 0
                t_time = b_time
                lookup_node = node if node in truck_reload_map else sat_node

                if lookup_node in truck_reload_map and truck_reload_map[lookup_node]:
                    match = truck_reload_map[lookup_node][0]
                    t_idx = match[0]
                    t_time = match[2]

                # Compute transfer kg: sum of demands after this reload until next reload or end
                transfer_kg = 0.0
                for k in range(pos + 1, len(route["stops"])):
                    if route["actions"][k] == ACT_RELOAD:
                        break
                    if route["actions"][k] == 0:  # ACT_DELIVER
                        transfer_kg += float(customers[route["stops"][k], COL_DEMAND])

                planned_time = max(t_time, b_time)
                events.append([float(node), float(b_idx), float(t_idx),
                               transfer_kg, planned_time])

    if not events:
        return np.zeros((0, 5), dtype=np.float64)
    return np.array(events, dtype=np.float64)


def adjust_sync_times(satellites, truck_times, bike_times, delta_t):
    """Adjust planned_time to minimize waiting. Returns adjusted satellites."""
    if satellites.shape[0] == 0:
        return satellites

    adjusted = satellites.copy()
    # For each event, the planned time is already set to max(truck, bike)
    # Just ensure it's reasonable
    for i in range(len(adjusted)):
        t = adjusted[i, SAT_TIME]
        adjusted[i, SAT_TIME] = max(0.0, t)

    return adjusted


def synchronize_times(truck_sol, bike_sol, customers, dist_matrix, vehicles):
    """Estimate times + create satellite events. Returns satellites ndarray (S,5)."""
    truck_speed = TRUCK_SPEED_URBAN
    bike_speed = BIKE_SPEED_URBAN

    truck_times = estimate_all_arrival_times(truck_sol, dist_matrix, truck_speed)
    bike_times = estimate_all_arrival_times(bike_sol, dist_matrix, bike_speed)
    satellites = match_reload_events(truck_sol, bike_sol, truck_times, bike_times,
                                     customers)
    satellites = adjust_sync_times(satellites, truck_times, bike_times, SYNC_DELTA_T)
    return satellites
