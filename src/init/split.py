"""Split giant tour into truck trips (greedy capacity + time split).

All units: meters (i64), seconds (i64), grams (i64).
"""
import numpy as np

from ..data.constants import (
    ACT_DELIVER, ACT_RELOAD, COL_DEMAND, COL_TW_OPEN, COL_SERVICE,
    travel_time_s,
)
from ..data.cost import TRUCK_SPEED_US_PER_M, DAY_LENGTH

# Re-export bike split functions for backward compatibility
from .split_bike import split_bike_gt, _split_gt_segments  # noqa: F401


def _compute_route_distance(stops, dist_matrix):
    """Total distance in meters: depot -> stops -> depot."""
    if len(stops) == 0:
        return 0
    total = int(dist_matrix[0, stops[0] + 1])
    for i in range(len(stops) - 1):
        total += int(dist_matrix[stops[i] + 1, stops[i + 1] + 1])
    total += int(dist_matrix[stops[-1] + 1, 0])
    return total


def _estimate_trip_time(stops, customers, dist_matrix, speed_us=None):
    """Estimate time in seconds for 1 trip: depot -> stops -> depot."""
    if len(stops) == 0:
        return 0
    if speed_us is None:
        speed_us = TRUCK_SPEED_US_PER_M
    clock = 0
    prev = 0
    for s in stops:
        d = int(dist_matrix[prev, s + 1])
        clock += travel_time_s(d, speed_us)
        tw_open = int(customers[s, COL_TW_OPEN])
        if clock < tw_open:
            clock = tw_open
        clock += int(customers[s, COL_SERVICE])
        prev = s + 1
    clock += travel_time_s(int(dist_matrix[prev, 0]), speed_us)
    return clock


def split_to_trips(giant_tour, customers, dist_matrix, capacity,
                   speed_us=None):
    """Greedy split: fill each trip until capacity or DAY_LENGTH.
    Returns list of trip dicts."""
    n = len(giant_tour)
    if n == 0:
        return []
    if speed_us is None:
        speed_us = TRUCK_SPEED_US_PER_M
    trips = []
    i = 0

    while i < n:
        trip_stops = []
        trip_actions = []
        trip_demand = 0
        clock = 0
        prev_dm = 0

        while i < n:
            node = giant_tour[i]["node"]
            demand = giant_tour[i]["demand"]

            if trip_demand + demand > capacity:
                break

            d = int(dist_matrix[prev_dm, node + 1])
            arrive = clock + travel_time_s(d, speed_us)
            tw_open = int(customers[node, COL_TW_OPEN])
            if arrive < tw_open:
                arrive = tw_open
            service = int(customers[node, COL_SERVICE])
            depot_d = int(dist_matrix[node + 1, 0])
            depot_return = travel_time_s(depot_d, speed_us)

            if arrive + service + depot_return > DAY_LENGTH:
                break

            trip_stops.append(node)
            action = (ACT_RELOAD if giant_tour[i]["type"] == "satellite"
                      else ACT_DELIVER)
            trip_actions.append(action)
            trip_demand += demand
            clock = arrive + service
            prev_dm = node + 1
            i += 1

        if trip_stops:
            stops_arr = np.array(trip_stops, dtype=np.int32)
            trips.append({
                "stops": stops_arr,
                "actions": np.array(trip_actions, dtype=np.int8),
                "total_demand": trip_demand,
                "total_distance": _compute_route_distance(stops_arr,
                                                          dist_matrix),
            })
        else:
            i += 1

    return trips


def group_trips_to_trucks(trips, n_trucks, customers, dist_matrix,
                          speed_us=None):
    """1 trip = 1 vehicle. Extra trips beyond n_trucks are dropped."""
    trip_times = [
        (i, _estimate_trip_time(t["stops"], customers, dist_matrix, speed_us))
        for i, t in enumerate(trips)
    ]
    trip_times.sort(key=lambda x: x[1])

    result = []
    used_trucks = 0

    for trip_idx, trip_time in trip_times:
        if used_trucks >= n_trucks:
            break
        if trip_time > DAY_LENGTH:
            continue
        trip = trips[trip_idx]
        result.append({
            "stops": (np.array(trip["stops"], dtype=np.int32)
                      if not isinstance(trip["stops"], np.ndarray)
                      else trip["stops"]),
            "actions": (np.array(trip["actions"], dtype=np.int8)
                        if not isinstance(trip["actions"], np.ndarray)
                        else trip["actions"]),
            "total_demand": trip["total_demand"],
            "total_distance": trip.get("total_distance", 0),
            "truck_id": used_trucks,
        })
        used_trucks += 1
    return result
