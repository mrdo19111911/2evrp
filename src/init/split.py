"""Split giant tour into truck routes (optimal shortest-path split)."""
import numpy as np

from ..data.constants import ACT_DELIVER, ACT_RELOAD, COL_DEMAND
from ..data.cost import TRUCK_SPEED_URBAN, DAY_LENGTH
from .constraints import calc_service_time, check_trip_feasibility


def backtrack_split(pred, n, giant_tour):
    """Backtrack predecessor array -> route dicts (1 trip each)."""
    boundaries = []
    j = n
    while j > 0:
        i = int(pred[j])
        boundaries.append((i, j))
        j = i
    boundaries.reverse()

    routes = []
    for start, end in boundaries:
        stops_list = []
        actions_list = []
        total_demand = 0.0
        for k in range(start, end):
            stop = giant_tour[k]
            stops_list.append(stop["node"])
            action = ACT_RELOAD if stop["type"] == "satellite" else ACT_DELIVER
            actions_list.append(action)
            total_demand += stop["demand"]

        routes.append({
            "stops": np.array(stops_list, dtype=np.int32),
            "actions": np.array(actions_list, dtype=np.int8),
            "total_demand": total_demand,
            "total_distance": 0.0,
        })
    return routes


def _compute_route_distance(route, dist_matrix):
    """Compute total distance for a route (depot -> stops -> depot)."""
    stops = route["stops"]
    if len(stops) == 0:
        return 0.0
    total = dist_matrix[0, stops[0] + 1]
    for i in range(len(stops) - 1):
        total += dist_matrix[stops[i] + 1, stops[i + 1] + 1]
    total += dist_matrix[stops[-1] + 1, 0]
    return float(total)


def _estimate_trip_time(stops, customers, dist_matrix, speed=None):
    """Estimate time for 1 trip: depot -> stops -> depot.
    Includes travel + wait (arrive before tw_open) + service."""
    if len(stops) == 0:
        return 0.0
    from ..data.constants import COL_TW_OPEN
    if speed is None:
        speed = TRUCK_SPEED_URBAN
    clock = 0.0
    prev = 0
    for s in stops:
        travel = dist_matrix[prev, s + 1] / speed * 60.0
        clock += travel
        tw_open = float(customers[s, COL_TW_OPEN])
        if clock < tw_open:
            clock = tw_open
        clock += calc_service_time(float(customers[s, COL_DEMAND]))
        prev = s + 1
    clock += dist_matrix[prev, 0] / speed * 60.0
    return clock


def group_trips_to_trucks(trips, n_trucks, customers, dist_matrix, speed=None):
    """1 trip = 1 vehicle. No multi-trip. No reload at depot.
    Each vehicle does exactly one trip: depot -> customers -> depot.
    Extra trips beyond n_trucks are dropped (ALNS will handle unserved).
    Assigns shortest-time trips first to maximize utilization."""
    trip_times = [(i, _estimate_trip_time(t["stops"], customers, dist_matrix, speed))
                  for i, t in enumerate(trips)]
    trip_times.sort(key=lambda x: x[1])

    result = []
    used_trucks = 0

    for trip_idx, trip_time in trip_times:
        if used_trucks >= n_trucks:
            break  # no more trucks available, remaining trips dropped
        if trip_time > DAY_LENGTH:
            continue  # trip itself exceeds DAY_LENGTH, skip

        trip = trips[trip_idx]
        result.append({
            "stops": np.array(trip["stops"], dtype=np.int32) if not isinstance(trip["stops"], np.ndarray) else trip["stops"],
            "actions": np.array(trip["actions"], dtype=np.int8) if not isinstance(trip["actions"], np.ndarray) else trip["actions"],
            "total_demand": trip["total_demand"],
            "total_distance": trip.get("total_distance", 0.0),
            "truck_id": used_trucks,
        })
        used_trucks += 1
    return result


def split_to_trips(giant_tour, customers, dist_matrix, capacity, speed=None):
    """Greedy split: fill each trip until capacity or DAY_LENGTH, then start new trip.
    Each trip: depot -> stops -> depot, within capacity + DAY_LENGTH.
    Returns list of trip dicts."""
    n = len(giant_tour)
    if n == 0:
        return []

    from ..data.constants import COL_TW_OPEN
    if speed is None:
        speed = TRUCK_SPEED_URBAN
    trips = []
    i = 0

    while i < n:
        trip_stops = []
        trip_actions = []
        trip_demand = 0.0
        clock = 0.0
        prev_dm = 0  # depot

        while i < n:
            node = giant_tour[i]["node"]
            demand = giant_tour[i]["demand"]

            # Capacity check
            if trip_demand + demand > capacity:
                break

            # Time check: travel + wait + service + return to depot
            travel = dist_matrix[prev_dm, node + 1] / speed * 60.0
            arrive = clock + travel
            tw_open = float(customers[node, COL_TW_OPEN])
            if arrive < tw_open:
                arrive = tw_open
            service = calc_service_time(demand)
            depot_return = dist_matrix[node + 1, 0] / speed * 60.0

            if arrive + service + depot_return > DAY_LENGTH:
                break

            # Add to trip
            trip_stops.append(node)
            action = ACT_RELOAD if giant_tour[i]["type"] == "satellite" else ACT_DELIVER
            trip_actions.append(action)
            trip_demand += demand
            clock = arrive + service
            prev_dm = node + 1
            i += 1

        if trip_stops:
            trip = {
                "stops": np.array(trip_stops, dtype=np.int32),
                "actions": np.array(trip_actions, dtype=np.int8),
                "total_demand": trip_demand,
                "total_distance": _compute_route_distance(
                    {"stops": np.array(trip_stops, dtype=np.int32)}, dist_matrix),
            }
            trips.append(trip)
        else:
            # Current stop can't fit in any trip alone — skip it
            i += 1

    return trips
