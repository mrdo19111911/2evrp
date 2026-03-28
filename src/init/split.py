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


def _estimate_trip_time(stops, customers, dist_matrix):
    """Estimate time for 1 trip: depot -> stops -> depot."""
    if len(stops) == 0:
        return 0.0
    speed = TRUCK_SPEED_URBAN
    clock = 0.0
    prev = 0
    for s in stops:
        clock += dist_matrix[prev, s + 1] / speed * 60.0
        clock += calc_service_time(float(customers[s, COL_DEMAND]))
        prev = s + 1
    clock += dist_matrix[prev, 0] / speed * 60.0
    return clock


def group_trips_to_trucks(trips, n_trucks, customers, dist_matrix):
    """Group trips into trucks, multiple trips per truck within DAY_LENGTH.
    Assigns to least-busy truck. Returns list of truck route dicts."""
    truck_times = np.zeros(n_trucks, dtype=np.float64)
    truck_routes = [{"stops": [], "actions": [], "total_demand": 0.0,
                     "total_distance": 0.0, "n_trips": 0}
                    for _ in range(n_trucks)]

    for trip in trips:
        trip_time = _estimate_trip_time(trip["stops"], customers, dist_matrix)
        tid = int(np.argmin(truck_times))

        if truck_times[tid] + trip_time > DAY_LENGTH:
            # All trucks busy — try to find any truck that fits
            fits = np.where(truck_times + trip_time <= DAY_LENGTH)[0]
            if len(fits) > 0:
                tid = int(fits[np.argmin(truck_times[fits])])
            else:
                # No truck fits — assign to least-busy anyway (ALNS fixes later)
                tid = int(np.argmin(truck_times))

        tr = truck_routes[tid]
        # Insert RELOAD (return to depot) between trips
        if tr["n_trips"] > 0:
            # Use depot node -1 convention? No — truck returns to depot implicitly.
            # In simulation, consecutive stops are connected via dist_matrix.
            # We don't need explicit RELOAD for truck — truck auto-returns to depot
            # between trips. But we need a depot marker for load reset.
            # Actually: truck load doesn't reset at depot in current model.
            # Truck load = cumulative. So no RELOAD needed, just append stops.
            pass

        tr["stops"].extend(trip["stops"].tolist())
        tr["actions"].extend(trip["actions"].tolist())
        tr["total_demand"] += trip["total_demand"]
        tr["total_distance"] += trip.get("total_distance", 0.0)
        tr["n_trips"] += 1
        truck_times[tid] += trip_time

    # Convert to numpy
    result = []
    for tid, tr in enumerate(truck_routes):
        if len(tr["stops"]) == 0:
            continue
        result.append({
            "stops": np.array(tr["stops"], dtype=np.int32),
            "actions": np.array(tr["actions"], dtype=np.int8),
            "total_demand": tr["total_demand"],
            "total_distance": tr["total_distance"],
            "truck_id": tid,
        })
    return result


def split_to_trips(giant_tour, customers, dist_matrix, truck_capacity):
    """Split giant tour into individual trips. Each trip respects capacity + DAY_LENGTH.
    Returns list of trip dicts."""
    n = len(giant_tour)
    if n == 0:
        return []

    speed = TRUCK_SPEED_URBAN
    cost = np.full(n + 1, np.inf)
    pred = np.full(n + 1, -1, dtype=np.int64)
    cost[0] = 0.0

    for i in range(n):
        if cost[i] == np.inf:
            continue
        load = 0.0
        clock = 0.0
        prev_dm = 0  # depot

        for j in range(i, n):
            node = giant_tour[j]["node"]
            load += giant_tour[j]["demand"]
            if load > truck_capacity:
                break

            d = dist_matrix[prev_dm, node + 1]
            clock += d / speed * 60.0
            clock += calc_service_time(giant_tour[j]["demand"])

            return_time = dist_matrix[node + 1, 0] / speed * 60.0
            if clock + return_time > DAY_LENGTH:
                break

            # Distance cost for split graph
            if j == i:
                dist = dist_matrix[0, node + 1]
            else:
                dist = dist_matrix[0, giant_tour[i]["node"] + 1]
                for k in range(i, j):
                    dist += dist_matrix[giant_tour[k]["node"] + 1,
                                        giant_tour[k + 1]["node"] + 1]
            dist_with_return = dist + dist_matrix[node + 1, 0]

            if cost[i] + dist_with_return < cost[j + 1]:
                cost[j + 1] = cost[i] + dist_with_return
                pred[j + 1] = i

            prev_dm = node + 1

    trips = backtrack_split(pred, n, giant_tour)

    for t in trips:
        t["total_distance"] = _compute_route_distance(t, dist_matrix)

    return trips
