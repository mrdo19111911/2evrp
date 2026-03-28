"""Split giant tour into truck routes (optimal shortest-path split)."""
import numpy as np

from ..data.constants import ACT_DELIVER, ACT_RELOAD, COL_DEMAND
from ..data.cost import TRUCK_SPEED_URBAN, DAY_LENGTH, RELOAD_SERVICE_TIME
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


def _split_gt_segments(giant_tour):
    """Split GT into segments: each segment = [reload, deliver, deliver, ...].
    First segment may start without reload (from depot).
    Returns list of (reload_node_or_None, [deliver_stops])."""
    segments = []
    current_reload = None
    current_delivers = []

    for stop in giant_tour:
        if stop["type"] == "reload":
            if current_delivers:
                segments.append((current_reload, current_delivers))
                current_delivers = []
            current_reload = stop
        else:
            current_delivers.append(stop)

    if current_delivers:
        segments.append((current_reload, current_delivers))
    return segments


def _assign_to_best_bike(stop, bike_assigned, bike_clocks, bike_prev_dm,
                         sat_node, capacity, dist_matrix, speed, available):
    """Find best bike for a deliver stop. Returns bike index or -1."""
    node = stop["node"]
    demand = stop["demand"]
    best_b = -1
    best_score = np.inf
    for b in available:
        cur_load = sum(d["demand"] for d in bike_assigned[b])
        if cur_load + demand > capacity:
            continue
        prev = sat_node + 1 if sat_node is not None else bike_prev_dm[b]
        if bike_assigned[b]:
            prev = bike_assigned[b][-1]["node"] + 1
        travel = dist_matrix[prev, node + 1] / speed * 60.0
        score = bike_clocks[b] + travel + len(bike_assigned[b]) * 1.0
        if score < best_score:
            best_score = score
            best_b = b
    return best_b


def split_bike_gt(giant_tour, customers, dist_matrix, capacity, speed, n_bikes):
    """Multi-trip bike split with regional assignment + multi-pass easy/hard.

    GT = [R0, D,D,D, R1, D,D,D, R2, D,D,D, ...]
    Each segment between reloads is split across bikes by capacity.
    Each bike: depot → delivers → SAT → delivers → SAT → ... → depot.
    """
    n = len(giant_tour)
    if n == 0:
        return []

    from ..data.constants import COL_TW_OPEN, COL_TW_CLOSE

    segments = _split_gt_segments(giant_tour)
    if not segments:
        return []

    all_bikes = list(range(n_bikes))

    # Initialize bike states
    bike_stops = [[] for _ in range(n_bikes)]
    bike_actions = [[] for _ in range(n_bikes)]
    bike_clocks = [0.0] * n_bikes
    bike_prev_dm = [0] * n_bikes  # all start at depot
    bike_loads = [0.0] * n_bikes

    for seg_idx, (reload_stop, delivers) in enumerate(segments):
        sat_node = reload_stop["node"] if reload_stop is not None else None
        available = all_bikes

        # Multi-pass: classify easy vs hard customers
        easy, hard = [], []
        for stop in delivers:
            node = stop["node"]
            tw_width = float(customers[node, COL_TW_CLOSE]) - float(customers[node, COL_TW_OPEN])
            if tw_width >= 60.0:
                easy.append(stop)
            else:
                hard.append(stop)

        bike_assigned = [[] for _ in range(n_bikes)]

        # Pass 1: assign easy customers (wide TW, flexible)
        for stop in easy:
            best_b = _assign_to_best_bike(
                stop, bike_assigned, bike_clocks, bike_prev_dm,
                sat_node, capacity, dist_matrix, speed, available)
            if best_b >= 0:
                bike_assigned[best_b].append(stop)

        # Pass 2: assign hard customers (tight TW) — prefer bike with most remaining time
        for stop in sorted(hard, key=lambda s: customers[s["node"], COL_TW_CLOSE]):
            best_b = _assign_to_best_bike(
                stop, bike_assigned, bike_clocks, bike_prev_dm,
                sat_node, capacity, dist_matrix, speed, available)
            if best_b >= 0:
                bike_assigned[best_b].append(stop)

        # Now send each bike that has work to satellite + deliver
        for b in range(n_bikes):
            if not bike_assigned[b]:
                continue

            # Simulate reload + delivers first, only commit if ≥1 deliver succeeds
            sim_clock = bike_clocks[b]
            sim_prev = bike_prev_dm[b]

            if sat_node is not None:
                travel = dist_matrix[sim_prev, sat_node + 1] / speed * 60.0
                sim_clock += travel + RELOAD_SERVICE_TIME
                sim_prev = sat_node + 1

            # Check which delivers are feasible after reload
            feasible = []
            fc, fp = sim_clock, sim_prev
            for stop in bike_assigned[b]:
                node = stop["node"]
                demand = stop["demand"]
                travel = dist_matrix[fp, node + 1] / speed * 60.0
                arrive = fc + travel
                tw_open = float(customers[node, COL_TW_OPEN])
                if arrive < tw_open:
                    arrive = tw_open
                service = calc_service_time(demand)
                depot_return = dist_matrix[node + 1, 0] / speed * 60.0
                if arrive + service + depot_return > DAY_LENGTH:
                    break
                feasible.append((stop, arrive + service, node + 1))
                fc = arrive + service
                fp = node + 1

            if not feasible:
                continue  # no delivers feasible → don't reload, don't waste time

            # Commit: reload + delivers
            if sat_node is not None:
                bike_stops[b].append(sat_node)
                bike_actions[b].append(ACT_RELOAD)
                bike_clocks[b] = sim_clock
                bike_prev_dm[b] = sat_node + 1
                bike_loads[b] = 0.0

            for stop, end_clock, end_dm in feasible:
                bike_stops[b].append(stop["node"])
                bike_actions[b].append(ACT_DELIVER)
                bike_clocks[b] = end_clock
                bike_prev_dm[b] = end_dm
                bike_loads[b] += stop["demand"]

    # Pack routes
    routes = []
    for b in range(n_bikes):
        if not bike_stops[b]:
            continue
        routes.append({
            "stops": np.array(bike_stops[b], dtype=np.int32),
            "actions": np.array(bike_actions[b], dtype=np.int8),
            "total_demand": 0.0,
            "total_distance": _compute_route_distance(
                {"stops": np.array(bike_stops[b], dtype=np.int32)}, dist_matrix),
            "bike_id": b,
        })
    return routes
