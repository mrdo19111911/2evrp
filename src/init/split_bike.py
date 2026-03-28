"""Split bike giant tour into multi-trip bike routes.

All units: meters (i64), seconds (i64), grams (i64).
"""
import numpy as np

from ..data.constants import (
    ACT_DELIVER, ACT_RELOAD, COL_TW_OPEN, COL_TW_CLOSE, COL_SERVICE,
    travel_time_s,
)
from ..data.cost import DAY_LENGTH, RELOAD_SERVICE_TIME


def _compute_route_distance(stops, dist_matrix):
    """Total distance in meters: depot -> stops -> depot."""
    if len(stops) == 0:
        return 0
    total = int(dist_matrix[0, stops[0] + 1])
    for i in range(len(stops) - 1):
        total += int(dist_matrix[stops[i] + 1, stops[i + 1] + 1])
    total += int(dist_matrix[stops[-1] + 1, 0])
    return total


def _split_gt_segments(giant_tour):
    """Split GT into segments: each = [reload, deliver, deliver, ...].
    Returns list of (reload_stop_or_None, [deliver_stops])."""
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
                         sat_node, capacity, dist_matrix, speed_us,
                         available):
    """Find best bike for a deliver stop. Returns bike index or -1."""
    node = stop["node"]
    demand = stop["demand"]
    best_b = -1
    best_score = 2**62
    for b in available:
        cur_load = sum(d["demand"] for d in bike_assigned[b])
        if cur_load + demand > capacity:
            continue
        prev = sat_node + 1 if sat_node is not None else bike_prev_dm[b]
        if bike_assigned[b]:
            prev = bike_assigned[b][-1]["node"] + 1
        d = int(dist_matrix[prev, node + 1])
        t = travel_time_s(d, speed_us)
        score = bike_clocks[b] + t + len(bike_assigned[b])
        if score < best_score:
            best_score = score
            best_b = b
    return best_b


def _process_segment(seg_idx, reload_stop, delivers, customers, dist_matrix,
                     capacity, speed_us, n_bikes, all_bikes,
                     bike_stops, bike_actions, bike_clocks, bike_prev_dm,
                     bike_loads):
    """Process one GT segment: assign delivers to bikes."""
    sat_node = reload_stop["node"] if reload_stop is not None else None
    available = all_bikes

    easy, hard = [], []
    for stop in delivers:
        node = stop["node"]
        tw_w = int(customers[node, COL_TW_CLOSE]) - int(customers[node, COL_TW_OPEN])
        if tw_w >= 3600:
            easy.append(stop)
        else:
            hard.append(stop)

    bike_assigned = [[] for _ in range(n_bikes)]

    for stop in easy:
        best_b = _assign_to_best_bike(
            stop, bike_assigned, bike_clocks, bike_prev_dm,
            sat_node, capacity, dist_matrix, speed_us, available)
        if best_b >= 0:
            bike_assigned[best_b].append(stop)

    for stop in sorted(hard,
                       key=lambda s: customers[s["node"], COL_TW_CLOSE]):
        best_b = _assign_to_best_bike(
            stop, bike_assigned, bike_clocks, bike_prev_dm,
            sat_node, capacity, dist_matrix, speed_us, available)
        if best_b >= 0:
            bike_assigned[best_b].append(stop)

    _commit_bike_assignments(
        bike_assigned, sat_node, customers, dist_matrix, speed_us,
        n_bikes, bike_stops, bike_actions, bike_clocks, bike_prev_dm,
        bike_loads)


def _commit_bike_assignments(bike_assigned, sat_node, customers, dist_matrix,
                             speed_us, n_bikes, bike_stops, bike_actions,
                             bike_clocks, bike_prev_dm, bike_loads):
    """Commit assigned delivers for each bike."""
    for b in range(n_bikes):
        if not bike_assigned[b]:
            continue

        sim_clock = bike_clocks[b]
        sim_prev = bike_prev_dm[b]

        if sat_node is not None:
            d = int(dist_matrix[sim_prev, sat_node + 1])
            sim_clock += travel_time_s(d, speed_us) + RELOAD_SERVICE_TIME
            sim_prev = sat_node + 1

        feasible = []
        fc, fp = sim_clock, sim_prev
        for stop in bike_assigned[b]:
            node = stop["node"]
            d = int(dist_matrix[fp, node + 1])
            arrive = fc + travel_time_s(d, speed_us)
            tw_open = int(customers[node, COL_TW_OPEN])
            if arrive < tw_open:
                arrive = tw_open
            service = int(customers[node, COL_SERVICE])
            depot_d = int(dist_matrix[node + 1, 0])
            depot_ret = travel_time_s(depot_d, speed_us)
            if arrive + service + depot_ret > DAY_LENGTH:
                break
            feasible.append((stop, arrive + service, node + 1))
            fc = arrive + service
            fp = node + 1

        if not feasible:
            continue

        if sat_node is not None:
            bike_stops[b].append(sat_node)
            bike_actions[b].append(ACT_RELOAD)
            bike_clocks[b] = sim_clock
            bike_prev_dm[b] = sat_node + 1
            bike_loads[b] = 0

        for stop, end_clock, end_dm in feasible:
            bike_stops[b].append(stop["node"])
            bike_actions[b].append(ACT_DELIVER)
            bike_clocks[b] = end_clock
            bike_prev_dm[b] = end_dm
            bike_loads[b] += stop["demand"]


def split_bike_gt(giant_tour, customers, dist_matrix, capacity, speed_us,
                  n_bikes):
    """Multi-trip bike split. Returns list of route dicts."""
    n = len(giant_tour)
    if n == 0:
        return []

    segments = _split_gt_segments(giant_tour)
    if not segments:
        return []

    all_bikes = list(range(n_bikes))
    bike_stops = [[] for _ in range(n_bikes)]
    bike_actions = [[] for _ in range(n_bikes)]
    bike_clocks = [0] * n_bikes
    bike_prev_dm = [0] * n_bikes
    bike_loads = [0] * n_bikes

    for seg_idx, (reload_stop, delivers) in enumerate(segments):
        _process_segment(
            seg_idx, reload_stop, delivers, customers, dist_matrix,
            capacity, speed_us, n_bikes, all_bikes,
            bike_stops, bike_actions, bike_clocks, bike_prev_dm, bike_loads)

    routes = []
    for b in range(n_bikes):
        if not bike_stops[b]:
            continue
        stops_arr = np.array(bike_stops[b], dtype=np.int32)
        routes.append({
            "stops": stops_arr,
            "actions": np.array(bike_actions[b], dtype=np.int8),
            "total_demand": bike_loads[b],
            "total_distance": _compute_route_distance(stops_arr, dist_matrix),
            "bike_id": b,
        })
    return routes
