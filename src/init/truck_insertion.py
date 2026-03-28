"""Cheapest insertion of bike customers into truck routes.

All units: meters (i64), seconds (i64), grams (i64).
Hot inner loop uses @njit. No np.insert — pre-allocated trial arrays.
"""
import numpy as np
from numba import njit

from ..data.constants import (
    ACT_DELIVER, COL_DEMAND, COL_RESTRICTED, travel_time_s,
    COL_TW_CLOSE, COL_SERVICE, ACT_RELOAD,
)
from ..data.cost import (
    TRUCK_SPEED_US_PER_M, TRUCK_CAPACITY_G, DAY_LENGTH,
    RELOAD_SERVICE_TIME,
)


@njit(cache=True)
def _insertion_cost(stops, n, pos, node, dist_matrix):
    """Extra distance (meters) if inserting node at pos in stops[:n]."""
    node_dm = node + 1
    if n == 0:
        return dist_matrix[0, node_dm] + dist_matrix[node_dm, 0]
    if pos == 0:
        prev_dm = np.int32(0)
        next_dm = stops[0] + 1
    elif pos == n:
        prev_dm = stops[n - 1] + 1
        next_dm = np.int32(0)
    else:
        prev_dm = stops[pos - 1] + 1
        next_dm = stops[pos] + 1
    old_dist = dist_matrix[prev_dm, next_dm]
    new_dist = dist_matrix[prev_dm, node_dm] + dist_matrix[node_dm, next_dm]
    return new_dist - old_dist


@njit(cache=True)
def _insert_into_trial(src_stops, src_actions, n, pos, node,
                       dst_stops, dst_actions):
    """Insert node at pos into trial arrays. Length becomes n+1."""
    for i in range(pos):
        dst_stops[i] = src_stops[i]
        dst_actions[i] = src_actions[i]
    dst_stops[pos] = node
    dst_actions[pos] = ACT_DELIVER
    for i in range(pos, n):
        dst_stops[i + 1] = src_stops[i]
        dst_actions[i + 1] = src_actions[i]


@njit(cache=True)
def _check_feasibility_fast(stops, actions, n, customers, dist_matrix,
                            capacity, speed_us):
    """Combined cap + TW check. Inline for speed."""
    load = np.int64(0)
    clock = np.int64(0)
    prev = np.int32(0)
    for i in range(n):
        if actions[i] == ACT_RELOAD:
            load = np.int64(0)
            d = dist_matrix[prev, stops[i] + 1]
            clock += travel_time_s(d, speed_us)
            clock += RELOAD_SERVICE_TIME
        elif actions[i] == ACT_DELIVER:
            load += customers[stops[i], COL_DEMAND]
            if load > capacity:
                return False
            d = dist_matrix[prev, stops[i] + 1]
            clock += travel_time_s(d, speed_us)
            if clock > customers[stops[i], COL_TW_CLOSE]:
                return False
            clock += customers[stops[i], COL_SERVICE]
        prev = stops[i] + 1
    clock += travel_time_s(dist_matrix[prev, 0], speed_us)
    return clock <= DAY_LENGTH


@njit(cache=True)
def _find_best_insertion_njit(route_stops, route_actions, n, node,
                              customers, dist_matrix, capacity,
                              trial_stops, trial_actions):
    """Find cheapest feasible insertion pos. Returns (best_pos, best_cost)."""
    best_pos = np.int32(-1)
    best_cost = np.int64(2**62)
    for pos in range(n + 1):
        cost = _insertion_cost(route_stops, n, pos, node, dist_matrix)
        if cost >= best_cost:
            continue
        _insert_into_trial(route_stops, route_actions, n, pos, node,
                           trial_stops, trial_actions)
        if _check_feasibility_fast(trial_stops, trial_actions, n + 1,
                                   customers, dist_matrix, capacity,
                                   TRUCK_SPEED_US_PER_M):
            best_pos = pos
            best_cost = cost
    return best_pos, best_cost


def find_best_insertion(route_stops, route_actions, node, customers,
                        dist_matrix, capacity, trial_stops, trial_actions):
    """Python wrapper. trial arrays are pre-allocated buffers."""
    n = len(route_stops)
    pos, cost = _find_best_insertion_njit(
        route_stops, route_actions, n, node,
        customers, dist_matrix, capacity,
        trial_stops, trial_actions)
    if pos < 0:
        return None, 2**62
    return int(pos), int(cost)


def insert_bike_customers_into_trips(trips, clusters, customers,
                                      dist_matrix):
    """Cheapest insertion of non-restricted bike customers into trips."""
    inserted = set()
    candidates = []
    for cluster in clusters:
        for c in cluster["bike_nodes"]:
            c = int(c)
            if int(customers[c, COL_RESTRICTED]) == 0:
                candidates.append(c)
    candidates.sort(key=lambda c: -int(customers[c, COL_DEMAND]))

    # Pre-allocate trial buffers (reused across all calls)
    max_len = max((len(t["stops"]) for t in trips), default=0) + len(candidates) + 2
    trial_stops = np.empty(max_len, dtype=np.int32)
    trial_actions = np.empty(max_len, dtype=np.int8)

    for cust in candidates:
        best_trip = None
        best_pos = None
        best_cost = 2**62

        for t_idx, trip in enumerate(trips):
            pos, cost = find_best_insertion(
                trip["stops"], trip["actions"], cust,
                customers, dist_matrix, TRUCK_CAPACITY_G,
                trial_stops, trial_actions)
            if pos is not None and cost < best_cost:
                best_trip = t_idx
                best_pos = pos
                best_cost = cost

        if best_trip is not None:
            trip = trips[best_trip]
            trip["stops"] = np.insert(trip["stops"], best_pos, cust)
            trip["actions"] = np.insert(trip["actions"], best_pos, ACT_DELIVER)
            trip["total_demand"] += int(customers[cust, COL_DEMAND])
            inserted.add(cust)

    return trips, inserted


def build_new_trips_from_remaining(candidates, customers, dist_matrix,
                                    inserted, truck_capacity):
    """Build new truck trips from remaining bike customers using NN."""
    remaining = [c for c in candidates
                 if c not in inserted
                 and int(customers[c, COL_RESTRICTED]) == 0]
    if not remaining:
        return [], inserted

    remaining.sort(key=lambda c: -int(dist_matrix[0, c + 1]))

    trial_stops = np.empty(len(remaining) + 2, dtype=np.int32)
    trial_actions = np.empty(len(remaining) + 2, dtype=np.int8)

    new_trips = []
    used = set()

    while remaining:
        trip_stops = []
        trip_actions = []
        trip_load = 0
        current_dm = 0
        skipped = set()

        seed = remaining[0]
        demand = int(customers[seed, COL_DEMAND])
        if demand <= truck_capacity:
            trip_stops.append(seed)
            trip_actions.append(ACT_DELIVER)
            trip_load = demand
            current_dm = seed + 1
            used.add(seed)
            remaining = [c for c in remaining if c not in used]

        while True:
            best_node = None
            best_dist = 2**62
            for c in remaining:
                if c in used or c in skipped:
                    continue
                d = int(dist_matrix[current_dm, c + 1])
                if d < best_dist:
                    best_dist = d
                    best_node = c
            if best_node is None:
                break

            demand = int(customers[best_node, COL_DEMAND])
            if trip_load + demand > truck_capacity:
                skipped.add(best_node)
                continue

            # Quick feasibility check via njit
            n = len(trip_stops)
            for i in range(n):
                trial_stops[i] = trip_stops[i]
                trial_actions[i] = trip_actions[i]
            trial_stops[n] = best_node
            trial_actions[n] = ACT_DELIVER
            if not _check_feasibility_fast(
                    trial_stops, trial_actions, n + 1,
                    customers, dist_matrix, truck_capacity,
                    TRUCK_SPEED_US_PER_M):
                skipped.add(best_node)
                continue

            trip_stops.append(best_node)
            trip_actions.append(ACT_DELIVER)
            trip_load += demand
            current_dm = best_node + 1
            used.add(best_node)

        remaining = [c for c in remaining if c not in used]

        if trip_stops:
            new_trips.append({
                "stops": np.array(trip_stops, dtype=np.int32),
                "actions": np.array(trip_actions, dtype=np.int8),
                "total_demand": trip_load,
                "total_distance": 0,
            })
        else:
            break

    inserted = inserted | used
    return new_trips, inserted


def remove_inserted_from_clusters(clusters, inserted):
    """Remove inserted customers from cluster bike_nodes."""
    for cluster in clusters:
        mask = np.array([int(c) not in inserted
                         for c in cluster["bike_nodes"]])
        cluster["bike_nodes"] = cluster["bike_nodes"][mask]
    return clusters
