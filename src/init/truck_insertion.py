"""Cheapest insertion of bike customers into truck routes."""
import numpy as np

from ..data.constants import ACT_DELIVER, COL_DEMAND, COL_TW_CLOSE
from ..data.cost import TRUCK_SPEED_URBAN, TRUCK_CAPACITY, DAY_LENGTH
from .constraints import check_trip_feasibility, check_trip_time_window, calc_service_time


def _insertion_cost(stops, pos, node, dist_matrix):
    """Extra distance if inserting node at position pos in stops."""
    n = len(stops)
    if n == 0:
        return dist_matrix[0, node + 1] + dist_matrix[node + 1, 0]

    if pos == 0:
        prev_dm = 0  # depot
        next_dm = stops[0] + 1
    elif pos == n:
        prev_dm = stops[-1] + 1
        next_dm = 0  # depot
    else:
        prev_dm = stops[pos - 1] + 1
        next_dm = stops[pos] + 1

    old_dist = dist_matrix[prev_dm, next_dm]
    new_dist = dist_matrix[prev_dm, node + 1] + dist_matrix[node + 1, next_dm]
    return new_dist - old_dist


def find_best_insertion(route_stops, route_actions, node, customers,
                        dist_matrix, capacity):
    """Find cheapest feasible insertion position for node in route.
    Returns (best_pos, best_cost) or (None, inf) if infeasible."""
    n = len(route_stops)
    best_pos = None
    best_cost = np.inf

    for pos in range(n + 1):
        cost = _insertion_cost(route_stops, pos, node, dist_matrix)
        if cost >= best_cost:
            continue

        # Trial insert
        trial_stops = np.insert(route_stops, pos, node)
        trial_actions = np.insert(route_actions, pos, ACT_DELIVER)

        feasible, _, _, _, _ = check_trip_feasibility(
            trial_stops, trial_actions, customers, dist_matrix,
            capacity, TRUCK_SPEED_URBAN)
        if feasible:
            best_pos = pos
            best_cost = cost

    return best_pos, best_cost


def insert_bike_customers_into_trips(trips, clusters, customers,
                                      dist_matrix, restricted):
    """Cheapest insertion: insert bike customers into individual trips.
    Each trip is depot -> stops -> depot, checked independently.
    Only inserts non-restricted customers (trucks can't enter alleys).
    Returns updated trips and set of inserted customer indices."""
    inserted = set()

    # Collect all bike customers that trucks CAN serve (not restricted)
    candidates = []
    for cluster in clusters:
        for c in cluster["bike_nodes"]:
            c = int(c)
            if restricted[c] == 0:  # truck allowed
                candidates.append(c)

    # Sort by demand descending — insert heaviest first (fills truck capacity better)
    candidates.sort(key=lambda c: -customers[c, COL_DEMAND])

    for cust in candidates:
        best_trip = None
        best_pos = None
        best_cost = np.inf

        for t_idx, trip in enumerate(trips):
            pos, cost = find_best_insertion(
                trip["stops"], trip["actions"], cust,
                customers, dist_matrix, TRUCK_CAPACITY)
            if pos is not None and cost < best_cost:
                best_trip = t_idx
                best_pos = pos
                best_cost = cost

        if best_trip is not None:
            trip = trips[best_trip]
            trip["stops"] = np.insert(trip["stops"], best_pos, cust)
            trip["actions"] = np.insert(trip["actions"], best_pos, ACT_DELIVER)
            trip["total_demand"] += float(customers[cust, COL_DEMAND])
            inserted.add(cust)

    return trips, inserted


def build_new_trips_from_remaining(candidates, customers, dist_matrix, restricted,
                                    inserted, truck_capacity):
    """Build new truck trips from remaining bike customers using nearest neighbor.
    Each trip: depot -> customers -> depot, within capacity + DAY_LENGTH.
    Returns new trips and updated inserted set."""
    remaining = [c for c in candidates if c not in inserted and restricted[c] == 0]
    if not remaining:
        return [], inserted

    # Sort farthest-from-depot first: serve periphery before core
    remaining.sort(key=lambda c: -dist_matrix[0, c + 1])

    new_trips = []
    used = set()

    while remaining:
        # Start new trip: pick farthest unserved as seed, then nearest-neighbor
        trip_stops = []
        trip_actions = []
        trip_load = 0.0
        current_dm = 0  # depot
        skipped_this_trip = set()

        # First stop: farthest remaining customer (already sorted)
        seed = remaining[0]
        demand = float(customers[seed, COL_DEMAND])
        if demand <= truck_capacity:
            trip_stops.append(seed)
            trip_actions.append(ACT_DELIVER)
            trip_load = demand
            current_dm = seed + 1
            used.add(seed)
            remaining = [c for c in remaining if c not in used]

        while True:
            # Find nearest feasible customer from current position
            best_node = None
            best_dist = np.inf
            for c in remaining:
                if c in used or c in skipped_this_trip:
                    continue
                d = dist_matrix[current_dm, c + 1]
                if d < best_dist:
                    best_dist = d
                    best_node = c

            if best_node is None:
                break

            demand = float(customers[best_node, COL_DEMAND])
            if trip_load + demand > truck_capacity:
                skipped_this_trip.add(best_node)
                continue  # try another customer

            # Trial: check time with this customer added
            trial_stops = np.array(trip_stops + [best_node], dtype=np.int32)
            trial_actions = np.array(trip_actions + [ACT_DELIVER], dtype=np.int8)
            tw_ok, _, total_time = check_trip_time_window(
                trial_stops, trial_actions, customers, dist_matrix,
                TRUCK_SPEED_URBAN)
            if not tw_ok or total_time > DAY_LENGTH:
                skipped_this_trip.add(best_node)
                continue  # try another customer in this trip

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
                "total_distance": 0.0,
            })
        else:
            break  # can't build any more trips

    inserted = inserted | used
    return new_trips, inserted


def remove_inserted_from_clusters(clusters, inserted):
    """Remove inserted customers from cluster bike_nodes."""
    for cluster in clusters:
        mask = np.array([int(c) not in inserted for c in cluster["bike_nodes"]])
        cluster["bike_nodes"] = cluster["bike_nodes"][mask]
    return clusters
