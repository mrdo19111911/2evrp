"""Initial solution orchestrator — convoy model (truck + bikes)."""
import numpy as np
from ..data.cost import TRUCK_CAPACITY, BIKE_CAPACITY, DAY_LENGTH, BIKE_SPEED_URBAN
from .clustering import cluster_customers, rebalance_clusters
from .giant_tour import build_giant_tour, build_bike_giant_tour
from .split import split_to_trips, group_trips_to_trucks
from .truck_insertion import (insert_bike_customers_into_trips,
                              build_new_trips_from_remaining,
                              remove_inserted_from_clusters)
from .convoy import assign_customers_to_satellites, build_convoy_bike_routes
from .sync import synchronize_times
from .constraints import calc_service_time
from ..solution.structure import create_solution, rebuild_index
from ..solution._helpers import update_route_distance
from ..data.constants import (
    ACT_DELIVER, ACT_RELOAD, COL_DEMAND, COL_TW_OPEN, VEH_TRUCK, VEH_BIKE,
)


def build_initial_solution(customers, restricted, depot, vehicles,
                           dist_matrix, n_trucks, n_bikes, seed=42):
    """Convoy model: truck routes + bikes follow trucks at satellites.

    Pipeline:
      1. Cluster → truck giant tour (big nodes + satellites)
      2. Split truck GT into trips (capacity + DAY_LENGTH)
      3. Insert non-restricted bike customers into truck trips
      4. Group trips → trucks (1 trip per truck)
      5. Build bike giant tour (insert bike customers into truck GT)
      6. Split bike GT into bike routes (capacity + DAY_LENGTH)
      7. Fallback: remaining unserved → direct bike routes
      8. Sync truck-bike at satellites
    """
    rng = np.random.default_rng(seed)

    # 1. Cluster + truck giant tour
    clusters = cluster_customers(customers, restricted, dist_matrix, rng)
    clusters = rebalance_clusters(clusters, customers, TRUCK_CAPACITY)
    giant_tour = build_giant_tour(clusters, customers, depot, dist_matrix)

    # 2. Split truck GT into trips
    trips = split_to_trips(giant_tour, customers, dist_matrix, TRUCK_CAPACITY)

    # 3. Insert non-restricted bike customers into truck trips
    trips, inserted = insert_bike_customers_into_trips(
        trips, clusters, customers, dist_matrix, restricted)
    all_bike_cands = [int(c) for cl in clusters for c in cl["bike_nodes"]]
    new_trips, inserted = build_new_trips_from_remaining(
        all_bike_cands, customers, dist_matrix, restricted,
        inserted, TRUCK_CAPACITY)
    trips.extend(new_trips)
    clusters = remove_inserted_from_clusters(clusters, inserted)

    # 4. Group trips → trucks (1 trip per truck). Excess trips dropped.
    truck_sol = group_trips_to_trucks(trips, n_trucks, customers, dist_matrix)

    # Recover customers from dropped trips → back to bike pool
    truck_placed = set()
    for r in truck_sol:
        for s, a in zip(r["stops"], r["actions"]):
            if a == ACT_DELIVER:
                truck_placed.add(int(s))
    dropped_back = [int(c) for c in inserted if c not in truck_placed]

    # 5. Build bike GT from truck GT + remaining bike customers + recovered
    remaining_bike = [int(c) for cl in clusters for c in cl["bike_nodes"]]
    remaining_bike.extend(dropped_back)
    bike_gt = build_bike_giant_tour(giant_tour, remaining_bike, customers, dist_matrix)

    # 6. Split bike GT into bike routes (same algorithm as truck split)
    from .split import split_to_trips as split_trips
    bike_trips = split_trips(bike_gt, customers, dist_matrix, BIKE_CAPACITY,
                             speed=BIKE_SPEED_URBAN)
    bike_sol = _group_bike_trips(bike_trips, n_bikes, customers, dist_matrix)

    # 7. Fallback: any unserved customers get direct bike routes
    served = set()
    for r in truck_sol:
        for s, a in zip(r["stops"], r["actions"]):
            if a == ACT_DELIVER:
                served.add(int(s))
    for r in bike_sol:
        for s, a in zip(r["stops"], r["actions"]):
            if a == ACT_DELIVER:
                served.add(int(s))

    unserved = [c for c in range(len(customers)) if c not in served]
    if unserved:
        fallback_routes = _build_fallback_bike_routes(
            unserved, customers, dist_matrix, n_bikes, bike_sol)
        bike_sol.extend(fallback_routes)

    # 6. Sync
    satellites = synchronize_times(truck_sol, bike_sol, customers,
                                   dist_matrix, vehicles)

    sol = pack_solution(truck_sol, bike_sol, satellites, n_trucks, n_bikes,
                        len(customers), customers)
    # Recompute all distances from scratch (convoy bike routes don't track distance)
    _recompute_all_distances(sol, dist_matrix, n_trucks, n_bikes)
    return sol


def _group_bike_trips(bike_trips, n_bikes, customers, dist_matrix):
    """Assign bike trips to bikes. Same as group_trips_to_trucks but for bikes."""
    from .split import group_trips_to_trucks
    routes = group_trips_to_trucks(bike_trips, n_bikes, customers, dist_matrix,
                                   speed=BIKE_SPEED_URBAN)
    # Rename truck_id -> bike_id
    for i, r in enumerate(routes):
        r["bike_id"] = r.pop("truck_id")
    return routes


def _build_fallback_bike_routes(unserved, customers, dist_matrix, n_bikes,
                                existing_bike_sol):
    """Nearest-neighbor bike routes for unserved customers.
    Enforces BIKE_CAPACITY and DAY_LENGTH per route."""
    from ..data.cost import BIKE_SPEED_URBAN
    from ..data.constants import COL_TW_OPEN
    from ..init.constraints import calc_service_time

    used_bike_ids = {r.get("bike_id", -1) for r in existing_bike_sol}
    free_bikes = [b for b in range(n_bikes) if b not in used_bike_ids]
    if not free_bikes:
        free_bikes = list(range(n_bikes))

    speed = BIKE_SPEED_URBAN
    routes = []
    remaining = list(unserved)
    bike_idx = 0

    while remaining and bike_idx < len(free_bikes):
        bid = free_bikes[bike_idx]
        trip_stops = []
        trip_actions = []
        trip_load = 0.0
        trip_clock = 0.0
        current_dm = 0  # depot

        tried = set()
        while True:
            best_node = None
            best_dist = np.inf
            for c in remaining:
                if c in tried:
                    continue
                d = dist_matrix[current_dm, c + 1]
                if d < best_dist:
                    best_dist = d
                    best_node = c

            if best_node is None:
                break

            demand = float(customers[best_node, COL_DEMAND])
            if trip_load + demand > BIKE_CAPACITY:
                tried.add(best_node)
                continue

            # Time check: travel + wait + service + return to depot
            travel = dist_matrix[current_dm, best_node + 1] / speed * 60.0
            arrive = trip_clock + travel
            tw_open = float(customers[best_node, COL_TW_OPEN])
            if arrive < tw_open:
                arrive = tw_open
            service = calc_service_time(demand)
            depot_return = dist_matrix[best_node + 1, 0] / speed * 60.0
            if arrive + service + depot_return > DAY_LENGTH:
                tried.add(best_node)
                continue

            trip_stops.append(best_node)
            trip_actions.append(ACT_DELIVER)
            trip_load += demand
            trip_clock = arrive + service
            current_dm = best_node + 1
            remaining.remove(best_node)

        if trip_stops:
            routes.append({
                "stops": np.array(trip_stops, dtype=np.int32),
                "actions": np.array(trip_actions, dtype=np.int8),
                "total_demand": trip_load,
                "total_distance": 0.0,
                "bike_id": bid,
            })
        bike_idx += 1

    return routes


def _recompute_all_distances(sol, dist_matrix, n_trucks, n_bikes):
    """Recompute distance cache for all routes."""
    for t in range(n_trucks):
        update_route_distance(sol, VEH_TRUCK, t, dist_matrix)
    for b in range(n_bikes):
        update_route_distance(sol, VEH_BIKE, b, dist_matrix)


def _build_convoy_bikes(truck_sol, giant_tour, clusters, customers,
                        dist_matrix, n_trucks, n_bikes):
    """Assign bikes to truck convoys, build routes at satellites."""
    # Collect all remaining bike customers
    all_bike = [int(c) for cl in clusters for c in cl["bike_nodes"]]
    if not all_bike:
        return []

    # Collect satellite nodes from giant tour
    sat_nodes = [s["node"] for s in giant_tour if s["type"] == "satellite"]
    if not sat_nodes:
        return []

    # Determine convoys: 3 bikes per truck, pick trucks with most satellites
    BIKES_PER_CONVOY = 3
    n_convoys = min(n_bikes // BIKES_PER_CONVOY, len(truck_sol))
    truck_sat_count = []
    for i, r in enumerate(truck_sol):
        n_sats = sum(1 for s in r["stops"] if int(s) in sat_nodes)
        truck_sat_count.append((n_sats, i))
    truck_sat_count.sort(key=lambda x: -x[0])
    convoy_trucks = [idx for _, idx in truck_sat_count[:n_convoys]]

    # Only use satellites from convoy trucks
    convoy_sats = []
    for t_idx in convoy_trucks:
        for s in truck_sol[t_idx]["stops"]:
            if int(s) in sat_nodes and int(s) not in convoy_sats:
                convoy_sats.append(int(s))

    # Assign customers to nearest convoy satellite
    sat_customers = assign_customers_to_satellites(
        all_bike, convoy_sats if convoy_sats else sat_nodes, dist_matrix)

    # Build bike routes per convoy
    all_bike_routes = []
    bike_offset = 0

    for c_idx, t_idx in enumerate(convoy_trucks):
        truck_route = truck_sol[t_idx]
        if len(truck_route["stops"]) == 0:
            continue

        truck_sats = [int(s) for s in truck_route["stops"]
                      if int(s) in sat_nodes]
        if not truck_sats:
            continue

        k_bikes = min(BIKES_PER_CONVOY, n_bikes - bike_offset)
        if k_bikes <= 0:
            break

        routes = build_convoy_bike_routes(
            truck_route, truck_sats, sat_customers,
            k_bikes, customers, dist_matrix)

        for r in routes:
            r["bike_id"] = bike_offset + r["bike_id"]
        all_bike_routes.extend(routes)
        bike_offset += k_bikes

    return all_bike_routes


def pack_solution(truck_sol, bike_sol, satellites, n_trucks, n_bikes,
                  n_customers, customers=None):
    """Convert route dicts -> numpy solution dict."""
    sol = create_solution(n_trucks, n_bikes, n_customers)
    _assign_routes(sol, truck_sol, "truck", n_trucks, customers)
    _assign_routes(sol, bike_sol, "bike", n_bikes, customers)
    sol["satellites"] = satellites
    rebuild_index(sol, n_customers)
    return sol


def _assign_routes(sol, routes, vtype_str, n_vehicles, customers):
    """Assign routes to vehicles."""
    key_stops = f"{vtype_str}_stops"
    key_actions = f"{vtype_str}_actions"
    key_lengths = f"{vtype_str}_lengths"
    key_loads = f"{vtype_str}_loads"
    key_dists = f"{vtype_str}_distances"

    for r_idx, route in enumerate(routes):
        vid = route.get("bike_id", route.get("truck_id", r_idx))
        if vid < 0 or vid >= n_vehicles:
            continue

        offset = int(sol[key_lengths][vid])
        n_stops = len(route["stops"])
        if offset + n_stops > sol["max_route_len"]:
            n_stops = sol["max_route_len"] - offset
            if n_stops <= 0:
                continue

        sol[key_stops][vid, offset:offset + n_stops] = route["stops"][:n_stops]
        sol[key_actions][vid, offset:offset + n_stops] = route["actions"][:n_stops]
        sol[key_lengths][vid] = offset + n_stops
        sol[key_dists][vid] += route.get("total_distance", 0.0)

    if customers is not None:
        for vid in range(n_vehicles):
            total = 0.0
            L = int(sol[key_lengths][vid])
            for i in range(L):
                if sol[key_actions][vid, i] == ACT_DELIVER:
                    total += customers[int(sol[key_stops][vid, i]), COL_DEMAND]
            sol[key_loads][vid] = total
