"""Initial solution orchestrator — convoy model (truck + bikes)."""
import numpy as np
from ..data.cost import TRUCK_CAPACITY, BIKE_CAPACITY, DAY_LENGTH
from .clustering import cluster_customers, rebalance_clusters
from .giant_tour import build_giant_tour
from .split import split_to_trips, group_trips_to_trucks
from .truck_insertion import (insert_bike_customers_into_trips,
                              build_new_trips_from_remaining,
                              remove_inserted_from_clusters)
from .convoy import assign_customers_to_satellites, build_convoy_bike_routes
from .sync import synchronize_times
from ..solution.structure import create_solution, rebuild_index
from ..data.constants import ACT_DELIVER, ACT_RELOAD, COL_DEMAND


def build_initial_solution(customers, restricted, depot, vehicles,
                           dist_matrix, n_trucks, n_bikes, seed=42):
    """Convoy model: truck routes + bikes follow trucks at satellites."""
    rng = np.random.default_rng(seed)

    # 1. Cluster + giant tour + split → truck trips
    clusters = cluster_customers(customers, restricted, dist_matrix, rng)
    clusters = rebalance_clusters(clusters, customers, TRUCK_CAPACITY)
    giant_tour = build_giant_tour(clusters, customers, depot, dist_matrix)
    trips = split_to_trips(giant_tour, customers, dist_matrix, TRUCK_CAPACITY)

    # 2. Cheapest insertion of bike customers into truck trips
    trips, inserted = insert_bike_customers_into_trips(
        trips, clusters, customers, dist_matrix, restricted)
    all_bike_cands = [int(c) for cl in clusters for c in cl["bike_nodes"]]
    new_trips, inserted = build_new_trips_from_remaining(
        all_bike_cands, customers, dist_matrix, restricted,
        inserted, TRUCK_CAPACITY)
    trips.extend(new_trips)
    clusters = remove_inserted_from_clusters(clusters, inserted)

    # 3. Group trips → trucks
    truck_sol = group_trips_to_trucks(trips, n_trucks, customers, dist_matrix)

    # 4. Convoy: bikes follow trucks at satellites
    bike_sol = _build_convoy_bikes(
        truck_sol, giant_tour, clusters, customers, dist_matrix,
        n_trucks, n_bikes)

    # 5. Sync
    satellites = synchronize_times(truck_sol, bike_sol, customers,
                                   dist_matrix, vehicles)

    return pack_solution(truck_sol, bike_sol, satellites, n_trucks, n_bikes,
                         len(customers), customers)


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
