"""Initial solution orchestrator -- convoy model (truck + bikes).

All arrays are i64/i32/i8. No separate restricted array --
restricted is customers[:, COL_RESTRICTED].
"""
import numpy as np

from ..data.cost import TRUCK_CAPACITY_G, BIKE_CAPACITY_G, BIKE_SPEED_US_PER_M
from ..data.constants import ACT_DELIVER, ACT_RELOAD
from .clustering import cluster_customers, rebalance_clusters
from .giant_tour import build_giant_tour, build_bike_giant_tour
from .split import split_to_trips, group_trips_to_trucks
from .split_bike import split_bike_gt
from .truck_insertion import (insert_bike_customers_into_trips,
                              build_new_trips_from_remaining,
                              remove_inserted_from_clusters)
from .convoy import assign_customers_to_satellites, build_convoy_bike_routes
from .sync import synchronize_times
from .builder_pack import (pack_solution, recompute_all_distances,
                           build_fallback_bike_routes)


def build_initial_solution(customers, depot, vehicles,
                           dist_matrix, n_trucks, n_bikes, seed=42):
    """Convoy model: truck routes + bikes follow trucks at satellites.

    Pipeline:
      1. Cluster -> truck giant tour
      2. Split truck GT into trips
      3. Insert non-restricted bike customers into truck trips
      4. Group trips -> trucks
      5. Build bike giant tour
      6. Split bike GT into bike routes
      7. Fallback: remaining unserved -> direct bike routes
      8. Sync truck-bike at satellites
    """
    rng = np.random.default_rng(seed)

    # 1. Cluster + truck giant tour
    clusters = cluster_customers(customers, dist_matrix, rng)
    clusters = rebalance_clusters(clusters, customers, TRUCK_CAPACITY_G)
    giant_tour = build_giant_tour(clusters, customers, depot, dist_matrix)

    # 2. Split truck GT into trips
    trips = split_to_trips(giant_tour, customers, dist_matrix,
                           TRUCK_CAPACITY_G)

    # 3. Insert non-restricted bike customers into truck trips
    trips, inserted = insert_bike_customers_into_trips(
        trips, clusters, customers, dist_matrix)
    all_bike_cands = [int(c) for cl in clusters for c in cl["bike_nodes"]]
    new_trips, inserted = build_new_trips_from_remaining(
        all_bike_cands, customers, dist_matrix,
        inserted, TRUCK_CAPACITY_G)
    trips.extend(new_trips)
    clusters = remove_inserted_from_clusters(clusters, inserted)

    # 4. Group trips -> trucks
    truck_sol = group_trips_to_trucks(trips, n_trucks, customers, dist_matrix)

    # Recover dropped-trip customers -> bike pool
    truck_placed = _collect_served(truck_sol)
    dropped_back = [int(c) for c in inserted if c not in truck_placed]

    # 5. Build bike routes
    remaining_bike = [int(c) for cl in clusters for c in cl["bike_nodes"]]
    remaining_bike.extend(dropped_back)
    bike_gt = build_bike_giant_tour(giant_tour, remaining_bike, customers,
                                    dist_matrix)

    # 6. Split bike GT
    bike_sol = split_bike_gt(bike_gt, customers, dist_matrix,
                             BIKE_CAPACITY_G,
                             speed_us=BIKE_SPEED_US_PER_M, n_bikes=n_bikes)

    # 7. Fallback: unserved customers get direct bike routes
    served = _collect_served(truck_sol) | _collect_served(bike_sol)
    unserved = [c for c in range(len(customers)) if c not in served]
    if unserved:
        fallback = build_fallback_bike_routes(
            unserved, customers, dist_matrix, n_bikes, bike_sol)
        bike_sol.extend(fallback)

    # 8. Sync
    satellites = synchronize_times(truck_sol, bike_sol, customers,
                                   dist_matrix, vehicles)

    sol = pack_solution(truck_sol, bike_sol, satellites, n_trucks, n_bikes,
                        len(customers), customers)
    recompute_all_distances(sol, dist_matrix, n_trucks, n_bikes)
    return sol


def _collect_served(route_dicts):
    """Collect set of customer indices served (ACT_DELIVER) across routes."""
    served = set()
    for r in route_dicts:
        for s, a in zip(r["stops"], r["actions"]):
            if a == ACT_DELIVER:
                served.add(int(s))
    return served
