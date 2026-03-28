"""Bike route orchestrator: 1 route per cluster, greedy NN + satellite reload."""
import numpy as np

from ..data.constants import COL_DEMAND, ACT_DELIVER
from ..data.cost import BIKE_SPEED_URBAN, DAY_LENGTH
from .bike_savings import collect_reload_points
from .bike_split import build_cluster_bike_route, pack_bike_route
from .constraints import calc_service_time


def estimate_route_time(route, customers, dist_matrix):
    """Actual time for route: depot -> stops -> depot."""
    stops = route["stops"]
    if len(stops) == 0:
        return 0.0
    speed = BIKE_SPEED_URBAN
    clock = 0.0
    prev = 0
    for s in stops:
        clock += dist_matrix[prev, s + 1] / speed * 60.0
        clock += calc_service_time(float(customers[s, COL_DEMAND]))
        prev = s + 1
    clock += dist_matrix[prev, 0] / speed * 60.0
    return clock


def build_bike_routes(clusters, truck_sol, customers, dist_matrix,
                      n_bikes, bike_capacity, giant_tour=None):
    """Build bike routes: 1 route per cluster, greedy NN, satellite reload.

    Each cluster's bike_nodes get served by 1 bike route.
    If more clusters than bikes, excess routes are dropped.
    Unserved customers within a route (TW miss) are skipped.
    """
    reload_dm = collect_reload_points(giant_tour)

    routes = []
    all_served = set()

    for cluster in clusters:
        if len(cluster["bike_nodes"]) == 0:
            continue

        stops, actions, served = build_cluster_bike_route(
            cluster["bike_nodes"], reload_dm, customers, dist_matrix)
        all_served |= served

        route = pack_bike_route(stops, actions)
        if route is not None:
            routes.append(route)

    # Assign 1 route per bike, sorted by size (biggest first)
    routes.sort(key=lambda r: -len(r["stops"]))
    for i, route in enumerate(routes):
        route["bike_id"] = i if i < n_bikes else -1

    return routes
