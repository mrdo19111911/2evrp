"""Pack route dicts into solution tuple + helper functions.

All numpy arrays produced are i64/i32/i8.
"""
import numpy as np

from ..data.constants import (
    ACT_DELIVER, ACT_RELOAD, COL_DEMAND, COL_TW_OPEN, COL_SERVICE,
    VEH_TRUCK, VEH_BIKE, travel_time_s,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_TRUCK_LENGTHS, SOL_TRUCK_LOADS,
    SOL_TRUCK_DISTANCES, SOL_BIKE_STOPS, SOL_BIKE_ACTIONS, SOL_BIKE_LENGTHS,
    SOL_BIKE_LOADS, SOL_BIKE_DISTANCES,
    SOL_SATELLITES, SOL_META,
    META_N_TRUCKS, META_N_BIKES, META_MAX_ROUTE_LEN, META_N_SATELLITES,
)
from ..data.cost import BIKE_CAPACITY_G, DAY_LENGTH, BIKE_SPEED_US_PER_M
from ..solution.structure import create_solution, rebuild_index
from ..solution._helpers import update_route_distance
from ..solution.satellite_ops import add_satellite_event


def pack_solution(truck_sol, bike_sol, satellites, n_trucks, n_bikes,
                  n_customers, customers=None):
    """Convert route dicts -> solution tuple."""
    sol = create_solution(n_trucks, n_bikes, n_customers)
    _assign_truck_routes(sol, truck_sol, n_trucks, customers)
    _assign_bike_routes(sol, bike_sol, n_bikes, customers)
    _write_satellites(sol, satellites)
    rebuild_index(sol)
    return sol


def _assign_truck_routes(sol, routes, n_trucks, customers):
    """Write truck route dicts into solution tuple arrays."""
    stops_arr = sol[SOL_TRUCK_STOPS]
    actions_arr = sol[SOL_TRUCK_ACTIONS]
    lengths_arr = sol[SOL_TRUCK_LENGTHS]
    loads_arr = sol[SOL_TRUCK_LOADS]
    dists_arr = sol[SOL_TRUCK_DISTANCES]
    max_len = sol[SOL_META][META_MAX_ROUTE_LEN]

    for route in routes:
        vid = route.get("truck_id", -1)
        if vid < 0 or vid >= n_trucks:
            continue
        _write_route(stops_arr, actions_arr, lengths_arr, dists_arr,
                     vid, route, max_len)

    if customers is not None:
        _recompute_loads(stops_arr, actions_arr, lengths_arr, loads_arr,
                         n_trucks, customers)


def _assign_bike_routes(sol, routes, n_bikes, customers):
    """Write bike route dicts into solution tuple arrays."""
    stops_arr = sol[SOL_BIKE_STOPS]
    actions_arr = sol[SOL_BIKE_ACTIONS]
    lengths_arr = sol[SOL_BIKE_LENGTHS]
    loads_arr = sol[SOL_BIKE_LOADS]
    dists_arr = sol[SOL_BIKE_DISTANCES]
    max_len = sol[SOL_META][META_MAX_ROUTE_LEN]

    for route in routes:
        vid = route.get("bike_id", -1)
        if vid < 0 or vid >= n_bikes:
            continue
        _write_route(stops_arr, actions_arr, lengths_arr, dists_arr,
                     vid, route, max_len)

    if customers is not None:
        _recompute_loads(stops_arr, actions_arr, lengths_arr, loads_arr,
                         n_bikes, customers)


def _write_route(stops_arr, actions_arr, lengths_arr, dists_arr,
                 vid, route, max_len):
    """Write one route dict into pre-allocated arrays."""
    offset = int(lengths_arr[vid])
    n_stops = len(route["stops"])
    if offset + n_stops > max_len:
        return
    stops_arr[vid, offset:offset + n_stops] = route["stops"][:n_stops]
    actions_arr[vid, offset:offset + n_stops] = route["actions"][:n_stops]
    lengths_arr[vid] = offset + n_stops
    dists_arr[vid] += route.get("total_distance", 0)


def _recompute_loads(stops_arr, actions_arr, lengths_arr, loads_arr,
                     n_vehicles, customers):
    """Recompute load totals (grams) from route arrays."""
    for vid in range(n_vehicles):
        total = 0
        L = int(lengths_arr[vid])
        for i in range(L):
            if actions_arr[vid, i] == ACT_DELIVER:
                total += int(customers[int(stops_arr[vid, i]), COL_DEMAND])
        loads_arr[vid] = total


def _write_satellites(sol, satellites):
    """Write satellite events into pre-allocated array."""
    if satellites is None or len(satellites) == 0:
        return
    n_events = len(satellites)
    sats = sol[SOL_SATELLITES]
    meta = sol[SOL_META]
    sats[:n_events] = satellites[:n_events]
    meta[META_N_SATELLITES] = n_events


def recompute_all_distances(sol, dist_matrix, n_trucks, n_bikes):
    """Recompute distance cache for all routes."""
    for t in range(n_trucks):
        update_route_distance(sol, VEH_TRUCK, t, dist_matrix)
    for b in range(n_bikes):
        update_route_distance(sol, VEH_BIKE, b, dist_matrix)


def build_fallback_bike_routes(unserved, customers, dist_matrix, n_bikes,
                               existing_bike_sol):
    """NN bike routes for unserved customers. Returns list of route dicts."""
    used_bike_ids = {r.get("bike_id", -1) for r in existing_bike_sol}
    free_bikes = [b for b in range(n_bikes) if b not in used_bike_ids]
    if not free_bikes:
        return []

    routes = []
    remaining = list(unserved)
    bike_idx = 0

    while remaining and bike_idx < len(free_bikes):
        bid = free_bikes[bike_idx]
        trip_stops = []
        trip_actions = []
        trip_load = 0
        trip_clock = 0
        current_dm = 0

        tried = set()
        while True:
            best_node = None
            best_dist = 2**62
            for c in remaining:
                if c in tried:
                    continue
                d = int(dist_matrix[current_dm, c + 1])
                if d < best_dist:
                    best_dist = d
                    best_node = c

            if best_node is None:
                break

            demand = int(customers[best_node, COL_DEMAND])
            if trip_load + demand > BIKE_CAPACITY_G:
                tried.add(best_node)
                continue

            d = int(dist_matrix[current_dm, best_node + 1])
            arrive = trip_clock + travel_time_s(d, BIKE_SPEED_US_PER_M)
            tw_open = int(customers[best_node, COL_TW_OPEN])
            if arrive < tw_open:
                arrive = tw_open
            service = int(customers[best_node, COL_SERVICE])
            dep_d = int(dist_matrix[best_node + 1, 0])
            depot_ret = travel_time_s(dep_d, BIKE_SPEED_US_PER_M)
            if arrive + service + depot_ret > DAY_LENGTH:
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
                "total_distance": 0,
                "bike_id": bid,
            })
        bike_idx += 1
    return routes
