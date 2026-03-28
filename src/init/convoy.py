"""Convoy builder: truck (queen) + bikes (workers) at each satellite.

All units: meters (i64), seconds (i64), grams (i64).
"""
import numpy as np

from ..data.constants import (
    COL_DEMAND, COL_TW_CLOSE, COL_SERVICE,
    ACT_DELIVER, ACT_RELOAD, travel_time_s,
)
from ..data.cost import (
    BIKE_SPEED_US_PER_M, TRUCK_SPEED_US_PER_M,
    BIKE_CAPACITY_G, DAY_LENGTH, RELOAD_SERVICE_TIME,
)


def assign_customers_to_satellites(bike_customers, satellite_nodes,
                                   dist_matrix):
    """Each customer -> nearest satellite. Returns {sat: [cust_list]}."""
    assignment = {s: [] for s in satellite_nodes}
    for c in bike_customers:
        best_s = satellite_nodes[0]
        best_d = 2**62
        for s in satellite_nodes:
            d = int(dist_matrix[c + 1, s + 1])
            if d < best_d:
                best_d = d
                best_s = s
        assignment[best_s].append(int(c))
    return assignment


def build_bike_trips_at_satellite(sat_node, customers_at_sat, n_bikes,
                                   customers, dist_matrix, clock_start):
    """Build 1 round of bike trips at satellite. Greedy NN.

    Returns list of trips, remaining customers.
    """
    remaining = list(customers_at_sat)
    trips = []

    for bike_idx in range(n_bikes):
        if not remaining:
            break

        trip_stops = []
        trip_load = 0
        trip_clock = clock_start
        current_dm = sat_node + 1

        while remaining:
            best_node = None
            best_dist = 2**62
            for c in remaining:
                d = int(dist_matrix[current_dm, c + 1])
                if d < best_dist:
                    best_dist = d
                    best_node = c

            if best_node is None:
                break

            demand = int(customers[best_node, COL_DEMAND])
            if trip_load + demand > BIKE_CAPACITY_G:
                break

            d = int(dist_matrix[current_dm, best_node + 1])
            arrival = trip_clock + travel_time_s(d, BIKE_SPEED_US_PER_M)

            if arrival > int(customers[best_node, COL_TW_CLOSE]):
                break

            service = int(customers[best_node, COL_SERVICE])
            ret_d = int(dist_matrix[best_node + 1, sat_node + 1])
            ret_sat = travel_time_s(ret_d, BIKE_SPEED_US_PER_M)
            dep_d = int(dist_matrix[sat_node + 1, 0])
            dep_ret = travel_time_s(dep_d, BIKE_SPEED_US_PER_M)
            if arrival + service + ret_sat + dep_ret > DAY_LENGTH:
                break

            trip_clock = arrival + service
            trip_stops.append(best_node)
            trip_load += demand
            current_dm = best_node + 1
            remaining.remove(best_node)

        if trip_stops:
            trips.append({"stops": trip_stops, "bike_idx": bike_idx,
                          "last_dm": current_dm, "clock_end": trip_clock})
    return trips, remaining


def build_convoy_bike_routes(truck_trip, satellite_order, sat_customers,
                              n_bikes, customers, dist_matrix):
    """Build bike routes for 1 convoy following truck through satellites.

    Returns list of route dicts (1 per bike).
    """
    bike_stops = [[] for _ in range(n_bikes)]
    bike_actions = [[] for _ in range(n_bikes)]
    bike_clocks = [0] * n_bikes
    bike_prev_dm = [0] * n_bikes

    truck_clock = 0
    truck_prev_dm = 0

    for sat_node in satellite_order:
        custs_here = list(sat_customers.get(sat_node, []))
        if not custs_here:
            d = int(dist_matrix[truck_prev_dm, sat_node + 1])
            truck_clock += travel_time_s(d, TRUCK_SPEED_US_PER_M)
            truck_clock += RELOAD_SERVICE_TIME
            truck_prev_dm = sat_node + 1
            continue

        d = int(dist_matrix[truck_prev_dm, sat_node + 1])
        truck_clock += travel_time_s(d, TRUCK_SPEED_US_PER_M)

        for b in range(n_bikes):
            bd = int(dist_matrix[bike_prev_dm[b], sat_node + 1])
            bike_clocks[b] += travel_time_s(bd, BIKE_SPEED_US_PER_M)
            bike_prev_dm[b] = sat_node + 1

        sync_time = max(truck_clock, max(bike_clocks))
        remaining = list(custs_here)
        round_clock = sync_time

        while remaining:
            for b in range(n_bikes):
                bike_stops[b].append(sat_node)
                bike_actions[b].append(ACT_RELOAD)
                bike_clocks[b] = round_clock
                bike_prev_dm[b] = sat_node + 1

            trips, remaining = build_bike_trips_at_satellite(
                sat_node, remaining, n_bikes, customers, dist_matrix,
                round_clock)
            if not trips:
                break

            for trip in trips:
                b = trip["bike_idx"]
                for c in trip["stops"]:
                    bike_stops[b].append(c)
                    bike_actions[b].append(ACT_DELIVER)
                bike_clocks[b] = trip["clock_end"]
                bike_prev_dm[b] = trip["last_dm"]

            for b in range(n_bikes):
                bd = int(dist_matrix[bike_prev_dm[b], sat_node + 1])
                bike_clocks[b] += travel_time_s(bd, BIKE_SPEED_US_PER_M)
                bike_prev_dm[b] = sat_node + 1

            round_clock = max(bike_clocks)

            max_return = max(
                bike_clocks[b] + travel_time_s(
                    int(dist_matrix[bike_prev_dm[b], 0]),
                    BIKE_SPEED_US_PER_M)
                for b in range(n_bikes))
            if max_return > DAY_LENGTH:
                break

        truck_clock = max(sync_time + RELOAD_SERVICE_TIME,
                          max(bike_clocks))
        truck_prev_dm = sat_node + 1

    routes = []
    first_sat = satellite_order[0] if satellite_order else 0
    for b in range(n_bikes):
        if not bike_stops[b]:
            continue
        bike_sat = first_sat
        for k, act in enumerate(bike_actions[b]):
            if act == ACT_RELOAD:
                bike_sat = bike_stops[b][k]
                break
        routes.append({
            "stops": np.array(bike_stops[b], dtype=np.int32),
            "actions": np.array(bike_actions[b], dtype=np.int8),
            "initial_load": 0,
            "satellite_node": bike_sat,
            "cluster_idx": 0,
            "bike_id": b,
        })
    return routes
