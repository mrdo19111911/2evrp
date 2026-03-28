"""Convoy builder: truck (queen) + bikes (workers) at each satellite."""
import numpy as np

from ..data.constants import COL_DEMAND, COL_TW_CLOSE, ACT_DELIVER, ACT_RELOAD
from ..data.cost import BIKE_SPEED_URBAN, BIKE_CAPACITY, DAY_LENGTH
from .constraints import calc_service_time


def assign_customers_to_satellites(bike_customers, satellite_nodes, dist_matrix):
    """Each customer -> nearest satellite. Returns dict {sat_node: [cust_list]}."""
    assignment = {s: [] for s in satellite_nodes}
    for c in bike_customers:
        best_s = satellite_nodes[0]
        best_d = np.inf
        for s in satellite_nodes:
            d = dist_matrix[c + 1, s + 1]
            if d < best_d:
                best_d = d
                best_s = s
        assignment[best_s].append(int(c))
    return assignment


def build_bike_trips_at_satellite(sat_node, customers_at_sat, n_bikes,
                                   customers, dist_matrix, clock_start):
    """Build 1 round of bike trips at satellite. Each bike does greedy NN, max 60kg.

    Returns list of trips: [{stops, bike_idx, time}], remaining customers, clock_after.
    """
    speed = BIKE_SPEED_URBAN
    remaining = list(customers_at_sat)
    trips = []

    for bike_idx in range(n_bikes):
        if not remaining:
            break

        trip_stops = []
        trip_load = 0.0
        trip_clock = clock_start
        current_dm = sat_node + 1

        while remaining:
            # Nearest unvisited
            best_node = None
            best_dist = np.inf
            for c in remaining:
                d = dist_matrix[current_dm, c + 1]
                if d < best_dist:
                    best_dist = d
                    best_node = c

            if best_node is None:
                break

            demand = float(customers[best_node, COL_DEMAND])
            if trip_load + demand > BIKE_CAPACITY:
                break

            # Time check
            travel = dist_matrix[current_dm, best_node + 1] / speed * 60.0
            service = calc_service_time(demand)
            arrival = trip_clock + travel

            if arrival > float(customers[best_node, COL_TW_CLOSE]):
                remaining.remove(best_node)
                continue

            # Check can return to satellite after serving
            return_to_sat = dist_matrix[best_node + 1, sat_node + 1] / speed * 60.0
            if arrival + service + return_to_sat > DAY_LENGTH:
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

    Each bike: depot -> [S1,RELOAD] -> custs -> [S2,RELOAD] -> custs -> ... -> depot
    Returns list of route dicts (1 per bike).
    """
    speed = BIKE_SPEED_URBAN
    # Initialize bike states
    bike_stops = [[] for _ in range(n_bikes)]
    bike_actions = [[] for _ in range(n_bikes)]
    bike_clocks = [0.0] * n_bikes
    bike_prev_dm = [0] * n_bikes  # all start from depot

    # Truck clock
    truck_clock = 0.0
    truck_prev_dm = 0  # depot

    for sat_node in satellite_order:
        custs_here = list(sat_customers.get(sat_node, []))
        if not custs_here:
            # Truck just passes through, no bike work
            truck_clock += dist_matrix[truck_prev_dm, sat_node + 1] / speed * 60.0
            truck_clock += calc_service_time(float(customers[sat_node, COL_DEMAND]))
            truck_prev_dm = sat_node + 1
            continue

        # Truck arrives at satellite
        truck_travel = dist_matrix[truck_prev_dm, sat_node + 1] / speed * 60.0
        truck_clock += truck_travel

        # Bikes travel to satellite
        for b in range(n_bikes):
            bike_travel = dist_matrix[bike_prev_dm[b], sat_node + 1] / speed * 60.0
            bike_clocks[b] += bike_travel
            bike_prev_dm[b] = sat_node + 1

        # Sync: everyone waits for slowest
        sync_time = max(truck_clock, max(bike_clocks))

        # Multi-round: bikes fan out, return to satellite, reload, repeat
        remaining = list(custs_here)
        round_clock = sync_time

        while remaining:
            # RELOAD at satellite for all bikes
            for b in range(n_bikes):
                bike_stops[b].append(sat_node)
                bike_actions[b].append(ACT_RELOAD)
                bike_clocks[b] = round_clock
                bike_prev_dm[b] = sat_node + 1

            # Bikes fan out
            trips, remaining = build_bike_trips_at_satellite(
                sat_node, remaining, n_bikes, customers, dist_matrix,
                round_clock)

            if not trips:
                break  # no bike could serve any remaining customer

            for trip in trips:
                b = trip["bike_idx"]
                for c in trip["stops"]:
                    bike_stops[b].append(c)
                    bike_actions[b].append(ACT_DELIVER)
                bike_clocks[b] = trip["clock_end"]
                bike_prev_dm[b] = trip["last_dm"]

            # Next round starts when all bikes return to satellite
            # Bikes travel back to satellite
            for b in range(n_bikes):
                back_travel = dist_matrix[bike_prev_dm[b], sat_node + 1] / speed * 60.0
                bike_clocks[b] += back_travel
                bike_prev_dm[b] = sat_node + 1

            round_clock = max(bike_clocks)

            # Check DAY_LENGTH — stop if too late
            if round_clock > DAY_LENGTH * 0.9:
                break

        # Truck service at satellite + wait for bikes
        truck_service = calc_service_time(float(customers[sat_node, COL_DEMAND]))
        truck_clock = max(sync_time + truck_service, max(bike_clocks))
        truck_prev_dm = sat_node + 1

    # Pack routes
    routes = []
    for b in range(n_bikes):
        if not bike_stops[b]:
            continue
        routes.append({
            "stops": np.array(bike_stops[b], dtype=np.int32),
            "actions": np.array(bike_actions[b], dtype=np.int8),
            "initial_load": 0.0,
            "satellite_node": 0,
            "cluster_idx": 0,
            "bike_id": b,
        })
    return routes


def _reassign_overflow(overflow, satellite_order, current_sat, sat_customers):
    """Move overflow customers to next satellite in order."""
    idx = satellite_order.index(current_sat)
    if idx + 1 < len(satellite_order):
        next_sat = satellite_order[idx + 1]
        sat_customers.setdefault(next_sat, []).extend(overflow)
