"""Build bike routes per cluster: greedy NN + satellite reload."""
import numpy as np

from ..data.constants import COL_DEMAND, COL_TW_CLOSE, ACT_DELIVER, ACT_RELOAD
from ..data.cost import BIKE_SPEED_URBAN, BIKE_CAPACITY, DAY_LENGTH
from .constraints import calc_service_time


def find_nearest_reload(cust, reload_dm, dist_matrix):
    """Nearest reload point (satellite or depot) to customer. Returns dm index."""
    best_dm = 0
    best_d = np.inf
    for r_dm in reload_dm:
        d = dist_matrix[cust + 1, r_dm]
        if d < best_d:
            best_d = d
            best_dm = r_dm
    return best_dm, best_d


def build_cluster_bike_route(bike_nodes, reload_dm, customers, dist_matrix):
    """Build 1 bike route for a cluster using greedy NN + reload at nearest satellite.

    Bike starts from depot, visits nearest unvisited customer, reloads when
    capacity exceeded (at nearest satellite), stops when DAY_LENGTH reached.
    Returns (stops, actions, served_set).
    """
    speed = BIKE_SPEED_URBAN
    remaining = set(int(c) for c in bike_nodes)
    stops = []
    actions = []
    load = 0.0
    clock = 0.0
    prev_dm = 0  # depot
    served = set()

    while remaining:
        # Find nearest unvisited customer
        best_node = None
        best_dist = np.inf
        for c in remaining:
            d = dist_matrix[prev_dm, c + 1]
            if d < best_dist:
                best_dist = d
                best_node = c

        if best_node is None:
            break

        demand = float(customers[best_node, COL_DEMAND])

        # Need reload?
        if load + demand > BIKE_CAPACITY:
            # Find nearest satellite (not depot — depot has no customer index)
            sat_dm = _find_satellite_reload(prev_dm, best_node, reload_dm, dist_matrix)
            if sat_dm is None or sat_dm == 0:
                break  # no satellite available, end route

            reload_node = sat_dm - 1  # customer index
            travel_to_sat = dist_matrix[prev_dm, sat_dm] / speed * 60.0
            travel_from_sat = dist_matrix[sat_dm, best_node + 1] / speed * 60.0
            service = calc_service_time(demand)
            arrival = clock + travel_to_sat + travel_from_sat

            # Check time
            nearest_return = dist_matrix[best_node + 1, 0] / speed * 60.0
            if arrival + service + nearest_return > DAY_LENGTH:
                break

            # Check TW
            if arrival > float(customers[best_node, COL_TW_CLOSE]):
                remaining.discard(best_node)
                continue  # skip this customer, try next

            stops.append(reload_node)
            actions.append(ACT_RELOAD)
            clock += travel_to_sat
            prev_dm = sat_dm
            load = 0.0

        # Check time to serve + return
        travel_time = dist_matrix[prev_dm, best_node + 1] / speed * 60.0
        service_time = calc_service_time(demand)
        arrival = clock + travel_time
        nearest_return = dist_matrix[best_node + 1, 0] / speed * 60.0

        if arrival + service_time + nearest_return > DAY_LENGTH:
            break  # out of time

        if arrival > float(customers[best_node, COL_TW_CLOSE]):
            remaining.discard(best_node)
            continue  # TW missed, skip

        # Serve customer
        stops.append(best_node)
        actions.append(ACT_DELIVER)
        load += demand
        clock = arrival + service_time
        prev_dm = best_node + 1
        served.add(best_node)
        remaining.discard(best_node)

    return stops, actions, served


def _find_satellite_reload(prev_dm, next_cust, reload_dm, dist_matrix):
    """Find satellite (not depot) minimizing detour. Returns dm index or None."""
    best_dm = None
    best_cost = np.inf
    for r_dm in reload_dm:
        if r_dm == 0:
            continue  # skip depot
        cost = dist_matrix[prev_dm, r_dm] + dist_matrix[r_dm, next_cust + 1]
        if cost < best_cost:
            best_cost = cost
            best_dm = r_dm
    return best_dm


def pack_bike_route(stops, actions):
    """Pack into route dict."""
    if not stops:
        return None
    return {
        "stops": np.array(stops, dtype=np.int32),
        "actions": np.array(actions, dtype=np.int8),
        "initial_load": 0.0,
        "satellite_node": 0,
        "cluster_idx": 0,
    }
