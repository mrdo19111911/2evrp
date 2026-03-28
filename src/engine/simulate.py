"""Route simulation — core engine."""
import numpy as np

from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD,
    VEH_TRUCK, VEH_BIKE,
    COL_DEMAND, COL_TW_OPEN, COL_TW_CLOSE,
    VCOL_CAPACITY, VCOL_SPEED,
    ST_COLS,
)
from src.data.cost import SERVICE_BASE, SERVICE_PER_100KG
from src.engine.reload import truck_reload_at_stop, bike_reload_at_stop


def calc_service_time(demand):
    """Service time = 10 min base + 5 min per 100kg."""
    return SERVICE_BASE + SERVICE_PER_100KG * (demand / 100.0)


def simulate_route(stops, actions, vehicle_type, vehicle_capacity, vehicle_speed,
                   customers, dist_matrix, satellites, vehicle_id, delta_t):
    """Simulate 1 route stop-by-stop. Load starts at 0, increases with each delivery.

    Returns (state, return_time, total_distance, feasible).
    """
    l_actual = int(np.sum(stops >= 0))
    if l_actual == 0:
        return np.zeros((0, ST_COLS), dtype=np.float64), 0.0, 0.0, True

    state = np.zeros((l_actual, ST_COLS), dtype=np.float64)
    prev_node, prev_depart = 0, 0.0
    current_load, total_dist = 0.0, 0.0
    feasible = True

    for i in range(l_actual):
        cust = int(stops[i])
        cust_node = cust + 1
        action = int(actions[i])

        d = dist_matrix[prev_node, cust_node]
        total_dist += d
        arrive = prev_depart + d / vehicle_speed * 60.0

        tw_open = customers[cust, COL_TW_OPEN]
        tw_close = customers[cust, COL_TW_CLOSE]
        tw_wait = max(0.0, tw_open - arrive)
        sync_wait = 0.0

        if action == ACT_DELIVER:
            if arrive > tw_close:
                feasible = False
            start = arrive + tw_wait
            demand = customers[cust, COL_DEMAND]
            service = calc_service_time(demand)
            load_before = current_load
            current_load += demand
            if current_load > vehicle_capacity:
                feasible = False

        else:  # ACT_RELOAD
            if vehicle_type == VEH_BIKE:
                # Bike at satellite: got new goods from truck, reset load
                load_before = current_load
                _, service, sync_wait = bike_reload_at_stop(
                    cust, vehicle_id, arrive + tw_wait, satellites)
                current_load = 0.0
            else:
                # Truck at satellite: hands off goods to bikes
                # Truck load += transfer_kg (truck carried these from depot)
                load_before = current_load
                load_change, service, sync_wait = truck_reload_at_stop(
                    cust, vehicle_id, current_load, arrive + tw_wait, satellites)
                current_load += abs(load_change)  # load_change is negative, we want +
            tw_wait += sync_wait
            start = arrive + tw_wait
            if current_load > vehicle_capacity:
                feasible = False

        depart = start + service
        state[i] = [cust, action, arrive, tw_wait + sync_wait, start,
                     service, depart, load_before, current_load, float(feasible)]
        prev_node = cust_node
        prev_depart = depart

    d_back = dist_matrix[prev_node, 0]
    total_dist += d_back
    return_time = prev_depart + d_back / vehicle_speed * 60.0
    return state, return_time, total_dist, feasible


def simulate_all_routes(truck_stops, truck_actions, bike_stops, bike_actions,
                        vehicles, customers, dist_matrix, satellites, delta_t):
    """Simulate all routes. Returns (truck_states, bike_states, return_times, distances, feasible)."""
    n_trucks = len(truck_stops)
    n_bikes = len(bike_stops)

    truck_states = []
    bike_states = []
    truck_return_times = np.zeros(n_trucks, dtype=np.float64)
    bike_return_times = np.zeros(n_bikes, dtype=np.float64)
    truck_distances = np.zeros(n_trucks, dtype=np.float64)
    bike_distances = np.zeros(n_bikes, dtype=np.float64)
    all_feasible = True

    for i in range(n_trucks):
        vid = i
        st, rt, td, f = simulate_route(
            truck_stops[i], truck_actions[i], VEH_TRUCK,
            vehicles[vid, VCOL_CAPACITY], vehicles[vid, VCOL_SPEED],
            customers, dist_matrix, satellites, i, delta_t)
        truck_states.append(st)
        truck_return_times[i] = rt
        truck_distances[i] = td
        if not f:
            all_feasible = False

    for i in range(n_bikes):
        vid = n_trucks + i
        st, rt, td, f = simulate_route(
            bike_stops[i], bike_actions[i], VEH_BIKE,
            vehicles[vid, VCOL_CAPACITY], vehicles[vid, VCOL_SPEED],
            customers, dist_matrix, satellites, i, delta_t)
        bike_states.append(st)
        bike_return_times[i] = rt
        bike_distances[i] = td
        if not f:
            all_feasible = False

    return (truck_states, bike_states, truck_return_times, bike_return_times,
            truck_distances, bike_distances, all_feasible)
