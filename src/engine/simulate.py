"""Route simulation — core engine. All i64. Fully @njit."""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD,
    VEH_TRUCK, VEH_BIKE,
    COL_DEMAND, COL_TW_OPEN, COL_TW_CLOSE, COL_SERVICE,
    VCOL_CAPACITY, VCOL_SPEED,
    ST_COLS,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_BIKE_STOPS, SOL_BIKE_ACTIONS,
    SOL_TRUCK_LENGTHS, SOL_BIKE_LENGTHS,
    SOL_SATELLITES, SOL_META,
    META_N_TRUCKS, META_N_BIKES, META_N_SATELLITES,
)
from src.data.cost import RELOAD_SERVICE_TIME
from src.engine.reload import truck_reload_at_stop, bike_reload_at_stop


@njit(cache=True)
def simulate_route_into(out_state, stops, actions, length,
                        vehicle_type, vehicle_capacity, speed_us_per_m,
                        customers, dist_matrix, satellites, n_satellites,
                        vehicle_id, delta_t_s):
    """Simulate 1 route into out_state (i64[:,:]). Returns (return_time_s, total_dist_m, feasible)."""
    L = length
    if L == 0:
        return np.int64(0), np.int64(0), True

    prev_node = np.int32(0)
    prev_depart = np.int64(0)
    current_load = np.int64(0)
    total_dist = np.int64(0)
    feasible = True
    reload_service_s = np.int64(RELOAD_SERVICE_TIME)

    for i in range(L):
        cust = stops[i]
        cust_node = cust + 1
        action = actions[i]

        d = dist_matrix[prev_node, cust_node]
        total_dist += d
        travel_s = d * speed_us_per_m // np.int64(1_000_000)
        arrive = prev_depart + travel_s

        tw_open = customers[cust, COL_TW_OPEN]
        tw_close = customers[cust, COL_TW_CLOSE]
        tw_wait = max(np.int64(0), tw_open - arrive)
        sync_wait = np.int64(0)

        if action == ACT_DELIVER:
            if arrive > tw_close:
                feasible = False
            start = arrive + tw_wait
            demand = customers[cust, COL_DEMAND]
            service = customers[cust, COL_SERVICE]
            load_before = current_load
            current_load += demand
            if current_load > vehicle_capacity:
                feasible = False
        else:  # ACT_RELOAD
            if arrive > customers[cust, COL_TW_CLOSE]:
                feasible = False
            if vehicle_type == VEH_BIKE:
                load_before = current_load
                transfer_g, service, sync_wait = bike_reload_at_stop(
                    cust, vehicle_id, arrive + tw_wait,
                    satellites, n_satellites, reload_service_s)
                current_load += transfer_g
            else:
                load_before = current_load
                load_change, service, sync_wait = truck_reload_at_stop(
                    cust, vehicle_id, current_load, arrive + tw_wait,
                    satellites, n_satellites, reload_service_s)
                current_load += load_change
            tw_wait += sync_wait
            start = arrive + tw_wait
            if current_load > vehicle_capacity:
                feasible = False

        depart = start + service
        out_state[i, 0] = cust
        out_state[i, 1] = action
        out_state[i, 2] = arrive
        out_state[i, 3] = tw_wait
        out_state[i, 4] = start
        out_state[i, 5] = service
        out_state[i, 6] = depart
        out_state[i, 7] = load_before
        out_state[i, 8] = current_load
        out_state[i, 9] = np.int64(1) if feasible else np.int64(0)
        prev_node = cust_node
        prev_depart = depart

    d_back = dist_matrix[prev_node, 0]
    total_dist += d_back
    travel_back_s = d_back * speed_us_per_m // np.int64(1_000_000)
    return_time = prev_depart + travel_back_s
    return return_time, total_dist, feasible


@njit(cache=True)
def simulate_all_routes(sol, vehicles, customers, dist_matrix, delta_t_s):
    """Simulate all routes. Fully @njit — no Python overhead."""
    meta = sol[SOL_META]
    n_trucks = meta[META_N_TRUCKS]
    n_bikes = meta[META_N_BIKES]
    max_len = sol[SOL_TRUCK_STOPS].shape[1]
    n_sats = meta[META_N_SATELLITES]
    sats = sol[SOL_SATELLITES]

    truck_sim = np.zeros((n_trucks, max_len, ST_COLS), dtype=np.int64)
    bike_sim = np.zeros((n_bikes, max_len, ST_COLS), dtype=np.int64)
    truck_rt = np.zeros(n_trucks, dtype=np.int64)
    bike_rt = np.zeros(n_bikes, dtype=np.int64)
    truck_dist = np.zeros(n_trucks, dtype=np.int64)
    bike_dist = np.zeros(n_bikes, dtype=np.int64)
    all_feasible = True

    truck_stops = sol[SOL_TRUCK_STOPS]
    truck_actions = sol[SOL_TRUCK_ACTIONS]
    truck_lengths = sol[SOL_TRUCK_LENGTHS]
    for i in range(n_trucks):
        rt, td, f = simulate_route_into(
            truck_sim[i], truck_stops[i], truck_actions[i], truck_lengths[i],
            VEH_TRUCK, vehicles[i, VCOL_CAPACITY], vehicles[i, VCOL_SPEED],
            customers, dist_matrix, sats, n_sats, i, delta_t_s)
        truck_rt[i] = rt
        truck_dist[i] = td
        if not f:
            all_feasible = False

    bike_stops = sol[SOL_BIKE_STOPS]
    bike_actions = sol[SOL_BIKE_ACTIONS]
    bike_lengths = sol[SOL_BIKE_LENGTHS]
    for i in range(n_bikes):
        vid = n_trucks + i
        rt, td, f = simulate_route_into(
            bike_sim[i], bike_stops[i], bike_actions[i], bike_lengths[i],
            VEH_BIKE, vehicles[vid, VCOL_CAPACITY], vehicles[vid, VCOL_SPEED],
            customers, dist_matrix, sats, n_sats, i, delta_t_s)
        bike_rt[i] = rt
        bike_dist[i] = td
        if not f:
            all_feasible = False

    return (truck_sim, bike_sim, truck_rt, bike_rt,
            truck_dist, bike_dist, all_feasible)
