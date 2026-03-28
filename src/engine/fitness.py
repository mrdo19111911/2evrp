"""Fitness evaluation orchestrator — fully @njit. Returns eval result as i64 array."""
import numpy as np
from numba import njit

from src.data.constants import (
    VEH_TRUCK, VEH_BIKE,
    SOL_TRUCK_LENGTHS, SOL_BIKE_LENGTHS, SOL_TRUCK_STOPS,
    SOL_META, META_N_TRUCKS, META_N_BIKES, META_N_SATELLITES,
    SOL_SATELLITES, ST_CUST,
    EV_FITNESS, EV_COST, EV_SYNC_COST, EV_MAKESPAN,
    EV_TOTAL_PENALTY, EV_TOTAL_WAIT, EV_FEASIBLE,
    EV_SIM_FEASIBLE, EV_VALID, EV_N_FEASIBLE_ROUTES,
    EV_SIZE, ACT_DELIVER,
)
from src.engine.simulate import simulate_all_routes
from src.engine.validate import validate_all
from src.engine.route_cost import (
    compute_route_cost, compute_makespan, count_reloads_3d, total_wait_time_3d,
)
from src.engine.violations import compute_penalties
from src.engine.sync_cost import compute_sync_cost
from src.data.cost import DAY_LENGTH


FEASIBLE_ROUTE_BONUS = np.int64(2000000)


@njit(cache=True)
def compute_fitness(total_cost, sync_cost, total_penalty, n_feasible_routes=0):
    """VND sum fitness (i64). Lower = better."""
    return total_cost + sync_cost + total_penalty - np.int64(n_feasible_routes) * FEASIBLE_ROUTE_BONUS


@njit(cache=True)
def evaluate_solution(sol, customers, vehicles, dist_matrix,
                      delta_t_s, penalty_w):
    """One-stop: simulate -> validate -> fitness. Fully @njit. Returns i64 (EV_SIZE,)."""
    meta = sol[SOL_META]
    n_trucks = meta[META_N_TRUCKS]
    n_bikes = meta[META_N_BIKES]
    n_customers = len(customers)
    n_sats = meta[META_N_SATELLITES]
    sats = sol[SOL_SATELLITES]

    # 1. Simulate
    (truck_sim, bike_sim, truck_rt, bike_rt,
     truck_dist, bike_dist, sim_feasible) = simulate_all_routes(
        sol, vehicles, customers, dist_matrix, delta_t_s)

    truck_lengths = sol[SOL_TRUCK_LENGTHS]
    bike_lengths = sol[SOL_BIKE_LENGTHS]

    # 2. Validate
    (valid, unserved, n_unserved, n_dup,
     vr_viol, n_vr, cap_viol, n_cap, tw_viol, n_tw,
     sync_viol, n_sync) = validate_all(
        sol, truck_sim, bike_sim, customers,
        vehicles, n_customers, delta_t_s)

    # 3. Operating cost
    total_cost = np.int64(0)
    total_wait = np.int64(0)
    for i in range(n_trucks):
        L = np.int32(truck_lengths[i])
        wt = total_wait_time_3d(truck_sim, i, L)
        total_wait += wt
        total_cost += compute_route_cost(
            truck_dist[i], truck_rt[i],
            count_reloads_3d(truck_sim, i, L), VEH_TRUCK, wt, vehicles, i)
    for i in range(n_bikes):
        L = np.int32(bike_lengths[i])
        wt = total_wait_time_3d(bike_sim, i, L)
        total_wait += wt
        vid = n_trucks + i
        total_cost += compute_route_cost(
            bike_dist[i], bike_rt[i],
            count_reloads_3d(bike_sim, i, L), VEH_BIKE, wt, vehicles, vid)

    # 4. Makespan, sync, penalties
    makespan = compute_makespan(truck_rt, bike_rt)
    sync_cost = compute_sync_cost(
        truck_sim, truck_lengths, bike_sim, bike_lengths, sats, n_sats, delta_t_s)

    all_rt = np.empty(n_trucks + n_bikes, dtype=np.int64)
    for i in range(n_trucks):
        all_rt[i] = truck_rt[i]
    for i in range(n_bikes):
        all_rt[n_trucks + i] = bike_rt[i]

    total_penalty, _ = compute_penalties(
        unserved, n_unserved, n_dup,
        cap_viol, n_cap, tw_viol, n_tw,
        sync_viol, n_sync, n_vr,
        all_rt, n_trucks + n_bikes,
        penalty_w, customers, dist_matrix, vehicles)

    # 5. Count feasible routes — boolean flags instead of set()
    viol_flags = np.zeros(n_trucks + n_bikes, dtype=np.int8)
    for i in range(n_tw):
        vt = np.int32(tw_viol[i, 0])
        vid = np.int32(tw_viol[i, 1])
        if vt == 0:
            viol_flags[vid] = 1
        else:
            viol_flags[n_trucks + vid] = 1
    for i in range(n_cap):
        vt = np.int32(cap_viol[i, 0])
        vid = np.int32(cap_viol[i, 1])
        if vt == 0:
            viol_flags[vid] = 1
        else:
            viol_flags[n_trucks + vid] = 1
    for i in range(n_vr):
        cust_idx = np.int32(vr_viol[i])
        for t in range(n_trucks):
            L = np.int32(truck_lengths[t])
            for j in range(L):
                if np.int32(truck_sim[t, j, ST_CUST]) == cust_idx:
                    viol_flags[t] = 1

    day = np.int64(DAY_LENGTH)
    n_feasible = np.int32(0)
    for i in range(n_trucks):
        if truck_rt[i] <= day and viol_flags[i] == 0 and truck_rt[i] > 0:
            n_feasible += 1
    for i in range(n_bikes):
        if bike_rt[i] <= day and viol_flags[n_trucks + i] == 0 and bike_rt[i] > 0:
            n_feasible += 1

    fitness = total_cost + sync_cost + total_penalty - np.int64(n_feasible) * FEASIBLE_ROUTE_BONUS

    ev = np.zeros(EV_SIZE, dtype=np.int64)
    ev[EV_FITNESS] = fitness
    ev[EV_COST] = total_cost
    ev[EV_SYNC_COST] = sync_cost
    ev[EV_MAKESPAN] = makespan
    ev[EV_TOTAL_PENALTY] = total_penalty
    ev[EV_TOTAL_WAIT] = total_wait
    ev[EV_FEASIBLE] = np.int64(1) if (sim_feasible and valid) else np.int64(0)
    ev[EV_SIM_FEASIBLE] = np.int64(1) if sim_feasible else np.int64(0)
    ev[EV_VALID] = np.int64(1) if valid else np.int64(0)
    ev[EV_N_FEASIBLE_ROUTES] = np.int64(n_feasible)
    return ev
