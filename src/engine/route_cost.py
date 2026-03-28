"""Route and solution cost computation in VND. All i64. @njit on hot functions."""
import numpy as np
from numba import njit

from src.data.constants import (
    VEH_TRUCK, VEH_BIKE, ST_ACTION, ST_WAIT, ACT_RELOAD,
    VCOL_COST_M, VCOL_SPEED,
)
from src.data.cost import (
    TRUCK_DRIVER_SEC, TRUCK_FIXED_DAY, TRUCK_DEPLOY_COST, TRUCK_WAIT_COST_SEC,
    BIKE_DRIVER_SEC, BIKE_FIXED_DAY, BIKE_DEPLOY_COST, BIKE_WAIT_COST_SEC,
    RELOAD_HANDLING_COST,
)


@njit(cache=True)
def compute_route_cost(total_dist_m, total_time_s, n_reloads, vtype,
                       total_wait_s, vehicles, vid):
    """Total cost for 1 route in VND (i64). Returns i64 scalar."""
    cost_per_m = vehicles[vid, VCOL_COST_M]

    if vtype == VEH_TRUCK:
        driver_sec = np.int64(TRUCK_DRIVER_SEC)
        fixed = np.int64(TRUCK_FIXED_DAY)
        deploy = np.int64(TRUCK_DEPLOY_COST)
        wait_sec = np.int64(TRUCK_WAIT_COST_SEC)
    else:
        driver_sec = np.int64(BIKE_DRIVER_SEC)
        fixed = np.int64(BIKE_FIXED_DAY)
        deploy = np.int64(BIKE_DEPLOY_COST)
        wait_sec = np.int64(BIKE_WAIT_COST_SEC)

    distance_cost = total_dist_m * cost_per_m
    time_cost = total_time_s * driver_sec
    wait_cost = total_wait_s * wait_sec
    reload_cost = np.int64(n_reloads) * np.int64(RELOAD_HANDLING_COST)

    total = distance_cost + time_cost + wait_cost + fixed + deploy + reload_cost
    return total


@njit(cache=True)
def compute_makespan(truck_return_times, bike_return_times):
    """Latest vehicle return time (seconds, i64)."""
    t_max = np.int64(0)
    for i in range(len(truck_return_times)):
        if truck_return_times[i] > t_max:
            t_max = truck_return_times[i]
    for i in range(len(bike_return_times)):
        if bike_return_times[i] > t_max:
            t_max = bike_return_times[i]
    return t_max


@njit(cache=True)
def count_reloads_3d(sim_states, vid, length):
    """Count RELOAD actions in a 3D i64 state slice."""
    if length == 0:
        return np.int32(0)
    count = np.int32(0)
    for i in range(length):
        if sim_states[vid, i, ST_ACTION] == ACT_RELOAD:
            count += 1
    return count


@njit(cache=True)
def total_wait_time_3d(sim_states, vid, length):
    """Sum wait time (seconds, i64) from 3D state slice."""
    if length == 0:
        return np.int64(0)
    total = np.int64(0)
    for i in range(length):
        total += sim_states[vid, i, ST_WAIT]
    return total
