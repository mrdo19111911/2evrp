"""Route and solution cost computation in VND."""
import numpy as np

from src.data.constants import VEH_TRUCK, VEH_BIKE, ST_ACTION, ST_WAIT, ACT_RELOAD
from src.data.cost import (
    TRUCK_TOTAL_KM, TRUCK_DRIVER_HOUR, TRUCK_FIXED_DAY, TRUCK_DEPLOY_COST,
    TRUCK_WAIT_COST_MIN,
    BIKE_TOTAL_KM, BIKE_DRIVER_HOUR, BIKE_FIXED_DAY, BIKE_DEPLOY_COST,
    BIKE_WAIT_COST_MIN,
    RELOAD_HANDLING_COST,
)


def compute_route_cost(total_distance, total_time, n_reloads, vtype,
                       total_wait_time=0.0):
    """Total cost for 1 route in VND. Returns (cost, breakdown)."""
    if vtype == VEH_TRUCK:
        cost_km, driver_hour = TRUCK_TOTAL_KM, TRUCK_DRIVER_HOUR
        fixed, deploy, wait_min = TRUCK_FIXED_DAY, TRUCK_DEPLOY_COST, TRUCK_WAIT_COST_MIN
    else:
        cost_km, driver_hour = BIKE_TOTAL_KM, BIKE_DRIVER_HOUR
        fixed, deploy, wait_min = BIKE_FIXED_DAY, BIKE_DEPLOY_COST, BIKE_WAIT_COST_MIN

    distance_cost = total_distance * cost_km
    time_cost = (total_time / 60.0) * driver_hour
    wait_cost = total_wait_time * wait_min
    reload_cost = n_reloads * RELOAD_HANDLING_COST

    breakdown = {
        "distance_cost": distance_cost, "time_cost": time_cost,
        "wait_cost": wait_cost, "fixed_cost": fixed,
        "deploy_cost": deploy, "reload_cost": reload_cost,
    }
    total = distance_cost + time_cost + wait_cost + fixed + deploy + reload_cost
    return total, breakdown


def compute_makespan(truck_return_times, bike_return_times):
    """Latest vehicle return time."""
    t_max = truck_return_times.max() if len(truck_return_times) > 0 else 0.0
    b_max = bike_return_times.max() if len(bike_return_times) > 0 else 0.0
    return max(float(t_max), float(b_max))


def count_reloads(state):
    """Count RELOAD actions in a state array."""
    if len(state) == 0:
        return 0
    return int(np.sum(state[:, ST_ACTION] == ACT_RELOAD))


def total_wait_time(state):
    """Sum wait time from state array."""
    if len(state) == 0:
        return 0.0
    return float(np.sum(state[:, ST_WAIT]))
