"""Solution structure: creation, copy, index management. All int64 on hot path."""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_DELIVER, ACT_PAD, VEH_TRUCK, VEH_BIKE,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_BIKE_STOPS, SOL_BIKE_ACTIONS,
    SOL_TRUCK_LENGTHS, SOL_BIKE_LENGTHS, SOL_TRUCK_LOADS, SOL_BIKE_LOADS,
    SOL_TRUCK_DISTANCES, SOL_BIKE_DISTANCES,
    SOL_CUST_VEHICLE, SOL_CUST_VTYPE, SOL_CUST_ROUTE_POS,
    SOL_SATELLITES, SOL_META,
    META_N_TRUCKS, META_N_BIKES, META_MAX_ROUTE_LEN, META_N_CUSTOMERS, META_N_SATELLITES,
    MAX_SATELLITES,
)


def create_solution(n_trucks, n_bikes, n_customers, max_route_len=100):
    """Create empty solution tuple. i32 for IDs, i64 for measures, i8 for actions."""
    meta = np.array([n_trucks, n_bikes, max_route_len, n_customers, 0], dtype=np.int32)
    return (
        np.full((n_trucks, max_route_len), -1, dtype=np.int32),      # truck_stops
        np.full((n_trucks, max_route_len), ACT_PAD, dtype=np.int8),  # truck_actions
        np.full((n_bikes, max_route_len), -1, dtype=np.int32),       # bike_stops
        np.full((n_bikes, max_route_len), ACT_PAD, dtype=np.int8),   # bike_actions
        np.zeros(n_trucks, dtype=np.int32),                           # truck_lengths
        np.zeros(n_bikes, dtype=np.int32),                            # bike_lengths
        np.zeros(n_trucks, dtype=np.int64),                           # truck_loads (grams)
        np.zeros(n_bikes, dtype=np.int64),                            # bike_loads (grams)
        np.zeros(n_trucks, dtype=np.int64),                           # truck_distances (meters)
        np.zeros(n_bikes, dtype=np.int64),                            # bike_distances (meters)
        np.full(n_customers, -1, dtype=np.int32),                     # cust_vehicle
        np.full(n_customers, -1, dtype=np.int8),                      # cust_vtype
        np.full(n_customers, -1, dtype=np.int32),                     # cust_route_pos
        np.zeros((MAX_SATELLITES, 5), dtype=np.int64),                # satellites
        meta,
    )


def copy_solution(sol):
    """Deep copy all arrays in solution tuple."""
    return tuple(a.copy() for a in sol)


@njit(cache=True)
def copy_solution_into(src, dst):
    """Copy all arrays from src into pre-allocated dst. Zero-alloc."""
    dst[0][:] = src[0]
    dst[1][:] = src[1]
    dst[2][:] = src[2]
    dst[3][:] = src[3]
    dst[4][:] = src[4]
    dst[5][:] = src[5]
    dst[6][:] = src[6]
    dst[7][:] = src[7]
    dst[8][:] = src[8]
    dst[9][:] = src[9]
    dst[10][:] = src[10]
    dst[11][:] = src[11]
    dst[12][:] = src[12]
    dst[13][:] = src[13]
    dst[14][:] = src[14]


def rebuild_index(sol):
    """Rebuild customer index from routes."""
    cust_vehicle = sol[SOL_CUST_VEHICLE]
    cust_vtype = sol[SOL_CUST_VTYPE]
    cust_route_pos = sol[SOL_CUST_ROUTE_POS]
    meta = sol[SOL_META]
    n_trucks = meta[META_N_TRUCKS]
    n_bikes = meta[META_N_BIKES]

    cust_vehicle[:] = -1
    cust_vtype[:] = -1
    cust_route_pos[:] = -1

    truck_stops = sol[SOL_TRUCK_STOPS]
    truck_actions = sol[SOL_TRUCK_ACTIONS]
    truck_lengths = sol[SOL_TRUCK_LENGTHS]
    for t in range(n_trucks):
        for i in range(truck_lengths[t]):
            c = truck_stops[t, i]
            if truck_actions[t, i] == ACT_DELIVER:
                cust_vehicle[c] = t
                cust_vtype[c] = VEH_TRUCK
                cust_route_pos[c] = i

    bike_stops = sol[SOL_BIKE_STOPS]
    bike_actions = sol[SOL_BIKE_ACTIONS]
    bike_lengths = sol[SOL_BIKE_LENGTHS]
    for b in range(n_bikes):
        for i in range(bike_lengths[b]):
            c = bike_stops[b, i]
            if bike_actions[b, i] == ACT_DELIVER:
                cust_vehicle[c] = n_trucks + b
                cust_vtype[c] = VEH_BIKE
                cust_route_pos[c] = i
