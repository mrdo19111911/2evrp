"""Shared helpers for test_engine tests. ALL i64 per INTERFACE.md."""
import numpy as np

from src.data.constants import (
    ACT_DELIVER, ACT_PAD, PW_SIZE,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_BIKE_STOPS, SOL_BIKE_ACTIONS,
    SOL_TRUCK_LENGTHS, SOL_BIKE_LENGTHS, SOL_SATELLITES, SOL_META,
    META_N_TRUCKS, META_N_BIKES, META_N_SATELLITES,
    PW_UNSERVED, PW_DUPLICATE, PW_CAPACITY,
    PW_TIME_WINDOW, PW_SYNC, PW_VEHICLE_RESTRICTION,
)
from src.solution.structure import create_solution
from src.solution.satellite_ops import add_satellite_event


def make_sol_from_routes(n_trucks, n_bikes, n_customers,
                         truck_routes=None, bike_routes=None,
                         satellites=None, max_route_len=100):
    """Build a solution tuple from simple route descriptions.

    truck_routes / bike_routes: list of list of (customer_idx, action) tuples.
    satellites: list of (cust, bike, truck, grams, time_s) tuples — ALL int.
    """
    sol = create_solution(n_trucks, n_bikes, n_customers, max_route_len)

    if truck_routes:
        for vid, route in enumerate(truck_routes):
            for pos, (cust, act) in enumerate(route):
                sol[SOL_TRUCK_STOPS][vid, pos] = cust
                sol[SOL_TRUCK_ACTIONS][vid, pos] = act
            sol[SOL_TRUCK_LENGTHS][vid] = len(route)

    if bike_routes:
        for vid, route in enumerate(bike_routes):
            for pos, (cust, act) in enumerate(route):
                sol[SOL_BIKE_STOPS][vid, pos] = cust
                sol[SOL_BIKE_ACTIONS][vid, pos] = act
            sol[SOL_BIKE_LENGTHS][vid] = len(route)

    if satellites:
        for (cust, bike, truck, grams, time_s) in satellites:
            add_satellite_event(sol, cust, bike, truck, grams, time_s)

    return sol


def make_penalty_weights():
    """Default penalty weights as i64 (PW_SIZE,) array. VND."""
    pw = np.zeros(PW_SIZE, dtype=np.int64)
    pw[PW_UNSERVED] = 500_000
    pw[PW_DUPLICATE] = 500_000
    pw[PW_CAPACITY] = 50_000
    pw[PW_TIME_WINDOW] = 10_000
    pw[PW_SYNC] = 100_000
    pw[PW_VEHICLE_RESTRICTION] = 200_000
    return pw
