"""QA tests for local_search.py edge cases and bug detection. i64 interface."""
import numpy as np
import pytest

from src.data.constants import (
    ACT_DELIVER, ACT_PAD, VEH_TRUCK, VEH_BIKE, COL_DEMAND,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_TRUCK_LENGTHS,
    SOL_TRUCK_LOADS, SOL_TRUCK_DISTANCES,
    SOL_BIKE_LENGTHS, SOL_BIKE_LOADS,
    SOL_CUST_VEHICLE, SOL_CUST_VTYPE, SOL_CUST_ROUTE_POS,
)
from src.solution.structure import create_solution
from src.solution.route_ops import insert_stop
from src.alns.ls_advanced import ejection_chain_3, swap_star
from src.data.cost import (
    TRUCK_COST_PER_M, TRUCK_SPEED_US_PER_M, TRUCK_CAPACITY_G,
    BIKE_COST_PER_M, BIKE_SPEED_US_PER_M, BIKE_CAPACITY_G,
)


def _make_dm_4():
    """4x4 i64 distance matrix."""
    return np.array([
        [0, 10000, 20000, 30000],
        [10000, 0, 10000, 20000],
        [20000, 10000, 0, 10000],
        [30000, 20000, 10000, 0],
    ], dtype=np.int64)


def _customers_3():
    """3 customers, i64 (3, 7)."""
    return np.array([
        [0, 0, 10000, 0, 28800, 300, 0],
        [10000, 10000, 10000, 0, 28800, 300, 0],
        [20000, 20000, 10000, 0, 28800, 300, 0],
    ], dtype=np.int64)


def _vehicles_3trucks():
    return np.array([
        [0, TRUCK_CAPACITY_G, TRUCK_COST_PER_M, TRUCK_SPEED_US_PER_M],
        [0, TRUCK_CAPACITY_G, TRUCK_COST_PER_M, TRUCK_SPEED_US_PER_M],
        [0, TRUCK_CAPACITY_G, TRUCK_COST_PER_M, TRUCK_SPEED_US_PER_M],
    ], dtype=np.int64)


def _add_customer(sol, vtype, vid, cust_id, dist_matrix, customers):
    from src.alns.ls_helpers import get_lengths
    L = get_lengths(sol, vtype)[vid]
    insert_stop(sol, vtype, vid, L, cust_id, ACT_DELIVER, dist_matrix, customers)


# ============================================================================
# TEST 1: ejection_chain_3 with minimum case (3 routes, 1 customer each)
# ============================================================================

def test_ejection_chain_3_minimum_case():
    dm = _make_dm_4()
    customers = _customers_3()
    vehicles = _vehicles_3trucks()
    sol = create_solution(n_trucks=3, n_bikes=0, n_customers=3)

    _add_customer(sol, VEH_TRUCK, 0, 0, dm, customers)
    _add_customer(sol, VEH_TRUCK, 1, 1, dm, customers)
    _add_customer(sol, VEH_TRUCK, 2, 2, dm, customers)

    assert sol[SOL_TRUCK_LENGTHS][0] == 1
    assert sol[SOL_TRUCK_LENGTHS][1] == 1
    assert sol[SOL_TRUCK_LENGTHS][2] == 1

    result = ejection_chain_3(sol, VEH_TRUCK, dm, customers, vehicles)
    assert isinstance(result, (bool, np.bool_))
    total_custs = sum(sol[SOL_TRUCK_LENGTHS][t] for t in range(3))
    assert total_custs == 3


# ============================================================================
# TEST 2: swap_star truck-only (single truck -> False)
# ============================================================================

def test_swap_star_single_truck_returns_false():
    dm = np.array([
        [0, 10000, 20000],
        [10000, 0, 10000],
        [20000, 10000, 0],
    ], dtype=np.int64)
    customers = np.array([
        [0, 0, 5000, 0, 28800, 300, 0],
        [10000, 10000, 5000, 0, 28800, 300, 0],
    ], dtype=np.int64)
    sol = create_solution(n_trucks=1, n_bikes=1, n_customers=2)

    _add_customer(sol, VEH_TRUCK, 0, 0, dm, customers)
    _add_customer(sol, VEH_BIKE, 0, 1, dm, customers)

    result = swap_star(sol, VEH_TRUCK, dm, customers)
    assert isinstance(result, (bool, np.bool_))
    assert result == False


# ============================================================================
# TEST 3: ejection_chain_3 with empty route
# ============================================================================

def test_ejection_chain_3_with_empty_route():
    dm = _make_dm_4()
    customers = _customers_3()
    vehicles = _vehicles_3trucks()
    sol = create_solution(n_trucks=3, n_bikes=0, n_customers=3)

    _add_customer(sol, VEH_TRUCK, 0, 0, dm, customers)
    _add_customer(sol, VEH_TRUCK, 1, 1, dm, customers)
    # T2 remains empty

    result = ejection_chain_3(sol, VEH_TRUCK, dm, customers, vehicles)
    assert isinstance(result, (bool, np.bool_))
