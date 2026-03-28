"""Tests for cross-route operations: move_stop, swap_stops_between, clear_route, copy_solution.
Tuple-based solution format. All i64.
"""
import numpy as np
import pytest

from src.data.constants import (
    ACT_DELIVER, ACT_PAD, ACT_RELOAD, VEH_TRUCK, VEH_BIKE, COL_DEMAND,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_TRUCK_LENGTHS,
    SOL_BIKE_STOPS, SOL_BIKE_ACTIONS, SOL_BIKE_LENGTHS,
    SOL_TRUCK_LOADS, SOL_BIKE_LOADS,
    SOL_TRUCK_DISTANCES, SOL_BIKE_DISTANCES,
    SOL_CUST_VEHICLE, SOL_CUST_VTYPE, SOL_CUST_ROUTE_POS,
    SOL_META, META_N_TRUCKS, META_N_BIKES, META_MAX_ROUTE_LEN,
)
from src.solution.structure import create_solution, copy_solution
from src.solution.route_ops import insert_stop
from src.solution.solution_ops import (
    move_stop,
    swap_stops_between,
    clear_route,
    create_route_from_sequence,
)


@pytest.fixture
def sol_truck_with_2stops(tiny_instance, tiny_dist_matrix):
    """Truck route with 2 customers: [C0, C1]."""
    sol = create_solution(
        tiny_instance["n_trucks"],
        tiny_instance["n_bikes"],
        tiny_instance["n_customers"],
    )
    customers = tiny_instance["customers"]
    dist_matrix = tiny_dist_matrix

    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, dist_matrix, customers)
    insert_stop(sol, VEH_TRUCK, 0, 1, 1, ACT_DELIVER, dist_matrix, customers)

    return sol, customers, dist_matrix


@pytest.fixture
def sol_two_bikes_with_stops(tiny_instance, tiny_dist_matrix):
    """Two bike routes: bike 0 has [C2], bike 1 has [C3]."""
    sol = create_solution(
        tiny_instance["n_trucks"],
        tiny_instance["n_bikes"],
        tiny_instance["n_customers"],
    )
    customers = tiny_instance["customers"]
    dist_matrix = tiny_dist_matrix

    insert_stop(sol, VEH_BIKE, 0, 0, 2, ACT_DELIVER, dist_matrix, customers)
    insert_stop(sol, VEH_BIKE, 1, 0, 3, ACT_DELIVER, dist_matrix, customers)

    return sol, customers, dist_matrix


# ===========================================================================
# TEST: move_stop within same route
# ===========================================================================

def test_move_stop_within_same_route_forward(sol_truck_with_2stops):
    """Move C1 from pos 1 to pos 0 (reorder in same route)."""
    sol, customers, dist_matrix = sol_truck_with_2stops

    assert sol[SOL_TRUCK_LENGTHS][0] == 2
    assert int(sol[SOL_TRUCK_STOPS][0, 0]) == 0
    assert int(sol[SOL_TRUCK_STOPS][0, 1]) == 1

    move_stop(sol, VEH_TRUCK, 0, 1, VEH_TRUCK, 0, 0, dist_matrix, customers)

    assert sol[SOL_TRUCK_LENGTHS][0] == 2
    assert int(sol[SOL_TRUCK_STOPS][0, 0]) == 1
    assert int(sol[SOL_TRUCK_STOPS][0, 1]) == 0
    assert sol[SOL_CUST_ROUTE_POS][1] == 0
    assert sol[SOL_CUST_ROUTE_POS][0] == 1


@pytest.mark.xfail(reason="Known bug: move_stop same-route backward position adjustment")
def test_move_stop_within_same_route_backward(sol_truck_with_2stops):
    """Move C0 from pos 0 to pos 1."""
    sol, customers, dist_matrix = sol_truck_with_2stops

    assert int(sol[SOL_TRUCK_STOPS][0, 0]) == 0
    assert int(sol[SOL_TRUCK_STOPS][0, 1]) == 1

    move_stop(sol, VEH_TRUCK, 0, 0, VEH_TRUCK, 0, 1, dist_matrix, customers)

    assert sol[SOL_TRUCK_LENGTHS][0] == 2
    assert int(sol[SOL_TRUCK_STOPS][0, 0]) == 1
    assert int(sol[SOL_TRUCK_STOPS][0, 1]) == 0


@pytest.mark.xfail(reason="Known bug: move_stop same-route position adjustment")
def test_move_stop_within_same_route_three_stops(tiny_dist_matrix):
    """Move middle stop in a 3-stop route: [A, B, C] -> [A, C, B]."""
    customers = np.array([
        [    0,     0, 10000, 0, 100000, 5000, 0],
        [10000,     0, 10000, 0, 100000, 5000, 0],
        [20000,     0, 10000, 0, 100000, 5000, 0],
    ], dtype=np.int64)

    depot = np.array([0, 0], dtype=np.int64)
    # Build dist_matrix manually (i64 meters)
    n = len(customers)
    coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
    diff = coords[:, None, :].astype(np.float64) - coords[None, :, :].astype(np.float64)
    dist_matrix = np.round(np.sqrt((diff ** 2).sum(axis=2))).astype(np.int64)

    sol = create_solution(1, 0, 3)
    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, dist_matrix, customers)
    insert_stop(sol, VEH_TRUCK, 0, 1, 1, ACT_DELIVER, dist_matrix, customers)
    insert_stop(sol, VEH_TRUCK, 0, 2, 2, ACT_DELIVER, dist_matrix, customers)

    assert sol[SOL_TRUCK_LENGTHS][0] == 3
    assert int(sol[SOL_TRUCK_STOPS][0, 0]) == 0
    assert int(sol[SOL_TRUCK_STOPS][0, 1]) == 1
    assert int(sol[SOL_TRUCK_STOPS][0, 2]) == 2

    move_stop(sol, VEH_TRUCK, 0, 1, VEH_TRUCK, 0, 2, dist_matrix, customers)

    assert sol[SOL_TRUCK_LENGTHS][0] == 3
    assert int(sol[SOL_TRUCK_STOPS][0, 0]) == 0
    assert int(sol[SOL_TRUCK_STOPS][0, 1]) == 2
    assert int(sol[SOL_TRUCK_STOPS][0, 2]) == 1
    assert sol[SOL_CUST_ROUTE_POS][0] == 0
    assert sol[SOL_CUST_ROUTE_POS][2] == 1
    assert sol[SOL_CUST_ROUTE_POS][1] == 2


# ===========================================================================
# TEST: swap_stops_between different vtypes (truck <-> bike)
# ===========================================================================

def test_swap_stops_truck_to_bike_different_routes(tiny_instance, tiny_dist_matrix):
    """Swap C0 (truck) with C2 (bike 0)."""
    sol = create_solution(
        tiny_instance["n_trucks"],
        tiny_instance["n_bikes"],
        tiny_instance["n_customers"],
    )
    customers = tiny_instance["customers"]
    dist_matrix = tiny_dist_matrix

    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, dist_matrix, customers)
    insert_stop(sol, VEH_TRUCK, 0, 1, 1, ACT_DELIVER, dist_matrix, customers)
    insert_stop(sol, VEH_BIKE, 0, 0, 2, ACT_DELIVER, dist_matrix, customers)
    insert_stop(sol, VEH_BIKE, 1, 0, 3, ACT_DELIVER, dist_matrix, customers)

    assert int(sol[SOL_TRUCK_STOPS][0, 0]) == 0
    assert int(sol[SOL_BIKE_STOPS][0, 0]) == 2
    assert sol[SOL_CUST_VEHICLE][0] == 0
    assert sol[SOL_CUST_VTYPE][0] == VEH_TRUCK
    assert sol[SOL_CUST_VEHICLE][2] == 1  # n_trucks(1) + bike_idx(0)
    assert sol[SOL_CUST_VTYPE][2] == VEH_BIKE

    swap_stops_between(sol, VEH_TRUCK, 0, 0, VEH_BIKE, 0, 0, dist_matrix, customers)

    assert int(sol[SOL_TRUCK_STOPS][0, 0]) == 2
    assert int(sol[SOL_TRUCK_STOPS][0, 1]) == 1
    assert int(sol[SOL_BIKE_STOPS][0, 0]) == 0

    assert sol[SOL_CUST_VEHICLE][2] == 0
    assert sol[SOL_CUST_VTYPE][2] == VEH_TRUCK
    assert sol[SOL_CUST_VEHICLE][0] == 1  # n_trucks(1) + bike_idx(0)
    assert sol[SOL_CUST_VTYPE][0] == VEH_BIKE


def test_swap_stops_bike_to_truck(tiny_instance, tiny_dist_matrix):
    """Swap C1 (truck) with C3 (bike 1)."""
    sol = create_solution(
        tiny_instance["n_trucks"],
        tiny_instance["n_bikes"],
        tiny_instance["n_customers"],
    )
    customers = tiny_instance["customers"]
    dist_matrix = tiny_dist_matrix

    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, dist_matrix, customers)
    insert_stop(sol, VEH_TRUCK, 0, 1, 1, ACT_DELIVER, dist_matrix, customers)
    insert_stop(sol, VEH_BIKE, 0, 0, 2, ACT_DELIVER, dist_matrix, customers)
    insert_stop(sol, VEH_BIKE, 1, 0, 3, ACT_DELIVER, dist_matrix, customers)

    swap_stops_between(sol, VEH_TRUCK, 0, 1, VEH_BIKE, 1, 0, dist_matrix, customers)

    assert int(sol[SOL_TRUCK_STOPS][0, 0]) == 0
    assert int(sol[SOL_TRUCK_STOPS][0, 1]) == 3
    assert int(sol[SOL_BIKE_STOPS][1, 0]) == 1

    assert sol[SOL_CUST_VTYPE][3] == VEH_TRUCK
    assert sol[SOL_CUST_VEHICLE][3] == 0
    assert sol[SOL_CUST_VTYPE][1] == VEH_BIKE
    assert sol[SOL_CUST_VEHICLE][1] == 2  # n_trucks(1) + bike1(1)


# ===========================================================================
# TEST: clear_route
# ===========================================================================

def test_clear_route_returns_all_pairs(sol_truck_with_2stops):
    """clear_route should return ndarray of (customer, action) pairs."""
    sol, customers, dist_matrix = sol_truck_with_2stops

    assert sol[SOL_TRUCK_LENGTHS][0] == 2
    assert sol[SOL_TRUCK_LOADS][0] == 120000  # C0: 100kg + C1: 20kg = 120000g

    removed = clear_route(sol, VEH_TRUCK, 0)

    assert removed.shape == (2, 2)
    # Check both customers present (order preserved)
    custs_removed = set(removed[:, 0].tolist())
    assert 0 in custs_removed
    assert 1 in custs_removed


def test_clear_route_clears_route_state(sol_truck_with_2stops):
    sol, customers, dist_matrix = sol_truck_with_2stops

    assert sol[SOL_TRUCK_LENGTHS][0] == 2
    assert sol[SOL_TRUCK_LOADS][0] == 120000
    assert sol[SOL_TRUCK_DISTANCES][0] > 0

    clear_route(sol, VEH_TRUCK, 0)

    assert sol[SOL_TRUCK_LENGTHS][0] == 0
    assert sol[SOL_TRUCK_LOADS][0] == 0
    assert sol[SOL_TRUCK_DISTANCES][0] == 0
    assert np.all(sol[SOL_TRUCK_STOPS][0, :] == -1)
    assert np.all(sol[SOL_TRUCK_ACTIONS][0, :] == ACT_PAD)


def test_clear_route_clears_customer_index(sol_truck_with_2stops):
    sol, customers, dist_matrix = sol_truck_with_2stops

    assert sol[SOL_CUST_VEHICLE][0] == 0
    assert sol[SOL_CUST_VTYPE][0] == VEH_TRUCK
    assert sol[SOL_CUST_ROUTE_POS][0] == 0
    assert sol[SOL_CUST_VEHICLE][1] == 0
    assert sol[SOL_CUST_VTYPE][1] == VEH_TRUCK
    assert sol[SOL_CUST_ROUTE_POS][1] == 1

    clear_route(sol, VEH_TRUCK, 0)

    assert sol[SOL_CUST_VEHICLE][0] == -1
    assert sol[SOL_CUST_VTYPE][0] == -1
    assert sol[SOL_CUST_ROUTE_POS][0] == -1
    assert sol[SOL_CUST_VEHICLE][1] == -1
    assert sol[SOL_CUST_VTYPE][1] == -1
    assert sol[SOL_CUST_ROUTE_POS][1] == -1


def test_clear_route_empty_route(sol_truck_with_2stops):
    """clear_route on empty route returns empty array."""
    sol, customers, dist_matrix = sol_truck_with_2stops

    removed = clear_route(sol, VEH_BIKE, 0)

    assert removed.shape == (0, 2)
    assert sol[SOL_BIKE_LENGTHS][0] == 0


def test_clear_route_with_reloads():
    customers = np.array([
        [    0,     0, 50000, 0, 100000, 5000, 0],
        [10000,     0, 50000, 0, 100000, 5000, 0],
        [20000,     0, 50000, 0, 100000, 5000, 0],
    ], dtype=np.int64)

    depot = np.array([0, 0], dtype=np.int64)
    coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
    diff = coords[:, None, :].astype(np.float64) - coords[None, :, :].astype(np.float64)
    dist_matrix = np.round(np.sqrt((diff ** 2).sum(axis=2))).astype(np.int64)

    sol = create_solution(1, 0, 3)

    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, dist_matrix, customers)
    # Manually insert RELOAD
    sol[SOL_TRUCK_STOPS][0, 1] = -1
    sol[SOL_TRUCK_ACTIONS][0, 1] = ACT_RELOAD
    sol[SOL_TRUCK_LENGTHS][0] = 2
    insert_stop(sol, VEH_TRUCK, 0, 2, 1, ACT_DELIVER, dist_matrix, customers)

    removed = clear_route(sol, VEH_TRUCK, 0)

    assert removed.shape[0] == 3
    custs_removed = removed[:, 0].tolist()
    acts_removed = removed[:, 1].tolist()
    assert 0 in custs_removed
    assert 1 in custs_removed
    assert ACT_RELOAD in acts_removed


# ===========================================================================
# TEST: copy_solution isolation
# ===========================================================================

def test_copy_solution_deep_copy_arrays(sol_truck_with_2stops):
    sol_orig, customers, dist_matrix = sol_truck_with_2stops

    sol_copy = copy_solution(sol_orig)

    assert sol_orig[SOL_TRUCK_STOPS][0, 0] == sol_copy[SOL_TRUCK_STOPS][0, 0]
    assert sol_orig[SOL_TRUCK_LENGTHS][0] == sol_copy[SOL_TRUCK_LENGTHS][0]

    sol_copy[SOL_TRUCK_STOPS][0, 0] = 999

    assert sol_orig[SOL_TRUCK_STOPS][0, 0] == 0
    assert sol_copy[SOL_TRUCK_STOPS][0, 0] == 999


def test_copy_solution_isolated_mutation(sol_truck_with_2stops):
    sol_orig, customers, dist_matrix = sol_truck_with_2stops

    sol_copy = copy_solution(sol_orig)

    move_stop(sol_copy, VEH_TRUCK, 0, 1, VEH_TRUCK, 0, 0, dist_matrix, customers)

    assert sol_orig[SOL_TRUCK_STOPS][0, 0] == 0
    assert sol_orig[SOL_TRUCK_STOPS][0, 1] == 1
    assert sol_orig[SOL_CUST_ROUTE_POS][0] == 0
    assert sol_orig[SOL_CUST_ROUTE_POS][1] == 1

    assert sol_copy[SOL_TRUCK_STOPS][0, 0] == 1
    assert sol_copy[SOL_TRUCK_STOPS][0, 1] == 0


def test_copy_solution_clear_isolation(sol_truck_with_2stops):
    sol_orig, customers, dist_matrix = sol_truck_with_2stops

    sol_copy = copy_solution(sol_orig)
    clear_route(sol_copy, VEH_TRUCK, 0)

    assert sol_orig[SOL_TRUCK_LENGTHS][0] == 2
    assert sol_orig[SOL_TRUCK_LOADS][0] == 120000  # 120kg in grams
    assert int(sol_orig[SOL_TRUCK_STOPS][0, 0]) == 0

    assert sol_copy[SOL_TRUCK_LENGTHS][0] == 0
    assert sol_copy[SOL_TRUCK_LOADS][0] == 0


def test_copy_solution_scalar_preserved(sol_truck_with_2stops):
    sol_orig, _, _ = sol_truck_with_2stops
    sol_copy = copy_solution(sol_orig)

    assert sol_copy[SOL_META][META_N_TRUCKS] == sol_orig[SOL_META][META_N_TRUCKS]
    assert sol_copy[SOL_META][META_N_BIKES] == sol_orig[SOL_META][META_N_BIKES]
    assert sol_copy[SOL_META][META_MAX_ROUTE_LEN] == sol_orig[SOL_META][META_MAX_ROUTE_LEN]
