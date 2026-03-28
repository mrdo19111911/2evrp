"""Tests for solution structure: create, copy, rebuild_index -- tuple-based solution."""
import numpy as np
import pytest

from src.solution.structure import create_solution, copy_solution, rebuild_index
from src.data.constants import (
    ACT_DELIVER, ACT_PAD, VEH_TRUCK, VEH_BIKE,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_BIKE_STOPS, SOL_BIKE_ACTIONS,
    SOL_TRUCK_LENGTHS, SOL_BIKE_LENGTHS, SOL_TRUCK_LOADS, SOL_BIKE_LOADS,
    SOL_TRUCK_DISTANCES, SOL_BIKE_DISTANCES,
    SOL_CUST_VEHICLE, SOL_CUST_VTYPE, SOL_CUST_ROUTE_POS,
    SOL_SATELLITES, SOL_META,
    META_N_TRUCKS, META_N_BIKES, META_MAX_ROUTE_LEN, META_N_SATELLITES,
    MAX_SATELLITES,
)


# =========================================================================
# TEST: create_solution with 0 trucks 0 bikes
# =========================================================================

def test_create_solution_zero_trucks_zero_bikes():
    sol = create_solution(0, 0, 10, max_route_len=100)

    assert sol[SOL_TRUCK_STOPS].shape == (0, 100)
    assert sol[SOL_TRUCK_ACTIONS].shape == (0, 100)
    assert sol[SOL_BIKE_STOPS].shape == (0, 100)
    assert sol[SOL_BIKE_ACTIONS].shape == (0, 100)

    assert sol[SOL_TRUCK_LENGTHS].shape == (0,)
    assert sol[SOL_TRUCK_LOADS].shape == (0,)
    assert sol[SOL_BIKE_LENGTHS].shape == (0,)
    assert sol[SOL_BIKE_LOADS].shape == (0,)

    assert sol[SOL_CUST_VEHICLE].shape == (10,)
    assert sol[SOL_CUST_VTYPE].shape == (10,)
    assert sol[SOL_CUST_ROUTE_POS].shape == (10,)
    assert np.all(sol[SOL_CUST_VEHICLE] == -1)
    assert np.all(sol[SOL_CUST_VTYPE] == -1)
    assert np.all(sol[SOL_CUST_ROUTE_POS] == -1)

    assert sol[SOL_META][META_N_TRUCKS] == 0
    assert sol[SOL_META][META_N_BIKES] == 0


def test_create_solution_zero_trucks_zero_bikes_zero_customers():
    sol = create_solution(0, 0, 0, max_route_len=100)

    assert sol[SOL_CUST_VEHICLE].shape == (0,)
    assert sol[SOL_CUST_VTYPE].shape == (0,)
    assert sol[SOL_CUST_ROUTE_POS].shape == (0,)


# =========================================================================
# TEST: create_solution normal cases
# =========================================================================

def test_create_solution_basic_shapes():
    sol = create_solution(3, 2, 10, max_route_len=100)

    assert sol[SOL_TRUCK_STOPS].shape == (3, 100)
    assert sol[SOL_TRUCK_ACTIONS].shape == (3, 100)
    assert sol[SOL_BIKE_STOPS].shape == (2, 100)
    assert sol[SOL_BIKE_ACTIONS].shape == (2, 100)

    assert sol[SOL_TRUCK_LENGTHS].shape == (3,)
    assert sol[SOL_TRUCK_LOADS].shape == (3,)
    assert sol[SOL_BIKE_LENGTHS].shape == (2,)
    assert sol[SOL_BIKE_LOADS].shape == (2,)

    assert sol[SOL_CUST_VEHICLE].shape == (10,)
    assert sol[SOL_CUST_VTYPE].shape == (10,)
    assert sol[SOL_CUST_ROUTE_POS].shape == (10,)

    assert sol[SOL_TRUCK_DISTANCES].shape == (3,)
    assert sol[SOL_BIKE_DISTANCES].shape == (2,)

    # Satellites pre-allocated
    assert sol[SOL_SATELLITES].shape == (MAX_SATELLITES, 5)
    assert sol[SOL_META][META_N_SATELLITES] == 0


def test_create_solution_initial_values():
    sol = create_solution(2, 1, 5)

    assert np.all(sol[SOL_TRUCK_STOPS] == -1)
    assert np.all(sol[SOL_TRUCK_ACTIONS] == ACT_PAD)
    assert np.all(sol[SOL_BIKE_STOPS] == -1)
    assert np.all(sol[SOL_BIKE_ACTIONS] == ACT_PAD)

    assert np.all(sol[SOL_TRUCK_LENGTHS] == 0)
    assert np.all(sol[SOL_TRUCK_LOADS] == 0)
    assert np.all(sol[SOL_TRUCK_DISTANCES] == 0)

    assert np.all(sol[SOL_BIKE_LENGTHS] == 0)
    assert np.all(sol[SOL_BIKE_LOADS] == 0)
    assert np.all(sol[SOL_BIKE_DISTANCES] == 0)

    assert np.all(sol[SOL_CUST_VEHICLE] == -1)
    assert np.all(sol[SOL_CUST_VTYPE] == -1)
    assert np.all(sol[SOL_CUST_ROUTE_POS] == -1)


def test_create_solution_dtypes():
    sol = create_solution(2, 1, 5)

    assert sol[SOL_TRUCK_STOPS].dtype == np.int32
    assert sol[SOL_TRUCK_ACTIONS].dtype == np.int8
    assert sol[SOL_TRUCK_LENGTHS].dtype == np.int32
    assert sol[SOL_TRUCK_LOADS].dtype == np.int64
    assert sol[SOL_TRUCK_DISTANCES].dtype == np.int64

    assert sol[SOL_BIKE_STOPS].dtype == np.int32
    assert sol[SOL_BIKE_ACTIONS].dtype == np.int8
    assert sol[SOL_BIKE_LENGTHS].dtype == np.int32
    assert sol[SOL_BIKE_LOADS].dtype == np.int64
    assert sol[SOL_BIKE_DISTANCES].dtype == np.int64

    assert sol[SOL_CUST_VEHICLE].dtype == np.int32
    assert sol[SOL_CUST_VTYPE].dtype == np.int8
    assert sol[SOL_CUST_ROUTE_POS].dtype == np.int32


def test_create_solution_metadata():
    sol = create_solution(3, 2, 10, max_route_len=50)

    assert sol[SOL_META][META_N_TRUCKS] == 3
    assert sol[SOL_META][META_N_BIKES] == 2
    assert sol[SOL_META][META_MAX_ROUTE_LEN] == 50


# =========================================================================
# TEST: copy_solution deep copy
# =========================================================================

def test_copy_solution_modifying_copy_does_not_affect_original():
    sol = create_solution(2, 1, 5)
    sol[SOL_TRUCK_STOPS][0, 0] = 42
    sol[SOL_TRUCK_LENGTHS][0] = 3
    sol[SOL_CUST_VEHICLE][0] = 1

    cp = copy_solution(sol)
    cp[SOL_TRUCK_STOPS][0, 0] = 99
    cp[SOL_TRUCK_LENGTHS][0] = 7
    cp[SOL_CUST_VEHICLE][0] = -1

    assert sol[SOL_TRUCK_STOPS][0, 0] == 42, "Original truck_stops modified"
    assert sol[SOL_TRUCK_LENGTHS][0] == 3, "Original truck_lengths modified"
    assert sol[SOL_CUST_VEHICLE][0] == 1, "Original cust_vehicle modified"


def test_copy_solution_modifying_bike_arrays():
    sol = create_solution(1, 2, 5)
    sol[SOL_BIKE_STOPS][0, 0] = 10
    sol[SOL_BIKE_ACTIONS][0, 0] = ACT_DELIVER
    sol[SOL_BIKE_LENGTHS][0] = 1

    cp = copy_solution(sol)
    cp[SOL_BIKE_STOPS][0, 0] = 20
    cp[SOL_BIKE_ACTIONS][0, 0] = ACT_PAD
    cp[SOL_BIKE_LENGTHS][0] = 0

    assert sol[SOL_BIKE_STOPS][0, 0] == 10, "Original bike_stops modified"
    assert sol[SOL_BIKE_ACTIONS][0, 0] == ACT_DELIVER, "Original bike_actions modified"
    assert sol[SOL_BIKE_LENGTHS][0] == 1, "Original bike_lengths modified"


def test_copy_solution_preserves_values():
    sol = create_solution(2, 1, 5)
    sol[SOL_TRUCK_STOPS][1, 0] = 3
    sol[SOL_TRUCK_ACTIONS][1, 0] = ACT_DELIVER
    sol[SOL_TRUCK_LENGTHS][1] = 1
    sol[SOL_TRUCK_LOADS][1] = 50000       # 50kg in grams
    sol[SOL_TRUCK_DISTANCES][1] = 12300   # 12300 meters
    sol[SOL_CUST_VEHICLE][2] = 1
    sol[SOL_CUST_VTYPE][2] = VEH_TRUCK
    sol[SOL_CUST_ROUTE_POS][2] = 0

    cp = copy_solution(sol)

    assert cp[SOL_TRUCK_STOPS][1, 0] == 3
    assert cp[SOL_TRUCK_ACTIONS][1, 0] == ACT_DELIVER
    assert cp[SOL_TRUCK_LENGTHS][1] == 1
    assert cp[SOL_TRUCK_LOADS][1] == 50000
    assert cp[SOL_TRUCK_DISTANCES][1] == 12300
    assert cp[SOL_CUST_VEHICLE][2] == 1
    assert cp[SOL_CUST_VTYPE][2] == VEH_TRUCK
    assert cp[SOL_CUST_ROUTE_POS][2] == 0


def test_copy_solution_no_shared_memory():
    sol = create_solution(2, 1, 5)
    cp = copy_solution(sol)

    assert not np.shares_memory(sol[SOL_TRUCK_STOPS], cp[SOL_TRUCK_STOPS])
    assert not np.shares_memory(sol[SOL_TRUCK_ACTIONS], cp[SOL_TRUCK_ACTIONS])
    assert not np.shares_memory(sol[SOL_TRUCK_LENGTHS], cp[SOL_TRUCK_LENGTHS])
    assert not np.shares_memory(sol[SOL_TRUCK_LOADS], cp[SOL_TRUCK_LOADS])
    assert not np.shares_memory(sol[SOL_TRUCK_DISTANCES], cp[SOL_TRUCK_DISTANCES])

    assert not np.shares_memory(sol[SOL_BIKE_STOPS], cp[SOL_BIKE_STOPS])
    assert not np.shares_memory(sol[SOL_BIKE_ACTIONS], cp[SOL_BIKE_ACTIONS])
    assert not np.shares_memory(sol[SOL_BIKE_LENGTHS], cp[SOL_BIKE_LENGTHS])
    assert not np.shares_memory(sol[SOL_BIKE_LOADS], cp[SOL_BIKE_LOADS])
    assert not np.shares_memory(sol[SOL_BIKE_DISTANCES], cp[SOL_BIKE_DISTANCES])

    assert not np.shares_memory(sol[SOL_CUST_VEHICLE], cp[SOL_CUST_VEHICLE])
    assert not np.shares_memory(sol[SOL_CUST_VTYPE], cp[SOL_CUST_VTYPE])
    assert not np.shares_memory(sol[SOL_CUST_ROUTE_POS], cp[SOL_CUST_ROUTE_POS])

    assert not np.shares_memory(sol[SOL_SATELLITES], cp[SOL_SATELLITES])


def test_copy_solution_scalar_values_preserved():
    sol = create_solution(3, 2, 10, max_route_len=50)
    cp = copy_solution(sol)

    assert cp[SOL_META][META_N_TRUCKS] == 3
    assert cp[SOL_META][META_N_BIKES] == 2
    assert cp[SOL_META][META_MAX_ROUTE_LEN] == 50


# =========================================================================
# TEST: rebuild_index
# =========================================================================

def test_rebuild_index_empty_solution():
    sol = create_solution(2, 1, 5)
    rebuild_index(sol)

    assert np.all(sol[SOL_CUST_VEHICLE] == -1)
    assert np.all(sol[SOL_CUST_VTYPE] == -1)
    assert np.all(sol[SOL_CUST_ROUTE_POS] == -1)


def test_rebuild_index_single_truck_route():
    sol = create_solution(2, 1, 5)
    sol[SOL_TRUCK_STOPS][0, 0] = 0
    sol[SOL_TRUCK_ACTIONS][0, 0] = ACT_DELIVER
    sol[SOL_TRUCK_LENGTHS][0] = 1

    rebuild_index(sol)

    assert sol[SOL_CUST_VEHICLE][0] == 0
    assert sol[SOL_CUST_VTYPE][0] == VEH_TRUCK
    assert sol[SOL_CUST_ROUTE_POS][0] == 0


def test_rebuild_index_single_bike_route():
    sol = create_solution(2, 1, 5)
    sol[SOL_BIKE_STOPS][0, 0] = 1
    sol[SOL_BIKE_ACTIONS][0, 0] = ACT_DELIVER
    sol[SOL_BIKE_LENGTHS][0] = 1

    rebuild_index(sol)

    assert sol[SOL_CUST_VEHICLE][1] == 2  # n_trucks + bike_idx = 2 + 0
    assert sol[SOL_CUST_VTYPE][1] == VEH_BIKE
    assert sol[SOL_CUST_ROUTE_POS][1] == 0


def test_rebuild_index_customer_in_both_truck_and_bike():
    """Customer assigned to BOTH truck and bike routes. Last route (bike) wins."""
    sol = create_solution(2, 1, 5)

    sol[SOL_TRUCK_STOPS][0, 0] = 0
    sol[SOL_TRUCK_ACTIONS][0, 0] = ACT_DELIVER
    sol[SOL_TRUCK_LENGTHS][0] = 1

    sol[SOL_BIKE_STOPS][0, 0] = 0
    sol[SOL_BIKE_ACTIONS][0, 0] = ACT_DELIVER
    sol[SOL_BIKE_LENGTHS][0] = 1

    rebuild_index(sol)

    assert sol[SOL_CUST_VEHICLE][0] == 2, \
        "Customer 0 should be in bike route (n_trucks + 0)"
    assert sol[SOL_CUST_VTYPE][0] == VEH_BIKE
    assert sol[SOL_CUST_ROUTE_POS][0] == 0


def test_rebuild_index_multiple_customers_mixed_routes():
    sol = create_solution(2, 2, 10)

    # Truck 0: customers 0, 1
    sol[SOL_TRUCK_STOPS][0, 0] = 0
    sol[SOL_TRUCK_STOPS][0, 1] = 1
    sol[SOL_TRUCK_ACTIONS][0, 0] = ACT_DELIVER
    sol[SOL_TRUCK_ACTIONS][0, 1] = ACT_DELIVER
    sol[SOL_TRUCK_LENGTHS][0] = 2

    # Truck 1: customer 2
    sol[SOL_TRUCK_STOPS][1, 0] = 2
    sol[SOL_TRUCK_ACTIONS][1, 0] = ACT_DELIVER
    sol[SOL_TRUCK_LENGTHS][1] = 1

    # Bike 0: customers 3, 4
    sol[SOL_BIKE_STOPS][0, 0] = 3
    sol[SOL_BIKE_STOPS][0, 1] = 4
    sol[SOL_BIKE_ACTIONS][0, 0] = ACT_DELIVER
    sol[SOL_BIKE_ACTIONS][0, 1] = ACT_DELIVER
    sol[SOL_BIKE_LENGTHS][0] = 2

    # Bike 1: customer 5
    sol[SOL_BIKE_STOPS][1, 0] = 5
    sol[SOL_BIKE_ACTIONS][1, 0] = ACT_DELIVER
    sol[SOL_BIKE_LENGTHS][1] = 1

    rebuild_index(sol)

    # Truck assignments
    assert sol[SOL_CUST_VEHICLE][0] == 0 and sol[SOL_CUST_VTYPE][0] == VEH_TRUCK
    assert sol[SOL_CUST_VEHICLE][1] == 0 and sol[SOL_CUST_VTYPE][1] == VEH_TRUCK
    assert sol[SOL_CUST_VEHICLE][2] == 1 and sol[SOL_CUST_VTYPE][2] == VEH_TRUCK

    # Bike assignments
    assert sol[SOL_CUST_VEHICLE][3] == 2 and sol[SOL_CUST_VTYPE][3] == VEH_BIKE
    assert sol[SOL_CUST_VEHICLE][4] == 2 and sol[SOL_CUST_VTYPE][4] == VEH_BIKE
    assert sol[SOL_CUST_VEHICLE][5] == 3 and sol[SOL_CUST_VTYPE][5] == VEH_BIKE

    # Unassigned
    for c in range(6, 10):
        assert sol[SOL_CUST_VEHICLE][c] == -1


def test_rebuild_index_overwrites_stale_data():
    sol = create_solution(2, 1, 5)

    sol[SOL_CUST_VEHICLE][2] = 99
    sol[SOL_CUST_VTYPE][2] = VEH_TRUCK
    sol[SOL_CUST_ROUTE_POS][2] = 5

    rebuild_index(sol)

    assert sol[SOL_CUST_VEHICLE][2] == -1
    assert sol[SOL_CUST_VTYPE][2] == -1
    assert sol[SOL_CUST_ROUTE_POS][2] == -1


# =========================================================================
# TEST: is_customer_assigned boundary cases
# =========================================================================

def test_is_customer_assigned_unassigned():
    sol = create_solution(2, 1, 5)
    assert sol[SOL_CUST_VEHICLE][0] == -1
    assert sol[SOL_CUST_VTYPE][0] == -1


def test_is_customer_assigned_in_truck():
    sol = create_solution(2, 1, 5)
    sol[SOL_TRUCK_STOPS][0, 0] = 0
    sol[SOL_TRUCK_ACTIONS][0, 0] = ACT_DELIVER
    sol[SOL_TRUCK_LENGTHS][0] = 1
    rebuild_index(sol)

    assert sol[SOL_CUST_VEHICLE][0] != -1
    assert sol[SOL_CUST_VTYPE][0] == VEH_TRUCK


def test_is_customer_assigned_in_bike():
    sol = create_solution(2, 1, 5)
    sol[SOL_BIKE_STOPS][0, 0] = 1
    sol[SOL_BIKE_ACTIONS][0, 0] = ACT_DELIVER
    sol[SOL_BIKE_LENGTHS][0] = 1
    rebuild_index(sol)

    assert sol[SOL_CUST_VEHICLE][1] != -1
    assert sol[SOL_CUST_VTYPE][1] == VEH_BIKE


def test_is_customer_assigned_boundary_customer_id():
    sol = create_solution(1, 1, 3)

    sol[SOL_TRUCK_STOPS][0, 0] = 0
    sol[SOL_TRUCK_ACTIONS][0, 0] = ACT_DELIVER
    sol[SOL_TRUCK_LENGTHS][0] = 1

    sol[SOL_BIKE_STOPS][0, 0] = 2
    sol[SOL_BIKE_ACTIONS][0, 0] = ACT_DELIVER
    sol[SOL_BIKE_LENGTHS][0] = 1

    rebuild_index(sol)

    assert sol[SOL_CUST_VEHICLE][0] == 0
    assert sol[SOL_CUST_VEHICLE][2] == 1  # n_trucks(1) + bike_idx(0)
    assert sol[SOL_CUST_VEHICLE][1] == -1


# =========================================================================
# ADVANCED EDGE CASES
# =========================================================================

def test_rebuild_index_with_non_deliver_actions():
    sol = create_solution(1, 1, 5)

    sol[SOL_TRUCK_STOPS][0, 0] = 0
    sol[SOL_TRUCK_ACTIONS][0, 0] = ACT_PAD
    sol[SOL_TRUCK_LENGTHS][0] = 1

    rebuild_index(sol)

    assert sol[SOL_CUST_VEHICLE][0] == -1, "ACT_PAD should not assign customer"


def test_rebuild_index_gaps_in_route():
    sol = create_solution(1, 0, 5)

    sol[SOL_TRUCK_STOPS][0, 0] = 0
    sol[SOL_TRUCK_ACTIONS][0, 0] = ACT_DELIVER
    sol[SOL_TRUCK_STOPS][0, 1] = 999  # Garbage beyond length
    sol[SOL_TRUCK_ACTIONS][0, 1] = ACT_DELIVER
    sol[SOL_TRUCK_LENGTHS][0] = 1  # Only 1 valid stop

    rebuild_index(sol)

    assert sol[SOL_CUST_VEHICLE][0] == 0


def test_rebuild_index_bike_vehicle_id_calculation():
    sol = create_solution(5, 3, 20)

    sol[SOL_BIKE_STOPS][0, 0] = 0
    sol[SOL_BIKE_ACTIONS][0, 0] = ACT_DELIVER
    sol[SOL_BIKE_LENGTHS][0] = 1

    sol[SOL_BIKE_STOPS][2, 0] = 1
    sol[SOL_BIKE_ACTIONS][2, 0] = ACT_DELIVER
    sol[SOL_BIKE_LENGTHS][2] = 1

    rebuild_index(sol)

    assert sol[SOL_CUST_VEHICLE][0] == 5, "Bike 0 should be vehicle 5 (n_trucks + 0)"
    assert sol[SOL_CUST_VEHICLE][1] == 7, "Bike 2 should be vehicle 7 (n_trucks + 2)"


def test_copy_solution_with_satellites():
    sol = create_solution(1, 1, 5)

    # Add a satellite via the pre-allocated array (i64 values)
    sol[SOL_SATELLITES][0] = [0, 0, 1, 10000, 5200]
    sol[SOL_META][META_N_SATELLITES] = 1

    cp = copy_solution(sol)

    # Modify copy
    cp[SOL_SATELLITES][0] = [1, 1, 1, 20000, 10200]

    assert sol[SOL_SATELLITES][0, 0] == 0, "Original satellite modified"
    assert cp[SOL_SATELLITES][0, 0] == 1, "Copy not independent"


def test_copy_solution_independent_scalar_metadata():
    sol = create_solution(2, 1, 5, max_route_len=100)
    cp = copy_solution(sol)

    # Modify copy metadata
    cp[SOL_META][META_N_TRUCKS] = 99

    assert sol[SOL_META][META_N_TRUCKS] == 2
    assert cp[SOL_META][META_N_TRUCKS] == 99


def test_rebuild_index_large_customer_id():
    sol = create_solution(1, 0, 10)

    sol[SOL_TRUCK_STOPS][0, 0] = 999
    sol[SOL_TRUCK_ACTIONS][0, 0] = ACT_DELIVER
    sol[SOL_TRUCK_LENGTHS][0] = 1

    with pytest.raises(IndexError):
        rebuild_index(sol)


def test_copy_solution_empty_solution():
    sol = create_solution(0, 0, 0)
    cp = copy_solution(sol)

    assert cp[SOL_META][META_N_TRUCKS] == 0
    assert cp[SOL_META][META_N_BIKES] == 0
    assert len(cp) == len(sol)


def test_rebuild_index_sequence_positions():
    sol = create_solution(1, 0, 5)

    sol[SOL_TRUCK_STOPS][0, 0] = 0
    sol[SOL_TRUCK_STOPS][0, 1] = 2
    sol[SOL_TRUCK_STOPS][0, 2] = 1
    sol[SOL_TRUCK_ACTIONS][0, 0] = ACT_DELIVER
    sol[SOL_TRUCK_ACTIONS][0, 1] = ACT_DELIVER
    sol[SOL_TRUCK_ACTIONS][0, 2] = ACT_DELIVER
    sol[SOL_TRUCK_LENGTHS][0] = 3

    rebuild_index(sol)

    assert sol[SOL_CUST_ROUTE_POS][0] == 0
    assert sol[SOL_CUST_ROUTE_POS][2] == 1
    assert sol[SOL_CUST_ROUTE_POS][1] == 2
