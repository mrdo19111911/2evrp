"""Tests for O(1) lookups and solution queries -- tuple-based solution. All i64."""
import numpy as np
import pytest

from src.solution.structure import create_solution
from src.solution.route_ops import insert_stop
from src.solution.query import (
    get_unassigned_customers,
    get_assigned_customers,
    get_route_customers_only,
    is_customer_assigned,
    get_customer_info,
)
from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, ACT_PAD, VEH_TRUCK, VEH_BIKE,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_TRUCK_LENGTHS,
    SOL_BIKE_STOPS, SOL_BIKE_ACTIONS, SOL_BIKE_LENGTHS,
    SOL_CUST_VEHICLE,
)


# ---------------------------------------------------------------------------
# get_unassigned_customers
# ---------------------------------------------------------------------------

def test_empty_all_unassigned(tiny_instance, tiny_dist_matrix):
    sol = create_solution(1, 2, 5)
    unassigned = get_unassigned_customers(sol, 5)
    np.testing.assert_array_equal(np.sort(unassigned), np.array([0, 1, 2, 3, 4]))


def test_some_assigned(tiny_instance, tiny_dist_matrix):
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix
    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 0, 3, ACT_DELIVER, dm, custs)
    unassigned = get_unassigned_customers(sol, 5)
    np.testing.assert_array_equal(np.sort(unassigned), np.array([1, 2, 4]))


def test_all_assigned_empty(tiny_instance, tiny_dist_matrix):
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix
    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 0, 1, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 1, 2, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 1, 0, 3, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 1, 1, 4, ACT_DELIVER, dm, custs)
    unassigned = get_unassigned_customers(sol, 5)
    assert len(unassigned) == 0


# ---------------------------------------------------------------------------
# get_assigned_customers
# ---------------------------------------------------------------------------

def test_empty_none_assigned(tiny_instance, tiny_dist_matrix):
    sol = create_solution(1, 2, 5)
    assigned = get_assigned_customers(sol, 5)
    assert len(assigned) == 0


def test_assigned_some(tiny_instance, tiny_dist_matrix):
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix
    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 0, 3, ACT_DELIVER, dm, custs)
    assigned = get_assigned_customers(sol, 5)
    np.testing.assert_array_equal(np.sort(assigned), np.array([0, 3]))


# ---------------------------------------------------------------------------
# get_route_customers_only
# ---------------------------------------------------------------------------

def test_route_customers_mixed(tiny_instance, tiny_dist_matrix):
    """Route with DELIVER and RELOAD: only DELIVER customers returned."""
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix

    insert_stop(sol, VEH_BIKE, 0, 0, 3, ACT_DELIVER, dm, custs)
    # Insert a reload at same location
    insert_stop(sol, VEH_BIKE, 0, 1, 3, ACT_RELOAD, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 2, 4, ACT_DELIVER, dm, custs)

    result = get_route_customers_only(sol, VEH_BIKE, 0)
    assert len(result) == 2
    assert 3 in result
    assert 4 in result


# ---------------------------------------------------------------------------
# is_customer_assigned
# ---------------------------------------------------------------------------

def test_assigned_true(tiny_instance, tiny_dist_matrix):
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix
    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, dm, custs)
    assert is_customer_assigned(sol, 0) is True


def test_unassigned_false(tiny_instance, tiny_dist_matrix):
    sol = create_solution(1, 2, 5)
    assert is_customer_assigned(sol, 0) is False


def test_reload_not_assigned(tiny_instance, tiny_dist_matrix):
    """RELOAD does not count as assignment."""
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix
    insert_stop(sol, VEH_BIKE, 0, 0, 3, ACT_RELOAD, dm, custs)
    assert is_customer_assigned(sol, 3) is False


# ---------------------------------------------------------------------------
# get_customer_info
# ---------------------------------------------------------------------------

def test_customer_info_assigned(tiny_instance, tiny_dist_matrix):
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix
    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, dm, custs)
    vtype, vid, pos = get_customer_info(sol, 0)
    assert vtype == VEH_TRUCK
    assert vid == 0
    assert pos == 0


def test_customer_info_unassigned(tiny_instance, tiny_dist_matrix):
    sol = create_solution(1, 2, 5)
    vtype, vid, pos = get_customer_info(sol, 2)
    assert vtype == -1
    assert vid == -1
    assert pos == -1


def test_customer_info_after_insert(tiny_instance, tiny_dist_matrix):
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix
    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_TRUCK, 0, 1, 1, ACT_DELIVER, dm, custs)
    vtype, vid, pos = get_customer_info(sol, 1)
    assert vtype == VEH_TRUCK
    assert vid == 0
    assert pos == 1
