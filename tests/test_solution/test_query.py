"""Tests for O(1) lookups and solution queries — Data Model v2."""
import numpy as np
import pytest

from src.solution.structure import create_solution
from src.solution.route_ops import insert_stop
from src.solution.query import (
    get_unassigned_customers,
    get_assigned_customers,
    get_route_as_list,
    get_route_customers_only,
    is_customer_assigned,
    get_customer_info,
)
from src.data.constants import (
    ACT_DELIVER, ACT_PICKUP, ACT_PAD,
    ORD_LOC,
)


def _make_sol(data):
    return create_solution(data["n_vehicles"], data["n_customers"],
                           n_sku=data["n_sku"])


def _ins(sol, v, pos, customer, data, action=ACT_DELIVER):
    orders = data["orders"]
    loc_id = int(orders[customer, ORD_LOC])
    insert_stop(sol, v, pos, loc_id, action, data["dist_matrix"],
                data["vehicles"], orders,
                customer=customer if action == ACT_DELIVER else -1)


# ---------------------------------------------------------------------------
# get_unassigned_customers
# ---------------------------------------------------------------------------

def test_empty_all_unassigned(tiny_data):
    sol = _make_sol(tiny_data)
    unassigned = get_unassigned_customers(sol, 5)
    np.testing.assert_array_equal(np.sort(unassigned), np.array([0, 1, 2, 3, 4]))


def test_some_assigned(tiny_data):
    sol = _make_sol(tiny_data)
    _ins(sol, 0, 0, 0, tiny_data)
    _ins(sol, 1, 0, 3, tiny_data)
    unassigned = get_unassigned_customers(sol, 5)
    np.testing.assert_array_equal(np.sort(unassigned), np.array([1, 2, 4]))


def test_all_assigned_empty(tiny_data):
    sol = _make_sol(tiny_data)
    _ins(sol, 0, 0, 0, tiny_data)
    _ins(sol, 1, 0, 1, tiny_data)
    _ins(sol, 1, 1, 2, tiny_data)
    _ins(sol, 2, 0, 3, tiny_data)
    _ins(sol, 2, 1, 4, tiny_data)
    unassigned = get_unassigned_customers(sol, 5)
    assert len(unassigned) == 0


# ---------------------------------------------------------------------------
# get_assigned_customers
# ---------------------------------------------------------------------------

def test_empty_none_assigned(tiny_data):
    sol = _make_sol(tiny_data)
    assigned = get_assigned_customers(sol, 5)
    assert len(assigned) == 0


def test_assigned_some(tiny_data):
    sol = _make_sol(tiny_data)
    _ins(sol, 0, 0, 0, tiny_data)
    _ins(sol, 1, 0, 3, tiny_data)
    assigned = get_assigned_customers(sol, 5)
    np.testing.assert_array_equal(np.sort(assigned), np.array([0, 3]))


# ---------------------------------------------------------------------------
# get_route_as_list
# ---------------------------------------------------------------------------

def test_route_as_list(tiny_data):
    sol = _make_sol(tiny_data)
    _ins(sol, 1, 0, 3, tiny_data)
    _ins(sol, 1, 1, 4, tiny_data)
    route_list = get_route_as_list(sol, 1)
    assert len(route_list) == 2


def test_route_as_list_empty(tiny_data):
    sol = _make_sol(tiny_data)
    route_list = get_route_as_list(sol, 0)
    assert route_list == []


# ---------------------------------------------------------------------------
# get_route_customers_only
# ---------------------------------------------------------------------------

def test_route_customers_mixed(tiny_data):
    """Route with DELIVER and PICKUP: only DELIVER customers returned."""
    sol = _make_sol(tiny_data)
    orders = tiny_data["orders"]
    _ins(sol, 1, 0, 3, tiny_data, ACT_DELIVER)
    # Insert a pickup at same location
    loc_id = int(orders[3, ORD_LOC])
    insert_stop(sol, 1, 1, loc_id, ACT_PICKUP, tiny_data["dist_matrix"],
                tiny_data["vehicles"], orders, customer=-1)
    _ins(sol, 1, 2, 4, tiny_data, ACT_DELIVER)

    result = get_route_customers_only(sol, 1, orders)
    assert len(result) >= 2


# ---------------------------------------------------------------------------
# is_customer_assigned
# ---------------------------------------------------------------------------

def test_assigned_true(tiny_data):
    sol = _make_sol(tiny_data)
    _ins(sol, 0, 0, 0, tiny_data)
    assert is_customer_assigned(sol, 0) is True


def test_unassigned_false(tiny_data):
    sol = _make_sol(tiny_data)
    assert is_customer_assigned(sol, 0) is False


def test_pickup_not_assigned(tiny_data):
    """PICKUP does not count as assignment."""
    sol = _make_sol(tiny_data)
    orders = tiny_data["orders"]
    loc_id = int(orders[3, ORD_LOC])
    insert_stop(sol, 1, 0, loc_id, ACT_PICKUP, tiny_data["dist_matrix"],
                tiny_data["vehicles"], orders, customer=-1)
    assert is_customer_assigned(sol, 3) is False


# ---------------------------------------------------------------------------
# get_customer_info
# ---------------------------------------------------------------------------

def test_customer_info_assigned(tiny_data):
    sol = _make_sol(tiny_data)
    _ins(sol, 0, 0, 0, tiny_data)
    vid, pos = get_customer_info(sol, 0)
    assert vid == 0
    assert pos == 0


def test_customer_info_unassigned(tiny_data):
    sol = _make_sol(tiny_data)
    vid, pos = get_customer_info(sol, 2)
    assert vid == -1
    assert pos == -1


def test_customer_info_after_insert(tiny_data):
    sol = _make_sol(tiny_data)
    _ins(sol, 0, 0, 0, tiny_data)
    _ins(sol, 0, 1, 1, tiny_data)
    vid, pos = get_customer_info(sol, 1)
    assert vid == 0
    assert pos == 1
