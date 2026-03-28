"""Tests for O(1) constraint checks — Data Model v2."""
import numpy as np
import pytest

from src.solution.structure import create_solution
from src.solution.route_ops import insert_stop
from src.solution.check import (
    can_insert_customer,
    can_swap_customers,
    check_route_capacity_quick,
    check_all_assigned,
)
from src.data.constants import (
    ACT_DELIVER, VTYPE_TRUCK, VTYPE_BIKE,
    ORD_LOC, ORD_QTY, ORD_UNIT_W,
    VEH_TYPE, VEH_CAP_KG,
)


def _make_sol_and_data(tiny_data):
    sol = create_solution(tiny_data["n_vehicles"], tiny_data["n_customers"],
                          n_sku=tiny_data["n_sku"])
    return sol


def _insert(sol, v, pos, customer, data):
    """Helper to insert a DELIVER stop."""
    orders = data["orders"]
    loc_id = int(orders[customer, ORD_LOC])
    insert_stop(sol, v, pos, loc_id, ACT_DELIVER, data["dist_matrix"],
                data["vehicles"], orders, customer=customer)


# ---------------------------------------------------------------------------
# can_insert_customer
# ---------------------------------------------------------------------------

def test_truck_unrestricted(tiny_data):
    """C1 (not restricted) in truck (cap=2000) -> True."""
    sol = _make_sol_and_data(tiny_data)
    result = can_insert_customer(
        sol, 0, 1, tiny_data["orders"], tiny_data["vehicles"],
        tiny_data["locations"], tiny_data["allowed_bike"],
    )
    assert result == True


def test_truck_restricted_customer(tiny_data):
    """C0 (allowed_bike=0, heavy) in truck -> should be allowed (trucks can serve all)."""
    sol = _make_sol_and_data(tiny_data)
    # In v2, allowed_bike[0]=0 means NOT allowed on bike, but trucks can serve
    result = can_insert_customer(
        sol, 0, 0, tiny_data["orders"], tiny_data["vehicles"],
        tiny_data["locations"], tiny_data["allowed_bike"],
    )
    assert result == True


def test_bike_allowed_customer(tiny_data):
    """C2 (allowed_bike=1) on bike -> True."""
    sol = _make_sol_and_data(tiny_data)
    result = can_insert_customer(
        sol, 1, 2, tiny_data["orders"], tiny_data["vehicles"],
        tiny_data["locations"], tiny_data["allowed_bike"],
    )
    assert result == True


def test_capacity_overload_bike(tiny_data):
    """C0 (demand=100) in bike (cap=60) -> False."""
    sol = _make_sol_and_data(tiny_data)
    result = can_insert_customer(
        sol, 1, 0, tiny_data["orders"], tiny_data["vehicles"],
        tiny_data["locations"], tiny_data["allowed_bike"],
    )
    # C0 demand=100 > bike cap=60, also allowed_bike[0]=0
    assert result == False


# ---------------------------------------------------------------------------
# check_route_capacity_quick
# ---------------------------------------------------------------------------

def test_under_capacity(tiny_data):
    sol = _make_sol_and_data(tiny_data)
    _insert(sol, 1, 0, 3, tiny_data)  # C3: 8kg
    _insert(sol, 1, 1, 4, tiny_data)  # C4: 5kg
    assert check_route_capacity_quick(sol, 1, tiny_data["vehicles"]) == True


def test_over_capacity(tiny_data):
    sol = _make_sol_and_data(tiny_data)
    sol["loads_kg"][1] = 65.0
    assert check_route_capacity_quick(sol, 1, tiny_data["vehicles"]) == False


# ---------------------------------------------------------------------------
# check_all_assigned
# ---------------------------------------------------------------------------

def test_none_assigned(tiny_data):
    sol = _make_sol_and_data(tiny_data)
    assert check_all_assigned(sol, 5) == False


def test_all_assigned(tiny_data):
    sol = _make_sol_and_data(tiny_data)
    _insert(sol, 0, 0, 0, tiny_data)
    _insert(sol, 1, 0, 1, tiny_data)
    _insert(sol, 1, 1, 2, tiny_data)
    _insert(sol, 2, 0, 3, tiny_data)
    _insert(sol, 2, 1, 4, tiny_data)
    assert check_all_assigned(sol, 5) == True
