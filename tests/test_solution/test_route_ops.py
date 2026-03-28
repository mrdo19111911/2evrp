"""Tests for single-route operations — Data Model v2."""
import numpy as np
import pytest

from src.solution.structure import create_solution
from src.solution.route_ops import (
    insert_stop,
    remove_stop,
    swap_stops_within,
    reverse_segment,
)
from src.data.constants import (
    ACT_DELIVER, ACT_PICKUP, ACT_PAD,
    VTYPE_TRUCK, VTYPE_BIKE,
    ORD_LOC, ORD_QTY, ORD_UNIT_W,
    DM_DIST,
)


def _make_sol(data):
    return create_solution(data["n_vehicles"], data["n_customers"],
                           n_sku=data["n_sku"])


def _ins(sol, v, pos, customer, data, action=ACT_DELIVER):
    orders = data["orders"]
    loc_id = int(orders[customer, ORD_LOC])
    insert_stop(sol, v, pos, loc_id, action, data["dist_matrix"],
                data["vehicles"], orders, customer=customer if action == ACT_DELIVER else -1)


def _loc(data, customer):
    return int(data["orders"][customer, ORD_LOC])


def _demand(data, customer):
    o = data["orders"]
    return o[customer, ORD_QTY] * o[customer, ORD_UNIT_W]


# ---------------------------------------------------------------------------
# insert_stop
# ---------------------------------------------------------------------------

def test_insert_empty_route(tiny_data):
    """Insert C0 at pos=0 in empty truck route -> length=1."""
    sol = _make_sol(tiny_data)
    _ins(sol, 0, 0, 0, tiny_data)
    assert sol["lengths"][0] == 1
    assert sol["stops"][0, 0] == _loc(tiny_data, 0)
    assert sol["actions"][0, 0] == ACT_DELIVER
    assert sol["distances"][0] > 0.0


def test_insert_middle_shifts(tiny_data):
    """Insert C1 at pos=1 in route [C3, C4] -> [C3, C1, C4]."""
    sol = _make_sol(tiny_data)
    _ins(sol, 1, 0, 3, tiny_data)
    _ins(sol, 1, 1, 4, tiny_data)
    assert sol["lengths"][1] == 2

    _ins(sol, 1, 1, 1, tiny_data)
    assert sol["lengths"][1] == 3
    assert sol["stops"][1, 0] == _loc(tiny_data, 3)
    assert sol["stops"][1, 1] == _loc(tiny_data, 1)
    assert sol["stops"][1, 2] == _loc(tiny_data, 4)


def test_insert_updates_customer_index(tiny_data):
    """DELIVER insert sets cust_vehicle, cust_route_pos."""
    sol = _make_sol(tiny_data)
    _ins(sol, 0, 0, 0, tiny_data)
    assert sol["cust_vehicle"][0] == 0
    assert sol["cust_route_pos"][0] == 0


def test_insert_pickup_no_customer_index(tiny_data):
    """PICKUP does not update customer index."""
    sol = _make_sol(tiny_data)
    loc_id = _loc(tiny_data, 3)
    insert_stop(sol, 1, 0, loc_id, ACT_PICKUP, tiny_data["dist_matrix"],
                tiny_data["vehicles"], tiny_data["orders"], customer=-1)
    assert sol["cust_vehicle"][3] == -1


def test_insert_shifts_route_pos(tiny_data):
    """After insert at pos=0, subsequent customers get updated route_pos."""
    sol = _make_sol(tiny_data)
    _ins(sol, 1, 0, 3, tiny_data)
    _ins(sol, 1, 1, 4, tiny_data)

    _ins(sol, 1, 0, 1, tiny_data)
    # All 3 customers should be assigned to vehicle 1
    assert sol["cust_vehicle"][1] == 1
    assert sol["cust_vehicle"][3] == 1
    assert sol["cust_vehicle"][4] == 1
    # Route should have 3 stops
    assert sol["lengths"][1] == 3


# ---------------------------------------------------------------------------
# remove_stop
# ---------------------------------------------------------------------------

def test_remove_single(tiny_data):
    """Remove only stop -> empty route."""
    sol = _make_sol(tiny_data)
    _ins(sol, 0, 0, 0, tiny_data)
    cust, act = remove_stop(sol, 0, 0, tiny_data["dist_matrix"],
                            tiny_data["vehicles"], tiny_data["orders"], customer=0)
    assert cust == _loc(tiny_data, 0)
    assert sol["lengths"][0] == 0


def test_remove_shifts_left(tiny_data):
    """Remove middle -> remaining shift left."""
    sol = _make_sol(tiny_data)
    _ins(sol, 1, 0, 3, tiny_data)
    _ins(sol, 1, 1, 1, tiny_data)
    _ins(sol, 1, 2, 4, tiny_data)

    remove_stop(sol, 1, 1, tiny_data["dist_matrix"],
                tiny_data["vehicles"], tiny_data["orders"], customer=1)
    assert sol["lengths"][1] == 2
    assert sol["stops"][1, 0] == _loc(tiny_data, 3)
    assert sol["stops"][1, 1] == _loc(tiny_data, 4)


def test_remove_clears_customer_index(tiny_data):
    """Removing DELIVER sets cust_vehicle = -1."""
    sol = _make_sol(tiny_data)
    _ins(sol, 0, 0, 0, tiny_data)
    assert sol["cust_vehicle"][0] == 0

    remove_stop(sol, 0, 0, tiny_data["dist_matrix"],
                tiny_data["vehicles"], tiny_data["orders"], customer=0)
    assert sol["cust_vehicle"][0] == -1


# ---------------------------------------------------------------------------
# swap_stops_within
# ---------------------------------------------------------------------------

def test_swap_first_last(tiny_data):
    """Route [C3, C1, C4] swap(0,2) -> [C4, C1, C3]."""
    sol = _make_sol(tiny_data)
    _ins(sol, 1, 0, 3, tiny_data)
    _ins(sol, 1, 1, 1, tiny_data)
    _ins(sol, 1, 2, 4, tiny_data)

    swap_stops_within(sol, 1, 0, 2, tiny_data["dist_matrix"], tiny_data["vehicles"])

    assert sol["stops"][1, 0] == _loc(tiny_data, 4)
    assert sol["stops"][1, 1] == _loc(tiny_data, 1)
    assert sol["stops"][1, 2] == _loc(tiny_data, 3)
    assert sol["lengths"][1] == 3


# ---------------------------------------------------------------------------
# reverse_segment
# ---------------------------------------------------------------------------

def test_reverse_middle(tiny_data):
    """Route [C0, C3, C4, C1, C2], reverse [1,3] -> [C0, C1, C4, C3, C2]."""
    sol = _make_sol(tiny_data)
    for i, c in enumerate([0, 3, 4, 1, 2]):
        _ins(sol, 0, i, c, tiny_data)

    reverse_segment(sol, 0, 1, 3, tiny_data["dist_matrix"], tiny_data["vehicles"])

    assert sol["stops"][0, 0] == _loc(tiny_data, 0)
    assert sol["stops"][0, 1] == _loc(tiny_data, 1)
    assert sol["stops"][0, 2] == _loc(tiny_data, 4)
    assert sol["stops"][0, 3] == _loc(tiny_data, 3)
    assert sol["stops"][0, 4] == _loc(tiny_data, 2)


def test_reverse_entire_route(tiny_data):
    """Reverse [C3, C4] -> [C4, C3]."""
    sol = _make_sol(tiny_data)
    _ins(sol, 1, 0, 3, tiny_data)
    _ins(sol, 1, 1, 4, tiny_data)

    reverse_segment(sol, 1, 0, 1, tiny_data["dist_matrix"], tiny_data["vehicles"])

    assert sol["stops"][1, 0] == _loc(tiny_data, 4)
    assert sol["stops"][1, 1] == _loc(tiny_data, 3)
