"""Tests for cross-route operations — Data Model v2."""
import numpy as np
import pytest

from src.solution.structure import create_solution
from src.solution.route_ops import insert_stop
from src.solution.solution_ops import (
    move_stop,
    swap_stops_between,
    clear_route,
    create_route_from_list,
)
from src.data.constants import (
    ACT_DELIVER, ACT_PICKUP, ACT_PAD,
    VTYPE_TRUCK, VTYPE_BIKE,
    ORD_LOC, ORD_QTY, ORD_UNIT_W,
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


def _loc(data, customer):
    return int(data["orders"][customer, ORD_LOC])


# ---------------------------------------------------------------------------
# move_stop
# ---------------------------------------------------------------------------

def test_move_truck_to_bike(tiny_data):
    """Move C1 from v=0 (truck) to v=1 (bike)."""
    sol = _make_sol(tiny_data)
    dm = tiny_data["dist_matrix"]
    veh = tiny_data["vehicles"]
    orders = tiny_data["orders"]

    _ins(sol, 0, 0, 0, tiny_data)  # truck: [C0, C1]
    _ins(sol, 0, 1, 1, tiny_data)

    move_stop(sol, 0, 1, 1, 0, dm, veh, orders, customer=1)

    assert sol["lengths"][0] == 1  # truck: [C0]
    assert sol["lengths"][1] == 1  # bike: [C1]
    assert sol["cust_vehicle"][1] == 1


def test_move_updates_distances(tiny_data):
    """Both routes have distances recalculated."""
    sol = _make_sol(tiny_data)
    dm = tiny_data["dist_matrix"]
    veh = tiny_data["vehicles"]
    orders = tiny_data["orders"]

    _ins(sol, 0, 0, 0, tiny_data)
    _ins(sol, 0, 1, 1, tiny_data)

    move_stop(sol, 0, 1, 1, 0, dm, veh, orders, customer=1)

    assert sol["distances"][0] > 0.0
    assert sol["distances"][1] > 0.0


def test_move_within_same_type(tiny_data):
    """Move between two bike routes."""
    sol = _make_sol(tiny_data)
    dm = tiny_data["dist_matrix"]
    veh = tiny_data["vehicles"]
    orders = tiny_data["orders"]

    _ins(sol, 1, 0, 3, tiny_data)  # bike 0: [C3, C4]
    _ins(sol, 1, 1, 4, tiny_data)

    move_stop(sol, 1, 1, 2, 0, dm, veh, orders, customer=4)

    assert sol["lengths"][1] == 1
    assert sol["lengths"][2] == 1
    assert sol["cust_vehicle"][4] == 2


# ---------------------------------------------------------------------------
# swap_stops_between
# ---------------------------------------------------------------------------

def test_swap_between_routes(tiny_data):
    """Swap C0 (truck) and C3 (bike)."""
    sol = _make_sol(tiny_data)
    dm = tiny_data["dist_matrix"]
    veh = tiny_data["vehicles"]
    orders = tiny_data["orders"]

    _ins(sol, 0, 0, 0, tiny_data)  # truck: [C0]
    _ins(sol, 1, 0, 3, tiny_data)  # bike: [C3]

    swap_stops_between(sol, 0, 0, 1, 0, dm, veh, orders, cust_a=0, cust_b=3)

    assert sol["stops"][0, 0] == _loc(tiny_data, 3)
    assert sol["stops"][1, 0] == _loc(tiny_data, 0)


def test_swap_updates_customer_index(tiny_data):
    sol = _make_sol(tiny_data)
    dm = tiny_data["dist_matrix"]
    veh = tiny_data["vehicles"]
    orders = tiny_data["orders"]

    _ins(sol, 0, 0, 0, tiny_data)
    _ins(sol, 1, 0, 3, tiny_data)

    swap_stops_between(sol, 0, 0, 1, 0, dm, veh, orders, cust_a=0, cust_b=3)

    assert sol["cust_vehicle"][3] == 0
    assert sol["cust_vehicle"][0] == 1


# ---------------------------------------------------------------------------
# clear_route
# ---------------------------------------------------------------------------

def test_clear_populated(tiny_data):
    """Clear route with 3 stops -> length=0."""
    sol = _make_sol(tiny_data)
    _ins(sol, 1, 0, 3, tiny_data)
    _ins(sol, 1, 1, 4, tiny_data)
    _ins(sol, 1, 2, 1, tiny_data)

    removed = clear_route(sol, 1)
    assert len(removed) == 3
    assert sol["lengths"][1] == 0
    assert sol["cust_vehicle"][3] == -1
    assert sol["cust_vehicle"][4] == -1
    assert sol["cust_vehicle"][1] == -1


def test_clear_empty(tiny_data):
    sol = _make_sol(tiny_data)
    removed = clear_route(sol, 0)
    assert removed == []


# ---------------------------------------------------------------------------
# create_route_from_list
# ---------------------------------------------------------------------------

def test_create_from_list(tiny_data):
    """Build bike route from list of (loc, action) pairs."""
    sol = _make_sol(tiny_data)
    dm = tiny_data["dist_matrix"]
    veh = tiny_data["vehicles"]
    orders = tiny_data["orders"]

    stop_list = [
        (_loc(tiny_data, 3), ACT_DELIVER),
        (_loc(tiny_data, 4), ACT_DELIVER),
        (_loc(tiny_data, 1), ACT_DELIVER),
    ]
    create_route_from_list(sol, 1, stop_list, dm, veh, orders)

    assert sol["lengths"][1] == 3
    assert sol["stops"][1, 0] == _loc(tiny_data, 3)
    assert sol["stops"][1, 1] == _loc(tiny_data, 4)
    assert sol["stops"][1, 2] == _loc(tiny_data, 1)


def test_create_empty_list(tiny_data):
    sol = _make_sol(tiny_data)
    dm = tiny_data["dist_matrix"]
    veh = tiny_data["vehicles"]
    orders = tiny_data["orders"]

    create_route_from_list(sol, 0, [], dm, veh, orders)
    assert sol["lengths"][0] == 0


def test_create_computes_distance(tiny_data):
    sol = _make_sol(tiny_data)
    dm = tiny_data["dist_matrix"]
    veh = tiny_data["vehicles"]
    orders = tiny_data["orders"]

    stop_list = [
        (_loc(tiny_data, 0), ACT_DELIVER),
        (_loc(tiny_data, 1), ACT_DELIVER),
    ]
    create_route_from_list(sol, 0, stop_list, dm, veh, orders)
    assert sol["distances"][0] > 0.0
