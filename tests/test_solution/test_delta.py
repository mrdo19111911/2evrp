"""Tests for delta cost evaluation — Data Model v2.

Unified vehicle arrays, (2, N, N) dist_matrix.
"""
import numpy as np
import pytest

from src.solution.structure import create_solution
from src.solution.route_ops import insert_stop, remove_stop
from src.solution.delta import (
    insertion_cost_delta,
    removal_cost_delta,
    best_insertion_pos,
    find_best_insertion_all_routes,
)
from src.data.constants import (
    ACT_DELIVER, ACT_PAD, VTYPE_TRUCK, VTYPE_BIKE,
    ORD_LOC, ORD_QTY, ORD_UNIT_W,
    DM_DIST,
)


def _make_sol(data):
    return create_solution(data["n_vehicles"], data["n_customers"],
                           n_sku=data["n_sku"])


def _ins(sol, v, pos, customer, data):
    orders = data["orders"]
    loc_id = int(orders[customer, ORD_LOC])
    insert_stop(sol, v, pos, loc_id, ACT_DELIVER, data["dist_matrix"],
                data["vehicles"], orders, customer=customer)


def _loc(data, customer):
    return int(data["orders"][customer, ORD_LOC])


# ---------------------------------------------------------------------------
# insertion_cost_delta
# ---------------------------------------------------------------------------

def test_insert_empty_route(tiny_data):
    """Empty route: delta = round-trip to customer location."""
    sol = _make_sol(tiny_data)
    dm = tiny_data["dist_matrix"]
    veh = tiny_data["vehicles"]
    loc_id = _loc(tiny_data, 0)
    delta = insertion_cost_delta(sol, 0, 0, loc_id, dm, veh)
    # Should be depot->loc->depot distance
    d = dm[DM_DIST, 0, loc_id] + dm[DM_DIST, loc_id, 0]
    assert pytest.approx(delta, abs=1e-4) == d


def test_insert_consistency(tiny_data):
    """old_distance + delta == new_distance after actual insert."""
    sol = _make_sol(tiny_data)
    dm = tiny_data["dist_matrix"]
    veh = tiny_data["vehicles"]
    orders = tiny_data["orders"]

    # Build route [C3, C4] on bike (v=1)
    _ins(sol, 1, 0, 3, tiny_data)
    _ins(sol, 1, 1, 4, tiny_data)
    old_dist = sol["distances"][1]

    loc_id = _loc(tiny_data, 1)  # C1
    delta = insertion_cost_delta(sol, 1, 1, loc_id, dm, veh)

    insert_stop(sol, 1, 1, loc_id, ACT_DELIVER, dm, veh, orders, customer=1)
    new_dist = sol["distances"][1]

    assert pytest.approx(old_dist + delta, abs=1e-4) == new_dist


# ---------------------------------------------------------------------------
# removal_cost_delta
# ---------------------------------------------------------------------------

def test_remove_only_stop(tiny_data):
    """Remove only stop: delta = -(round-trip distance)."""
    sol = _make_sol(tiny_data)
    dm = tiny_data["dist_matrix"]
    veh = tiny_data["vehicles"]

    _ins(sol, 0, 0, 0, tiny_data)
    dist_before = sol["distances"][0]
    delta = removal_cost_delta(sol, 0, 0, dm, veh)
    assert pytest.approx(delta, abs=1e-4) == -dist_before


def test_removal_consistency(tiny_data):
    """old + delta == new after actual removal."""
    sol = _make_sol(tiny_data)
    dm = tiny_data["dist_matrix"]
    veh = tiny_data["vehicles"]
    orders = tiny_data["orders"]

    _ins(sol, 1, 0, 3, tiny_data)
    _ins(sol, 1, 1, 1, tiny_data)
    _ins(sol, 1, 2, 4, tiny_data)

    old_dist = sol["distances"][1]
    delta = removal_cost_delta(sol, 1, 1, dm, veh)

    remove_stop(sol, 1, 1, dm, veh, orders, customer=1)
    new_dist = sol["distances"][1]

    assert pytest.approx(old_dist + delta, abs=1e-4) == new_dist


def test_removal_nonpositive(tiny_data):
    """Removal delta should be <= 0."""
    sol = _make_sol(tiny_data)
    dm = tiny_data["dist_matrix"]
    veh = tiny_data["vehicles"]

    _ins(sol, 1, 0, 3, tiny_data)
    _ins(sol, 1, 1, 1, tiny_data)
    _ins(sol, 1, 2, 4, tiny_data)

    for pos in range(3):
        delta = removal_cost_delta(sol, 1, pos, dm, veh)
        assert delta <= 1e-9, f"pos={pos} delta={delta}"


# ---------------------------------------------------------------------------
# best_insertion_pos
# ---------------------------------------------------------------------------

def test_best_pos_empty(tiny_data):
    """Empty route: only pos=0 available."""
    sol = _make_sol(tiny_data)
    dm = tiny_data["dist_matrix"]
    veh = tiny_data["vehicles"]
    loc_id = _loc(tiny_data, 0)
    pos, delta = best_insertion_pos(sol, 0, loc_id, dm, veh)
    assert pos == 0
    assert delta > 0


def test_best_pos_minimizes(tiny_data):
    """Best pos should have smallest delta."""
    sol = _make_sol(tiny_data)
    dm = tiny_data["dist_matrix"]
    veh = tiny_data["vehicles"]

    _ins(sol, 1, 0, 3, tiny_data)
    _ins(sol, 1, 1, 4, tiny_data)

    loc_id = _loc(tiny_data, 1)
    pos, delta = best_insertion_pos(sol, 1, loc_id, dm, veh)

    # Verify this is actually the best
    for p in range(sol["lengths"][1] + 1):
        d = insertion_cost_delta(sol, 1, p, loc_id, dm, veh)
        assert d >= delta - 1e-9


# ---------------------------------------------------------------------------
# find_best_insertion_all_routes
# ---------------------------------------------------------------------------

def test_find_best_single_route(tiny_data):
    """One bike with space -> returns that route."""
    sol = _make_sol(tiny_data)
    dm = tiny_data["dist_matrix"]
    veh = tiny_data["vehicles"]
    orders = tiny_data["orders"]

    _ins(sol, 1, 0, 3, tiny_data)  # bike 0

    loc_id = _loc(tiny_data, 4)
    vid, pos, delta = find_best_insertion_all_routes(
        sol, 4, loc_id, dm, veh, orders,
    )
    assert vid >= 0
    assert delta < np.inf
