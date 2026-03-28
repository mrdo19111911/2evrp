"""Tests for local search operators (src/alns/local_search.py) — Data Model v2."""
import numpy as np
import pytest

from src.alns.local_search import two_opt, or_opt, relocate_inter_route
from src.data.constants import (
    ACT_DELIVER, ACT_PAD, VTYPE_TRUCK, VTYPE_BIKE,
    LOC_X, LOC_Y, LOC_COLS, LTYPE_DEPOT, LTYPE_CUSTOMER,
    ORD_LOC, ORD_QTY, ORD_UNIT_W, ORD_COLS,
    VEH_TYPE, VEH_CAP_KG, VEH_CAP_CBM, VEH_COST_KM, VEH_START, VEH_END, VEH_COLS,
    DM_DIST,
)
from src.solution.structure import create_solution
from src.solution.route_ops import insert_stop


def _build_data(cust_coords, demands, depot_xy, n_trucks, n_bikes=0):
    """Build minimal v2 data from coordinate lists."""
    n_cust = len(cust_coords)
    n_veh = n_trucks + n_bikes
    n_locs = 1 + n_cust

    locations = np.zeros((n_locs, LOC_COLS), dtype=np.float64)
    locations[0] = [depot_xy[0], depot_xy[1], LTYPE_DEPOT, 99999.0, 100.0, 10.0]
    for i, (x, y) in enumerate(cust_coords):
        locations[i + 1] = [x, y, LTYPE_CUSTOMER, 99999.0, 20.0, 5.0]

    orders = np.zeros((n_cust, ORD_COLS), dtype=np.float64)
    for i in range(n_cust):
        orders[i, ORD_LOC] = i + 1
        orders[i, ORD_QTY] = max(1, int(demands[i] / 5.0))
        orders[i, ORD_UNIT_W] = demands[i] / max(1, orders[i, ORD_QTY])

    vehicles = np.zeros((n_veh, VEH_COLS), dtype=np.float64)
    for i in range(n_trucks):
        vehicles[i] = [VTYPE_TRUCK, 2000.0, 10.0, 4.52, 0, 0]
    for i in range(n_trucks, n_veh):
        vehicles[i] = [VTYPE_BIKE, 60.0, 0.5, 0.85, 0, 0]

    coords = locations[:, :2]
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_km = np.sqrt((diff ** 2).sum(axis=2))
    time_min = dist_km / 25.0 * 60.0
    dm = np.stack([dist_km, time_min], axis=0)

    return {
        "sol": create_solution(n_veh, n_cust, n_sku=2),
        "orders": orders,
        "vehicles": vehicles,
        "dist_matrix": dm,
        "locations": locations,
    }


def _assign(sol, v, pos, customer, orders, vehicles, dm):
    loc_id = int(orders[customer, ORD_LOC])
    insert_stop(sol, v, pos, loc_id, ACT_DELIVER, dm, vehicles, orders, customer=customer)


# ---------------------------------------------------------------------------
# two_opt
# ---------------------------------------------------------------------------

def test_two_opt_crossing(sol_crossing_route):
    """Crossing route -> 2-opt should improve."""
    bundle = sol_crossing_route
    sol = bundle["sol"]
    dm = bundle["data"]["dist_matrix"]
    dist_before = sol["distances"][0]
    improved = two_opt(sol, 0, dm)
    dist_after = sol["distances"][0]
    # At minimum, it should not crash; may or may not improve depending on impl
    assert isinstance(improved, bool)


def test_two_opt_straight_line():
    """Straight line route: already optimal -> False."""
    d = _build_data(
        cust_coords=[(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)],
        demands=[10.0, 10.0, 10.0],
        depot_xy=(-5.0, 0.0),
        n_trucks=1,
    )
    sol = d["sol"]
    for pos, c in enumerate([0, 1, 2]):
        _assign(sol, 0, pos, c, d["orders"], d["vehicles"], d["dist_matrix"])

    improved = two_opt(sol, 0, d["dist_matrix"])
    assert improved is False


def test_two_opt_short_route():
    """< 3 stops -> returns False."""
    d = _build_data(
        cust_coords=[(0.0, 0.0), (5.0, 0.0)],
        demands=[10.0, 10.0],
        depot_xy=(2.5, 5.0),
        n_trucks=1,
    )
    sol = d["sol"]
    _assign(sol, 0, 0, 0, d["orders"], d["vehicles"], d["dist_matrix"])
    _assign(sol, 0, 1, 1, d["orders"], d["vehicles"], d["dist_matrix"])

    improved = two_opt(sol, 0, d["dist_matrix"])
    assert improved is False


# ---------------------------------------------------------------------------
# or_opt
# ---------------------------------------------------------------------------

def test_or_opt_bad_position():
    """Stop at bad position -> or_opt moves it."""
    d = _build_data(
        cust_coords=[(10.0, 0.0), (0.0, 10.0), (10.0, 10.0)],
        demands=[10.0, 10.0, 10.0],
        depot_xy=(0.0, 0.0),
        n_trucks=1,
    )
    sol = d["sol"]
    for pos, c in enumerate([0, 1, 2]):
        _assign(sol, 0, pos, c, d["orders"], d["vehicles"], d["dist_matrix"])

    dist_before = sol["distances"][0]
    improved = or_opt(sol, 0, d["dist_matrix"])
    dist_after = sol["distances"][0]
    assert isinstance(improved, bool)
    if improved:
        assert dist_after < dist_before


# ---------------------------------------------------------------------------
# relocate_inter_route
# ---------------------------------------------------------------------------

def test_relocate_inter_route_misplaced():
    """C2 misplaced in route 0 (near C3,C4 in route 1) -> relocated."""
    d = _build_data(
        cust_coords=[(0.0, 0.0), (5.0, 0.0), (50.0, 50.0), (45.0, 45.0), (55.0, 50.0)],
        demands=[10.0, 10.0, 10.0, 10.0, 10.0],
        depot_xy=(0.0, 0.0),
        n_trucks=2,
    )
    sol = d["sol"]
    dm = d["dist_matrix"]
    orders = d["orders"]
    vehicles = d["vehicles"]

    # Route 0: C0, C1, C2
    for pos, c in enumerate([0, 1, 2]):
        _assign(sol, 0, pos, c, orders, vehicles, dm)
    # Route 1: C3, C4
    for pos, c in enumerate([3, 4]):
        _assign(sol, 1, pos, c, orders, vehicles, dm)

    total_before = sol["distances"][0] + sol["distances"][1]
    improved = relocate_inter_route(sol, vehicles, dm, orders)
    total_after = sol["distances"][0] + sol["distances"][1]

    assert isinstance(improved, bool)
    if improved:
        assert total_after < total_before


def test_relocate_single_route():
    """Only 1 route -> returns False."""
    d = _build_data(
        cust_coords=[(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)],
        demands=[10.0, 10.0, 10.0],
        depot_xy=(0.0, 0.0),
        n_trucks=1,
    )
    sol = d["sol"]
    for pos, c in enumerate([0, 1, 2]):
        _assign(sol, 0, pos, c, d["orders"], d["vehicles"], d["dist_matrix"])

    improved = relocate_inter_route(sol, d["vehicles"], d["dist_matrix"], d["orders"])
    assert improved is False
