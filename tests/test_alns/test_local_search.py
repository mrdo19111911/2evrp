"""Tests for local search operators (src/alns/local_search.py). i64 interface."""
import numpy as np
import pytest

from src.alns.ls_intra import two_opt, or_opt
from src.alns.ls_inter import relocate_inter_route
from src.data.constants import (
    ACT_DELIVER, VEH_TRUCK, VEH_BIKE, COL_DEMAND, COL_X, COL_Y,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_TRUCK_LENGTHS,
    SOL_TRUCK_LOADS, SOL_TRUCK_DISTANCES,
    SOL_CUST_VEHICLE, SOL_CUST_VTYPE, SOL_CUST_ROUTE_POS,
)
from src.solution.structure import create_solution
from src.data.cost import (
    TRUCK_COST_PER_M, TRUCK_SPEED_US_PER_M, TRUCK_CAPACITY_G,
    BIKE_COST_PER_M, BIKE_SPEED_US_PER_M, BIKE_CAPACITY_G,
)


def _simple_dist_matrix_i64(coords_with_depot):
    """Euclidean distance matrix from (N+1, 2) coords (index 0 = depot). i64 meters."""
    n = len(coords_with_depot)
    coords = coords_with_depot.astype(np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.round(np.sqrt((diff ** 2).sum(axis=2))).astype(np.int64)


def _assign_simple(sol, customer_id, vtype, vid, pos, demand):
    """Manually assign customer to position in route."""
    if vtype == VEH_TRUCK:
        sol[SOL_TRUCK_STOPS][vid, pos] = customer_id
        sol[SOL_TRUCK_ACTIONS][vid, pos] = ACT_DELIVER
        sol[SOL_TRUCK_LENGTHS][vid] = max(sol[SOL_TRUCK_LENGTHS][vid], pos + 1)
        sol[SOL_TRUCK_LOADS][vid] += demand
    sol[SOL_CUST_VEHICLE][customer_id] = vid
    sol[SOL_CUST_VTYPE][customer_id] = vtype
    sol[SOL_CUST_ROUTE_POS][customer_id] = pos


def _compute_route_distance(sol, vid, dm):
    L = sol[SOL_TRUCK_LENGTHS][vid]
    if L == 0:
        return 0
    stops = sol[SOL_TRUCK_STOPS][vid]
    d = dm[0, stops[0] + 1]
    for i in range(L - 1):
        d += dm[stops[i] + 1, stops[i + 1] + 1]
    d += dm[stops[L - 1] + 1, 0]
    sol[SOL_TRUCK_DISTANCES][vid] = d
    return d


def _make_vehicles_1truck():
    return np.array([
        [0, TRUCK_CAPACITY_G, TRUCK_COST_PER_M, TRUCK_SPEED_US_PER_M],
    ], dtype=np.int64)


# ---------------------------------------------------------------------------
# two_opt
# ---------------------------------------------------------------------------

def test_two_opt_short_route():
    """< 3 stops -> returns False."""
    customers = np.array([
        [0, 0, 10000, 0, 28800, 300, 0],
        [5000, 0, 10000, 0, 28800, 300, 0],
    ], dtype=np.int64)
    depot = np.array([2500, 5000], dtype=np.int64)
    coords = np.vstack([depot.reshape(1, 2), customers[:, [COL_X, COL_Y]]])
    dm = _simple_dist_matrix_i64(coords)

    sol = create_solution(1, 0, 2)
    _assign_simple(sol, 0, VEH_TRUCK, 0, 0, 10000)
    _assign_simple(sol, 1, VEH_TRUCK, 0, 1, 10000)
    _compute_route_distance(sol, 0, dm)

    improved = two_opt(sol, VEH_TRUCK, 0, dm)
    assert improved is False


def test_two_opt_straight_line():
    """Straight line route: already optimal -> False."""
    customers = np.array([
        [0, 0, 10000, 0, 28800, 300, 0],
        [5000, 0, 10000, 0, 28800, 300, 0],
        [10000, 0, 10000, 0, 28800, 300, 0],
    ], dtype=np.int64)
    depot = np.array([-5000, 0], dtype=np.int64)
    coords = np.vstack([depot.reshape(1, 2), customers[:, [COL_X, COL_Y]]])
    dm = _simple_dist_matrix_i64(coords)

    sol = create_solution(1, 0, 3)
    for pos, c in enumerate([0, 1, 2]):
        _assign_simple(sol, c, VEH_TRUCK, 0, pos, 10000)
    _compute_route_distance(sol, 0, dm)

    improved = two_opt(sol, VEH_TRUCK, 0, dm)
    assert improved is False


# ---------------------------------------------------------------------------
# or_opt
# ---------------------------------------------------------------------------

def test_or_opt_returns_bool():
    customers = np.array([
        [10000, 0, 10000, 0, 28800, 300, 0],
        [0, 10000, 10000, 0, 28800, 300, 0],
        [10000, 10000, 10000, 0, 28800, 300, 0],
    ], dtype=np.int64)
    depot = np.array([0, 0], dtype=np.int64)
    coords = np.vstack([depot.reshape(1, 2), customers[:, [COL_X, COL_Y]]])
    dm = _simple_dist_matrix_i64(coords)

    sol = create_solution(1, 0, 3)
    for pos, c in enumerate([0, 1, 2]):
        _assign_simple(sol, c, VEH_TRUCK, 0, pos, 10000)
    _compute_route_distance(sol, 0, dm)

    improved = or_opt(sol, VEH_TRUCK, 0, dm)
    assert isinstance(improved, bool)


# ---------------------------------------------------------------------------
# relocate_inter_route — signature: (sol, vtype, dist_matrix, customers, vehicles)
# ---------------------------------------------------------------------------

def test_relocate_single_route():
    """Only 1 truck route -> returns False (need >= 2 routes)."""
    customers = np.array([
        [0, 0, 10000, 0, 28800, 300, 0],
        [5000, 0, 10000, 0, 28800, 300, 0],
        [10000, 0, 10000, 0, 28800, 300, 0],
    ], dtype=np.int64)
    depot = np.array([0, 0], dtype=np.int64)
    coords = np.vstack([depot.reshape(1, 2), customers[:, [COL_X, COL_Y]]])
    dm = _simple_dist_matrix_i64(coords)
    vehicles = _make_vehicles_1truck()

    sol = create_solution(1, 0, 3)
    for pos, c in enumerate([0, 1, 2]):
        _assign_simple(sol, c, VEH_TRUCK, 0, pos, 10000)
    _compute_route_distance(sol, 0, dm)

    improved = relocate_inter_route(sol, VEH_TRUCK, dm, customers, vehicles)
    assert improved is False
