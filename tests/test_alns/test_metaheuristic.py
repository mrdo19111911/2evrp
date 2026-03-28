"""Metaheuristic property tests for the ALNS solver. i64 interface.

Tests stochastic algorithms via PROPERTIES and INVARIANTS.
No restricted param. No _make_dist_matrix import from conftest.
"""
import numpy as np
import pytest

from src.data.constants import (
    ACT_DELIVER, VEH_TRUCK, VEH_BIKE, COL_X, COL_Y, COL_DEMAND,
    SOL_TRUCK_LENGTHS, SOL_BIKE_LENGTHS, SOL_CUST_VEHICLE,
    CFG_Q_MIN, CFG_Q_MAX,
)
from src.solution.structure import create_solution, copy_solution
from src.alns.destroy_basic import random_removal
from src.alns.repair_ops import greedy_insertion
from src.alns.config import make_config
from src.data.cost import (
    TRUCK_COST_PER_M, TRUCK_SPEED_US_PER_M, TRUCK_CAPACITY_G,
    BIKE_COST_PER_M, BIKE_SPEED_US_PER_M, BIKE_CAPACITY_G,
)
from tests.test_alns.conftest import _assign_customer_simple


def _make_dm_i64(depot, customers):
    depot_xy = depot.reshape(1, 2).astype(np.float64)
    cust_xy = customers[:, [COL_X, COL_Y]].astype(np.float64)
    coords = np.vstack([depot_xy, cust_xy])
    diff = coords[:, None, :] - coords[None, :, :]
    return np.round(np.sqrt((diff ** 2).sum(axis=2))).astype(np.int64)


def _build_assigned_instance():
    """Build small instance with all 5 customers assigned. i64."""
    customers = np.array([
        [    0, 5000, 100000,    0, 7200,  900, 0],
        [10000, 5000,  20000, 3600, 10800, 300, 0],
        [ 5000, 10000, 10000,    0, 14400, 300, 1],
        [ 2000,     0,  8000, 1800, 9000,  300, 0],
        [ 8000,     0,  5000,    0, 28800, 300, 0],
    ], dtype=np.int64)
    vehicles = np.array([
        [0, TRUCK_CAPACITY_G, TRUCK_COST_PER_M, TRUCK_SPEED_US_PER_M],
        [1, BIKE_CAPACITY_G, BIKE_COST_PER_M, BIKE_SPEED_US_PER_M],
        [1, BIKE_CAPACITY_G, BIKE_COST_PER_M, BIKE_SPEED_US_PER_M],
    ], dtype=np.int64)
    depot = np.array([5000, 5000], dtype=np.int64)
    dm = _make_dm_i64(depot, customers)

    sol = create_solution(1, 2, 5)
    _assign_customer_simple(sol, 0, VEH_TRUCK, 0, 0, customers[0, COL_DEMAND])
    _assign_customer_simple(sol, 1, VEH_BIKE, 0, 0, customers[1, COL_DEMAND])
    _assign_customer_simple(sol, 2, VEH_BIKE, 0, 1, customers[2, COL_DEMAND])
    _assign_customer_simple(sol, 3, VEH_BIKE, 1, 0, customers[3, COL_DEMAND])
    _assign_customer_simple(sol, 4, VEH_BIKE, 1, 1, customers[4, COL_DEMAND])

    return sol, customers, vehicles, dm


def test_destroy_repair_preserves_all_customers():
    """After destroy + repair, all customers should be assigned."""
    sol, customers, vehicles, dm = _build_assigned_instance()
    sol = copy_solution(sol)
    config = make_config(5, overrides={CFG_Q_MIN: 1, CFG_Q_MAX: 3})

    removed = random_removal(sol, customers, dm, 42, config)
    greedy_insertion(sol, removed, customers, dm, vehicles, 7, config)

    for c in range(5):
        assert sol[SOL_CUST_VEHICLE][c] >= 0, f"C{c} unassigned after D+R"


def test_destroy_removes_customers():
    """Destroy should unassign some customers."""
    sol, customers, vehicles, dm = _build_assigned_instance()
    sol = copy_solution(sol)
    config = make_config(5, overrides={CFG_Q_MIN: 1, CFG_Q_MAX: 3})

    removed = random_removal(sol, customers, dm, 0, config)
    assert len(removed) >= 1
    for c in removed:
        assert sol[SOL_CUST_VEHICLE][c] == -1


def test_copy_solution_independence():
    """copy_solution creates independent arrays."""
    sol, _, _, _ = _build_assigned_instance()
    sol2 = copy_solution(sol)
    sol2[SOL_TRUCK_LENGTHS][0] = 99
    assert sol[SOL_TRUCK_LENGTHS][0] != 99
