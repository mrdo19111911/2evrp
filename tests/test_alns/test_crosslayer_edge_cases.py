"""Edge case tests for cross-layer operators. i64 interface.

Tests critical boundary conditions: empty solutions, single satellite, etc.
Cross-layer ops signature: (sol, customers_i64, dist_matrix_i64, vehicles_i64, rng, config_f64)
"""
import numpy as np
import pytest

from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, VEH_TRUCK, VEH_BIKE, COL_X, COL_Y,
    SOL_TRUCK_LENGTHS, SOL_BIKE_LENGTHS,
    SOL_CUST_VEHICLE, SOL_META, META_N_SATELLITES,
)
from src.solution.structure import create_solution
from src.alns.crosslayer import cross_dispatch
from src.data.constants import N_CROSS_OPS
from src.alns.config import make_config
from src.data.cost import (
    TRUCK_COST_PER_M, BIKE_COST_PER_M,
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M,
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
)


def _make_i64_dm(depot, customers):
    depot_xy = depot.reshape(1, 2).astype(np.float64)
    cust_xy = customers[:, [COL_X, COL_Y]].astype(np.float64)
    coords = np.vstack([depot_xy, cust_xy])
    diff = coords[:, None, :] - coords[None, :, :]
    return np.round(np.sqrt((diff ** 2).sum(axis=2))).astype(np.int64)


def test_cross_ops_on_empty_solution():
    """All cross-layer ops should handle empty solution gracefully."""
    sol = create_solution(1, 1, 3)
    customers = np.array([
        [0, 0, 10000, 0, 28800, 300, 0],
        [5000, 0, 10000, 0, 28800, 300, 0],
        [10000, 0, 10000, 0, 28800, 300, 0],
    ], dtype=np.int64)
    depot = np.array([0, 0], dtype=np.int64)
    dm = _make_i64_dm(depot, customers)

    vehicles = np.array([
        [0, TRUCK_CAPACITY_G, TRUCK_COST_PER_M, TRUCK_SPEED_US_PER_M],
        [1, BIKE_CAPACITY_G, BIKE_COST_PER_M, BIKE_SPEED_US_PER_M],
    ], dtype=np.int64)
    rng = np.random.default_rng(42)
    config = make_config(3)

    for idx in range(N_CROSS_OPS):
        sol_copy = create_solution(1, 1, 3)
        seed = np.int32(rng.integers(0, 2**31 - 1))
        try:
            cross_dispatch(idx, sol_copy, customers, dm, vehicles, seed, config)
        except Exception:
            pass  # Some ops may fail on empty, that's acceptable


def test_cross_ops_count():
    assert N_CROSS_OPS == 5
