"""Shared fixtures for viz tests: mock solutions, logs, archives, states — v2."""
import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_all_figures():
    """Close all matplotlib figures after every test."""
    import matplotlib.pyplot as plt
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Mock solution — Data Model v2 unified format
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_sol():
    """Minimal valid solution dict for 5 customers, 3 vehicles.

    Vehicle 0 (truck): loc1(deliver) -> loc2(deliver)
    Vehicle 1 (bike0): loc3(deliver) -> loc4(deliver)
    Vehicle 2 (bike1): loc5(deliver)
    """
    n_veh = 3
    L = 4
    return {
        "stops": np.array([
            [1, 2, -1, -1],
            [3, 4, -1, -1],
            [5, -1, -1, -1],
        ], dtype=np.int32),
        "actions": np.array([
            [0, 0, -1, -1],
            [0, 0, -1, -1],
            [0, -1, -1, -1],
        ], dtype=np.int8),
        "lengths": np.array([2, 2, 1], dtype=np.int32),
        "loads_kg": np.array([120.0, 18.0, 5.0], dtype=np.float64),
        "loads_cbm": np.array([0.1, 0.02, 0.01], dtype=np.float64),
        "distances": np.array([20.0, 12.0, 8.0], dtype=np.float64),
        "cust_vehicle": np.array([0, 0, 1, 1, 2], dtype=np.int32),
        "cust_route_pos": np.array([0, 1, 0, 1, 0], dtype=np.int32),
        "transfers": np.empty((0, 6), dtype=np.float64),
        "n_vehicles": n_veh,
        "n_customers": 5,
        "n_sku": 2,
        "max_route_len": L,
        "vehicle_sku_qty": np.zeros((n_veh, 5, 2), dtype=np.int32),
    }


@pytest.fixture
def empty_sol():
    """Solution with no assigned customers (all routes empty)."""
    n_veh = 3
    L = 4
    return {
        "stops": np.full((n_veh, L), -1, dtype=np.int32),
        "actions": np.full((n_veh, L), -1, dtype=np.int8),
        "lengths": np.zeros(n_veh, dtype=np.int32),
        "loads_kg": np.zeros(n_veh, dtype=np.float64),
        "loads_cbm": np.zeros(n_veh, dtype=np.float64),
        "distances": np.zeros(n_veh, dtype=np.float64),
        "cust_vehicle": np.full(5, -1, dtype=np.int32),
        "cust_route_pos": np.full(5, -1, dtype=np.int32),
        "transfers": np.empty((0, 6), dtype=np.float64),
        "n_vehicles": n_veh,
        "n_customers": 5,
        "n_sku": 2,
        "max_route_len": L,
        "vehicle_sku_qty": np.zeros((n_veh, 5, 2), dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Mock eval result
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_eval_result():
    return {
        "fitness": 742.3,
        "cost": 1_234_500.0,
        "makespan": 385.2,
        "penalty": 0.0,
        "feasible": True,
        "cap_violations": 0,
        "sync_violations": 0,
        "total_distance": 45.3,
        "vehicles_used": 3,
    }


@pytest.fixture
def mock_eval_result_infeasible():
    return {
        "fitness": 1500.0,
        "cost": 1_800_000.0,
        "makespan": 500.0,
        "penalty": 750.0,
        "feasible": False,
        "cap_violations": 1,
        "sync_violations": 0,
        "total_distance": 72.1,
        "vehicles_used": 3,
    }


# ---------------------------------------------------------------------------
# Mock vehicle states (from simulate) — unified
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vehicle_states():
    """List of state arrays, one per vehicle."""
    return [
        np.array([
            [1, 0, 10.0, 5.0, 15.0, 0.0, 100.0, 0.0, 0.01, 5.0, 1.0, 0],
            [2, 0, 30.0, 5.0, 35.0, 100.0, 120.0, 0.01, 0.02, 10.0, 1.0, 0],
        ], dtype=np.float64),
        np.array([
            [3, 0, 20.0, 5.0, 25.0, 0.0, 10.0, 0.0, 0.005, 5.0, 1.0, 1],
            [4, 0, 40.0, 5.0, 45.0, 10.0, 18.0, 0.005, 0.01, 7.0, 1.0, 1],
        ], dtype=np.float64),
        np.array([
            [5, 0, 15.0, 5.0, 20.0, 0.0, 5.0, 0.0, 0.005, 8.0, 1.0, 2],
        ], dtype=np.float64),
    ]


# Backward compat aliases
@pytest.fixture
def mock_truck_states(mock_vehicle_states):
    return mock_vehicle_states[:1]


@pytest.fixture
def mock_bike_states(mock_vehicle_states):
    return mock_vehicle_states[1:]


# ---------------------------------------------------------------------------
# Mock ALNS log
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_log_100():
    rng = np.random.default_rng(42)
    n = 100
    base = np.linspace(2000, 800, n)
    noise = rng.normal(0, 50, n)
    fitness = base + noise
    best_fitness = np.minimum.accumulate(fitness)
    cost = fitness * 1000
    makespan = np.linspace(500, 300, n) + rng.normal(0, 10, n)
    penalty = np.maximum(0, np.linspace(500, 0, n) + rng.normal(0, 20, n))
    temperature = 1000 * np.exp(-np.linspace(0, 5, n))
    feasible = (rng.random(n) > np.linspace(0.5, 0.1, n)).astype(np.float64)
    w3 = np.linspace(1.0, 5.0, n)
    accepted = (rng.random(n) > 0.3).astype(np.float64)

    n_snapshots = n // 10
    destroy_names = ["random_remove", "worst_remove", "shaw_remove", "zone_remove"]
    repair_names = ["greedy_insert", "regret_insert", "random_insert"]
    dw_raw = rng.dirichlet(np.ones(4), size=n_snapshots)
    rw_raw = rng.dirichlet(np.ones(3), size=n_snapshots)

    return {
        "iterations": np.arange(n),
        "fitness": fitness,
        "best_fitness": best_fitness,
        "cost": cost,
        "makespan": makespan,
        "penalty": penalty,
        "temperature": temperature,
        "feasible": feasible,
        "w3": w3,
        "accepted": accepted,
        "destroy_weights": dw_raw,
        "repair_weights": rw_raw,
        "destroy_names": destroy_names,
        "repair_names": repair_names,
        "destroy_scores": {name: rng.uniform(0, 100) for name in destroy_names},
        "destroy_counts": {name: rng.integers(50, 200) for name in destroy_names},
        "repair_scores": {name: rng.uniform(0, 100) for name in repair_names},
        "repair_counts": {name: rng.integers(50, 200) for name in repair_names},
        "weight_snapshot_iters": np.arange(0, n, 10),
    }


@pytest.fixture
def empty_log():
    return {
        "iterations": np.array([], dtype=np.int64),
        "fitness": np.array([], dtype=np.float64),
        "best_fitness": np.array([], dtype=np.float64),
        "cost": np.array([], dtype=np.float64),
        "makespan": np.array([], dtype=np.float64),
        "penalty": np.array([], dtype=np.float64),
        "temperature": np.array([], dtype=np.float64),
        "feasible": np.array([], dtype=np.float64),
        "w3": np.array([], dtype=np.float64),
        "accepted": np.array([], dtype=np.float64),
        "destroy_weights": np.empty((0, 4), dtype=np.float64),
        "repair_weights": np.empty((0, 3), dtype=np.float64),
        "destroy_names": [],
        "repair_names": [],
        "destroy_scores": {},
        "destroy_counts": {},
        "repair_scores": {},
        "repair_counts": {},
        "weight_snapshot_iters": np.array([], dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Mock Pareto archive
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_archive_5():
    return [
        {"cost": 500_000.0, "makespan": 480.0, "fitness": 900.0, "feasible": True},
        {"cost": 700_000.0, "makespan": 380.0, "fitness": 850.0, "feasible": True},
        {"cost": 900_000.0, "makespan": 320.0, "fitness": 820.0, "feasible": True},
        {"cost": 1_100_000.0, "makespan": 280.0, "fitness": 800.0, "feasible": True},
        {"cost": 1_300_000.0, "makespan": 250.0, "fitness": 790.0, "feasible": False},
    ]


@pytest.fixture
def empty_archive():
    return []
