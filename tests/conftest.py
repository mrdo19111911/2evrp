"""Shared fixtures: tiny instances for all tests."""
import numpy as np
import pytest


@pytest.fixture
def tiny_instance():
    """5 customers, 1 truck, 2 bikes. Simplest testable instance.

    Layout (10x10 km):
        C2(10kg,hẻm)
        |
    C0(100kg) --- Depot --- C1(20kg)
        |
    C3(8kg)      C4(5kg)

    Depot at (5, 5).
    C0: (0,5)  demand=100kg  → truck only (>60)
    C1: (10,5) demand=20kg
    C2: (5,10) demand=10kg   → restricted (bike only)
    C3: (2,0)  demand=8kg
    C4: (8,0)  demand=5kg
    """
    customers = np.array([
        # x,    y,   demand, tw_open, tw_close, service_time
        [0.0,  5.0, 100.0,  0.0,     120.0,    15.0],   # C0: heavy
        [10.0, 5.0,  20.0,  60.0,    180.0,    5.0],    # C1: normal
        [5.0, 10.0,  10.0,  0.0,     240.0,    5.0],    # C2: restricted
        [2.0,  0.0,   8.0,  30.0,    150.0,    5.0],    # C3: normal
        [8.0,  0.0,   5.0,  0.0,     480.0,    5.0],    # C4: normal
    ], dtype=np.float64)

    restricted = np.array([0, 0, 1, 0, 0], dtype=np.int8)

    depot = np.array([5.0, 5.0], dtype=np.float64)

    vehicles = np.array([
        [0, 2000.0, 4.52, 25.0],   # truck 0
        [1,   60.0, 0.85, 20.0],   # bike 0
        [1,   60.0, 0.85, 20.0],   # bike 1
    ], dtype=np.float64)

    return {
        "customers": customers,
        "restricted": restricted,
        "depot": depot,
        "vehicles": vehicles,
        "n_trucks": 1,
        "n_bikes": 2,
        "n_customers": 5,
    }


@pytest.fixture
def tiny_dist_matrix(tiny_instance):
    """Precomputed distance matrix for tiny_instance. (6x6), index 0=depot."""
    depot = tiny_instance["depot"]
    custs = tiny_instance["customers"][:, :2]
    coords = np.vstack([depot, custs])  # (6, 2)
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dm = np.sqrt((diff ** 2).sum(axis=2))
    return dm


@pytest.fixture
def medium_instance():
    """20 customers, 2 trucks, 5 bikes. For operator / integration tests.

    Grid 5x4, spacing 10km. Some heavy, some restricted.
    """
    rng = np.random.default_rng(42)
    n = 20
    gx = np.arange(5).repeat(4)
    gy = np.tile(np.arange(4), 5)
    x = gx * 10.0
    y = gy * 10.0
    demands = rng.choice([5.0, 10.0, 15.0, 30.0, 80.0, 200.0], size=n,
                         p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05])
    tw_open = rng.uniform(0, 200, n)
    tw_close = tw_open + rng.uniform(60, 180, n)
    tw_close = np.minimum(tw_close, 480.0)
    service = np.where(demands > 60, 15.0, 5.0)

    customers = np.column_stack([x, y, demands, tw_open, tw_close, service])
    restricted = np.zeros(n, dtype=np.int8)
    restricted[[3, 7, 11, 15]] = 1  # 4 restricted nodes

    depot = np.array([20.0, 15.0])
    vehicles = np.array([
        [0, 2000.0, 4.52, 25.0],
        [0, 2000.0, 4.52, 25.0],
        [1,   60.0, 0.85, 20.0],
        [1,   60.0, 0.85, 20.0],
        [1,   60.0, 0.85, 20.0],
        [1,   60.0, 0.85, 20.0],
        [1,   60.0, 0.85, 20.0],
    ], dtype=np.float64)

    return {
        "customers": customers,
        "restricted": restricted,
        "depot": depot,
        "vehicles": vehicles,
        "n_trucks": 2,
        "n_bikes": 5,
        "n_customers": n,
    }
