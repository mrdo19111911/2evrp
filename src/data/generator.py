"""Random and clustered test instance generation. All outputs i64."""
import numpy as np

from .constants import CUST_COLS, kmh_to_us_per_m
from .cost import (
    TRUCK_COST_PER_M, BIKE_COST_PER_M,
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M,
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
)
from .io import SERVICE_BASE_S, SERVICE_PER_G


def generate_random_instance(n_customers, n_trucks, n_bikes,
                             area_size_m=50000, demand_range_g=(3000, 120000),
                             seed=42):
    """Generate random instance. Returns (customers_i64, depot_i64, vehicles_i64)."""
    rng = np.random.default_rng(seed)
    depot = np.array([area_size_m // 2, area_size_m // 2], dtype=np.int64)
    customers = _make_customers(rng, n_customers, area_size_m, demand_range_g)
    restricted = _make_restricted(rng, n_customers, frac=0.15)
    customers[:, 6] = restricted
    vehicles = _make_vehicles(n_trucks, n_bikes)
    return customers, depot, vehicles


def generate_clustered_instance(n_customers, n_trucks, n_bikes,
                                n_clusters=5, cluster_std_m=3000,
                                area_size_m=50000, seed=42):
    """Generate clustered instance. Returns (customers_i64, depot_i64, vehicles_i64)."""
    rng = np.random.default_rng(seed)
    depot = np.array([area_size_m // 2, area_size_m // 2], dtype=np.int64)

    centers = rng.integers(cluster_std_m, area_size_m - cluster_std_m,
                           (n_clusters, 2))
    labels = rng.integers(0, n_clusters, n_customers)
    xy_f = centers[labels].astype(np.float64) + rng.normal(
        0, cluster_std_m, (n_customers, 2))
    xy = np.clip(xy_f, 0, area_size_m).astype(np.int64)

    customers = _make_customers_from_xy(rng, xy, demand_range_g=(3000, 120000))
    restricted = _make_restricted(rng, n_customers, frac=0.15)
    customers[:, 6] = restricted
    vehicles = _make_vehicles(n_trucks, n_bikes)
    return customers, depot, vehicles


# --- Internal helpers ---

def _make_customers(rng, n, area_size_m, demand_range_g):
    xy = rng.integers(0, area_size_m, (n, 2))
    return _make_customers_from_xy(rng, xy, demand_range_g)


def _make_customers_from_xy(rng, xy, demand_range_g, day_length_s=28800):
    n = len(xy)
    out = np.zeros((n, CUST_COLS), dtype=np.int64)
    out[:, 0] = xy[:, 0]
    out[:, 1] = xy[:, 1]

    demands_g = rng.integers(demand_range_g[0], demand_range_g[1], n)
    out[:, 2] = demands_g

    tw_open_s = rng.integers(0, int(day_length_s * 0.7), n)
    tw_width_s = rng.integers(1800, 10800, n)
    tw_close_s = np.minimum(tw_open_s + tw_width_s, day_length_s)
    tight = tw_close_s - tw_open_s < 60
    tw_close_s[tight] = tw_open_s[tight] + 60
    out[:, 3] = tw_open_s
    out[:, 4] = tw_close_s

    for i in range(n):
        out[i, 5] = int(SERVICE_BASE_S + SERVICE_PER_G * demands_g[i])

    out[:, 6] = 0  # restricted filled by caller
    return out


def _make_restricted(rng, n, frac=0.15):
    restricted = np.zeros(n, dtype=np.int64)
    k = max(1, int(n * frac))
    idx = rng.choice(n, size=k, replace=False)
    restricted[idx] = 1
    return restricted


def _make_vehicles(n_trucks, n_bikes):
    rows = []
    for _ in range(n_trucks):
        rows.append([0, TRUCK_CAPACITY_G, TRUCK_COST_PER_M,
                     TRUCK_SPEED_US_PER_M])
    for _ in range(n_bikes):
        rows.append([1, BIKE_CAPACITY_G, BIKE_COST_PER_M,
                     BIKE_SPEED_US_PER_M])
    return np.array(rows, dtype=np.int64)
