"""Random and clustered test instance generation."""
import numpy as np


def generate_random_instance(n_customers, n_trucks, n_bikes, area_size=50.0,
                             demand_range=(3.0, 120.0), seed=42):
    """Generate random instance. Returns (customers, restricted, depot, vehicles)."""
    rng = np.random.default_rng(seed)
    depot = np.array([area_size / 2, area_size / 2], dtype=np.float64)
    customers = _make_customers(rng, n_customers, area_size, demand_range)
    restricted = _make_restricted(rng, n_customers, frac=0.15)
    vehicles = _make_vehicles(n_trucks, n_bikes)
    return customers, restricted, depot, vehicles


def generate_clustered_instance(n_customers, n_trucks, n_bikes, n_clusters=5,
                                cluster_std=3.0, area_size=50.0, seed=42):
    """Generate clustered instance. Returns (customers, restricted, depot, vehicles)."""
    rng = np.random.default_rng(seed)
    depot = np.array([area_size / 2, area_size / 2], dtype=np.float64)

    centers = rng.uniform(cluster_std, area_size - cluster_std, (n_clusters, 2))
    labels = rng.integers(0, n_clusters, n_customers)
    xy = centers[labels] + rng.normal(0, cluster_std, (n_customers, 2))
    xy = np.clip(xy, 0.0, area_size)

    customers = _make_customers_from_xy(rng, xy, demand_range=(3.0, 120.0))
    restricted = _make_restricted(rng, n_customers, frac=0.15)
    vehicles = _make_vehicles(n_trucks, n_bikes)
    return customers, restricted, depot, vehicles


# --- Internal helpers ---

def _make_customers(rng, n, area_size, demand_range):
    xy = rng.uniform(0, area_size, (n, 2))
    return _make_customers_from_xy(rng, xy, demand_range)


def _make_customers_from_xy(rng, xy, demand_range, day_length=480.0):
    n = len(xy)
    demands = rng.uniform(demand_range[0], demand_range[1], n)
    tw_open = rng.uniform(0, day_length * 0.7, n)
    tw_width = rng.uniform(30.0, 180.0, n)
    tw_close = np.minimum(tw_open + tw_width, day_length)
    # Ensure tw_open < tw_close (min width 1 minute)
    tight = tw_close - tw_open < 1.0
    tw_close[tight] = tw_open[tight] + 1.0
    service = np.where(demands > 60, 15.0, 5.0)
    customers = np.column_stack([xy, demands, tw_open, tw_close, service])
    return customers.astype(np.float64)


def _make_restricted(rng, n, frac=0.15):
    restricted = np.zeros(n, dtype=np.int8)
    k = max(1, int(n * frac))
    idx = rng.choice(n, size=k, replace=False)
    restricted[idx] = 1
    return restricted


def _make_vehicles(n_trucks, n_bikes):
    rows = []
    for _ in range(n_trucks):
        rows.append([0, 2000.0, 4.52, 25.0])
    for _ in range(n_bikes):
        rows.append([1, 60.0, 0.85, 20.0])
    return np.array(rows, dtype=np.float64)
