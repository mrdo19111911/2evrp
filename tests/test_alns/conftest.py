"""Shared fixtures for ALNS operator tests — Data Model v2.

Provides manually-constructed solution dicts with concrete numpy arrays
so that operator tests have deterministic, inspectable inputs.
"""
import numpy as np
import pytest

from src.data.constants import (
    ACT_DELIVER, ACT_PICKUP, ACT_PAD,
    VTYPE_TRUCK, VTYPE_BIKE,
    LOC_X, LOC_Y, LOC_TYPE, LOC_MAX_WEIGHT, LOC_UNLOAD_RATE, LOC_BASE_SVC, LOC_COLS,
    LTYPE_DEPOT, LTYPE_CUSTOMER,
    ORD_LOC, ORD_SKU, ORD_QTY, ORD_UNIT_W, ORD_UNIT_CBM, ORD_COLS,
    VEH_TYPE, VEH_CAP_KG, VEH_CAP_CBM, VEH_COST_KM, VEH_START, VEH_END, VEH_COLS,
    DM_DIST, DM_TIME,
    TR_HUB, TR_BIKE, TR_TRUCK, TR_KG, TR_CBM, TR_TIME, TR_COLS,
)
from src.solution.structure import create_solution


def _build_dist_matrix(coords, speed_kmh=25.0):
    """Build (2, N, N) dist matrix from (N, 2) coords."""
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_km = np.sqrt((diff ** 2).sum(axis=2))
    time_min = dist_km / speed_kmh * 60.0
    return np.stack([dist_km, time_min], axis=0)


def _make_data(n_customers, loc_coords, demands_kg, n_trucks, n_bikes,
               allowed_bike=None, depot_xy=(20.0, 5.0)):
    """Build a v2 data dict from simple parameters."""
    n_veh = n_trucks + n_bikes
    n_locs = 1 + n_customers
    n_sku = 2

    locations = np.zeros((n_locs, LOC_COLS), dtype=np.float64)
    locations[0] = [depot_xy[0], depot_xy[1], LTYPE_DEPOT, 99999.0, 20.0, 10.0]
    for i in range(n_customers):
        x, y = loc_coords[i]
        locations[i + 1] = [x, y, LTYPE_CUSTOMER, 99999.0, 20.0, 5.0]

    orders = np.zeros((n_customers, ORD_COLS), dtype=np.float64)
    for i in range(n_customers):
        d = demands_kg[i]
        orders[i, ORD_LOC] = i + 1
        orders[i, ORD_SKU] = 0
        orders[i, ORD_QTY] = max(1, int(d / 5.0))
        orders[i, ORD_UNIT_W] = d / max(1, orders[i, ORD_QTY])
        orders[i, ORD_UNIT_CBM] = 0.01

    vehicles = np.zeros((n_veh, VEH_COLS), dtype=np.float64)
    for i in range(n_trucks):
        vehicles[i] = [VTYPE_TRUCK, 2000.0, 10.0, 4.52, 0, 0]
    for i in range(n_trucks, n_veh):
        vehicles[i] = [VTYPE_BIKE, 60.0, 0.5, 0.85, 0, 0]

    if allowed_bike is None:
        allowed_bike = np.ones(n_customers, dtype=np.int8)
    else:
        allowed_bike = np.array(allowed_bike, dtype=np.int8)

    coords = locations[:, :2]
    dist_matrix = _build_dist_matrix(coords)

    loc_to_cust = np.full(n_locs, -1, dtype=np.int32)
    for c in range(n_customers):
        loc_to_cust[int(orders[c, ORD_LOC])] = c

    return {
        "locations": locations,
        "orders": orders,
        "vehicles": vehicles,
        "dist_matrix": dist_matrix,
        "allowed_bike": allowed_bike,
        "loc_to_cust": loc_to_cust,
        "n_customers": n_customers,
        "n_vehicles": n_veh,
        "n_sku": n_sku,
        "max_working_time": np.full(n_veh, 420.0, dtype=np.float64),
        "temp_requirements": np.zeros((n_veh, n_sku), dtype=np.int8),
        "demand_per_sku": np.zeros((n_customers, n_sku), dtype=np.float64),
    }


def _assign_customer(sol, v, pos, customer, demand_kg, orders, vehicles, dist_matrix):
    """Place a DELIVER stop and update index."""
    from src.solution.route_ops import insert_stop
    loc_id = int(orders[customer, ORD_LOC])
    insert_stop(sol, v, pos, loc_id, ACT_DELIVER, dist_matrix, vehicles, orders,
                customer=customer)


def _add_pickup(sol, v, pos, loc_id, dist_matrix, vehicles, orders):
    """Insert a PICKUP (transfer) stop; no customer index update."""
    from src.solution.route_ops import insert_stop
    insert_stop(sol, v, pos, loc_id, ACT_PICKUP, dist_matrix, vehicles, orders,
                customer=-1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sol_10_assigned():
    """Solution with 10 customers, all assigned. Unified v2 format."""
    N = 10
    loc_coords = [
        (0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (40.0, 0.0),
        (0.0, 10.0), (10.0, 10.0), (20.0, 10.0), (30.0, 10.0), (40.0, 10.0),
    ]
    demands = [50.0, 40.0, 30.0, 25.0, 20.0, 15.0, 10.0, 8.0, 5.0, 3.0]

    data = _make_data(N, loc_coords, demands, n_trucks=1, n_bikes=2,
                      depot_xy=(20.0, 5.0))
    sol = create_solution(data["n_vehicles"], N, n_sku=data["n_sku"])
    orders = data["orders"]
    vehicles = data["vehicles"]
    dm = data["dist_matrix"]

    # Truck 0 (v=0): C0-C5
    for pos, c in enumerate([0, 1, 2, 3, 4, 5]):
        _assign_customer(sol, 0, pos, c, demands[c], orders, vehicles, dm)
    # Bike 0 (v=1): C6, C7
    for pos, c in enumerate([6, 7]):
        _assign_customer(sol, 1, pos, c, demands[c], orders, vehicles, dm)
    # Bike 1 (v=2): C8, C9
    for pos, c in enumerate([8, 9]):
        _assign_customer(sol, 2, pos, c, demands[c], orders, vehicles, dm)

    return {"sol": sol, "data": data, "N": N}


@pytest.fixture
def sol_crossing_route():
    """Solution with a single truck route whose stops form a crossing pattern."""
    N = 4
    loc_coords = [(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)]
    demands = [10.0, 10.0, 10.0, 10.0]

    data = _make_data(N, loc_coords, demands, n_trucks=1, n_bikes=0,
                      depot_xy=(5.0, 5.0))
    sol = create_solution(data["n_vehicles"], N, n_sku=data["n_sku"])
    orders = data["orders"]
    vehicles = data["vehicles"]
    dm = data["dist_matrix"]

    for pos, c in enumerate([0, 1, 2, 3]):
        _assign_customer(sol, 0, pos, c, 10.0, orders, vehicles, dm)

    return {"sol": sol, "data": data, "N": N}


@pytest.fixture
def sol_with_transfer():
    """Solution with 1 transfer (hub at C2 location).

    Truck 0: C0 -> hub(PICKUP) -> C1
    Bike 0:  hub(PICKUP) -> C3 -> C4
    Transfer: hub=loc3, bike=1, truck=0, kg=13, time=50
    """
    N = 5
    loc_coords = [
        (0.0, 0.0), (20.0, 0.0), (10.0, 5.0), (8.0, 8.0), (12.0, 8.0),
    ]
    demands = [100.0, 80.0, 10.0, 8.0, 5.0]

    data = _make_data(N, loc_coords, demands, n_trucks=1, n_bikes=1,
                      allowed_bike=[0, 0, 1, 1, 1], depot_xy=(10.0, 0.0))
    sol = create_solution(data["n_vehicles"], N, n_sku=data["n_sku"])
    orders = data["orders"]
    vehicles = data["vehicles"]
    dm = data["dist_matrix"]

    # Truck 0 (v=0): C0(DELIVER) -> hub_loc=3(PICKUP) -> C1(DELIVER)
    _assign_customer(sol, 0, 0, 0, 100.0, orders, vehicles, dm)
    hub_loc = int(orders[2, ORD_LOC])  # loc 3
    _add_pickup(sol, 0, 1, hub_loc, dm, vehicles, orders)
    _assign_customer(sol, 0, 2, 1, 80.0, orders, vehicles, dm)

    # Bike 0 (v=1): hub_loc(PICKUP) -> C3 -> C4
    _add_pickup(sol, 1, 0, hub_loc, dm, vehicles, orders)
    _assign_customer(sol, 1, 1, 3, 8.0, orders, vehicles, dm)
    _assign_customer(sol, 1, 2, 4, 5.0, orders, vehicles, dm)

    # Transfer event
    from src.solution.transfer_ops import add_transfer_event
    add_transfer_event(sol, hub_loc=hub_loc, bike_id=1, truck_id=0,
                       kg=13.0, cbm=0.01, time=50.0)

    return {"sol": sol, "data": data, "N": N}


@pytest.fixture
def default_params():
    """Default ALNS params for destroy/repair operators."""
    return {
        "q_min": 3,
        "q_max": 5,
        "q_max_pct": 0.15,
        "shaw_randomness": 6.0,
        "worst_noise": 0.1,
        "zone_pct": 15,
        "regret_k": 3,
        "transfer_threshold_km": 5.0,
    }
