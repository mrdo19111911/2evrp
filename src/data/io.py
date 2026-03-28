"""Load/save problem instances and solutions. All outputs are i64/i32/i8."""
import json
import numpy as np

from .constants import (
    COL_X, COL_Y, COL_DEMAND, COL_TW_OPEN, COL_TW_CLOSE,
    COL_SERVICE, COL_RESTRICTED, CUST_COLS,
    VCOL_TYPE, VCOL_CAPACITY, VCOL_COST_M, VCOL_SPEED,
    kmh_to_us_per_m,
)
from .cost import (
    TRUCK_COST_PER_M, BIKE_COST_PER_M,
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M,
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
)

# Service time constants (cold path only, used at load time)
SERVICE_BASE_S = 600        # 10 min base
SERVICE_PER_G = 0.003       # 5 min / 100 kg = 300s / 100000g


def load_instance(filepath):
    """Load JSON -> (customers_i64, depot_i64, vehicles_i64)."""
    with open(filepath, "r") as f:
        data = json.load(f)

    customers = _parse_customers(data)
    depot = _parse_depot(data)
    vehicles = _parse_vehicles(data)
    _validate(customers)
    return customers, depot, vehicles


def save_solution(filepath, truck_stops, truck_actions, bike_stops,
                  bike_actions, satellites):
    """Save solution arrays to JSON."""
    output = {
        "truck_stops": truck_stops.tolist(),
        "truck_actions": truck_actions.tolist(),
        "bike_stops": bike_stops.tolist(),
        "bike_actions": bike_actions.tolist(),
        "satellites": satellites.tolist(),
    }
    with open(filepath, "w") as f:
        json.dump(output, f)


def load_solution(filepath):
    """Load solution from JSON."""
    with open(filepath, "r") as f:
        data = json.load(f)

    truck_stops = np.array(data["truck_stops"], dtype=np.int32)
    truck_actions = np.array(data["truck_actions"], dtype=np.int32)
    bike_stops = np.array(data["bike_stops"], dtype=np.int32)
    bike_actions = np.array(data["bike_actions"], dtype=np.int32)
    sats = data["satellites"]
    satellites = (np.array(sats, dtype=np.int64) if len(sats) > 0
                  else np.empty((0, 5), dtype=np.int64))
    return truck_stops, truck_actions, bike_stops, bike_actions, satellites


# --- Internal helpers ---

def _parse_customers(data):
    """Parse customers -> i64 (N, 7). Units: m, g, s."""
    raw = data["customers"]
    if len(raw) == 0:
        return np.empty((0, CUST_COLS), dtype=np.int64)

    if isinstance(raw[0], dict):
        return _parse_customers_dict(raw)
    return _parse_customers_flat(raw, data)


def _parse_customers_dict(raw):
    """Dict-format customers -> i64 (N, 7)."""
    n = len(raw)
    out = np.zeros((n, CUST_COLS), dtype=np.int64)
    for i, c in enumerate(raw):
        x_m = int(c["x"] * 1000)
        y_m = int(c["y"] * 1000)
        demand_g = int(c["demand_kg"] * 1000)
        tw_open_s = int(c["tw_open"] * 60)
        tw_close_s = int(c["tw_close"] * 60)
        service_s = int(SERVICE_BASE_S + SERVICE_PER_G * demand_g)
        restricted = int(c.get("restricted", False))
        out[i] = [x_m, y_m, demand_g, tw_open_s, tw_close_s,
                  service_s, restricted]
    return out


def _parse_customers_flat(raw, data):
    """Flat-format customers -> i64 (N, 7)."""
    n = len(raw)
    restricted_raw = data.get("restricted", [0] * n)
    out = np.zeros((n, CUST_COLS), dtype=np.int64)
    for i, r in enumerate(raw):
        x_m = int(r[0] * 1000)
        y_m = int(r[1] * 1000)
        demand_g = int(r[2] * 1000)
        tw_open_s = int(r[3] * 60)
        tw_close_s = int(r[4] * 60)
        service_s = int(SERVICE_BASE_S + SERVICE_PER_G * demand_g)
        restricted = int(restricted_raw[i])
        out[i] = [x_m, y_m, demand_g, tw_open_s, tw_close_s,
                  service_s, restricted]
    return out


def _parse_depot(data):
    """Parse depot -> i64 (2,) in meters."""
    raw = data["depot"]
    if isinstance(raw, dict):
        return np.array([int(raw["x"] * 1000), int(raw["y"] * 1000)],
                        dtype=np.int64)
    return np.array([int(raw[0] * 1000), int(raw[1] * 1000)], dtype=np.int64)


def _parse_vehicles(data):
    """Parse vehicles -> i64 (K, 4): [type, cap_g, cost_m, speed_us_m]."""
    raw = data.get("vehicles", {})
    if isinstance(raw, dict):
        return _parse_vehicles_dict(raw)
    return _parse_vehicles_flat(raw)


def _parse_vehicles_dict(raw):
    """Dict-format vehicles -> i64 (K, 4)."""
    rows = []
    for t in raw.get("trucks", []):
        spd = kmh_to_us_per_m(t["speed_kmh"])
        cost_m = int(t.get("cost_per_km", TRUCK_COST_PER_M * 1000) / 1000)
        if cost_m == 0:
            cost_m = TRUCK_COST_PER_M
        cap_g = int(t["capacity_kg"] * 1000)
        rows.append([0, cap_g, cost_m, spd])
    for b in raw.get("bikes", []):
        spd = kmh_to_us_per_m(b["speed_kmh"])
        cost_m = int(b.get("cost_per_km", BIKE_COST_PER_M * 1000) / 1000)
        if cost_m == 0:
            cost_m = BIKE_COST_PER_M
        cap_g = int(b["capacity_kg"] * 1000)
        rows.append([1, cap_g, cost_m, spd])
    if not rows:
        return np.empty((0, 4), dtype=np.int64)
    return np.array(rows, dtype=np.int64)


def _parse_vehicles_flat(raw):
    """Flat-format vehicles -> i64 (K, 4)."""
    if not raw:
        return np.empty((0, 4), dtype=np.int64)
    rows = []
    for r in raw:
        vtype = int(r[0])
        cap_g = int(r[1] * 1000)
        cost_m = int(r[2]) if r[2] > 100 else int(r[2] * 1000)
        spd = kmh_to_us_per_m(r[3]) if r[3] > 1000 else int(r[3])
        rows.append([vtype, cap_g, cost_m, spd])
    return np.array(rows, dtype=np.int64)


def _validate(customers):
    """Validate: demand > 0, tw_open <= tw_close."""
    if len(customers) == 0:
        return
    demands = customers[:, COL_DEMAND]
    if np.any(demands <= 0):
        raise ValueError(f"All demands must be > 0, found min={demands.min()}")
    tw_open = customers[:, COL_TW_OPEN]
    tw_close = customers[:, COL_TW_CLOSE]
    if np.any(tw_open > tw_close):
        bad = np.where(tw_open > tw_close)[0]
        raise ValueError(f"tw_open > tw_close at customers {bad.tolist()}")
