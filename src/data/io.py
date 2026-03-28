"""Load/save problem instances and solutions."""
import json
import numpy as np


def load_instance(filepath):
    """Load problem instance from JSON. Returns (customers, restricted, depot, vehicles)."""
    with open(filepath, "r") as f:
        data = json.load(f)

    customers, restricted = _parse_customers(data)
    depot = _parse_depot(data)
    vehicles = _parse_vehicles(data)
    _validate_instance(customers, restricted)

    return customers, restricted, depot, vehicles


def save_solution(filepath, truck_stops, truck_actions, bike_stops, bike_actions, satellites):
    """Save solution to JSON."""
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
    """Load solution from JSON. Returns (truck_stops, truck_actions, bike_stops, bike_actions, satellites)."""
    with open(filepath, "r") as f:
        data = json.load(f)

    truck_stops = np.array(data["truck_stops"], dtype=np.int32)
    truck_actions = np.array(data["truck_actions"], dtype=np.int32)
    bike_stops = np.array(data["bike_stops"], dtype=np.int32)
    bike_actions = np.array(data["bike_actions"], dtype=np.int32)
    sats = data["satellites"]
    satellites = np.array(sats, dtype=np.float64) if len(sats) > 0 else np.empty((0, 5), dtype=np.float64)

    return truck_stops, truck_actions, bike_stops, bike_actions, satellites


# --- Internal helpers ---

def _parse_customers(data):
    """Parse customers from structured (list of dicts) or flat (list of lists) format."""
    raw = data["customers"]
    if len(raw) == 0:
        return np.empty((0, 6), dtype=np.float64), np.empty(0, dtype=np.int8)

    if isinstance(raw[0], dict):
        rows = [[c["x"], c["y"], c["demand_kg"], c["tw_open"], c["tw_close"], c["service_time"]]
                for c in raw]
        restricted = np.array([int(c.get("restricted", False)) for c in raw], dtype=np.int8)
    else:
        rows = [r[:6] for r in raw]
        restricted_raw = data.get("restricted", [0] * len(raw))
        restricted = np.array(restricted_raw, dtype=np.int8)

    customers = np.array(rows, dtype=np.float64)
    return customers, restricted


def _parse_depot(data):
    """Parse depot from dict or list format."""
    raw = data["depot"]
    if isinstance(raw, dict):
        return np.array([raw["x"], raw["y"]], dtype=np.float64)
    return np.array(raw, dtype=np.float64)


def _parse_vehicles(data):
    """Parse vehicles from structured or flat format."""
    raw = data.get("vehicles", {})
    if isinstance(raw, dict):
        rows = []
        for t in raw.get("trucks", []):
            rows.append([0, t["capacity_kg"], t["cost_per_km"], t["speed_kmh"]])
        for b in raw.get("bikes", []):
            rows.append([1, b["capacity_kg"], b["cost_per_km"], b["speed_kmh"]])
        return np.array(rows, dtype=np.float64) if rows else np.empty((0, 4), dtype=np.float64)
    return np.array(raw, dtype=np.float64)


def _validate_instance(customers, restricted):
    """Validate: demand > 0, tw_open <= tw_close."""
    if len(customers) == 0:
        return
    demands = customers[:, 2]
    if np.any(demands <= 0):
        raise ValueError(f"All demands must be > 0, found min={demands.min()}")
    tw_open = customers[:, 3]
    tw_close = customers[:, 4]
    if np.any(tw_open > tw_close):
        bad = np.where(tw_open > tw_close)[0]
        raise ValueError(f"tw_open > tw_close at customers {bad.tolist()}")
