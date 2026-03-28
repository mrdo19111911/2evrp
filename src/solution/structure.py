"""Solution structure: creation, copy, index management."""
import numpy as np

from src.data.constants import ACT_DELIVER, ACT_PAD, VEH_TRUCK, VEH_BIKE


def create_solution(n_trucks, n_bikes, n_customers, max_route_len=100):
    """Create empty solution dict with customer index."""
    return {
        "truck_stops": np.full((n_trucks, max_route_len), -1, dtype=np.int32),
        "truck_actions": np.full((n_trucks, max_route_len), ACT_PAD, dtype=np.int8),
        "bike_stops": np.full((n_bikes, max_route_len), -1, dtype=np.int32),
        "bike_actions": np.full((n_bikes, max_route_len), ACT_PAD, dtype=np.int8),
        "satellites": np.empty((0, 5), dtype=np.float64),
        "truck_lengths": np.zeros(n_trucks, dtype=np.int32),
        "bike_lengths": np.zeros(n_bikes, dtype=np.int32),
        "truck_loads": np.zeros(n_trucks, dtype=np.float64),
        "bike_loads": np.zeros(n_bikes, dtype=np.float64),
        "truck_distances": np.zeros(n_trucks, dtype=np.float64),
        "bike_distances": np.zeros(n_bikes, dtype=np.float64),
        "cust_vehicle": np.full(n_customers, -1, dtype=np.int32),
        "cust_vtype": np.full(n_customers, -1, dtype=np.int8),
        "cust_route_pos": np.full(n_customers, -1, dtype=np.int32),
        "n_trucks": n_trucks,
        "n_bikes": n_bikes,
        "max_route_len": max_route_len,
    }


def copy_solution(sol):
    """Deep copy solution. Modify copy without affecting original."""
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sol.items()}


def rebuild_index(sol, n_customers):
    """Rebuild customer index from routes. Safety/debug function."""
    sol["cust_vehicle"][:] = -1
    sol["cust_vtype"][:] = -1
    sol["cust_route_pos"][:] = -1

    for t in range(sol["n_trucks"]):
        for i in range(sol["truck_lengths"][t]):
            c = sol["truck_stops"][t, i]
            if sol["truck_actions"][t, i] == ACT_DELIVER:
                sol["cust_vehicle"][c] = t
                sol["cust_vtype"][c] = VEH_TRUCK
                sol["cust_route_pos"][c] = i

    for b in range(sol["n_bikes"]):
        for i in range(sol["bike_lengths"][b]):
            c = sol["bike_stops"][b, i]
            if sol["bike_actions"][b, i] == ACT_DELIVER:
                sol["cust_vehicle"][c] = sol["n_trucks"] + b
                sol["cust_vtype"][c] = VEH_BIKE
                sol["cust_route_pos"][c] = i
