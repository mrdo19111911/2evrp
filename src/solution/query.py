"""O(1) lookups and solution queries."""
import numpy as np

from src.data.constants import ACT_DELIVER, VEH_TRUCK, VEH_BIKE


def _get_route_arrays(sol, vtype):
    if vtype == VEH_TRUCK:
        return sol["truck_stops"], sol["truck_actions"], sol["truck_lengths"]
    return sol["bike_stops"], sol["bike_actions"], sol["bike_lengths"]


def get_unassigned_customers(sol, n_customers):
    """Customers not yet delivered. O(N) vectorized."""
    return np.where(sol["cust_vehicle"][:n_customers] == -1)[0].astype(np.int32)


def get_assigned_customers(sol, n_customers):
    """Customers already delivered. O(N) vectorized."""
    return np.where(sol["cust_vehicle"][:n_customers] >= 0)[0].astype(np.int32)


def get_route_as_list(sol, vtype, vid):
    """Route as [(customer, action), ...]. For debug."""
    stops, actions, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    return [(int(stops[vid, i]), int(actions[vid, i])) for i in range(L)]


def get_route_customers_only(sol, vtype, vid):
    """Only DELIVER customers in route. ndarray int32."""
    stops, actions, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    if L == 0:
        return np.array([], dtype=np.int32)
    mask = actions[vid, :L] == ACT_DELIVER
    return stops[vid, :L][mask].astype(np.int32)


def is_customer_assigned(sol, customer):
    """O(1) check."""
    return bool(sol["cust_vehicle"][customer] >= 0)


def get_customer_info(sol, customer):
    """O(1) lookup. Returns (vtype, vid, pos) or (-1,-1,-1)."""
    veh = sol["cust_vehicle"][customer]
    if veh == -1:
        return (-1, -1, -1)
    vtype = int(sol["cust_vtype"][customer])
    pos = int(sol["cust_route_pos"][customer])
    vid = int(veh) if vtype == VEH_TRUCK else int(veh) - sol["n_trucks"]
    return (vtype, vid, pos)
