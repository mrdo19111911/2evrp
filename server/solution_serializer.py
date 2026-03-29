"""Convert solver numpy solution tuple + eval array into JSON-serializable dicts.

Units preserved: meters (i64), seconds (i64), grams (i64), VND (i64).
"""
import numpy as np

from src.data.constants import (
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_BIKE_STOPS, SOL_BIKE_ACTIONS,
    SOL_TRUCK_LENGTHS, SOL_BIKE_LENGTHS,
    SOL_TRUCK_LOADS, SOL_BIKE_LOADS,
    SOL_TRUCK_DISTANCES, SOL_BIKE_DISTANCES,
    SOL_SATELLITES, SOL_META,
    META_N_TRUCKS, META_N_BIKES, META_N_CUSTOMERS, META_N_SATELLITES,
    ACT_DELIVER, ACT_RELOAD, ACT_PAD,
    SAT_CUST, SAT_BIKE, SAT_TRUCK, SAT_KG, SAT_TIME,
    COL_X, COL_Y, COL_DEMAND,
    EV_FITNESS, EV_COST, EV_SYNC_COST, EV_MAKESPAN,
    EV_TOTAL_PENALTY, EV_TOTAL_WAIT, EV_FEASIBLE,
)

_ACTION_NAMES = {int(ACT_DELIVER): "deliver", int(ACT_RELOAD): "reload",
                 int(ACT_PAD): "pad"}


def _serialize_routes(stops, actions, lengths, loads, distances, customers):
    """Convert route arrays into list of route dicts."""
    routes = []
    n_vehicles = len(lengths)
    for v in range(n_vehicles):
        length = int(lengths[v])
        if length == 0:
            continue
        route_stops = []
        for i in range(length):
            cid = int(stops[v, i])
            act = int(actions[v, i])
            stop = {"customer_id": cid,
                    "action": _ACTION_NAMES.get(act, str(act))}
            if 0 <= cid < len(customers):
                stop["x"] = int(customers[cid, COL_X])
                stop["y"] = int(customers[cid, COL_Y])
                stop["demand_g"] = int(customers[cid, COL_DEMAND])
            route_stops.append(stop)
        routes.append({
            "vehicle_id": int(v),
            "stops": route_stops,
            "load_g": int(loads[v]),
            "distance_m": int(distances[v]),
        })
    return routes


def _serialize_satellites(sat_array, n_satellites):
    """Convert satellite array into list of dicts."""
    sats = []
    for s in range(int(n_satellites)):
        sats.append({
            "customer_id": int(sat_array[s, SAT_CUST]),
            "bike_id": int(sat_array[s, SAT_BIKE]),
            "truck_id": int(sat_array[s, SAT_TRUCK]),
            "load_g": int(sat_array[s, SAT_KG]),
            "time_s": int(sat_array[s, SAT_TIME]),
        })
    return sats


def serialize_solution(sol, customers, depot):
    """Convert solution tuple + instance data -> JSON-serializable dict."""
    meta = sol[SOL_META]
    n_trucks = int(meta[META_N_TRUCKS])
    n_bikes = int(meta[META_N_BIKES])
    n_customers = int(meta[META_N_CUSTOMERS])
    n_sats = int(meta[META_N_SATELLITES])

    truck_routes = _serialize_routes(
        sol[SOL_TRUCK_STOPS], sol[SOL_TRUCK_ACTIONS],
        sol[SOL_TRUCK_LENGTHS], sol[SOL_TRUCK_LOADS],
        sol[SOL_TRUCK_DISTANCES], customers)

    bike_routes = _serialize_routes(
        sol[SOL_BIKE_STOPS], sol[SOL_BIKE_ACTIONS],
        sol[SOL_BIKE_LENGTHS], sol[SOL_BIKE_LOADS],
        sol[SOL_BIKE_DISTANCES], customers)

    satellites = _serialize_satellites(sol[SOL_SATELLITES], n_sats)

    return {
        "truck_routes": truck_routes,
        "bike_routes": bike_routes,
        "satellites": satellites,
        "depot": {"x": int(depot[0]), "y": int(depot[1])},
        "summary": {
            "n_trucks_used": len(truck_routes),
            "n_bikes_used": len(bike_routes),
            "n_customers": n_customers,
            "n_satellites": n_sats,
        },
    }


def serialize_eval(ev):
    """Convert eval result array -> JSON-serializable dict."""
    return {
        "fitness": int(ev[EV_FITNESS]),
        "cost": int(ev[EV_COST]),
        "sync_cost": int(ev[EV_SYNC_COST]),
        "makespan": int(ev[EV_MAKESPAN]),
        "total_penalty": int(ev[EV_TOTAL_PENALTY]),
        "total_wait": int(ev[EV_TOTAL_WAIT]),
        "feasible": bool(ev[EV_FEASIBLE]),
    }
