"""Sync cost between truck and bike at satellite nodes."""
import numpy as np

from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD,
    ST_ACTION, ST_CUST, ST_DEPART, ST_ARRIVE,
    SAT_CUST, SAT_BIKE, SAT_TRUCK,
)
from src.data.cost import PENALTY_MISSING_RELOAD, PENALTY_SYNC_GAP_MIN


def compute_sync_cost(truck_states, bike_states, satellites, delta_t):
    """Sync cost in VND. Returns (total_sync_cost, breakdown_list)."""
    total = 0.0
    breakdown = []

    for s in range(len(satellites)):
        cust = int(satellites[s, SAT_CUST])
        bike_id = int(satellites[s, SAT_BIKE])
        truck_id = int(satellites[s, SAT_TRUCK])

        truck_depart = _find_truck_depart(truck_states, truck_id, cust)
        if truck_depart is None:
            total += PENALTY_MISSING_RELOAD
            breakdown.append({"sat": s, "cust": cust, "type": "missing",
                              "cost": PENALTY_MISSING_RELOAD})
            continue

        bike_arrive = _find_bike_arrive(bike_states, bike_id, cust)
        if bike_arrive is None:
            total += PENALTY_MISSING_RELOAD
            breakdown.append({"sat": s, "cust": cust, "type": "missing",
                              "cost": PENALTY_MISSING_RELOAD})
            continue

        gap = abs(truck_depart - bike_arrive)
        if gap > delta_t:
            cost = PENALTY_SYNC_GAP_MIN * (gap - delta_t)
            total += cost
            breakdown.append({"sat": s, "cust": cust, "type": "gap",
                              "gap": gap, "cost": cost})
        else:
            breakdown.append({"sat": s, "cust": cust, "type": "ok",
                              "gap": gap, "cost": 0.0})

    return total, breakdown


def _find_truck_depart(truck_states, truck_id, cust):
    """Find truck RELOAD depart time at satellite node."""
    if truck_id >= len(truck_states):
        return None
    t_state = truck_states[truck_id]
    if len(t_state) == 0:
        return None
    t_mask = (t_state[:, ST_CUST] == cust) & (t_state[:, ST_ACTION] == ACT_RELOAD)
    if not np.any(t_mask):
        return None
    return float(t_state[t_mask][0, ST_DEPART])


def _find_bike_arrive(bike_states, bike_id, cust):
    """Find bike RELOAD arrive time at customer node."""
    if bike_id >= len(bike_states):
        return None
    b_state = bike_states[bike_id]
    if len(b_state) == 0:
        return None
    b_mask = (b_state[:, ST_CUST] == cust) & (b_state[:, ST_ACTION] == ACT_RELOAD)
    if not np.any(b_mask):
        return None
    return float(b_state[b_mask][0, ST_ARRIVE])
