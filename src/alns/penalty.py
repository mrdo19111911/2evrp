"""Adaptive penalty weight scheduling."""
import numpy as np

from src.data.cost import (
    PENALTY_UNSERVED, PENALTY_LATE, PENALTY_OVERLOAD_KG,
    PENALTY_SYNC_FAIL, PENALTY_RESTRICTION,
)


def init_penalty_weights(config):
    """Initialize penalty weights + w3. Returns (penalty_weights, w3)."""
    penalty_weights = {
        "unserved": PENALTY_UNSERVED,
        "duplicate": PENALTY_UNSERVED,
        "capacity": PENALTY_OVERLOAD_KG,
        "time_window": PENALTY_LATE,
        "sync": PENALTY_SYNC_FAIL,
        "vehicle_restriction": PENALTY_RESTRICTION,
    }
    w3 = config["penalty_w3_start"]
    return penalty_weights, float(w3)


def decay_penalty_weight(w3, config):
    """Geometric decay per iteration."""
    decay_rate = config.get("penalty_decay_rate", 0.9999)
    w3_end = config.get("penalty_w3_end", 0.1)
    return max(w3 * decay_rate, w3_end)


def adaptive_penalty_adjustment(w3, feasible_history, config):
    """Feedback: increase w3 if too few feasible, decrease if stable."""
    feas_rate = float(np.mean(feasible_history))
    decay_rate = config.get("penalty_decay_rate", 0.9999)
    w3_start = config.get("penalty_w3_start", 10.0)
    w3_end = config.get("penalty_w3_end", 0.1)

    if feas_rate < 0.3:
        new_w3 = min(w3 * 2.0, w3_start)
    elif feas_rate > 0.8:
        new_w3 = w3 * (decay_rate ** 5)
    else:
        new_w3 = w3 * decay_rate

    return max(new_w3, w3_end)
