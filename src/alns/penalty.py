"""Adaptive penalty weight scheduling — penalty_w is i64 VND array."""
import numpy as np
from numba import njit

from src.data.constants import (
    PW_UNSERVED, PW_DUPLICATE, PW_CAPACITY, PW_TIME_WINDOW, PW_SYNC, PW_VEHICLE_RESTRICTION,
    PW_SIZE,
    CFG_PENALTY_W3_START, CFG_PENALTY_W3_END, CFG_PENALTY_DECAY_RATE,
)
from src.data.cost import (
    PENALTY_UNSERVED, PENALTY_LATE, PENALTY_OVERLOAD_PER_PCT,
    PENALTY_SYNC_FAIL, PENALTY_RESTRICTION,
)


def init_penalty_weights(config):
    """Initialize penalty weights + w3. Returns (pw i64 array, w3 scalar f64)."""
    pw = np.zeros(PW_SIZE, dtype=np.int64)
    pw[PW_UNSERVED] = PENALTY_UNSERVED
    pw[PW_DUPLICATE] = PENALTY_UNSERVED
    pw[PW_CAPACITY] = PENALTY_OVERLOAD_PER_PCT
    pw[PW_TIME_WINDOW] = PENALTY_LATE
    pw[PW_SYNC] = PENALTY_SYNC_FAIL
    pw[PW_VEHICLE_RESTRICTION] = PENALTY_RESTRICTION
    w3 = float(config[CFG_PENALTY_W3_START])
    return pw, w3


@njit(cache=True)
def adaptive_penalty_adjustment(w3, feasible_history, config):
    """Feedback: increase w3 if too few feasible, decrease if stable."""
    n = len(feasible_history)
    if n == 0:
        return w3
    any_feas = False
    feas_sum = 0.0
    for i in range(n):
        if feasible_history[i]:
            any_feas = True
            feas_sum += 1.0
    if not any_feas:
        return w3
    feas_rate = feas_sum / n
    decay_rate = config[CFG_PENALTY_DECAY_RATE]
    w3_start = config[CFG_PENALTY_W3_START]
    w3_end = config[CFG_PENALTY_W3_END]

    if feas_rate < 0.3:
        new_w3 = min(w3 * 2.0, w3_start)
    elif feas_rate > 0.8:
        new_w3 = w3 * (decay_rate ** 5)
    else:
        new_w3 = w3 * decay_rate

    return max(new_w3, w3_end)
