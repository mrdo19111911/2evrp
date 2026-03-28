"""ALNS hyperparameters — float64 array with named index constants.
Config stays f64 (tuning knobs). CFG_DELTA_T in seconds (i64 value stored as f64).
"""
import numpy as np

from src.data.constants import (
    CFG_MAX_ITERATIONS, CFG_SEGMENT_LENGTH, CFG_NO_IMPROVE_LIMIT,
    CFG_SA_INITIAL_TEMP, CFG_SA_COOLING_RATE, CFG_SA_MIN_TEMP,
    CFG_REACTION_FACTOR,
    CFG_SIGMA_1, CFG_SIGMA_2, CFG_SIGMA_3,
    CFG_PENALTY_W3_START, CFG_PENALTY_W3_END, CFG_PENALTY_DECAY_RATE,
    CFG_LS_FREQUENCY, CFG_LS_ON_NEW_BEST, CFG_CROSS_FREQUENCY,
    CFG_Q_MIN, CFG_Q_MAX_PCT, CFG_Q_MAX,
    CFG_SHAW_RANDOMNESS, CFG_WORST_NOISE, CFG_ZONE_PCT,
    CFG_REGRET_K, CFG_REGRET_NOISE, CFG_SAT_THRESHOLD_KM,
    CFG_REHEAT_THRESHOLD_PCT, CFG_REHEAT_TEMP_PCT,
    CFG_ELITE_POOL_SIZE, CFG_ELITE_MIN_HAMMING_PCT,
    CFG_RESTART_THRESHOLD_PCT, CFG_MAX_RESTARTS,
    CFG_QL_ENABLED, CFG_QL_ALPHA, CFG_QL_GAMMA,
    CFG_QL_EPSILON, CFG_QL_EPSILON_DECAY, CFG_QL_EPSILON_MIN,
    CFG_DELTA_T, CFG_BLINK_PROB,
    CFG_SISR_MAX_STRING_LEN, CFG_SISR_MAX_ROUTES,
    CFG_HISTORY_NOISE,
    CFG_SIZE,
)
from src.data.cost import SYNC_DELTA_T


def make_config(n_customers, overrides=None):
    """Scale-adaptive config. Returns float64 array."""
    cfg = np.zeros(CFG_SIZE, dtype=np.float64)

    cfg[CFG_MAX_ITERATIONS] = 50000
    cfg[CFG_SEGMENT_LENGTH] = 100
    cfg[CFG_NO_IMPROVE_LIMIT] = 5000
    cfg[CFG_SA_INITIAL_TEMP] = 100.0
    cfg[CFG_SA_COOLING_RATE] = 0.9997
    cfg[CFG_SA_MIN_TEMP] = 0.01
    cfg[CFG_REACTION_FACTOR] = 0.1
    cfg[CFG_SIGMA_1] = 13
    cfg[CFG_SIGMA_2] = 9
    cfg[CFG_SIGMA_3] = 33
    cfg[CFG_PENALTY_W3_START] = 50.0
    cfg[CFG_PENALTY_W3_END] = 1.0
    cfg[CFG_PENALTY_DECAY_RATE] = 0.9998
    cfg[CFG_LS_FREQUENCY] = 5
    cfg[CFG_LS_ON_NEW_BEST] = 1.0
    cfg[CFG_CROSS_FREQUENCY] = 10
    cfg[CFG_Q_MIN] = 3
    cfg[CFG_Q_MAX_PCT] = 0.15
    cfg[CFG_SHAW_RANDOMNESS] = 6.0
    cfg[CFG_WORST_NOISE] = 0.1
    cfg[CFG_ZONE_PCT] = 15
    cfg[CFG_REGRET_K] = 3
    cfg[CFG_REGRET_NOISE] = 0.1
    cfg[CFG_SAT_THRESHOLD_KM] = 5000  # meters (was km, now i64 meters)
    cfg[CFG_REHEAT_THRESHOLD_PCT] = 0.5
    cfg[CFG_REHEAT_TEMP_PCT] = 0.5
    cfg[CFG_ELITE_POOL_SIZE] = 5
    cfg[CFG_ELITE_MIN_HAMMING_PCT] = 0.10
    cfg[CFG_RESTART_THRESHOLD_PCT] = 0.80
    cfg[CFG_MAX_RESTARTS] = 3
    cfg[CFG_QL_ENABLED] = 0.0
    cfg[CFG_QL_ALPHA] = 0.1
    cfg[CFG_QL_GAMMA] = 0.9
    cfg[CFG_QL_EPSILON] = 0.15
    cfg[CFG_QL_EPSILON_DECAY] = 0.9999
    cfg[CFG_QL_EPSILON_MIN] = 0.02
    cfg[CFG_DELTA_T] = SYNC_DELTA_T  # seconds (i64 stored as f64)
    cfg[CFG_BLINK_PROB] = 0.0
    cfg[CFG_SISR_MAX_STRING_LEN] = 10
    cfg[CFG_SISR_MAX_ROUTES] = 3
    cfg[CFG_HISTORY_NOISE] = 0.1

    # Scale-adaptive iteration limits
    if n_customers < 100:
        cfg[CFG_MAX_ITERATIONS] = 20000
        cfg[CFG_NO_IMPROVE_LIMIT] = 2000
    elif n_customers < 500:
        cfg[CFG_MAX_ITERATIONS] = 50000
        cfg[CFG_NO_IMPROVE_LIMIT] = 5000
    else:
        cfg[CFG_MAX_ITERATIONS] = 100000
        cfg[CFG_NO_IMPROVE_LIMIT] = 10000

    cfg[CFG_Q_MAX] = max(5, int(n_customers * cfg[CFG_Q_MAX_PCT]))

    if overrides:
        for key, val in overrides.items():
            cfg[key] = val

    return cfg
