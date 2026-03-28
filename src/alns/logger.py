"""Iteration logging — pre-allocated 2D numpy array. @njit on hot path."""
import numpy as np
from numba import njit

from src.data.constants import (
    LOG_ITERATION, LOG_FITNESS, LOG_COST, LOG_MAKESPAN,
    LOG_PENALTY, LOG_FEASIBLE, LOG_ACCEPTED, LOG_TEMPERATURE,
    LOG_W3, LOG_DESTROY_OP, LOG_REPAIR_OP, LOG_BEST_FITNESS, LOG_PARETO_SIZE,
    LOG_COLS, N_DESTROY_OPS, N_REPAIR_OPS,
)


def create_logger(max_iterations):
    """Create pre-allocated log arrays. Returns (log_data, weight_log_d, weight_log_r, n_logged, n_wsnap)."""
    log_data = np.zeros((max_iterations, LOG_COLS), dtype=np.float64)
    max_segments = max_iterations // 100 + 1
    weight_log_d = np.zeros((max_segments, N_DESTROY_OPS), dtype=np.float64)
    weight_log_r = np.zeros((max_segments, N_REPAIR_OPS), dtype=np.float64)
    return log_data, weight_log_d, weight_log_r, 0, 0


@njit(cache=True)
def log_iteration(log_data, n_logged, iteration, fitness, cost, makespan, penalty,
                  feasible, accepted, temperature, w3,
                  destroy_idx, repair_idx, best_fitness, pareto_size):
    """Write 1 row to log. Returns new n_logged."""
    if n_logged >= len(log_data):
        return n_logged
    log_data[n_logged, LOG_ITERATION] = iteration
    log_data[n_logged, LOG_FITNESS] = fitness
    log_data[n_logged, LOG_COST] = cost
    log_data[n_logged, LOG_MAKESPAN] = makespan
    log_data[n_logged, LOG_PENALTY] = penalty
    log_data[n_logged, LOG_FEASIBLE] = 1.0 if feasible else 0.0
    log_data[n_logged, LOG_ACCEPTED] = 1.0 if accepted else 0.0
    log_data[n_logged, LOG_TEMPERATURE] = temperature
    log_data[n_logged, LOG_W3] = w3
    log_data[n_logged, LOG_DESTROY_OP] = destroy_idx
    log_data[n_logged, LOG_REPAIR_OP] = repair_idx
    log_data[n_logged, LOG_BEST_FITNESS] = best_fitness
    log_data[n_logged, LOG_PARETO_SIZE] = pareto_size
    return n_logged + 1


@njit(cache=True)
def log_weights_snapshot(weight_log_d, weight_log_r, n_wsnap,
                         destroy_weights, repair_weights):
    """Snapshot operator weights. Returns new n_wsnap."""
    if n_wsnap >= len(weight_log_d):
        return n_wsnap
    weight_log_d[n_wsnap, :len(destroy_weights)] = destroy_weights
    weight_log_r[n_wsnap, :len(repair_weights)] = repair_weights
    return n_wsnap + 1


def log_to_dict(log_data, n_logged, weight_log_d, weight_log_r, n_wsnap):
    """Convert pre-allocated arrays back to dict for viz compatibility."""
    active = log_data[:n_logged]
    return {
        "iterations": active[:, LOG_ITERATION].tolist(),
        "fitness": active[:, LOG_FITNESS].tolist(),
        "cost": active[:, LOG_COST].tolist(),
        "makespan": active[:, LOG_MAKESPAN].tolist(),
        "penalty": active[:, LOG_PENALTY].tolist(),
        "feasible": (active[:, LOG_FEASIBLE] > 0.5).tolist(),
        "accepted": (active[:, LOG_ACCEPTED] > 0.5).tolist(),
        "temperature": active[:, LOG_TEMPERATURE].tolist(),
        "w3": active[:, LOG_W3].tolist(),
        "destroy_op": active[:, LOG_DESTROY_OP].astype(int).tolist(),
        "repair_op": active[:, LOG_REPAIR_OP].astype(int).tolist(),
        "best_fitness": active[:, LOG_BEST_FITNESS].tolist(),
        "pareto_size": active[:, LOG_PARETO_SIZE].astype(int).tolist(),
        "destroy_weights": weight_log_d[:n_wsnap].tolist(),
        "repair_weights": weight_log_r[:n_wsnap].tolist(),
    }
