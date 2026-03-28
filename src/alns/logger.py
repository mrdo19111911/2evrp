"""Iteration logging for debug + visualization."""
import json
import numpy as np


def create_logger():
    """Create empty log dict with all tracked fields."""
    return {
        "iterations": [],
        "fitness": [],
        "cost": [],
        "makespan": [],
        "penalty": [],
        "feasible": [],
        "temperature": [],
        "w3": [],
        "destroy_op": [],
        "repair_op": [],
        "accepted": [],
        "best_fitness": [],
        "destroy_weights": [],
        "repair_weights": [],
        "pareto_size": [],
    }


def log_iteration(log, iteration, fitness, cost, makespan, penalty,
                  feasible, accepted, temperature, w3,
                  destroy_idx, repair_idx, best_fitness, pareto_size):
    """Append 1 iteration to log. O(1)."""
    log["iterations"].append(iteration)
    log["fitness"].append(float(fitness))
    log["cost"].append(float(cost))
    log["makespan"].append(float(makespan))
    log["penalty"].append(float(penalty))
    log["feasible"].append(bool(feasible))
    log["accepted"].append(bool(accepted))
    log["temperature"].append(float(temperature))
    log["w3"].append(float(w3))
    log["destroy_op"].append(int(destroy_idx))
    log["repair_op"].append(int(repair_idx))
    log["best_fitness"].append(float(best_fitness))
    log["pareto_size"].append(int(pareto_size))


def log_weights_snapshot(log, destroy_weights, repair_weights):
    """Snapshot operator weights at end of segment."""
    log["destroy_weights"].append(destroy_weights.tolist() if isinstance(destroy_weights, np.ndarray) else list(destroy_weights))
    log["repair_weights"].append(repair_weights.tolist() if isinstance(repair_weights, np.ndarray) else list(repair_weights))


def save_log(log, filepath):
    """Save log to JSON."""
    with open(filepath, "w") as f:
        json.dump(log, f)
