"""Simulated Annealing acceptance criterion."""
import numpy as np

from src.alns.destroy import random_removal
from src.alns.repair import greedy_insertion
from src.solution.structure import copy_solution


def sa_accept(current_fitness, new_fitness, temperature, rng):
    """Accept better always, worse with probability. Returns bool."""
    delta = new_fitness - current_fitness
    if delta <= 0:
        return True
    if temperature <= 1e-12:
        return False
    prob = np.exp(-delta / temperature)
    return bool(rng.random() < prob)


def cool_temperature(temperature, cooling_rate):
    """Geometric cooling. Returns new temperature."""
    return temperature * cooling_rate


def compute_initial_temperature(sol, customers, restricted, vehicles, dist_matrix,
                                config, rng, target_accept_rate=0.8, n_samples=100):
    """Auto-calibrate T0 from random perturbations."""
    deltas = []
    N = len(customers)
    params = {
        "q_min": max(1, min(3, N)),
        "q_max": max(1, min(5, N)),
    }

    for _ in range(n_samples):
        sol_copy = copy_solution(sol)
        removed = random_removal(sol_copy, customers, dist_matrix, rng, params)
        if len(removed) > 0:
            greedy_insertion(sol_copy, removed, customers, restricted,
                             dist_matrix, vehicles, rng, params)
        # Use total distance as proxy for fitness
        d_before = sol["truck_distances"].sum() + sol["bike_distances"].sum()
        d_after = sol_copy["truck_distances"].sum() + sol_copy["bike_distances"].sum()
        delta = abs(d_after - d_before)
        if delta > 0:
            deltas.append(delta)

    if len(deltas) == 0:
        return config.get("sa_initial_temp", 100.0)

    avg_delta = np.mean(deltas)
    # T0 such that exp(-avg_delta/T0) = target_accept_rate
    # => T0 = -avg_delta / ln(target_accept_rate)
    T0 = -avg_delta / np.log(target_accept_rate)
    return float(max(T0, 0.01))
