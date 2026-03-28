"""Simulated Annealing acceptance criterion."""
import numpy as np
from numba import njit

from src.data.constants import (
    CFG_SA_INITIAL_TEMP, CFG_Q_MIN, CFG_Q_MAX,
    SOL_TRUCK_DISTANCES, SOL_BIKE_DISTANCES,
    STF_NO_IMPROVE_COUNT, STF_TEMPERATURE, STF_INITIAL_TEMP, STF_REHEAT_DONE,
    CFG_NO_IMPROVE_LIMIT, CFG_REHEAT_THRESHOLD_PCT, CFG_REHEAT_TEMP_PCT,
)
from src.alns.destroy_basic import random_removal
from src.alns.repair_ops import greedy_insertion
from src.solution.structure import copy_solution


@njit(cache=True)
def sa_accept(current_fitness, new_fitness, temperature, seed):
    """Accept better always, worse with probability. Fitness is i64 VND."""
    delta = new_fitness - current_fitness
    if delta <= 0:
        return True
    if temperature <= 1e-12:
        return False
    prob = np.exp(-delta / temperature)
    np.random.seed(seed)
    return np.random.random() < prob


@njit(cache=True)
def cool_temperature(temperature, cooling_rate):
    """Geometric cooling."""
    return temperature * cooling_rate


def compute_initial_temperature(sol, customers, vehicles, dist_matrix,
                                config, rng, target_accept_rate=0.8, n_samples=100):
    """Auto-calibrate T0 from random perturbations. No restricted param."""
    deltas = []
    N = len(customers)
    mini_cfg = config.copy()
    mini_cfg[CFG_Q_MIN] = max(1, min(3, N))
    mini_cfg[CFG_Q_MAX] = max(1, min(5, N))

    for _ in range(n_samples):
        sol_copy = copy_solution(sol)
        d_seed = int(rng.integers(0, 2**31))
        r_seed = int(rng.integers(0, 2**31))
        removed = random_removal(sol_copy, customers, dist_matrix, d_seed, mini_cfg)
        if len(removed) > 0:
            greedy_insertion(sol_copy, removed, customers, dist_matrix,
                             vehicles, r_seed, mini_cfg)
        d_before = sol[SOL_TRUCK_DISTANCES].sum() + sol[SOL_BIKE_DISTANCES].sum()
        d_after = sol_copy[SOL_TRUCK_DISTANCES].sum() + sol_copy[SOL_BIKE_DISTANCES].sum()
        delta = abs(int(d_after - d_before))
        if delta > 0:
            deltas.append(delta)

    if len(deltas) == 0:
        return config[CFG_SA_INITIAL_TEMP]

    avg_delta = np.mean(deltas)
    T0 = -avg_delta / np.log(target_accept_rate)
    return float(max(T0, 0.01))


@njit(cache=True)
def maybe_reheat(state_f, config):
    """Reheat temperature if stagnation threshold reached."""
    if state_f[STF_REHEAT_DONE] > 0.5:
        return
    threshold = int(config[CFG_NO_IMPROVE_LIMIT] * config[CFG_REHEAT_THRESHOLD_PCT])
    if state_f[STF_NO_IMPROVE_COUNT] >= threshold:
        state_f[STF_TEMPERATURE] = state_f[STF_INITIAL_TEMP] * config[CFG_REHEAT_TEMP_PCT]
        state_f[STF_REHEAT_DONE] = 1.0
