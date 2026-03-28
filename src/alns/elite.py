"""Elite pool: diverse solution archive with restart logic. Fitness is i64 VND."""
import numpy as np

from src.data.constants import (
    SOL_CUST_VEHICLE,
    STF_NO_IMPROVE_COUNT, STF_REHEAT_DONE, STF_TEMPERATURE, STF_INITIAL_TEMP, STF_RESTART_COUNT,
    CFG_ELITE_MIN_HAMMING_PCT, CFG_ELITE_POOL_SIZE,
    CFG_NO_IMPROVE_LIMIT, CFG_RESTART_THRESHOLD_PCT, CFG_MAX_RESTARTS,
)
from src.solution.structure import copy_solution


def create_elite_pool():
    return []


def hamming_distance(sol_a, sol_b):
    return int(np.sum(sol_a[SOL_CUST_VEHICLE] != sol_b[SOL_CUST_VEHICLE]))


def is_diverse_enough(sol, pool, min_dist):
    for entry in pool:
        if hamming_distance(sol, entry[0]) < min_dist:
            return False
    return True


def maybe_add_to_pool(pool, sol, fitness, config):
    """fitness is i64 VND."""
    n_cust = len(sol[SOL_CUST_VEHICLE])
    min_dist = max(1, int(n_cust * config[CFG_ELITE_MIN_HAMMING_PCT]))
    max_size = int(config[CFG_ELITE_POOL_SIZE])
    if not is_diverse_enough(sol, pool, min_dist):
        return
    entry = (copy_solution(sol), fitness)
    if len(pool) < max_size:
        pool.append(entry)
    else:
        worst_idx = max(range(len(pool)), key=lambda i: pool[i][1])
        if fitness < pool[worst_idx][1]:
            pool[worst_idx] = entry


def pick_random_elite(pool, rng):
    if not pool:
        return None
    idx = rng.integers(0, len(pool))
    return copy_solution(pool[idx][0])


def maybe_restart_from_elite(state_f, elite_pool, config, rng):
    """Check if restart is warranted. Returns new sol or None."""
    if state_f[STF_RESTART_COUNT] >= config[CFG_MAX_RESTARTS]:
        return None
    threshold = int(config[CFG_NO_IMPROVE_LIMIT] * config[CFG_RESTART_THRESHOLD_PCT])
    if state_f[STF_NO_IMPROVE_COUNT] < threshold:
        return None
    if not elite_pool:
        return None
    new_sol = pick_random_elite(elite_pool, rng)
    state_f[STF_NO_IMPROVE_COUNT] = 0
    state_f[STF_REHEAT_DONE] = 0.0
    state_f[STF_TEMPERATURE] = state_f[STF_INITIAL_TEMP]
    state_f[STF_RESTART_COUNT] += 1
    return new_sol
