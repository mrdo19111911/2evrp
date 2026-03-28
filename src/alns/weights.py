"""Adaptive operator weight management."""
import numpy as np
from numba import njit


@njit(cache=True)
def select_operator(weights, seed):
    """Roulette wheel selection. Returns operator index."""
    n = len(weights)
    total = 0.0
    all_finite = True
    for i in range(n):
        if not np.isfinite(weights[i]):
            all_finite = False
            break
        total += weights[i]

    if not all_finite or total < 1e-10:
        np.random.seed(seed)
        return int(np.random.randint(0, n))

    # Cumulative sum roulette
    np.random.seed(seed)
    r = np.random.random() * total
    cumsum = 0.0
    for i in range(n):
        cumsum += weights[i]
        if r <= cumsum:
            return i
    return n - 1
