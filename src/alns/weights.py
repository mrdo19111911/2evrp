"""Adaptive operator weight management."""
import numpy as np


def init_weights(n_destroy, n_repair, n_cross):
    """Uniform weights + zero scores/counts."""
    d_w = np.full(n_destroy, 1.0 / n_destroy)
    r_w = np.full(n_repair, 1.0 / n_repair)
    c_w = np.full(n_cross, 1.0 / n_cross)
    d_scores = np.zeros(n_destroy, dtype=np.float64)
    r_scores = np.zeros(n_repair, dtype=np.float64)
    c_scores = np.zeros(n_cross, dtype=np.float64)
    d_counts = np.zeros(n_destroy, dtype=np.int32)
    r_counts = np.zeros(n_repair, dtype=np.int32)
    c_counts = np.zeros(n_cross, dtype=np.int32)
    return (d_w, r_w, c_w, d_scores, r_scores, c_scores,
            d_counts, r_counts, c_counts)


def select_operator(weights, rng):
    """Roulette wheel selection. Returns operator index."""
    return int(rng.choice(len(weights), p=weights))


def update_scores(scores, counts, op_idx, result, score_values=(1, 2, 5)):
    """Add score for operator based on result quality."""
    if result > 0:
        scores[op_idx] += score_values[result - 1]
    counts[op_idx] += 1


def update_weights(weights, scores, counts, reaction_factor=0.1):
    """End-of-segment: update weights from scores, normalize, reset."""
    for i in range(len(weights)):
        if counts[i] > 0:
            performance = scores[i] / counts[i]
            weights[i] = (1 - reaction_factor) * weights[i] + reaction_factor * performance

    n = len(weights)
    min_w = 0.01
    np.maximum(weights, min_w, out=weights)
    weights /= weights.sum()
    # Enforce minimum post-normalize: fix small weights, redistribute excess
    low = weights < min_w
    if low.any():
        deficit = np.sum(min_w - weights[low])
        weights[low] = min_w
        high = ~low
        if high.any():
            weights[high] -= deficit * weights[high] / weights[high].sum()
            np.maximum(weights, min_w, out=weights)
        weights /= weights.sum()

    scores[:] = 0
    counts[:] = 0
