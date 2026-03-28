"""Precompute k-nearest neighbor matrix for granular LS restriction."""
import numpy as np


def precompute_neighbors(dist_matrix, n_customers, k=30):
    """For each customer c, find k nearest customer IDs.
    Returns neighbors array (n_customers, k) int32."""
    k = min(k, n_customers - 1)
    if k <= 0:
        return np.empty((n_customers, 0), dtype=np.int32)
    neighbors = np.empty((n_customers, k), dtype=np.int32)
    for c in range(n_customers):
        dists = dist_matrix[c + 1, 1:n_customers + 1].copy()
        dists[c] = np.inf
        top_k = np.argpartition(dists, k)[:k]
        neighbors[c] = top_k[np.argsort(dists[top_k])]
    return neighbors


def is_neighbor(ca, cb, neighbors):
    """O(log k) check: is cb in neighbors[ca] OR ca in neighbors[cb]?"""
    k = neighbors.shape[1]
    if k == 0:
        return True  # no restriction
    idx = np.searchsorted(neighbors[ca], cb)
    if idx < k and neighbors[ca, idx] == cb:
        return True
    idx = np.searchsorted(neighbors[cb], ca)
    if idx < k and neighbors[cb, idx] == ca:
        return True
    return False
