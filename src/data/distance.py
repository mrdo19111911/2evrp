"""Distance matrix computation."""
import numpy as np


def compute_dist_matrix(depot, customers):
    """Euclidean distance matrix (N+1, N+1). Index 0 = depot."""
    coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))
