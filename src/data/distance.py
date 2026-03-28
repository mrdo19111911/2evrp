"""Distance matrix computation. Returns i64 in meters."""
import numpy as np

from .constants import COL_X, COL_Y


def compute_dist_matrix(depot, customers):
    """Euclidean distance matrix (N+1, N+1), i64 meters. Row/col 0 = depot.

    Both depot and customers are i64 with coordinates in meters.
    """
    depot_xy = depot.reshape(1, 2).astype(np.float64)
    cust_xy = customers[:, [COL_X, COL_Y]].astype(np.float64)
    coords = np.vstack([depot_xy, cust_xy])
    diff = coords[:, None, :] - coords[None, :, :]
    dist_f = np.sqrt((diff ** 2).sum(axis=2))
    return np.round(dist_f).astype(np.int64)
