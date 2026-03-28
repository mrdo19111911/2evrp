"""Customer clustering for initial solution."""
import numpy as np
from math import ceil

from ..data.constants import COL_X, COL_Y, COL_DEMAND, COL_RESTRICTED
from ..data.cost import BIKE_CAPACITY_G


def estimate_n_clusters(customers, n_trucks, capacity):
    """Estimate k from total demand / capacity."""
    total_demand = customers[:, COL_DEMAND].sum()
    return max(1, int(ceil(total_demand / capacity)))


def kmeans(coords, k, rng, max_iter=50):
    """Simple k-means, pure numpy. Returns labels ndarray."""
    n = len(coords)
    indices = rng.choice(n, size=k, replace=False)
    centroids = coords[indices].astype(np.float64).copy()

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        dists = np.linalg.norm(
            coords[:, None, :].astype(np.float64) - centroids[None, :, :],
            axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = coords[mask].astype(np.float64).mean(axis=0)
    return labels


def group_by_label(labels, customers):
    """Labels -> list of cluster dicts with members, big_nodes, bike_nodes."""
    unique_labels = np.unique(labels)
    clusters = []
    for lbl in unique_labels:
        members = np.where(labels == lbl)[0].astype(np.int32)
        demands = customers[members, COL_DEMAND]
        big_mask = demands > BIKE_CAPACITY_G
        rest_mask = customers[members, COL_RESTRICTED] == 1
        big_nodes = members[big_mask]
        bike_nodes = members[~big_mask]
        centroid = customers[members][:, [COL_X, COL_Y]].astype(
            np.float64).mean(axis=0)
        clusters.append({
            "members": members,
            "big_nodes": big_nodes,
            "bike_nodes": bike_nodes,
            "restricted": members[rest_mask],
            "total_demand": int(demands.sum()),
            "centroid": centroid,
        })
    return clusters


def cluster_customers(customers, dist_matrix, rng):
    """K-means clustering. Returns list of cluster dicts."""
    n_trucks = max(1, len(customers) // 5)
    coords = customers[:, [COL_X, COL_Y]]
    n_clusters = estimate_n_clusters(customers, n_trucks, 2000000)
    n_clusters = min(n_clusters, len(customers))
    labels = kmeans(coords, n_clusters, rng)
    return group_by_label(labels, customers)


def make_single_cluster(customer_idx, customers):
    """1 customer -> 1 cluster dict."""
    c = customer_idx
    demand = int(customers[c, COL_DEMAND])
    members = np.array([c], dtype=np.int32)
    is_big = demand > BIKE_CAPACITY_G
    is_rest = int(customers[c, COL_RESTRICTED]) == 1
    return {
        "members": members,
        "big_nodes": members if is_big else np.array([], dtype=np.int32),
        "bike_nodes": np.array([], dtype=np.int32) if is_big else members,
        "restricted": members if is_rest else np.array([], dtype=np.int32),
        "total_demand": demand,
        "centroid": customers[c, [COL_X, COL_Y]].astype(np.float64).copy(),
    }


def split_cluster(members, customers, k=2):
    """Split 1 cluster into k sub-clusters."""
    if len(members) <= k:
        demands = customers[members, COL_DEMAND]
        big_mask = demands > BIKE_CAPACITY_G
        rest_mask = customers[members, COL_RESTRICTED] == 1
        centroid = customers[members][:, [COL_X, COL_Y]].astype(
            np.float64).mean(axis=0)
        return [{
            "members": members,
            "big_nodes": members[big_mask],
            "bike_nodes": members[~big_mask],
            "restricted": members[rest_mask],
            "total_demand": int(demands.sum()),
            "centroid": centroid,
        }]

    coords = customers[members][:, [COL_X, COL_Y]]
    rng = np.random.default_rng(0)
    labels = kmeans(coords, k, rng)
    result = []
    for lbl in np.unique(labels):
        sub = members[labels == lbl]
        demands = customers[sub, COL_DEMAND]
        big_mask = demands > BIKE_CAPACITY_G
        rest_mask = customers[sub, COL_RESTRICTED] == 1
        centroid = customers[sub][:, [COL_X, COL_Y]].astype(
            np.float64).mean(axis=0)
        result.append({
            "members": sub,
            "big_nodes": sub[big_mask],
            "bike_nodes": sub[~big_mask],
            "restricted": sub[rest_mask],
            "total_demand": int(demands.sum()),
            "centroid": centroid,
        })
    return result


def rebalance_clusters(clusters, customers, truck_capacity):
    """Split heavy clusters (>capacity/2). Keep single heavy customers."""
    result = []
    half_cap = truck_capacity // 2

    for cluster in clusters:
        if cluster["total_demand"] <= half_cap:
            result.append(cluster)
            continue

        heavy_mask = customers[cluster["members"], COL_DEMAND] > half_cap
        heavy = cluster["members"][heavy_mask]

        if len(heavy) > 0 and len(cluster["members"]) == len(heavy):
            for c in heavy:
                result.append(make_single_cluster(c, customers))
            continue

        if len(heavy) > 0:
            for c in heavy:
                result.append(make_single_cluster(c, customers))
            remaining = np.setdiff1d(cluster["members"], heavy)
        else:
            remaining = cluster["members"]

        sub_clusters = split_cluster(remaining, customers, k=2)
        result.extend(sub_clusters)
    return result
