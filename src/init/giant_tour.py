"""Giant tour through big nodes + cluster satellites."""
import numpy as np

from ..data.constants import COL_X, COL_Y, COL_DEMAND


def find_cluster_satellite(cluster, customers, dist_matrix):
    """Node in cluster closest to centroid. Returns customer index."""
    members = cluster["members"]
    coords = customers[members][:, [COL_X, COL_Y]]
    centroid = cluster["centroid"]
    dists = np.linalg.norm(coords - centroid, axis=1)
    return int(members[np.argmin(dists)])


def cw_savings_order(stops, dist_matrix):
    """Clarke-Wright Savings ordering.
    s(i,j) = d(depot,i) + d(depot,j) - d(i,j).
    Greedily link nodes with highest savings into a chain."""
    n = len(stops)
    if n == 0:
        return []
    if n == 1:
        return list(stops)

    nodes = [s["node"] for s in stops]

    # Compute savings for all pairs
    savings = []
    for i in range(n):
        di = dist_matrix[0, nodes[i] + 1]
        for j in range(i + 1, n):
            dj = dist_matrix[0, nodes[j] + 1]
            dij = dist_matrix[nodes[i] + 1, nodes[j] + 1]
            s = di + dj - dij
            savings.append((s, i, j))

    savings.sort(key=lambda x: -x[0])  # descending

    # Build chain: each node has at most 2 neighbors (prev, next)
    # Track chain endpoints
    neighbor = [[] for _ in range(n)]

    for s, i, j in savings:
        if len(neighbor[i]) >= 2 or len(neighbor[j]) >= 2:
            continue
        # Check no cycle (unless it would close the full tour)
        if _would_create_short_cycle(neighbor, i, j, n):
            continue
        neighbor[i].append(j)
        neighbor[j].append(i)

    # Traverse chain from an endpoint (node with < 2 neighbors)
    start = 0
    for i in range(n):
        if len(neighbor[i]) < 2:
            start = i
            break

    ordered = []
    visited = set()
    current = start
    while current is not None and current not in visited:
        visited.add(current)
        ordered.append(stops[current])
        next_node = None
        for nb in neighbor[current]:
            if nb not in visited:
                next_node = nb
                break
        current = next_node

    return ordered


def _would_create_short_cycle(neighbor, i, j, n):
    """Check if linking i-j would create a cycle shorter than n."""
    # Walk from i through existing links, see if we reach j
    visited = {i}
    current = i
    length = 1
    while True:
        next_node = None
        for nb in neighbor[current]:
            if nb not in visited:
                next_node = nb
                break
        if next_node is None:
            break
        visited.add(next_node)
        current = next_node
        length += 1
        if current == j:
            return length < n  # cycle shorter than full tour
    return False


def build_giant_tour(clusters, customers, depot, dist_matrix):
    """Ordered tour using CW Savings: big nodes + satellite nodes."""
    stops = []

    for i, cluster in enumerate(clusters):
        for c in cluster["big_nodes"]:
            stops.append({"node": int(c), "type": "deliver",
                          "demand": float(customers[c, COL_DEMAND]),
                          "cluster_idx": -1})

        if len(cluster["bike_nodes"]) > 0:
            sat_node = find_cluster_satellite(cluster, customers, dist_matrix)
            sat_demand = float(customers[cluster["bike_nodes"], COL_DEMAND].sum())
            stops.append({"node": sat_node, "type": "satellite",
                          "demand": sat_demand, "cluster_idx": i})

    return cw_savings_order(stops, dist_matrix)
