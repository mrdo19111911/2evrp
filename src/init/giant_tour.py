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
            # Pick satellite from bike_nodes (not big_nodes) to avoid dual-role conflict
            bike_members = cluster["bike_nodes"]
            sat_node = find_cluster_satellite(
                {"members": bike_members, "centroid": cluster["centroid"]},
                customers, dist_matrix)
            # Satellite demand = bike customers only, excluding sat_node's own demand
            # (sat_node serves as transfer point, its own delivery is separate)
            other_bikes = bike_members[bike_members != sat_node]
            sat_demand = float(customers[other_bikes, COL_DEMAND].sum()) if len(other_bikes) > 0 else 0.0
            stops.append({"node": sat_node, "type": "satellite",
                          "demand": sat_demand, "cluster_idx": i})

    return cw_savings_order(stops, dist_matrix)


def build_bike_giant_tour(truck_gt, bike_customers, customers, dist_matrix):
    """Cheapest insertion of bike customers into truck GT.

    GT_all starts as copy of truck GT. For each bike customer, find the
    position in GT_all where insertion cost is minimum:
      delta = d(GT[pos], c) + d(c, GT[pos+1]) - d(GT[pos], GT[pos+1])
    Insert at that position. GT grows with each insertion.

    Truck stops become RELOAD points for bikes.
    Returns combined GT (deliver + reload stops).
    """
    if len(truck_gt) == 0 and len(bike_customers) == 0:
        return []

    # Start with truck GT as backbone (RELOAD stops)
    gt_all = []
    for s in truck_gt:
        gt_all.append({
            "node": s["node"], "type": "reload",
            "demand": 0.0,
            "cluster_idx": s.get("cluster_idx", -1),
        })

    if len(bike_customers) == 0:
        return gt_all

    # If truck GT is empty, build NN tour from bike customers only
    if len(gt_all) == 0:
        for c in bike_customers:
            c = int(c)
            gt_all.append({
                "node": c, "type": "deliver",
                "demand": float(customers[c, COL_DEMAND]),
                "cluster_idx": -1,
            })
        return gt_all

    # Cheapest insertion: insert each bike customer one by one
    for c in bike_customers:
        c = int(c)
        cn = c + 1  # dist_matrix index
        n = len(gt_all)

        best_pos = 0
        best_delta = np.inf

        for pos in range(n):
            node_a = gt_all[pos]["node"]
            node_b = gt_all[(pos + 1) % n]["node"]

            delta = (dist_matrix[node_a + 1, cn]
                     + dist_matrix[cn, node_b + 1]
                     - dist_matrix[node_a + 1, node_b + 1])

            if delta < best_delta:
                best_delta = delta
                best_pos = pos + 1  # insert AFTER pos

        gt_all.insert(best_pos, {
            "node": c, "type": "deliver",
            "demand": float(customers[c, COL_DEMAND]),
            "cluster_idx": -1,
        })

    return gt_all
