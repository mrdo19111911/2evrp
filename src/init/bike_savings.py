"""CW Savings computation for bike giant tour."""
import numpy as np


def collect_reload_points(giant_tour):
    """Depot (dm index 0) + satellite nodes. Returns list of dm indices."""
    reload_dm = [0]
    if giant_tour is not None:
        for s in giant_tour:
            if s["type"] == "satellite":
                dm_idx = s["node"] + 1
                if dm_idx not in reload_dm:
                    reload_dm.append(dm_idx)
    return reload_dm


def compute_bike_savings(bike_customers, reload_dm, dist_matrix):
    """saving(A,B) = min_S{ d(A,S) + d(B,S) } - d(A,B). Sorted descending."""
    n = len(bike_customers)
    min_to_reload = np.empty(n, dtype=np.float64)
    for i in range(n):
        c = bike_customers[i]
        min_to_reload[i] = min(dist_matrix[c + 1, r] for r in reload_dm)

    savings = []
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = bike_customers[i], bike_customers[j]
            s = min_to_reload[i] + min_to_reload[j] - dist_matrix[ci + 1, cj + 1]
            savings.append((s, i, j))

    savings.sort(key=lambda x: -x[0])
    return savings


def build_bike_giant_tour(bike_customers, savings):
    """CW link into single chain (Bike GT). Returns ordered customer indices."""
    n = len(bike_customers)
    if n == 0:
        return []
    if n == 1:
        return [bike_customers[0]]

    neighbor = [[] for _ in range(n)]
    for s, i, j in savings:
        if s <= 0:
            break
        if len(neighbor[i]) >= 2 or len(neighbor[j]) >= 2:
            continue
        if _would_cycle(neighbor, i, j, n):
            continue
        neighbor[i].append(j)
        neighbor[j].append(i)

    ordered = _traverse_chain(neighbor, n)
    return [bike_customers[idx] for idx in ordered]


def _would_cycle(neighbor, i, j, n):
    """Check if linking i-j creates cycle shorter than n."""
    visited = {i}
    current = i
    length = 1
    while True:
        nxt = None
        for nb in neighbor[current]:
            if nb not in visited:
                nxt = nb
                break
        if nxt is None:
            break
        visited.add(nxt)
        current = nxt
        length += 1
        if current == j:
            return length < n
    return False


def _traverse_chain(neighbor, n):
    """Traverse all nodes via neighbor links. Returns list of indices."""
    visited = set()
    result = []

    endpoints = [i for i in range(n) if len(neighbor[i]) < 2]
    starts = endpoints if endpoints else [0]

    for start in starts:
        if start in visited:
            continue
        current = start
        while current is not None and current not in visited:
            visited.add(current)
            result.append(current)
            nxt = None
            for nb in neighbor[current]:
                if nb not in visited:
                    nxt = nb
                    break
            current = nxt

    for i in range(n):
        if i not in visited:
            result.append(i)

    return result
