"""Pareto archive for multi-objective optimization."""
import numpy as np


def create_pareto_archive(max_size=50):
    """Empty archive. Returns list."""
    return []


def _dominates(c1, m1, c2, m2):
    """True if (c1,m1) dominates (c2,m2): <= in both, < in at least one."""
    return (c1 <= c2 and m1 <= m2) and (c1 < c2 or m1 < m2)


def update_pareto_archive(archive, new_cost, new_makespan, new_sol, max_size=50):
    """Add if non-dominated, remove dominated. Returns True if added."""
    # Check if new is dominated by any existing
    for cost, makespan, sol in archive:
        if _dominates(cost, makespan, new_cost, new_makespan):
            return False

    # Remove entries dominated by new
    archive[:] = [(c, m, s) for c, m, s in archive
                  if not _dominates(new_cost, new_makespan, c, m)]

    archive.append((new_cost, new_makespan, new_sol))

    # Prune if exceeding max_size using crowding distance
    if len(archive) > max_size:
        _prune_archive(archive, max_size)

    return True


def _prune_archive(archive, max_size):
    """Remove entries with smallest crowding distance until at max_size."""
    while len(archive) > max_size:
        n = len(archive)
        if n <= 2:
            break
        # Sort by cost for crowding distance
        archive.sort(key=lambda x: x[0])
        crowding = [float('inf')] * n
        for i in range(1, n - 1):
            cost_range = archive[-1][0] - archive[0][0]
            make_range = max(a[1] for a in archive) - min(a[1] for a in archive)
            cd = 0.0
            if cost_range > 0:
                cd += (archive[i + 1][0] - archive[i - 1][0]) / cost_range
            if make_range > 0:
                cd += (abs(archive[i - 1][1] - archive[i + 1][1])) / make_range
            crowding[i] = cd
        # Remove entry with smallest crowding distance (not endpoints)
        min_idx = min(range(1, n - 1), key=lambda i: crowding[i])
        archive.pop(min_idx)


def get_pareto_front(archive):
    """Extract (costs, makespans) arrays sorted by cost. For visualization."""
    if len(archive) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    entries = sorted(archive, key=lambda x: x[0])
    costs = np.array([e[0] for e in entries], dtype=np.float64)
    makespans = np.array([e[1] for e in entries], dtype=np.float64)
    return costs, makespans
