"""Intra-route local search: two-opt, or-opt. All @njit, i64 distances."""
from numba import njit

from src.solution.route_ops import reverse_segment
from src.alns.ls_helpers import get_lengths, two_opt_delta, or_opt_delta, do_or_opt_move


@njit(cache=True)
def two_opt(sol, vtype, vid, dist_matrix):
    """Reverse segment in route. Returns True if improved."""
    L = get_lengths(sol, vtype)[vid]
    if L < 3:
        return False
    improved = False
    for i in range(L - 1):
        for j in range(i + 2, L):
            if two_opt_delta(sol, vtype, vid, i, j, dist_matrix) < -1:
                reverse_segment(sol, vtype, vid, i, j, dist_matrix)
                improved = True
    return improved


@njit(cache=True)
def or_opt(sol, vtype, vid, dist_matrix):
    """Move segment (1-3 stops) within route. Returns True if improved."""
    L = get_lengths(sol, vtype)[vid]
    improved = False
    for seg_len in (1, 2, 3):
        for i in range(L - seg_len + 1):
            for j in range(L + 1):
                if j >= i and j <= i + seg_len:
                    continue
                if or_opt_delta(sol, vtype, vid, i, seg_len, j, dist_matrix) < -1:
                    do_or_opt_move(sol, vtype, vid, i, seg_len, j, dist_matrix)
                    improved = True
                    break
    return improved
