"""Tests for repair operators (src/alns/repair.py) — Data Model v2."""
import numpy as np
import pytest

from src.alns.repair import (
    greedy_insertion,
    regret_k_insertion,
    transfer_aware_insertion,
    REPAIR_OPS,
)
from src.alns.destroy import random_removal
from src.solution.structure import copy_solution


def _remove_customers_v2(sol, to_remove):
    """Manually unassign customers from unified solution."""
    from src.data.constants import ACT_DELIVER, ACT_PAD
    for c in to_remove:
        vid = sol["cust_vehicle"][c]
        pos = sol["cust_route_pos"][c]
        if vid < 0:
            continue
        # Clear stop
        sol["stops"][vid, pos] = -1
        sol["actions"][vid, pos] = ACT_PAD
        # Compact route (shift left)
        L = sol["lengths"][vid]
        for j in range(pos, L - 1):
            sol["stops"][vid, j] = sol["stops"][vid, j + 1]
            sol["actions"][vid, j] = sol["actions"][vid, j + 1]
            sc = sol["stops"][vid, j]
            if sc >= 0 and sol["actions"][vid, j] == ACT_DELIVER:
                # Find customer for this location
                pass  # simplified; rely on rebuild
        sol["stops"][vid, L - 1] = -1
        sol["actions"][vid, L - 1] = ACT_PAD
        sol["lengths"][vid] -= 1
        sol["cust_vehicle"][c] = -1
        sol["cust_route_pos"][c] = -1


# ---------------------------------------------------------------------------
# greedy_insertion
# ---------------------------------------------------------------------------

def test_greedy_all_reinserted(sol_10_assigned, default_params):
    """Remove 3, greedy insert -> all re-assigned."""
    bundle = sol_10_assigned
    sol = copy_solution(bundle["sol"])
    data = bundle["data"]
    rng = np.random.default_rng(0)

    # Use destroy operator to remove
    removed = random_removal(sol, data, rng, default_params)
    assert len(removed) >= 3

    greedy_insertion(sol, removed, data, np.random.default_rng(1), default_params)
    for c in removed:
        assert sol["cust_vehicle"][c] >= 0, f"C{c} not reinserted"


def test_greedy_preserves_assigned(sol_10_assigned, default_params):
    """After reinserting, every customer assigned."""
    bundle = sol_10_assigned
    sol = copy_solution(bundle["sol"])
    data = bundle["data"]
    rng = np.random.default_rng(42)

    removed = random_removal(sol, data, rng, default_params)
    greedy_insertion(sol, removed, data, np.random.default_rng(7), default_params)

    for c in range(bundle["N"]):
        assert sol["cust_vehicle"][c] >= 0, f"C{c} unassigned after repair"


# ---------------------------------------------------------------------------
# regret_k_insertion
# ---------------------------------------------------------------------------

def test_regret_all_reinserted(sol_10_assigned, default_params):
    """Remove some, regret-k insert -> all re-assigned."""
    bundle = sol_10_assigned
    sol = copy_solution(bundle["sol"])
    data = bundle["data"]
    rng = np.random.default_rng(0)

    removed = random_removal(sol, data, rng, default_params)
    params = {**default_params, "regret_k": 3}
    regret_k_insertion(sol, removed, data, np.random.default_rng(7), params)

    for c in removed:
        assert sol["cust_vehicle"][c] >= 0, f"C{c} not reinserted"


# ---------------------------------------------------------------------------
# transfer_aware_insertion
# ---------------------------------------------------------------------------

def test_transfer_aware_with_event(sol_with_transfer, default_params):
    """Transfer exists -> transfer_aware_insertion works."""
    bundle = sol_with_transfer
    sol = copy_solution(bundle["sol"])
    data = bundle["data"]
    rng = np.random.default_rng(0)

    removed = random_removal(sol, data, rng, {**default_params, "q_min": 1, "q_max": 2})
    if len(removed) > 0:
        transfer_aware_insertion(sol, removed, data, np.random.default_rng(1), default_params)
        for c in removed:
            assert sol["cust_vehicle"][c] >= 0, f"C{c} not reinserted"


def test_transfer_aware_no_transfers(sol_10_assigned, default_params):
    """No transfers -> falls back to greedy."""
    bundle = sol_10_assigned
    sol = copy_solution(bundle["sol"])
    data = bundle["data"]
    rng = np.random.default_rng(0)

    removed = random_removal(sol, data, rng, default_params)
    transfer_aware_insertion(sol, removed, data, np.random.default_rng(1), default_params)
    for c in removed:
        assert sol["cust_vehicle"][c] >= 0, f"C{c} not reinserted"


# ---------------------------------------------------------------------------
# REPAIR_OPS list
# ---------------------------------------------------------------------------

def test_repair_ops_length():
    assert len(REPAIR_OPS) == 3


def test_repair_ops_names():
    names = {f.__name__ for f in REPAIR_OPS}
    expected = {"greedy_insertion", "regret_k_insertion", "transfer_aware_insertion"}
    assert names == expected
