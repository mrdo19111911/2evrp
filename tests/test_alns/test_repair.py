"""Tests for repair operators (src/alns/repair.py). i64 interface.
Repair signature: (sol, removed, customers_i64, dist_matrix_i64, vehicles_i64, seed_int, config_f64)
No restricted param.
"""
import numpy as np
import pytest

from src.alns.repair import repair_dispatch
from src.alns.repair_ops import greedy_insertion, regret_k_insertion
from src.alns.destroy_basic import random_removal
from src.alns.config import make_config
from src.solution.structure import copy_solution
from src.data.constants import (
    SOL_CUST_VEHICLE, CFG_Q_MIN, CFG_Q_MAX, CFG_REGRET_K,
)


# ---------------------------------------------------------------------------
# greedy_insertion — no restricted param
# ---------------------------------------------------------------------------

def test_greedy_all_reinserted(sol_5_assigned):
    bundle = sol_5_assigned
    sol = copy_solution(bundle["sol"])
    config = make_config(bundle["N"], overrides={CFG_Q_MIN: 1, CFG_Q_MAX: 3})

    removed = random_removal(sol, bundle["customers"], bundle["dist_matrix"],
                             0, config)
    assert len(removed) >= 1

    greedy_insertion(sol, removed, bundle["customers"],
                     bundle["dist_matrix"], bundle["vehicles"],
                     1, config)
    for c in removed:
        assert sol[SOL_CUST_VEHICLE][c] >= 0, f"C{c} not reinserted"


def test_greedy_preserves_assigned(sol_5_assigned):
    bundle = sol_5_assigned
    sol = copy_solution(bundle["sol"])
    config = make_config(bundle["N"], overrides={CFG_Q_MIN: 1, CFG_Q_MAX: 2})

    removed = random_removal(sol, bundle["customers"], bundle["dist_matrix"],
                             42, config)
    greedy_insertion(sol, removed, bundle["customers"],
                     bundle["dist_matrix"], bundle["vehicles"],
                     7, config)

    for c in range(bundle["N"]):
        assert sol[SOL_CUST_VEHICLE][c] >= 0, f"C{c} unassigned after repair"


# ---------------------------------------------------------------------------
# regret_k_insertion — no restricted param
# ---------------------------------------------------------------------------

def test_regret_all_reinserted(sol_5_assigned):
    bundle = sol_5_assigned
    sol = copy_solution(bundle["sol"])
    config = make_config(bundle["N"], overrides={
        CFG_Q_MIN: 1, CFG_Q_MAX: 2, CFG_REGRET_K: 3,
    })

    removed = random_removal(sol, bundle["customers"], bundle["dist_matrix"],
                             0, config)
    regret_k_insertion(sol, removed, bundle["customers"],
                       bundle["dist_matrix"], bundle["vehicles"],
                       7, config)

    for c in removed:
        assert sol[SOL_CUST_VEHICLE][c] >= 0, f"C{c} not reinserted"


# ---------------------------------------------------------------------------
# REPAIR_OPS list
# ---------------------------------------------------------------------------

def test_repair_dispatch_callable():
    assert callable(repair_dispatch)
