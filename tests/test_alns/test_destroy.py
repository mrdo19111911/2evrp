"""Tests for destroy operators (src/alns/destroy.py). i64 interface.
Destroy signature: (sol, customers_i64, dist_matrix_i64, seed_int, config_f64) -> i32[:]
"""
import numpy as np
import pytest

from src.alns.destroy import destroy_dispatch
from src.alns.destroy_basic import random_removal, worst_cost_removal, shaw_removal, zone_removal
from src.alns.config import make_config
from src.solution.structure import copy_solution
from src.data.constants import (
    SOL_CUST_VEHICLE, CFG_Q_MIN, CFG_Q_MAX, CFG_WORST_NOISE,
    CFG_SHAW_RANDOMNESS,
)


# ---------------------------------------------------------------------------
# random_removal
# ---------------------------------------------------------------------------

def test_random_removed_count(sol_5_assigned):
    bundle = sol_5_assigned
    config = make_config(bundle["N"], overrides={CFG_Q_MIN: 2, CFG_Q_MAX: 3})
    removed = random_removal(bundle["sol"], bundle["customers"],
                             bundle["dist_matrix"], 0, config)
    assert 2 <= len(removed) <= 3


def test_random_removed_unassigned(sol_5_assigned):
    bundle = sol_5_assigned
    config = make_config(bundle["N"], overrides={CFG_Q_MIN: 1, CFG_Q_MAX: 3})
    removed = random_removal(bundle["sol"], bundle["customers"],
                             bundle["dist_matrix"], 1, config)
    sol = bundle["sol"]
    for c in removed:
        assert sol[SOL_CUST_VEHICLE][c] == -1, f"C{c} still assigned"


def test_random_non_removed_assigned(sol_5_assigned):
    bundle = sol_5_assigned
    config = make_config(bundle["N"], overrides={CFG_Q_MIN: 1, CFG_Q_MAX: 2})
    removed = random_removal(bundle["sol"], bundle["customers"],
                             bundle["dist_matrix"], 2, config)
    removed_set = set(int(c) for c in removed)
    sol = bundle["sol"]
    for c in range(bundle["N"]):
        if c not in removed_set:
            assert sol[SOL_CUST_VEHICLE][c] >= 0, f"C{c} wrongly unassigned"


def test_random_deterministic(sol_5_assigned):
    bundle = sol_5_assigned
    config = make_config(bundle["N"], overrides={CFG_Q_MIN: 1, CFG_Q_MAX: 3})
    sol1 = copy_solution(bundle["sol"])
    sol2 = copy_solution(bundle["sol"])
    r1 = random_removal(sol1, bundle["customers"], bundle["dist_matrix"], 99, config)
    r2 = random_removal(sol2, bundle["customers"], bundle["dist_matrix"], 99, config)
    np.testing.assert_array_equal(np.sort(r1), np.sort(r2))


# ---------------------------------------------------------------------------
# worst_cost_removal
# ---------------------------------------------------------------------------

def test_worst_cost_removed_count(sol_5_assigned):
    bundle = sol_5_assigned
    config = make_config(bundle["N"], overrides={CFG_Q_MIN: 1, CFG_Q_MAX: 3,
                                                  CFG_WORST_NOISE: 0.0})
    removed = worst_cost_removal(bundle["sol"], bundle["customers"],
                                  bundle["dist_matrix"], 0, config)
    assert len(removed) >= 1


def test_worst_cost_removed_unassigned(sol_5_assigned):
    bundle = sol_5_assigned
    config = make_config(bundle["N"], overrides={CFG_Q_MIN: 1, CFG_Q_MAX: 3})
    removed = worst_cost_removal(bundle["sol"], bundle["customers"],
                                  bundle["dist_matrix"], 3, config)
    for c in removed:
        assert bundle["sol"][SOL_CUST_VEHICLE][c] == -1


# ---------------------------------------------------------------------------
# shaw_removal
# ---------------------------------------------------------------------------

def test_shaw_removed_count(sol_5_assigned):
    bundle = sol_5_assigned
    config = make_config(bundle["N"], overrides={CFG_Q_MIN: 1, CFG_Q_MAX: 3,
                                                  CFG_SHAW_RANDOMNESS: 100.0})
    removed = shaw_removal(bundle["sol"], bundle["customers"],
                           bundle["dist_matrix"], 7, config)
    assert len(removed) >= 1


def test_shaw_valid_indices(sol_5_assigned):
    bundle = sol_5_assigned
    config = make_config(bundle["N"], overrides={CFG_Q_MIN: 1, CFG_Q_MAX: 3})
    removed = shaw_removal(bundle["sol"], bundle["customers"],
                           bundle["dist_matrix"], 42, config)
    for c in removed:
        assert 0 <= c < bundle["N"]


# ---------------------------------------------------------------------------
# zone_removal
# ---------------------------------------------------------------------------

def test_zone_removed_valid(sol_5_assigned):
    bundle = sol_5_assigned
    config = make_config(bundle["N"])
    removed = zone_removal(bundle["sol"], bundle["customers"],
                           bundle["dist_matrix"], 5, config)
    assert len(removed) >= 1
    for c in removed:
        assert 0 <= c < bundle["N"]


# ---------------------------------------------------------------------------
# DESTROY_OPS list
# ---------------------------------------------------------------------------

def test_destroy_dispatch_callable():
    assert callable(destroy_dispatch)
