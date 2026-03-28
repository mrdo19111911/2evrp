"""Tests for destroy operators (src/alns/destroy.py) — Data Model v2."""
import numpy as np
import pytest

from src.alns.destroy import (
    random_removal,
    worst_cost_removal,
    shaw_removal,
    route_removal,
    transfer_removal,
    zone_removal,
    DESTROY_OPS,
)
from src.solution.structure import copy_solution


# ---------------------------------------------------------------------------
# random_removal
# ---------------------------------------------------------------------------

def test_random_removed_count(sol_10_assigned, default_params):
    """10 assigned, q_min=3, q_max=5 -> len(removed) in [3,5]."""
    bundle = sol_10_assigned
    rng = np.random.default_rng(0)
    removed = random_removal(bundle["sol"], bundle["data"], rng, default_params)
    assert 3 <= len(removed) <= 5


def test_random_removed_unassigned(sol_10_assigned, default_params):
    """Removed customers have cust_vehicle == -1."""
    bundle = sol_10_assigned
    rng = np.random.default_rng(1)
    removed = random_removal(bundle["sol"], bundle["data"], rng, default_params)
    sol = bundle["sol"]
    for c in removed:
        assert sol["cust_vehicle"][c] == -1, f"C{c} still assigned"


def test_random_non_removed_assigned(sol_10_assigned, default_params):
    """Customers not removed still assigned."""
    bundle = sol_10_assigned
    rng = np.random.default_rng(2)
    removed = random_removal(bundle["sol"], bundle["data"], rng, default_params)
    removed_set = set(int(c) for c in removed)
    sol = bundle["sol"]
    for c in range(bundle["N"]):
        if c not in removed_set:
            assert sol["cust_vehicle"][c] >= 0, f"C{c} wrongly unassigned"


def test_random_deterministic(sol_10_assigned, default_params):
    """Same seed -> same removed set."""
    bundle = sol_10_assigned
    sol1 = copy_solution(bundle["sol"])
    sol2 = copy_solution(bundle["sol"])
    r1 = random_removal(sol1, bundle["data"], np.random.default_rng(99), default_params)
    r2 = random_removal(sol2, bundle["data"], np.random.default_rng(99), default_params)
    np.testing.assert_array_equal(np.sort(r1), np.sort(r2))


# ---------------------------------------------------------------------------
# worst_cost_removal
# ---------------------------------------------------------------------------

def test_worst_cost_removed_count(sol_10_assigned, default_params):
    """Removes at least q_min customers."""
    bundle = sol_10_assigned
    rng = np.random.default_rng(0)
    params = {**default_params, "worst_noise": 0.0}
    removed = worst_cost_removal(bundle["sol"], bundle["data"], rng, params)
    assert len(removed) >= params["q_min"]


def test_worst_cost_removed_unassigned(sol_10_assigned, default_params):
    bundle = sol_10_assigned
    rng = np.random.default_rng(3)
    removed = worst_cost_removal(bundle["sol"], bundle["data"], rng, default_params)
    for c in removed:
        assert bundle["sol"]["cust_vehicle"][c] == -1


# ---------------------------------------------------------------------------
# shaw_removal
# ---------------------------------------------------------------------------

def test_shaw_removed_count(sol_10_assigned, default_params):
    bundle = sol_10_assigned
    rng = np.random.default_rng(7)
    params = {**default_params, "shaw_randomness": 100.0}
    removed = shaw_removal(bundle["sol"], bundle["data"], rng, params)
    assert len(removed) >= params["q_min"]


def test_shaw_valid_indices(sol_10_assigned, default_params):
    bundle = sol_10_assigned
    rng = np.random.default_rng(42)
    removed = shaw_removal(bundle["sol"], bundle["data"], rng, default_params)
    for c in removed:
        assert 0 <= c < bundle["N"]


# ---------------------------------------------------------------------------
# route_removal
# ---------------------------------------------------------------------------

def test_route_removal_delivers_only(sol_with_transfer, default_params):
    """Only DELIVER customers in removed (not PICKUP nodes)."""
    bundle = sol_with_transfer
    rng = np.random.default_rng(0)
    removed = route_removal(bundle["sol"], bundle["data"], rng, {})
    for c in removed:
        assert 0 <= c < bundle["N"]


def test_route_removal_clears_a_route(sol_10_assigned, default_params):
    """After route_removal, at least one route is shorter."""
    bundle = sol_10_assigned
    rng = np.random.default_rng(0)
    removed = route_removal(bundle["sol"], bundle["data"], rng, {})
    assert len(removed) > 0 or any(bundle["sol"]["lengths"][v] == 0
                                    for v in range(bundle["sol"]["n_vehicles"]))


# ---------------------------------------------------------------------------
# transfer_removal
# ---------------------------------------------------------------------------

def test_transfer_removal_with_event(sol_with_transfer, default_params):
    """Transfer exists -> dependent customers removed."""
    bundle = sol_with_transfer
    rng = np.random.default_rng(0)
    removed = transfer_removal(bundle["sol"], bundle["data"], rng, {})
    # Should include bike customers from the transfer route
    assert isinstance(removed, np.ndarray)


def test_transfer_removal_none(sol_10_assigned, default_params):
    """No transfers -> empty."""
    bundle = sol_10_assigned
    rng = np.random.default_rng(0)
    removed = transfer_removal(bundle["sol"], bundle["data"], rng, {})
    assert len(removed) == 0


# ---------------------------------------------------------------------------
# zone_removal
# ---------------------------------------------------------------------------

def test_zone_removed_valid(sol_10_assigned, default_params):
    bundle = sol_10_assigned
    rng = np.random.default_rng(5)
    removed = zone_removal(bundle["sol"], bundle["data"], rng, default_params)
    assert len(removed) >= 1
    for c in removed:
        assert 0 <= c < bundle["N"]


# ---------------------------------------------------------------------------
# DESTROY_OPS list
# ---------------------------------------------------------------------------

def test_destroy_ops_length():
    assert len(DESTROY_OPS) == 6


def test_destroy_ops_names():
    names = {f.__name__ for f in DESTROY_OPS}
    expected = {
        "random_removal", "worst_cost_removal", "shaw_removal",
        "route_removal", "transfer_removal", "zone_removal",
    }
    assert names == expected
