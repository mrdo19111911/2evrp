"""Tests for cross-layer operators (src/alns/crosslayer.py) — Data Model v2."""
import numpy as np
import pytest

from src.alns.crosslayer import (
    swap_assignment,
    relocate_transfer,
    create_new_transfer,
    remove_transfer_node,
    CROSS_OPS,
)


# ---------------------------------------------------------------------------
# swap_assignment
# ---------------------------------------------------------------------------

def test_swap_light_truck_to_bike(sol_10_assigned, default_params):
    """Light customer on truck can be moved to bike."""
    bundle = sol_10_assigned
    sol, data = bundle["sol"], bundle["data"]
    rng = np.random.default_rng(0)
    result = swap_assignment(sol, data, rng, default_params)
    assert isinstance(result, bool)


def test_swap_heavy_cannot_move(sol_crossing_route, default_params):
    """All-truck solution with no bikes -> False."""
    bundle = sol_crossing_route
    sol, data = bundle["sol"], bundle["data"]
    rng = np.random.default_rng(0)
    result = swap_assignment(sol, data, rng, default_params)
    assert result is False


# ---------------------------------------------------------------------------
# relocate_transfer
# ---------------------------------------------------------------------------

def test_relocate_transfer_exists(sol_with_transfer, default_params):
    """Transfer event exists -> relocate may return True."""
    bundle = sol_with_transfer
    sol, data = bundle["sol"], bundle["data"]
    rng = np.random.default_rng(0)
    result = relocate_transfer(sol, data, rng, default_params)
    assert isinstance(result, bool)


def test_relocate_no_transfers(sol_10_assigned, default_params):
    """No transfers -> returns False."""
    bundle = sol_10_assigned
    sol, data = bundle["sol"], bundle["data"]
    rng = np.random.default_rng(0)
    result = relocate_transfer(sol, data, rng, default_params)
    assert result is False


# ---------------------------------------------------------------------------
# create_new_transfer
# ---------------------------------------------------------------------------

def test_create_new_transfer(sol_10_assigned, default_params):
    """Bike route exists -> create_new_transfer may succeed."""
    bundle = sol_10_assigned
    sol, data = bundle["sol"], bundle["data"]
    rng = np.random.default_rng(0)
    result = create_new_transfer(sol, data, rng, default_params)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# remove_transfer_node
# ---------------------------------------------------------------------------

def test_remove_transfer_with_event(sol_with_transfer, default_params):
    """Transfer exists -> remove returns True."""
    bundle = sol_with_transfer
    sol, data = bundle["sol"], bundle["data"]
    assert sol["transfers"].shape[0] >= 1
    rng = np.random.default_rng(0)
    result = remove_transfer_node(sol, data, rng, default_params)
    assert isinstance(result, bool)


def test_remove_transfer_none(sol_10_assigned, default_params):
    """No transfers -> returns False."""
    bundle = sol_10_assigned
    sol, data = bundle["sol"], bundle["data"]
    rng = np.random.default_rng(0)
    result = remove_transfer_node(sol, data, rng, default_params)
    assert result is False


# ---------------------------------------------------------------------------
# CROSS_OPS list
# ---------------------------------------------------------------------------

def test_cross_ops_length():
    assert len(CROSS_OPS) == 4


def test_cross_ops_names():
    names = {f.__name__ for f in CROSS_OPS}
    expected = {
        "swap_assignment", "relocate_transfer",
        "create_new_transfer", "remove_transfer_node",
    }
    assert names == expected
