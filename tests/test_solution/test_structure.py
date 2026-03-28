"""Tests for solution structure: create, copy, rebuild_index — Data Model v2."""
import numpy as np
import pytest

from src.solution.structure import create_solution, copy_solution, rebuild_index
from src.data.constants import ACT_DELIVER, ACT_PICKUP, ACT_PAD


# ---------------------------------------------------------------------------
# create_solution
# ---------------------------------------------------------------------------

def test_basic_shapes():
    """create_solution(3, 5, n_sku=2) produces arrays with correct shapes."""
    sol = create_solution(3, 5, n_sku=2, max_route_len=100)

    assert sol["stops"].shape == (3, 100)
    assert sol["actions"].shape == (3, 100)
    assert sol["lengths"].shape == (3,)
    assert sol["loads_kg"].shape == (3,)
    assert sol["loads_cbm"].shape == (3,)
    assert sol["distances"].shape == (3,)
    assert sol["cust_vehicle"].shape == (5,)
    assert sol["cust_route_pos"].shape == (5,)
    assert sol["vehicle_sku_qty"].shape == (3, 5, 2)


def test_initial_fill_values():
    """All stops = -1, actions = ACT_PAD, lengths = 0."""
    sol = create_solution(3, 5)

    assert np.all(sol["stops"] == -1)
    assert np.all(sol["actions"] == ACT_PAD)
    assert np.all(sol["lengths"] == 0)
    assert np.all(sol["loads_kg"] == 0.0)
    assert np.all(sol["loads_cbm"] == 0.0)
    assert np.all(sol["distances"] == 0.0)
    assert np.all(sol["cust_vehicle"] == -1)
    assert np.all(sol["cust_route_pos"] == -1)


def test_dtypes():
    sol = create_solution(2, 5)

    assert sol["stops"].dtype == np.int32
    assert sol["actions"].dtype == np.int8
    assert sol["lengths"].dtype == np.int32
    assert sol["loads_kg"].dtype == np.float64
    assert sol["cust_vehicle"].dtype == np.int32
    assert sol["cust_route_pos"].dtype == np.int32


def test_config_stored():
    sol = create_solution(3, 20, n_sku=10, max_route_len=50)
    assert sol["n_vehicles"] == 3
    assert sol["n_customers"] == 20
    assert sol["n_sku"] == 10
    assert sol["max_route_len"] == 50


def test_transfers_initial():
    """Transfers array starts empty with TR_COLS columns."""
    sol = create_solution(2, 5)
    assert sol["transfers"].shape[0] == 0
    assert sol["transfers"].shape[1] == 6


def test_dict_access():
    sol = create_solution(2, 5)
    _ = sol["stops"]
    _ = sol["cust_vehicle"]


# ---------------------------------------------------------------------------
# copy_solution
# ---------------------------------------------------------------------------

def test_copy_independent():
    """Modifying copy does not alter original."""
    sol = create_solution(2, 5)
    sol["stops"][0, 0] = 42
    sol["lengths"][0] = 1
    sol["cust_vehicle"][0] = 0

    cp = copy_solution(sol)
    cp["stops"][0, 0] = 99
    cp["lengths"][0] = 0
    cp["cust_vehicle"][0] = -1

    assert sol["stops"][0, 0] == 42
    assert sol["lengths"][0] == 1
    assert sol["cust_vehicle"][0] == 0


def test_copy_preserves_values():
    sol = create_solution(2, 5)
    sol["stops"][1, 0] = 3
    sol["actions"][1, 0] = ACT_DELIVER
    sol["lengths"][1] = 1

    cp = copy_solution(sol)
    assert cp["stops"][1, 0] == 3
    assert cp["actions"][1, 0] == ACT_DELIVER
    assert cp["lengths"][1] == 1


def test_copy_no_shared_memory():
    sol = create_solution(2, 5)
    cp = copy_solution(sol)
    assert not np.shares_memory(sol["stops"], cp["stops"])
    assert not np.shares_memory(sol["cust_vehicle"], cp["cust_vehicle"])


def test_copy_scalars():
    sol = create_solution(3, 10)
    cp = copy_solution(sol)
    assert cp["n_vehicles"] == 3
    assert cp["n_customers"] == 10
    assert cp["max_route_len"] == sol["max_route_len"]


# ---------------------------------------------------------------------------
# rebuild_index
# ---------------------------------------------------------------------------

def test_rebuild_empty():
    """Empty solution: all = -1."""
    sol = create_solution(2, 5)
    loc_to_cust = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    rebuild_index(sol, loc_to_cust)
    assert np.all(sol["cust_vehicle"] == -1)
    assert np.all(sol["cust_route_pos"] == -1)


def test_rebuild_single_route():
    """Vehicle 0 has route with location 1 and 2 as DELIVER."""
    sol = create_solution(2, 5)
    sol["stops"][0, 0] = 1   # loc 1
    sol["stops"][0, 1] = 2   # loc 2
    sol["actions"][0, 0] = ACT_DELIVER
    sol["actions"][0, 1] = ACT_DELIVER
    sol["lengths"][0] = 2

    loc_to_cust = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    rebuild_index(sol, loc_to_cust)

    assert sol["cust_vehicle"][0] == 0
    assert sol["cust_route_pos"][0] == 0
    assert sol["cust_vehicle"][1] == 0
    assert sol["cust_route_pos"][1] == 1
    for c in [2, 3, 4]:
        assert sol["cust_vehicle"][c] == -1


def test_rebuild_pickup_not_counted():
    """PICKUP stops don't update customer index."""
    sol = create_solution(2, 5)
    sol["stops"][0, 0] = 3   # loc 3 as DELIVER
    sol["stops"][0, 1] = 3   # loc 3 as PICKUP
    sol["stops"][0, 2] = 4   # loc 4 as DELIVER
    sol["actions"][0, 0] = ACT_DELIVER
    sol["actions"][0, 1] = ACT_PICKUP
    sol["actions"][0, 2] = ACT_DELIVER
    sol["lengths"][0] = 3

    loc_to_cust = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    rebuild_index(sol, loc_to_cust)

    assert sol["cust_vehicle"][2] == 0   # C2 at loc 3 DELIVER
    assert sol["cust_route_pos"][2] == 0
    assert sol["cust_vehicle"][3] == 0   # C3 at loc 4 DELIVER
    assert sol["cust_route_pos"][3] == 2


def test_rebuild_overwrites_stale():
    sol = create_solution(2, 5)
    sol["cust_vehicle"][2] = 99
    sol["cust_route_pos"][2] = 5

    loc_to_cust = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    rebuild_index(sol, loc_to_cust)

    assert sol["cust_vehicle"][2] == -1
    assert sol["cust_route_pos"][2] == -1
