"""Tests for satellite_ops -- pre-allocated i64 array, tuple-based solution."""
import numpy as np
import pytest

from src.solution.structure import create_solution
from src.solution.satellite_ops import (
    add_satellite_event,
    remove_satellite_event,
    remove_satellites_for_customer,
    remove_satellites_for_bike,
    update_satellite_time,
    get_satellites,
)
from src.data.constants import (
    SOL_SATELLITES, SOL_META, META_N_SATELLITES,
    SAT_CUST, SAT_BIKE, SAT_TRUCK, SAT_KG, SAT_TIME,
)


def test_add_satellite_event():
    sol = create_solution(2, 3, 10)
    add_satellite_event(sol, customer_idx=5, bike_id=1, truck_id=0,
                        transfer_g=20000, planned_time_s=100)

    assert sol[SOL_META][META_N_SATELLITES] == 1
    sats = get_satellites(sol)
    assert sats.shape == (1, 5)
    assert sats[0, SAT_CUST] == 5
    assert sats[0, SAT_BIKE] == 1
    assert sats[0, SAT_TRUCK] == 0
    assert sats[0, SAT_KG] == 20000
    assert sats[0, SAT_TIME] == 100


def test_add_multiple_satellites():
    sol = create_solution(2, 3, 10)
    add_satellite_event(sol, 5, 1, 0, 20000, 100)
    add_satellite_event(sol, 7, 2, 1, 15000, 200)

    assert sol[SOL_META][META_N_SATELLITES] == 2
    sats = get_satellites(sol)
    assert sats.shape == (2, 5)


def test_remove_satellite_by_index():
    sol = create_solution(2, 3, 10)
    add_satellite_event(sol, 5, 1, 0, 20000, 100)
    add_satellite_event(sol, 7, 2, 1, 15000, 200)

    removed = remove_satellite_event(sol, sat_idx=0)

    assert sol[SOL_META][META_N_SATELLITES] == 1
    assert removed[SAT_CUST] == 5
    # Remaining should be the second event (swapped to index 0)
    sats = get_satellites(sol)
    assert sats[0, SAT_CUST] == 7


def test_remove_satellite_by_customer():
    sol = create_solution(2, 3, 10)
    add_satellite_event(sol, 5, 1, 0, 20000, 100)
    add_satellite_event(sol, 7, 2, 1, 15000, 200)

    removed = remove_satellite_event(sol, customer_idx=7)

    assert sol[SOL_META][META_N_SATELLITES] == 1
    assert removed[SAT_CUST] == 7


def test_remove_satellite_not_found():
    sol = create_solution(2, 3, 10)
    add_satellite_event(sol, 5, 1, 0, 20000, 100)

    removed = remove_satellite_event(sol, customer_idx=99)
    assert removed is None
    assert sol[SOL_META][META_N_SATELLITES] == 1


def test_remove_satellites_for_customer():
    sol = create_solution(2, 3, 10)
    add_satellite_event(sol, 5, 1, 0, 20000, 100)
    add_satellite_event(sol, 5, 2, 1, 15000, 200)
    add_satellite_event(sol, 7, 0, 0, 10000, 300)

    count = remove_satellites_for_customer(sol, 5)

    assert count == 2
    assert sol[SOL_META][META_N_SATELLITES] == 1
    sats = get_satellites(sol)
    assert sats[0, SAT_CUST] == 7


def test_remove_satellites_for_bike():
    sol = create_solution(2, 3, 10)
    add_satellite_event(sol, 5, 1, 0, 20000, 100)
    add_satellite_event(sol, 7, 1, 1, 15000, 200)
    add_satellite_event(sol, 8, 2, 0, 10000, 300)

    count = remove_satellites_for_bike(sol, 1)

    assert count == 2
    assert sol[SOL_META][META_N_SATELLITES] == 1
    sats = get_satellites(sol)
    assert sats[0, SAT_BIKE] == 2


def test_update_satellite_time():
    sol = create_solution(2, 3, 10)
    add_satellite_event(sol, 5, 1, 0, 20000, 100)

    update_satellite_time(sol, 0, 250)

    assert sol[SOL_SATELLITES][0, SAT_TIME] == 250


def test_get_satellites_empty():
    sol = create_solution(2, 3, 10)
    sats = get_satellites(sol)
    assert sats.shape == (0, 5)
