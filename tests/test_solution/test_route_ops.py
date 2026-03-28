"""Tests for single-route operations -- tuple-based solution. All i64."""
import numpy as np
import pytest

from src.solution.structure import create_solution
from src.solution.route_ops import (
    insert_stop,
    remove_stop,
    swap_stops_within,
    reverse_segment,
)
from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, ACT_PAD, VEH_TRUCK, VEH_BIKE, COL_DEMAND,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_TRUCK_LENGTHS,
    SOL_BIKE_STOPS, SOL_BIKE_ACTIONS, SOL_BIKE_LENGTHS,
    SOL_TRUCK_DISTANCES, SOL_BIKE_DISTANCES,
    SOL_CUST_VEHICLE, SOL_CUST_VTYPE, SOL_CUST_ROUTE_POS,
)


# ---------------------------------------------------------------------------
# insert_stop
# ---------------------------------------------------------------------------

def test_insert_empty_route(tiny_instance, tiny_dist_matrix):
    """Insert C0 at pos=0 in empty truck route -> length=1."""
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, tiny_dist_matrix, custs)

    assert sol[SOL_TRUCK_LENGTHS][0] == 1
    assert sol[SOL_TRUCK_STOPS][0, 0] == 0
    assert sol[SOL_TRUCK_ACTIONS][0, 0] == ACT_DELIVER
    assert sol[SOL_TRUCK_DISTANCES][0] > 0


def test_insert_middle_shifts(tiny_instance, tiny_dist_matrix):
    """Insert C1 at pos=1 in route [C3, C4] -> [C3, C1, C4]."""
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix

    insert_stop(sol, VEH_BIKE, 0, 0, 3, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 1, 4, ACT_DELIVER, dm, custs)
    assert sol[SOL_BIKE_LENGTHS][0] == 2

    insert_stop(sol, VEH_BIKE, 0, 1, 1, ACT_DELIVER, dm, custs)
    assert sol[SOL_BIKE_LENGTHS][0] == 3
    assert sol[SOL_BIKE_STOPS][0, 0] == 3
    assert sol[SOL_BIKE_STOPS][0, 1] == 1
    assert sol[SOL_BIKE_STOPS][0, 2] == 4


def test_insert_updates_customer_index(tiny_instance, tiny_dist_matrix):
    """DELIVER insert sets cust_vehicle, cust_route_pos."""
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, tiny_dist_matrix, custs)

    assert sol[SOL_CUST_VEHICLE][0] == 0
    assert sol[SOL_CUST_ROUTE_POS][0] == 0


def test_insert_reload_no_customer_index(tiny_instance, tiny_dist_matrix):
    """RELOAD does not update customer index for delivery."""
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    insert_stop(sol, VEH_BIKE, 0, 0, 3, ACT_RELOAD, tiny_dist_matrix, custs)

    assert sol[SOL_CUST_VEHICLE][3] == -1


def test_insert_shifts_route_pos(tiny_instance, tiny_dist_matrix):
    """After insert at pos=0, subsequent customers get updated route_pos."""
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix

    insert_stop(sol, VEH_BIKE, 0, 0, 3, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 1, 4, ACT_DELIVER, dm, custs)

    insert_stop(sol, VEH_BIKE, 0, 0, 1, ACT_DELIVER, dm, custs)

    # All 3 customers assigned to bike 0, global_vid = n_trucks(1) + 0 = 1
    assert sol[SOL_CUST_VEHICLE][1] == 1
    assert sol[SOL_CUST_VEHICLE][3] == 1
    assert sol[SOL_CUST_VEHICLE][4] == 1
    assert sol[SOL_BIKE_LENGTHS][0] == 3


# ---------------------------------------------------------------------------
# remove_stop
# ---------------------------------------------------------------------------

def test_remove_single(tiny_instance, tiny_dist_matrix):
    """Remove only stop -> empty route."""
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix
    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, dm, custs)

    cust, act = remove_stop(sol, VEH_TRUCK, 0, 0, dm, custs)
    assert cust == 0
    assert sol[SOL_TRUCK_LENGTHS][0] == 0


def test_remove_shifts_left(tiny_instance, tiny_dist_matrix):
    """Remove middle -> remaining shift left."""
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix

    insert_stop(sol, VEH_BIKE, 0, 0, 3, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 1, 1, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 2, 4, ACT_DELIVER, dm, custs)

    remove_stop(sol, VEH_BIKE, 0, 1, dm, custs)
    assert sol[SOL_BIKE_LENGTHS][0] == 2
    assert sol[SOL_BIKE_STOPS][0, 0] == 3
    assert sol[SOL_BIKE_STOPS][0, 1] == 4


def test_remove_clears_customer_index(tiny_instance, tiny_dist_matrix):
    """Removing DELIVER sets cust_vehicle = -1."""
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix

    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, dm, custs)
    assert sol[SOL_CUST_VEHICLE][0] == 0

    remove_stop(sol, VEH_TRUCK, 0, 0, dm, custs)
    assert sol[SOL_CUST_VEHICLE][0] == -1


# ---------------------------------------------------------------------------
# swap_stops_within
# ---------------------------------------------------------------------------

def test_swap_first_last(tiny_instance, tiny_dist_matrix):
    """Route [C3, C1, C4] swap(0,2) -> [C4, C1, C3]."""
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix

    insert_stop(sol, VEH_BIKE, 0, 0, 3, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 1, 1, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 2, 4, ACT_DELIVER, dm, custs)

    swap_stops_within(sol, VEH_BIKE, 0, 0, 2, dm)

    assert sol[SOL_BIKE_STOPS][0, 0] == 4
    assert sol[SOL_BIKE_STOPS][0, 1] == 1
    assert sol[SOL_BIKE_STOPS][0, 2] == 3
    assert sol[SOL_BIKE_LENGTHS][0] == 3


# ---------------------------------------------------------------------------
# reverse_segment
# ---------------------------------------------------------------------------

def test_reverse_middle(tiny_instance, tiny_dist_matrix):
    """Route [C0, C3, C4, C1, C2], reverse [1,3] -> [C0, C1, C4, C3, C2]."""
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix

    for i, c in enumerate([0, 3, 4, 1, 2]):
        insert_stop(sol, VEH_TRUCK, 0, i, c, ACT_DELIVER, dm, custs)

    reverse_segment(sol, VEH_TRUCK, 0, 1, 3, dm)

    assert sol[SOL_TRUCK_STOPS][0, 0] == 0
    assert sol[SOL_TRUCK_STOPS][0, 1] == 1
    assert sol[SOL_TRUCK_STOPS][0, 2] == 4
    assert sol[SOL_TRUCK_STOPS][0, 3] == 3
    assert sol[SOL_TRUCK_STOPS][0, 4] == 2


def test_reverse_entire_route(tiny_instance, tiny_dist_matrix):
    """Reverse [C3, C4] -> [C4, C3]."""
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix

    insert_stop(sol, VEH_BIKE, 0, 0, 3, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 1, 4, ACT_DELIVER, dm, custs)

    reverse_segment(sol, VEH_BIKE, 0, 0, 1, dm)

    assert sol[SOL_BIKE_STOPS][0, 0] == 4
    assert sol[SOL_BIKE_STOPS][0, 1] == 3
