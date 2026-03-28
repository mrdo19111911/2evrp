"""Tests for O(1) constraint checks -- tuple-based solution. All i64."""
import numpy as np
import pytest

from src.solution.structure import create_solution
from src.solution.route_ops import insert_stop
from src.solution.check import (
    can_insert_customer,
    can_swap_customers,
    check_route_capacity_quick,
    check_all_assigned,
)
from src.data.constants import (
    ACT_DELIVER, VEH_TRUCK, VEH_BIKE, COL_DEMAND,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_TRUCK_LENGTHS,
    SOL_BIKE_STOPS, SOL_BIKE_ACTIONS, SOL_BIKE_LENGTHS,
    SOL_TRUCK_LOADS, SOL_BIKE_LOADS,
    SOL_CUST_VEHICLE,
)
from src.data.cost import TRUCK_CAPACITY_G, BIKE_CAPACITY_G


# ---------------------------------------------------------------------------
# can_insert_customer
# ---------------------------------------------------------------------------

def test_truck_unrestricted(tiny_instance, tiny_dist_matrix):
    """C1 (not restricted) in truck (cap=2000kg) -> True."""
    sol = create_solution(1, 2, 5)
    customers = tiny_instance["customers"]
    result = can_insert_customer(
        sol, VEH_TRUCK, 0, 1, customers, TRUCK_CAPACITY_G,
    )
    assert result is True


def test_truck_restricted_customer(tiny_instance, tiny_dist_matrix):
    """C2 (restricted=1) in truck -> should be False (restricted means bike-only)."""
    sol = create_solution(1, 2, 5)
    customers = tiny_instance["customers"]
    result = can_insert_customer(
        sol, VEH_TRUCK, 0, 2, customers, TRUCK_CAPACITY_G,
    )
    assert result is False


def test_bike_allowed_customer(tiny_instance, tiny_dist_matrix):
    """C2 (restricted=1, bike-only, demand=10kg) on bike -> True."""
    sol = create_solution(1, 2, 5)
    customers = tiny_instance["customers"]
    result = can_insert_customer(
        sol, VEH_BIKE, 0, 2, customers, BIKE_CAPACITY_G,
    )
    assert result is True


def test_capacity_overload_bike(tiny_instance, tiny_dist_matrix):
    """C0 (demand=100kg=100000g) in bike (cap=60kg=60000g) -> False."""
    sol = create_solution(1, 2, 5)
    customers = tiny_instance["customers"]
    result = can_insert_customer(
        sol, VEH_BIKE, 0, 0, customers, BIKE_CAPACITY_G,
    )
    assert result is False


# ---------------------------------------------------------------------------
# can_swap_customers
# ---------------------------------------------------------------------------

def test_swap_customers_basic(tiny_instance, tiny_dist_matrix):
    """Swap two assigned customers -- basic feasibility."""
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix
    vehicles = tiny_instance["vehicles"]

    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 0, 3, ACT_DELIVER, dm, custs)

    result = can_swap_customers(sol, 0, 3, custs, vehicles)
    # C0=100kg cannot go to bike (60kg cap), so should be False
    assert result is False


# ---------------------------------------------------------------------------
# check_route_capacity_quick
# ---------------------------------------------------------------------------

def test_under_capacity(tiny_instance, tiny_dist_matrix):
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix
    insert_stop(sol, VEH_BIKE, 0, 0, 3, ACT_DELIVER, dm, custs)  # C3: 8kg
    insert_stop(sol, VEH_BIKE, 0, 1, 4, ACT_DELIVER, dm, custs)  # C4: 5kg
    assert check_route_capacity_quick(sol, VEH_BIKE, 0, BIKE_CAPACITY_G) is True


def test_over_capacity(tiny_instance, tiny_dist_matrix):
    sol = create_solution(1, 2, 5)
    sol[SOL_BIKE_LOADS][0] = 65000  # 65kg in grams > 60kg cap
    assert check_route_capacity_quick(sol, VEH_BIKE, 0, BIKE_CAPACITY_G) is False


# ---------------------------------------------------------------------------
# check_all_assigned
# ---------------------------------------------------------------------------

def test_none_assigned(tiny_instance, tiny_dist_matrix):
    sol = create_solution(1, 2, 5)
    assert check_all_assigned(sol, 5) is False


def test_all_assigned(tiny_instance, tiny_dist_matrix):
    sol = create_solution(1, 2, 5)
    custs = tiny_instance["customers"]
    dm = tiny_dist_matrix
    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 0, 1, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 1, 2, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 1, 0, 3, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 1, 1, 4, ACT_DELIVER, dm, custs)
    assert check_all_assigned(sol, 5) is True
