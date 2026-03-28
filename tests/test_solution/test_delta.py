"""Tests for delta cost evaluation -- tuple-based solution. All i64."""
import numpy as np
import pytest

from src.solution.structure import create_solution
from src.solution.route_ops import insert_stop, remove_stop
from src.solution.delta import (
    insertion_cost_delta,
    removal_cost_delta,
    best_insertion_pos,
    find_best_insertion_all_routes,
)
from src.data.constants import (
    ACT_DELIVER, ACT_PAD, VEH_TRUCK, VEH_BIKE,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_TRUCK_LENGTHS,
    SOL_BIKE_STOPS, SOL_BIKE_ACTIONS, SOL_BIKE_LENGTHS,
    SOL_TRUCK_DISTANCES, SOL_BIKE_DISTANCES,
)


# ---------------------------------------------------------------------------
# insertion_cost_delta
# ---------------------------------------------------------------------------

def test_insert_empty_route(tiny_instance, tiny_dist_matrix):
    """Empty route: delta = round-trip to customer location."""
    sol = create_solution(1, 2, 5)
    dm = tiny_dist_matrix
    delta = insertion_cost_delta(sol, VEH_TRUCK, 0, 0, 0, dm)
    # depot->C0->depot: dm[0, 1] + dm[1, 0]
    expected = dm[0, 1] + dm[1, 0]
    assert delta == expected


def test_insert_consistency(tiny_instance, tiny_dist_matrix):
    """old_distance + delta == new_distance after actual insert."""
    sol = create_solution(1, 2, 5)
    dm = tiny_dist_matrix
    custs = tiny_instance["customers"]

    insert_stop(sol, VEH_BIKE, 0, 0, 3, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 1, 4, ACT_DELIVER, dm, custs)
    old_dist = sol[SOL_BIKE_DISTANCES][0]

    delta = insertion_cost_delta(sol, VEH_BIKE, 0, 1, 1, dm)

    insert_stop(sol, VEH_BIKE, 0, 1, 1, ACT_DELIVER, dm, custs)
    new_dist = sol[SOL_BIKE_DISTANCES][0]

    assert old_dist + delta == new_dist


# ---------------------------------------------------------------------------
# removal_cost_delta
# ---------------------------------------------------------------------------

def test_remove_only_stop(tiny_instance, tiny_dist_matrix):
    """Remove only stop: delta = -(round-trip distance)."""
    sol = create_solution(1, 2, 5)
    dm = tiny_dist_matrix
    custs = tiny_instance["customers"]

    insert_stop(sol, VEH_TRUCK, 0, 0, 0, ACT_DELIVER, dm, custs)
    dist_before = sol[SOL_TRUCK_DISTANCES][0]
    delta = removal_cost_delta(sol, VEH_TRUCK, 0, 0, dm)
    assert delta == -dist_before


def test_removal_consistency(tiny_instance, tiny_dist_matrix):
    """old + delta == new after actual removal."""
    sol = create_solution(1, 2, 5)
    dm = tiny_dist_matrix
    custs = tiny_instance["customers"]

    insert_stop(sol, VEH_BIKE, 0, 0, 3, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 1, 1, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 2, 4, ACT_DELIVER, dm, custs)

    old_dist = sol[SOL_BIKE_DISTANCES][0]
    delta = removal_cost_delta(sol, VEH_BIKE, 0, 1, dm)

    remove_stop(sol, VEH_BIKE, 0, 1, dm, custs)
    new_dist = sol[SOL_BIKE_DISTANCES][0]

    assert old_dist + delta == new_dist


def test_removal_nonpositive(tiny_instance, tiny_dist_matrix):
    """Removal delta should be <= 0."""
    sol = create_solution(1, 2, 5)
    dm = tiny_dist_matrix
    custs = tiny_instance["customers"]

    insert_stop(sol, VEH_BIKE, 0, 0, 3, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 1, 1, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 2, 4, ACT_DELIVER, dm, custs)

    for pos in range(3):
        delta = removal_cost_delta(sol, VEH_BIKE, 0, pos, dm)
        assert delta <= 0, f"pos={pos} delta={delta}"


# ---------------------------------------------------------------------------
# best_insertion_pos
# ---------------------------------------------------------------------------

def test_best_pos_empty(tiny_instance, tiny_dist_matrix):
    """Empty route: only pos=0 available."""
    sol = create_solution(1, 2, 5)
    dm = tiny_dist_matrix
    pos, delta = best_insertion_pos(sol, VEH_TRUCK, 0, 0, dm)
    assert pos == 0
    assert delta > 0


def test_best_pos_minimizes(tiny_instance, tiny_dist_matrix):
    """Best pos should have smallest delta."""
    sol = create_solution(1, 2, 5)
    dm = tiny_dist_matrix
    custs = tiny_instance["customers"]

    insert_stop(sol, VEH_BIKE, 0, 0, 3, ACT_DELIVER, dm, custs)
    insert_stop(sol, VEH_BIKE, 0, 1, 4, ACT_DELIVER, dm, custs)

    pos, delta = best_insertion_pos(sol, VEH_BIKE, 0, 1, dm)

    # Verify this is actually the best
    for p in range(sol[SOL_BIKE_LENGTHS][0] + 1):
        d = insertion_cost_delta(sol, VEH_BIKE, 0, p, 1, dm)
        assert d >= delta


# ---------------------------------------------------------------------------
# find_best_insertion_all_routes
# ---------------------------------------------------------------------------

def test_find_best_single_route(tiny_instance, tiny_dist_matrix):
    """One bike with space -> returns that route."""
    sol = create_solution(1, 2, 5)
    dm = tiny_dist_matrix
    custs = tiny_instance["customers"]
    vehicles = tiny_instance["vehicles"]

    insert_stop(sol, VEH_BIKE, 0, 0, 3, ACT_DELIVER, dm, custs)

    vid, pos, delta = find_best_insertion_all_routes(
        sol, VEH_BIKE, 4, dm, custs, vehicles,
    )
    assert vid >= 0
    assert delta < 2_000_000_000_000
