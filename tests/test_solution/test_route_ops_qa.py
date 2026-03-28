"""QA tests for src/solution/route_ops.py - Single-route operations.
Tuple-based solution format. All i64.

Test plan:
1. insert_stop + remove_stop (roundtrip) - verify complete state recovery
2. remove_stop on empty route (L=0)
3. reverse_segment of length 1 (no-op behavior)
4. swap_stops_within adjacent positions (ensures neighboring swaps work)
"""
import numpy as np
import pytest

from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, ACT_PAD, VEH_TRUCK, VEH_BIKE, COL_DEMAND,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_TRUCK_LENGTHS,
    SOL_BIKE_STOPS, SOL_BIKE_ACTIONS, SOL_BIKE_LENGTHS,
    SOL_TRUCK_LOADS, SOL_BIKE_LOADS,
    SOL_TRUCK_DISTANCES, SOL_BIKE_DISTANCES,
    SOL_CUST_VEHICLE, SOL_CUST_VTYPE, SOL_CUST_ROUTE_POS,
    SOL_META, META_N_TRUCKS, META_MAX_ROUTE_LEN,
)
from src.solution.structure import create_solution, copy_solution
from src.solution.route_ops import (
    insert_stop, remove_stop, swap_stops_within, reverse_segment
)


@pytest.fixture
def minimal_setup():
    """Minimal setup: 1 truck, 1 bike, 5 customers. All i64."""
    n_trucks, n_bikes, n_customers = 1, 1, 5
    sol = create_solution(n_trucks, n_bikes, n_customers, max_route_len=10)

    # i64 (N, 7): [x_m, y_m, demand_g, tw_open_s, tw_close_s, service_s, restricted]
    customers = np.array([
        [    0,     0,     0, 0, 1000000, 0, 0],  # customer 0 (depot-like, unused)
        [10000, 10000,  1000, 0, 1000000, 5000, 0],  # customer 1: 1kg
        [20000, 20000,  2000, 0, 1000000, 5000, 0],  # customer 2: 2kg
        [30000, 30000,  1500, 0, 1000000, 5000, 0],  # customer 3: 1.5kg
        [40000, 40000,  2500, 0, 1000000, 5000, 0],  # customer 4: 2.5kg
    ], dtype=np.int64)

    depot = np.array([0, 0], dtype=np.int64)

    # Build i64 dist_matrix
    coords = np.vstack([depot.reshape(1, 2), customers[:, :2]]).astype(np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.round(np.sqrt((diff ** 2).sum(axis=2))).astype(np.int64)

    return sol, customers, dist_matrix


class TestInsertRemoveRoundtrip:

    def test_insert_deliver_then_remove(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        assert sol[SOL_TRUCK_LENGTHS][vid] == 0
        assert sol[SOL_CUST_VEHICLE][1] == -1

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)

        assert sol[SOL_TRUCK_LENGTHS][vid] == 1
        assert sol[SOL_TRUCK_STOPS][vid, 0] == 1
        assert sol[SOL_TRUCK_ACTIONS][vid, 0] == ACT_DELIVER
        assert sol[SOL_CUST_VEHICLE][1] == 0
        assert sol[SOL_CUST_VTYPE][1] == VEH_TRUCK
        assert sol[SOL_CUST_ROUTE_POS][1] == 0
        assert sol[SOL_TRUCK_LOADS][vid] == 1000  # 1kg = 1000g

        removed_cust, removed_action = remove_stop(sol, vtype, vid, 0, dist_matrix, customers)

        assert removed_cust == 1
        assert removed_action == ACT_DELIVER
        assert sol[SOL_TRUCK_LENGTHS][vid] == 0
        assert sol[SOL_CUST_VEHICLE][1] == -1
        assert sol[SOL_CUST_VTYPE][1] == -1
        assert sol[SOL_CUST_ROUTE_POS][1] == -1
        assert sol[SOL_TRUCK_LOADS][vid] == 0
        assert sol[SOL_TRUCK_STOPS][vid, 0] == -1
        assert sol[SOL_TRUCK_ACTIONS][vid, 0] == ACT_PAD

    def test_insert_multiple_then_remove_middle(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 1, 2, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 2, 3, ACT_DELIVER, dist_matrix, customers)

        assert sol[SOL_TRUCK_LENGTHS][vid] == 3
        assert list(sol[SOL_TRUCK_STOPS][vid, :3]) == [1, 2, 3]

        removed_cust, _ = remove_stop(sol, vtype, vid, 1, dist_matrix, customers)

        assert removed_cust == 2
        assert sol[SOL_TRUCK_LENGTHS][vid] == 2
        assert list(sol[SOL_TRUCK_STOPS][vid, :2]) == [1, 3]
        assert sol[SOL_CUST_ROUTE_POS][3] == 1
        assert sol[SOL_CUST_VEHICLE][2] == -1
        assert sol[SOL_TRUCK_LOADS][vid] == 1000 + 1500  # C1 + C3


class TestRemoveFromEmpty:

    def test_remove_from_empty_truck_route(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        assert sol[SOL_TRUCK_LENGTHS][vid] == 0

        removed_cust, removed_action = remove_stop(sol, vtype, vid, 0, dist_matrix, customers)

        assert removed_cust == -1
        assert removed_action == -1
        assert sol[SOL_TRUCK_LENGTHS][vid] == 0

    def test_remove_from_empty_bike_route(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_BIKE, 0

        assert sol[SOL_BIKE_LENGTHS][vid] == 0

        removed_cust, removed_action = remove_stop(sol, vtype, vid, 0, dist_matrix, customers)

        assert removed_cust == -1
        assert removed_action == -1
        assert sol[SOL_BIKE_LENGTHS][vid] == 0

    def test_remove_out_of_bounds_position(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        assert sol[SOL_TRUCK_LENGTHS][vid] == 1

        removed_cust, removed_action = remove_stop(sol, vtype, vid, 1, dist_matrix, customers)

        assert removed_cust == -1
        assert removed_action == -1
        assert sol[SOL_TRUCK_LENGTHS][vid] == 1


class TestReverseSegmentSingleLength:

    def test_reverse_single_element_no_change(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        dist_before = sol[SOL_TRUCK_DISTANCES][vid]

        reverse_segment(sol, vtype, vid, 0, 0, dist_matrix)

        assert sol[SOL_TRUCK_STOPS][vid, 0] == 1
        assert sol[SOL_TRUCK_ACTIONS][vid, 0] == ACT_DELIVER
        assert sol[SOL_CUST_ROUTE_POS][1] == 0
        assert sol[SOL_TRUCK_DISTANCES][vid] == dist_before

    def test_reverse_single_element_among_three(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 1, 2, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 2, 3, ACT_DELIVER, dist_matrix, customers)

        route_before = list(sol[SOL_TRUCK_STOPS][vid, :3])
        dist_before = sol[SOL_TRUCK_DISTANCES][vid]

        reverse_segment(sol, vtype, vid, 1, 1, dist_matrix)

        route_after = list(sol[SOL_TRUCK_STOPS][vid, :3])
        assert route_before == route_after
        assert sol[SOL_TRUCK_DISTANCES][vid] == dist_before


class TestSwapAdjacentPositions:

    def test_swap_positions_0_and_1(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 1, 2, ACT_DELIVER, dist_matrix, customers)

        assert list(sol[SOL_TRUCK_STOPS][vid, :2]) == [1, 2]

        swap_stops_within(sol, vtype, vid, 0, 1, dist_matrix)

        assert list(sol[SOL_TRUCK_STOPS][vid, :2]) == [2, 1]
        assert sol[SOL_CUST_ROUTE_POS][1] == 1
        assert sol[SOL_CUST_ROUTE_POS][2] == 0

    def test_swap_middle_elements_in_three(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 1, 2, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 2, 3, ACT_DELIVER, dist_matrix, customers)

        swap_stops_within(sol, vtype, vid, 1, 2, dist_matrix)

        assert list(sol[SOL_TRUCK_STOPS][vid, :3]) == [1, 3, 2]
        assert sol[SOL_CUST_ROUTE_POS][2] == 2
        assert sol[SOL_CUST_ROUTE_POS][3] == 1

    def test_swap_same_position(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 1, 2, ACT_DELIVER, dist_matrix, customers)

        route_before = list(sol[SOL_TRUCK_STOPS][vid, :2])
        pos_1_before = int(sol[SOL_CUST_ROUTE_POS][1])
        pos_2_before = int(sol[SOL_CUST_ROUTE_POS][2])

        swap_stops_within(sol, vtype, vid, 0, 0, dist_matrix)

        route_after = list(sol[SOL_TRUCK_STOPS][vid, :2])
        assert route_before == route_after
        assert sol[SOL_CUST_ROUTE_POS][1] == pos_1_before
        assert sol[SOL_CUST_ROUTE_POS][2] == pos_2_before


class TestLoadTracking:

    def test_load_increases_on_insert_deliver(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        assert sol[SOL_TRUCK_LOADS][vid] == 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        assert sol[SOL_TRUCK_LOADS][vid] == 1000  # 1kg

        insert_stop(sol, vtype, vid, 1, 2, ACT_DELIVER, dist_matrix, customers)
        assert sol[SOL_TRUCK_LOADS][vid] == 3000  # 1kg + 2kg

    def test_load_unchanged_on_insert_reload(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        load_after_deliver = sol[SOL_TRUCK_LOADS][vid]

        # Insert RELOAD at a customer node (not -1, since customer must be valid index)
        insert_stop(sol, vtype, vid, 1, 0, ACT_RELOAD, dist_matrix, customers)

        assert sol[SOL_TRUCK_LOADS][vid] == load_after_deliver

    def test_load_decreases_on_remove(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 1, 2, ACT_DELIVER, dist_matrix, customers)
        assert sol[SOL_TRUCK_LOADS][vid] == 3000  # 1kg + 2kg

        remove_stop(sol, vtype, vid, 0, dist_matrix, customers)
        assert sol[SOL_TRUCK_LOADS][vid] == 2000  # 2kg


class TestBikeOperations:

    def test_insert_remove_bike_roundtrip(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_BIKE, 0
        n_trucks = sol[SOL_META][META_N_TRUCKS]

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        assert sol[SOL_CUST_VEHICLE][1] == n_trucks  # bike global_vid
        assert sol[SOL_CUST_VTYPE][1] == VEH_BIKE

        remove_stop(sol, vtype, vid, 0, dist_matrix, customers)
        assert sol[SOL_CUST_VEHICLE][1] == -1


class TestShiftingLogic:

    def test_insert_at_end_updates_length(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 1, 2, ACT_DELIVER, dist_matrix, customers)
        assert sol[SOL_TRUCK_LENGTHS][vid] == 2

        insert_stop(sol, vtype, vid, 2, 3, ACT_DELIVER, dist_matrix, customers)
        assert sol[SOL_TRUCK_LENGTHS][vid] == 3
        assert list(sol[SOL_TRUCK_STOPS][vid, :3]) == [1, 2, 3]

    def test_insert_at_start_shifts_all(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 1, 2, ACT_DELIVER, dist_matrix, customers)

        insert_stop(sol, vtype, vid, 0, 3, ACT_DELIVER, dist_matrix, customers)

        assert sol[SOL_TRUCK_LENGTHS][vid] == 3
        assert list(sol[SOL_TRUCK_STOPS][vid, :3]) == [3, 1, 2]
        assert sol[SOL_CUST_ROUTE_POS][1] == 1
        assert sol[SOL_CUST_ROUTE_POS][2] == 2
        assert sol[SOL_CUST_ROUTE_POS][3] == 0

    def test_remove_updates_all_positions_after(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 1, 2, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 2, 3, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 3, 4, ACT_DELIVER, dist_matrix, customers)

        assert sol[SOL_CUST_ROUTE_POS][4] == 3

        remove_stop(sol, vtype, vid, 0, dist_matrix, customers)

        assert list(sol[SOL_TRUCK_STOPS][vid, :3]) == [2, 3, 4]
        assert sol[SOL_CUST_ROUTE_POS][2] == 0
        assert sol[SOL_CUST_ROUTE_POS][3] == 1
        assert sol[SOL_CUST_ROUTE_POS][4] == 2


class TestReverseSegmentEdgeCases:

    def test_reverse_full_route(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 1, 2, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 2, 3, ACT_DELIVER, dist_matrix, customers)

        L = sol[SOL_TRUCK_LENGTHS][vid]
        route_before = list(sol[SOL_TRUCK_STOPS][vid, :L])

        reverse_segment(sol, vtype, vid, 0, 2, dist_matrix)

        route_after = list(sol[SOL_TRUCK_STOPS][vid, :L])
        assert route_after == route_before[::-1]
        assert list(route_after) == [3, 2, 1]

        assert sol[SOL_CUST_ROUTE_POS][1] == 2
        assert sol[SOL_CUST_ROUTE_POS][2] == 1
        assert sol[SOL_CUST_ROUTE_POS][3] == 0

    def test_reverse_middle_segment(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 1, 2, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 2, 3, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 3, 4, ACT_DELIVER, dist_matrix, customers)

        reverse_segment(sol, vtype, vid, 1, 2, dist_matrix)

        assert list(sol[SOL_TRUCK_STOPS][vid, :4]) == [1, 3, 2, 4]
        assert sol[SOL_CUST_ROUTE_POS][2] == 2
        assert sol[SOL_CUST_ROUTE_POS][3] == 1


class TestInsertBoundaryConditions:

    def test_insert_at_max_capacity_fails(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0
        max_len = sol[SOL_META][META_MAX_ROUTE_LEN]

        for pos in range(max_len):
            insert_stop(sol, vtype, vid, pos, (pos % 4) + 1, ACT_DELIVER, dist_matrix, customers)

        assert sol[SOL_TRUCK_LENGTHS][vid] == max_len

        insert_stop(sol, vtype, vid, max_len, 1, ACT_DELIVER, dist_matrix, customers)

        assert sol[SOL_TRUCK_LENGTHS][vid] == max_len


class TestReloadActions:

    def test_reload_not_indexed(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        # Use customer 0 as the RELOAD node (valid index)
        insert_stop(sol, vtype, vid, 0, 0, ACT_RELOAD, dist_matrix, customers)

        assert sol[SOL_TRUCK_LENGTHS][vid] == 1
        assert sol[SOL_TRUCK_ACTIONS][vid, 0] == ACT_RELOAD
        assert sol[SOL_CUST_VEHICLE][0] == -1

    def test_reload_removed_silently(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 1, 0, ACT_RELOAD, dist_matrix, customers)

        assert sol[SOL_TRUCK_LENGTHS][vid] == 2

        removed_cust, removed_action = remove_stop(sol, vtype, vid, 1, dist_matrix, customers)

        assert removed_action == ACT_RELOAD
        assert removed_cust == 0
        assert sol[SOL_TRUCK_LENGTHS][vid] == 1
        assert sol[SOL_CUST_VEHICLE][1] == 0


class TestDistanceRecalculation:

    def test_distance_nonzero_after_operations(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        dist_1 = sol[SOL_TRUCK_DISTANCES][vid]
        assert dist_1 > 0

        insert_stop(sol, vtype, vid, 1, 2, ACT_DELIVER, dist_matrix, customers)
        dist_2 = sol[SOL_TRUCK_DISTANCES][vid]
        assert dist_2 > dist_1

    def test_distance_zero_when_empty(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        remove_stop(sol, vtype, vid, 0, dist_matrix, customers)

        assert sol[SOL_TRUCK_LENGTHS][vid] == 0
        assert sol[SOL_TRUCK_DISTANCES][vid] == 0


class TestSwapStopsWithinBug:

    def test_swap_updates_correct_positions_after_swap(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 1, 2, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 2, 3, ACT_DELIVER, dist_matrix, customers)

        assert sol[SOL_CUST_ROUTE_POS][1] == 0
        assert sol[SOL_CUST_ROUTE_POS][2] == 1
        assert sol[SOL_CUST_ROUTE_POS][3] == 2

        swap_stops_within(sol, vtype, vid, 0, 2, dist_matrix)

        assert list(sol[SOL_TRUCK_STOPS][vid, :3]) == [3, 2, 1]
        assert sol[SOL_CUST_ROUTE_POS][3] == 0
        assert sol[SOL_CUST_ROUTE_POS][1] == 2
        assert sol[SOL_CUST_ROUTE_POS][2] == 1

    def test_swap_with_reload_action(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 1, 0, ACT_RELOAD, dist_matrix, customers)

        assert sol[SOL_TRUCK_STOPS][vid, 0] == 1
        assert sol[SOL_TRUCK_ACTIONS][vid, 0] == ACT_DELIVER
        assert sol[SOL_TRUCK_STOPS][vid, 1] == 0
        assert sol[SOL_TRUCK_ACTIONS][vid, 1] == ACT_RELOAD

        swap_stops_within(sol, vtype, vid, 0, 1, dist_matrix)

        assert sol[SOL_TRUCK_STOPS][vid, 0] == 0
        assert sol[SOL_TRUCK_ACTIONS][vid, 0] == ACT_RELOAD
        assert sol[SOL_TRUCK_STOPS][vid, 1] == 1
        assert sol[SOL_TRUCK_ACTIONS][vid, 1] == ACT_DELIVER

        assert sol[SOL_CUST_ROUTE_POS][1] == 1


class TestInsertWithMixedActions:

    def test_insert_between_delivers_and_reload(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 1, 0, ACT_RELOAD, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 2, 2, ACT_DELIVER, dist_matrix, customers)

        assert list(sol[SOL_TRUCK_STOPS][vid, :3]) == [1, 0, 2]
        assert list(sol[SOL_TRUCK_ACTIONS][vid, :3]) == [ACT_DELIVER, ACT_RELOAD, ACT_DELIVER]

        insert_stop(sol, vtype, vid, 1, 3, ACT_DELIVER, dist_matrix, customers)

        assert list(sol[SOL_TRUCK_STOPS][vid, :4]) == [1, 3, 0, 2]
        assert list(sol[SOL_TRUCK_ACTIONS][vid, :4]) == [ACT_DELIVER, ACT_DELIVER, ACT_RELOAD, ACT_DELIVER]

        assert sol[SOL_CUST_ROUTE_POS][1] == 0
        assert sol[SOL_CUST_ROUTE_POS][3] == 1
        assert sol[SOL_CUST_ROUTE_POS][2] == 3


class TestReverseSegmentBoundaryBug:

    def test_reverse_with_start_greater_than_end(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 1, 2, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 2, 3, ACT_DELIVER, dist_matrix, customers)

        route_before = list(sol[SOL_TRUCK_STOPS][vid, :3])

        reverse_segment(sol, vtype, vid, 2, 1, dist_matrix)

        route_after = list(sol[SOL_TRUCK_STOPS][vid, :3])
        assert route_after == route_before

    def test_reverse_valid_call_works(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 1, 2, ACT_DELIVER, dist_matrix, customers)
        insert_stop(sol, vtype, vid, 2, 3, ACT_DELIVER, dist_matrix, customers)

        reverse_segment(sol, vtype, vid, 0, 2, dist_matrix)

        assert list(sol[SOL_TRUCK_STOPS][vid, :3]) == [3, 2, 1]



class TestRemoveCustomerIndexCleanup:

    def test_remove_deliver_clears_all_indices(self, minimal_setup):
        sol, customers, dist_matrix = minimal_setup
        vtype, vid = VEH_TRUCK, 0

        insert_stop(sol, vtype, vid, 0, 1, ACT_DELIVER, dist_matrix, customers)

        assert sol[SOL_CUST_VEHICLE][1] != -1
        assert sol[SOL_CUST_VTYPE][1] != -1
        assert sol[SOL_CUST_ROUTE_POS][1] != -1

        remove_stop(sol, vtype, vid, 0, dist_matrix, customers)

        assert sol[SOL_CUST_VEHICLE][1] == -1
        assert sol[SOL_CUST_VTYPE][1] == -1
        assert sol[SOL_CUST_ROUTE_POS][1] == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
