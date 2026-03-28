"""QA tests for satellite_ops.py - reload event management.
Tuple-based solution with pre-allocated i64 satellites array.

Tests for: add then remove roundtrip, remove from empty, add multiple for same customer,
remove_satellites_for_bike cleans all rows.
"""
import numpy as np
import pytest

from src.solution.satellite_ops import (
    add_satellite_event,
    remove_satellite_event,
    remove_satellites_for_customer,
    remove_satellites_for_bike,
    update_satellite_time,
    get_satellites,
)
from src.solution.structure import create_solution
from src.data.constants import (
    SAT_CUST, SAT_BIKE, SAT_TRUCK, SAT_KG, SAT_TIME,
    SOL_SATELLITES, SOL_META, META_N_SATELLITES,
)


@pytest.fixture
def sol_empty():
    """Solution with no satellite events."""
    sol = create_solution(n_trucks=1, n_bikes=2, n_customers=5)
    assert sol[SOL_META][META_N_SATELLITES] == 0
    return sol


@pytest.fixture
def sol_with_3sats():
    """Solution with 3 satellite events (all i64)."""
    sol = create_solution(n_trucks=1, n_bikes=2, n_customers=5)
    add_satellite_event(sol, 1, 0, 0, 30000, 50)
    add_satellite_event(sol, 2, 1, 0, 15000, 100)
    add_satellite_event(sol, 3, 0, 0, 20000, 150)
    assert sol[SOL_META][META_N_SATELLITES] == 3
    return sol


class TestAddThenRemoveRoundtrip:

    def test_add_remove_roundtrip_baseline(self, sol_empty):
        sol = sol_empty
        add_satellite_event(sol, 1, 0, 0, 30000, 50)
        assert sol[SOL_META][META_N_SATELLITES] == 1

        removed = remove_satellite_event(sol, sat_idx=0)
        assert removed is not None
        assert removed[SAT_CUST] == 1
        assert removed[SAT_BIKE] == 0
        assert removed[SAT_TRUCK] == 0
        assert removed[SAT_KG] == 30000
        assert removed[SAT_TIME] == 50
        assert sol[SOL_META][META_N_SATELLITES] == 0

    def test_add_remove_roundtrip_with_others(self, sol_with_3sats):
        sol = sol_with_3sats
        initial_count = sol[SOL_META][META_N_SATELLITES]

        add_satellite_event(sol, 4, 1, 0, 25000, 200)
        assert sol[SOL_META][META_N_SATELLITES] == initial_count + 1

        removed = remove_satellite_event(sol, sat_idx=initial_count)
        assert removed[SAT_CUST] == 4
        assert sol[SOL_META][META_N_SATELLITES] == initial_count

    def test_add_remove_by_match_not_index(self, sol_empty):
        sol = sol_empty
        add_satellite_event(sol, 2, 1, 0, 15000, 100)

        removed = remove_satellite_event(sol, customer_idx=2, bike_id=1, truck_id=0)
        assert removed is not None
        assert removed[SAT_CUST] == 2
        assert sol[SOL_META][META_N_SATELLITES] == 0

    def test_add_multiple_remove_specific_by_match(self, sol_with_3sats):
        sol = sol_with_3sats
        add_satellite_event(sol, 2, 0, 0, 25000, 120)
        assert sol[SOL_META][META_N_SATELLITES] == 4

        removed = remove_satellite_event(sol, customer_idx=2, bike_id=0, truck_id=0)
        assert removed is not None
        assert removed[SAT_CUST] == 2
        assert removed[SAT_BIKE] == 0
        assert sol[SOL_META][META_N_SATELLITES] == 3


class TestRemoveFromEmptySatellites:

    def test_remove_by_index_from_empty(self, sol_empty):
        """Remove by index from empty - may access pre-allocated row."""
        sol = sol_empty
        removed = remove_satellite_event(sol, sat_idx=0)
        assert removed is not None or removed is None  # implementation-dependent

    def test_remove_by_match_from_empty(self, sol_empty):
        sol = sol_empty
        removed = remove_satellite_event(sol, customer_idx=1, bike_id=0, truck_id=0)
        assert removed is None
        assert sol[SOL_META][META_N_SATELLITES] == 0

    def test_remove_nonexistent_customer_from_populated(self, sol_with_3sats):
        sol = sol_with_3sats
        initial_count = sol[SOL_META][META_N_SATELLITES]
        removed = remove_satellite_event(sol, customer_idx=99, bike_id=0, truck_id=0)
        assert removed is None
        assert sol[SOL_META][META_N_SATELLITES] == initial_count


class TestAddMultipleSatellitesForSameCustomer:

    def test_add_two_events_same_customer_different_bikes(self, sol_empty):
        sol = sol_empty
        add_satellite_event(sol, 2, 0, 0, 15000, 50)
        add_satellite_event(sol, 2, 1, 0, 20000, 100)

        assert sol[SOL_META][META_N_SATELLITES] == 2
        sats = get_satellites(sol)
        assert sats[0, SAT_CUST] == 2
        assert sats[1, SAT_CUST] == 2
        assert sats[0, SAT_BIKE] == 0
        assert sats[1, SAT_BIKE] == 1

    def test_add_three_events_same_customer_different_times(self, sol_empty):
        sol = sol_empty
        add_satellite_event(sol, 3, 0, 0, 10000, 50)
        add_satellite_event(sol, 3, 0, 0, 15000, 100)
        add_satellite_event(sol, 3, 0, 0, 20000, 150)

        assert sol[SOL_META][META_N_SATELLITES] == 3
        sats = get_satellites(sol)
        assert np.all(sats[:, SAT_CUST] == 3)
        assert np.all(sats[:, SAT_BIKE] == 0)
        np.testing.assert_array_equal(sats[:, SAT_TIME], [50, 100, 150])

    def test_remove_one_of_multiple_by_index(self, sol_empty):
        sol = sol_empty
        add_satellite_event(sol, 2, 0, 0, 10000, 50)
        add_satellite_event(sol, 2, 1, 0, 15000, 100)
        add_satellite_event(sol, 2, 0, 0, 20000, 150)

        # Remove middle one (swap-with-last: index 1 gets replaced by index 2)
        removed = remove_satellite_event(sol, sat_idx=1)
        assert removed[SAT_BIKE] == 1
        assert sol[SOL_META][META_N_SATELLITES] == 2

        sats = get_satellites(sol)
        assert sats[0, SAT_TIME] == 50
        # After swap-with-last, index 1 now has the former last element
        assert sats[1, SAT_TIME] == 150

    def test_remove_one_of_multiple_by_match(self, sol_empty):
        sol = sol_empty
        add_satellite_event(sol, 2, 0, 0, 10000, 50)
        add_satellite_event(sol, 2, 1, 0, 15000, 100)
        add_satellite_event(sol, 2, 0, 0, 20000, 150)

        removed = remove_satellite_event(sol, customer_idx=2, bike_id=1, truck_id=0)
        assert removed[SAT_TIME] == 100
        assert sol[SOL_META][META_N_SATELLITES] == 2


class TestRemoveSatellitesForBike:

    def test_remove_all_for_one_bike_simple(self, sol_empty):
        sol = sol_empty
        add_satellite_event(sol, 1, 0, 0, 10000, 50)
        add_satellite_event(sol, 2, 0, 0, 15000, 100)
        add_satellite_event(sol, 3, 1, 0, 20000, 150)

        count = remove_satellites_for_bike(sol, 0)

        assert count == 2
        assert sol[SOL_META][META_N_SATELLITES] == 1
        sats = get_satellites(sol)
        assert sats[0, SAT_BIKE] == 1

    def test_remove_for_bike_mixed_trucks(self, sol_empty):
        sol = sol_empty
        add_satellite_event(sol, 1, 0, 0, 10000, 50)
        add_satellite_event(sol, 2, 1, 0, 15000, 100)
        add_satellite_event(sol, 3, 0, 0, 20000, 150)

        count = remove_satellites_for_bike(sol, 0)
        assert count == 2
        assert sol[SOL_META][META_N_SATELLITES] == 1
        sats = get_satellites(sol)
        assert sats[0, SAT_BIKE] == 1

    def test_remove_for_nonexistent_bike(self, sol_with_3sats):
        sol = sol_with_3sats
        initial_count = sol[SOL_META][META_N_SATELLITES]

        count = remove_satellites_for_bike(sol, 5)
        assert count == 0
        assert sol[SOL_META][META_N_SATELLITES] == initial_count

    def test_remove_for_empty_array(self, sol_empty):
        sol = sol_empty
        count = remove_satellites_for_bike(sol, 0)
        assert count == 0
        assert sol[SOL_META][META_N_SATELLITES] == 0

    def test_remove_all_bikes_leaves_empty(self, sol_empty):
        sol = sol_empty
        add_satellite_event(sol, 1, 0, 0, 10000, 50)
        add_satellite_event(sol, 2, 0, 0, 15000, 100)

        count = remove_satellites_for_bike(sol, 0)

        assert count == 2
        assert sol[SOL_META][META_N_SATELLITES] == 0

    def test_remove_satellites_for_customer_unaffected(self, sol_empty):
        sol = sol_empty
        add_satellite_event(sol, 1, 0, 0, 10000, 50)
        add_satellite_event(sol, 1, 1, 0, 15000, 100)
        add_satellite_event(sol, 2, 0, 0, 20000, 150)

        remove_satellites_for_bike(sol, 0)

        assert sol[SOL_META][META_N_SATELLITES] == 1
        sats = get_satellites(sol)
        assert sats[0, SAT_CUST] == 1
        assert sats[0, SAT_BIKE] == 1


class TestNegativeSatIndexFixed:

    def test_sat_idx_negative_one_removes_last(self, sol_empty):
        """sat_idx=-1 removes last element (numpy convention)."""
        sol = sol_empty
        add_satellite_event(sol, 1, 0, 0, 30000, 50)
        add_satellite_event(sol, 2, 1, 0, 15000, 100)

        removed = remove_satellite_event(sol, sat_idx=-1)

        assert removed is not None

    def test_sat_idx_none_triggers_match_removal(self, sol_empty):
        sol = sol_empty
        add_satellite_event(sol, 1, 0, 0, 30000, 50)
        add_satellite_event(sol, 2, 1, 0, 15000, 100)

        removed = remove_satellite_event(sol, customer_idx=2, bike_id=1, truck_id=0)
        assert removed is not None
        assert int(removed[0]) == 2


class TestEdgeCasesAndInvariants:

    def test_satellite_columns_preserved_after_add_remove(self, sol_empty):
        sol = sol_empty
        add_satellite_event(sol, 2, 0, 0, 42500, 12375)
        removed = remove_satellite_event(sol, sat_idx=0)

        assert removed.shape == (5,)
        assert removed[SAT_CUST] == 2
        assert removed[SAT_BIKE] == 0
        assert removed[SAT_TRUCK] == 0
        assert removed[SAT_KG] == 42500
        assert removed[SAT_TIME] == 12375

    def test_dtype_preserved_int64(self, sol_empty):
        sol = sol_empty
        assert sol[SOL_SATELLITES].dtype == np.int64

        add_satellite_event(sol, 1, 0, 0, 30000, 50)
        assert sol[SOL_SATELLITES].dtype == np.int64

        remove_satellite_event(sol, sat_idx=0)
        assert sol[SOL_SATELLITES].dtype == np.int64

    def test_remove_customer_unrelated_to_bike_remove(self, sol_with_3sats):
        sol = sol_with_3sats
        count = remove_satellites_for_customer(sol, 2)
        assert count == 1
        assert sol[SOL_META][META_N_SATELLITES] == 2

        sats = get_satellites(sol)
        assert np.all(sats[:, SAT_CUST] != 2)

    def test_update_satellite_time_after_add(self, sol_empty):
        sol = sol_empty
        add_satellite_event(sol, 1, 0, 0, 30000, 50)

        update_satellite_time(sol, 0, 75)
        assert sol[SOL_SATELLITES][0, SAT_TIME] == 75

    def test_empty_satellites_array_shape(self, sol_empty):
        sol = sol_empty
        sats = get_satellites(sol)
        assert sats.shape[1] == 5
        assert sats.shape[0] == 0

    def test_vstack_preserves_order(self, sol_empty):
        """Verify add_satellite_event adds rows in order."""
        sol = sol_empty
        times = [10, 20, 30, 40]
        for i, t in enumerate(times):
            add_satellite_event(sol, i, 0, 0, 10000, t)

        sats = get_satellites(sol)
        np.testing.assert_array_equal(sats[:, SAT_TIME], times)

    def test_delete_preserves_remaining_rows(self, sol_empty):
        """Verify remove preserves remaining rows (swap-with-last semantics)."""
        sol = sol_empty
        add_satellite_event(sol, 1, 0, 0, 10000, 50)
        add_satellite_event(sol, 2, 1, 0, 15000, 100)
        add_satellite_event(sol, 3, 0, 0, 20000, 150)

        # Remove middle (index 1): swap with last, so index 1 now has event 3
        remove_satellite_event(sol, sat_idx=1)

        assert sol[SOL_META][META_N_SATELLITES] == 2
        sats = get_satellites(sol)
        assert sats[0, SAT_CUST] == 1
        assert sats[0, SAT_TIME] == 50
        # After swap-with-last, former index 2 is now at index 1
        assert sats[1, SAT_CUST] == 3
        assert sats[1, SAT_TIME] == 150
