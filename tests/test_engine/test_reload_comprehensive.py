"""Comprehensive QA tests for src/engine/reload.py. All i64.

API:
- truck_reload_at_stop(customer_idx, truck_id, current_load_g, truck_arrive_s,
                       satellites, n_satellites, reload_service_s)
- bike_reload_at_stop(customer_idx, bike_id, bike_arrive_s,
                      satellites, n_satellites, reload_service_s)
"""
import numpy as np
import pytest

from src.engine.reload import truck_reload_at_stop, bike_reload_at_stop
from src.data.constants import MAX_SATELLITES
from src.data.cost import RELOAD_SERVICE_TIME


def _make_sats(*rows):
    """Build satellites array from (cust, bike, truck, grams, time_s) tuples. All i64."""
    sats = np.zeros((MAX_SATELLITES, 5), dtype=np.int64)
    for i, row in enumerate(rows):
        sats[i] = row
    return sats, len(rows)


RELOAD_S = np.int64(RELOAD_SERVICE_TIME)  # 300s


# ===========================================================================
# TRUCK: BOUNDARY CONDITIONS
# ===========================================================================

class TestTruckReloadZeroKg:

    def test_truck_reload_zero_kg_single_bike(self):
        sats, n = _make_sats((0, 0, 0, 0, 6000))
        load_change, total_service, sync_wait = truck_reload_at_stop(
            0, 0, 500000, 5400, sats, n, RELOAD_S,
        )
        assert load_change == 0
        assert total_service == RELOAD_S
        assert sync_wait == 600

    def test_truck_reload_zero_kg_multiple_bikes(self):
        sats, n = _make_sats(
            (0, 0, 0, 0, 6000),
            (0, 1, 0, 0, 6300),
            (0, 2, 0, 0, 6600),
        )
        load_change, total_service, sync_wait = truck_reload_at_stop(
            0, 0, 500000, 5400, sats, n, RELOAD_S,
        )
        assert load_change == 0
        assert total_service == 3 * RELOAD_S
        assert sync_wait == 600


class TestTruckReloadMaxCapacity:

    def test_truck_unload_1000kg_from_capacity(self):
        sats, n = _make_sats((0, 0, 0, 1000000, 6000))  # 1000kg
        load_change, total_service, sync_wait = truck_reload_at_stop(
            0, 0, 2000000, 5400, sats, n, RELOAD_S,
        )
        assert load_change == -1000000
        assert 2000000 + load_change == 1000000


class TestTruckReloadEmptySatellites:

    def test_truck_no_matching_events_wrong_truck(self):
        sats, n = _make_sats(
            (0, 0, 1, 100000, 6000),
            (0, 1, 1, 100000, 6300),
        )
        load_change, total_service, sync_wait = truck_reload_at_stop(
            0, 0, 500000, 5400, sats, n, RELOAD_S,
        )
        assert load_change == 0
        assert total_service == 0
        assert sync_wait == 0

    def test_truck_no_matching_events_wrong_customer(self):
        sats, n = _make_sats(
            (1, 0, 0, 100000, 6000),
            (1, 1, 0, 100000, 6300),
        )
        load_change, total_service, sync_wait = truck_reload_at_stop(
            0, 0, 500000, 5400, sats, n, RELOAD_S,
        )
        assert load_change == 0
        assert total_service == 0
        assert sync_wait == 0


class TestTruckReloadEmptyArrayCompletely:

    def test_empty_satellites_array(self):
        sats = np.zeros((MAX_SATELLITES, 5), dtype=np.int64)
        load_change, total_service, sync_wait = truck_reload_at_stop(
            0, 0, 500000, 5400, sats, 0, RELOAD_S,
        )
        assert load_change == 0
        assert total_service == 0
        assert sync_wait == 0


# ===========================================================================
# TRUCK: TIME EDGE CASES
# ===========================================================================

class TestTruckReloadSyncWaitNegative:

    def test_truck_arrives_after_first_bike(self):
        sats, n = _make_sats(
            (0, 0, 0, 100000, 6000),
            (0, 1, 0, 100000, 6300),
        )
        load_change, total_service, sync_wait = truck_reload_at_stop(
            0, 0, 500000, 9000, sats, n, RELOAD_S,
        )
        assert sync_wait >= 0
        assert sync_wait == 0
        assert total_service == 2 * RELOAD_S


class TestTruckReloadEventOrdering:

    def test_truck_reload_unsorted_events(self):
        sats, n = _make_sats(
            (0, 0, 0, 100000, 6600),
            (0, 1, 0, 100000, 6000),
            (0, 2, 0, 100000, 6300),
        )
        load_change, total_service, sync_wait = truck_reload_at_stop(
            0, 0, 500000, 5400, sats, n, RELOAD_S,
        )
        assert sync_wait == 600
        assert total_service == 3 * RELOAD_S
        assert load_change == -300000


# ===========================================================================
# TRUCK: CURSOR LOGIC
# ===========================================================================

class TestTruckReloadCursorWait:

    def test_truck_reload_cursor_respects_bike_time(self):
        sats, n = _make_sats(
            (0, 0, 0, 100000, 6600),
            (0, 1, 0, 100000, 12000),
        )
        load_change, total_service, sync_wait = truck_reload_at_stop(
            0, 0, 500000, 6000, sats, n, RELOAD_S,
        )
        assert sync_wait == 600
        # First: start at 6600, end at 6900. Second: wait until 12000, end 12300.
        # total_service = 12300 - 6600 = 5700
        assert total_service == 5700


# ===========================================================================
# BIKE: BOUNDARY CONDITIONS
# ===========================================================================

class TestBikeReloadZeroKg:

    def test_bike_reload_zero_kg(self):
        sats, n = _make_sats((0, 0, 0, 0, 6000))
        transfer_g, service_time, sync_wait = bike_reload_at_stop(
            0, 0, 5400, sats, n, RELOAD_S,
        )
        assert transfer_g == 0
        assert service_time == RELOAD_S
        assert sync_wait == 600


class TestBikeReloadEmptySatellites:

    def test_bike_reload_no_matching_bike(self):
        sats, n = _make_sats(
            (0, 1, 0, 100000, 6000),
            (0, 1, 0, 100000, 6300),
        )
        transfer_g, service_time, sync_wait = bike_reload_at_stop(
            0, 0, 5400, sats, n, RELOAD_S,
        )
        assert transfer_g == 0
        assert service_time == 0
        assert sync_wait == 0

    def test_bike_reload_empty_array(self):
        sats = np.zeros((MAX_SATELLITES, 5), dtype=np.int64)
        transfer_g, service_time, sync_wait = bike_reload_at_stop(
            0, 0, 5400, sats, 0, RELOAD_S,
        )
        assert transfer_g == 0
        assert service_time == 0
        assert sync_wait == 0


# ===========================================================================
# BIKE: TIME EDGE CASES
# ===========================================================================

class TestBikeReloadSyncWaitNegative:

    def test_bike_arrives_after_truck_ready(self):
        sats, n = _make_sats((0, 0, 0, 100000, 6000))
        transfer_g, service_time, sync_wait = bike_reload_at_stop(
            0, 0, 12000, sats, n, RELOAD_S,
        )
        assert sync_wait >= 0
        assert sync_wait == 0
        assert transfer_g == 100000
        assert service_time == RELOAD_S


# ===========================================================================
# CONSERVATION
# ===========================================================================

class TestReloadConsistencyTruckBikePair:

    def test_truck_bike_conservation_single_transfer(self):
        sats, n = _make_sats((0, 0, 0, 100000, 6000))

        truck_load_change, truck_service, truck_sync = truck_reload_at_stop(
            0, 0, 500000, 5400, sats, n, RELOAD_S,
        )
        bike_transfer_g, bike_service, bike_sync = bike_reload_at_stop(
            0, 0, 5700, sats, n, RELOAD_S,
        )

        assert truck_load_change == -bike_transfer_g
        assert bike_transfer_g == 100000
        assert bike_sync < truck_sync
        assert truck_sync == 600
        assert bike_sync == 300
        assert truck_service >= RELOAD_S
        assert bike_service >= RELOAD_S


class TestReloadConsistencyMultipleBikes:

    def test_truck_three_bikes_conservation(self):
        sats, n = _make_sats(
            (0, 0, 0, 50000, 6000),
            (0, 1, 0, 50000, 6600),
            (0, 2, 0, 50000, 7200),
        )

        truck_load_change, truck_service, truck_sync = truck_reload_at_stop(
            0, 0, 500000, 5400, sats, n, RELOAD_S,
        )
        assert truck_load_change == -150000

        total_bike_transfer = 0
        for bike_id in range(3):
            bike_transfer_g, _, _ = bike_reload_at_stop(
                0, bike_id, 5400, sats, n, RELOAD_S,
            )
            total_bike_transfer += bike_transfer_g
        assert total_bike_transfer == 150000
        assert total_bike_transfer == -truck_load_change


class TestReloadConsistencyNegativeLoadChange:

    def test_truck_load_change_always_negative(self):
        test_cases = [
            ([(0, 0, 0, 10000, 6000)], 5400, 10000),
            ([(0, 0, 0, 100000, 6000), (0, 1, 0, 50000, 6300)], 5400, 150000),
            ([(0, 0, 0, 0, 6000)], 5400, 0),
        ]
        for events, truck_arrive, expected_sum in test_cases:
            sats, n = _make_sats(*events)
            load_change, _, _ = truck_reload_at_stop(
                0, 0, 500000, truck_arrive, sats, n, RELOAD_S,
            )
            assert load_change <= 0
            assert load_change == -expected_sum


# ===========================================================================
# EDGE: MULTIPLE TRUCKS AT SAME SATELLITE
# ===========================================================================

class TestTruckReloadMultipleTrucks:

    def test_truck_reload_filters_by_truck_id(self):
        sats, n = _make_sats(
            (0, 0, 0, 100000, 6000),
            (0, 1, 1, 200000, 6000),
            (0, 2, 0, 50000, 6300),
        )

        truck0_change, _, _ = truck_reload_at_stop(0, 0, 500000, 5400, sats, n, RELOAD_S)
        truck1_change, _, _ = truck_reload_at_stop(0, 1, 500000, 5400, sats, n, RELOAD_S)

        assert truck0_change == -150000
        assert truck1_change == -200000


# ===========================================================================
# EDGE: MULTIPLE BIKES
# ===========================================================================

class TestBikeReloadMultipleBikes:

    def test_bike_reload_takes_first_match_only(self):
        sats, n = _make_sats(
            (0, 0, 0, 100000, 6000),
            (0, 1, 0, 50000, 6000),
            (0, 1, 0, 30000, 6300),
            (0, 2, 0, 60000, 6000),
        )
        bike1_transfer, _, _ = bike_reload_at_stop(0, 1, 5400, sats, n, RELOAD_S)
        assert bike1_transfer == 50000


# ===========================================================================
# EDGE: LARGE VALUES
# ===========================================================================

class TestReloadLargeValues:

    def test_truck_reload_large_values(self):
        sats, n = _make_sats((0, 0, 0, 50000000, 1000100))  # 50 tons, ~16.7min
        load_change, _, sync_wait = truck_reload_at_stop(
            0, 0, 100000000, 1000000, sats, n, RELOAD_S,
        )
        assert load_change == -50000000
        assert sync_wait == 100


# ===========================================================================
# INTEGRATION: Output structure
# ===========================================================================

class TestReloadInSimulateRoute:

    def test_reload_output_structure(self):
        sats, n = _make_sats((0, 0, 0, 100000, 6000))

        result = truck_reload_at_stop(0, 0, 500000, 5400, sats, n, RELOAD_S)
        assert len(result) == 3
        load_change, service, sync = result
        assert isinstance(load_change, (int, np.integer))
        assert isinstance(service, (int, np.integer))
        assert isinstance(sync, (int, np.integer))

        result = bike_reload_at_stop(0, 0, 5400, sats, n, RELOAD_S)
        assert len(result) == 3
        transfer_g, service, sync = result
        assert isinstance(transfer_g, (int, np.integer))
        assert isinstance(service, (int, np.integer))
        assert isinstance(sync, (int, np.integer))
