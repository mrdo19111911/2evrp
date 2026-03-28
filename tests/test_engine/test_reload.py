"""Tests for src/engine/reload.py -- truck/bike reload at satellite stops. All i64.

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


class TestTruckReloadBasic:

    def test_truck_reload_single_bike(self):
        sats, n = _make_sats((0, 0, 0, 100000, 6000))  # 100kg, t=6000s
        load_change, service, sync_wait = truck_reload_at_stop(
            0, 0, 500000, 5400, sats, n, RELOAD_S,
        )
        assert load_change == -100000
        assert service == RELOAD_S
        assert sync_wait == 600  # 6000 - 5400

    def test_truck_reload_no_matching_events(self):
        sats, n = _make_sats((0, 0, 1, 100000, 6000))  # truck_id=1
        load_change, service, sync_wait = truck_reload_at_stop(
            0, 0, 500000, 5400, sats, n, RELOAD_S,
        )
        assert load_change == 0
        assert service == 0
        assert sync_wait == 0

    def test_truck_reload_empty_satellites(self):
        sats = np.zeros((MAX_SATELLITES, 5), dtype=np.int64)
        load_change, service, sync_wait = truck_reload_at_stop(
            0, 0, 500000, 5400, sats, 0, RELOAD_S,
        )
        assert load_change == 0
        assert service == 0
        assert sync_wait == 0


class TestBikeReloadBasic:

    def test_bike_reload_single_event(self):
        sats, n = _make_sats((0, 0, 0, 50000, 6000))  # 50kg, t=6000s
        transfer_g, service, sync_wait = bike_reload_at_stop(
            0, 0, 5400, sats, n, RELOAD_S,
        )
        assert transfer_g == 50000
        assert service == RELOAD_S
        assert sync_wait == 600

    def test_bike_reload_no_matching_events(self):
        sats, n = _make_sats((0, 1, 0, 50000, 6000))  # bike_id=1
        transfer_g, service, sync_wait = bike_reload_at_stop(
            0, 0, 5400, sats, n, RELOAD_S,
        )
        assert transfer_g == 0
        assert service == 0
        assert sync_wait == 0

    def test_bike_reload_empty_satellites(self):
        sats = np.zeros((MAX_SATELLITES, 5), dtype=np.int64)
        transfer_g, service, sync_wait = bike_reload_at_stop(
            0, 0, 5400, sats, 0, RELOAD_S,
        )
        assert transfer_g == 0
        assert service == 0
        assert sync_wait == 0


class TestTruckBikeConservation:

    def test_conservation_single_transfer(self):
        sats, n = _make_sats((0, 0, 0, 100000, 6000))

        truck_load_change, _, _ = truck_reload_at_stop(
            0, 0, 500000, 5400, sats, n, RELOAD_S,
        )
        bike_transfer_g, _, _ = bike_reload_at_stop(
            0, 0, 5700, sats, n, RELOAD_S,
        )
        assert truck_load_change == -bike_transfer_g
        assert bike_transfer_g == 100000
