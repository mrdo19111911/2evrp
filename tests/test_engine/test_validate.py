"""Tests for src/engine/validate.py — Data Model v2.

Unified arrays, transfers instead of satellites.
"""
import numpy as np
import pytest

from src.engine.validate import validate_delivery_uniqueness, validate_sync
from src.data.constants import (
    ACT_DELIVER, ACT_PICKUP, ACT_PAD,
    ST_LOC, ST_ACTION, ST_CUMTIME, ST_SERVICE, ST_DEPART,
    ST_LOAD_KG_BEF, ST_LOAD_KG_AFT, ST_FEASIBLE, ST_COLS,
    TR_HUB, TR_BIKE, TR_TRUCK, TR_KG, TR_CBM, TR_TIME, TR_COLS,
    ORD_LOC, ORD_QTY, ORD_UNIT_W, ORD_COLS,
)


# ---------------------------------------------------------------------------
# validate_delivery_uniqueness — unified arrays
# ---------------------------------------------------------------------------

def test_all_delivered_once(tiny_data):
    """5 customers each delivered once -> valid."""
    n_cust = 5
    n_veh = 3
    L = 4
    orders = tiny_data["orders"]

    stops = np.full((n_veh, L), -1, dtype=np.int32)
    actions = np.full((n_veh, L), ACT_PAD, dtype=np.int8)

    # Vehicle 0 (truck): C0, C1
    stops[0, 0] = int(orders[0, ORD_LOC]); actions[0, 0] = ACT_DELIVER
    stops[0, 1] = int(orders[1, ORD_LOC]); actions[0, 1] = ACT_DELIVER

    # Vehicle 1 (bike 0): C2, C3
    stops[1, 0] = int(orders[2, ORD_LOC]); actions[1, 0] = ACT_DELIVER
    stops[1, 1] = int(orders[3, ORD_LOC]); actions[1, 1] = ACT_DELIVER

    # Vehicle 2 (bike 1): C4
    stops[2, 0] = int(orders[4, ORD_LOC]); actions[2, 0] = ACT_DELIVER

    valid, unserved, duplicates = validate_delivery_uniqueness(
        stops, actions, orders, n_cust,
    )
    assert valid is True
    assert len(unserved) == 0
    assert len(duplicates) == 0


def test_customer_missing(tiny_data):
    """Customer 3 not delivered -> unserved."""
    n_cust = 5
    n_veh = 2
    L = 3
    orders = tiny_data["orders"]

    stops = np.full((n_veh, L), -1, dtype=np.int32)
    actions = np.full((n_veh, L), ACT_PAD, dtype=np.int8)

    # V0: C0, C1
    stops[0, 0] = int(orders[0, ORD_LOC]); actions[0, 0] = ACT_DELIVER
    stops[0, 1] = int(orders[1, ORD_LOC]); actions[0, 1] = ACT_DELIVER

    # V1: C2, C4 (C3 missing)
    stops[1, 0] = int(orders[2, ORD_LOC]); actions[1, 0] = ACT_DELIVER
    stops[1, 1] = int(orders[4, ORD_LOC]); actions[1, 1] = ACT_DELIVER

    valid, unserved, duplicates = validate_delivery_uniqueness(
        stops, actions, orders, n_cust,
    )
    assert valid is False
    assert 3 in unserved


def test_customer_duplicate(tiny_data):
    """Customer 2 delivered by both V0 and V1 -> duplicates."""
    n_cust = 5
    n_veh = 2
    L = 4
    orders = tiny_data["orders"]

    stops = np.full((n_veh, L), -1, dtype=np.int32)
    actions = np.full((n_veh, L), ACT_PAD, dtype=np.int8)

    # V0: C0, C1, C2
    stops[0, 0] = int(orders[0, ORD_LOC]); actions[0, 0] = ACT_DELIVER
    stops[0, 1] = int(orders[1, ORD_LOC]); actions[0, 1] = ACT_DELIVER
    stops[0, 2] = int(orders[2, ORD_LOC]); actions[0, 2] = ACT_DELIVER  # dup

    # V1: C2, C3, C4
    stops[1, 0] = int(orders[2, ORD_LOC]); actions[1, 0] = ACT_DELIVER  # dup
    stops[1, 1] = int(orders[3, ORD_LOC]); actions[1, 1] = ACT_DELIVER
    stops[1, 2] = int(orders[4, ORD_LOC]); actions[1, 2] = ACT_DELIVER

    valid, unserved, duplicates = validate_delivery_uniqueness(
        stops, actions, orders, n_cust,
    )
    assert valid is False
    assert 2 in duplicates


def test_pickup_not_counted(tiny_data):
    """PICKUP at a location does not count as delivery."""
    n_cust = 5
    n_veh = 2
    L = 4
    orders = tiny_data["orders"]

    stops = np.full((n_veh, L), -1, dtype=np.int32)
    actions = np.full((n_veh, L), ACT_PAD, dtype=np.int8)

    # V0: C0(deliver), C1(deliver), C2_loc(PICKUP)
    stops[0, 0] = int(orders[0, ORD_LOC]); actions[0, 0] = ACT_DELIVER
    stops[0, 1] = int(orders[1, ORD_LOC]); actions[0, 1] = ACT_DELIVER
    stops[0, 2] = int(orders[2, ORD_LOC]); actions[0, 2] = ACT_PICKUP  # not delivery

    # V1: C2(deliver), C3(deliver), C4(deliver)
    stops[1, 0] = int(orders[2, ORD_LOC]); actions[1, 0] = ACT_DELIVER
    stops[1, 1] = int(orders[3, ORD_LOC]); actions[1, 1] = ACT_DELIVER
    stops[1, 2] = int(orders[4, ORD_LOC]); actions[1, 2] = ACT_DELIVER

    valid, unserved, duplicates = validate_delivery_uniqueness(
        stops, actions, orders, n_cust,
    )
    assert valid is True
    assert len(unserved) == 0
    assert len(duplicates) == 0


# ---------------------------------------------------------------------------
# validate_sync — transfer-based
# ---------------------------------------------------------------------------

def test_sync_valid(tiny_data):
    """Transfer with states within delta_t -> valid."""
    delta_t = 15.0
    n_veh = tiny_data["n_vehicles"]

    # Build minimal states: one row per vehicle
    states = []
    for v in range(n_veh):
        row = np.zeros((0, ST_COLS), dtype=np.float64)
        states.append(row)

    transfers = np.empty((0, TR_COLS), dtype=np.float64)
    valid, violations = validate_sync(states, transfers, delta_t)
    assert valid is True
    assert len(violations) == 0
