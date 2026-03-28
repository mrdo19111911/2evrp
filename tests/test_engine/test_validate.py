"""Tests for src/engine/validate.py -- flat violation arrays. All i64.

New @njit signatures:
- validate_delivery_uniqueness(truck_stops, truck_actions, truck_lengths,
    bike_stops, bike_actions, bike_lengths, n_trucks, n_bikes, n_customers)
- validate_vehicle_restrictions(truck_stops, truck_actions, truck_lengths,
    n_trucks, customers)
"""
import numpy as np
import pytest

from src.engine.validate import (
    validate_delivery_uniqueness, validate_vehicle_restrictions, validate_all,
)
from src.data.constants import ACT_DELIVER, ACT_RELOAD, ACT_PAD


def test_all_delivered_once():
    """5 customers each delivered once -> valid."""
    n_cust = 5
    L = 4

    truck_stops = np.full((1, L), -1, dtype=np.int32)
    truck_actions = np.full((1, L), ACT_PAD, dtype=np.int8)
    truck_stops[0, 0] = 0; truck_actions[0, 0] = ACT_DELIVER
    truck_stops[0, 1] = 1; truck_actions[0, 1] = ACT_DELIVER
    truck_lengths = np.array([2], dtype=np.int32)

    bike_stops = np.full((2, L), -1, dtype=np.int32)
    bike_actions = np.full((2, L), ACT_PAD, dtype=np.int8)
    bike_stops[0, 0] = 2; bike_actions[0, 0] = ACT_DELIVER
    bike_stops[0, 1] = 3; bike_actions[0, 1] = ACT_DELIVER
    bike_stops[1, 0] = 4; bike_actions[1, 0] = ACT_DELIVER
    bike_lengths = np.array([2, 1], dtype=np.int32)

    unserved, n_unserved, n_dup = validate_delivery_uniqueness(
        truck_stops, truck_actions, truck_lengths,
        bike_stops, bike_actions, bike_lengths, 1, 2, n_cust)
    assert n_unserved == 0
    assert n_dup == 0


def test_customer_missing():
    """Customer 3 not delivered -> unserved."""
    n_cust = 5
    L = 3

    truck_stops = np.full((1, L), -1, dtype=np.int32)
    truck_actions = np.full((1, L), ACT_PAD, dtype=np.int8)
    truck_stops[0, 0] = 0; truck_actions[0, 0] = ACT_DELIVER
    truck_stops[0, 1] = 1; truck_actions[0, 1] = ACT_DELIVER
    truck_lengths = np.array([2], dtype=np.int32)

    bike_stops = np.full((1, L), -1, dtype=np.int32)
    bike_actions = np.full((1, L), ACT_PAD, dtype=np.int8)
    bike_stops[0, 0] = 2; bike_actions[0, 0] = ACT_DELIVER
    bike_stops[0, 1] = 4; bike_actions[0, 1] = ACT_DELIVER
    bike_lengths = np.array([2], dtype=np.int32)

    unserved, n_unserved, n_dup = validate_delivery_uniqueness(
        truck_stops, truck_actions, truck_lengths,
        bike_stops, bike_actions, bike_lengths, 1, 1, n_cust)
    assert n_unserved > 0
    assert 3 in unserved[:n_unserved]


def test_customer_duplicate():
    """Customer 2 delivered by both truck and bike -> duplicates."""
    n_cust = 5
    L = 4

    truck_stops = np.full((1, L), -1, dtype=np.int32)
    truck_actions = np.full((1, L), ACT_PAD, dtype=np.int8)
    truck_stops[0, 0] = 0; truck_actions[0, 0] = ACT_DELIVER
    truck_stops[0, 1] = 1; truck_actions[0, 1] = ACT_DELIVER
    truck_stops[0, 2] = 2; truck_actions[0, 2] = ACT_DELIVER
    truck_lengths = np.array([3], dtype=np.int32)

    bike_stops = np.full((1, L), -1, dtype=np.int32)
    bike_actions = np.full((1, L), ACT_PAD, dtype=np.int8)
    bike_stops[0, 0] = 2; bike_actions[0, 0] = ACT_DELIVER
    bike_stops[0, 1] = 3; bike_actions[0, 1] = ACT_DELIVER
    bike_stops[0, 2] = 4; bike_actions[0, 2] = ACT_DELIVER
    bike_lengths = np.array([3], dtype=np.int32)

    unserved, n_unserved, n_dup = validate_delivery_uniqueness(
        truck_stops, truck_actions, truck_lengths,
        bike_stops, bike_actions, bike_lengths, 1, 1, n_cust)
    assert n_dup > 0


def test_reload_not_counted():
    """RELOAD at a location does not count as delivery."""
    n_cust = 5
    L = 4

    truck_stops = np.full((1, L), -1, dtype=np.int32)
    truck_actions = np.full((1, L), ACT_PAD, dtype=np.int8)
    truck_stops[0, 0] = 0; truck_actions[0, 0] = ACT_DELIVER
    truck_stops[0, 1] = 1; truck_actions[0, 1] = ACT_DELIVER
    truck_stops[0, 2] = 2; truck_actions[0, 2] = ACT_RELOAD
    truck_lengths = np.array([3], dtype=np.int32)

    bike_stops = np.full((1, L), -1, dtype=np.int32)
    bike_actions = np.full((1, L), ACT_PAD, dtype=np.int8)
    bike_stops[0, 0] = 2; bike_actions[0, 0] = ACT_DELIVER
    bike_stops[0, 1] = 3; bike_actions[0, 1] = ACT_DELIVER
    bike_stops[0, 2] = 4; bike_actions[0, 2] = ACT_DELIVER
    bike_lengths = np.array([3], dtype=np.int32)

    unserved, n_unserved, n_dup = validate_delivery_uniqueness(
        truck_stops, truck_actions, truck_lengths,
        bike_stops, bike_actions, bike_lengths, 1, 1, n_cust)
    assert n_unserved == 0
    assert n_dup == 0


def test_restriction_violation(tiny_instance):
    """Restricted customer 2 on truck -> violation."""
    L = 3
    truck_stops = np.full((1, L), -1, dtype=np.int32)
    truck_actions = np.full((1, L), ACT_PAD, dtype=np.int8)
    truck_stops[0, 0] = 2; truck_actions[0, 0] = ACT_DELIVER
    truck_lengths = np.array([1], dtype=np.int32)

    vr_viol, n_vr = validate_vehicle_restrictions(
        truck_stops, truck_actions, truck_lengths, 1,
        tiny_instance["customers"])
    assert n_vr >= 1
    assert 2 in vr_viol[:n_vr]


def test_restriction_ok(tiny_instance):
    """Non-restricted customers on truck -> ok."""
    L = 3
    truck_stops = np.full((1, L), -1, dtype=np.int32)
    truck_actions = np.full((1, L), ACT_PAD, dtype=np.int8)
    truck_stops[0, 0] = 0; truck_actions[0, 0] = ACT_DELIVER
    truck_stops[0, 1] = 1; truck_actions[0, 1] = ACT_DELIVER
    truck_lengths = np.array([2], dtype=np.int32)

    vr_viol, n_vr = validate_vehicle_restrictions(
        truck_stops, truck_actions, truck_lengths, 1,
        tiny_instance["customers"])
    assert n_vr == 0
