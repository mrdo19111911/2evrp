"""QA tests for src/data/io.py -- edge cases & robustness.

API (i64):
  load_instance returns (customers_i64, depot_i64, vehicles_i64)
  customers: i64 (N, 7) [x_m, y_m, demand_g, tw_open_s, tw_close_s, service_s, restricted]
  depot: i64 (2,) in meters
  vehicles: i64 (K, 4) [type, capacity_g, cost_per_m_vnd, speed_us_per_m]
"""
import json
import os
import tempfile

import numpy as np
import pytest

from src.data.io import load_instance, save_solution, load_solution
from src.data.constants import CUST_COLS


# ===========================================================================
# TC1: load_instance with missing JSON fields
# ===========================================================================

def test_load_instance_missing_customers_key():
    """JSON missing 'customers' key should raise KeyError."""
    with tempfile.TemporaryDirectory() as tmp:
        bad = {
            "depot": [0.0, 0.0],
            "vehicles": [[0, 2000000, 12, 144000]],
        }
        filepath = os.path.join(tmp, "missing_customers.json")
        with open(filepath, "w") as f:
            json.dump(bad, f)

        with pytest.raises(KeyError):
            load_instance(filepath)


def test_load_instance_missing_depot_key():
    """JSON missing 'depot' key should raise KeyError."""
    with tempfile.TemporaryDirectory() as tmp:
        bad = {
            "customers": [[1.0, 1.0, 400.0, 0.0, 120.0, 5.0]],
            "vehicles": [[0, 2000000, 12, 144000]],
        }
        filepath = os.path.join(tmp, "missing_depot.json")
        with open(filepath, "w") as f:
            json.dump(bad, f)

        with pytest.raises(KeyError):
            load_instance(filepath)


def test_load_instance_missing_vehicle_key_allowed():
    """JSON missing 'vehicles' key -> empty vehicle array via .get()."""
    with tempfile.TemporaryDirectory() as tmp:
        ok_ish = {
            "depot": [0.0, 0.0],
            "customers": [[1.0, 1.0, 400.0, 0.0, 120.0, 5.0]],
        }
        filepath = os.path.join(tmp, "missing_vehicles.json")
        with open(filepath, "w") as f:
            json.dump(ok_ish, f)

        customers, depot, vehicles = load_instance(filepath)
        assert vehicles.shape[0] == 0


def test_load_instance_missing_customer_fields_dict_format():
    """Customer dict missing required fields -> KeyError."""
    with tempfile.TemporaryDirectory() as tmp:
        bad = {
            "depot": {"x": 0.0, "y": 0.0},
            "customers": [
                {"x": 1.0, "y": 2.0}  # MISSING: demand_kg, tw_open, tw_close
            ],
            "vehicles": {"trucks": [], "bikes": []},
        }
        filepath = os.path.join(tmp, "missing_cust_fields.json")
        with open(filepath, "w") as f:
            json.dump(bad, f)

        with pytest.raises(KeyError):
            load_instance(filepath)


def test_load_instance_missing_restricted_field_flat_format():
    """Flat format without 'restricted' key -> defaults to 0 for all."""
    with tempfile.TemporaryDirectory() as tmp:
        ok = {
            "depot": [0.0, 0.0],
            "customers": [[1.0, 1.0, 400.0, 0.0, 120.0, 5.0],
                          [2.0, 2.0, 300.0, 0.0, 120.0, 5.0]],
            "vehicles": [[0, 2000000, 12, 144000]],
        }
        filepath = os.path.join(tmp, "missing_restricted.json")
        with open(filepath, "w") as f:
            json.dump(ok, f)

        customers, depot, vehicles = load_instance(filepath)
        assert customers.shape[0] == 2
        assert customers.shape[1] == CUST_COLS  # 7 columns
        # restricted is column 6 of customers
        assert (customers[:, 6] == 0).all()


# ===========================================================================
# TC2: load_instance with 0 customers
# ===========================================================================

def test_load_instance_zero_customers():
    """Instance with empty customers list should load successfully."""
    with tempfile.TemporaryDirectory() as tmp:
        empty = {
            "depot": [50.0, 50.0],
            "customers": [],
            "vehicles": [[0, 2000000, 12, 144000]],
        }
        filepath = os.path.join(tmp, "zero_customers.json")
        with open(filepath, "w") as f:
            json.dump(empty, f)

        customers, depot, vehicles = load_instance(filepath)
        assert customers.shape == (0, CUST_COLS)
        assert depot.shape == (2,)
        assert depot.dtype == np.int64


def test_load_instance_zero_customers_zero_vehicles():
    """Minimal valid instance (no customers, no vehicles)."""
    with tempfile.TemporaryDirectory() as tmp:
        minimal = {
            "depot": [0.0, 0.0],
            "customers": [],
            "vehicles": [],
        }
        filepath = os.path.join(tmp, "minimal.json")
        with open(filepath, "w") as f:
            json.dump(minimal, f)

        customers, depot, vehicles = load_instance(filepath)
        assert customers.shape[0] == 0
        assert vehicles.shape[0] == 0


# ===========================================================================
# TC3: save_solution then load_solution round-trip
# ===========================================================================

def test_save_load_roundtrip_preserves_dtypes():
    """Round-trip should preserve dtypes (int32 for stops/actions, int64 for satellites)."""
    with tempfile.TemporaryDirectory() as tmp:
        filepath = os.path.join(tmp, "sol_dtype.json")

        truck_stops = np.array([[0, 3, 1, -1], [0, 2, 4, 0]], dtype=np.int32)
        truck_actions = np.array([[0, 0, 0, -1], [0, 0, 0, 0]], dtype=np.int32)
        bike_stops = np.array([[0], [1]], dtype=np.int32)
        bike_actions = np.array([[-1], [-1]], dtype=np.int32)
        satellites = np.array([[1, 2, 3, 50000, 1800]], dtype=np.int64)

        save_solution(filepath, truck_stops, truck_actions,
                      bike_stops, bike_actions, satellites)
        ts, ta, bs, ba, sats = load_solution(filepath)

        assert ts.shape == truck_stops.shape
        assert ta.shape == truck_actions.shape
        assert bs.shape == bike_stops.shape
        assert ba.shape == bike_actions.shape
        assert ts.dtype == np.int32
        assert ta.dtype == np.int32
        assert bs.dtype == np.int32
        assert ba.dtype == np.int32
        assert sats.dtype == np.int64


def test_save_load_roundtrip_preserves_values():
    """Round-trip should preserve exact numeric values."""
    with tempfile.TemporaryDirectory() as tmp:
        filepath = os.path.join(tmp, "sol_values.json")

        truck_stops = np.array([[0, 3, 1, -1]], dtype=np.int32)
        truck_actions = np.array([[0, 0, 0, -1]], dtype=np.int32)
        bike_stops = np.array([[5]], dtype=np.int32)
        bike_actions = np.array([[1]], dtype=np.int32)
        satellites = np.array([[10, 11, 12, 50000, 3600]], dtype=np.int64)

        save_solution(filepath, truck_stops, truck_actions,
                      bike_stops, bike_actions, satellites)
        ts, ta, bs, ba, sats = load_solution(filepath)

        np.testing.assert_array_equal(ts, truck_stops)
        np.testing.assert_array_equal(ta, truck_actions)
        np.testing.assert_array_equal(bs, bike_stops)
        np.testing.assert_array_equal(ba, bike_actions)
        np.testing.assert_array_equal(sats, satellites)


def test_save_load_empty_satellites_array():
    """Empty satellites array should reconstruct as (0, 5) shape."""
    with tempfile.TemporaryDirectory() as tmp:
        filepath = os.path.join(tmp, "sol_empty_sats.json")

        truck_stops = np.array([[0]], dtype=np.int32)
        truck_actions = np.array([[0]], dtype=np.int32)
        bike_stops = np.array([[1]], dtype=np.int32)
        bike_actions = np.array([[1]], dtype=np.int32)
        satellites = np.empty((0, 5), dtype=np.int64)

        save_solution(filepath, truck_stops, truck_actions,
                      bike_stops, bike_actions, satellites)
        ts, ta, bs, ba, sats = load_solution(filepath)

        assert sats.shape == (0, 5)


# ===========================================================================
# TC4: Vehicle count mismatch
# ===========================================================================

def test_load_instance_vehicle_mismatch_types():
    """Vehicles with invalid type values (not 0 or 1) pass through (no validation)."""
    with tempfile.TemporaryDirectory() as tmp:
        bad = {
            "depot": [0.0, 0.0],
            "customers": [[1.0, 1.0, 400.0, 0.0, 120.0, 5.0]],
            "vehicles": [[2, 2000000, 12, 144000]],
        }
        filepath = os.path.join(tmp, "bad_vehicle_type.json")
        with open(filepath, "w") as f:
            json.dump(bad, f)

        customers, depot, vehicles = load_instance(filepath)
        assert vehicles[0, 0] == 2


def test_load_instance_vehicle_missing_fields():
    """Vehicle dict missing required fields -> KeyError."""
    with tempfile.TemporaryDirectory() as tmp:
        bad = {
            "depot": {"x": 0.0, "y": 0.0},
            "customers": [],
            "vehicles": {
                "trucks": [{"capacity_kg": 2000}],  # MISSING: speed_kmh
                "bikes": []
            },
        }
        filepath = os.path.join(tmp, "missing_veh_fields.json")
        with open(filepath, "w") as f:
            json.dump(bad, f)

        with pytest.raises(KeyError):
            load_instance(filepath)


def test_load_instance_vehicle_negative_capacity():
    """Vehicle with negative capacity passes through (no validation).
    Flat parser multiplies capacity by 1000 (kg->g), so -2000 kg -> -2000000 g.
    """
    with tempfile.TemporaryDirectory() as tmp:
        bad = {
            "depot": [0.0, 0.0],
            "customers": [[1.0, 1.0, 400.0, 0.0, 120.0, 5.0]],
            "vehicles": [[0, -2000.0, 12, 144000]],
        }
        filepath = os.path.join(tmp, "neg_capacity.json")
        with open(filepath, "w") as f:
            json.dump(bad, f)

        customers, depot, vehicles = load_instance(filepath)
        assert vehicles[0, 1] < 0  # negative capacity passed through


def test_load_instance_vehicle_count_mismatch_actual():
    """Metadata vs actual vehicle count mismatch is not validated."""
    with tempfile.TemporaryDirectory() as tmp:
        suspicious = {
            "metadata": {
                "name": "test",
                "n_customers": 1,
                "n_trucks": 5,
                "n_bikes": 0,
            },
            "depot": [0.0, 0.0],
            "customers": [[1.0, 1.0, 400.0, 0.0, 120.0, 5.0]],
            "vehicles": [
                [0, 2000000, 12, 144000],
            ],
        }
        filepath = os.path.join(tmp, "veh_mismatch.json")
        with open(filepath, "w") as f:
            json.dump(suspicious, f)

        customers, depot, vehicles = load_instance(filepath)
        assert vehicles.shape[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
