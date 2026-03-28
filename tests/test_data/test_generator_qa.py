"""QA tests for src/data/generator.py -- shape, value, and column order validation.

API (i64):
  generate_random_instance returns (customers_i64, depot_i64, vehicles_i64)
  customers: i64 (N, 7) [x_m, y_m, demand_g, tw_open_s, tw_close_s, service_s, restricted]
  depot: i64 (2,) in meters
  vehicles: i64 (K, 4) [type, capacity_g, cost_per_m_vnd, speed_us_per_m]
"""
import numpy as np
import pytest
from src.data.generator import (
    generate_random_instance,
    generate_clustered_instance,
    _make_customers,
    _make_customers_from_xy,
    _make_restricted,
    _make_vehicles,
)
from src.data.constants import (
    COL_X, COL_Y, COL_DEMAND, COL_TW_OPEN, COL_TW_CLOSE,
    COL_SERVICE, COL_RESTRICTED, CUST_COLS,
    VCOL_TYPE, VCOL_CAPACITY, VCOL_COST_M, VCOL_SPEED,
    VEH_TRUCK, VEH_BIKE,
)
from src.data.cost import (
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
    TRUCK_COST_PER_M, BIKE_COST_PER_M,
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M,
    DAY_LENGTH,
)


# ===================================================================
# SECTION 1: generate_random_instance() -- Shapes & Dtypes
# ===================================================================

class TestRandomInstanceShapes:

    def test_returns_three_elements(self):
        result = generate_random_instance(n_customers=10, n_trucks=2, n_bikes=3)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_customers_shape(self):
        n = 15
        customers, _, _ = generate_random_instance(n_customers=n, n_trucks=2, n_bikes=3)
        assert customers.shape == (n, CUST_COLS)

    def test_customers_dtype_int64(self):
        customers, _, _ = generate_random_instance(n_customers=10, n_trucks=2, n_bikes=3)
        assert customers.dtype == np.int64

    def test_depot_shape(self):
        _, depot, _ = generate_random_instance(n_customers=10, n_trucks=2, n_bikes=3)
        assert depot.shape == (2,)

    def test_depot_dtype_int64(self):
        _, depot, _ = generate_random_instance(n_customers=10, n_trucks=2, n_bikes=3)
        assert depot.dtype == np.int64

    def test_vehicles_shape(self):
        n_trucks, n_bikes = 3, 5
        _, _, vehicles = generate_random_instance(n_customers=10, n_trucks=n_trucks, n_bikes=n_bikes)
        expected_rows = n_trucks + n_bikes
        assert vehicles.shape == (expected_rows, 4)

    def test_vehicles_dtype_int64(self):
        _, _, vehicles = generate_random_instance(n_customers=10, n_trucks=2, n_bikes=3)
        assert vehicles.dtype == np.int64


# ===================================================================
# SECTION 2: Customer Data Validation (i64 units: meters, grams, seconds)
# ===================================================================

class TestRandomInstanceCustomerData:

    def test_all_demands_positive(self):
        customers, _, _ = generate_random_instance(n_customers=100, n_trucks=2, n_bikes=3)
        demands = customers[:, COL_DEMAND]
        assert np.all(demands > 0)

    def test_demands_in_specified_range_grams(self):
        """Default demand_range_g=(3000, 120000) grams."""
        customers, _, _ = generate_random_instance(n_customers=50, n_trucks=2, n_bikes=3)
        demands = customers[:, COL_DEMAND]
        assert np.all(demands >= 3000)
        assert np.all(demands <= 120000)

    def test_tw_open_less_than_close(self):
        customers, _, _ = generate_random_instance(n_customers=100, n_trucks=2, n_bikes=3)
        tw_open = customers[:, COL_TW_OPEN]
        tw_close = customers[:, COL_TW_CLOSE]
        assert np.all(tw_open < tw_close)

    def test_tw_close_not_exceed_day_length_seconds(self):
        """tw_close <= DAY_LENGTH (28800 seconds = 8 hours)."""
        customers, _, _ = generate_random_instance(n_customers=50, n_trucks=2, n_bikes=3)
        tw_close = customers[:, COL_TW_CLOSE]
        assert np.all(tw_close <= DAY_LENGTH)

    def test_tw_minimum_width_60_seconds(self):
        """tw_close - tw_open >= 60 seconds (1 minute minimum)."""
        customers, _, _ = generate_random_instance(n_customers=100, n_trucks=2, n_bikes=3)
        widths = customers[:, COL_TW_CLOSE] - customers[:, COL_TW_OPEN]
        assert np.all(widths >= 60)

    def test_service_times_positive(self):
        """Service time = SERVICE_BASE_S + SERVICE_PER_G * demand_g."""
        customers, _, _ = generate_random_instance(n_customers=100, n_trucks=2, n_bikes=3)
        service_times = customers[:, COL_SERVICE]
        assert np.all(service_times > 0)

    def test_coordinates_within_area_meters(self):
        area_m = 50000
        customers, _, _ = generate_random_instance(
            n_customers=50, n_trucks=2, n_bikes=3, area_size_m=area_m)
        assert np.all(customers[:, COL_X] >= 0)
        assert np.all(customers[:, COL_X] <= area_m)
        assert np.all(customers[:, COL_Y] >= 0)
        assert np.all(customers[:, COL_Y] <= area_m)


# ===================================================================
# SECTION 3: Vehicle Column Order (i64 VCOL constants)
# ===================================================================

class TestRandomInstanceVehicleColumns:

    def test_vehicle_column_order_truck_explicit(self):
        _, _, vehicles = generate_random_instance(n_customers=10, n_trucks=2, n_bikes=0)
        assert vehicles.shape[0] == 2
        truck = vehicles[0]
        assert truck[VCOL_TYPE] == VEH_TRUCK
        assert truck[VCOL_CAPACITY] == TRUCK_CAPACITY_G
        assert truck[VCOL_COST_M] == TRUCK_COST_PER_M
        assert truck[VCOL_SPEED] == TRUCK_SPEED_US_PER_M

    def test_vehicle_column_order_bike_explicit(self):
        _, _, vehicles = generate_random_instance(n_customers=10, n_trucks=0, n_bikes=2)
        assert vehicles.shape[0] == 2
        bike = vehicles[0]
        assert bike[VCOL_TYPE] == VEH_BIKE
        assert bike[VCOL_CAPACITY] == BIKE_CAPACITY_G
        assert bike[VCOL_COST_M] == BIKE_COST_PER_M
        assert bike[VCOL_SPEED] == BIKE_SPEED_US_PER_M

    def test_vehicle_type_column_matches_vtype_constant(self):
        _, _, vehicles = generate_random_instance(n_trucks=2, n_bikes=3, n_customers=10)
        types = vehicles[:, VCOL_TYPE]
        assert np.all((types == 0) | (types == 1))

    def test_truck_then_bikes_order(self):
        n_trucks, n_bikes = 2, 3
        _, _, vehicles = generate_random_instance(n_trucks=n_trucks, n_bikes=n_bikes, n_customers=10)
        assert np.all(vehicles[:n_trucks, VCOL_TYPE] == VEH_TRUCK)
        assert np.all(vehicles[n_trucks:, VCOL_TYPE] == VEH_BIKE)

    def test_all_trucks_identical(self):
        _, _, vehicles = generate_random_instance(n_trucks=5, n_bikes=0, n_customers=10)
        for i in range(1, 5):
            np.testing.assert_array_equal(vehicles[0], vehicles[i])

    def test_all_bikes_identical(self):
        _, _, vehicles = generate_random_instance(n_trucks=0, n_bikes=5, n_customers=10)
        for i in range(1, 5):
            np.testing.assert_array_equal(vehicles[0], vehicles[i])


# ===================================================================
# SECTION 4: Depot & Restricted
# ===================================================================

class TestRandomInstanceDepotRestricted:

    def test_depot_at_area_center_meters(self):
        area_m = 50000
        _, depot, _ = generate_random_instance(
            n_customers=10, n_trucks=2, n_bikes=3, area_size_m=area_m)
        expected = np.array([area_m // 2, area_m // 2], dtype=np.int64)
        np.testing.assert_array_equal(depot, expected)

    def test_restricted_only_0_or_1(self):
        customers, _, _ = generate_random_instance(n_customers=100, n_trucks=2, n_bikes=3)
        restricted = customers[:, COL_RESTRICTED]
        unique = np.unique(restricted)
        assert set(unique.tolist()) <= {0, 1}

    def test_restricted_fraction_approximately_15_percent(self):
        customers, _, _ = generate_random_instance(n_customers=1000, n_trucks=2, n_bikes=3)
        restricted = customers[:, COL_RESTRICTED]
        frac = np.sum(restricted) / len(restricted)
        assert 0.10 < frac < 0.20


# ===================================================================
# SECTION 5: Determinism & Edge Cases
# ===================================================================

class TestRandomInstanceDeterminism:

    def test_seed_reproducibility(self):
        c1, d1, v1 = generate_random_instance(n_customers=20, n_trucks=2, n_bikes=2, seed=42)
        c2, d2, v2 = generate_random_instance(n_customers=20, n_trucks=2, n_bikes=2, seed=42)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(v1, v2)

    def test_different_seeds_different_instances(self):
        c1, _, _ = generate_random_instance(n_customers=20, n_trucks=2, n_bikes=2, seed=1)
        c2, _, _ = generate_random_instance(n_customers=20, n_trucks=2, n_bikes=2, seed=2)
        assert not np.array_equal(c1, c2)

    def test_zero_trucks(self):
        _, _, vehicles = generate_random_instance(n_customers=10, n_trucks=0, n_bikes=5)
        assert vehicles.shape[0] == 5
        assert np.all(vehicles[:, VCOL_TYPE] == VEH_BIKE)

    def test_zero_bikes(self):
        _, _, vehicles = generate_random_instance(n_customers=10, n_trucks=5, n_bikes=0)
        assert vehicles.shape[0] == 5
        assert np.all(vehicles[:, VCOL_TYPE] == VEH_TRUCK)

    def test_large_instance(self):
        customers, _, _ = generate_random_instance(n_customers=1000, n_trucks=10, n_bikes=20)
        assert customers.shape == (1000, CUST_COLS)
        assert np.all(customers[:, COL_DEMAND] > 0)
        assert np.all(customers[:, COL_TW_OPEN] < customers[:, COL_TW_CLOSE])


# ===================================================================
# SECTION 6: generate_clustered_instance()
# ===================================================================

class TestClusteredInstance:

    def test_clustered_shapes(self):
        customers, depot, vehicles = generate_clustered_instance(
            n_customers=50, n_trucks=2, n_bikes=3)
        assert customers.shape == (50, CUST_COLS)
        assert depot.shape == (2,)
        assert vehicles.shape == (5, 4)

    def test_clustered_all_demands_positive(self):
        customers, _, _ = generate_clustered_instance(n_customers=100, n_trucks=2, n_bikes=3)
        assert np.all(customers[:, COL_DEMAND] > 0)

    def test_clustered_tw_open_less_than_close(self):
        customers, _, _ = generate_clustered_instance(n_customers=100, n_trucks=2, n_bikes=3)
        assert np.all(customers[:, COL_TW_OPEN] < customers[:, COL_TW_CLOSE])

    def test_clustered_vehicle_column_order(self):
        _, _, vehicles = generate_clustered_instance(n_customers=50, n_trucks=2, n_bikes=2)
        trucks = vehicles[:2]
        bikes = vehicles[2:]
        assert np.all(trucks[:, VCOL_TYPE] == VEH_TRUCK)
        assert np.all(bikes[:, VCOL_TYPE] == VEH_BIKE)
        assert trucks[0, VCOL_CAPACITY] == TRUCK_CAPACITY_G
        assert bikes[0, VCOL_CAPACITY] == BIKE_CAPACITY_G

    def test_clustered_coordinates_within_area(self):
        area_m = 80000
        customers, _, _ = generate_clustered_instance(
            n_customers=50, n_trucks=2, n_bikes=2, area_size_m=area_m)
        assert np.all(customers[:, COL_X] >= 0)
        assert np.all(customers[:, COL_X] <= area_m)
        assert np.all(customers[:, COL_Y] >= 0)
        assert np.all(customers[:, COL_Y] <= area_m)

    def test_clustered_deterministic(self):
        c1, d1, v1 = generate_clustered_instance(
            n_customers=30, n_trucks=2, n_bikes=2, seed=456)
        c2, d2, v2 = generate_clustered_instance(
            n_customers=30, n_trucks=2, n_bikes=2, seed=456)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(d1, d2)


# ===================================================================
# SECTION 7: Internal Helpers Validation
# ===================================================================

class TestMakeVehiclesHelper:

    def test_make_vehicles_shape_and_order(self):
        vehicles = _make_vehicles(n_trucks=2, n_bikes=3)
        assert vehicles.shape == (5, 4)
        assert vehicles.dtype == np.int64
        assert np.all(vehicles[:2, VCOL_TYPE] == VEH_TRUCK)
        assert np.all(vehicles[2:, VCOL_TYPE] == VEH_BIKE)

    def test_make_vehicles_zero_case(self):
        vehicles = _make_vehicles(n_trucks=0, n_bikes=0)
        assert vehicles.shape[0] == 0


class TestMakeRestrictedHelper:

    def test_restricted_contains_only_01(self):
        rng = np.random.default_rng(42)
        restricted = _make_restricted(rng, n=100, frac=0.15)
        unique = np.unique(restricted)
        assert set(unique.tolist()) <= {0, 1}


class TestMakeCustomersFromXyHelper:

    def test_preserves_xy_coordinates(self):
        rng = np.random.default_rng(42)
        xy = np.array([[10000, 20000], [30000, 40000], [5000, 15000]],
                       dtype=np.int64)
        customers = _make_customers_from_xy(rng, xy, demand_range_g=(3000, 120000))
        np.testing.assert_array_equal(customers[:, COL_X], xy[:, 0])
        np.testing.assert_array_equal(customers[:, COL_Y], xy[:, 1])

    def test_tw_width_minimum_60s(self):
        rng = np.random.default_rng(42)
        xy = np.ones((100, 2), dtype=np.int64) * 5000
        customers = _make_customers_from_xy(rng, xy, demand_range_g=(3000, 120000))
        widths = customers[:, COL_TW_CLOSE] - customers[:, COL_TW_OPEN]
        assert np.all(widths >= 60)
