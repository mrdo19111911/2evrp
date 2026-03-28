"""QA tests for src/data/cost.py -- constant consistency checks.

All costs are now integer (VND), per-meter or per-second units.
No float, no per-km constants.
"""
import pytest
from src.data import cost


class TestTruckCostConsistency:
    """Truck operating cost decomposition (VND per meter)."""

    def test_truck_total_per_m_equals_sum_of_components(self):
        """TRUCK_COST_PER_M = sum of per-meter components."""
        expected_total = (
            cost.TRUCK_FUEL_COST_M
            + cost.TRUCK_MAINTENANCE_M
            + cost.TRUCK_DEPRECIATION_M
            + cost.TRUCK_DRIVER_M
            + cost.TRUCK_OTHER_M
        )
        assert cost.TRUCK_COST_PER_M == expected_total, (
            f"TRUCK_COST_PER_M ({cost.TRUCK_COST_PER_M}) != "
            f"sum of components ({expected_total})")

    def test_truck_cost_per_m_is_integer(self):
        assert isinstance(cost.TRUCK_COST_PER_M, int)


class TestBikeCostConsistency:
    """Bike operating cost decomposition (VND per meter)."""

    def test_bike_total_per_m_equals_sum_of_components(self):
        """BIKE_COST_PER_M = sum of per-meter components."""
        expected_total = (
            cost.BIKE_FUEL_COST_M
            + cost.BIKE_MAINTENANCE_M
            + cost.BIKE_DEPRECIATION_M
            + cost.BIKE_DRIVER_M
            + cost.BIKE_OTHER_M
        )
        assert cost.BIKE_COST_PER_M == expected_total, (
            f"BIKE_COST_PER_M ({cost.BIKE_COST_PER_M}) != "
            f"sum of components ({expected_total})")

    def test_bike_cost_per_m_is_integer(self):
        assert isinstance(cost.BIKE_COST_PER_M, int)


class TestPenalties:
    """Penalty values must be positive integers (VND or VND/s)."""

    def test_all_penalties_are_positive(self):
        penalties = [
            ("PENALTY_UNSERVED", cost.PENALTY_UNSERVED),
            ("PENALTY_UNSERVED_MULTIPLIER", cost.PENALTY_UNSERVED_MULTIPLIER),
            ("PENALTY_LATE", cost.PENALTY_LATE),
            ("PENALTY_OVERLOAD_PER_PCT", cost.PENALTY_OVERLOAD_PER_PCT),
            ("PENALTY_SYNC_FAIL", cost.PENALTY_SYNC_FAIL),
            ("PENALTY_RESTRICTION", cost.PENALTY_RESTRICTION),
            ("PENALTY_OVERTIME", cost.PENALTY_OVERTIME),
            ("PENALTY_MISSING_RELOAD", cost.PENALTY_MISSING_RELOAD),
            ("PENALTY_SYNC_GAP_SEC", cost.PENALTY_SYNC_GAP_SEC),
        ]
        for name, value in penalties:
            assert value > 0, f"{name} = {value} is not positive"

    def test_penalties_are_integers(self):
        assert isinstance(cost.PENALTY_UNSERVED, int)
        assert isinstance(cost.PENALTY_LATE, int)
        assert isinstance(cost.PENALTY_SYNC_GAP_SEC, int)


class TestTimeConstraints:
    """Time-related constants in seconds (i64)."""

    def test_day_length_is_positive_seconds(self):
        assert cost.DAY_LENGTH > 0
        assert cost.DAY_LENGTH == 28800  # 8 hours in seconds

    def test_sync_delta_t_is_positive_seconds(self):
        assert cost.SYNC_DELTA_T > 0
        assert cost.SYNC_DELTA_T == 900  # 15 minutes in seconds

    def test_reload_service_time_is_positive_seconds(self):
        assert cost.RELOAD_SERVICE_TIME > 0
        assert cost.RELOAD_SERVICE_TIME == 300  # 5 minutes in seconds

    def test_time_constants_are_integers(self):
        assert isinstance(cost.DAY_LENGTH, int)
        assert isinstance(cost.SYNC_DELTA_T, int)
        assert isinstance(cost.RELOAD_SERVICE_TIME, int)


class TestCapacityConstraints:
    """Capacity in grams (i64)."""

    def test_bike_capacity_less_than_truck_capacity(self):
        assert cost.BIKE_CAPACITY_G < cost.TRUCK_CAPACITY_G

    def test_both_capacities_are_positive(self):
        assert cost.BIKE_CAPACITY_G > 0
        assert cost.TRUCK_CAPACITY_G > 0

    def test_truck_capacity_2000kg_in_grams(self):
        assert cost.TRUCK_CAPACITY_G == 2000000  # 2000 kg

    def test_bike_capacity_60kg_in_grams(self):
        assert cost.BIKE_CAPACITY_G == 60000  # 60 kg

    def test_capacities_are_integers(self):
        assert isinstance(cost.TRUCK_CAPACITY_G, int)
        assert isinstance(cost.BIKE_CAPACITY_G, int)


class TestSpeedConstraints:
    """Speed in microseconds per meter (us/m)."""

    def test_truck_speed_is_positive(self):
        assert cost.TRUCK_SPEED_US_PER_M > 0

    def test_bike_speed_is_positive(self):
        assert cost.BIKE_SPEED_US_PER_M > 0

    def test_bike_slower_than_truck(self):
        """Higher us/m = slower. Bike (20 km/h) slower than truck (25 km/h)."""
        assert cost.BIKE_SPEED_US_PER_M > cost.TRUCK_SPEED_US_PER_M

    def test_speeds_are_integers(self):
        assert isinstance(cost.TRUCK_SPEED_US_PER_M, int)
        assert isinstance(cost.BIKE_SPEED_US_PER_M, int)


class TestWaitCosts:
    """Wait costs in VND per second."""

    def test_truck_wait_cost_is_positive(self):
        assert cost.TRUCK_WAIT_COST_SEC > 0

    def test_bike_wait_cost_is_positive(self):
        assert cost.BIKE_WAIT_COST_SEC > 0

    def test_wait_costs_are_integers(self):
        assert isinstance(cost.TRUCK_WAIT_COST_SEC, int)
        assert isinstance(cost.BIKE_WAIT_COST_SEC, int)


class TestDeployCosts:
    """Deploy and fixed costs in VND."""

    def test_deploy_costs_are_positive(self):
        assert cost.TRUCK_DEPLOY_COST > 0
        assert cost.BIKE_DEPLOY_COST > 0

    def test_fixed_day_costs_are_positive(self):
        assert cost.TRUCK_FIXED_DAY > 0
        assert cost.BIKE_FIXED_DAY > 0

    def test_deploy_costs_are_integers(self):
        assert isinstance(cost.TRUCK_DEPLOY_COST, int)
        assert isinstance(cost.BIKE_DEPLOY_COST, int)


class TestReloadCosts:
    """Reload handling costs in VND."""

    def test_reload_handling_cost_is_positive(self):
        assert cost.RELOAD_HANDLING_COST > 0

    def test_reload_handling_cost_is_integer(self):
        assert isinstance(cost.RELOAD_HANDLING_COST, int)
