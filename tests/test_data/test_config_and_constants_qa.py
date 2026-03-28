"""QA tests for src/alns/config.py and src/data/constants.py.

Verifies:
1. make_config scaling behavior for small/medium/large instances
2. ST_COLS definition matches simulate.py state array width
3. All column indices within bounds, no collisions
4. Constants use correct i64 names (COL_SERVICE, VCOL_COST_M, CUST_COLS=7)
"""
import numpy as np
import pytest

from src.alns.config import make_config
from src.data.constants import (
    # Customer columns (i64[:, 7])
    COL_X, COL_Y, COL_DEMAND, COL_TW_OPEN, COL_TW_CLOSE,
    COL_SERVICE, COL_RESTRICTED, CUST_COLS,
    # Vehicle columns (i64[:, 4])
    VCOL_TYPE, VCOL_CAPACITY, VCOL_COST_M, VCOL_SPEED,
    # Action codes
    ACT_DELIVER, ACT_RELOAD, ACT_PAD,
    # Vehicle types
    VEH_TRUCK, VEH_BIKE,
    # Satellite columns
    SAT_CUST, SAT_BIKE, SAT_TRUCK, SAT_KG, SAT_TIME,
    # Route state columns
    ST_CUST, ST_ACTION, ST_ARRIVE, ST_WAIT, ST_START,
    ST_SERVICE, ST_DEPART, ST_LOAD_BEF, ST_LOAD_AFT, ST_FEASIBLE,
    ST_COLS,
    # Config indices
    CFG_MAX_ITERATIONS, CFG_NO_IMPROVE_LIMIT, CFG_Q_MAX, CFG_Q_MAX_PCT,
    CFG_SA_INITIAL_TEMP, CFG_SA_COOLING_RATE, CFG_SEGMENT_LENGTH,
)


class TestConfigSmallInstance:
    """Test make_config with n=50 (small, < 100)."""

    def test_n50_max_iterations_is_20000(self):
        cfg = make_config(50)
        assert cfg[CFG_MAX_ITERATIONS] == 20000

    def test_n50_no_improve_limit_is_2000(self):
        cfg = make_config(50)
        assert cfg[CFG_NO_IMPROVE_LIMIT] == 2000

    def test_n50_q_max_computed(self):
        cfg = make_config(50)
        expected = max(5, int(50 * cfg[CFG_Q_MAX_PCT]))
        assert cfg[CFG_Q_MAX] == expected

    def test_n50_returns_ndarray(self):
        cfg = make_config(50)
        assert isinstance(cfg, np.ndarray)
        assert cfg.dtype == np.float64


class TestConfigMediumInstance:
    """Test make_config with n=300 (medium, 100 <= n < 500)."""

    def test_n300_max_iterations_is_50000(self):
        cfg = make_config(300)
        assert cfg[CFG_MAX_ITERATIONS] == 50000

    def test_n300_no_improve_limit_is_5000(self):
        cfg = make_config(300)
        assert cfg[CFG_NO_IMPROVE_LIMIT] == 5000

    def test_n300_q_max_computed(self):
        cfg = make_config(300)
        expected = max(5, int(300 * cfg[CFG_Q_MAX_PCT]))
        assert cfg[CFG_Q_MAX] == expected


class TestConfigLargeInstance:
    """Test make_config with n=1000 (large, >= 500)."""

    def test_n1000_max_iterations_is_100000(self):
        cfg = make_config(1000)
        assert cfg[CFG_MAX_ITERATIONS] == 100000

    def test_n1000_no_improve_limit_is_10000(self):
        cfg = make_config(1000)
        assert cfg[CFG_NO_IMPROVE_LIMIT] == 10000

    def test_n1000_q_max_scales_with_n(self):
        cfg_small = make_config(50)
        cfg_large = make_config(1000)
        assert cfg_large[CFG_Q_MAX] > cfg_small[CFG_Q_MAX]


class TestConfigBoundaries:

    def test_n99_uses_small_thresholds(self):
        cfg = make_config(99)
        assert cfg[CFG_MAX_ITERATIONS] == 20000

    def test_n100_uses_medium_thresholds(self):
        cfg = make_config(100)
        assert cfg[CFG_MAX_ITERATIONS] == 50000

    def test_n500_uses_large_thresholds(self):
        cfg = make_config(500)
        assert cfg[CFG_MAX_ITERATIONS] == 100000

    def test_q_max_minimum_5_for_small_n(self):
        cfg = make_config(1)
        assert cfg[CFG_Q_MAX] >= 5


class TestConfigOverrides:

    def test_override_single_key(self):
        cfg = make_config(50, overrides={CFG_MAX_ITERATIONS: 999})
        assert cfg[CFG_MAX_ITERATIONS] == 999

    def test_override_multiple_keys(self):
        overrides = {CFG_MAX_ITERATIONS: 1000, CFG_SA_INITIAL_TEMP: 200.0}
        cfg = make_config(50, overrides=overrides)
        assert cfg[CFG_MAX_ITERATIONS] == 1000
        assert cfg[CFG_SA_INITIAL_TEMP] == 200.0

    def test_none_overrides_equals_no_overrides(self):
        cfg1 = make_config(50)
        cfg2 = make_config(50, overrides=None)
        np.testing.assert_array_equal(cfg1, cfg2)


class TestConstantsCustomerColumns:

    def test_customer_columns_sequential(self):
        assert COL_X == 0
        assert COL_Y == 1
        assert COL_DEMAND == 2
        assert COL_TW_OPEN == 3
        assert COL_TW_CLOSE == 4
        assert COL_SERVICE == 5
        assert COL_RESTRICTED == 6

    def test_customer_columns_unique(self):
        cols = [COL_X, COL_Y, COL_DEMAND, COL_TW_OPEN, COL_TW_CLOSE,
                COL_SERVICE, COL_RESTRICTED]
        assert len(cols) == len(set(cols))

    def test_cust_cols_is_7(self):
        assert CUST_COLS == 7


class TestConstantsVehicleColumns:

    def test_vehicle_columns_sequential(self):
        assert VCOL_TYPE == 0
        assert VCOL_CAPACITY == 1
        assert VCOL_COST_M == 2
        assert VCOL_SPEED == 3

    def test_vehicle_columns_unique(self):
        cols = [VCOL_TYPE, VCOL_CAPACITY, VCOL_COST_M, VCOL_SPEED]
        assert len(cols) == len(set(cols))


class TestConstantsActionCodes:

    def test_action_codes_distinct(self):
        codes = [ACT_DELIVER, ACT_RELOAD, ACT_PAD]
        assert len(codes) == len(set(codes))

    def test_deliver_is_zero(self):
        assert ACT_DELIVER == 0

    def test_reload_is_one(self):
        assert ACT_RELOAD == 1

    def test_pad_is_negative(self):
        assert ACT_PAD < 0


class TestConstantsVehicleTypes:

    def test_vehicle_types_distinct(self):
        assert VEH_TRUCK != VEH_BIKE

    def test_truck_is_zero(self):
        assert VEH_TRUCK == 0

    def test_bike_is_one(self):
        assert VEH_BIKE == 1


class TestConstantsSatelliteColumns:

    def test_satellite_columns_sequential(self):
        assert SAT_CUST == 0
        assert SAT_BIKE == 1
        assert SAT_TRUCK == 2
        assert SAT_KG == 3
        assert SAT_TIME == 4

    def test_satellite_columns_unique(self):
        cols = [SAT_CUST, SAT_BIKE, SAT_TRUCK, SAT_KG, SAT_TIME]
        assert len(cols) == len(set(cols))


class TestConstantsRouteStateColumns:

    def test_route_state_columns_sequential(self):
        assert ST_CUST == 0
        assert ST_ACTION == 1
        assert ST_ARRIVE == 2
        assert ST_WAIT == 3
        assert ST_START == 4
        assert ST_SERVICE == 5
        assert ST_DEPART == 6
        assert ST_LOAD_BEF == 7
        assert ST_LOAD_AFT == 8
        assert ST_FEASIBLE == 9

    def test_route_state_columns_unique(self):
        cols = [ST_CUST, ST_ACTION, ST_ARRIVE, ST_WAIT, ST_START,
                ST_SERVICE, ST_DEPART, ST_LOAD_BEF, ST_LOAD_AFT, ST_FEASIBLE]
        assert len(cols) == len(set(cols))

    def test_st_cols_is_10(self):
        assert ST_COLS == 10

    def test_all_st_indices_within_bounds(self):
        cols = [ST_CUST, ST_ACTION, ST_ARRIVE, ST_WAIT, ST_START,
                ST_SERVICE, ST_DEPART, ST_LOAD_BEF, ST_LOAD_AFT, ST_FEASIBLE]
        for col_idx in cols:
            assert col_idx < ST_COLS

    def test_st_cols_exact_range(self):
        cols = [ST_CUST, ST_ACTION, ST_ARRIVE, ST_WAIT, ST_START,
                ST_SERVICE, ST_DEPART, ST_LOAD_BEF, ST_LOAD_AFT, ST_FEASIBLE]
        assert set(cols) == set(range(ST_COLS))


class TestConfigAndScaling:

    def test_config_produces_valid_q_max(self):
        for n in [50, 300, 1000]:
            cfg = make_config(n)
            q_max = int(cfg[CFG_Q_MAX])
            assert q_max <= n
            assert q_max >= 5

    def test_config_iteration_scales_appropriately(self):
        cfg_small = make_config(50)
        cfg_med = make_config(300)
        cfg_large = make_config(1000)
        assert cfg_small[CFG_MAX_ITERATIONS] <= cfg_med[CFG_MAX_ITERATIONS]
        assert cfg_med[CFG_MAX_ITERATIONS] <= cfg_large[CFG_MAX_ITERATIONS]
