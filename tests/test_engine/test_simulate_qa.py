"""QA tests for src/engine/simulate.py -- edge cases. All i64.

Units: meters, seconds, grams, microseconds-per-meter.
"""
import numpy as np
import pytest

from src.engine.simulate import simulate_route_into, simulate_all_routes
from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, ACT_PAD,
    VEH_TRUCK, VEH_BIKE,
    COL_DEMAND, COL_TW_OPEN, COL_TW_CLOSE, COL_SERVICE,
    SAT_CUST, SAT_BIKE, SAT_TRUCK, SAT_KG, SAT_TIME,
    ST_CUST, ST_ACTION, ST_ARRIVE, ST_WAIT, ST_START,
    ST_SERVICE, ST_DEPART, ST_LOAD_BEF, ST_LOAD_AFT, ST_FEASIBLE, ST_COLS,
    MAX_SATELLITES,
)
from src.data.cost import (
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M,
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
    RELOAD_SERVICE_TIME,
)


def _dist_matrix_i64(depot_m, customer_coords_m):
    """Build (N+1, N+1) i64 meters distance matrix."""
    coords = np.vstack([depot_m.reshape(1, 2), customer_coords_m]).astype(np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.round(np.sqrt((diff ** 2).sum(axis=2))).astype(np.int64)


def _simulate_one(stops_list, actions_list, vtype, capacity_g, speed_us_per_m,
                   customers, dist_matrix, satellites=None, n_satellites=0,
                   vehicle_id=0, delta_t_s=900):
    """Helper: simulate a single route. All i64."""
    length = len(stops_list)
    out = np.zeros((max(length, 1), ST_COLS), dtype=np.int64)
    stops = np.array(stops_list, dtype=np.int32)
    actions = np.array(actions_list, dtype=np.int8)
    if satellites is None:
        satellites = np.zeros((MAX_SATELLITES, 5), dtype=np.int64)
    rt, td, feasible = simulate_route_into(
        out, stops, actions, length,
        vtype, capacity_g, speed_us_per_m,
        customers, dist_matrix, satellites, n_satellites,
        vehicle_id, delta_t_s,
    )
    return out[:length], rt, td, feasible


# ============================================================================
# SKETCH 1: empty route
# ============================================================================
class TestSimulateRouteZeroStops:

    def test_empty_array_zero_stops(self, tiny_instance, tiny_dist_matrix):
        out = np.zeros((1, ST_COLS), dtype=np.int64)
        stops = np.array([], dtype=np.int32)
        actions = np.array([], dtype=np.int8)
        sats = np.zeros((MAX_SATELLITES, 5), dtype=np.int64)

        rt, td, feasible = simulate_route_into(
            out, stops, actions, 0,
            VEH_TRUCK, TRUCK_CAPACITY_G, TRUCK_SPEED_US_PER_M,
            tiny_instance["customers"], tiny_dist_matrix,
            sats, 0, 0, 900,
        )
        assert rt == 0
        assert td == 0
        assert feasible is True


# ============================================================================
# SKETCH 2: truck reload -- load decreases
# ============================================================================
class TestSimulateRouteTruckReload:

    def test_truck_reload_decreases_load(self, tiny_instance, tiny_dist_matrix):
        sats = np.zeros((MAX_SATELLITES, 5), dtype=np.int64)
        sats[0] = [0, 0, 0, 30000, 300]  # cust=0, bike=0, truck=0, 30kg, t=300s
        n_sats = 1

        state, rt, td, feasible = _simulate_one(
            [0, 1], [ACT_RELOAD, ACT_DELIVER],
            VEH_TRUCK, TRUCK_CAPACITY_G, TRUCK_SPEED_US_PER_M,
            tiny_instance["customers"], tiny_dist_matrix,
            satellites=sats, n_satellites=n_sats, vehicle_id=0,
        )
        assert state.shape[0] >= 1


# ============================================================================
# SKETCH 3: bike reload -- load increases
# ============================================================================
class TestSimulateRouteBikeReload:

    def test_bike_reload_increases_load(self, tiny_instance, tiny_dist_matrix):
        sats = np.zeros((MAX_SATELLITES, 5), dtype=np.int64)
        sats[0] = [0, 0, 0, 20000, 600]  # 20kg, t=600s
        n_sats = 1

        state, rt, td, feasible = _simulate_one(
            [0, 2], [ACT_RELOAD, ACT_DELIVER],
            VEH_BIKE, BIKE_CAPACITY_G, BIKE_SPEED_US_PER_M,
            tiny_instance["customers"], tiny_dist_matrix,
            satellites=sats, n_satellites=n_sats, vehicle_id=0,
        )
        assert state.shape[0] >= 1
        assert state[0, ST_ACTION] == ACT_RELOAD
        assert state[0, ST_LOAD_AFT] > state[0, ST_LOAD_BEF]


# ============================================================================
# SKETCH 4: TW violation
# ============================================================================
class TestSimulateRouteTimeWindowViolation:

    def test_tw_violation_on_deliver(self):
        # Customer at 100000m, tw_close=5000s. Travel > 5000s at truck speed.
        customers = np.array([
            [100000, 0, 50000, 0, 5000, 300, 0],
        ], dtype=np.int64)
        dm = np.array([[0, 100000], [100000, 0]], dtype=np.int64)

        state, rt, td, feasible = _simulate_one(
            [0], [ACT_DELIVER], VEH_TRUCK, TRUCK_CAPACITY_G, TRUCK_SPEED_US_PER_M,
            customers, dm,
        )
        assert feasible is False
        assert state[0, ST_ARRIVE] > 5000

    def test_tw_violation_on_reload(self):
        customers = np.array([
            [100000, 0, 0, 0, 5000, 0, 0],
        ], dtype=np.int64)
        dm = np.array([[0, 100000], [100000, 0]], dtype=np.int64)

        sats = np.zeros((MAX_SATELLITES, 5), dtype=np.int64)
        sats[0] = [0, 0, 0, 10000, 14400]
        n_sats = 1

        state, rt, td, feasible = _simulate_one(
            [0], [ACT_RELOAD], VEH_TRUCK, TRUCK_CAPACITY_G, TRUCK_SPEED_US_PER_M,
            customers, dm,
            satellites=sats, n_satellites=n_sats,
        )
        assert state[0, ST_ARRIVE] > 5000


# ============================================================================
# SKETCH 5: return time
# ============================================================================
class TestSimulateRouteReturnTime:

    def test_return_time_single_stop(self, tiny_instance, tiny_dist_matrix):
        state, rt, td, feasible = _simulate_one(
            [0], [ACT_DELIVER], VEH_TRUCK, TRUCK_CAPACITY_G, TRUCK_SPEED_US_PER_M,
            tiny_instance["customers"], tiny_dist_matrix,
        )
        last_depart = state[0, ST_DEPART]
        d_back = tiny_dist_matrix[1, 0]
        travel_back = d_back * TRUCK_SPEED_US_PER_M // 1_000_000
        assert rt == last_depart + travel_back

    def test_return_time_multi_stop(self, tiny_instance, tiny_dist_matrix):
        state, rt, td, feasible = _simulate_one(
            [0, 3, 1], [ACT_DELIVER, ACT_DELIVER, ACT_DELIVER],
            VEH_TRUCK, TRUCK_CAPACITY_G, TRUCK_SPEED_US_PER_M,
            tiny_instance["customers"], tiny_dist_matrix,
        )
        last_depart = state[-1, ST_DEPART]
        last_cust = int(state[-1, ST_CUST])
        d_back = tiny_dist_matrix[last_cust + 1, 0]
        travel_back = d_back * TRUCK_SPEED_US_PER_M // 1_000_000
        assert rt == last_depart + travel_back


# ============================================================================
# SKETCH 6: load accumulation
# ============================================================================
class TestSimulateRouteLoadAccumulation:

    def test_load_accumulation_three_stops(self, tiny_instance, tiny_dist_matrix):
        state, rt, td, feasible = _simulate_one(
            [4, 3, 1], [ACT_DELIVER, ACT_DELIVER, ACT_DELIVER],
            VEH_TRUCK, TRUCK_CAPACITY_G, TRUCK_SPEED_US_PER_M,
            tiny_instance["customers"], tiny_dist_matrix,
        )
        # C4=5000g, C3=8000g, C1=20000g
        assert state[0, ST_LOAD_BEF] == 0
        assert state[0, ST_LOAD_AFT] == 5000
        assert state[1, ST_LOAD_BEF] == 5000
        assert state[1, ST_LOAD_AFT] == 13000
        assert state[2, ST_LOAD_BEF] == 13000
        assert state[2, ST_LOAD_AFT] == 33000


# ============================================================================
# SKETCH 7: capacity violation at reload
# ============================================================================
class TestSimulateRouteCapacityViolationDetection:

    def test_capacity_violation_at_reload(self, tiny_instance, tiny_dist_matrix):
        sats = np.zeros((MAX_SATELLITES, 5), dtype=np.int64)
        sats[0] = [0, 0, 0, 65000, 300]  # 65kg > bike 60kg cap
        n_sats = 1

        state, rt, td, feasible = _simulate_one(
            [0], [ACT_RELOAD], VEH_BIKE, BIKE_CAPACITY_G, BIKE_SPEED_US_PER_M,
            tiny_instance["customers"], tiny_dist_matrix,
            satellites=sats, n_satellites=n_sats, vehicle_id=0,
        )
        assert feasible is False
        assert state[0, ST_LOAD_AFT] == 65000


# ============================================================================
# SKETCH 8: wait time
# ============================================================================
class TestSimulateRouteWaitTime:

    def test_wait_time_tw_open(self):
        # Customer at 1000m, tw_open=3600s. Arrive fast -> must wait.
        customers = np.array([
            [1000, 0, 20000, 3600, 10800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        dm = _dist_matrix_i64(depot, customers[:, :2])

        state, rt, td, feasible = _simulate_one(
            [0], [ACT_DELIVER], VEH_TRUCK, TRUCK_CAPACITY_G, TRUCK_SPEED_US_PER_M,
            customers, dm,
        )
        arrive = state[0, ST_ARRIVE]
        wait = state[0, ST_WAIT]
        assert wait == max(0, 3600 - arrive)


# ============================================================================
# SKETCH 9: service time from customer data
# ============================================================================
class TestSimulateRouteServiceTime:

    def test_service_time_delivery(self):
        # Customer with service_s = 600
        customers = np.array([
            [0, 0, 50000, 0, 28800, 600, 0],
        ], dtype=np.int64)
        dm = np.array([[0, 0], [0, 0]], dtype=np.int64)

        state, rt, td, feasible = _simulate_one(
            [0], [ACT_DELIVER], VEH_TRUCK, TRUCK_CAPACITY_G, TRUCK_SPEED_US_PER_M,
            customers, dm,
        )
        assert state[0, ST_SERVICE] == 600
