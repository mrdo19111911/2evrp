"""Tests for src/engine/simulate.py -- route simulation core. All i64.

Load model: start at 0, increase with each delivery (accumulated weight in grams).
Units: meters, seconds, grams, microseconds-per-meter.
"""
import numpy as np
import pytest

from src.engine.simulate import simulate_route_into, simulate_all_routes
from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, ACT_PAD,
    VEH_TRUCK, VEH_BIKE,
    COL_X, COL_Y, COL_DEMAND, COL_TW_OPEN, COL_TW_CLOSE, COL_SERVICE,
    ST_CUST, ST_ACTION, ST_ARRIVE, ST_WAIT, ST_START,
    ST_SERVICE, ST_DEPART, ST_LOAD_BEF, ST_LOAD_AFT, ST_FEASIBLE, ST_COLS,
    MAX_SATELLITES,
)
from src.data.cost import (
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M,
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
    TRUCK_COST_PER_M, BIKE_COST_PER_M,
    RELOAD_SERVICE_TIME,
)
from tests.test_engine.conftest import make_sol_from_routes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist_matrix_i64(depot_m, customer_coords_m):
    """Build (N+1, N+1) Euclidean distance matrix, i64 meters. Index 0 = depot."""
    coords = np.vstack([depot_m.reshape(1, 2), customer_coords_m]).astype(np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_f = np.sqrt((diff ** 2).sum(axis=2))
    return np.round(dist_f).astype(np.int64)


def _simulate_one(stops_list, actions_list, vtype, capacity_g, speed_us_per_m,
                   customers, dist_matrix, satellites=None, n_satellites=0,
                   vehicle_id=0, delta_t_s=900):
    """Helper: simulate a single route using simulate_route_into. All i64."""
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


# ---------------------------------------------------------------------------
# simulate_route_into
# ---------------------------------------------------------------------------

class TestSimulateRouteEmpty:
    """TC1: Route with 0 length gives return_time=0, feasible."""

    def test_empty_route(self, tiny_instance, tiny_dist_matrix):
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


class TestSimulateRouteOneDeliver:
    """TC2: depot(5000,5000) -> C0(0,5000), demand=100000g.

    Distance = 5000m. travel_s = 5000 * 144000 / 1_000_000 = 720s.
    tw_open=0, arrive=720 => wait=0, start=720.
    service=900s, depart=1620.
    Return: dist(C0,depot) = 5000m, return_time = 1620 + 720 = 2340.
    total_distance = 5000 + 5000 = 10000m.
    """

    def test_one_deliver_basic(self, tiny_instance, tiny_dist_matrix):
        state, rt, td, feasible = _simulate_one(
            [0], [ACT_DELIVER], VEH_TRUCK, TRUCK_CAPACITY_G, TRUCK_SPEED_US_PER_M,
            tiny_instance["customers"], tiny_dist_matrix,
        )

        assert feasible is True
        d = tiny_dist_matrix[0, 1]  # depot -> C0
        travel_s = d * TRUCK_SPEED_US_PER_M // 1_000_000
        assert td == d * 2  # round trip
        assert state[0, ST_ARRIVE] == travel_s
        assert state[0, ST_WAIT] == 0
        assert state[0, ST_START] == travel_s
        assert state[0, ST_SERVICE] == 900  # C0 service time
        assert state[0, ST_DEPART] == travel_s + 900
        assert state[0, ST_LOAD_BEF] == 0
        assert state[0, ST_LOAD_AFT] == 100000  # C0 demand = 100kg
        assert rt == state[0, ST_DEPART] + travel_s


class TestSimulateRouteCapacityViolation:
    """TC3: capacity=50000g, C0 demand=100000g -> infeasible."""

    def test_capacity_violation(self, tiny_instance, tiny_dist_matrix):
        state, rt, td, feasible = _simulate_one(
            [0], [ACT_DELIVER], VEH_TRUCK, 50000, TRUCK_SPEED_US_PER_M,
            tiny_instance["customers"], tiny_dist_matrix,
        )
        assert feasible is False
        assert state[0, ST_LOAD_AFT] == 100000


class TestSimulateRouteTWViolation:
    """TC4: Far customer with tight TW -> infeasible."""

    def test_tw_violation(self):
        # Customer 62500m away, tw_close=5000s. At truck speed: 62500*144000/1e6 = 9000s > 5000
        customers = np.array([
            [62500, 0, 50000, 0, 5000, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        dm = _dist_matrix_i64(depot, customers[:, :2])

        state, rt, td, feasible = _simulate_one(
            [0], [ACT_DELIVER], VEH_TRUCK, TRUCK_CAPACITY_G, TRUCK_SPEED_US_PER_M,
            customers, dm,
        )
        assert feasible is False
        assert state[0, ST_ARRIVE] > 5000


class TestSimulateRouteTWWait:
    """TC5: arrive before tw_open -> must wait."""

    def test_tw_wait(self):
        # Customer at 1000m, truck speed 144000 us/m -> arrive = 1000*144000/1e6 = 144s
        # tw_open=3600 -> wait = 3600-144 = 3456
        customers = np.array([
            [1000, 0, 20000, 3600, 10800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        dm = _dist_matrix_i64(depot, customers[:, :2])

        state, rt, td, feasible = _simulate_one(
            [0], [ACT_DELIVER], VEH_TRUCK, TRUCK_CAPACITY_G, TRUCK_SPEED_US_PER_M,
            customers, dm,
        )

        assert feasible is True
        arrive = state[0, ST_ARRIVE]
        wait = state[0, ST_WAIT]
        assert wait == max(0, 3600 - arrive)
        assert state[0, ST_START] == arrive + wait
        assert state[0, ST_SERVICE] == 300
        assert state[0, ST_DEPART] == state[0, ST_START] + 300


class TestSimulateRouteRoundTrip:
    """TC6: depot -> C1 -> depot. total_distance = dist(depot, C1) * 2."""

    def test_round_trip_distance(self, tiny_instance, tiny_dist_matrix):
        state, rt, td, feasible = _simulate_one(
            [1], [ACT_DELIVER], VEH_TRUCK, TRUCK_CAPACITY_G, TRUCK_SPEED_US_PER_M,
            tiny_instance["customers"], tiny_dist_matrix,
        )
        d = tiny_dist_matrix[0, 2]  # depot -> C1
        assert td == d * 2


class TestSimulateRouteThreeDelivers:
    """TC7: 3 DELIVER stops -- cumulative load increase."""

    def test_three_delivers(self, tiny_instance, tiny_dist_matrix):
        custs = tiny_instance["customers"]
        state, rt, td, feasible = _simulate_one(
            [4, 3, 1], [ACT_DELIVER, ACT_DELIVER, ACT_DELIVER],
            VEH_TRUCK, TRUCK_CAPACITY_G, TRUCK_SPEED_US_PER_M,
            custs, tiny_dist_matrix,
        )

        assert feasible is True

        # Load accumulation: C4=5000g, C3=8000g, C1=20000g
        assert state[0, ST_LOAD_BEF] == 0
        assert state[0, ST_LOAD_AFT] == 5000
        assert state[1, ST_LOAD_BEF] == 5000
        assert state[1, ST_LOAD_AFT] == 13000
        assert state[2, ST_LOAD_BEF] == 13000
        assert state[2, ST_LOAD_AFT] == 33000

        # Total distance
        d0 = tiny_dist_matrix[0, 5]   # depot -> C4
        d1 = tiny_dist_matrix[5, 4]   # C4 -> C3
        d2 = tiny_dist_matrix[4, 2]   # C3 -> C1
        d_back = tiny_dist_matrix[2, 0]  # C1 -> depot
        assert td == d0 + d1 + d2 + d_back

        # Time ordering
        assert state[0, ST_DEPART] < state[1, ST_ARRIVE]
        assert state[1, ST_DEPART] < state[2, ST_ARRIVE]
        assert rt > state[2, ST_DEPART]


# ---------------------------------------------------------------------------
# simulate_all_routes
# ---------------------------------------------------------------------------

class TestSimulateAllRoutes:
    """TC: 1 truck + 2 bikes via solution tuple."""

    def test_one_truck_one_bike(self, tiny_instance, tiny_dist_matrix):
        """Truck delivers C0, bike delivers C2."""
        sol = make_sol_from_routes(
            n_trucks=1, n_bikes=2, n_customers=5,
            truck_routes=[[(0, ACT_DELIVER)]],
            bike_routes=[[(2, ACT_DELIVER)], []],
        )

        result = simulate_all_routes(
            sol, tiny_instance["vehicles"],
            tiny_instance["customers"], tiny_dist_matrix, 900,
        )
        truck_sim, bike_sim = result[0], result[1]
        truck_rt, bike_rt = result[2], result[3]
        truck_dist, bike_dist = result[4], result[5]
        all_feasible = result[6]

        assert truck_sim.shape[0] == 1
        assert truck_sim[0, 0, ST_CUST] == 0
        assert bike_sim.shape[0] == 2
        assert bike_sim[0, 0, ST_CUST] == 2

        assert truck_rt[0] > 0
        assert bike_rt[0] > 0
        assert bike_rt[1] == 0
        assert truck_dist[0] > 0
        assert bike_dist[0] > 0
        assert bike_dist[1] == 0
        assert all_feasible is True

    def test_all_empty_routes(self, tiny_instance, tiny_dist_matrix):
        """All routes empty -> feasible, all zeros."""
        sol = make_sol_from_routes(
            n_trucks=1, n_bikes=2, n_customers=5,
            truck_routes=[[]],
            bike_routes=[[], []],
        )

        result = simulate_all_routes(
            sol, tiny_instance["vehicles"],
            tiny_instance["customers"], tiny_dist_matrix, 900,
        )
        all_feasible = result[6]
        assert all_feasible is True
        assert np.all(result[2] == 0)
        assert np.all(result[3] == 0)
