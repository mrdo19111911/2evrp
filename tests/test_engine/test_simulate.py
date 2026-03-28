"""Tests for src/engine/simulate.py — route simulation core.

Load model: start at 0, increase with each delivery (accumulated weight picked up).
"""
import numpy as np
import pytest

from src.engine.simulate import simulate_route, simulate_all_routes
from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, ACT_PAD,
    VEH_TRUCK, VEH_BIKE,
    COL_X, COL_Y, COL_DEMAND, COL_TW_OPEN, COL_TW_CLOSE, COL_SERVICE_TIME,
    ST_CUST, ST_ACTION, ST_ARRIVE, ST_WAIT, ST_START,
    ST_SERVICE, ST_DEPART, ST_LOAD_BEF, ST_LOAD_AFT, ST_FEASIBLE, ST_COLS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist_matrix_from_coords(depot, customer_coords):
    """Build (N+1, N+1) Euclidean distance matrix.  Index 0 = depot."""
    coords = np.vstack([depot.reshape(1, 2), customer_coords])
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


# ---------------------------------------------------------------------------
# simulate_route
# ---------------------------------------------------------------------------

class TestSimulateRouteEmpty:
    """TC1: Route with all padding (-1) gives 0 stops, return_time=0, feasible."""

    def test_empty_route_all_pad(self, tiny_instance, tiny_dist_matrix):
        stops = np.array([-1, -1, -1], dtype=np.int32)
        actions = np.array([ACT_PAD, ACT_PAD, ACT_PAD], dtype=np.int8)
        satellites = np.empty((0, 5), dtype=np.float64)

        state, return_time, total_distance, feasible = simulate_route(
            stops=stops,
            actions=actions,
            vehicle_type=VEH_TRUCK,
            vehicle_capacity=2000.0,
            vehicle_speed=25.0,
            customers=tiny_instance["customers"],
            dist_matrix=tiny_dist_matrix,
            satellites=satellites,
            vehicle_id=0,
            delta_t=15.0,
        )

        assert state.shape == (0, ST_COLS)
        assert return_time == 0.0
        assert total_distance == 0.0
        assert feasible is True


class TestSimulateRouteOneDeliver:
    """TC2: depot(5,5) -> C0(0,5), demand=100, speed=25 km/h.

    Distance = 5 km.  travel_time = 5/25*60 = 12 min.
    tw_open=0, arrive=12 => wait=0, start=12.
    service=15 min, depart=27.
    load: 0 -> 100 (accumulated).
    Return: dist(C0,depot) = 5 km, return_time = 27 + 12 = 39.
    total_distance = 5 + 5 = 10.
    """

    def test_one_deliver_basic(self, tiny_instance, tiny_dist_matrix):
        stops = np.array([0, -1], dtype=np.int32)
        actions = np.array([ACT_DELIVER, ACT_PAD], dtype=np.int8)
        satellites = np.empty((0, 5), dtype=np.float64)

        state, return_time, total_distance, feasible = simulate_route(
            stops=stops,
            actions=actions,
            vehicle_type=VEH_TRUCK,
            vehicle_capacity=2000.0,
            vehicle_speed=25.0,
            customers=tiny_instance["customers"],
            dist_matrix=tiny_dist_matrix,
            satellites=satellites,
            vehicle_id=0,
            delta_t=15.0,
        )

        assert state.shape == (1, ST_COLS)
        assert feasible is True

        # Distance depot(5,5)->C0(0,5) = 5.0 km
        assert total_distance == pytest.approx(10.0, abs=1e-9)

        # Travel time = 5/25*60 = 12 min
        assert state[0, ST_ARRIVE] == pytest.approx(12.0, abs=1e-9)
        assert state[0, ST_WAIT] == pytest.approx(0.0, abs=1e-9)
        assert state[0, ST_START] == pytest.approx(12.0, abs=1e-9)
        assert state[0, ST_SERVICE] == pytest.approx(15.0, abs=1e-9)
        assert state[0, ST_DEPART] == pytest.approx(27.0, abs=1e-9)

        # Load: 0 -> 0+100 = 100 (accumulated)
        assert state[0, ST_LOAD_BEF] == pytest.approx(0.0, abs=1e-9)
        assert state[0, ST_LOAD_AFT] == pytest.approx(100.0, abs=1e-9)

        # Return: depot dist=5, time = 27 + 5/25*60 = 27+12 = 39
        assert return_time == pytest.approx(39.0, abs=1e-9)


class TestSimulateRouteCapacityViolation:
    """TC3: capacity=50, C0 demand=100 -> load exceeds capacity -> infeasible."""

    def test_capacity_violation(self, tiny_instance, tiny_dist_matrix):
        stops = np.array([0], dtype=np.int32)
        actions = np.array([ACT_DELIVER], dtype=np.int8)
        satellites = np.empty((0, 5), dtype=np.float64)

        state, return_time, total_distance, feasible = simulate_route(
            stops=stops,
            actions=actions,
            vehicle_type=VEH_TRUCK,
            vehicle_capacity=50.0,
            vehicle_speed=25.0,
            customers=tiny_instance["customers"],
            dist_matrix=tiny_dist_matrix,
            satellites=satellites,
            vehicle_id=0,
            delta_t=15.0,
        )

        assert feasible is False
        # Load goes 0 + 100 = 100, exceeds capacity 50
        assert state[0, ST_LOAD_AFT] == pytest.approx(100.0, abs=1e-9)


class TestSimulateRouteTWViolation:
    """TC4: arrive=150 (approx), tw_close=120 for C0 -> infeasible."""

    def test_tw_violation(self):
        customers = np.array([
            [62.5, 0.0, 50.0, 0.0, 120.0, 10.0],
        ], dtype=np.float64)
        depot = np.array([0.0, 0.0])
        dm = _dist_matrix_from_coords(depot, customers[:, :2])

        stops = np.array([0], dtype=np.int32)
        actions = np.array([ACT_DELIVER], dtype=np.int8)
        satellites = np.empty((0, 5), dtype=np.float64)

        state, return_time, total_distance, feasible = simulate_route(
            stops=stops,
            actions=actions,
            vehicle_type=VEH_TRUCK,
            vehicle_capacity=2000.0,
            vehicle_speed=25.0,
            customers=customers,
            dist_matrix=dm,
            satellites=satellites,
            vehicle_id=0,
            delta_t=15.0,
        )

        assert feasible is False
        assert state[0, ST_ARRIVE] == pytest.approx(150.0, abs=1e-9)


class TestSimulateRouteTWWait:
    """TC5: arrive=10, tw_open=60 -> wait=50, start=60."""

    def test_tw_wait(self):
        dist_km = 10.0 * 25.0 / 60.0  # = 4.16667 km => arrive at t=10 min
        customers = np.array([
            [dist_km, 0.0, 20.0, 60.0, 200.0, 5.0],
        ], dtype=np.float64)
        depot = np.array([0.0, 0.0])
        dm = _dist_matrix_from_coords(depot, customers[:, :2])

        stops = np.array([0], dtype=np.int32)
        actions = np.array([ACT_DELIVER], dtype=np.int8)
        satellites = np.empty((0, 5), dtype=np.float64)

        state, return_time, total_distance, feasible = simulate_route(
            stops=stops,
            actions=actions,
            vehicle_type=VEH_TRUCK,
            vehicle_capacity=2000.0,
            vehicle_speed=25.0,
            customers=customers,
            dist_matrix=dm,
            satellites=satellites,
            vehicle_id=0,
            delta_t=15.0,
        )

        assert feasible is True
        assert state[0, ST_ARRIVE] == pytest.approx(10.0, abs=1e-9)
        assert state[0, ST_WAIT] == pytest.approx(50.0, abs=1e-9)
        assert state[0, ST_START] == pytest.approx(60.0, abs=1e-9)
        # service_time = 10 + 5*(20/100) = 11.0 min
        assert state[0, ST_SERVICE] == pytest.approx(11.0, abs=1e-9)
        assert state[0, ST_DEPART] == pytest.approx(71.0, abs=1e-9)


class TestSimulateRouteRoundTrip:
    """TC6: depot -> C1 -> depot.  total_distance = dist(depot, C1) * 2."""

    def test_round_trip_distance(self, tiny_instance, tiny_dist_matrix):
        stops = np.array([1], dtype=np.int32)
        actions = np.array([ACT_DELIVER], dtype=np.int8)
        satellites = np.empty((0, 5), dtype=np.float64)

        state, return_time, total_distance, feasible = simulate_route(
            stops=stops,
            actions=actions,
            vehicle_type=VEH_TRUCK,
            vehicle_capacity=2000.0,
            vehicle_speed=25.0,
            customers=tiny_instance["customers"],
            dist_matrix=tiny_dist_matrix,
            satellites=satellites,
            vehicle_id=0,
            delta_t=15.0,
        )

        d = tiny_dist_matrix[0, 2]  # should be 5.0
        assert d == pytest.approx(5.0, abs=1e-9)
        assert total_distance == pytest.approx(d * 2, abs=1e-9)


class TestSimulateRouteThreeDelivers:
    """TC7: 3 DELIVER stops — cumulative load increase, cumulative time increase.

    Route: depot(5,5) -> C4(8,0) -> C3(2,0) -> C1(10,5).
    Truck, speed=25 km/h, capacity=2000.

    Demands: C4=5, C3=8, C1=20.  Total=33.
    Load: 0 -> 5 -> 13 -> 33.
    """

    def test_three_delivers(self, tiny_instance, tiny_dist_matrix):
        stops = np.array([4, 3, 1, -1], dtype=np.int32)
        actions = np.array([ACT_DELIVER, ACT_DELIVER, ACT_DELIVER, ACT_PAD],
                           dtype=np.int8)
        satellites = np.empty((0, 5), dtype=np.float64)

        state, return_time, total_distance, feasible = simulate_route(
            stops=stops,
            actions=actions,
            vehicle_type=VEH_TRUCK,
            vehicle_capacity=2000.0,
            vehicle_speed=25.0,
            customers=tiny_instance["customers"],
            dist_matrix=tiny_dist_matrix,
            satellites=satellites,
            vehicle_id=0,
            delta_t=15.0,
        )

        assert state.shape == (3, ST_COLS)
        assert feasible is True

        # --- Stop 0: C4 ---
        d0 = tiny_dist_matrix[0, 5]  # depot->C4
        assert d0 == pytest.approx(np.sqrt(34.0), abs=1e-4)
        travel0 = d0 / 25.0 * 60.0
        assert state[0, ST_ARRIVE] == pytest.approx(travel0, abs=1e-4)
        assert state[0, ST_LOAD_BEF] == pytest.approx(0.0, abs=1e-9)
        assert state[0, ST_LOAD_AFT] == pytest.approx(5.0, abs=1e-9)  # 0+5

        # --- Stop 1: C3 ---
        d1 = tiny_dist_matrix[5, 4]  # C4->C3
        assert d1 == pytest.approx(6.0, abs=1e-4)
        assert state[1, ST_LOAD_BEF] == pytest.approx(5.0, abs=1e-9)
        assert state[1, ST_LOAD_AFT] == pytest.approx(13.0, abs=1e-9)  # 5+8

        # --- Stop 2: C1 ---
        d2 = tiny_dist_matrix[4, 2]  # C3->C1
        assert d2 == pytest.approx(np.sqrt(89.0), abs=1e-4)
        assert state[2, ST_LOAD_BEF] == pytest.approx(13.0, abs=1e-9)
        assert state[2, ST_LOAD_AFT] == pytest.approx(33.0, abs=1e-9)  # 13+20

        # Total distance = d0 + d1 + d2 + d_back
        d_back = tiny_dist_matrix[2, 0]  # C1->depot = 5.0
        expected_total = d0 + d1 + d2 + d_back
        assert total_distance == pytest.approx(expected_total, abs=1e-4)

        # Time is strictly increasing
        assert state[0, ST_DEPART] < state[1, ST_ARRIVE]
        assert state[1, ST_DEPART] < state[2, ST_ARRIVE]

        # Return time > last depart
        assert return_time > state[2, ST_DEPART]


# ---------------------------------------------------------------------------
# simulate_all_routes
# ---------------------------------------------------------------------------

class TestSimulateAllRoutes:
    """TC: 1 truck + 1 bike -> separate states."""

    def test_one_truck_one_bike(self, tiny_instance, tiny_dist_matrix):
        """Truck delivers C0, bike delivers C2."""
        L = 3
        truck_stops = np.full((1, L), -1, dtype=np.int32)
        truck_actions = np.full((1, L), ACT_PAD, dtype=np.int8)
        truck_stops[0, 0] = 0  # C0
        truck_actions[0, 0] = ACT_DELIVER

        bike_stops = np.full((2, L), -1, dtype=np.int32)
        bike_actions = np.full((2, L), ACT_PAD, dtype=np.int8)
        bike_stops[0, 0] = 2  # C2
        bike_actions[0, 0] = ACT_DELIVER

        satellites = np.empty((0, 5), dtype=np.float64)

        result = simulate_all_routes(
            truck_stops=truck_stops,
            truck_actions=truck_actions,
            bike_stops=bike_stops,
            bike_actions=bike_actions,
            vehicles=tiny_instance["vehicles"],
            customers=tiny_instance["customers"],
            dist_matrix=tiny_dist_matrix,
            satellites=satellites,
            delta_t=15.0,
        )

        truck_states, bike_states = result[0], result[1]
        truck_return_times = result[2]
        bike_return_times = result[3]
        truck_distances = result[4]
        bike_distances = result[5]
        all_feasible = result[6]

        # Truck state: 1 stop
        assert len(truck_states) == 1
        assert truck_states[0].shape == (1, ST_COLS)
        assert truck_states[0][0, ST_CUST] == 0  # C0

        # Bike 0 state: 1 stop
        assert len(bike_states) == 2
        assert bike_states[0].shape == (1, ST_COLS)
        assert bike_states[0][0, ST_CUST] == 2  # C2

        # Bike 1 state: empty
        assert bike_states[1].shape == (0, ST_COLS)

        # Return times are positive for active vehicles
        assert truck_return_times[0] > 0.0
        assert bike_return_times[0] > 0.0
        assert bike_return_times[1] == 0.0

        # Distances are positive for active vehicles
        assert truck_distances[0] > 0.0
        assert bike_distances[0] > 0.0
        assert bike_distances[1] == 0.0

        assert all_feasible is True

    def test_all_empty_routes(self, tiny_instance, tiny_dist_matrix):
        """All routes empty -> feasible, all zeros."""
        L = 2
        truck_stops = np.full((1, L), -1, dtype=np.int32)
        truck_actions = np.full((1, L), ACT_PAD, dtype=np.int8)
        bike_stops = np.full((2, L), -1, dtype=np.int32)
        bike_actions = np.full((2, L), ACT_PAD, dtype=np.int8)
        satellites = np.empty((0, 5), dtype=np.float64)

        result = simulate_all_routes(
            truck_stops=truck_stops,
            truck_actions=truck_actions,
            bike_stops=bike_stops,
            bike_actions=bike_actions,
            vehicles=tiny_instance["vehicles"],
            customers=tiny_instance["customers"],
            dist_matrix=tiny_dist_matrix,
            satellites=satellites,
            delta_t=15.0,
        )

        all_feasible = result[6]
        assert all_feasible is True
        assert np.all(result[2] == 0.0)  # truck_return_times
        assert np.all(result[3] == 0.0)  # bike_return_times
