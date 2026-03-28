"""Tests for src/solution/_helpers.py -- tuple-based solution. All i64."""
import numpy as np
import pytest

from src.data.constants import (
    ACT_DELIVER, ACT_PAD, ACT_RELOAD, VEH_TRUCK, VEH_BIKE, COL_DEMAND,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_TRUCK_LENGTHS,
    SOL_BIKE_STOPS, SOL_BIKE_ACTIONS, SOL_BIKE_LENGTHS,
    SOL_TRUCK_LOADS, SOL_BIKE_LOADS,
    SOL_TRUCK_DISTANCES, SOL_BIKE_DISTANCES,
)
from src.solution.structure import create_solution
from src.solution._helpers import (
    global_vid,
    update_route_distance,
    get_route_arrays,
    get_loads,
    get_distances,
    update_route_load,
)


class TestGlobalVid:

    def test_global_vid_truck_zero(self):
        assert global_vid(VEH_TRUCK, 0, n_trucks=5) == 0

    def test_global_vid_truck_nonzero(self):
        assert global_vid(VEH_TRUCK, 3, n_trucks=5) == 3

    def test_global_vid_bike_zero(self):
        assert global_vid(VEH_BIKE, 0, n_trucks=5) == 5

    def test_global_vid_bike_nonzero(self):
        assert global_vid(VEH_BIKE, 2, n_trucks=5) == 7

    def test_global_vid_bike_last(self):
        assert global_vid(VEH_BIKE, 4, n_trucks=5) == 9

    def test_global_vid_symmetry_n_trucks_varies(self):
        for n_trucks in [1, 2, 5, 10]:
            assert global_vid(VEH_TRUCK, 0, n_trucks=n_trucks) == 0
            assert global_vid(VEH_BIKE, 0, n_trucks=n_trucks) == n_trucks


class TestUpdateRouteDistance:

    def test_update_route_distance_empty_route(self, tiny_dist_matrix):
        sol = create_solution(n_trucks=1, n_bikes=1, n_customers=5)
        assert sol[SOL_TRUCK_LENGTHS][0] == 0

        update_route_distance(sol, VEH_TRUCK, 0, tiny_dist_matrix)
        assert sol[SOL_TRUCK_DISTANCES][0] == 0

    def test_update_route_distance_single_stop(self, tiny_dist_matrix):
        sol = create_solution(n_trucks=1, n_bikes=1, n_customers=5)
        sol[SOL_TRUCK_STOPS][0, 0] = 0
        sol[SOL_TRUCK_LENGTHS][0] = 1

        update_route_distance(sol, VEH_TRUCK, 0, tiny_dist_matrix)

        expected = tiny_dist_matrix[0, 1] + tiny_dist_matrix[1, 0]
        actual = sol[SOL_TRUCK_DISTANCES][0]
        assert actual == expected

    def test_update_route_distance_multi_stop(self, tiny_dist_matrix):
        sol = create_solution(n_trucks=1, n_bikes=1, n_customers=5)
        sol[SOL_TRUCK_STOPS][0, 0] = 0
        sol[SOL_TRUCK_STOPS][0, 1] = 1
        sol[SOL_TRUCK_LENGTHS][0] = 2

        update_route_distance(sol, VEH_TRUCK, 0, tiny_dist_matrix)

        expected = (tiny_dist_matrix[0, 1] +
                    tiny_dist_matrix[1, 2] +
                    tiny_dist_matrix[2, 0])
        actual = sol[SOL_TRUCK_DISTANCES][0]
        assert actual == expected

    def test_update_route_distance_bike_route(self, tiny_dist_matrix):
        sol = create_solution(n_trucks=1, n_bikes=2, n_customers=5)
        sol[SOL_BIKE_STOPS][0, 0] = 2
        sol[SOL_BIKE_LENGTHS][0] = 1

        update_route_distance(sol, VEH_BIKE, 0, tiny_dist_matrix)

        expected = tiny_dist_matrix[0, 3] + tiny_dist_matrix[3, 0]
        actual = sol[SOL_BIKE_DISTANCES][0]
        assert actual == expected

    def test_update_route_distance_different_vehicles(self, tiny_dist_matrix):
        sol = create_solution(n_trucks=2, n_bikes=2, n_customers=5)

        sol[SOL_TRUCK_STOPS][0, 0] = 0
        sol[SOL_TRUCK_LENGTHS][0] = 1

        sol[SOL_TRUCK_STOPS][1, 0] = 1
        sol[SOL_TRUCK_STOPS][1, 1] = 2
        sol[SOL_TRUCK_LENGTHS][1] = 2

        update_route_distance(sol, VEH_TRUCK, 0, tiny_dist_matrix)
        update_route_distance(sol, VEH_TRUCK, 1, tiny_dist_matrix)

        expected_0 = tiny_dist_matrix[0, 1] + tiny_dist_matrix[1, 0]
        expected_1 = (tiny_dist_matrix[0, 2] +
                      tiny_dist_matrix[2, 3] +
                      tiny_dist_matrix[3, 0])

        assert sol[SOL_TRUCK_DISTANCES][0] == expected_0
        assert sol[SOL_TRUCK_DISTANCES][1] == expected_1


class TestGetRouteArrays:

    def test_get_route_arrays_truck(self, tiny_dist_matrix):
        sol = create_solution(n_trucks=1, n_bikes=2, n_customers=5)
        stops, actions, lengths = get_route_arrays(sol, VEH_TRUCK)

        assert stops is sol[SOL_TRUCK_STOPS]
        assert actions is sol[SOL_TRUCK_ACTIONS]
        assert lengths is sol[SOL_TRUCK_LENGTHS]

    def test_get_route_arrays_bike(self, tiny_dist_matrix):
        sol = create_solution(n_trucks=1, n_bikes=2, n_customers=5)
        stops, actions, lengths = get_route_arrays(sol, VEH_BIKE)

        assert stops is sol[SOL_BIKE_STOPS]
        assert actions is sol[SOL_BIKE_ACTIONS]
        assert lengths is sol[SOL_BIKE_LENGTHS]

    def test_get_route_arrays_truck_shape(self):
        n_trucks, n_bikes, n_customers = 2, 3, 10
        sol = create_solution(n_trucks=n_trucks, n_bikes=n_bikes, n_customers=n_customers)
        stops, actions, lengths = get_route_arrays(sol, VEH_TRUCK)

        assert stops.shape[0] == n_trucks
        assert actions.shape[0] == n_trucks
        assert lengths.shape[0] == n_trucks

    def test_get_route_arrays_bike_shape(self):
        n_trucks, n_bikes, n_customers = 2, 3, 10
        sol = create_solution(n_trucks=n_trucks, n_bikes=n_bikes, n_customers=n_customers)
        stops, actions, lengths = get_route_arrays(sol, VEH_BIKE)

        assert stops.shape[0] == n_bikes
        assert actions.shape[0] == n_bikes
        assert lengths.shape[0] == n_bikes

    def test_get_route_arrays_truck_initial_state(self):
        sol = create_solution(n_trucks=1, n_bikes=1, n_customers=5)
        stops, actions, lengths = get_route_arrays(sol, VEH_TRUCK)

        assert (stops == -1).all()
        assert (actions == ACT_PAD).all()
        assert (lengths == 0).all()


class TestUpdateRouteLoad:

    def test_update_route_load_empty_route(self, tiny_instance):
        sol = create_solution(
            n_trucks=1, n_bikes=1,
            n_customers=tiny_instance["n_customers"]
        )
        customers = tiny_instance["customers"]

        assert sol[SOL_TRUCK_LENGTHS][0] == 0
        update_route_load(sol, VEH_TRUCK, 0, customers)

        assert sol[SOL_TRUCK_LOADS][0] == 0

    def test_update_route_load_single_delivery(self, tiny_instance):
        sol = create_solution(
            n_trucks=1, n_bikes=1,
            n_customers=tiny_instance["n_customers"]
        )
        customers = tiny_instance["customers"]

        sol[SOL_TRUCK_STOPS][0, 0] = 0
        sol[SOL_TRUCK_ACTIONS][0, 0] = ACT_DELIVER
        sol[SOL_TRUCK_LENGTHS][0] = 1

        update_route_load(sol, VEH_TRUCK, 0, customers)

        expected = 100000  # C0: 100kg = 100000g
        actual = sol[SOL_TRUCK_LOADS][0]
        assert actual == expected

    def test_update_route_load_multiple_deliveries(self, tiny_instance):
        sol = create_solution(
            n_trucks=1, n_bikes=1,
            n_customers=tiny_instance["n_customers"]
        )
        customers = tiny_instance["customers"]

        sol[SOL_TRUCK_STOPS][0, 0] = 0
        sol[SOL_TRUCK_ACTIONS][0, 0] = ACT_DELIVER
        sol[SOL_TRUCK_STOPS][0, 1] = 1
        sol[SOL_TRUCK_ACTIONS][0, 1] = ACT_DELIVER
        sol[SOL_TRUCK_LENGTHS][0] = 2

        update_route_load(sol, VEH_TRUCK, 0, customers)

        expected = 100000 + 20000  # C0: 100kg + C1: 20kg
        actual = sol[SOL_TRUCK_LOADS][0]
        assert actual == expected

    def test_update_route_load_with_reload(self, tiny_instance):
        sol = create_solution(
            n_trucks=1, n_bikes=1,
            n_customers=tiny_instance["n_customers"]
        )
        customers = tiny_instance["customers"]

        sol[SOL_TRUCK_STOPS][0, 0] = 0
        sol[SOL_TRUCK_ACTIONS][0, 0] = ACT_DELIVER
        sol[SOL_TRUCK_STOPS][0, 1] = 0
        sol[SOL_TRUCK_ACTIONS][0, 1] = ACT_RELOAD
        sol[SOL_TRUCK_STOPS][0, 2] = 1
        sol[SOL_TRUCK_ACTIONS][0, 2] = ACT_DELIVER
        sol[SOL_TRUCK_LENGTHS][0] = 3

        update_route_load(sol, VEH_TRUCK, 0, customers)

        expected = 100000 + 20000  # Only DELIVER actions count
        actual = sol[SOL_TRUCK_LOADS][0]
        assert actual == expected

    def test_update_route_load_bike(self, tiny_instance):
        sol = create_solution(
            n_trucks=1, n_bikes=2,
            n_customers=tiny_instance["n_customers"]
        )
        customers = tiny_instance["customers"]

        sol[SOL_BIKE_STOPS][0, 0] = 1
        sol[SOL_BIKE_ACTIONS][0, 0] = ACT_DELIVER
        sol[SOL_BIKE_STOPS][0, 1] = 3
        sol[SOL_BIKE_ACTIONS][0, 1] = ACT_DELIVER
        sol[SOL_BIKE_LENGTHS][0] = 2

        update_route_load(sol, VEH_BIKE, 0, customers)

        expected = 20000 + 8000  # C1: 20kg + C3: 8kg
        actual = sol[SOL_BIKE_LOADS][0]
        assert actual == expected


class TestGetLoadsAndDistances:

    def test_get_loads_truck(self):
        sol = create_solution(n_trucks=2, n_bikes=2, n_customers=5)
        loads = get_loads(sol, VEH_TRUCK)
        assert loads is sol[SOL_TRUCK_LOADS]

    def test_get_loads_bike(self):
        sol = create_solution(n_trucks=2, n_bikes=2, n_customers=5)
        loads = get_loads(sol, VEH_BIKE)
        assert loads is sol[SOL_BIKE_LOADS]

    def test_get_distances_truck(self):
        sol = create_solution(n_trucks=2, n_bikes=2, n_customers=5)
        distances = get_distances(sol, VEH_TRUCK)
        assert distances is sol[SOL_TRUCK_DISTANCES]

    def test_get_distances_bike(self):
        sol = create_solution(n_trucks=2, n_bikes=2, n_customers=5)
        distances = get_distances(sol, VEH_BIKE)
        assert distances is sol[SOL_BIKE_DISTANCES]
