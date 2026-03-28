"""QA tests for src/engine/fitness.py -- bug detection. All i64.

API:
- evaluate_solution(sol, customers, vehicles, dist_matrix, delta_t_s, penalty_w) -> i64[EV_SIZE]
- compute_route_cost(total_dist_m, total_time_s, n_reloads, vtype, total_wait_s, vehicles, vid) -> i64
"""
import numpy as np
import pytest

from src.engine.fitness import compute_fitness, evaluate_solution, FEASIBLE_ROUTE_BONUS
from src.engine.route_cost import compute_route_cost, compute_makespan
from src.data.constants import (
    ACT_DELIVER, ACT_PAD, VEH_TRUCK, VEH_BIKE, ST_DEPART,
    EV_FITNESS, EV_COST, EV_SYNC_COST, EV_MAKESPAN,
    EV_TOTAL_PENALTY, EV_FEASIBLE, EV_N_FEASIBLE_ROUTES,
)
from src.data.cost import (
    DAY_LENGTH,
    TRUCK_COST_PER_M, TRUCK_DRIVER_SEC, TRUCK_FIXED_DAY, TRUCK_DEPLOY_COST,
    BIKE_COST_PER_M, BIKE_DRIVER_SEC, BIKE_FIXED_DAY, BIKE_DEPLOY_COST,
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M,
)
from tests.test_engine.conftest import make_sol_from_routes, make_penalty_weights


def _make_vehicles(n_trucks=1, n_bikes=1):
    rows = []
    for _ in range(n_trucks):
        rows.append([0, TRUCK_CAPACITY_G, TRUCK_COST_PER_M, TRUCK_SPEED_US_PER_M])
    for _ in range(n_bikes):
        rows.append([1, BIKE_CAPACITY_G, BIKE_COST_PER_M, BIKE_SPEED_US_PER_M])
    return np.array(rows, dtype=np.int64)


# ============================================================================
# TEST 1: Empty Solution
# ============================================================================

class TestEmptySolution:

    def test_evaluate_solution_empty_solution(self, tiny_instance, tiny_dist_matrix):
        sol = make_sol_from_routes(
            n_trucks=1, n_bikes=2, n_customers=5,
            truck_routes=[[]],
            bike_routes=[[], []],
        )
        pw = make_penalty_weights()

        ev = evaluate_solution(
            sol, tiny_instance["customers"],
            tiny_instance["vehicles"], tiny_dist_matrix, 900, pw,
        )

        assert ev[EV_FEASIBLE] == 0, "Empty solution cannot be feasible"
        assert ev[EV_MAKESPAN] == 0
        assert ev[EV_TOTAL_PENALTY] > 0, "All customers unserved -> penalty"

    def test_empty_makespan_is_zero(self):
        truck_rt = np.empty(0, dtype=np.int64)
        bike_rt = np.empty(0, dtype=np.int64)
        assert compute_makespan(truck_rt, bike_rt) == 0


# ============================================================================
# TEST 2: Fitness Monotonicity
# ============================================================================

class TestFitnessMonotonicity:

    def test_fitness_decreases_with_feasible_route(self, tiny_instance, tiny_dist_matrix):
        pw = make_penalty_weights()

        sol1 = make_sol_from_routes(
            n_trucks=1, n_bikes=2, n_customers=5,
            truck_routes=[[(0, ACT_DELIVER), (1, ACT_DELIVER)]],
            bike_routes=[[], []],
        )
        sol2 = make_sol_from_routes(
            n_trucks=1, n_bikes=2, n_customers=5,
            truck_routes=[[(0, ACT_DELIVER), (1, ACT_DELIVER)]],
            bike_routes=[[(3, ACT_DELIVER)], []],
        )

        ev1 = evaluate_solution(
            sol1, tiny_instance["customers"],
            tiny_instance["vehicles"], tiny_dist_matrix, 900, pw,
        )
        ev2 = evaluate_solution(
            sol2, tiny_instance["customers"],
            tiny_instance["vehicles"], tiny_dist_matrix, 900, pw,
        )

        assert ev2[EV_COST] >= ev1[EV_COST]
        assert ev2[EV_TOTAL_PENALTY] <= ev1[EV_TOTAL_PENALTY]


# ============================================================================
# TEST 3: All Unserved Penalty
# ============================================================================

class TestUnservedPenalty:
    def test_all_customers_unserved(self, tiny_instance, tiny_dist_matrix):
        sol = make_sol_from_routes(
            n_trucks=1, n_bikes=2, n_customers=5,
            truck_routes=[[]],
            bike_routes=[[], []],
        )
        pw = make_penalty_weights()

        ev = evaluate_solution(
            sol, tiny_instance["customers"],
            tiny_instance["vehicles"], tiny_dist_matrix, 900, pw,
        )
        assert ev[EV_TOTAL_PENALTY] > 0
        assert ev[EV_TOTAL_PENALTY] >= 500_000


# ============================================================================
# TEST 4: Cost Non-Negative
# ============================================================================

class TestCostNonNegative:
    def test_all_cost_components_non_negative(self, tiny_instance, tiny_dist_matrix):
        sol = make_sol_from_routes(
            n_trucks=1, n_bikes=2, n_customers=5,
            truck_routes=[[(0, ACT_DELIVER), (1, ACT_DELIVER)]],
            bike_routes=[[(3, ACT_DELIVER)], []],
        )
        pw = make_penalty_weights()

        ev = evaluate_solution(
            sol, tiny_instance["customers"],
            tiny_instance["vehicles"], tiny_dist_matrix, 900, pw,
        )
        assert ev[EV_COST] >= 0

    def test_route_cost_zero_distance_is_nonnegative(self):
        vehicles = _make_vehicles(1, 0)
        cost = compute_route_cost(0, 0, 0, VEH_TRUCK, 0, vehicles, 0)
        assert cost >= 0


# ============================================================================
# TEST 5: Makespan = Max Return Time
# ============================================================================

class TestMakespanDefinition:
    def test_makespan_single_truck(self, tiny_instance, tiny_dist_matrix):
        sol = make_sol_from_routes(
            n_trucks=1, n_bikes=2, n_customers=5,
            truck_routes=[[(0, ACT_DELIVER), (1, ACT_DELIVER)]],
            bike_routes=[[], []],
        )
        pw = make_penalty_weights()

        ev = evaluate_solution(
            sol, tiny_instance["customers"],
            tiny_instance["vehicles"], tiny_dist_matrix, 900, pw,
        )
        assert ev[EV_MAKESPAN] > 0
        assert ev[EV_MAKESPAN] <= DAY_LENGTH

    def test_makespan_multi_vehicle(self, tiny_instance, tiny_dist_matrix):
        sol = make_sol_from_routes(
            n_trucks=1, n_bikes=2, n_customers=5,
            truck_routes=[[(0, ACT_DELIVER), (1, ACT_DELIVER)]],
            bike_routes=[[(3, ACT_DELIVER)], [(4, ACT_DELIVER)]],
        )
        pw = make_penalty_weights()

        ev = evaluate_solution(
            sol, tiny_instance["customers"],
            tiny_instance["vehicles"], tiny_dist_matrix, 900, pw,
        )
        assert ev[EV_MAKESPAN] > 0
        assert ev[EV_MAKESPAN] <= DAY_LENGTH


# ============================================================================
# TEST 6: FEASIBLE_ROUTE_BONUS
# ============================================================================

class TestFeasibleRouteBonus:
    def test_bonus_applied_to_feasible_routes(self, tiny_instance, tiny_dist_matrix):
        sol = make_sol_from_routes(
            n_trucks=1, n_bikes=2, n_customers=5,
            truck_routes=[[(0, ACT_DELIVER), (1, ACT_DELIVER)]],
            bike_routes=[[(3, ACT_DELIVER)], []],
        )
        pw = make_penalty_weights()

        ev = evaluate_solution(
            sol, tiny_instance["customers"],
            tiny_instance["vehicles"], tiny_dist_matrix, 900, pw,
        )
        sum_without_bonus = ev[EV_COST] + ev[EV_SYNC_COST] + ev[EV_TOTAL_PENALTY]
        assert ev[EV_FITNESS] <= sum_without_bonus

    def test_bonus_magnitude(self):
        assert FEASIBLE_ROUTE_BONUS == 2_000_000


# ============================================================================
# TEST 7: Fitness Formula
# ============================================================================

class TestFitnessFormula:
    def test_compute_fitness_basic(self):
        fitness = compute_fitness(10_000, 500, 2_000, n_feasible_routes=1)
        expected = 10_000 + 500 + 2_000 - 1 * FEASIBLE_ROUTE_BONUS
        assert fitness == expected

    def test_compute_fitness_no_bonus(self):
        fitness = compute_fitness(10_000, 0, 1_000, n_feasible_routes=0)
        assert fitness == 11_000

    def test_compute_fitness_multiple_bonuses(self):
        fitness = compute_fitness(100_000, 10_000, 50_000, n_feasible_routes=3)
        expected = 100_000 + 10_000 + 50_000 - 3 * FEASIBLE_ROUTE_BONUS
        assert fitness == expected
        assert fitness < 100_000
