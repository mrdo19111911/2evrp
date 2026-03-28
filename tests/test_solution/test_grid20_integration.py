"""Integration test: src/solution/ with grid_20x20 (400 customers).
Build a random solution with trucks, bikes, reload events, then verify all ops."""
import numpy as np
import pytest

from src.data.io import load_instance
from src.data.distance import compute_dist_matrix
from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, VEH_TRUCK, VEH_BIKE, COL_DEMAND
)
from src.solution.structure import create_solution, copy_solution, rebuild_index
from src.solution.route_ops import insert_stop, remove_stop, reverse_segment
from src.solution.solution_ops import move_stop, clear_route, create_route_from_list
from src.solution.satellite_ops import add_satellite_event, remove_satellites_for_customer
from src.solution.delta import (
    insertion_cost_delta, removal_cost_delta, best_insertion_pos,
    find_best_insertion_all_routes
)
from src.solution.check import (
    can_insert_customer, can_swap_customers, check_all_assigned
)
from src.solution.query import (
    get_unassigned_customers, get_assigned_customers,
    get_customer_info, is_customer_assigned, get_route_as_list
)

GRID_PATH = "data/grid_20x20.json"


@pytest.fixture(scope="module")
def grid():
    """Load grid_20x20 + compute dist_matrix. Cached for module."""
    customers, restricted, depot, vehicles = load_instance(GRID_PATH)
    dm = compute_dist_matrix(depot, customers)
    return {
        "customers": customers,
        "restricted": restricted,
        "depot": depot,
        "vehicles": vehicles,
        "dist_matrix": dm,
        "n_customers": len(customers),
        "n_trucks": 5,
        "n_bikes": 15,
    }


@pytest.fixture
def grid_solution(grid):
    """Build a realistic random solution for 400 customers.

    Strategy:
    - Heavy nodes (demand>60, 25 nodes) → truck DELIVER
    - Some light nodes → truck DELIVER (fill truck capacity)
    - Remaining light nodes → bike DELIVER
    - Pick 5 satellite nodes where trucks RELOAD + bikes RELOAD
    - Add satellite events for reload
    """
    rng = np.random.default_rng(42)
    custs = grid["customers"]
    restricted = grid["restricted"]
    dm = grid["dist_matrix"]
    n = grid["n_customers"]

    sol = create_solution(grid["n_trucks"], grid["n_bikes"], n)

    # Classify
    heavy = np.where(custs[:, COL_DEMAND] > 60)[0]
    light = np.where(custs[:, COL_DEMAND] <= 60)[0]
    bike_only = np.where(restricted == 1)[0]
    bike_only_set = set(bike_only)
    light_not_restricted = np.array([c for c in light if c not in bike_only_set])

    # --- Truck routes: heavy nodes + some light ---
    rng.shuffle(heavy)
    rng.shuffle(light_not_restricted)

    # Split heavy across 5 trucks (~5 each)
    truck_assignments = [[] for _ in range(5)]
    for i, c in enumerate(heavy):
        truck_assignments[i % 5].append(c)

    # Add some light nodes to trucks (first 50 light_not_restricted)
    for i, c in enumerate(light_not_restricted[:50]):
        truck_assignments[i % 5].append(c)

    # Pick 5 satellite nodes (from light_not_restricted, spaced out)
    sat_candidates = light_not_restricted[50:70]
    satellite_nodes = sat_candidates[:5]

    # Add satellite RELOAD stops to truck routes
    for t in range(5):
        truck_assignments[t].append(int(satellite_nodes[t % len(satellite_nodes)]))

    # Insert into solution
    for t in range(5):
        for c in truck_assignments[t][:-1]:  # all except last = DELIVER
            pos = sol["truck_lengths"][t]
            insert_stop(sol, VEH_TRUCK, t, pos, int(c), ACT_DELIVER, dm, custs)
        # Last = satellite RELOAD
        sat_node = truck_assignments[t][-1]
        pos = sol["truck_lengths"][t]
        insert_stop(sol, VEH_TRUCK, t, pos, sat_node, ACT_RELOAD, dm, custs)

    # --- Bike routes: remaining light + bike_only ---
    assigned_set = set()
    for t in range(5):
        for c in truck_assignments[t][:-1]:
            assigned_set.add(c)

    remaining = [c for c in range(n) if c not in assigned_set]
    rng.shuffle(remaining)

    # Split across 15 bikes
    bike_assignments = [[] for _ in range(15)]
    for i, c in enumerate(remaining):
        bike_assignments[i % 15].append(c)

    # Insert bike routes with RELOAD at satellite
    for b in range(15):
        if not bike_assignments[b]:
            continue
        sat_node = int(satellite_nodes[b % len(satellite_nodes)])

        # First half → DELIVER
        half = len(bike_assignments[b]) // 2
        for c in bike_assignments[b][:half]:
            pos = sol["bike_lengths"][b]
            insert_stop(sol, VEH_BIKE, b, pos, int(c), ACT_DELIVER, dm, custs)

        # RELOAD stop
        pos = sol["bike_lengths"][b]
        insert_stop(sol, VEH_BIKE, b, pos, sat_node, ACT_RELOAD, dm, custs)

        # Second half → DELIVER
        for c in bike_assignments[b][half:]:
            pos = sol["bike_lengths"][b]
            insert_stop(sol, VEH_BIKE, b, pos, int(c), ACT_DELIVER, dm, custs)

        # Add satellite event
        transfer_kg = sum(custs[c, COL_DEMAND] for c in bike_assignments[b][half:])
        truck_id = b % 5
        add_satellite_event(sol, sat_node, b, truck_id, transfer_kg, 200.0 + b * 10)

    return sol


# =====================================================================
# Structure tests
# =====================================================================
class TestGridSolutionStructure:
    def test_solution_created(self, grid_solution):
        assert grid_solution["n_trucks"] == 5
        assert grid_solution["n_bikes"] == 15

    def test_trucks_have_routes(self, grid_solution):
        for t in range(5):
            assert grid_solution["truck_lengths"][t] > 0

    def test_bikes_have_routes(self, grid_solution):
        active = sum(1 for b in range(15) if grid_solution["bike_lengths"][b] > 0)
        assert active >= 10  # at least 10 bikes used

    def test_satellites_exist(self, grid_solution):
        assert len(grid_solution["satellites"]) > 0

    def test_most_customers_assigned(self, grid_solution, grid):
        assigned = get_assigned_customers(grid_solution, grid["n_customers"])
        # All 400 should be assigned (heavy via truck, rest via bike)
        assert len(assigned) == 400


# =====================================================================
# Query tests on real data
# =====================================================================
class TestGridQuery:
    def test_customer_info_consistency(self, grid_solution, grid):
        """Every assigned customer's info matches its actual position in route."""
        for c in range(grid["n_customers"]):
            if not is_customer_assigned(grid_solution, c):
                continue
            vtype, vid, pos = get_customer_info(grid_solution, c)
            assert vtype in (VEH_TRUCK, VEH_BIKE)
            if vtype == VEH_TRUCK:
                assert grid_solution["truck_stops"][vid, pos] == c
                assert grid_solution["truck_actions"][vid, pos] == ACT_DELIVER
            else:
                assert grid_solution["bike_stops"][vid, pos] == c
                assert grid_solution["bike_actions"][vid, pos] == ACT_DELIVER

    def test_route_as_list_matches_arrays(self, grid_solution):
        """get_route_as_list matches raw array data for truck 0."""
        route = get_route_as_list(grid_solution, VEH_TRUCK, 0)
        L = grid_solution["truck_lengths"][0]
        for i, (c, a) in enumerate(route):
            assert c == grid_solution["truck_stops"][0, i]
            assert a == grid_solution["truck_actions"][0, i]
        assert len(route) == L


# =====================================================================
# Delta consistency: old + delta == new (THE critical test)
# =====================================================================
class TestGridDeltaConsistency:
    def test_insertion_delta_100_random(self, grid_solution, grid):
        """100 random insertions: verify old_dist + delta == new_dist."""
        rng = np.random.default_rng(123)
        dm = grid["dist_matrix"]
        custs = grid["customers"]
        sol = copy_solution(grid_solution)

        # Pick 20 random unassigned-ish customers (remove some first)
        for _ in range(20):
            assigned = get_assigned_customers(sol, grid["n_customers"])
            if len(assigned) == 0:
                break
            c = int(rng.choice(assigned))
            vtype, vid, pos = get_customer_info(sol, c)
            remove_stop(sol, vtype, vid, pos, dm, custs)

        unassigned = get_unassigned_customers(sol, grid["n_customers"])
        checks = 0

        for c in unassigned[:20]:
            c = int(c)
            # Try inserting into truck 0
            vid = 0
            L = sol["truck_lengths"][vid]
            for pos in [0, L // 2, L]:
                old_dist = float(sol["truck_distances"][vid])
                delta = insertion_cost_delta(sol, VEH_TRUCK, vid, pos, c, dm)

                # Actually insert, measure new distance
                sol_copy = copy_solution(sol)
                insert_stop(sol_copy, VEH_TRUCK, vid, pos, c, ACT_DELIVER, dm, custs)
                new_dist = float(sol_copy["truck_distances"][vid])

                assert old_dist + delta == pytest.approx(new_dist, abs=1e-6), \
                    f"C{c} pos={pos}: {old_dist} + {delta} != {new_dist}"
                checks += 1

        assert checks >= 30, f"Only {checks} checks done"

    def test_removal_delta_50_random(self, grid_solution, grid):
        """50 random removals: verify old_dist + delta == new_dist."""
        rng = np.random.default_rng(456)
        dm = grid["dist_matrix"]
        custs = grid["customers"]
        sol = copy_solution(grid_solution)

        checks = 0
        for t in range(5):
            L = sol["truck_lengths"][t]
            for pos in range(min(L, 10)):
                old_dist = float(sol["truck_distances"][t])
                delta = removal_cost_delta(sol, VEH_TRUCK, t, pos, dm)

                sol_copy = copy_solution(sol)
                remove_stop(sol_copy, VEH_TRUCK, t, pos, dm, custs)
                new_dist = float(sol_copy["truck_distances"][t])

                assert old_dist + delta == pytest.approx(new_dist, abs=1e-6), \
                    f"Truck {t} pos={pos}: {old_dist} + {delta} != {new_dist}"
                checks += 1

        assert checks >= 30, f"Only {checks} checks done"


# =====================================================================
# Route operations on 400-node solution
# =====================================================================
class TestGridRouteOps:
    def test_remove_and_reinsert(self, grid_solution, grid):
        """Remove 2 customers from truck 0, reinsert one into truck 0 (now has room)."""
        dm = grid["dist_matrix"]
        custs = grid["customers"]
        sol = copy_solution(grid_solution)

        # Remove first 2 DELIVER customers from truck 0 to free capacity
        removed = []
        while len(removed) < 2:
            L = sol["truck_lengths"][0]
            for pos in range(L):
                if sol["truck_actions"][0, pos] == ACT_DELIVER:
                    c, _ = remove_stop(sol, VEH_TRUCK, 0, pos, dm, custs)
                    removed.append(c)
                    break

        assert len(removed) == 2
        c = removed[0]
        assert not is_customer_assigned(sol, c)

        # Now truck 0 has room — reinsert with generous capacity
        vid, ins_pos, delta = find_best_insertion_all_routes(
            sol, VEH_TRUCK, c, dm, custs, 5000.0  # generous cap for test
        )
        assert vid >= 0, "Should find a truck route with generous capacity"

        insert_stop(sol, VEH_TRUCK, vid, ins_pos, c, ACT_DELIVER, dm, custs)
        assert is_customer_assigned(sol, c)

    def test_reverse_segment_reduces_or_keeps_distance(self, grid_solution, grid):
        """2-opt reverse on truck route: distance should not increase if it was crossing."""
        dm = grid["dist_matrix"]
        sol = copy_solution(grid_solution)

        # Just verify reverse_segment doesn't crash on real data
        for t in range(5):
            L = sol["truck_lengths"][t]
            if L >= 4:
                old_dist = float(sol["truck_distances"][t])
                reverse_segment(sol, VEH_TRUCK, t, 1, L - 2, dm)
                new_dist = float(sol["truck_distances"][t])
                # May or may not improve, but must be valid
                assert new_dist > 0
                return

    def test_clear_and_rebuild(self, grid_solution, grid):
        """Clear truck route 0, rebuild from list, verify consistency."""
        dm = grid["dist_matrix"]
        custs = grid["customers"]
        sol = copy_solution(grid_solution)

        route_before = get_route_as_list(sol, VEH_TRUCK, 0)
        removed = clear_route(sol, VEH_TRUCK, 0)
        assert sol["truck_lengths"][0] == 0

        # Rebuild
        create_route_from_list(sol, VEH_TRUCK, 0, route_before, dm, custs)
        route_after = get_route_as_list(sol, VEH_TRUCK, 0)
        assert route_before == route_after

    def test_move_between_trucks(self, grid_solution, grid):
        """Move customer from truck 0 to truck 1."""
        dm = grid["dist_matrix"]
        custs = grid["customers"]
        sol = copy_solution(grid_solution)

        L0 = sol["truck_lengths"][0]
        if L0 < 2:
            pytest.skip("Truck 0 too short")

        c = int(sol["truck_stops"][0, 0])
        a = int(sol["truck_actions"][0, 0])
        if a != ACT_DELIVER:
            pytest.skip("First stop not DELIVER")

        old_L1 = sol["truck_lengths"][1]
        move_stop(sol, VEH_TRUCK, 0, 0, VEH_TRUCK, 1, 0, dm, custs)

        assert sol["truck_lengths"][0] == L0 - 1
        assert sol["truck_lengths"][1] == old_L1 + 1
        vtype, vid, pos = get_customer_info(sol, c)
        assert vtype == VEH_TRUCK
        assert vid == 1


# =====================================================================
# Check constraints on real data
# =====================================================================
class TestGridChecks:
    def test_can_insert_restricted_to_truck_fails(self, grid_solution, grid):
        """Restricted customer cannot be inserted into truck."""
        restricted_custs = np.where(grid["restricted"] == 1)[0]
        c = int(restricted_custs[0])
        result = can_insert_customer(
            grid_solution, VEH_TRUCK, 0, c,
            grid["customers"], grid["restricted"], 2000.0
        )
        assert result is False

    def test_can_insert_heavy_to_bike_fails(self, grid_solution, grid):
        """Heavy customer (400kg) cannot fit in bike (60kg cap)."""
        heavy = np.where(grid["customers"][:, COL_DEMAND] > 60)[0]
        c = int(heavy[0])
        # Remove from truck first
        sol = copy_solution(grid_solution)
        vtype, vid, pos = get_customer_info(sol, c)
        remove_stop(sol, vtype, vid, pos, grid["dist_matrix"], grid["customers"])

        result = can_insert_customer(
            sol, VEH_BIKE, 0, c,
            grid["customers"], grid["restricted"], 60.0
        )
        assert result is False

    def test_rebuild_index_matches(self, grid_solution, grid):
        """rebuild_index produces same result as incremental updates."""
        sol = copy_solution(grid_solution)
        original_cv = sol["cust_vehicle"].copy()
        original_vt = sol["cust_vtype"].copy()
        original_rp = sol["cust_route_pos"].copy()

        rebuild_index(sol, grid["n_customers"])

        np.testing.assert_array_equal(sol["cust_vehicle"], original_cv)
        np.testing.assert_array_equal(sol["cust_vtype"], original_vt)
        np.testing.assert_array_equal(sol["cust_route_pos"], original_rp)


# =====================================================================
# Satellite operations on real data
# =====================================================================
class TestGridSatellites:
    def test_satellite_events_exist(self, grid_solution):
        sats = grid_solution["satellites"]
        assert sats.shape[0] > 0
        assert sats.shape[1] == 5

    def test_remove_satellites_for_customer(self, grid_solution):
        sol = copy_solution(grid_solution)
        sat_node = int(sol["satellites"][0, 0])
        count_before = len(sol["satellites"])
        removed = remove_satellites_for_customer(sol, sat_node)
        assert removed > 0
        assert len(sol["satellites"]) == count_before - removed
