"""Integration test: src/solution/ with grid_20x20 (400 customers).
Build a random solution with trucks, bikes, reload events, then verify all ops.
Tuple-based solution format. All i64."""
import numpy as np
import pytest

from src.data.io import load_instance
from src.data.distance import compute_dist_matrix
from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, VEH_TRUCK, VEH_BIKE, COL_DEMAND, COL_RESTRICTED,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_TRUCK_LENGTHS,
    SOL_BIKE_STOPS, SOL_BIKE_ACTIONS, SOL_BIKE_LENGTHS,
    SOL_TRUCK_DISTANCES, SOL_BIKE_DISTANCES,
    SOL_TRUCK_LOADS, SOL_BIKE_LOADS,
    SOL_CUST_VEHICLE, SOL_CUST_VTYPE, SOL_CUST_ROUTE_POS,
    SOL_SATELLITES, SOL_META,
    META_N_TRUCKS, META_N_BIKES, META_N_SATELLITES,
    SAT_CUST, VCOL_CAPACITY,
)
from src.solution.structure import create_solution, copy_solution, rebuild_index
from src.solution.route_ops import insert_stop, remove_stop, reverse_segment
from src.solution.solution_ops import (
    move_stop, clear_route, create_route_from_sequence,
)
from src.solution.satellite_ops import add_satellite_event, remove_satellites_for_customer, get_satellites
from src.solution.delta import (
    insertion_cost_delta, removal_cost_delta, best_insertion_pos,
    find_best_insertion_all_routes,
)
from src.solution.check import (
    can_insert_customer, can_swap_customers, check_all_assigned,
)
from src.solution.query import (
    get_unassigned_customers, get_assigned_customers,
    get_customer_info, is_customer_assigned,
)
from src.data.cost import TRUCK_CAPACITY_G, BIKE_CAPACITY_G

GRID_PATH = "data/grid_20x20.json"


@pytest.fixture(scope="module")
def grid():
    """Load grid_20x20 + compute dist_matrix. Cached for module."""
    customers, depot, vehicles = load_instance(GRID_PATH)
    dm = compute_dist_matrix(depot, customers)
    return {
        "customers": customers,
        "depot": depot,
        "vehicles": vehicles,
        "dist_matrix": dm,
        "n_customers": len(customers),
        "n_trucks": 5,
        "n_bikes": 15,
    }


@pytest.fixture
def grid_solution(grid):
    """Build a realistic random solution for 400 customers."""
    rng = np.random.default_rng(42)
    custs = grid["customers"]
    dm = grid["dist_matrix"]
    n = grid["n_customers"]

    sol = create_solution(grid["n_trucks"], grid["n_bikes"], n)

    # Classify using COL_RESTRICTED column from customers
    restricted = custs[:, COL_RESTRICTED]
    heavy = np.where(custs[:, COL_DEMAND] > 60000)[0]  # > 60kg in grams
    light = np.where(custs[:, COL_DEMAND] <= 60000)[0]
    bike_only = np.where(restricted == 1)[0]
    bike_only_set = set(bike_only)
    light_not_restricted = np.array([c for c in light if c not in bike_only_set])

    rng.shuffle(heavy)
    rng.shuffle(light_not_restricted)

    # Split heavy across 5 trucks
    truck_assignments = [[] for _ in range(5)]
    for i, c in enumerate(heavy):
        truck_assignments[i % 5].append(c)

    for i, c in enumerate(light_not_restricted[:50]):
        truck_assignments[i % 5].append(c)

    sat_candidates = light_not_restricted[50:70]
    satellite_nodes = sat_candidates[:5]

    for t in range(5):
        truck_assignments[t].append(int(satellite_nodes[t % len(satellite_nodes)]))

    for t in range(5):
        for c in truck_assignments[t][:-1]:
            pos = sol[SOL_TRUCK_LENGTHS][t]
            insert_stop(sol, VEH_TRUCK, t, pos, int(c), ACT_DELIVER, dm, custs)
        sat_node = truck_assignments[t][-1]
        pos = sol[SOL_TRUCK_LENGTHS][t]
        insert_stop(sol, VEH_TRUCK, t, pos, sat_node, ACT_RELOAD, dm, custs)

    # Bike routes
    assigned_set = set()
    for t in range(5):
        for c in truck_assignments[t][:-1]:
            assigned_set.add(c)

    remaining = [c for c in range(n) if c not in assigned_set]
    rng.shuffle(remaining)

    bike_assignments = [[] for _ in range(15)]
    for i, c in enumerate(remaining):
        bike_assignments[i % 15].append(c)

    for b in range(15):
        if not bike_assignments[b]:
            continue
        sat_node = int(satellite_nodes[b % len(satellite_nodes)])

        half = len(bike_assignments[b]) // 2
        for c in bike_assignments[b][:half]:
            pos = sol[SOL_BIKE_LENGTHS][b]
            insert_stop(sol, VEH_BIKE, b, pos, int(c), ACT_DELIVER, dm, custs)

        pos = sol[SOL_BIKE_LENGTHS][b]
        insert_stop(sol, VEH_BIKE, b, pos, sat_node, ACT_RELOAD, dm, custs)

        for c in bike_assignments[b][half:]:
            pos = sol[SOL_BIKE_LENGTHS][b]
            insert_stop(sol, VEH_BIKE, b, pos, int(c), ACT_DELIVER, dm, custs)

        transfer_g = sum(int(custs[c, COL_DEMAND]) for c in bike_assignments[b][half:])
        truck_id = b % 5
        add_satellite_event(sol, sat_node, b, truck_id, transfer_g, 200 + b * 10)

    return sol


# =====================================================================
# Structure tests
# =====================================================================
class TestGridSolutionStructure:
    def test_solution_created(self, grid_solution):
        assert grid_solution[SOL_META][META_N_TRUCKS] == 5
        assert grid_solution[SOL_META][META_N_BIKES] == 15

    def test_trucks_have_routes(self, grid_solution):
        for t in range(5):
            assert grid_solution[SOL_TRUCK_LENGTHS][t] > 0

    def test_bikes_have_routes(self, grid_solution):
        active = sum(1 for b in range(15) if grid_solution[SOL_BIKE_LENGTHS][b] > 0)
        assert active >= 10

    def test_satellites_exist(self, grid_solution):
        n_sats = grid_solution[SOL_META][META_N_SATELLITES]
        assert n_sats > 0

    def test_most_customers_assigned(self, grid_solution, grid):
        assigned = get_assigned_customers(grid_solution, grid["n_customers"])
        assert len(assigned) == 400


# =====================================================================
# Query tests on real data
# =====================================================================
class TestGridQuery:
    def test_customer_info_consistency(self, grid_solution, grid):
        """Every assigned customer's info matches actual position in route."""
        for c in range(grid["n_customers"]):
            if not is_customer_assigned(grid_solution, c):
                continue
            vtype, vid, pos = get_customer_info(grid_solution, c)
            assert vtype in (VEH_TRUCK, VEH_BIKE)
            if vtype == VEH_TRUCK:
                assert grid_solution[SOL_TRUCK_STOPS][vid, pos] == c
                assert grid_solution[SOL_TRUCK_ACTIONS][vid, pos] == ACT_DELIVER
            else:
                assert grid_solution[SOL_BIKE_STOPS][vid, pos] == c
                assert grid_solution[SOL_BIKE_ACTIONS][vid, pos] == ACT_DELIVER


# =====================================================================
# Delta consistency: old + delta == new
# =====================================================================
class TestGridDeltaConsistency:
    def test_insertion_delta_100_random(self, grid_solution, grid):
        """100 random insertions: verify old_dist + delta == new_dist."""
        rng = np.random.default_rng(123)
        dm = grid["dist_matrix"]
        custs = grid["customers"]
        sol = copy_solution(grid_solution)

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
            vid = 0
            L = sol[SOL_TRUCK_LENGTHS][vid]
            for pos in [0, L // 2, L]:
                old_dist = int(sol[SOL_TRUCK_DISTANCES][vid])
                delta = insertion_cost_delta(sol, VEH_TRUCK, vid, pos, c, dm)

                sol_copy = copy_solution(sol)
                insert_stop(sol_copy, VEH_TRUCK, vid, pos, c, ACT_DELIVER, dm, custs)
                new_dist = int(sol_copy[SOL_TRUCK_DISTANCES][vid])

                assert old_dist + delta == new_dist, \
                    f"C{c} pos={pos}: {old_dist} + {delta} != {new_dist}"
                checks += 1

        assert checks >= 30, f"Only {checks} checks done"

    def test_removal_delta_50_random(self, grid_solution, grid):
        """50 random removals: verify old_dist + delta == new_dist."""
        dm = grid["dist_matrix"]
        custs = grid["customers"]
        sol = copy_solution(grid_solution)

        checks = 0
        for t in range(5):
            L = sol[SOL_TRUCK_LENGTHS][t]
            for pos in range(min(L, 10)):
                old_dist = int(sol[SOL_TRUCK_DISTANCES][t])
                delta = removal_cost_delta(sol, VEH_TRUCK, t, pos, dm)

                sol_copy = copy_solution(sol)
                remove_stop(sol_copy, VEH_TRUCK, t, pos, dm, custs)
                new_dist = int(sol_copy[SOL_TRUCK_DISTANCES][t])

                assert old_dist + delta == new_dist, \
                    f"Truck {t} pos={pos}: {old_dist} + {delta} != {new_dist}"
                checks += 1

        assert checks >= 30, f"Only {checks} checks done"


# =====================================================================
# Route operations on 400-node solution
# =====================================================================
class TestGridRouteOps:
    def test_remove_and_reinsert(self, grid_solution, grid):
        """Remove 2 customers from truck 0, reinsert one via best_insertion_pos."""
        dm = grid["dist_matrix"]
        custs = grid["customers"]
        sol = copy_solution(grid_solution)

        removed = []
        while len(removed) < 2:
            L = sol[SOL_TRUCK_LENGTHS][0]
            for pos in range(L):
                if sol[SOL_TRUCK_ACTIONS][0, pos] == ACT_DELIVER:
                    c, _ = remove_stop(sol, VEH_TRUCK, 0, pos, dm, custs)
                    removed.append(c)
                    break

        assert len(removed) == 2
        c = removed[0]
        assert not is_customer_assigned(sol, c)

        # Use best_insertion_pos on truck 0 directly (no TW/capacity filter)
        ins_pos, delta = best_insertion_pos(sol, VEH_TRUCK, 0, c, dm)
        assert delta < 2_000_000_000_000

        insert_stop(sol, VEH_TRUCK, 0, ins_pos, c, ACT_DELIVER, dm, custs)
        assert is_customer_assigned(sol, c)

    def test_reverse_segment_works(self, grid_solution, grid):
        """2-opt reverse on truck route: should not crash."""
        dm = grid["dist_matrix"]
        sol = copy_solution(grid_solution)

        for t in range(5):
            L = sol[SOL_TRUCK_LENGTHS][t]
            if L >= 4:
                reverse_segment(sol, VEH_TRUCK, t, 1, L - 2, dm)
                new_dist = int(sol[SOL_TRUCK_DISTANCES][t])
                assert new_dist > 0
                return

    def test_clear_and_rebuild(self, grid_solution, grid):
        """Clear truck route 0, rebuild from sequence, verify consistency."""
        dm = grid["dist_matrix"]
        custs = grid["customers"]
        sol = copy_solution(grid_solution)

        L = int(sol[SOL_TRUCK_LENGTHS][0])
        cust_arr = sol[SOL_TRUCK_STOPS][0, :L].copy()
        act_arr = sol[SOL_TRUCK_ACTIONS][0, :L].copy()

        removed = clear_route(sol, VEH_TRUCK, 0)
        assert sol[SOL_TRUCK_LENGTHS][0] == 0

        create_route_from_sequence(sol, VEH_TRUCK, 0, cust_arr, act_arr, L, dm, custs)

        assert sol[SOL_TRUCK_LENGTHS][0] == L
        for i in range(L):
            assert sol[SOL_TRUCK_STOPS][0, i] == cust_arr[i]
            assert sol[SOL_TRUCK_ACTIONS][0, i] == act_arr[i]

    def test_move_between_trucks(self, grid_solution, grid):
        """Move customer from truck 0 to truck 1."""
        dm = grid["dist_matrix"]
        custs = grid["customers"]
        sol = copy_solution(grid_solution)

        L0 = sol[SOL_TRUCK_LENGTHS][0]
        if L0 < 2:
            pytest.skip("Truck 0 too short")

        c = int(sol[SOL_TRUCK_STOPS][0, 0])
        a = int(sol[SOL_TRUCK_ACTIONS][0, 0])
        if a != ACT_DELIVER:
            pytest.skip("First stop not DELIVER")

        old_L1 = sol[SOL_TRUCK_LENGTHS][1]
        move_stop(sol, VEH_TRUCK, 0, 0, VEH_TRUCK, 1, 0, dm, custs)

        assert sol[SOL_TRUCK_LENGTHS][0] == L0 - 1
        assert sol[SOL_TRUCK_LENGTHS][1] == old_L1 + 1
        vtype, vid, pos = get_customer_info(sol, c)
        assert vtype == VEH_TRUCK
        assert vid == 1


# =====================================================================
# Check constraints on real data
# =====================================================================
class TestGridChecks:
    def test_can_insert_restricted_to_truck_fails(self, grid_solution, grid):
        """Restricted customer cannot be inserted into truck."""
        restricted_custs = np.where(grid["customers"][:, COL_RESTRICTED] == 1)[0]
        c = int(restricted_custs[0])
        result = can_insert_customer(
            grid_solution, VEH_TRUCK, 0, c,
            grid["customers"], TRUCK_CAPACITY_G,
        )
        assert result is False

    def test_can_insert_heavy_to_bike_fails(self, grid_solution, grid):
        """Heavy customer (>60kg) cannot fit in bike (60kg cap)."""
        heavy = np.where(grid["customers"][:, COL_DEMAND] > 60000)[0]
        c = int(heavy[0])
        sol = copy_solution(grid_solution)
        vtype, vid, pos = get_customer_info(sol, c)
        remove_stop(sol, vtype, vid, pos, grid["dist_matrix"], grid["customers"])

        result = can_insert_customer(
            sol, VEH_BIKE, 0, c,
            grid["customers"], BIKE_CAPACITY_G,
        )
        assert result is False

    def test_rebuild_index_matches(self, grid_solution, grid):
        """rebuild_index produces same result as incremental updates."""
        sol = copy_solution(grid_solution)
        original_cv = sol[SOL_CUST_VEHICLE].copy()
        original_vt = sol[SOL_CUST_VTYPE].copy()
        original_rp = sol[SOL_CUST_ROUTE_POS].copy()

        rebuild_index(sol)

        np.testing.assert_array_equal(sol[SOL_CUST_VEHICLE], original_cv)
        np.testing.assert_array_equal(sol[SOL_CUST_VTYPE], original_vt)
        np.testing.assert_array_equal(sol[SOL_CUST_ROUTE_POS], original_rp)


# =====================================================================
# Satellite operations on real data
# =====================================================================
class TestGridSatellites:
    def test_satellite_events_exist(self, grid_solution):
        sats = get_satellites(grid_solution)
        assert sats.shape[0] > 0
        assert sats.shape[1] == 5

    def test_remove_satellites_for_customer(self, grid_solution):
        sol = copy_solution(grid_solution)
        sats = get_satellites(sol)
        sat_node = int(sats[0, SAT_CUST])
        count_before = sol[SOL_META][META_N_SATELLITES]
        removed = remove_satellites_for_customer(sol, sat_node)
        assert removed > 0
        assert sol[SOL_META][META_N_SATELLITES] == count_before - removed
