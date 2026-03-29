"""Tests for server/solution_serializer.py."""
import numpy as np
import pytest

from src.data.io import load_instance
from src.data.distance import compute_dist_matrix
from src.init.builder import build_initial_solution
from src.data.constants import EV_SIZE
from server.solution_serializer import serialize_solution, serialize_eval


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def grid_solution():
    """Load grid_20x20, build initial solution, return (sol, customers, depot)."""
    customers, depot, vehicles = load_instance("data/grid_20x20.json")
    dist_matrix = compute_dist_matrix(depot, customers)
    n_trucks, n_bikes = 3, 6
    sol = build_initial_solution(customers, depot, vehicles,
                                 dist_matrix, n_trucks, n_bikes, seed=42)
    return sol, customers, depot


# ---------------------------------------------------------------------------
# serialize_solution
# ---------------------------------------------------------------------------
def test_serialize_solution_has_required_keys(grid_solution):
    sol, customers, depot = grid_solution
    result = serialize_solution(sol, customers, depot)
    assert "truck_routes" in result
    assert "bike_routes" in result
    assert "satellites" in result
    assert "depot" in result
    assert "summary" in result
    assert "x" in result["depot"]
    assert "y" in result["depot"]
    summary = result["summary"]
    assert "n_trucks_used" in summary
    assert "n_bikes_used" in summary
    assert "n_customers" in summary
    assert "n_satellites" in summary


def test_truck_route_has_stops_with_ids(grid_solution):
    sol, customers, depot = grid_solution
    result = serialize_solution(sol, customers, depot)
    all_routes = result["truck_routes"] + result["bike_routes"]
    assert len(all_routes) > 0, "Expected at least one route"
    for route in all_routes:
        assert "vehicle_id" in route
        assert "stops" in route
        assert "load_g" in route
        assert "distance_m" in route
        for stop in route["stops"]:
            assert "customer_id" in stop
            assert "action" in stop
            assert "x" in stop
            assert "y" in stop
            assert "demand_g" in stop


# ---------------------------------------------------------------------------
# serialize_eval
# ---------------------------------------------------------------------------
def test_serialize_eval_returns_dict():
    ev = np.zeros(EV_SIZE, dtype=np.int64)
    ev[0] = 999000   # fitness
    ev[1] = 500000   # cost
    ev[2] = 100000   # sync_cost
    ev[3] = 3600     # makespan
    ev[4] = 50000    # total_penalty
    ev[5] = 200      # total_wait
    ev[6] = 1        # feasible
    result = serialize_eval(ev)
    assert result["fitness"] == 999000
    assert result["cost"] == 500000
    assert result["sync_cost"] == 100000
    assert result["makespan"] == 3600
    assert result["total_penalty"] == 50000
    assert result["total_wait"] == 200
    assert result["feasible"] is True
