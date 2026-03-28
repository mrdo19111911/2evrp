"""Tests for Pareto archive (src/alns/pareto.py). Values are i64."""
import numpy as np
import pytest

from src.alns.pareto import (
    create_pareto_archive,
    update_pareto_archive,
    get_pareto_front,
)
from src.solution.structure import create_solution


def _dummy_sol(tag=0):
    return create_solution(1, 1, max(1, tag))


# ---------------------------------------------------------------------------
# create_pareto_archive
# ---------------------------------------------------------------------------

class TestCreateParetoArchive:

    def test_returns_empty_list(self):
        archive = create_pareto_archive(max_size=50)
        assert isinstance(archive, list)
        assert len(archive) == 0

    def test_custom_max_size(self):
        archive = create_pareto_archive(max_size=10)
        assert isinstance(archive, list)
        assert len(archive) == 0


# ---------------------------------------------------------------------------
# update_pareto_archive — cost/makespan are i64
# ---------------------------------------------------------------------------

class TestUpdateParetoArchive:

    def test_first_solution_always_added(self):
        archive = create_pareto_archive()
        added = update_pareto_archive(archive, 100000, 50000, _dummy_sol(1))
        assert added is True
        assert len(archive) == 1

    def test_dominated_solution_not_added(self):
        archive = create_pareto_archive()
        update_pareto_archive(archive, 100000, 50000, _dummy_sol(1))
        added = update_pareto_archive(archive, 150000, 60000, _dummy_sol(2))
        assert added is False
        assert len(archive) == 1

    def test_dominating_solution_removes_old(self):
        archive = create_pareto_archive()
        update_pareto_archive(archive, 100000, 50000, _dummy_sol(1))
        added = update_pareto_archive(archive, 80000, 40000, _dummy_sol(2))
        assert added is True
        assert len(archive) == 1
        assert archive[0][0] == 80000
        assert archive[0][1] == 40000

    def test_non_dominated_trade_off_both_kept(self):
        archive = create_pareto_archive()
        update_pareto_archive(archive, 100000, 50000, _dummy_sol(1))
        added = update_pareto_archive(archive, 80000, 70000, _dummy_sol(2))
        assert added is True
        assert len(archive) == 2

    def test_equal_objectives_not_dominated(self):
        archive = create_pareto_archive()
        update_pareto_archive(archive, 100000, 50000, _dummy_sol(1))
        added = update_pareto_archive(archive, 100000, 50000, _dummy_sol(2))
        assert added is True

    def test_multiple_dominated_removed(self):
        archive = create_pareto_archive()
        update_pareto_archive(archive, 100000, 50000, _dummy_sol(1))
        update_pareto_archive(archive, 90000, 60000, _dummy_sol(2))
        update_pareto_archive(archive, 110000, 45000, _dummy_sol(3))
        assert len(archive) == 3
        added = update_pareto_archive(archive, 80000, 40000, _dummy_sol(4))
        assert added is True
        assert len(archive) == 1
        assert archive[0][0] == 80000

    def test_archive_respects_max_size(self):
        archive = create_pareto_archive(max_size=3)
        update_pareto_archive(archive, 100000, 10000, _dummy_sol(1), max_size=3)
        update_pareto_archive(archive, 80000, 30000, _dummy_sol(2), max_size=3)
        update_pareto_archive(archive, 60000, 50000, _dummy_sol(3), max_size=3)
        update_pareto_archive(archive, 40000, 70000, _dummy_sol(4), max_size=3)
        assert len(archive) <= 3


# ---------------------------------------------------------------------------
# get_pareto_front — returns i64 arrays
# ---------------------------------------------------------------------------

class TestGetParetoFront:

    def test_sorted_by_cost(self):
        archive = create_pareto_archive()
        update_pareto_archive(archive, 100000, 10000, _dummy_sol(1))
        update_pareto_archive(archive, 50000, 80000, _dummy_sol(2))
        update_pareto_archive(archive, 75000, 40000, _dummy_sol(3))
        costs, makespans = get_pareto_front(archive)
        assert len(costs) == 3
        np.testing.assert_array_equal(costs, np.array([50000, 75000, 100000], dtype=np.int64))
        np.testing.assert_array_equal(makespans, np.array([80000, 40000, 10000], dtype=np.int64))

    def test_empty_archive(self):
        archive = create_pareto_archive()
        costs, makespans = get_pareto_front(archive)
        assert len(costs) == 0
        assert costs.dtype == np.int64

    def test_single_entry(self):
        archive = create_pareto_archive()
        update_pareto_archive(archive, 42000, 17000, _dummy_sol(1))
        costs, makespans = get_pareto_front(archive)
        np.testing.assert_array_equal(costs, np.array([42000], dtype=np.int64))
        np.testing.assert_array_equal(makespans, np.array([17000], dtype=np.int64))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestParetoEdgeCases:

    def test_prune_with_two_entries_at_boundary(self):
        archive = create_pareto_archive(max_size=2)
        update_pareto_archive(archive, 100000, 10000, _dummy_sol(1), max_size=2)
        update_pareto_archive(archive, 50000, 80000, _dummy_sol(2), max_size=2)
        assert len(archive) == 2
        added = update_pareto_archive(archive, 75000, 45000, _dummy_sol(3), max_size=2)
        assert len(archive) <= 2
        assert added is True

    def test_crowding_distance_zero_makespan_range(self):
        archive = create_pareto_archive(max_size=2)
        update_pareto_archive(archive, 100000, 50000, _dummy_sol(1), max_size=2)
        update_pareto_archive(archive, 80000, 50000, _dummy_sol(2), max_size=2)
        update_pareto_archive(archive, 60000, 50000, _dummy_sol(3), max_size=2)
        assert len(archive) <= 2

    def test_crowding_distance_zero_cost_range(self):
        archive = create_pareto_archive(max_size=2)
        update_pareto_archive(archive, 50000, 100000, _dummy_sol(1), max_size=2)
        update_pareto_archive(archive, 50000, 80000, _dummy_sol(2), max_size=2)
        update_pareto_archive(archive, 50000, 60000, _dummy_sol(3), max_size=2)
        assert len(archive) <= 2
