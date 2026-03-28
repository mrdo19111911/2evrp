"""Tests for Pareto archive (src/alns/pareto.py)."""
import numpy as np
import pytest

from src.alns.pareto import (
    create_pareto_archive,
    update_pareto_archive,
    get_pareto_front,
)


def _dummy_sol(tag=0):
    """Create a minimal dummy solution dict for archive storage."""
    return {"tag": tag, "truck_stops": np.array([[tag]], dtype=np.int32)}


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
# update_pareto_archive
# ---------------------------------------------------------------------------

class TestUpdateParetoArchive:

    def test_first_solution_always_added(self):
        """Empty archive -> any solution is non-dominated -> added."""
        archive = create_pareto_archive()
        added = update_pareto_archive(archive, 100.0, 50.0, _dummy_sol(1))
        assert added is True
        assert len(archive) == 1

    def test_dominated_solution_not_added(self):
        """New solution dominated by existing -> not added."""
        archive = create_pareto_archive()
        update_pareto_archive(archive, 100.0, 50.0, _dummy_sol(1))

        # New: cost=150, makespan=60 -- worse in BOTH objectives
        added = update_pareto_archive(archive, 150.0, 60.0, _dummy_sol(2))
        assert added is False
        assert len(archive) == 1

    def test_dominating_solution_removes_old(self):
        """New solution dominates existing -> existing removed, new added."""
        archive = create_pareto_archive()
        update_pareto_archive(archive, 100.0, 50.0, _dummy_sol(1))

        # New: cost=80, makespan=40 -- better in BOTH objectives
        added = update_pareto_archive(archive, 80.0, 40.0, _dummy_sol(2))
        assert added is True
        assert len(archive) == 1
        # The remaining entry should be the dominating one
        assert archive[0][0] == 80.0
        assert archive[0][1] == 40.0

    def test_non_dominated_trade_off_both_kept(self):
        """Non-dominated (trade-off) -> both solutions kept."""
        archive = create_pareto_archive()
        update_pareto_archive(archive, 100.0, 50.0, _dummy_sol(1))

        # New: cost=80 (better), makespan=70 (worse) -- trade-off
        added = update_pareto_archive(archive, 80.0, 70.0, _dummy_sol(2))
        assert added is True
        assert len(archive) == 2

    def test_equal_objectives_not_dominated(self):
        """Solution with equal cost and makespan -> is it added or not?

        Per spec: dominated means (c <= new_cost and m <= new_makespan)
        and (c < new_cost OR m < new_makespan). Equal is NOT dominated.
        So equal solution should be added.
        """
        archive = create_pareto_archive()
        update_pareto_archive(archive, 100.0, 50.0, _dummy_sol(1))
        added = update_pareto_archive(archive, 100.0, 50.0, _dummy_sol(2))
        assert added is True

    def test_multiple_dominated_removed(self):
        """New solution dominates multiple existing entries -> all removed."""
        archive = create_pareto_archive()
        update_pareto_archive(archive, 100.0, 50.0, _dummy_sol(1))
        update_pareto_archive(archive, 90.0, 60.0, _dummy_sol(2))  # trade-off
        update_pareto_archive(archive, 110.0, 45.0, _dummy_sol(3))  # trade-off
        assert len(archive) == 3

        # New dominates entries 1 and 2 (cost=80 < 100,90; makespan=40 < 50,60)
        # Entry 3 (110, 45): cost 80 < 110 and makespan 40 < 45 -> also dominated
        added = update_pareto_archive(archive, 80.0, 40.0, _dummy_sol(4))
        assert added is True
        assert len(archive) == 1
        assert archive[0][0] == 80.0

    def test_archive_respects_max_size(self):
        """Archive pruned when exceeding max_size."""
        archive = create_pareto_archive(max_size=3)
        # Add 4 non-dominated solutions (trade-offs)
        update_pareto_archive(archive, 100.0, 10.0, _dummy_sol(1), max_size=3)
        update_pareto_archive(archive, 80.0, 30.0, _dummy_sol(2), max_size=3)
        update_pareto_archive(archive, 60.0, 50.0, _dummy_sol(3), max_size=3)
        update_pareto_archive(archive, 40.0, 70.0, _dummy_sol(4), max_size=3)

        assert len(archive) <= 3


# ---------------------------------------------------------------------------
# get_pareto_front
# ---------------------------------------------------------------------------

class TestGetParetoFront:

    def test_sorted_by_cost(self):
        """Returns (costs, makespans) sorted by cost ascending."""
        archive = create_pareto_archive()
        update_pareto_archive(archive, 100.0, 10.0, _dummy_sol(1))
        update_pareto_archive(archive, 50.0, 80.0, _dummy_sol(2))
        update_pareto_archive(archive, 75.0, 40.0, _dummy_sol(3))

        costs, makespans = get_pareto_front(archive)

        assert len(costs) == 3
        assert len(makespans) == 3
        # Sorted by cost ascending
        np.testing.assert_array_equal(costs, np.array([50.0, 75.0, 100.0]))
        np.testing.assert_array_equal(makespans, np.array([80.0, 40.0, 10.0]))

    def test_empty_archive(self):
        """Empty archive -> empty arrays."""
        archive = create_pareto_archive()
        costs, makespans = get_pareto_front(archive)
        assert len(costs) == 0
        assert len(makespans) == 0
        assert costs.dtype == np.float64
        assert makespans.dtype == np.float64

    def test_single_entry(self):
        """Single entry returns 1-element arrays."""
        archive = create_pareto_archive()
        update_pareto_archive(archive, 42.0, 17.0, _dummy_sol(1))

        costs, makespans = get_pareto_front(archive)
        np.testing.assert_array_equal(costs, np.array([42.0]))
        np.testing.assert_array_equal(makespans, np.array([17.0]))
