"""Integration tests for the ALNS main loop (src/alns/loop.py). i64 interface."""
import numpy as np
import pytest

from src.alns.loop import solve
from src.data.constants import CFG_MAX_ITERATIONS


# ---------------------------------------------------------------------------
# solve — no restricted param. Signature:
# solve(customers, depot, vehicles, dist_matrix, n_trucks, n_bikes, ...)
# ---------------------------------------------------------------------------

class TestSolve:

    def test_returns_expected_tuple(self, tiny_instance, tiny_dist_matrix):
        data = tiny_instance
        result = solve(
            data["customers"], data["depot"], data["vehicles"],
            tiny_dist_matrix,
            data["n_trucks"], data["n_bikes"],
            config_overrides={CFG_MAX_ITERATIONS: 50},
            seed=42,
        )
        assert isinstance(result, tuple)
        assert len(result) == 4
        best_sol, best_fitness, archive, log = result
        assert isinstance(best_sol, tuple)
        assert isinstance(best_fitness, (int, float, np.integer, np.floating))
        assert isinstance(archive, list)
        assert isinstance(log, dict)

    def test_best_fitness_is_finite(self, tiny_instance, tiny_dist_matrix):
        data = tiny_instance
        _, best_fitness, _, _ = solve(
            data["customers"], data["depot"], data["vehicles"],
            tiny_dist_matrix,
            data["n_trucks"], data["n_bikes"],
            config_overrides={CFG_MAX_ITERATIONS: 50},
            seed=42,
        )
        assert np.isfinite(best_fitness)

    def test_log_has_entries(self, tiny_instance, tiny_dist_matrix):
        data = tiny_instance
        _, _, _, log = solve(
            data["customers"], data["depot"], data["vehicles"],
            tiny_dist_matrix,
            data["n_trucks"], data["n_bikes"],
            config_overrides={CFG_MAX_ITERATIONS: 30},
            seed=42,
        )
        assert "iterations" in log or "fitness" in log
        log_lengths = [len(v) for v in log.values() if isinstance(v, list)]
        assert any(length > 0 for length in log_lengths)

    def test_deterministic_with_same_seed(self, tiny_instance, tiny_dist_matrix):
        data = tiny_instance
        kwargs = dict(
            customers=data["customers"], depot=data["depot"],
            vehicles=data["vehicles"], dist_matrix=tiny_dist_matrix,
            n_trucks=data["n_trucks"], n_bikes=data["n_bikes"],
            config_overrides={CFG_MAX_ITERATIONS: 50},
            seed=42,
        )
        _, fitness1, _, _ = solve(**kwargs)
        _, fitness2, _, _ = solve(**kwargs)
        assert fitness1 == pytest.approx(fitness2)

    def test_pareto_archive_non_empty(self, tiny_instance, tiny_dist_matrix):
        data = tiny_instance
        _, _, archive, _ = solve(
            data["customers"], data["depot"], data["vehicles"],
            tiny_dist_matrix,
            data["n_trucks"], data["n_bikes"],
            config_overrides={CFG_MAX_ITERATIONS: 50},
            seed=42,
        )
        assert len(archive) >= 1
