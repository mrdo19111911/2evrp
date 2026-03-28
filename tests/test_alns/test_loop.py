"""Integration tests for the ALNS main loop (src/alns/loop.py)."""
import numpy as np
import pytest

from src.alns.loop import solve


# ---------------------------------------------------------------------------
# solve — integration tests
# ---------------------------------------------------------------------------

class TestSolve:

    def test_returns_expected_tuple(self, tiny_instance):
        """solve returns (best_sol, best_fitness, archive, log)."""
        result = solve(
            tiny_instance,
            config_overrides={"max_iterations": 100},
            seed=42,
        )

        assert isinstance(result, tuple)
        assert len(result) == 4

        best_sol, best_fitness, archive, log = result
        assert isinstance(best_sol, dict)
        assert isinstance(best_fitness, float)
        assert isinstance(archive, list)
        assert isinstance(log, dict)

    def test_best_fitness_is_finite(self, tiny_instance):
        """best_fitness is a finite number (not NaN or Inf)."""
        _, best_fitness, _, _ = solve(
            tiny_instance,
            config_overrides={"max_iterations": 100},
            seed=42,
        )

        assert np.isfinite(best_fitness), f"best_fitness={best_fitness} is not finite"

    def test_log_has_entries(self, tiny_instance):
        """Log should have at least 1 entry after running."""
        _, _, _, log = solve(
            tiny_instance,
            config_overrides={"max_iterations": 50},
            seed=42,
        )

        assert "iterations" in log or "fitness" in log
        # At least one of the log fields should have entries
        log_lengths = [len(v) for v in log.values() if isinstance(v, list)]
        assert any(l > 0 for l in log_lengths), "Log should have at least 1 entry"

    def test_deterministic_with_same_seed(self, tiny_instance):
        """Same seed -> identical best_fitness."""
        kwargs = dict(
            data=tiny_instance,
            config_overrides={"max_iterations": 100},
            seed=42,
        )

        _, fitness1, _, _ = solve(**kwargs)
        _, fitness2, _, _ = solve(**kwargs)

        assert fitness1 == pytest.approx(fitness2), (
            f"Same seed should give same fitness: {fitness1} vs {fitness2}"
        )

    def test_different_seed_may_differ(self, tiny_instance):
        """Different seeds can (but don't have to) produce different results."""
        base_kwargs = dict(
            data=tiny_instance,
            config_overrides={"max_iterations": 100},
        )

        _, fitness1, _, _ = solve(**base_kwargs, seed=42)
        _, fitness2, _, _ = solve(**base_kwargs, seed=999)

        # Not asserting they differ -- just that both are finite
        assert np.isfinite(fitness1)
        assert np.isfinite(fitness2)

    def test_callback_is_called(self, tiny_instance):
        """If callback is provided, it is called at least once."""
        call_count = [0]

        def my_callback(iteration, sol, eval_result, log):
            call_count[0] += 1

        solve(
            tiny_instance,
            config_overrides={"max_iterations": 20},
            seed=42,
            callback=my_callback,
        )

        assert call_count[0] > 0, "Callback should have been called at least once"

    def test_best_fitness_non_increasing_in_log(self, tiny_instance):
        """best_fitness entries in log should be monotone non-increasing."""
        _, _, _, log = solve(
            tiny_instance,
            config_overrides={"max_iterations": 100},
            seed=42,
        )

        if "best_fitness" in log and len(log["best_fitness"]) > 1:
            bf = np.array(log["best_fitness"])
            # Each entry should be <= the previous (non-increasing)
            for i in range(1, len(bf)):
                assert bf[i] <= bf[i - 1] + 1e-9, (
                    f"best_fitness increased at iteration {i}: "
                    f"{bf[i-1]:.6f} -> {bf[i]:.6f}"
                )

    def test_pareto_archive_non_empty(self, tiny_instance):
        """After running, Pareto archive should have at least 1 entry."""
        _, _, archive, _ = solve(
            tiny_instance,
            config_overrides={"max_iterations": 100},
            seed=42,
        )

        assert len(archive) >= 1, "Pareto archive should contain at least 1 solution"

    def test_medium_instance_runs(self, medium_data):
        """Smoke test: medium instance (20 customers) completes without error."""
        result = solve(
            medium_data,
            config_overrides={"max_iterations": 50},
            seed=42,
        )

        best_sol, best_fitness, archive, log = result
        assert np.isfinite(best_fitness)
