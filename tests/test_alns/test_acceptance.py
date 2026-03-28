"""Tests for SA acceptance criterion (src/alns/acceptance.py). i64 interface."""
import numpy as np
import pytest

from src.alns.acceptance import (
    sa_accept,
    cool_temperature,
    compute_initial_temperature,
)


# ---------------------------------------------------------------------------
# sa_accept — fitness is i64 VND, seed instead of rng
# ---------------------------------------------------------------------------

class TestSaAccept:

    def test_better_always_accepted(self):
        for seed in range(100):
            assert sa_accept(100000, 50000, 1.0, seed) is True
            assert sa_accept(100000, 99999, 0.001, seed + 1000) is True

    def test_equal_always_accepted(self):
        for seed in range(100):
            assert sa_accept(50000, 50000, 10.0, seed) is True

    def test_worse_high_temp_likely_accepted(self):
        accepts = sum(1 for s in range(1000) if sa_accept(100000, 101000, 1e9, s))
        rate = accepts / 1000
        assert rate > 0.95

    def test_worse_low_temp_likely_rejected(self):
        accepts = sum(1 for s in range(1000) if sa_accept(100000, 200000, 0.1, s))
        rate = accepts / 1000
        assert rate < 0.05

    def test_zero_temperature_rejects_worse(self):
        for seed in range(100):
            assert sa_accept(100000, 101000, 0.0, seed) is False
            assert sa_accept(100000, 101000, 1e-15, seed) is False

    def test_zero_temperature_accepts_better(self):
        assert sa_accept(100000, 50000, 0.0, 0) is True


# ---------------------------------------------------------------------------
# cool_temperature
# ---------------------------------------------------------------------------

class TestCoolTemperature:

    def test_geometric_cooling(self):
        new_temp = cool_temperature(100.0, 0.9997)
        np.testing.assert_allclose(new_temp, 99.97, atol=1e-6)

    def test_does_not_go_below_zero(self):
        temp = 0.02
        new_temp = cool_temperature(temp, 0.9997)
        assert new_temp > 0.0
        np.testing.assert_allclose(new_temp, 0.02 * 0.9997, atol=1e-10)

    def test_large_cooling_rate(self):
        new_temp = cool_temperature(100.0, 0.5)
        assert new_temp == pytest.approx(50.0, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_initial_temperature — no restricted param
# ---------------------------------------------------------------------------

class TestComputeInitialTemperature:

    def test_returns_positive_float(self, sol_5_assigned):
        """compute_initial_temperature returns a positive float."""
        from src.alns.config import make_config
        bundle = sol_5_assigned
        rng = np.random.default_rng(42)
        config = make_config(bundle["N"])

        T0 = compute_initial_temperature(
            bundle["sol"], bundle["customers"], bundle["vehicles"],
            bundle["dist_matrix"], config, rng,
            target_accept_rate=0.8, n_samples=100,
        )
        assert isinstance(T0, float)
        assert T0 > 0.0

    def test_deterministic_with_same_seed(self, sol_5_assigned):
        """Same seed -> same T0."""
        from src.alns.config import make_config
        from src.solution.structure import copy_solution
        bundle = sol_5_assigned
        config = make_config(bundle["N"])

        T1 = compute_initial_temperature(
            copy_solution(bundle["sol"]), bundle["customers"],
            bundle["vehicles"], bundle["dist_matrix"], config,
            np.random.default_rng(99),
        )
        T2 = compute_initial_temperature(
            copy_solution(bundle["sol"]), bundle["customers"],
            bundle["vehicles"], bundle["dist_matrix"], config,
            np.random.default_rng(99),
        )
        assert T1 == pytest.approx(T2)
