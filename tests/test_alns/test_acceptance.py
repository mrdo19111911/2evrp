"""Tests for SA acceptance criterion (src/alns/acceptance.py)."""
import numpy as np
import pytest

from src.alns.acceptance import (
    sa_accept,
    cool_temperature,
    compute_initial_temperature,
)


# ---------------------------------------------------------------------------
# sa_accept
# ---------------------------------------------------------------------------

class TestSaAccept:

    def test_better_always_accepted(self):
        """new_fitness < current_fitness -> always True, regardless of temperature."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            assert sa_accept(100.0, 50.0, 1.0, rng) is True
            assert sa_accept(100.0, 99.99, 0.001, rng) is True

    def test_equal_always_accepted(self):
        """new_fitness == current_fitness -> delta=0, prob=exp(0)=1 -> True."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            assert sa_accept(50.0, 50.0, 10.0, rng) is True

    def test_worse_high_temp_likely_accepted(self):
        """new > current, high temp -> most trials should accept."""
        rng = np.random.default_rng(42)
        accepts = 0
        n_trials = 1000
        for _ in range(n_trials):
            if sa_accept(100.0, 101.0, 1000.0, rng):
                accepts += 1
        # prob = exp(-1/1000) = exp(-0.001) ~ 0.999 -> almost always accept
        rate = accepts / n_trials
        assert rate > 0.95, f"Accept rate {rate:.3f} should be >0.95 at high temp"

    def test_worse_low_temp_likely_rejected(self):
        """new > current, low temp -> most trials should reject."""
        rng = np.random.default_rng(42)
        accepts = 0
        n_trials = 1000
        for _ in range(n_trials):
            if sa_accept(100.0, 200.0, 0.1, rng):
                accepts += 1
        # prob = exp(-100/0.1) = exp(-1000) ~ 0 -> almost never accept
        rate = accepts / n_trials
        assert rate < 0.05, f"Accept rate {rate:.3f} should be <0.05 at low temp"

    def test_zero_temperature_rejects_worse(self):
        """temp=0 (or near 0) -> always reject worse solutions."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            assert sa_accept(100.0, 101.0, 0.0, rng) is False
            assert sa_accept(100.0, 101.0, 1e-15, rng) is False

    def test_zero_temperature_accepts_better(self):
        """Even at temp=0, better solutions are accepted."""
        rng = np.random.default_rng(0)
        assert sa_accept(100.0, 50.0, 0.0, rng) is True


# ---------------------------------------------------------------------------
# cool_temperature
# ---------------------------------------------------------------------------

class TestCoolTemperature:

    def test_geometric_cooling(self):
        """100 * 0.9997 = 99.97."""
        new_temp = cool_temperature(100.0, 0.9997)
        np.testing.assert_allclose(new_temp, 99.97, atol=1e-6)

    def test_does_not_go_below_min(self):
        """Cooling cannot reduce temperature below sa_min_temp."""
        # After many coolings, temp should plateau at some minimum
        temp = 0.02
        new_temp = cool_temperature(temp, 0.9997)
        # 0.02 * 0.9997 = 0.019994 which is > 0.01 (default min)
        # So it should be 0.019994
        assert new_temp > 0.0
        np.testing.assert_allclose(new_temp, 0.02 * 0.9997, atol=1e-10)

    def test_large_cooling_rate(self):
        """Cooling rate of 0.5 halves the temperature."""
        new_temp = cool_temperature(100.0, 0.5)
        assert new_temp == pytest.approx(50.0, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_initial_temperature
# ---------------------------------------------------------------------------

class TestComputeInitialTemperature:

    def test_returns_positive_float(self, tiny_instance):
        """compute_initial_temperature returns a positive float."""
        from src.alns.config import DEFAULT_CONFIG
        from src.solution.structure import create_solution
        from tests.test_alns.conftest import _assign_customer

        data = tiny_instance
        n = data["n_customers"]
        sol = create_solution(data["n_vehicles"], n, n_sku=data["n_sku"])
        # Assign C0 to truck (heavy, 100kg)
        _assign_customer(sol, 0, 0, 0, 100.0,
                         data["orders"], data["vehicles"], data["dist_matrix"])

        rng = np.random.default_rng(42)
        config = DEFAULT_CONFIG.copy()

        T0 = compute_initial_temperature(
            sol, data, config, rng,
            target_accept_rate=0.8, n_samples=100,
        )

        assert isinstance(T0, float)
        assert T0 > 0.0, "Initial temperature must be positive"

    def test_deterministic_with_same_seed(self, tiny_instance):
        """Same seed -> same T0."""
        from src.alns.config import DEFAULT_CONFIG
        from src.solution.structure import create_solution, copy_solution
        from tests.test_alns.conftest import _assign_customer

        data = tiny_instance
        n = data["n_customers"]
        sol = create_solution(data["n_vehicles"], n, n_sku=data["n_sku"])
        _assign_customer(sol, 0, 0, 0, 100.0,
                         data["orders"], data["vehicles"], data["dist_matrix"])
        config = DEFAULT_CONFIG.copy()

        T1 = compute_initial_temperature(
            copy_solution(sol), data, config,
            np.random.default_rng(99),
        )
        T2 = compute_initial_temperature(
            copy_solution(sol), data, config,
            np.random.default_rng(99),
        )
        assert T1 == pytest.approx(T2)
