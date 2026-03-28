"""Tests for adaptive operator weight management (src/alns/weights.py)."""
import numpy as np
import pytest

from src.alns.weights import select_operator


# ---------------------------------------------------------------------------
# select_operator — now takes seed instead of rng
# ---------------------------------------------------------------------------

class TestSelectOperator:

    def test_deterministic_when_one_weight(self):
        weights = np.array([1.0, 0.0, 0.0])
        for seed in range(50):
            idx = select_operator(weights, seed)
            assert idx == 0

    def test_respects_distribution(self):
        weights = np.array([0.5, 0.3, 0.2])
        counts = np.zeros(3, dtype=int)
        n_trials = 2000
        for seed in range(n_trials):
            idx = select_operator(weights, seed)
            counts[idx] += 1
        assert 0.40 <= counts[0] / n_trials <= 0.60
        assert 0.12 <= counts[2] / n_trials <= 0.28

    def test_returns_valid_index(self):
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        for seed in range(100):
            idx = select_operator(weights, seed)
            assert 0 <= idx < 4


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestSelectOperatorEdgeCases:

    def test_select_with_all_zero_weights(self):
        weights = np.array([0.0, 0.0, 0.0])
        idx = select_operator(weights, 42)
        assert 0 <= idx < 3

    def test_select_with_nan_weights(self):
        weights = np.array([float('nan'), 0.5, 0.5])
        idx = select_operator(weights, 42)
        assert 0 <= idx < 3

    def test_select_with_one_dominant_weight(self):
        weights = np.array([0.9999, 0.00005, 0.00005])
        for seed in range(100):
            idx = select_operator(weights, seed + 1000)
            assert idx == 0

    def test_select_weights_not_normalized(self):
        """With njit roulette, non-normalized weights still work (select proportionally)."""
        weights = np.array([1.5, 0.5])
        idx = select_operator(weights, 42)
        assert 0 <= idx < 2
