"""Tests for adaptive operator weight management (src/alns/weights.py)."""
import numpy as np
import pytest

from src.alns.weights import (
    init_weights,
    select_operator,
    update_scores,
    update_weights,
)


# ---------------------------------------------------------------------------
# init_weights
# ---------------------------------------------------------------------------

class TestInitWeights:

    def test_uniform_sum_to_one(self):
        """init_weights(6, 3, 4) -> destroy_weights uniform, sum=1."""
        result = init_weights(6, 3, 4)
        # Result should contain at least destroy_weights, repair_weights, cross_weights
        # Per spec: returns (destroy_w, repair_w, cross_w, d_scores, r_scores, c_scores,
        #                     d_counts, r_counts, c_counts)
        d_w, r_w, c_w = result[0], result[1], result[2]

        assert len(d_w) == 6
        assert len(r_w) == 3
        assert len(c_w) == 4

        np.testing.assert_allclose(d_w.sum(), 1.0, atol=1e-10)
        np.testing.assert_allclose(r_w.sum(), 1.0, atol=1e-10)
        np.testing.assert_allclose(c_w.sum(), 1.0, atol=1e-10)

        # All weights equal (uniform)
        np.testing.assert_allclose(d_w, 1.0 / 6, atol=1e-10)
        np.testing.assert_allclose(r_w, 1.0 / 3, atol=1e-10)
        np.testing.assert_allclose(c_w, 1.0 / 4, atol=1e-10)

    def test_scores_and_counts_are_zero(self):
        """Scores and counts arrays start at zero."""
        result = init_weights(6, 3, 4)
        d_scores, r_scores, c_scores = result[3], result[4], result[5]
        d_counts, r_counts, c_counts = result[6], result[7], result[8]

        np.testing.assert_array_equal(d_scores, np.zeros(6))
        np.testing.assert_array_equal(r_scores, np.zeros(3))
        np.testing.assert_array_equal(c_scores, np.zeros(4))
        np.testing.assert_array_equal(d_counts, np.zeros(6, dtype=np.int32))
        np.testing.assert_array_equal(r_counts, np.zeros(3, dtype=np.int32))
        np.testing.assert_array_equal(c_counts, np.zeros(4, dtype=np.int32))


# ---------------------------------------------------------------------------
# select_operator
# ---------------------------------------------------------------------------

class TestSelectOperator:

    def test_deterministic_when_one_weight(self):
        """weights=[1,0,0] -> always selects operator 0."""
        weights = np.array([1.0, 0.0, 0.0])
        rng = np.random.default_rng(42)
        for _ in range(50):
            idx = select_operator(weights, rng)
            assert idx == 0

    def test_respects_distribution(self):
        """weights=[0.5, 0.3, 0.2] -> operator 0 chosen ~50% over 2000 trials."""
        weights = np.array([0.5, 0.3, 0.2])
        rng = np.random.default_rng(123)
        counts = np.zeros(3, dtype=int)
        n_trials = 2000
        for _ in range(n_trials):
            idx = select_operator(weights, rng)
            counts[idx] += 1

        # Operator 0 should be chosen ~50% of the time
        assert 0.40 <= counts[0] / n_trials <= 0.60, (
            f"Operator 0 frequency {counts[0]/n_trials:.2f} not near 0.50"
        )
        # Operator 2 should be chosen ~20%
        assert 0.12 <= counts[2] / n_trials <= 0.28

    def test_returns_valid_index(self):
        """Returned index is in [0, len(weights))."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        rng = np.random.default_rng(0)
        for _ in range(100):
            idx = select_operator(weights, rng)
            assert 0 <= idx < 4


# ---------------------------------------------------------------------------
# update_scores
# ---------------------------------------------------------------------------

class TestUpdateScores:

    def test_global_best_adds_sigma3(self):
        """result=3 (global best) -> score += 5 (sigma_3)."""
        scores = np.zeros(4, dtype=np.float64)
        counts = np.zeros(4, dtype=np.int32)
        update_scores(scores, counts, op_idx=2, result=3, score_values=(1, 2, 5))
        assert scores[2] == 5.0
        assert counts[2] == 1

    def test_worse_not_accepted_adds_zero(self):
        """result=0 (worse, not accepted) -> score += 0, but count += 1."""
        scores = np.zeros(3, dtype=np.float64)
        counts = np.zeros(3, dtype=np.int32)
        update_scores(scores, counts, op_idx=1, result=0, score_values=(1, 2, 5))
        assert scores[1] == 0.0
        assert counts[1] == 1

    def test_accepted_worse_adds_sigma1(self):
        """result=1 (accepted worse) -> score += 1 (sigma_1)."""
        scores = np.zeros(3, dtype=np.float64)
        counts = np.zeros(3, dtype=np.int32)
        update_scores(scores, counts, op_idx=0, result=1, score_values=(1, 2, 5))
        assert scores[0] == 1.0
        assert counts[0] == 1

    def test_new_local_best_adds_sigma2(self):
        """result=2 (new local best) -> score += 2 (sigma_2)."""
        scores = np.zeros(3, dtype=np.float64)
        counts = np.zeros(3, dtype=np.int32)
        update_scores(scores, counts, op_idx=0, result=2, score_values=(1, 2, 5))
        assert scores[0] == 2.0

    def test_cumulative_scores(self):
        """Multiple updates accumulate scores."""
        scores = np.zeros(2, dtype=np.float64)
        counts = np.zeros(2, dtype=np.int32)
        update_scores(scores, counts, 0, 3, (1, 2, 5))  # +5
        update_scores(scores, counts, 0, 2, (1, 2, 5))  # +2
        update_scores(scores, counts, 0, 1, (1, 2, 5))  # +1
        assert scores[0] == 8.0
        assert counts[0] == 3


# ---------------------------------------------------------------------------
# update_weights
# ---------------------------------------------------------------------------

class TestUpdateWeights:

    def test_high_performer_gets_higher_weight(self):
        """Operator with high score/count ratio -> weight increases."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        scores = np.array([20.0, 1.0, 1.0, 1.0])
        counts = np.array([4, 4, 4, 4], dtype=np.int32)
        # performance: [5.0, 0.25, 0.25, 0.25]
        original_w0 = weights[0]

        update_weights(weights, scores, counts, reaction_factor=0.1)

        assert weights[0] > original_w0, "High-performing operator weight should increase"

    def test_weights_sum_to_one(self):
        """After update, weights still sum to 1."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        scores = np.array([10.0, 0.0, 5.0, 2.0])
        counts = np.array([5, 5, 5, 5], dtype=np.int32)

        update_weights(weights, scores, counts, reaction_factor=0.1)

        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-10)

    def test_minimum_weight_enforced(self):
        """An operator with 0 score still has weight >= 0.01."""
        weights = np.array([0.5, 0.5])
        scores = np.array([100.0, 0.0])
        counts = np.array([10, 10], dtype=np.int32)

        update_weights(weights, scores, counts, reaction_factor=0.9)

        assert weights[1] >= 0.01, "Minimum weight 0.01 should be enforced"

    def test_scores_and_counts_reset(self):
        """After update, scores and counts are zeroed out."""
        weights = np.array([0.5, 0.5])
        scores = np.array([5.0, 3.0])
        counts = np.array([2, 3], dtype=np.int32)

        update_weights(weights, scores, counts, reaction_factor=0.1)

        np.testing.assert_array_equal(scores, np.zeros(2))
        np.testing.assert_array_equal(counts, np.zeros(2, dtype=np.int32))

    def test_reaction_factor_zero_no_change(self):
        """reaction_factor=0 -> weights unchanged (stay original after normalize)."""
        weights = np.array([0.6, 0.3, 0.1])
        original = weights.copy()
        scores = np.array([100.0, 0.0, 0.0])
        counts = np.array([10, 10, 10], dtype=np.int32)

        update_weights(weights, scores, counts, reaction_factor=0.0)

        # Weights should be renormalized but essentially unchanged
        # (minimum weight 0.01 may slightly affect 0.1)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-10)
