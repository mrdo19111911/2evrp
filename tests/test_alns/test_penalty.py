"""Tests for adaptive penalty weight scheduling (src/alns/penalty.py)."""
import numpy as np
import pytest

from src.alns.penalty import (
    init_penalty_weights,
    decay_penalty_weight,
    adaptive_penalty_adjustment,
)
from src.alns.config import DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# init_penalty_weights
# ---------------------------------------------------------------------------

class TestInitPenaltyWeights:

    def test_returns_weights_and_w3(self):
        """Returns (penalty_weights dict, w3 float)."""
        config = DEFAULT_CONFIG.copy()
        penalty_weights, w3 = init_penalty_weights(config)

        assert isinstance(penalty_weights, dict)
        assert isinstance(w3, float)
        assert w3 == config["penalty_w3_start"]

    def test_penalty_weights_have_expected_keys(self):
        """penalty_weights dict contains violation type keys."""
        config = DEFAULT_CONFIG.copy()
        penalty_weights, _ = init_penalty_weights(config)

        # Should have keys for each violation type
        assert "unserved" in penalty_weights
        # v2: capacity split into kg/cbm or just "capacity_kg"
        has_capacity = ("capacity" in penalty_weights
                        or "capacity_kg" in penalty_weights)
        assert has_capacity, f"Missing capacity key: {list(penalty_weights.keys())}"


# ---------------------------------------------------------------------------
# decay_penalty_weight
# ---------------------------------------------------------------------------

class TestDecayPenaltyWeight:

    def test_geometric_decay(self):
        """10.0 * 0.9999 = 9.999."""
        config = DEFAULT_CONFIG.copy()
        new_w3 = decay_penalty_weight(10.0, config)
        np.testing.assert_allclose(new_w3, 10.0 * 0.9999, atol=1e-8)

    def test_does_not_go_below_end(self):
        """w3 never drops below penalty_w3_end."""
        config = DEFAULT_CONFIG.copy()
        config["penalty_w3_end"] = 0.1
        new_w3 = decay_penalty_weight(0.1, config)
        assert new_w3 >= config["penalty_w3_end"]


# ---------------------------------------------------------------------------
# adaptive_penalty_adjustment
# ---------------------------------------------------------------------------

class TestAdaptivePenaltyAdjustment:

    def test_low_feasible_doubles_w3(self):
        """10% feasible (< 30%) -> w3 doubles."""
        config = DEFAULT_CONFIG.copy()
        config["penalty_w3_start"] = 10.0
        config["penalty_w3_end"] = 0.1

        # 10% feasible
        feasible_history = np.zeros(100, dtype=bool)
        feasible_history[:10] = True  # 10/100 = 10%

        w3 = 2.0
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)

        # Should double (but capped at penalty_w3_start)
        assert new_w3 == pytest.approx(4.0, abs=1e-6), \
            f"Expected w3 to double from 2.0 to 4.0, got {new_w3}"

    def test_high_feasible_decreases_faster(self):
        """90% feasible (> 80%) -> w3 decreases faster (decay^5)."""
        config = DEFAULT_CONFIG.copy()
        config["penalty_decay_rate"] = 0.9999
        config["penalty_w3_end"] = 0.1

        # 90% feasible
        feasible_history = np.ones(100, dtype=bool)
        feasible_history[:10] = False  # 90/100 = 90%

        w3 = 5.0
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)

        expected = 5.0 * (0.9999 ** 5)
        assert new_w3 == pytest.approx(expected, abs=1e-6), \
            f"Expected {expected}, got {new_w3}"

    def test_w3_never_below_end(self):
        """Regardless of history, w3 >= penalty_w3_end."""
        config = DEFAULT_CONFIG.copy()
        config["penalty_w3_end"] = 0.1
        config["penalty_decay_rate"] = 0.9999

        feasible_history = np.ones(100, dtype=bool)  # 100% feasible
        w3 = 0.1  # already at minimum

        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)
        assert new_w3 >= config["penalty_w3_end"]

    def test_medium_feasible_normal_decay(self):
        """50% feasible (between 30% and 80%) -> normal single decay."""
        config = DEFAULT_CONFIG.copy()
        config["penalty_decay_rate"] = 0.9999
        config["penalty_w3_end"] = 0.1

        feasible_history = np.zeros(100, dtype=bool)
        feasible_history[:50] = True  # 50%

        w3 = 5.0
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)

        expected = max(5.0 * 0.9999, config["penalty_w3_end"])
        assert new_w3 == pytest.approx(expected, abs=1e-6)

    def test_doubling_capped_at_start(self):
        """When w3 doubles, it should not exceed penalty_w3_start."""
        config = DEFAULT_CONFIG.copy()
        config["penalty_w3_start"] = 10.0
        config["penalty_w3_end"] = 0.1

        feasible_history = np.zeros(100, dtype=bool)  # 0% feasible

        w3 = 8.0
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)

        # 8.0 * 2 = 16.0, but capped at 10.0
        assert new_w3 <= config["penalty_w3_start"]
        assert new_w3 == pytest.approx(10.0, abs=1e-6)
