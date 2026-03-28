"""Tests for adaptive penalty weight scheduling (src/alns/penalty.py). i64 interface."""
import numpy as np
import pytest

from src.alns.penalty import (
    init_penalty_weights,
    adaptive_penalty_adjustment,
)
from src.alns.config import make_config
from src.data.constants import (
    PW_UNSERVED, PW_DUPLICATE, PW_CAPACITY, PW_TIME_WINDOW, PW_SYNC,
    PW_VEHICLE_RESTRICTION, PW_SIZE,
    CFG_PENALTY_W3_START, CFG_PENALTY_W3_END, CFG_PENALTY_DECAY_RATE,
)


def _cfg(overrides=None):
    cfg = make_config(50)
    if overrides:
        for k, v in overrides.items():
            cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------
# init_penalty_weights — returns i64 array + w3 float
# ---------------------------------------------------------------------------

class TestInitPenaltyWeights:

    def test_returns_weights_and_w3(self):
        config = _cfg()
        penalty_weights, w3 = init_penalty_weights(config)
        assert isinstance(penalty_weights, np.ndarray)
        assert penalty_weights.dtype == np.int64
        assert len(penalty_weights) == PW_SIZE
        assert isinstance(w3, float)
        assert w3 == config[CFG_PENALTY_W3_START]

    def test_penalty_weights_have_expected_size(self):
        config = _cfg()
        penalty_weights, _ = init_penalty_weights(config)
        assert len(penalty_weights) == PW_SIZE

    def test_init_returns_correct_w3_from_config(self):
        config = _cfg({CFG_PENALTY_W3_START: 42.0})
        _, w3 = init_penalty_weights(config)
        assert w3 == 42.0

    def test_init_weight_values_are_from_constants(self):
        from src.data.cost import (
            PENALTY_UNSERVED, PENALTY_LATE, PENALTY_OVERLOAD_PER_PCT,
            PENALTY_SYNC_FAIL, PENALTY_RESTRICTION,
        )
        config = _cfg()
        penalty_weights, _ = init_penalty_weights(config)
        assert penalty_weights[PW_UNSERVED] == PENALTY_UNSERVED
        assert penalty_weights[PW_DUPLICATE] == PENALTY_UNSERVED
        assert penalty_weights[PW_CAPACITY] == PENALTY_OVERLOAD_PER_PCT
        assert penalty_weights[PW_TIME_WINDOW] == PENALTY_LATE
        assert penalty_weights[PW_SYNC] == PENALTY_SYNC_FAIL
        assert penalty_weights[PW_VEHICLE_RESTRICTION] == PENALTY_RESTRICTION


# ---------------------------------------------------------------------------
# adaptive_penalty_adjustment
# ---------------------------------------------------------------------------

class TestAdaptivePenaltyAdjustment:

    def test_100_percent_feasible_should_decrease_w3(self):
        config = _cfg({CFG_PENALTY_DECAY_RATE: 0.9998, CFG_PENALTY_W3_END: 0.1})
        feasible_history = np.ones(100, dtype=bool)
        w3 = 10.0
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)
        expected = 10.0 * (0.9998 ** 5)
        assert new_w3 == pytest.approx(expected, abs=1e-6)
        assert new_w3 < w3

    def test_0_percent_feasible_returns_unchanged(self):
        config = _cfg({CFG_PENALTY_W3_START: 50.0, CFG_PENALTY_W3_END: 0.1})
        feasible_history = np.zeros(100, dtype=bool)
        w3 = 5.0
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)
        assert new_w3 == w3

    def test_w3_does_not_go_below_minimum(self):
        config = _cfg({CFG_PENALTY_W3_END: 1.0, CFG_PENALTY_DECAY_RATE: 0.9998})
        feasible_history = np.ones(100, dtype=bool)
        w3 = 0.5
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)
        assert new_w3 >= config[CFG_PENALTY_W3_END]

    def test_w3_does_not_go_below_zero(self):
        config = _cfg({CFG_PENALTY_W3_END: 0.0})
        feasible_history = np.ones(100, dtype=bool)
        w3 = 0.00001
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)
        assert new_w3 >= 0.0

    def test_low_feasible_doubles_w3(self):
        config = _cfg({CFG_PENALTY_W3_START: 100.0, CFG_PENALTY_W3_END: 0.1})
        feasible_history = np.zeros(100, dtype=bool)
        feasible_history[:10] = True
        w3 = 2.0
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)
        assert new_w3 == pytest.approx(4.0, abs=1e-6)

    def test_high_feasible_decreases_faster(self):
        config = _cfg({CFG_PENALTY_DECAY_RATE: 0.9999, CFG_PENALTY_W3_END: 0.1})
        feasible_history = np.ones(100, dtype=bool)
        feasible_history[:10] = False
        w3 = 5.0
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)
        expected = 5.0 * (0.9999 ** 5)
        assert new_w3 == pytest.approx(expected, abs=1e-6)

    def test_w3_never_below_end(self):
        config = _cfg({CFG_PENALTY_W3_END: 0.1, CFG_PENALTY_DECAY_RATE: 0.9999})
        feasible_history = np.ones(100, dtype=bool)
        w3 = 0.05
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)
        assert new_w3 >= config[CFG_PENALTY_W3_END]

    def test_medium_feasible_normal_decay(self):
        config = _cfg({CFG_PENALTY_DECAY_RATE: 0.9999, CFG_PENALTY_W3_END: 0.1})
        feasible_history = np.zeros(100, dtype=bool)
        feasible_history[:50] = True
        w3 = 5.0
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)
        expected = max(5.0 * 0.9999, config[CFG_PENALTY_W3_END])
        assert new_w3 == pytest.approx(expected, abs=1e-6)

    def test_doubling_capped_at_start(self):
        config = _cfg({CFG_PENALTY_W3_START: 10.0, CFG_PENALTY_W3_END: 0.1})
        feasible_history = np.zeros(100, dtype=bool)
        feasible_history[0] = True
        w3 = 8.0
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)
        assert new_w3 <= config[CFG_PENALTY_W3_START]
        assert new_w3 == pytest.approx(10.0, abs=1e-6)

    def test_boundary_exactly_30_percent_feasible(self):
        config = _cfg({CFG_PENALTY_DECAY_RATE: 0.9999, CFG_PENALTY_W3_END: 0.1})
        feasible_history = np.zeros(100, dtype=bool)
        feasible_history[:30] = True
        w3 = 5.0
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)
        expected = max(5.0 * 0.9999, config[CFG_PENALTY_W3_END])
        assert new_w3 == pytest.approx(expected, abs=1e-6)

    def test_boundary_exactly_80_percent_feasible(self):
        config = _cfg({CFG_PENALTY_DECAY_RATE: 0.9999, CFG_PENALTY_W3_END: 0.1})
        feasible_history = np.ones(100, dtype=bool)
        feasible_history[:20] = False
        w3 = 5.0
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)
        expected = max(5.0 * 0.9999, config[CFG_PENALTY_W3_END])
        assert new_w3 == pytest.approx(expected, abs=1e-6)

    def test_empty_feasible_history_returns_unchanged(self):
        config = _cfg({CFG_PENALTY_DECAY_RATE: 0.9999, CFG_PENALTY_W3_END: 0.1})
        feasible_history = np.array([], dtype=bool)
        w3 = 5.0
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)
        assert new_w3 == w3

    def test_decay_rate_exactly_1_0_no_decay(self):
        config = _cfg({CFG_PENALTY_DECAY_RATE: 1.0, CFG_PENALTY_W3_END: 0.1})
        feasible_history = np.zeros(100, dtype=bool)
        feasible_history[:50] = True
        w3 = 5.0
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)
        expected = max(5.0 * 1.0, config[CFG_PENALTY_W3_END])
        assert new_w3 == pytest.approx(expected, abs=1e-6)

    def test_w3_at_exactly_w3_end_boundary(self):
        config = _cfg({CFG_PENALTY_W3_END: 1.0, CFG_PENALTY_DECAY_RATE: 0.5})
        feasible_history = np.ones(100, dtype=bool)
        w3 = 1.0
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)
        assert new_w3 >= config[CFG_PENALTY_W3_END]
        assert new_w3 == pytest.approx(1.0, abs=1e-6)

    def test_very_large_w3_doubling(self):
        config = _cfg({CFG_PENALTY_W3_START: 1000.0, CFG_PENALTY_W3_END: 0.1})
        feasible_history = np.zeros(100, dtype=bool)
        feasible_history[0] = True
        w3 = 600.0
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)
        assert new_w3 == pytest.approx(1000.0, abs=1e-6)

    def test_very_small_w3_decay(self):
        config = _cfg({CFG_PENALTY_W3_END: 0.001, CFG_PENALTY_DECAY_RATE: 0.9})
        feasible_history = np.zeros(100, dtype=bool)
        feasible_history[:50] = True
        w3 = 0.0005
        new_w3 = adaptive_penalty_adjustment(w3, feasible_history, config)
        assert new_w3 >= config[CFG_PENALTY_W3_END]
