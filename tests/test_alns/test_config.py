"""Tests for ALNS configuration (src/alns/config.py)."""
import numpy as np
import pytest

from src.alns.config import make_config
from src.data.constants import (
    CFG_MAX_ITERATIONS, CFG_SEGMENT_LENGTH, CFG_NO_IMPROVE_LIMIT,
    CFG_SA_INITIAL_TEMP, CFG_SA_COOLING_RATE, CFG_REACTION_FACTOR,
    CFG_Q_MAX_PCT, CFG_Q_MAX,
)


class TestMakeConfig:

    def test_small_instance_20000_iterations(self):
        config = make_config(50)
        assert config[CFG_MAX_ITERATIONS] == 20000
        assert config[CFG_NO_IMPROVE_LIMIT] == 2000

    def test_medium_instance_50000_iterations(self):
        config = make_config(200)
        assert config[CFG_MAX_ITERATIONS] == 50000
        assert config[CFG_NO_IMPROVE_LIMIT] == 5000

    def test_large_instance_100000_iterations(self):
        config = make_config(1000)
        assert config[CFG_MAX_ITERATIONS] == 100000
        assert config[CFG_NO_IMPROVE_LIMIT] == 10000

    def test_q_max_computed(self):
        config = make_config(1000)
        expected_q_max = max(5, int(1000 * config[CFG_Q_MAX_PCT]))
        assert config[CFG_Q_MAX] == expected_q_max
        assert config[CFG_Q_MAX] == 150

    def test_q_max_minimum_5(self):
        config = make_config(10)
        assert config[CFG_Q_MAX] >= 5

    def test_override_max_iterations(self):
        config = make_config(50, overrides={CFG_MAX_ITERATIONS: 999})
        assert config[CFG_MAX_ITERATIONS] == 999

    def test_override_does_not_affect_other_keys(self):
        base = make_config(50)
        config = make_config(50, overrides={CFG_SA_INITIAL_TEMP: 200.0})
        assert config[CFG_SA_INITIAL_TEMP] == 200.0
        assert config[CFG_SA_COOLING_RATE] == base[CFG_SA_COOLING_RATE]
        assert config[CFG_SEGMENT_LENGTH] == base[CFG_SEGMENT_LENGTH]

    def test_no_overrides(self):
        config = make_config(50)
        base = make_config(50)
        assert config[CFG_SA_COOLING_RATE] == base[CFG_SA_COOLING_RATE]
        assert config[CFG_REACTION_FACTOR] == base[CFG_REACTION_FACTOR]

    def test_none_overrides(self):
        config = make_config(50, overrides=None)
        assert config[CFG_MAX_ITERATIONS] == 20000

    def test_returns_ndarray(self):
        config = make_config(50)
        assert isinstance(config, np.ndarray)
        assert config.dtype == np.float64
