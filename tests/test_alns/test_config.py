"""Tests for ALNS configuration (src/alns/config.py)."""
import pytest

from src.alns.config import make_config, DEFAULT_CONFIG


class TestMakeConfig:

    def test_small_instance_20000_iterations(self):
        """N=50 (< 100) -> max_iterations=20000, no_improve_limit=2000."""
        config = make_config(50)
        assert config["max_iterations"] == 20000
        assert config["no_improve_limit"] == 2000

    def test_medium_instance_50000_iterations(self):
        """N=200 (100 <= N < 500) -> max_iterations=50000."""
        config = make_config(200)
        assert config["max_iterations"] == 50000
        assert config["no_improve_limit"] == 5000

    def test_large_instance_100000_iterations(self):
        """N=1000 (>= 500) -> max_iterations=100000, no_improve_limit=10000."""
        config = make_config(1000)
        assert config["max_iterations"] == 100000
        assert config["no_improve_limit"] == 10000

    def test_q_max_computed(self):
        """q_max = max(5, int(N * q_max_pct))."""
        config = make_config(1000)
        expected_q_max = max(5, int(1000 * DEFAULT_CONFIG["q_max_pct"]))
        assert config["q_max"] == expected_q_max
        assert config["q_max"] == 150  # 1000 * 0.15 = 150

    def test_q_max_minimum_5(self):
        """For very small N, q_max is at least 5."""
        config = make_config(10)
        assert config["q_max"] >= 5

    def test_override_max_iterations(self):
        """Override max_iterations=999 -> config['max_iterations']=999."""
        config = make_config(50, overrides={"max_iterations": 999})
        assert config["max_iterations"] == 999

    def test_override_does_not_affect_other_keys(self):
        """Overriding one key preserves default values for others."""
        config = make_config(50, overrides={"sa_initial_temp": 200.0})
        assert config["sa_initial_temp"] == 200.0
        assert config["sa_cooling_rate"] == DEFAULT_CONFIG["sa_cooling_rate"]
        assert config["segment_length"] == DEFAULT_CONFIG["segment_length"]

    def test_no_overrides(self):
        """Calling without overrides (or None) uses only defaults + scale."""
        config = make_config(50)
        assert config["sa_cooling_rate"] == DEFAULT_CONFIG["sa_cooling_rate"]
        assert config["reaction_factor"] == DEFAULT_CONFIG["reaction_factor"]

    def test_none_overrides(self):
        """Passing overrides=None is equivalent to no overrides."""
        config = make_config(50, overrides=None)
        assert config["max_iterations"] == 20000

    def test_returns_dict(self):
        """make_config returns a plain dict."""
        config = make_config(50)
        assert isinstance(config, dict)

    def test_does_not_mutate_default(self):
        """Calling make_config should not mutate DEFAULT_CONFIG."""
        original = DEFAULT_CONFIG.copy()
        make_config(50, overrides={"max_iterations": 1})
        assert DEFAULT_CONFIG == original
