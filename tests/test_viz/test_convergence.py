"""Tests for src/viz/convergence.py — fitness convergence plots."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from src.viz.convergence import plot_convergence, update_convergence


# ---------------------------------------------------------------------------
# plot_convergence
# ---------------------------------------------------------------------------

class TestPlotConvergence:
    """Tests for plot_convergence."""

    def test_returns_fig_and_axes(self, mock_log_100):
        """plot_convergence returns (fig, axes) with multiple subplots."""
        fig, axes = plot_convergence(mock_log_100)
        assert isinstance(fig, Figure)
        # Spec says multi-panel: at least 2 axes
        assert isinstance(axes, (list, np.ndarray))
        assert len(axes) >= 2

    def test_all_panels_enabled(self, mock_log_100):
        """With all panels enabled, we get the full set of axes."""
        fig, axes = plot_convergence(
            mock_log_100,
            show_cost=True,
            show_makespan=True,
            show_penalty=True,
            show_temperature=True,
            show_feasibility=True,
        )
        assert isinstance(fig, Figure)
        # Should have multiple panels
        assert len(axes) >= 2

    def test_single_panel_cost_only(self, mock_log_100):
        """Enabling only cost panel should still work."""
        fig, axes = plot_convergence(
            mock_log_100,
            show_cost=True,
            show_makespan=False,
            show_penalty=False,
            show_temperature=False,
            show_feasibility=False,
        )
        assert isinstance(fig, Figure)

    def test_empty_log_no_crash(self, empty_log):
        """Empty log should not crash."""
        fig, axes = plot_convergence(empty_log)
        assert isinstance(fig, Figure)

    def test_custom_figsize(self, mock_log_100):
        """Custom figsize should be accepted."""
        fig, axes = plot_convergence(mock_log_100, figsize=(10, 8))
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# update_convergence
# ---------------------------------------------------------------------------

class TestUpdateConvergence:
    """Tests for update_convergence (live dashboard append)."""

    def test_update_with_growing_log(self, mock_log_100):
        """update_convergence should not crash when called with growing log."""
        fig, axes = plot_convergence(mock_log_100)

        # Simulate growing log: append 10 more iterations
        extended_log = dict(mock_log_100)
        rng = np.random.default_rng(99)
        n_new = 10
        extended_log["iterations"] = np.arange(110)
        extended_log["fitness"] = np.concatenate([
            mock_log_100["fitness"],
            mock_log_100["fitness"][-1] + rng.normal(0, 10, n_new),
        ])
        extended_log["best_fitness"] = np.minimum.accumulate(
            extended_log["fitness"]
        )

        # Should not crash
        update_convergence(fig, axes, extended_log)

    def test_update_empty_then_data(self, empty_log, mock_log_100):
        """Start with empty log, then update with data."""
        fig, axes = plot_convergence(empty_log)
        update_convergence(fig, axes, mock_log_100)
