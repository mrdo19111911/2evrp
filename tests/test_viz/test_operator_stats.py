"""Tests for src/viz/operator_stats.py — operator weights and performance."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from src.viz.operator_stats import plot_operator_weights, plot_operator_performance


# ---------------------------------------------------------------------------
# plot_operator_weights
# ---------------------------------------------------------------------------

class TestPlotOperatorWeights:
    """Tests for plot_operator_weights."""

    def test_returns_fig_and_axes(self, mock_log_100):
        """plot_operator_weights returns (fig, axes)."""
        fig, axes = plot_operator_weights(mock_log_100)
        assert isinstance(fig, Figure)
        # Should have axes (list or array, typically 2: destroy + repair)
        assert isinstance(axes, (list, np.ndarray))
        assert len(axes) >= 1

    def test_figure_created_with_mock_weights(self, mock_log_100):
        """Weights history produces a visible figure."""
        fig, axes = plot_operator_weights(mock_log_100)
        assert isinstance(fig, Figure)
        # Figure should have at least one axes child
        assert len(fig.get_axes()) >= 1

    def test_custom_figsize(self, mock_log_100):
        """Custom figsize should be accepted."""
        fig, axes = plot_operator_weights(mock_log_100, figsize=(8, 4))
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_operator_performance
# ---------------------------------------------------------------------------

class TestPlotOperatorPerformance:
    """Tests for plot_operator_performance."""

    def test_returns_fig_and_axes(self, mock_log_100):
        """plot_operator_performance returns (fig, axes)."""
        fig, axes = plot_operator_performance(mock_log_100)
        assert isinstance(fig, Figure)
        assert isinstance(axes, (list, np.ndarray))
        assert len(axes) >= 1

    def test_figure_created_with_mock_data(self, mock_log_100):
        """Performance data produces a visible figure."""
        fig, axes = plot_operator_performance(mock_log_100)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) >= 1

    def test_custom_figsize(self, mock_log_100):
        """Custom figsize should be accepted."""
        fig, axes = plot_operator_performance(mock_log_100, figsize=(10, 6))
        assert isinstance(fig, Figure)
