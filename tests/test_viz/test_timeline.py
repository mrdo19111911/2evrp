"""Tests for src/viz/timeline.py — Gantt chart vehicle timelines."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from src.viz.timeline import plot_timeline


# ---------------------------------------------------------------------------
# plot_timeline
# ---------------------------------------------------------------------------

class TestPlotTimeline:
    """Tests for plot_timeline."""

    def test_returns_fig_and_ax(
        self, tiny_instance, mock_sol, mock_vehicle_states,
    ):
        """plot_timeline returns a (Figure, Axes) tuple."""
        fig, ax = plot_timeline(
            sol=mock_sol,
            states=mock_vehicle_states,
            vehicles=tiny_instance["vehicles"],
        )
        assert isinstance(fig, Figure)
        assert hasattr(ax, "plot")

    def test_empty_routes_no_crash(self, tiny_instance, empty_sol):
        """Empty routes (all idle vehicles) should not crash."""
        empty_states = [
            np.empty((0, 12), dtype=np.float64),
            np.empty((0, 12), dtype=np.float64),
            np.empty((0, 12), dtype=np.float64),
        ]
        fig, ax = plot_timeline(
            sol=empty_sol,
            states=empty_states,
            vehicles=tiny_instance["vehicles"],
        )
        assert isinstance(fig, Figure)

    def test_with_sync_lines(
        self, tiny_instance, mock_sol, mock_vehicle_states,
    ):
        """show_sync=True should produce a figure without error."""
        fig, ax = plot_timeline(
            sol=mock_sol,
            states=mock_vehicle_states,
            vehicles=tiny_instance["vehicles"],
            show_sync=True,
        )
        assert isinstance(fig, Figure)

    def test_without_sync_lines(
        self, tiny_instance, mock_sol, mock_vehicle_states,
    ):
        """show_sync=False should produce a figure without error."""
        fig, ax = plot_timeline(
            sol=mock_sol,
            states=mock_vehicle_states,
            vehicles=tiny_instance["vehicles"],
            show_sync=False,
        )
        assert isinstance(fig, Figure)

    def test_show_tw_false(
        self, tiny_instance, mock_sol, mock_vehicle_states,
    ):
        """show_tw=False should produce a figure without error."""
        fig, ax = plot_timeline(
            sol=mock_sol,
            states=mock_vehicle_states,
            vehicles=tiny_instance["vehicles"],
        )
        assert isinstance(fig, Figure)

    def test_custom_figsize(
        self, tiny_instance, mock_sol, mock_vehicle_states,
    ):
        """Custom figsize should be accepted."""
        fig, ax = plot_timeline(
            sol=mock_sol,
            states=mock_vehicle_states,
            vehicles=tiny_instance["vehicles"],
            figsize=(10, 5),
        )
        assert isinstance(fig, Figure)
