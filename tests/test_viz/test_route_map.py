"""Tests for src/viz/route_map.py — 2D route map visualization."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.viz.route_map import plot_route_map, plot_single_route, show_route_map


# ---------------------------------------------------------------------------
# plot_route_map
# ---------------------------------------------------------------------------

class TestPlotRouteMap:
    """Tests for plot_route_map."""

    def test_returns_fig_and_ax(self, tiny_instance, tiny_dist_matrix, mock_sol):
        """plot_route_map returns a (Figure, Axes) tuple."""
        result = plot_route_map(
            sol=mock_sol,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
        )
        fig, ax = result
        assert isinstance(fig, Figure)
        # ax should be a matplotlib Axes
        assert hasattr(ax, "plot"), "ax should be a matplotlib Axes instance"

    def test_with_empty_solution_no_crash(
        self, tiny_instance, tiny_dist_matrix, empty_sol
    ):
        """Empty solution (no assigned customers) should not crash."""
        fig, ax = plot_route_map(
            sol=empty_sol,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
        )
        assert isinstance(fig, Figure)

    def test_highlight_violations_creates_figure(
        self, tiny_instance, tiny_dist_matrix, mock_sol,
        mock_vehicle_states,
    ):
        """highlight_violations=True with states should produce a figure."""
        fig, ax = plot_route_map(
            sol=mock_sol,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
            states=mock_vehicle_states,
            highlight_violations=True,
        )
        assert isinstance(fig, Figure)

    def test_show_labels_false(self, tiny_instance, tiny_dist_matrix, mock_sol):
        """show_labels=False should not crash."""
        fig, ax = plot_route_map(
            sol=mock_sol,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
            show_labels=False,
        )
        assert isinstance(fig, Figure)

    def test_show_demand_true(self, tiny_instance, tiny_dist_matrix, mock_sol):
        """show_demand=True should not crash."""
        fig, ax = plot_route_map(
            sol=mock_sol,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
            show_demand=True,
        )
        assert isinstance(fig, Figure)

    def test_custom_title_and_figsize(
        self, tiny_instance, tiny_dist_matrix, mock_sol
    ):
        """Custom title and figsize are accepted."""
        fig, ax = plot_route_map(
            sol=mock_sol,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
            title="Test Title",
            figsize=(8, 6),
        )
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_single_route
# ---------------------------------------------------------------------------

class TestPlotSingleRoute:
    """Tests for plot_single_route."""

    def test_returns_fig_and_axes_with_two_subplots(
        self, tiny_instance, tiny_dist_matrix, mock_sol
    ):
        """plot_single_route returns (fig, axes) with 2 subplots."""
        fig, axes = plot_single_route(
            sol=mock_sol,
            vid=0,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
        )
        assert isinstance(fig, Figure)
        # Should have 2 subplots (map + load/time profile)
        assert len(axes) == 2
        for ax in axes:
            assert hasattr(ax, "plot")

    def test_bike_route(self, tiny_instance, tiny_dist_matrix, mock_sol):
        """plot_single_route works for a bike route."""
        fig, axes = plot_single_route(
            sol=mock_sol,
            vid=1,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
        )
        assert isinstance(fig, Figure)
        assert len(axes) == 2

    def test_with_state(
        self, tiny_instance, tiny_dist_matrix, mock_sol, mock_vehicle_states
    ):
        """plot_single_route accepts optional state array."""
        fig, axes = plot_single_route(
            sol=mock_sol,
            vid=0,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
            state=mock_vehicle_states[0],
        )
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# show_route_map
# ---------------------------------------------------------------------------

class TestShowRouteMap:
    """Tests for show_route_map."""

    def test_show_route_map_no_crash(
        self, tiny_instance, tiny_dist_matrix, mock_sol
    ):
        """show_route_map with Agg backend should not crash."""
        plt.switch_backend("Agg")
        # show_route_map returns None (calls plt.show)
        show_route_map(
            sol=mock_sol,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
        )
