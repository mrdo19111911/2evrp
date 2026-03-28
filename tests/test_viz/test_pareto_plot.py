"""Tests for src/viz/pareto_plot.py — Pareto front visualization."""
import matplotlib
matplotlib.use("Agg")

from matplotlib.figure import Figure

from src.viz.pareto_plot import plot_pareto_front


# ---------------------------------------------------------------------------
# plot_pareto_front
# ---------------------------------------------------------------------------

class TestPlotParetoFront:
    """Tests for plot_pareto_front."""

    def test_returns_fig_and_ax(self, mock_archive_5):
        """plot_pareto_front with 5 solutions returns (fig, ax)."""
        fig, ax = plot_pareto_front(mock_archive_5)
        assert isinstance(fig, Figure)
        assert hasattr(ax, "plot")

    def test_scatter_points_present(self, mock_archive_5):
        """Scatter plot should contain data points."""
        fig, ax = plot_pareto_front(mock_archive_5)
        # At least one collection or line should be drawn
        has_content = (
            len(ax.collections) > 0
            or len(ax.lines) > 0
            or len(ax.patches) > 0
        )
        assert has_content, "Pareto plot should contain scatter points or lines"

    def test_empty_archive_no_crash(self, empty_archive):
        """Empty archive should not crash."""
        fig, ax = plot_pareto_front(empty_archive)
        assert isinstance(fig, Figure)

    def test_highlight_best_true(self, mock_archive_5):
        """highlight_best=True should produce a figure."""
        fig, ax = plot_pareto_front(mock_archive_5, highlight_best=True)
        assert isinstance(fig, Figure)

    def test_highlight_best_false(self, mock_archive_5):
        """highlight_best=False should produce a figure."""
        fig, ax = plot_pareto_front(mock_archive_5, highlight_best=False)
        assert isinstance(fig, Figure)

    def test_show_dominated_true(self, mock_archive_5):
        """show_dominated=True should not crash."""
        fig, ax = plot_pareto_front(mock_archive_5, show_dominated=True)
        assert isinstance(fig, Figure)

    def test_custom_figsize(self, mock_archive_5):
        """Custom figsize should be accepted."""
        fig, ax = plot_pareto_front(mock_archive_5, figsize=(10, 8))
        assert isinstance(fig, Figure)

    def test_single_solution_archive(self):
        """Archive with a single solution should not crash."""
        single = [{"cost": 1_000_000.0, "makespan": 300.0, "fitness": 800.0, "feasible": True}]
        fig, ax = plot_pareto_front(single)
        assert isinstance(fig, Figure)
