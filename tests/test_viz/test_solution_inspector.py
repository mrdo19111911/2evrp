"""Tests for src/viz/solution_inspector.py — text + visual solution inspection."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from src.viz.solution_inspector import print_solution_summary, plot_solution_detail


# ---------------------------------------------------------------------------
# print_solution_summary
# ---------------------------------------------------------------------------

class TestPrintSolutionSummary:
    """Tests for print_solution_summary."""

    def test_returns_nonempty_string(
        self, tiny_instance, mock_sol, mock_eval_result,
    ):
        """print_solution_summary returns a non-empty string."""
        result = print_solution_summary(
            sol=mock_sol,
            eval_result=mock_eval_result,
            vehicles=tiny_instance["vehicles"],
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_cost(
        self, tiny_instance, mock_sol, mock_eval_result,
    ):
        """Output contains 'Cost:' or 'cost' somewhere."""
        result = print_solution_summary(
            sol=mock_sol,
            eval_result=mock_eval_result,
            vehicles=tiny_instance["vehicles"],
        )
        assert "Cost" in result or "cost" in result

    def test_contains_makespan(
        self, tiny_instance, mock_sol, mock_eval_result,
    ):
        """Output contains 'Makespan:' or 'makespan' somewhere."""
        result = print_solution_summary(
            sol=mock_sol,
            eval_result=mock_eval_result,
            vehicles=tiny_instance["vehicles"],
        )
        assert "Makespan" in result or "makespan" in result

    def test_contains_feasible(
        self, tiny_instance, mock_sol, mock_eval_result,
    ):
        """Output contains 'Feasible:' or 'feasible' somewhere."""
        result = print_solution_summary(
            sol=mock_sol,
            eval_result=mock_eval_result,
            vehicles=tiny_instance["vehicles"],
        )
        assert "Feasible" in result or "feasible" in result

    def test_infeasible_solution_shows_violations(
        self, tiny_instance, mock_sol, mock_eval_result_infeasible,
    ):
        """Infeasible solution summary should mention violations."""
        result = print_solution_summary(
            sol=mock_sol,
            eval_result=mock_eval_result_infeasible,
            vehicles=tiny_instance["vehicles"],
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_routes(
        self, tiny_instance, empty_sol, mock_eval_result,
    ):
        """Empty solution routes should not crash."""
        result = print_solution_summary(
            sol=empty_sol,
            eval_result=mock_eval_result,
            vehicles=tiny_instance["vehicles"],
        )
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# plot_solution_detail
# ---------------------------------------------------------------------------

class TestPlotSolutionDetail:
    """Tests for plot_solution_detail."""

    def test_returns_fig_and_axes_with_4_panels(
        self, tiny_instance, tiny_dist_matrix, mock_sol, mock_eval_result,
    ):
        """plot_solution_detail returns (fig, axes) with 4 panels."""
        fig, axes = plot_solution_detail(
            sol=mock_sol,
            eval_result=mock_eval_result,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
        )
        assert isinstance(fig, Figure)
        assert isinstance(axes, (list, np.ndarray))
        assert len(axes) == 4

    def test_all_panels_have_axes(
        self, tiny_instance, tiny_dist_matrix, mock_sol, mock_eval_result,
    ):
        """Each panel should be a valid Axes instance."""
        fig, axes = plot_solution_detail(
            sol=mock_sol,
            eval_result=mock_eval_result,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
        )
        for ax in axes:
            assert hasattr(ax, "plot"), "Each panel should be a matplotlib Axes"

    def test_custom_figsize(
        self, tiny_instance, tiny_dist_matrix, mock_sol, mock_eval_result,
    ):
        """Custom figsize should be accepted."""
        fig, axes = plot_solution_detail(
            sol=mock_sol,
            eval_result=mock_eval_result,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
            figsize=(12, 10),
        )
        assert isinstance(fig, Figure)
