"""Tests for src/viz/dashboard.py — full 6-panel static dashboard."""
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from src.viz.dashboard import create_dashboard, save_dashboard


# ---------------------------------------------------------------------------
# create_dashboard
# ---------------------------------------------------------------------------

class TestCreateDashboard:
    """Tests for create_dashboard."""

    def test_returns_fig_and_axes_with_6_panels(
        self, tiny_instance, tiny_dist_matrix, mock_sol, mock_eval_result,
        mock_log_100, mock_archive_5, mock_vehicle_states,
    ):
        """create_dashboard returns (fig, axes) with 6 panels."""
        fig, axes = create_dashboard(
            sol=mock_sol,
            eval_result=mock_eval_result,
            log=mock_log_100,
            archive=mock_archive_5,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
            states=mock_vehicle_states,
        )
        assert isinstance(fig, Figure)
        assert isinstance(axes, (list, np.ndarray))
        assert len(axes) == 6

    def test_all_panels_are_axes(
        self, tiny_instance, tiny_dist_matrix, mock_sol, mock_eval_result,
        mock_log_100, mock_archive_5, mock_vehicle_states,
    ):
        """Each of the 6 panels should be a valid Axes."""
        fig, axes = create_dashboard(
            sol=mock_sol,
            eval_result=mock_eval_result,
            log=mock_log_100,
            archive=mock_archive_5,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
            states=mock_vehicle_states,
        )
        for i, ax in enumerate(axes):
            assert hasattr(ax, "plot"), f"Panel {i} should be a matplotlib Axes"

    def test_custom_figsize(
        self, tiny_instance, tiny_dist_matrix, mock_sol, mock_eval_result,
        mock_log_100, mock_archive_5, mock_vehicle_states,
    ):
        """Custom figsize should be accepted."""
        fig, axes = create_dashboard(
            sol=mock_sol,
            eval_result=mock_eval_result,
            log=mock_log_100,
            archive=mock_archive_5,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
            states=mock_vehicle_states,
            figsize=(16, 12),
        )
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# save_dashboard
# ---------------------------------------------------------------------------

class TestSaveDashboard:
    """Tests for save_dashboard."""

    def test_save_creates_png_file(
        self, tmp_path, tiny_instance, tiny_dist_matrix, mock_sol,
        mock_eval_result, mock_log_100, mock_archive_5,
        mock_vehicle_states,
    ):
        """save_dashboard writes a PNG file to disk."""
        fig, axes = create_dashboard(
            sol=mock_sol,
            eval_result=mock_eval_result,
            log=mock_log_100,
            archive=mock_archive_5,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
            states=mock_vehicle_states,
        )
        filepath = str(tmp_path / "test_dashboard.png")
        save_dashboard(fig, filepath=filepath)
        assert os.path.isfile(filepath)
        assert os.path.getsize(filepath) > 0

    def test_save_custom_dpi(
        self, tmp_path, tiny_instance, tiny_dist_matrix, mock_sol,
        mock_eval_result, mock_log_100, mock_archive_5,
        mock_vehicle_states,
    ):
        """save_dashboard with custom dpi should not crash."""
        fig, axes = create_dashboard(
            sol=mock_sol,
            eval_result=mock_eval_result,
            log=mock_log_100,
            archive=mock_archive_5,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
            states=mock_vehicle_states,
        )
        filepath = str(tmp_path / "test_dashboard_hi.png")
        save_dashboard(fig, filepath=filepath, dpi=72)
        assert os.path.isfile(filepath)
