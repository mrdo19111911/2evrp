"""Tests for src/viz/export.py — PNG, JSON, CSV export."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")

from src.viz.export import (
    export_dashboard_png,
    export_solution_json,
    export_pareto_csv,
)


# ---------------------------------------------------------------------------
# export_solution_json
# ---------------------------------------------------------------------------

class TestExportSolutionJson:
    """Tests for export_solution_json."""

    def test_creates_valid_json_file(self, tmp_path, mock_sol, mock_eval_result):
        """export_solution_json writes a valid JSON file."""
        filepath = str(tmp_path / "solution.json")
        result = export_solution_json(
            sol=mock_sol,
            eval_result=mock_eval_result,
            filepath=filepath,
        )
        # Return value is the filepath
        assert isinstance(result, str)
        assert os.path.isfile(filepath)
        assert os.path.getsize(filepath) > 0

        # Must be valid JSON
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_json_contains_solution_key(self, tmp_path, mock_sol, mock_eval_result):
        """JSON output contains a 'solution' key."""
        filepath = str(tmp_path / "solution.json")
        export_solution_json(
            sol=mock_sol,
            eval_result=mock_eval_result,
            filepath=filepath,
        )
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "solution" in data

    def test_json_contains_evaluation_key(
        self, tmp_path, mock_sol, mock_eval_result,
    ):
        """JSON output contains an 'evaluation' key."""
        filepath = str(tmp_path / "solution.json")
        export_solution_json(
            sol=mock_sol,
            eval_result=mock_eval_result,
            filepath=filepath,
        )
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "evaluation" in data

    def test_json_evaluation_has_fitness(
        self, tmp_path, mock_sol, mock_eval_result,
    ):
        """JSON evaluation section contains 'fitness'."""
        filepath = str(tmp_path / "solution.json")
        export_solution_json(
            sol=mock_sol,
            eval_result=mock_eval_result,
            filepath=filepath,
        )
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "fitness" in data["evaluation"]


# ---------------------------------------------------------------------------
# export_pareto_csv
# ---------------------------------------------------------------------------

class TestExportParetoCsv:
    """Tests for export_pareto_csv."""

    def test_creates_valid_csv_file(self, tmp_path, mock_archive_5):
        """export_pareto_csv writes a valid CSV file."""
        filepath = str(tmp_path / "pareto.csv")
        result = export_pareto_csv(
            archive=mock_archive_5,
            filepath=filepath,
        )
        assert isinstance(result, str)
        assert os.path.isfile(filepath)
        assert os.path.getsize(filepath) > 0

        # Must be readable as CSV
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Header + 5 data rows
        assert len(rows) >= 6  # at least header + 5 solutions

    def test_csv_has_cost_and_makespan_columns(self, tmp_path, mock_archive_5):
        """CSV should have 'cost' and 'makespan' columns."""
        filepath = str(tmp_path / "pareto.csv")
        export_pareto_csv(archive=mock_archive_5, filepath=filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
        assert fieldnames is not None
        lower_fields = [f.lower() for f in fieldnames]
        assert "cost" in lower_fields
        assert "makespan" in lower_fields

    def test_csv_row_count_matches_archive(self, tmp_path, mock_archive_5):
        """Number of data rows should match archive size."""
        filepath = str(tmp_path / "pareto.csv")
        export_pareto_csv(archive=mock_archive_5, filepath=filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data_rows = list(reader)
        assert len(data_rows) == 5

    def test_empty_archive_creates_file(self, tmp_path, empty_archive):
        """Empty archive should still produce a (possibly header-only) CSV."""
        filepath = str(tmp_path / "pareto_empty.csv")
        export_pareto_csv(archive=empty_archive, filepath=filepath)
        assert os.path.isfile(filepath)


# ---------------------------------------------------------------------------
# export_dashboard_png
# ---------------------------------------------------------------------------

class TestExportDashboardPng:
    """Tests for export_dashboard_png."""

    def test_creates_png_file(
        self, tmp_path, tiny_instance, tiny_dist_matrix, mock_sol,
        mock_eval_result, mock_log_100, mock_archive_5,
        mock_vehicle_states,
    ):
        """export_dashboard_png writes a PNG file to disk."""
        filepath = str(tmp_path / "dashboard.png")
        result = export_dashboard_png(
            sol=mock_sol,
            eval_result=mock_eval_result,
            log=mock_log_100,
            archive=mock_archive_5,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
            states=mock_vehicle_states,
            filepath=filepath,
        )
        assert isinstance(result, str)
        assert os.path.isfile(filepath)
        assert os.path.getsize(filepath) > 0

    def test_png_file_starts_with_png_signature(
        self, tmp_path, tiny_instance, tiny_dist_matrix, mock_sol,
        mock_eval_result, mock_log_100, mock_archive_5,
        mock_vehicle_states,
    ):
        """PNG file should start with the standard PNG signature bytes."""
        filepath = str(tmp_path / "dashboard_sig.png")
        export_dashboard_png(
            sol=mock_sol,
            eval_result=mock_eval_result,
            log=mock_log_100,
            archive=mock_archive_5,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
            states=mock_vehicle_states,
            filepath=filepath,
        )
        with open(filepath, "rb") as f:
            header = f.read(8)
        # PNG magic bytes
        assert header[:4] == b"\x89PNG"

    def test_custom_dpi(
        self, tmp_path, tiny_instance, tiny_dist_matrix, mock_sol,
        mock_eval_result, mock_log_100, mock_archive_5,
        mock_vehicle_states,
    ):
        """Custom dpi parameter should be accepted."""
        filepath = str(tmp_path / "dashboard_lodpi.png")
        export_dashboard_png(
            sol=mock_sol,
            eval_result=mock_eval_result,
            log=mock_log_100,
            archive=mock_archive_5,
            locations=tiny_instance["locations"],
            orders=tiny_instance["orders"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
            states=mock_vehicle_states,
            filepath=filepath,
            dpi=72,
        )
        assert os.path.isfile(filepath)
