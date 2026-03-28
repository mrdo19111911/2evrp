"""Export: PNG, HTML report, JSON, CSV."""
import matplotlib
matplotlib.use("Agg")

import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from src.viz.dashboard import create_dashboard, save_dashboard


# ---------------------------------------------------------------------------
# JSON helper for numpy types
# ---------------------------------------------------------------------------

def _to_serializable(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_dashboard_png(sol, eval_result, log, archive, customers, vehicles,
                         depot, dist_matrix, truck_states, bike_states,
                         filepath="dashboard.png", dpi=150):
    """Create dashboard + save PNG. Returns filepath."""
    fig, axes = create_dashboard(sol, eval_result, log, archive, customers,
                                  vehicles, depot, dist_matrix,
                                  truck_states, bike_states)
    save_dashboard(fig, filepath=filepath, dpi=dpi)
    plt.close(fig)
    return filepath


def export_html_report(sol, eval_result, log, archive, customers, vehicles,
                       depot, dist_matrix, truck_states, bike_states,
                       filepath="report.html"):
    """Generate a simple HTML report. Returns filepath."""
    feasible = eval_result.get("feasible", False)
    cost = eval_result.get("cost", 0)
    makespan = eval_result.get("makespan", 0)
    penalty = eval_result.get("penalty", 0)

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>2E-VRP Solution Report</title></head>
<body>
<h1>2E-VRP Solution Report</h1>
<div id="summary">
  <p>Feasible: {'Yes' if feasible else 'No'} | Cost: {cost:,.0f} VND | Makespan: {makespan:.1f} min | Penalty: {penalty:,.0f}</p>
</div>
<h2>Evaluation</h2>
<table border="1" cellpadding="4">
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Fitness</td><td>{eval_result.get('fitness', 0):.1f}</td></tr>
  <tr><td>Cost</td><td>{cost:,.0f}</td></tr>
  <tr><td>Makespan</td><td>{makespan:.1f}</td></tr>
  <tr><td>Penalty</td><td>{penalty:,.0f}</td></tr>
  <tr><td>Feasible</td><td>{'Yes' if feasible else 'No'}</td></tr>
</table>
<h2>Pareto Archive ({len(archive)} solutions)</h2>
<table border="1" cellpadding="4">
  <tr><th>Cost</th><th>Makespan</th></tr>
"""
    for entry in archive:
        html += f"  <tr><td>{entry.get('cost', 0):,.0f}</td><td>{entry.get('makespan', 0):.1f}</td></tr>\n"

    html += """</table>
</body>
</html>"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    return filepath


def export_solution_json(sol, eval_result, filepath="solution.json"):
    """Save solution + eval as JSON. Returns filepath."""
    # Build solution dict with serializable arrays
    solution_data = {}
    for key in ("truck_stops", "truck_actions", "bike_stops", "bike_actions",
                "satellites"):
        if key in sol:
            solution_data[key] = _to_serializable(sol[key])

    # Build evaluation dict
    evaluation_data = {}
    for key in ("fitness", "cost", "makespan", "penalty", "feasible",
                "tw_violations", "cap_violations", "sync_violations",
                "total_distance", "vehicles_used"):
        if key in eval_result:
            evaluation_data[key] = _to_serializable(eval_result[key])

    output = {
        "solution": solution_data,
        "evaluation": evaluation_data,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=_to_serializable)
    return filepath


def export_pareto_csv(archive, filepath="pareto.csv"):
    """Save Pareto front as CSV. Returns filepath."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cost", "makespan"])
        for entry in archive:
            writer.writerow([entry.get("cost", 0), entry.get("makespan", 0)])
    return filepath
