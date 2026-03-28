"""Detailed solution inspection -- text + visual."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.data.constants import (
    COL_DEMAND, COL_X, COL_Y, ACT_PAD, ACT_DELIVER, ACT_RELOAD,
    VEH_TRUCK, VEH_BIKE, VCOL_CAPACITY,
)
from src.data.cost import TRUCK_CAPACITY, BIKE_CAPACITY
from src.viz.draw_helpers import (
    draw_load_profile, draw_stats_table, draw_route_map_on_ax,
)


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def _format_route_line(stops, actions, customers, vtype, vid):
    """Format one vehicle's route as a text line."""
    vname = f"Truck {vid}" if vtype == VEH_TRUCK else f"Bike {vid}"
    mask = stops != ACT_PAD
    valid_stops = stops[mask]
    valid_actions = actions[mask]
    if len(valid_stops) == 0:
        return f"  {vname}: (empty)"
    parts = ["depot"]
    for s, a in zip(valid_stops, valid_actions):
        d = customers[s, COL_DEMAND]
        tag = "D" if a == ACT_DELIVER else "R"
        parts.append(f"C{s}({tag},{d:.0f}kg)")
    parts.append("depot")
    return f"  {vname}: {' -> '.join(parts)}"


def print_solution_summary(sol, eval_result, customers, vehicles):
    """Console text report: routes, costs, violations. Returns formatted string."""
    lines = []
    sep = "=" * 59
    lines.append(sep)
    lines.append("2E-VRP SOLUTION SUMMARY")
    lines.append(sep)

    feasible = eval_result.get("feasible", False)
    fmark = "Y" if feasible else "X"
    lines.append(f"Feasible: {fmark}")
    lines.append(f"Total Cost: {eval_result.get('cost', 0):,.0f} VND")
    lines.append(f"Makespan:   {eval_result.get('makespan', 0):.1f} min")
    lines.append(f"Fitness:    {eval_result.get('fitness', 0):.1f}")
    lines.append("")

    # Trucks
    n_trucks = sol.get("n_trucks", len(sol["truck_stops"]))
    lines.append("-- TRUCKS " + "-" * 49)
    for tid in range(n_trucks):
        lines.append(_format_route_line(
            sol["truck_stops"][tid], sol["truck_actions"][tid],
            customers, VEH_TRUCK, tid))
    lines.append("")

    # Bikes
    n_bikes = sol.get("n_bikes", len(sol["bike_stops"]))
    lines.append("-- BIKES " + "-" * 50)
    for bid in range(n_bikes):
        lines.append(_format_route_line(
            sol["bike_stops"][bid], sol["bike_actions"][bid],
            customers, VEH_BIKE, bid))
    lines.append("")

    # Violations
    lines.append("-- VIOLATIONS " + "-" * 45)
    if feasible:
        lines.append("  (none)")
    else:
        tw_v = eval_result.get("tw_violations", 0)
        cap_v = eval_result.get("cap_violations", 0)
        sync_v = eval_result.get("sync_violations", 0)
        if tw_v:
            lines.append(f"  [TW]   {tw_v} time-window violation(s)")
        if cap_v:
            lines.append(f"  [CAP]  {cap_v} capacity violation(s)")
        if sync_v:
            lines.append(f"  [SYNC] {sync_v} sync violation(s)")
        penalty = eval_result.get("penalty", 0)
        lines.append(f"  Total penalty: {penalty:,.0f}")
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4-panel detail plot
# ---------------------------------------------------------------------------

def plot_solution_detail(sol, eval_result, customers, vehicles, depot, dist_matrix,
                         figsize=(16, 12)):
    """4-panel: route map + truck loads + bike loads + stats. Returns (fig, axes)."""
    fig = plt.figure(figsize=figsize)
    axes = []

    ax1 = fig.add_subplot(2, 2, 1)
    draw_route_map_on_ax(ax1, sol, customers, depot)
    axes.append(ax1)

    ax2 = fig.add_subplot(2, 2, 2)
    draw_load_profile(ax2, sol, customers, VEH_TRUCK, TRUCK_CAPACITY)
    axes.append(ax2)

    ax3 = fig.add_subplot(2, 2, 3)
    draw_load_profile(ax3, sol, customers, VEH_BIKE, BIKE_CAPACITY)
    axes.append(ax3)

    ax4 = fig.add_subplot(2, 2, 4)
    draw_stats_table(ax4, eval_result)
    axes.append(ax4)

    fig.suptitle("2E-VRP Solution Detail", fontsize=14)
    fig.tight_layout()
    return fig, axes


def show_solution_detail(sol, eval_result, customers, vehicles, depot, dist_matrix):
    """plot + show."""
    fig, axes = plot_solution_detail(sol, eval_result, customers, vehicles,
                                     depot, dist_matrix)
    plt.show()
