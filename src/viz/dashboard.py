"""Full 6-panel static dashboard."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.data.constants import COL_X, COL_Y, ACT_PAD, ACT_RELOAD


# ---------------------------------------------------------------------------
# Helpers — draw into existing axes
# ---------------------------------------------------------------------------

def _draw_route_map_panel(ax, sol, customers, depot):
    """Simplified route map on a dashboard panel."""

    ax.plot(depot[0], depot[1], "s", color="red", markersize=10, zorder=10)
    ax.scatter(customers[:, COL_X], customers[:, COL_Y], s=30,
               c="steelblue", edgecolors="black", linewidths=0.3, zorder=5)

    n_trucks = sol.get("n_trucks", len(sol["truck_stops"]))
    for tid in range(n_trucks):
        stops = sol["truck_stops"][tid]
        valid = stops[stops != ACT_PAD]
        if len(valid) == 0:
            continue
        xs = [depot[0]] + [customers[s, COL_X] for s in valid] + [depot[0]]
        ys = [depot[1]] + [customers[s, COL_Y] for s in valid] + [depot[1]]
        ax.plot(xs, ys, "-", linewidth=2, alpha=0.7, label=f"T{tid}")

    n_bikes = sol.get("n_bikes", len(sol["bike_stops"]))
    for bid in range(n_bikes):
        stops = sol["bike_stops"][bid]
        valid = stops[stops != ACT_PAD]
        if len(valid) == 0:
            continue
        xs = [depot[0]] + [customers[s, COL_X] for s in valid] + [depot[0]]
        ys = [depot[1]] + [customers[s, COL_Y] for s in valid] + [depot[1]]
        ax.plot(xs, ys, "--", linewidth=1, alpha=0.7, label=f"B{bid}")

    ax.set_title("Route Map")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=6)


def _draw_convergence_panel(ax, log):
    """Fitness convergence on a dashboard panel."""
    iters = log.get("iterations", np.array([]))
    fitness = log.get("fitness", np.array([]))
    best = log.get("best_fitness", np.array([]))
    if len(iters) == 0:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes)
    else:
        ax.plot(iters, fitness, color="gray", alpha=0.4, linewidth=0.5)
        ax.plot(iters, best, color="red", linewidth=1.5)
    ax.set_title("Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")


def _draw_timeline_panel(ax, sol, truck_states, bike_states):
    """Simplified Gantt on a dashboard panel."""
    n_trucks = sol.get("n_trucks", len(sol["truck_stops"]))
    n_bikes = sol.get("n_bikes", len(sol["bike_stops"]))

    for tid in range(n_trucks):
        states = truck_states[tid] if tid < len(truck_states) else np.empty((0, 5))
        for row in states:
            t_start = row[3]
            dur = max(5.0, row[4] if len(row) > 4 else 5.0)
            color = "gold" if int(row[1]) == ACT_RELOAD else "steelblue"
            ax.barh(tid, dur, left=t_start, height=0.6, color=color,
                    edgecolor="black", linewidth=0.3, alpha=0.8)

    for bid in range(n_bikes):
        states = bike_states[bid] if bid < len(bike_states) else np.empty((0, 5))
        for row in states:
            t_start = row[3]
            dur = max(5.0, row[4] if len(row) > 4 else 5.0)
            color = "gold" if int(row[1]) == ACT_RELOAD else "mediumseagreen"
            ax.barh(n_trucks + bid, dur, left=t_start, height=0.6, color=color,
                    edgecolor="black", linewidth=0.3, alpha=0.8)

    labels = [f"T{i}" for i in range(n_trucks)] + [f"B{i}" for i in range(n_bikes)]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Timeline")
    ax.set_xlabel("Time (min)")
    ax.invert_yaxis()


def _draw_pareto_panel(ax, archive):
    """Pareto scatter on a dashboard panel."""
    if len(archive) == 0:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                transform=ax.transAxes)
    else:
        costs = [s["cost"] for s in archive]
        makespans = [s["makespan"] for s in archive]
        order = np.argsort(costs)
        ax.plot(np.array(costs)[order], np.array(makespans)[order], "o-",
                color="steelblue", markersize=4)
    ax.set_title("Pareto Front")
    ax.set_xlabel("Cost")
    ax.set_ylabel("Makespan")


def _draw_operator_panel(ax, log):
    """Operator weights stacked area on a dashboard panel."""
    dw = log.get("destroy_weights", np.empty((0, 0)))
    d_names = log.get("destroy_names", [])
    if dw.size > 0 and len(d_names) > 0:
        x = np.arange(len(dw))
        ax.stackplot(x, dw.T, labels=d_names, alpha=0.8)
        ax.legend(fontsize=5, loc="upper right")
    else:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("Operator Weights")
    ax.set_xlabel("Segment")


def _draw_violations_panel(ax, eval_result):
    """Violations summary text on a dashboard panel."""
    ax.axis("off")
    feasible = eval_result.get("feasible", False)
    lines = [
        f"Feasible: {'Yes' if feasible else 'No'}",
        f"Cost:     {eval_result.get('cost', 0):,.0f} VND",
        f"Makespan: {eval_result.get('makespan', 0):.1f} min",
        f"Penalty:  {eval_result.get('penalty', 0):,.0f}",
        f"TW viol:  {eval_result.get('tw_violations', 0)}",
        f"Cap viol: {eval_result.get('cap_violations', 0)}",
        f"Sync viol:{eval_result.get('sync_violations', 0)}",
    ]
    ax.text(0.1, 0.9, "\n".join(lines), transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title("Summary + Violations")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_dashboard(sol, eval_result, log, archive, customers, vehicles,
                     depot, dist_matrix, truck_states, bike_states,
                     figsize=(20, 16)):
    """6-panel dashboard. Returns (fig, axes) where axes is a list of 6 Axes."""
    fig = plt.figure(figsize=figsize)

    # 3 rows x 2 cols grid
    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6)
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    _draw_route_map_panel(ax1, sol, customers, depot)
    _draw_convergence_panel(ax2, log)
    _draw_timeline_panel(ax3, sol, truck_states, bike_states)
    _draw_pareto_panel(ax4, archive)
    _draw_operator_panel(ax5, log)
    _draw_violations_panel(ax6, eval_result)

    fig.suptitle("2E-VRP Dashboard", fontsize=16)
    fig.tight_layout()
    return fig, axes


def save_dashboard(fig, filepath="dashboard.png", dpi=150):
    """Save dashboard as image."""
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")


def show_dashboard(sol, eval_result, log, archive, customers, vehicles,
                   depot, dist_matrix, truck_states, bike_states):
    """create + show."""
    fig, axes = create_dashboard(sol, eval_result, log, archive, customers,
                                  vehicles, depot, dist_matrix,
                                  truck_states, bike_states)
    plt.show()
