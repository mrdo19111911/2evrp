"""Pareto front visualization."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_costs_makespans(archive):
    """Extract cost and makespan arrays from archive list."""
    costs = np.array([s["cost"] for s in archive])
    makespans = np.array([s["makespan"] for s in archive])
    return costs, makespans


def _draw_pareto_scatter(ax, archive, highlight_best):
    """Draw scatter + connected Pareto front line."""
    if len(archive) == 0:
        ax.text(0.5, 0.5, "(empty archive)", ha="center", va="center",
                transform=ax.transAxes)
        return

    costs, makespans = _extract_costs_makespans(archive)

    # Sort by cost for connected line
    order = np.argsort(costs)
    sorted_costs = costs[order]
    sorted_makespans = makespans[order]

    # Pareto front line
    ax.plot(sorted_costs, sorted_makespans, "o-", color="steelblue",
            linewidth=1.5, markersize=6, label="Pareto Front", zorder=5)

    # Highlight best cost and best makespan
    if highlight_best and len(archive) > 0:
        best_cost_idx = np.argmin(costs)
        best_ms_idx = np.argmin(makespans)
        ax.plot(costs[best_cost_idx], makespans[best_cost_idx], "*",
                color="blue", markersize=14, zorder=8, label="Best Cost")
        ax.plot(costs[best_ms_idx], makespans[best_ms_idx], "*",
                color="orange", markersize=14, zorder=8, label="Best Makespan")

        # Ideal point
        ax.plot(costs[best_cost_idx], makespans[best_ms_idx], "x",
                color="red", markersize=10, markeredgewidth=2, zorder=7,
                label="Ideal Point")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_pareto_front(archive, highlight_best=True, show_dominated=False,
                      figsize=(8, 6)):
    """Cost vs makespan scatter + Pareto front line. Returns (fig, ax)."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    _draw_pareto_scatter(ax, archive, highlight_best)

    ax.set_xlabel("Cost (VND)")
    ax.set_ylabel("Makespan (min)")
    ax.set_title("Pareto Front: Cost vs Makespan")
    if len(archive) > 0:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def update_pareto_front(fig, ax, archive):
    """Update for live dashboard."""
    ax.clear()
    _draw_pareto_scatter(ax, archive, highlight_best=True)
    ax.set_xlabel("Cost (VND)")
    ax.set_ylabel("Makespan (min)")
    ax.set_title("Pareto Front: Cost vs Makespan")
    if len(archive) > 0:
        ax.legend(fontsize=8)
    fig.canvas.draw_idle()


def show_pareto_front(archive, **kwargs):
    """plot + show."""
    fig, ax = plot_pareto_front(archive, **kwargs)
    plt.show()
