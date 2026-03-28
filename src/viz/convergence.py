"""Fitness convergence plots."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_fitness_panel(ax, log):
    """Panel 1: fitness + best fitness over iterations."""
    iters = log.get("iterations", np.array([]))
    fitness = log.get("fitness", np.array([]))
    best = log.get("best_fitness", np.array([]))
    if len(iters) == 0:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Fitness")
        return
    ax.plot(iters, fitness, color="gray", alpha=0.4, linewidth=0.5, label="All")
    ax.plot(iters, best, color="red", linewidth=1.5, label="Best")
    ax.set_title("Fitness + Best Fitness")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.legend(fontsize=7)


def _draw_cost_makespan_panel(ax, log):
    """Panel 2: cost + makespan (dual Y-axis)."""
    iters = log.get("iterations", np.array([]))
    cost = log.get("cost", np.array([]))
    makespan = log.get("makespan", np.array([]))
    if len(iters) == 0 or len(cost) == 0:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Cost + Makespan")
        return
    n = min(len(iters), len(cost), len(makespan))
    ax.plot(iters[:n], cost[:n], color="steelblue", linewidth=1, label="Cost")
    ax.set_ylabel("Cost (VND)", color="steelblue")
    ax.set_xlabel("Iteration")
    ax2 = ax.twinx()
    ax2.plot(iters[:n], makespan[:n], color="orange", linewidth=1, label="Makespan")
    ax2.set_ylabel("Makespan (min)", color="orange")
    ax.set_title("Cost + Makespan")


def _draw_penalty_panel(ax, log):
    """Panel 3: penalty + feasibility rate."""
    iters = log.get("iterations", np.array([]))
    penalty = log.get("penalty", np.array([]))
    feasible = log.get("feasible", np.array([]))
    if len(iters) == 0 or len(penalty) == 0:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Penalty + Feasibility")
        return
    n = min(len(iters), len(penalty), len(feasible))
    ax.plot(iters[:n], penalty[:n], color="red", linewidth=1, label="Penalty")
    ax.set_ylabel("Penalty", color="red")
    ax.set_xlabel("Iteration")
    ax2 = ax.twinx()
    window = min(100, n)
    if window > 0:
        kernel = np.ones(window) / window
        feas_rate = np.convolve(feasible[:n], kernel, mode="same") * 100
        ax2.plot(iters[:n], feas_rate, color="green", linewidth=1, label="Feasibility %")
        ax2.set_ylabel("Feasibility %", color="green")
        ax2.set_ylim(-5, 105)
    ax.set_title("Penalty + Feasibility")


def _draw_temperature_panel(ax, log):
    """Panel 4: temperature + w3."""
    iters = log.get("iterations", np.array([]))
    temp = log.get("temperature", np.array([]))
    w3 = log.get("w3", np.array([]))
    if len(iters) == 0 or len(temp) == 0:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Temperature + w3")
        return
    n = min(len(iters), len(temp), len(w3))
    ax.plot(iters[:n], temp[:n], color="orange", linewidth=1, label="Temperature")
    ax.set_ylabel("Temperature", color="orange")
    ax.set_xlabel("Iteration")
    if n > 0 and np.all(temp[:n] > 0):
        ax.set_yscale("log")
    ax2 = ax.twinx()
    ax2.plot(iters[:n], w3[:n], color="purple", linewidth=1, label="w3")
    ax2.set_ylabel("w3", color="purple")
    ax.set_title("Temperature + w3")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_convergence(log, show_cost=True, show_makespan=True, show_penalty=True,
                     show_temperature=True, show_feasibility=True, figsize=(14, 10)):
    """Multi-panel convergence: fitness, cost, penalty, temperature. Returns (fig, axes)."""
    # Always show fitness panel; conditionally show others
    panels = [("fitness", _draw_fitness_panel)]
    if show_cost or show_makespan:
        panels.append(("cost_makespan", _draw_cost_makespan_panel))
    if show_penalty or show_feasibility:
        panels.append(("penalty", _draw_penalty_panel))
    if show_temperature:
        panels.append(("temperature", _draw_temperature_panel))

    n_panels = max(len(panels), 2)
    fig, ax_arr = plt.subplots(n_panels, 1, figsize=figsize, squeeze=False)
    axes = [ax_arr[i, 0] for i in range(n_panels)]

    for idx, (name, draw_fn) in enumerate(panels):
        draw_fn(axes[idx], log)

    # Hide unused axes
    for idx in range(len(panels), n_panels):
        axes[idx].set_visible(False)

    fig.suptitle("ALNS Convergence", fontsize=14)
    fig.tight_layout()
    return fig, axes


def update_convergence(fig, axes, log):
    """Append data for live dashboard."""
    # Redraw all panels with new log data
    draw_fns = [_draw_fitness_panel, _draw_cost_makespan_panel,
                _draw_penalty_panel, _draw_temperature_panel]
    for idx, ax in enumerate(axes):
        if not ax.get_visible():
            continue
        ax.clear()
        if idx < len(draw_fns):
            draw_fns[idx](ax, log)
    fig.canvas.draw_idle()


def show_convergence(log, **kwargs):
    """plot + show."""
    fig, axes = plot_convergence(log, **kwargs)
    plt.show()
