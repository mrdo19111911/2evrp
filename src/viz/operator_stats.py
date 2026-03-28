"""Operator weight and performance analysis."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_stacked_area(ax, weights, names, title):
    """Draw stacked area chart for operator weights."""
    if len(weights) == 0 or len(names) == 0:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return
    n_snapshots, n_ops = weights.shape
    x = np.arange(n_snapshots)
    ax.stackplot(x, weights.T, labels=names, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper right")


def _draw_performance_bars(ax, scores, counts, names, title):
    """Draw grouped bar chart: score, count, avg performance."""
    if len(names) == 0:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return
    score_vals = np.array([float(scores.get(n, 0)) for n in names])
    count_vals = np.array([float(counts.get(n, 0)) for n in names])

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width / 2, score_vals, width, color="steelblue", label="Score")
    ax.bar(x + width / 2, count_vals, width, color="gray", alpha=0.7, label="Count")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_title(title)
    ax.set_ylabel("Score / Count")
    ax.legend(fontsize=7)

    # Secondary axis: average performance
    ax2 = ax.twinx()
    safe_counts = np.where(count_vals > 0, count_vals, 1)
    avg_perf = score_vals / safe_counts
    ax2.plot(x, avg_perf, "o-", color="red", linewidth=1.5, label="Avg Perf")
    ax2.set_ylabel("Avg Performance", color="red")
    ax2.legend(fontsize=7, loc="upper right")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_operator_weights(log, figsize=(12, 6)):
    """Stacked area: weights over time. Returns (fig, axes)."""
    dw = log.get("destroy_weights", np.empty((0, 0)))
    rw = log.get("repair_weights", np.empty((0, 0)))
    d_names = log.get("destroy_names", [])
    r_names = log.get("repair_names", [])

    fig, ax_arr = plt.subplots(1, 2, figsize=figsize)
    axes = list(ax_arr)

    _draw_stacked_area(axes[0], dw, d_names, "Destroy Operator Weights")
    _draw_stacked_area(axes[1], rw, r_names, "Repair Operator Weights")

    fig.suptitle("Operator Weights Over Time", fontsize=13)
    fig.tight_layout()
    return fig, axes


def plot_operator_performance(log, figsize=(12, 8)):
    """Bar chart: score/count per operator. Returns (fig, axes)."""
    d_names = log.get("destroy_names", [])
    r_names = log.get("repair_names", [])
    d_scores = log.get("destroy_scores", {})
    d_counts = log.get("destroy_counts", {})
    r_scores = log.get("repair_scores", {})
    r_counts = log.get("repair_counts", {})

    fig, ax_arr = plt.subplots(2, 1, figsize=figsize)
    axes = list(ax_arr)

    _draw_performance_bars(axes[0], d_scores, d_counts, d_names,
                           "Destroy Operator Performance")
    _draw_performance_bars(axes[1], r_scores, r_counts, r_names,
                           "Repair Operator Performance")

    fig.suptitle("Operator Performance Analysis", fontsize=13)
    fig.tight_layout()
    return fig, axes


def show_operator_stats(log, **kwargs):
    """plot + show."""
    plot_operator_weights(log, **kwargs)
    plot_operator_performance(log, **kwargs)
    plt.show()
