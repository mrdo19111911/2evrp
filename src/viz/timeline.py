"""Gantt chart: vehicle timelines."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.data.constants import ACT_PAD, ACT_DELIVER, ACT_RELOAD, COL_TW_OPEN, COL_TW_CLOSE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_vehicle_labels(sol):
    """Build ordered labels: trucks first, then bikes."""
    n_trucks = sol.get("n_trucks", len(sol["truck_stops"]))
    n_bikes = sol.get("n_bikes", len(sol["bike_stops"]))
    labels = [f"Truck {i}" for i in range(n_trucks)]
    labels += [f"Bike {i}" for i in range(n_bikes)]
    return labels


def _draw_gantt_bars(ax, states, y_pos, color, label_prefix):
    """Draw horizontal bars for one vehicle's timeline segments."""
    if len(states) == 0:
        return
    for i, row in enumerate(states):
        cid = int(row[0])
        action = int(row[1])
        # Use time (col 3) as start, assume service duration ~ 5 min
        t_start = row[3]
        duration = max(5.0, row[4] if len(row) > 4 else 5.0)

        if action == ACT_RELOAD:
            bar_color = 'gold'
        else:
            bar_color = color

        ax.barh(y_pos, duration, left=t_start, height=0.6,
                color=bar_color, edgecolor='black', linewidth=0.5,
                alpha=0.8)
        ax.text(t_start + duration / 2, y_pos, f"C{cid}",
                ha='center', va='center', fontsize=6)


def _draw_sync_lines(ax, sol, truck_states, bike_states, n_trucks):
    """Draw vertical sync lines connecting truck-bike at satellite reload events."""
    sats = sol.get("satellites", np.empty((0, 5)))
    if len(sats) == 0:
        return
    for sat in sats:
        cid = int(sat[0])
        bid = int(sat[1])
        tid = int(sat[2])
        t = sat[4]
        truck_y = tid
        bike_y = n_trucks + bid
        ax.plot([t, t], [truck_y, bike_y], '-', color='purple',
                linewidth=1.5, alpha=0.6, zorder=5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_timeline(sol, truck_states, bike_states, customers, vehicles,
                  show_sync=True, show_tw=True, figsize=(16, 8)):
    """Gantt chart: each vehicle = 1 row, time horizontal. Returns (fig, ax)."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    labels = _build_vehicle_labels(sol)
    n_trucks = sol.get("n_trucks", len(sol["truck_stops"]))
    n_bikes = sol.get("n_bikes", len(sol["bike_stops"]))
    n_total = n_trucks + n_bikes

    # Draw truck bars
    for tid in range(n_trucks):
        states = truck_states[tid] if tid < len(truck_states) else np.empty((0, 5))
        _draw_gantt_bars(ax, states, y_pos=tid, color='steelblue',
                         label_prefix="Truck")

    # Draw bike bars
    for bid in range(n_bikes):
        states = bike_states[bid] if bid < len(bike_states) else np.empty((0, 5))
        _draw_gantt_bars(ax, states, y_pos=n_trucks + bid, color='mediumseagreen',
                         label_prefix="Bike")

    # Separator line between trucks and bikes
    if n_trucks > 0 and n_bikes > 0:
        ax.axhline(y=n_trucks - 0.5, color='black', linewidth=1,
                   linestyle='--', alpha=0.5)

    # Sync lines
    if show_sync:
        _draw_sync_lines(ax, sol, truck_states, bike_states, n_trucks)

    # Time window brackets
    if show_tw:
        _draw_tw_brackets(ax, sol, truck_states, bike_states, customers,
                          n_trucks)

    ax.set_yticks(range(n_total))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (min)")
    ax.set_title("2E-VRP Vehicle Timeline (Gantt)")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig, ax


def _draw_tw_brackets(ax, sol, truck_states, bike_states, customers, n_trucks):
    """Draw time window markers on deliver segments."""
    all_states = []
    for tid, ts in enumerate(truck_states):
        if len(ts) > 0:
            for row in ts:
                all_states.append((tid, row))
    for bid, bs in enumerate(bike_states):
        if len(bs) > 0:
            for row in bs:
                all_states.append((n_trucks + bid, row))

    for y_pos, row in all_states:
        cid = int(row[0])
        if cid < 0 or cid >= len(customers):
            continue
        t_arrive = row[3]
        tw_open = customers[cid, COL_TW_OPEN]
        tw_close = customers[cid, COL_TW_CLOSE]
        color = 'green' if tw_open <= t_arrive <= tw_close else 'red'
        ax.plot([tw_open, tw_close], [y_pos + 0.35, y_pos + 0.35],
                '-', color=color, linewidth=2, alpha=0.4)


def update_timeline(fig, ax, sol, truck_states, bike_states, **kwargs):
    """Update for live dashboard."""
    ax.clear()
    customers = kwargs.get("customers", np.empty((0, 6)))
    vehicles = kwargs.get("vehicles", np.empty((0, 4)))

    labels = _build_vehicle_labels(sol)
    n_trucks = sol.get("n_trucks", len(sol["truck_stops"]))
    n_bikes = sol.get("n_bikes", len(sol["bike_stops"]))

    for tid in range(n_trucks):
        states = truck_states[tid] if tid < len(truck_states) else np.empty((0, 5))
        _draw_gantt_bars(ax, states, y_pos=tid, color='steelblue',
                         label_prefix="Truck")

    for bid in range(n_bikes):
        states = bike_states[bid] if bid < len(bike_states) else np.empty((0, 5))
        _draw_gantt_bars(ax, states, y_pos=n_trucks + bid, color='mediumseagreen',
                         label_prefix="Bike")

    ax.set_yticks(range(n_trucks + n_bikes))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (min)")
    ax.set_title("2E-VRP Vehicle Timeline (Gantt)")
    ax.invert_yaxis()
    fig.canvas.draw_idle()


def show_timeline(sol, truck_states, bike_states, customers, vehicles, **kwargs):
    """plot + show."""
    fig, ax = plot_timeline(sol, truck_states, bike_states, customers,
                            vehicles, **kwargs)
    plt.show()
