"""2D route map visualization."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.data.constants import (
    COL_X, COL_Y, COL_DEMAND, ACT_DELIVER, ACT_RELOAD, ACT_PAD,
    VEH_TRUCK, VEH_BIKE,
)
from src.viz.draw_helpers import (
    get_route_stops, draw_depot, draw_customers, draw_vehicle_route,
    draw_satellites, draw_violations,
)

# Color palettes
_TRUCK_COLORS = plt.cm.tab10.colors
_BIKE_COLORS = plt.cm.Pastel1.colors

# Keep old private names as aliases for backward compat within this module
_get_route_stops = get_route_stops
_draw_depot = draw_depot
_draw_customers = draw_customers
_draw_vehicle_route = draw_vehicle_route
_draw_satellites = draw_satellites
_draw_violations = draw_violations


def plot_route_map(sol, customers, depot, dist_matrix, truck_states=None,
                   bike_states=None, highlight_violations=True, show_satellites=True,
                   show_labels=True, show_demand=False, title=None, figsize=(14, 10)):
    """Full 2D map: depot, customers, routes, satellites, violations. Returns (fig, ax)."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    draw_customers(ax, customers, sol, show_labels, show_demand)
    draw_depot(ax, depot)

    _draw_all_truck_routes(ax, sol, customers, depot)
    _draw_all_bike_routes(ax, sol, customers, depot)

    if show_satellites:
        draw_satellites(ax, sol, customers)

    if highlight_violations and (truck_states or bike_states):
        draw_violations(ax, truck_states, bike_states, customers)

    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_title(title or "2E-VRP Route Map")
    ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    return fig, ax


def update_route_map(fig, ax, sol, customers, depot, **kwargs):
    """Clear + redraw for live dashboard."""
    ax.clear()
    dist_matrix = kwargs.pop("dist_matrix", None)

    draw_customers(ax, customers, sol,
                   kwargs.get("show_labels", True),
                   kwargs.get("show_demand", False))
    draw_depot(ax, depot)
    _draw_all_truck_routes(ax, sol, customers, depot)
    _draw_all_bike_routes(ax, sol, customers, depot)

    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(kwargs.get("title", "2E-VRP Route Map"))
    fig.canvas.draw_idle()


def show_route_map(sol, customers, depot, dist_matrix, **kwargs):
    """plot + plt.show()."""
    fig, ax = plot_route_map(sol, customers, depot, dist_matrix, **kwargs)
    plt.show()


def plot_single_route(sol, vtype, vid, customers, depot, dist_matrix,
                      state=None, figsize=(12, 5)):
    """Zoom 1 route: map + load/time profile. Returns (fig, axes)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_map, ax_profile = axes

    stops, actions = get_route_stops(sol, vtype, vid)
    vname = f"Truck {vid}" if vtype == VEH_TRUCK else f"Bike {vid}"

    _draw_single_route_map(ax_map, stops, actions, customers, depot, vtype, vname)
    _draw_single_route_profile(ax_profile, stops, actions, customers, state, vname)

    fig.tight_layout()
    return fig, axes


# ------------------------------------------------------------------
# Internal helpers (thin wrappers to reduce duplication)
# ------------------------------------------------------------------

def _draw_all_truck_routes(ax, sol, customers, depot):
    """Draw all truck routes on ax."""
    n_trucks = sol.get("n_trucks", len(sol["truck_stops"]))
    for tid in range(n_trucks):
        stops, actions = get_route_stops(sol, VEH_TRUCK, tid)
        color = _TRUCK_COLORS[tid % len(_TRUCK_COLORS)]
        draw_vehicle_route(ax, stops, actions, customers, depot,
                           color=color, linestyle='-', linewidth=2.5,
                           label=f"Truck {tid}")


def _draw_all_bike_routes(ax, sol, customers, depot):
    """Draw all bike routes on ax."""
    n_bikes = sol.get("n_bikes", len(sol["bike_stops"]))
    for bid in range(n_bikes):
        stops, actions = get_route_stops(sol, VEH_BIKE, bid)
        color = _BIKE_COLORS[bid % len(_BIKE_COLORS)]
        draw_vehicle_route(ax, stops, actions, customers, depot,
                           color=color, linestyle='--', linewidth=1.2,
                           label=f"Bike {bid}")


def _draw_single_route_map(ax, stops, actions, customers, depot, vtype, vname):
    """Draw the map subplot for a single route."""
    draw_depot(ax, depot)
    if len(stops) > 0:
        color = 'steelblue' if vtype == VEH_TRUCK else 'green'
        draw_vehicle_route(ax, stops, actions, customers, depot,
                           color=color, linestyle='-', linewidth=2,
                           label=vname)
        for idx, (s, a) in enumerate(zip(stops, actions)):
            x, y = customers[s, COL_X], customers[s, COL_Y]
            d = customers[s, COL_DEMAND]
            ax.annotate(f"#{s} D={d:.0f}kg", (x, y), fontsize=7,
                        textcoords="offset points", xytext=(5, 5))
            ax.annotate(str(idx + 1), (x, y), fontsize=8,
                        fontweight='bold', ha='center', va='center',
                        color='white',
                        bbox=dict(boxstyle='circle', fc=color, alpha=0.7))

    ax.set_title(f"{vname} - Route Map")
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize=8)


def _draw_single_route_profile(ax, stops, actions, customers, state, vname):
    """Draw the load/time profile subplot for a single route."""
    if state is not None and len(state) > 0:
        indices = np.arange(len(state))
        loads = state[:, 2]  # load column
        times = state[:, 3]  # time column
        ax.bar(indices, loads, color='steelblue', alpha=0.7, label='Load')
        ax2 = ax.twinx()
        ax2.plot(indices, times, 'o-', color='orange', label='Time')
        ax2.set_ylabel("Time (min)")
    elif len(stops) > 0:
        indices = np.arange(len(stops))
        demands = np.array([customers[s, COL_DEMAND] for s in stops])
        cumload = np.cumsum(demands)
        ax.bar(indices, cumload, color='steelblue', alpha=0.7, label='Cum. Load')
    else:
        ax.text(0.5, 0.5, "(empty route)", ha='center', va='center',
                transform=ax.transAxes)

    ax.set_title(f"{vname} - Load Profile")
    ax.set_xlabel("Stop index")
    ax.set_ylabel("Load (kg)")
    ax.legend(fontsize=8, loc='upper left')
