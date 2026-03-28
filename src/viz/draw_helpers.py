"""Shared low-level drawing helpers for viz modules."""
import matplotlib.pyplot as plt
import numpy as np

from src.data.constants import (
    LOC_X, LOC_Y, ACT_DELIVER, ACT_PICKUP, ACT_PAD,
    VTYPE_TRUCK, VTYPE_BIKE, VEH_TYPE, VEH_CAP_KG,
    ST_LOC, ST_FEASIBLE, ST_LOAD_KG_AFT,
)


def get_route_stops(sol, v):
    """Extract (stops, actions) for vehicle v, filtering out padding."""
    stops = sol["stops"][v]
    actions = sol["actions"][v]
    length = int(sol["lengths"][v])
    return stops[:length], actions[:length]


def draw_depot(ax, locations, depot_loc):
    """Draw depot as red square. depot_loc is location_id."""
    ax.plot(locations[depot_loc, LOC_X], locations[depot_loc, LOC_Y],
            's', color='red', markersize=14, zorder=10, label='Depot')


def draw_customers(ax, locations, orders, sol, show_labels, show_demand):
    """Draw customer nodes. orders[:,0]=ORD_LOC, weight=ORD_QTY*ORD_UNIT_W."""
    from src.data.constants import ORD_LOC, ORD_QTY, ORD_UNIT_W
    n = len(orders)
    for i in range(n):
        loc = int(orders[i, ORD_LOC])
        x, y = locations[loc, LOC_X], locations[loc, LOC_Y]
        demand_kg = orders[i, ORD_QTY] * orders[i, ORD_UNIT_W]
        size = max(20, min(200, demand_kg * 2))
        ax.scatter(x, y, s=size, c='steelblue', edgecolors='black',
                   linewidths=0.5, zorder=5)
        if show_labels:
            ax.annotate(f"C{i}", (x, y), fontsize=7, ha='center', va='bottom',
                        textcoords="offset points", xytext=(0, 5))
        if show_demand:
            ax.annotate(f"{demand_kg:.0f}kg", (x, y), fontsize=6, ha='center',
                        va='top', textcoords="offset points", xytext=(0, -8))


def draw_vehicle_route(ax, stops, actions, locations, depot_loc, color,
                       linestyle, linewidth, label):
    """Draw a single vehicle route with arrows."""
    if len(stops) == 0:
        return
    dx, dy = locations[depot_loc, LOC_X], locations[depot_loc, LOC_Y]
    xs = [dx]
    ys = [dy]
    for s in stops:
        xs.append(locations[int(s), LOC_X])
        ys.append(locations[int(s), LOC_Y])
    xs.append(dx)
    ys.append(dy)

    ax.plot(xs, ys, linestyle=linestyle, linewidth=linewidth,
            color=color, alpha=0.7, label=label, zorder=3)

    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i + 1], ys[i + 1]),
                    xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="->", color=color,
                                    lw=linewidth * 0.6, alpha=0.5))

    for s, a in zip(stops, actions):
        if a == ACT_PICKUP:
            ax.plot(locations[int(s), LOC_X], locations[int(s), LOC_Y], 'D',
                    color='gold', markersize=8, markeredgecolor='black',
                    zorder=7)


def draw_satellites(ax, sol, locations):
    """Draw satellite/transfer markers (yellow diamonds)."""
    transfers = sol.get("transfers", np.empty((0, 6)))
    if len(transfers) == 0:
        return
    from src.data.constants import TR_HUB
    hub_locs = np.unique(transfers[:, TR_HUB].astype(int))
    for hub in hub_locs:
        x, y = locations[hub, LOC_X], locations[hub, LOC_Y]
        ax.plot(x, y, 'D', color='gold', markersize=12,
                markeredgecolor='black', markeredgewidth=1.5, zorder=8)
        ax.annotate("HUB", (x, y), fontsize=6, ha='center', va='bottom',
                    textcoords="offset points", xytext=(0, 8),
                    fontweight='bold')


def draw_violations(ax, states, locations):
    """Overlay red X on nodes with violations (ST_FEASIBLE==0)."""
    for state in states:
        if state is None or len(state) == 0:
            continue
        if state.shape[1] <= ST_FEASIBLE:
            continue
        for row in state:
            loc = int(row[ST_LOC])
            if loc < 0 or loc >= len(locations):
                continue
            if row[ST_FEASIBLE] == 0:
                x, y = locations[loc, LOC_X], locations[loc, LOC_Y]
                ax.plot(x, y, 'x', color='red', markersize=12,
                        markeredgewidth=3, zorder=9)


def draw_load_profile(ax, sol, states, vehicles, vtype):
    """Draw load profile for all vehicles of given type."""
    n_vehicles = int(sol["n_vehicles"])
    vname = "Truck" if vtype == VTYPE_TRUCK else "Bike"
    cap = 0.0

    for v in range(n_vehicles):
        if vehicles[v, VEH_TYPE] != vtype:
            continue
        cap = max(cap, vehicles[v, VEH_CAP_KG])
        st = states[v]
        if st is None or len(st) == 0:
            continue
        loads = st[:, ST_LOAD_KG_AFT]
        ax.plot(np.arange(len(loads)), loads, 'o-',
                label=f"{vname} {v}", markersize=4)

    if cap > 0:
        ax.axhline(y=cap, color='red', linestyle='--', linewidth=1,
                   label=f"Cap={cap:.0f}kg")
    ax.set_title(f"{vname} Load Profile")
    ax.set_xlabel("Stop index")
    ax.set_ylabel("Cumulative Load (kg)")
    ax.legend(fontsize=7)


def draw_stats_table(ax, eval_result):
    """Draw summary stats as text on an axes panel."""
    ax.axis('off')
    lines = [
        f"Total Cost:     {eval_result.get('cost', 0):,.0f} VND",
        f"Makespan:       {eval_result.get('makespan', 0):.1f} min",
        f"Total Distance: {eval_result.get('total_distance', 0):.1f} km",
        f"Vehicles Used:  {eval_result.get('vehicles_used', 0)}",
        f"Feasible:       {'Yes' if eval_result.get('feasible') else 'No'}",
        f"Penalty:        {eval_result.get('penalty', 0):,.0f}",
    ]
    text = "\n".join(lines)
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title("Summary Stats")


def draw_route_map_on_ax(ax, sol, locations, vehicles, depot_loc):
    """Simplified route map drawing directly on a given axes."""
    dx, dy = locations[depot_loc, LOC_X], locations[depot_loc, LOC_Y]
    ax.plot(dx, dy, 's', color='red', markersize=10, zorder=10)

    from src.data.constants import ORD_LOC
    # scatter all location points used by orders -- skip, just draw routes
    n_vehicles = int(sol["n_vehicles"])
    for v in range(n_vehicles):
        stops, actions = get_route_stops(sol, v)
        if len(stops) == 0:
            continue
        is_truck = vehicles[v, VEH_TYPE] == VTYPE_TRUCK
        prefix = "T" if is_truck else "B"
        style = '-' if is_truck else '--'
        lw = 2 if is_truck else 1
        xs = [dx] + [locations[int(s), LOC_X] for s in stops] + [dx]
        ys = [dy] + [locations[int(s), LOC_Y] for s in stops] + [dy]
        ax.plot(xs, ys, style, linewidth=lw, alpha=0.7, label=f"{prefix}{v}")

    ax.set_title("Route Map")
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize=7)
