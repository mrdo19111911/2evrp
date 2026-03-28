"""Visualize 2E-VRP grid_20x20 input data as a publication-quality PNG."""
import sys
sys.path.insert(0, "e:/2evrp")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from src.data.constants import COL_X, COL_Y, COL_DEMAND, VCOL_TYPE, VCOL_CAPACITY, VEH_TRUCK, VEH_BIKE


def load_data(path="e:/2evrp/data/grid_20x20.npz"):
    d = np.load(path)
    return d["customers"], d["restricted"], d["depot"], d["vehicles"]


def split_customers(customers, restricted):
    demand = customers[:, COL_DEMAND]
    heavy_mask = demand > 60
    light_mask = ~heavy_mask
    restricted_mask = restricted.astype(bool)
    return heavy_mask, light_mask, restricted_mask


def compute_stats(customers, vehicles):
    demand = customers[:, COL_DEMAND]
    truck_cap = vehicles[vehicles[:, VCOL_TYPE] == VEH_TRUCK, VCOL_CAPACITY][0]
    bike_cap = vehicles[vehicles[:, VCOL_TYPE] == VEH_BIKE, VCOL_CAPACITY][0]
    return demand.sum(), truck_cap, bike_cap


def draw_plot(customers, restricted, depot, vehicles, out_path):
    heavy_mask, light_mask, restricted_mask = split_customers(customers, restricted)
    total_demand, truck_cap, bike_cap = compute_stats(customers, vehicles)
    x, y, demand = customers[:, COL_X], customers[:, COL_Y], customers[:, COL_DEMAND]

    fig, ax = plt.subplots(figsize=(12, 12))

    # Light customers: small black dots
    ax.scatter(x[light_mask], y[light_mask], s=10, c="black", zorder=3, label="Light ($\\leq$60 kg)")

    # Heavy customers: red circles sized by demand
    heavy_sizes = demand[heavy_mask] / demand[heavy_mask].max() * 300 + 50
    sc = ax.scatter(x[heavy_mask], y[heavy_mask], s=heavy_sizes, c="red", alpha=0.7,
                    edgecolors="darkred", linewidths=0.8, zorder=4, label="Heavy (>60 kg)")

    # Annotate heavy nodes with demand values
    for xi, yi, di in zip(x[heavy_mask], y[heavy_mask], demand[heavy_mask]):
        ax.annotate(f"{int(di)}", (xi, yi), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, color="darkred", fontweight="bold")

    # Restricted customers: green ring
    ax.scatter(x[restricted_mask], y[restricted_mask], s=80, facecolors="none",
               edgecolors="green", linewidths=1.5, zorder=5, label="Restricted (bike-only)")

    # Depot: red square
    ax.scatter(depot[0], depot[1], s=200, c="red", marker="s",
               edgecolors="black", linewidths=1.5, zorder=6, label="KHO (depot)")

    # Grid lines
    ax.set_axisbelow(True)
    ax.grid(True, color="lightgray", linewidth=0.5)
    ax.set_xticks(np.arange(0, 105, 5))
    ax.set_yticks(np.arange(0, 105, 5))

    # Labels and title
    n_heavy = heavy_mask.sum()
    n_light = light_mask.sum()
    n_rest = restricted_mask.sum()
    ax.set_title(
        f"2E-VRP Input: {len(customers)} customers, {n_heavy} heavy (400kg), "
        f"{n_light} light (10kg), {n_rest} restricted",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("X (km)", fontsize=12)
    ax.set_ylabel("Y (km)", fontsize=12)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_aspect("equal")

    # Legend
    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="red",
               markeredgecolor="black", markersize=12, label="KHO (depot)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markeredgecolor="darkred", markersize=10, alpha=0.7, label="Heavy (>60 kg)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=5, label="Light ($\\leq$60 kg)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="green", markeredgewidth=1.5, markersize=10,
               label="Restricted (bike-only)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.9)

    # Stats text box
    stats_text = (
        f"Total demand: {int(total_demand)} kg\n"
        f"Truck capacity: {int(truck_cap)} kg\n"
        f"Bike capacity: {int(bike_cap)} kg"
    )
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    customers, restricted, depot, vehicles = load_data()
    draw_plot(customers, restricted, depot, vehicles, "e:/2evrp/data/grid_20x20_input.png")


if __name__ == "__main__":
    main()
