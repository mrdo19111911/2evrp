"""Run ALNS solver on grid_20x20 and produce dashboard with initial + final view."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time

from src.data.io import load_instance
from src.data.distance import compute_dist_matrix
from src.init.builder import build_initial_solution
from src.init.clustering import cluster_customers, rebalance_clusters
from src.init.giant_tour import build_giant_tour
from src.alns.loop import solve
from src.engine.fitness import evaluate_solution
from src.engine.simulate import simulate_all_routes
from src.data.constants import (
    COL_X, COL_Y, COL_DEMAND, ACT_DELIVER, ACT_RELOAD,
    ST_START, ST_SERVICE, ST_ACTION, ST_LOAD_AFT,
)
from src.data.cost import (
    PENALTY_UNSERVED, PENALTY_OVERLOAD_KG, PENALTY_LATE,
    PENALTY_SYNC_FAIL, PENALTY_RESTRICTION, PENALTY_MISSING_RELOAD,
    SYNC_DELTA_T, DAY_LENGTH, TRUCK_CAPACITY, BIKE_CAPACITY,
)

# === Load ===
print("Loading grid_20x20...")
custs, restr, depot, vehs = load_instance("data/grid_20x20.json")
dm = compute_dist_matrix(depot, custs)
N = len(custs)
n_trucks, n_bikes = 10, 15

# Extend vehicles array if needed (original has 5T+15B=20)
if len(vehs) < n_trucks + n_bikes:
    truck_row = vehs[0:1]  # template truck
    bike_row = vehs[-1:]   # template bike
    new_vehs = np.vstack([
        np.repeat(truck_row, n_trucks, axis=0),
        np.repeat(bike_row, n_bikes, axis=0),
    ])
    vehs = new_vehs

pw = {"unserved": PENALTY_UNSERVED, "duplicate": PENALTY_UNSERVED,
      "capacity": PENALTY_OVERLOAD_KG, "time_window": PENALTY_LATE,
      "sync": PENALTY_SYNC_FAIL, "vehicle_restriction": PENALTY_RESTRICTION,
      "missing_reload": PENALTY_MISSING_RELOAD}

# === Initial Solution ===
print("Building initial solution...")
# Build clusters + giant tour for visualization
rng_init = np.random.default_rng(42)
clusters = cluster_customers(custs, restr, dm, rng_init)
clusters = rebalance_clusters(clusters, custs, TRUCK_CAPACITY)
giant_tour = build_giant_tour(clusters, custs, depot, dm)

init_sol = build_initial_solution(custs, restr, depot, vehs, dm,
                                  n_trucks, n_bikes, seed=42)
init_result = evaluate_solution(init_sol, custs, restr, vehs, dm, SYNC_DELTA_T, pw)
(init_truck_st, init_bike_st, init_truck_rt, init_bike_rt,
 init_truck_d, init_bike_d, _) = simulate_all_routes(
    init_sol["truck_stops"], init_sol["truck_actions"],
    init_sol["bike_stops"], init_sol["bike_actions"],
    vehs, custs, dm, init_sol["satellites"], SYNC_DELTA_T)

init_rp = init_result.get("report", {})
print(f"  Initial: Cost={init_result['cost']:,.0f}  Penalty={init_result['total_penalty']:,.0f}  "
      f"TW={len(init_rp.get('time_windows',{}).get('violations',[]))}  "
      f"Cap={len(init_rp.get('capacity',{}).get('violations',[]))}  "
      f"Unserved={len(init_rp.get('delivery_uniqueness',{}).get('unserved',[]))}")

# === Solve ===
MAX_ITER = 500
print(f"Solving ({MAX_ITER} iter, {N} customers)...")
t0 = time.time()
best, fitness, archive, log = solve(
    custs, restr, depot, vehs, dm, n_trucks=n_trucks, n_bikes=n_bikes,
    config_overrides={"max_iterations": MAX_ITER, "no_improve_limit": MAX_ITER},
    seed=42)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")

# === Evaluate final ===
result = evaluate_solution(best, custs, restr, vehs, dm, SYNC_DELTA_T, pw)
(truck_states, bike_states, truck_rt, bike_rt,
 truck_dists, bike_dists, _) = simulate_all_routes(
    best["truck_stops"], best["truck_actions"],
    best["bike_stops"], best["bike_actions"],
    vehs, custs, dm, best["satellites"], SYNC_DELTA_T)

bd = result.get("cost_breakdown", {})
rp = result.get("report", {})
mr = rp.get("missing_reload", {}).get("violations", [])
print(f"\n{'='*60}")
print(f"Feasible: {result['feasible']}  |  Cost: {result['cost']:,.0f} VND  |  Makespan: {result['makespan']:.0f} min")
print(f"Wait: {result['total_wait_time']:.0f} min  |  Penalty: {result['total_penalty']:,.0f}")
print(f"TW violations: {len(rp.get('time_windows',{}).get('violations',[]))}")
print(f"Capacity violations: {len(rp.get('capacity',{}).get('violations',[]))}")
for k, v in bd.items():
    print(f"  {k:18s}: {v:>14,.0f} VND")


# ======================================================================
# HELPER: draw route map on axes
# ======================================================================
def draw_route_map(ax, sol, title, gt=None, clust=None):
    ax.plot(depot[0], depot[1], "s", color="red", markersize=12, zorder=10, label="Depot")
    heavy_mask = custs[:, COL_DEMAND] > 60
    ax.scatter(custs[~heavy_mask, COL_X], custs[~heavy_mask, COL_Y], s=5, c="black", zorder=3)
    ax.scatter(custs[heavy_mask, COL_X], custs[heavy_mask, COL_Y], s=40, c="red", zorder=4, label="Heavy")

    # Giant tour overlay (offset a few pixels for visibility)
    if gt is not None and len(gt) > 0:
        off = 0.15  # slight offset so it doesn't overlap routes
        gt_xs = [depot[0]+off] + [custs[s["node"], COL_X]+off for s in gt] + [depot[0]+off]
        gt_ys = [depot[1]+off] + [custs[s["node"], COL_Y]+off for s in gt] + [depot[1]+off]
        ax.plot(gt_xs, gt_ys, "--", color="black", linewidth=2.5, alpha=0.6,
                zorder=6, label="Giant Tour")
        # Mark satellite nodes with star
        for s in gt:
            if s["type"] == "satellite":
                ax.plot(custs[s["node"], COL_X], custs[s["node"], COL_Y],
                        "*", color="gold", markersize=14, markeredgecolor="black",
                        markeredgewidth=0.5, zorder=8)
        # Number the giant tour stops
        for idx, s in enumerate(gt):
            ax.annotate(str(idx), (custs[s["node"], COL_X], custs[s["node"], COL_Y]),
                        fontsize=5, fontweight="bold", color="purple",
                        ha="center", va="bottom", zorder=9)

    # Cluster coloring
    if clust is not None:
        cmap = plt.colormaps.get_cmap("tab20").resampled(len(clust))
        for ci, cl in enumerate(clust):
            members = cl["members"]
            ax.scatter(custs[members, COL_X], custs[members, COL_Y],
                       s=20, color=cmap(ci), alpha=0.3, zorder=2)

    # Truck routes
    for t in range(n_trucks):
        stops = sol["truck_stops"][t]; valid = stops[stops >= 0]
        if len(valid) == 0: continue
        xs = [depot[0]] + [custs[s, COL_X] for s in valid] + [depot[0]]
        ys = [depot[1]] + [custs[s, COL_Y] for s in valid] + [depot[1]]
        ax.plot(xs, ys, "-", linewidth=7, alpha=0.7)
    # Bike routes
    for b in range(n_bikes):
        stops = sol["bike_stops"][b]; valid = stops[stops >= 0]
        if len(valid) == 0: continue
        xs = [depot[0]] + [custs[s, COL_X] for s in valid] + [depot[0]]
        ys = [depot[1]] + [custs[s, COL_Y] for s in valid] + [depot[1]]
        ax.plot(xs, ys, "--", linewidth=3, alpha=0.4)
    ax.set_title(title); ax.set_aspect("equal"); ax.legend(fontsize=5)


def draw_gantt(ax, t_states, b_states, title):
    for tid in range(n_trucks):
        for row in t_states[tid]:
            c = "gold" if int(row[ST_ACTION]) == ACT_RELOAD else "steelblue"
            ax.barh(tid, row[ST_SERVICE], left=row[ST_START], height=0.6,
                    color=c, edgecolor="black", linewidth=0.2)
    for bid in range(n_bikes):
        for row in b_states[bid]:
            c = "gold" if int(row[ST_ACTION]) == ACT_RELOAD else "mediumseagreen"
            ax.barh(n_trucks+bid, row[ST_SERVICE], left=row[ST_START], height=0.6,
                    color=c, edgecolor="black", linewidth=0.2)
    ax.axhline(y=n_trucks-0.5, color="black", linewidth=1, linestyle="--")
    ax.axvline(x=DAY_LENGTH, color="red", linewidth=2, linestyle="--", label="8h")
    labels = [f"T{i}" for i in range(n_trucks)] + [f"B{i}" for i in range(n_bikes)]
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=5)
    ax.set_title(title); ax.set_xlabel("Time (min)"); ax.invert_yaxis(); ax.legend(fontsize=6)


def draw_load(ax, states, n_vehs, capacity, vtype_prefix, title):
    for vid in range(n_vehs):
        st = states[vid]
        if len(st) == 0: continue
        ax.plot(np.arange(len(st)), st[:, ST_LOAD_AFT], "o-", markersize=2,
                linewidth=0.8, label=f"{vtype_prefix}{vid}")
    ax.axhline(y=capacity, color="red", linestyle="--", linewidth=1, label=f"Cap {capacity:.0f}kg")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_title(title); ax.set_xlabel("Stop"); ax.set_ylabel("Load (kg)")
    ax.legend(fontsize=5, ncol=3)


def draw_summary(ax, res, bd_, rp_, extra=""):
    ax.axis("off")
    mr_ = rp_.get("missing_reload", {}).get("violations", [])
    stats = [
        f"Feasible:      {'YES' if res['feasible'] else 'NO'}",
        f"Cost:          {res['cost']:>12,.0f} VND",
        f"  Distance:    {bd_.get('distance_cost',0):>12,.0f}",
        f"  Driver time: {bd_.get('time_cost',0):>12,.0f}",
        f"  Wait cost:   {bd_.get('wait_cost',0):>12,.0f}",
        f"  Deploy:      {bd_.get('deploy_cost',0):>12,.0f}",
        f"  Fixed/day:   {bd_.get('fixed_cost',0):>12,.0f}",
        f"  Reload:      {bd_.get('reload_cost',0):>12,.0f}",
        f"Makespan:      {res['makespan']:>8.0f} min",
        f"Penalty:       {res['total_penalty']:>12,.0f}",
        f"TW viol:       {len(rp_.get('time_windows',{}).get('violations',[]))}",
        f"Cap viol:      {len(rp_.get('capacity',{}).get('violations',[]))}",
        f"Unserved:      {len(rp_.get('delivery_uniqueness',{}).get('unserved',[]))}",
    ]
    if extra:
        stats.append(extra)
    ax.text(0.05, 0.95, "\n".join(stats), transform=ax.transAxes, fontsize=7,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))


# ======================================================================
# DASHBOARD: 4 rows x 4 cols = 16 panels
# Row 1-2: Initial Solution (route, gantt, truck load, summary)
# Row 3-4: Final Solution   (route, gantt, convergence, summary)
# ======================================================================
fig = plt.figure(figsize=(28, 24))
fig.suptitle(f"2E-VRP | {N} custs | {n_trucks}T+{n_bikes}B | {MAX_ITER} iter | {elapsed:.1f}s",
             fontsize=14, fontweight="bold")

# --- ROW 1: INITIAL SOLUTION ---
ax = fig.add_subplot(4, 4, 1)
draw_route_map(ax, init_sol, f"INITIAL: Routes + Giant Tour", gt=giant_tour, clust=clusters)

ax = fig.add_subplot(4, 4, 2)
draw_gantt(ax, init_truck_st, init_bike_st, "INITIAL: Gantt")

ax = fig.add_subplot(4, 4, 3)
draw_load(ax, init_truck_st, n_trucks, TRUCK_CAPACITY, "T", "INITIAL: Truck Load")

ax = fig.add_subplot(4, 4, 4)
draw_summary(ax, init_result, init_result.get("cost_breakdown", {}), init_rp)
ax.set_title("INITIAL: Summary")

# --- ROW 2: INITIAL BIKE LOAD + PENALTY BREAKDOWN ---
ax = fig.add_subplot(4, 4, 5)
draw_load(ax, init_bike_st, n_bikes, BIKE_CAPACITY, "B", "INITIAL: Bike Load")

ax = fig.add_subplot(4, 4, 6)
# Initial cost breakdown
init_bd = init_result.get("cost_breakdown", {})
labels_bd = [k for k in init_bd.keys() if init_bd[k] > 0]
values_bd = [init_bd[k] for k in labels_bd]
if values_bd:
    colors_bd = ["#2196F3","#FF9800","#F44336","#4CAF50","#9C27B0","#795548","#E91E63"]
    ax.barh(labels_bd, values_bd, color=colors_bd[:len(labels_bd)])
    for bar, val in zip(ax.patches, values_bd):
        ax.text(bar.get_width()+max(values_bd)*0.01, bar.get_y()+bar.get_height()/2,
                f"{val:,.0f}", va="center", fontsize=5)
ax.set_title(f"INITIAL: Cost Breakdown ({init_result['cost']:,.0f} VND)")

# --- ROW 3: FINAL SOLUTION ---
ax = fig.add_subplot(4, 4, 7)
draw_route_map(ax, best, f"FINAL: Route Map")

ax = fig.add_subplot(4, 4, 8)
draw_gantt(ax, truck_states, bike_states, "FINAL: Gantt")

ax = fig.add_subplot(4, 4, 9)
draw_load(ax, truck_states, n_trucks, TRUCK_CAPACITY, "T", "FINAL: Truck Load")

ax = fig.add_subplot(4, 4, 10)
draw_summary(ax, result, bd, rp, f"Pareto: {len(archive)} sols\nTime: {elapsed:.1f}s")
ax.set_title("FINAL: Summary")

# --- ROW 4: FINAL DETAILS + CONVERGENCE ---
ax = fig.add_subplot(4, 4, 11)
draw_load(ax, bike_states, n_bikes, BIKE_CAPACITY, "B", "FINAL: Bike Load")

ax = fig.add_subplot(4, 4, 12)
labels_bd2 = [k for k in bd.keys() if bd[k] > 0]
values_bd2 = [bd[k] for k in labels_bd2]
if values_bd2:
    ax.barh(labels_bd2, values_bd2, color=["#2196F3","#FF9800","#F44336","#4CAF50","#9C27B0","#795548","#E91E63"][:len(labels_bd2)])
    for bar, val in zip(ax.patches, values_bd2):
        ax.text(bar.get_width()+max(values_bd2)*0.01, bar.get_y()+bar.get_height()/2,
                f"{val:,.0f}", va="center", fontsize=5)
ax.set_title(f"FINAL: Cost Breakdown ({result['cost']:,.0f} VND)")

# Convergence
iters = log["iterations"]
ax = fig.add_subplot(4, 4, 13)
ax.plot(iters, log["fitness"], color="gray", alpha=0.3, linewidth=0.5, label="Current")
ax.plot(iters, log["best_fitness"], color="red", linewidth=1.5, label="Best")
ax.set_title("Fitness Convergence"); ax.set_xlabel("Iter"); ax.set_yscale("log")
ax.legend(fontsize=6)

ax = fig.add_subplot(4, 4, 14)
ax.plot(iters, log["cost"], color="steelblue", linewidth=0.8, label="Cost")
ax2 = ax.twinx()
ax2.plot(iters, log["makespan"], color="orange", linewidth=0.8, label="Makespan")
ax.set_title("Cost + Makespan"); ax.set_xlabel("Iter")
ax.set_ylabel("Cost", color="steelblue"); ax2.set_ylabel("Makespan", color="orange")

ax = fig.add_subplot(4, 4, 15)
ax.plot(iters, log["penalty"], color="red", linewidth=0.8)
ax3 = ax.twinx()
w = min(50, len(iters))
if w > 0:
    feas = np.convolve(log["feasible"], np.ones(w)/w, mode="same")*100
    ax3.plot(iters, feas, color="green", linewidth=1)
    ax3.set_ylim(-5, 105); ax3.set_ylabel("Feasible %", color="green")
ax.set_title("Penalty + Feasibility"); ax.set_xlabel("Iter")

# Improvement: initial vs final
ax = fig.add_subplot(4, 4, 16)
ax.axis("off")
impr_cost = (1 - result["cost"] / max(init_result["cost"], 1)) * 100
impr_pen = (1 - result["total_penalty"] / max(init_result["total_penalty"], 1)) * 100
impr_fit = (1 - result["fitness"] / max(init_result["fitness"], 1)) * 100
compare = [
    f"{'':15s} {'INITIAL':>14s} {'FINAL':>14s} {'CHANGE':>8s}",
    f"{'Cost':15s} {init_result['cost']:>14,.0f} {result['cost']:>14,.0f} {impr_cost:>+7.1f}%",
    f"{'Penalty':15s} {init_result['total_penalty']:>14,.0f} {result['total_penalty']:>14,.0f} {impr_pen:>+7.1f}%",
    f"{'Fitness':15s} {init_result['fitness']:>14,.0f} {result['fitness']:>14,.0f} {impr_fit:>+7.1f}%",
    f"{'Makespan':15s} {init_result['makespan']:>14,.0f} {result['makespan']:>14,.0f}",
    "",
    f"Iterations: {len(iters)}",
    f"Pareto: {len(archive)} solutions",
    f"Time: {elapsed:.1f}s",
]
ax.text(0.05, 0.95, "\n".join(compare), transform=ax.transAxes, fontsize=7,
        va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.9))
ax.set_title("Initial vs Final")

fig.tight_layout()
fig.savefig("data/alns_dashboard.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: data/alns_dashboard.png")
plt.close(fig)
