"""Run ALNS solver on grid_20x20 and produce dashboard."""
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
from src.alns.penalty import init_penalty_weights
from src.alns.config import make_config
from src.data.constants import (
    COL_X, COL_Y, COL_DEMAND, ACT_DELIVER, ACT_RELOAD,
    ST_START, ST_SERVICE, ST_ACTION, ST_LOAD_AFT,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_BIKE_STOPS, SOL_BIKE_ACTIONS,
    SOL_TRUCK_LENGTHS, SOL_BIKE_LENGTHS,
    SOL_META, META_N_TRUCKS, META_N_BIKES, VCOL_CAPACITY,
    EV_FITNESS, EV_COST, EV_MAKESPAN, EV_TOTAL_PENALTY, EV_FEASIBLE,
    CFG_MAX_ITERATIONS, CFG_NO_IMPROVE_LIMIT,
)
from src.data.cost import (
    SYNC_DELTA_T, DAY_LENGTH, TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
)

# === Load ===
print("Loading grid_20x20...")
custs, depot, vehs = load_instance("data/grid_20x20.json")
dm = compute_dist_matrix(depot, custs)
N = len(custs)
n_trucks, n_bikes = 10, 15

if len(vehs) < n_trucks + n_bikes:
    truck_row = vehs[0:1]
    bike_row = vehs[-1:]
    vehs = np.vstack([np.repeat(truck_row, n_trucks, axis=0),
                      np.repeat(bike_row, n_bikes, axis=0)])

cfg = make_config(N)
pw, _ = init_penalty_weights(cfg)

# === Initial Solution ===
print("Building initial solution...")
rng_init = np.random.default_rng(42)
clusters = cluster_customers(custs, dm, rng_init)
clusters = rebalance_clusters(clusters, custs, TRUCK_CAPACITY_G)
giant_tour = build_giant_tour(clusters, custs, depot, dm)

init_sol = build_initial_solution(custs, depot, vehs, dm, n_trucks, n_bikes, seed=42)
init_ev = evaluate_solution(init_sol, custs, vehs, dm, SYNC_DELTA_T, pw)
(init_truck_sim, init_bike_sim, init_truck_rt, init_bike_rt,
 init_truck_d, init_bike_d, _) = simulate_all_routes(init_sol, vehs, custs, dm, SYNC_DELTA_T)

print(f"  Initial: Cost={init_ev[EV_COST]:,.0f}  Penalty={init_ev[EV_TOTAL_PENALTY]:,.0f}")

# === Solve ===
MAX_ITER = 100000
print(f"Solving ({MAX_ITER} iter, {N} customers)...")
t0 = time.time()
best, fitness, archive, log = solve(
    custs, depot, vehs, dm, n_trucks=n_trucks, n_bikes=n_bikes,
    config_overrides={CFG_MAX_ITERATIONS: MAX_ITER, CFG_NO_IMPROVE_LIMIT: MAX_ITER},
    seed=42)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")

# === Evaluate final ===
ev = evaluate_solution(best, custs, vehs, dm, SYNC_DELTA_T, pw)
(truck_sim, bike_sim, truck_rt, bike_rt,
 truck_dists, bike_dists, _) = simulate_all_routes(best, vehs, custs, dm, SYNC_DELTA_T)

print(f"\n{'='*60}")
print(f"Feasible: {ev[EV_FEASIBLE]>0}  |  Cost: {ev[EV_COST]:,.0f} VND  |  Makespan: {ev[EV_MAKESPAN]} s")
print(f"Penalty: {ev[EV_TOTAL_PENALTY]:,.0f}")


# -- Dashboard helpers --
def draw_route_map(ax, sol, title, gt=None, clust=None):
    ax.plot(depot[0], depot[1], "s", color="red", markersize=12, zorder=10, label="Depot")
    ax.scatter(custs[:, COL_X], custs[:, COL_Y], s=5, c="black", zorder=3)
    if gt is not None:
        off = 0.15
        gt_xs = [depot[0]+off] + [custs[s["node"], COL_X]+off for s in gt] + [depot[0]+off]
        gt_ys = [depot[1]+off] + [custs[s["node"], COL_Y]+off for s in gt] + [depot[1]+off]
        ax.plot(gt_xs, gt_ys, "--", color="black", linewidth=2.5, alpha=0.6, zorder=6)
    if clust is not None:
        cmap = plt.colormaps.get_cmap("tab20").resampled(len(clust))
        for ci, cl in enumerate(clust):
            ax.scatter(custs[cl["members"], COL_X], custs[cl["members"], COL_Y],
                       s=20, color=cmap(ci), alpha=0.3, zorder=2)
    for t in range(n_trucks):
        stops = sol[SOL_TRUCK_STOPS][t]; valid = stops[stops >= 0]
        if len(valid) == 0: continue
        xs = [depot[0]] + [custs[s, COL_X] for s in valid] + [depot[0]]
        ys = [depot[1]] + [custs[s, COL_Y] for s in valid] + [depot[1]]
        ax.plot(xs, ys, "-", linewidth=7, alpha=0.7)
    for b in range(n_bikes):
        stops = sol[SOL_BIKE_STOPS][b]; valid = stops[stops >= 0]
        if len(valid) == 0: continue
        xs = [depot[0]] + [custs[s, COL_X] for s in valid] + [depot[0]]
        ys = [depot[1]] + [custs[s, COL_Y] for s in valid] + [depot[1]]
        ax.plot(xs, ys, "--", linewidth=3, alpha=0.4)
    ax.set_title(title); ax.set_aspect("equal")


def draw_gantt(ax, t_sim, b_sim, title):
    t_lens = init_sol[SOL_TRUCK_LENGTHS] if 'init' in title.lower() else best[SOL_TRUCK_LENGTHS]
    b_lens = init_sol[SOL_BIKE_LENGTHS] if 'init' in title.lower() else best[SOL_BIKE_LENGTHS]
    for tid in range(n_trucks):
        for r in range(int(t_lens[tid])):
            c = "gold" if int(t_sim[tid, r, ST_ACTION]) == ACT_RELOAD else "steelblue"
            ax.barh(tid, t_sim[tid, r, ST_SERVICE], left=t_sim[tid, r, ST_START],
                    height=0.6, color=c, edgecolor="black", linewidth=0.2)
    for bid in range(n_bikes):
        for r in range(int(b_lens[bid])):
            c = "gold" if int(b_sim[bid, r, ST_ACTION]) == ACT_RELOAD else "mediumseagreen"
            ax.barh(n_trucks+bid, b_sim[bid, r, ST_SERVICE], left=b_sim[bid, r, ST_START],
                    height=0.6, color=c, edgecolor="black", linewidth=0.2)
    ax.axvline(x=DAY_LENGTH, color="red", linewidth=2, linestyle="--")
    ax.set_title(title); ax.invert_yaxis()


# -- Dashboard --
fig = plt.figure(figsize=(28, 16))
fig.suptitle(f"2E-VRP | {N} custs | {n_trucks}T+{n_bikes}B | {MAX_ITER} iter | {elapsed:.1f}s",
             fontsize=14, fontweight="bold")

ax = fig.add_subplot(2, 4, 1)
draw_route_map(ax, init_sol, "INITIAL: Routes", gt=giant_tour, clust=clusters)
ax = fig.add_subplot(2, 4, 2)
draw_gantt(ax, init_truck_sim, init_bike_sim, "INITIAL: Gantt")

ax = fig.add_subplot(2, 4, 5)
draw_route_map(ax, best, "FINAL: Routes")
ax = fig.add_subplot(2, 4, 6)
draw_gantt(ax, truck_sim, bike_sim, "FINAL: Gantt")

# Convergence
iters = log["iterations"]
ax = fig.add_subplot(2, 4, 3)
ax.plot(iters, log["fitness"], color="gray", alpha=0.3, linewidth=0.5)
ax.plot(iters, log["best_fitness"], color="red", linewidth=1.5)
ax.set_title("Fitness"); ax.set_yscale("log")

ax = fig.add_subplot(2, 4, 7)
ax.plot(iters, log["penalty"], color="red", linewidth=0.8)
ax.set_title("Penalty")

# Summary
ax = fig.add_subplot(2, 4, 4); ax.axis("off")
impr = (1 - ev[EV_COST] / max(init_ev[EV_COST], 1)) * 100
txt = f"INITIAL Cost: {init_ev[EV_COST]:,.0f}\nFINAL   Cost: {ev[EV_COST]:,.0f}\nChange: {impr:+.1f}%\nPareto: {len(archive)} sols\nTime: {elapsed:.1f}s"
ax.text(0.1, 0.5, txt, transform=ax.transAxes, fontsize=10, va="center", fontfamily="monospace")
ax.set_title("Summary")

fig.tight_layout()
fig.savefig("data/alns_dashboard.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: data/alns_dashboard.png")
plt.close(fig)
