"""Fitness evaluation orchestrator."""
from src.data.constants import VEH_TRUCK, VEH_BIKE
from src.engine.simulate import simulate_all_routes
from src.engine.validate import validate_all
from src.engine.route_cost import (
    compute_route_cost, compute_makespan, count_reloads, total_wait_time,
)
from src.engine.violations import compute_penalties, calc_unserved_penalty
from src.engine.sync_cost import compute_sync_cost


FEASIBLE_ROUTE_BONUS = 2000000  # VND bonus per fully feasible route


def compute_fitness(total_cost, sync_cost, total_penalty, n_feasible_routes=0):
    """VND sum fitness with feasibility bonus. Lower = better."""
    return total_cost + sync_cost + total_penalty - n_feasible_routes * FEASIBLE_ROUTE_BONUS


def evaluate_solution(sol, customers, restricted, vehicles, dist_matrix,
                      delta_t, penalty_weights):
    """One-stop: simulate -> validate -> fitness. Returns full result dict."""
    truck_stops = sol["truck_stops"]
    truck_actions = sol["truck_actions"]
    bike_stops = sol["bike_stops"]
    bike_actions = sol["bike_actions"]
    satellites = sol["satellites"]
    n_customers = len(customers)

    # 1. Simulate
    (truck_states, bike_states, truck_return_times, bike_return_times,
     truck_distances, bike_distances, sim_feasible) = simulate_all_routes(
        truck_stops, truck_actions, bike_stops, bike_actions,
        vehicles, customers, dist_matrix, satellites, delta_t)

    # 2. Validate
    valid, report = validate_all(
        truck_stops, truck_actions, bike_stops, bike_actions,
        truck_states, bike_states, satellites, customers, restricted,
        vehicles, n_customers, delta_t)

    # 3. Operating cost
    total_cost, total_wait, cost_breakdown_agg = _aggregate_costs(
        truck_states, bike_states, truck_distances, bike_distances,
        truck_return_times, bike_return_times)

    # 4-7. Makespan, sync, penalties, fitness
    makespan = compute_makespan(truck_return_times, bike_return_times)
    sync_cost, sync_breakdown = compute_sync_cost(
        truck_states, bike_states, satellites, delta_t)
    # Add return_times to report for overtime penalty calculation
    all_return_times = list(truck_return_times) + list(bike_return_times)
    report["return_times"] = all_return_times

    total_penalty, _ = compute_penalties(report, penalty_weights,
                                         customers, dist_matrix)

    # Count feasible routes (all stops within TW + return within DAY_LENGTH)
    from src.data.cost import DAY_LENGTH
    tw_viol_routes = set()
    for (vtype, vid, *_) in report["time_windows"]["violations"]:
        tw_viol_routes.add((vtype, vid))
    n_feasible = 0
    for i, rt in enumerate(truck_return_times):
        if rt <= DAY_LENGTH and ("truck", i) not in tw_viol_routes and truck_return_times[i] > 0:
            n_feasible += 1
    for i, rt in enumerate(bike_return_times):
        if rt <= DAY_LENGTH and ("bike", i) not in tw_viol_routes and bike_return_times[i] > 0:
            n_feasible += 1

    fitness = compute_fitness(total_cost, sync_cost, total_penalty, n_feasible)
    report["return_times"] = all_return_times

    return {
        "fitness": fitness, "cost": total_cost, "sync_cost": sync_cost,
        "makespan": makespan, "total_penalty": total_penalty,
        "total_wait_time": total_wait, "feasible": sim_feasible and valid,
        "report": report, "cost_breakdown": cost_breakdown_agg,
        "sync_breakdown": sync_breakdown,
        "truck_states": truck_states, "bike_states": bike_states,
    }


def _aggregate_costs(truck_states, bike_states, truck_distances, bike_distances,
                     truck_return_times, bike_return_times):
    """Sum costs across all vehicles. Returns (total, total_wait, breakdown)."""
    total_cost = 0.0
    total_wait = 0.0
    agg = {"distance_cost": 0, "time_cost": 0, "wait_cost": 0,
           "fixed_cost": 0, "deploy_cost": 0, "reload_cost": 0}

    for i, st in enumerate(truck_states):
        wt = total_wait_time(st)
        total_wait += wt
        c, bd = compute_route_cost(truck_distances[i], truck_return_times[i],
                                   count_reloads(st), VEH_TRUCK, wt)
        total_cost += c
        for k in agg:
            agg[k] += bd.get(k, 0)

    for i, st in enumerate(bike_states):
        wt = total_wait_time(st)
        total_wait += wt
        c, bd = compute_route_cost(bike_distances[i], bike_return_times[i],
                                   count_reloads(st), VEH_BIKE, wt)
        total_cost += c
        for k in agg:
            agg[k] += bd.get(k, 0)

    return total_cost, total_wait, agg
