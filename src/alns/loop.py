"""Main ALNS loop — short orchestrator pattern."""
from .acceptance import sa_accept, cool_temperature
from .penalty import adaptive_penalty_adjustment
from .logger import log_iteration, save_log
from .weights import select_operator
from .destroy import DESTROY_OPS
from .repair import REPAIR_OPS
from .crosslayer import CROSS_OPS
from .local_search import run_local_search
from .setup import (setup, eval_sol, init_search_state,
                    classify_result, update_op_scores, update_best,
                    update_pareto, update_feasible_history,
                    update_weights_if_segment_end)
from ..init.builder import build_initial_solution
from ..solution.structure import copy_solution


def solve(customers, restricted, depot, vehicles, dist_matrix,
          n_trucks, n_bikes, config_overrides=None, seed=42, callback=None):
    """Main entry: build initial -> ALNS loop -> Pareto front. 10 lines of logic."""
    config, rng = setup(customers, config_overrides, seed)
    sol = build_initial_solution(customers, restricted, depot, vehicles,
                                 dist_matrix, n_trucks, n_bikes, seed)
    state = init_search_state(sol, customers, restricted, vehicles,
                               dist_matrix, config, rng)

    for iteration in range(config["max_iterations"]):
        new_sol = destroy_and_repair(sol, state, customers, restricted,
                                      dist_matrix, vehicles, rng, config)
        new_sol = maybe_local_search(new_sol, iteration, state, dist_matrix,
                                      customers, restricted, config)
        new_sol = maybe_cross_layer(new_sol, iteration, state, customers,
                                     restricted, dist_matrix, rng, config)
        sol, state = accept_and_update(sol, new_sol, state, iteration,
                                        customers, restricted, vehicles,
                                        dist_matrix, rng, config)
        log_and_callback(state, iteration, callback)

        if should_stop(state, config):
            break

    return finalize(sol, state)


def destroy_and_repair(sol, state, customers, restricted, dist_matrix,
                        vehicles, rng, config):
    """Copy -> select destroy -> apply -> select repair -> apply. Returns new_sol."""
    new_sol = copy_solution(sol)
    d_idx = select_operator(state["destroy_w"], rng)
    removed = DESTROY_OPS[d_idx](new_sol, customers, dist_matrix, rng, config)
    r_idx = select_operator(state["repair_w"], rng)
    REPAIR_OPS[r_idx](new_sol, removed, customers, restricted,
                       dist_matrix, vehicles, rng, config)
    state["last_d_idx"] = d_idx
    state["last_r_idx"] = r_idx
    return new_sol


def maybe_local_search(sol, iteration, state, dist_matrix, customers,
                       restricted, config):
    """Run LS if on schedule. Returns sol."""
    if iteration % config["ls_frequency"] != 0:
        return sol
    run_local_search(sol, dist_matrix, customers, restricted)
    return sol


def maybe_cross_layer(sol, iteration, state, customers, restricted,
                       dist_matrix, rng, config):
    """Run cross-layer op if on schedule. Returns sol."""
    if iteration % config["cross_frequency"] != 0:
        return sol
    cx_idx = select_operator(state["cross_w"], rng)
    CROSS_OPS[cx_idx](sol, customers, restricted, dist_matrix,
                       sol["satellites"], rng, {})
    state["last_cx_idx"] = cx_idx
    state["cross_used"] = True
    return sol


def accept_and_update(sol, new_sol, state, iteration, customers, restricted,
                       vehicles, dist_matrix, rng, config):
    """SA accept -> update best -> scores -> weights -> cool -> penalty."""
    new_eval = eval_sol(new_sol, customers, restricted, vehicles,
                        dist_matrix, state, config)
    accepted = sa_accept(state["current_fitness"], new_eval["fitness"],
                          state["temperature"], rng)

    score = classify_result(accepted, new_eval, state, config)
    update_op_scores(state, score)

    if accepted:
        sol = new_sol
        state["current_fitness"] = new_eval["fitness"]

    update_best(state, sol, new_eval)
    update_pareto(state, new_eval, sol)
    update_feasible_history(state, iteration, new_eval, config)
    update_weights_if_segment_end(state, iteration, config)
    state["temperature"] = cool_temperature(state["temperature"],
                                             config["sa_cooling_rate"])
    state["w3"] = adaptive_penalty_adjustment(state["w3"],
                                               state["feasible_history"], config)
    state["last_eval"] = new_eval
    state["accepted"] = accepted
    return sol, state


def log_and_callback(state, iteration, callback):
    """Log iteration + invoke callback."""
    ev = state["last_eval"]
    log_iteration(state["log"], iteration,
                  ev["fitness"], ev["cost"], ev["makespan"],
                  ev["total_penalty"], ev["feasible"], state["accepted"],
                  state["temperature"], state["w3"],
                  state["last_d_idx"], state["last_r_idx"],
                  state["best_fitness"], len(state["archive"]))
    if callback is not None:
        callback(iteration, state["best_sol"], ev, state["log"])


def should_stop(state, config):
    """No-improve limit reached?"""
    return state["no_improve_count"] >= config["no_improve_limit"]


def finalize(sol, state):
    """Pick best feasible, save log. Returns (best_sol, fitness, archive, log)."""
    best_sol = state["best_sol"]
    return best_sol, state["best_fitness"], state["archive"], state["log"]
