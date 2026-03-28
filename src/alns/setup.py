"""ALNS setup helpers -- init state, weights, scoring, penalty updates."""
import numpy as np

from .config import make_config
from .acceptance import compute_initial_temperature
from .penalty import init_penalty_weights
from .pareto import create_pareto_archive, update_pareto_archive
from .logger import create_logger, log_weights_snapshot
from .destroy import DESTROY_OPS
from .repair import REPAIR_OPS
from .crosslayer import CROSS_OPS
from ..solution.structure import copy_solution
from ..engine.fitness import evaluate_solution
from ..data.cost import SYNC_DELTA_T


def setup(customers, config_overrides, seed):
    """Config + rng."""
    n_cust = len(customers)
    config = make_config(n_cust, config_overrides or {})
    rng = np.random.default_rng(seed)
    return config, rng


def eval_sol(sol, customers, restricted, vehicles, dist_matrix, state, config):
    """Helper: evaluate solution with current penalty weights."""
    pw = state["penalty_weights"]
    delta_t = config.get("delta_t", SYNC_DELTA_T)
    return evaluate_solution(sol, customers, restricted, vehicles,
                             dist_matrix, delta_t, pw)


def init_search_state(sol, customers, restricted, vehicles, dist_matrix,
                      config, rng):
    """Init all state: fitness, best, temperature, weights, log, archive."""
    pw, w3 = init_penalty_weights(config)
    state = {"penalty_weights": pw, "w3": w3}

    eval_result = eval_sol(sol, customers, restricted, vehicles,
                           dist_matrix, state, config)
    temp = compute_initial_temperature(sol, customers, restricted, vehicles,
                                       dist_matrix, config, rng)
    dw, rw, cw, scores, counts = _init_all_weights()

    return {
        "current_fitness": eval_result["fitness"],
        "best_sol": copy_solution(sol),
        "best_fitness": eval_result["fitness"],
        "temperature": temp,
        "w3": w3,
        "penalty_weights": pw,
        "destroy_w": dw, "repair_w": rw, "cross_w": cw,
        "scores": scores, "counts": counts,
        "archive": create_pareto_archive(),
        "log": create_logger(),
        "feasible_history": np.zeros(config["segment_length"], dtype=bool),
        "no_improve_count": 0,
        "last_eval": eval_result,
        "last_d_idx": 0, "last_r_idx": 0, "last_cx_idx": 0,
        "accepted": False,
        "iteration": 0,
    }


def _init_all_weights():
    """Initialize uniform weights for all operator groups."""
    nd, nr, nc = len(DESTROY_OPS), len(REPAIR_OPS), len(CROSS_OPS)
    dw = np.ones(nd, dtype=np.float64) / nd
    rw = np.ones(nr, dtype=np.float64) / nr
    cw = np.ones(nc, dtype=np.float64) / nc
    scores = {"destroy": np.zeros(nd), "repair": np.zeros(nr),
              "cross": np.zeros(nc)}
    counts = {"destroy": np.zeros(nd), "repair": np.zeros(nr),
              "cross": np.zeros(nc)}
    return dw, rw, cw, scores, counts


def classify_result(accepted, new_eval, state):
    """Return score: 5=new global best, 2=improving, 1=accepted worse, 0=rejected."""
    if new_eval["fitness"] < state["best_fitness"] - 1e-10:
        return 5
    if accepted and new_eval["fitness"] < state["current_fitness"] - 1e-10:
        return 2
    if accepted:
        return 1
    return 0


def update_op_scores(state, score):
    """Add score to the operators used this iteration."""
    state["scores"]["destroy"][state["last_d_idx"]] += score
    state["counts"]["destroy"][state["last_d_idx"]] += 1
    state["scores"]["repair"][state["last_r_idx"]] += score
    state["counts"]["repair"][state["last_r_idx"]] += 1
    if state.get("cross_used", False):
        state["scores"]["cross"][state["last_cx_idx"]] += score
        state["counts"]["cross"][state["last_cx_idx"]] += 1
        state["cross_used"] = False


def update_best(state, sol, new_eval):
    """Update best solution if improved."""
    if new_eval["fitness"] < state["best_fitness"] - 1e-10:
        state["best_sol"] = copy_solution(sol)
        state["best_fitness"] = new_eval["fitness"]
        state["no_improve_count"] = 0
    else:
        state["no_improve_count"] += 1


def update_pareto(state, eval_result, sol):
    """Try adding to Pareto archive."""
    update_pareto_archive(state["archive"], eval_result["cost"],
                          eval_result["makespan"], sol)


def update_feasible_history(state, iteration, eval_result, config):
    """Track feasibility in circular buffer."""
    idx = iteration % config["segment_length"]
    state["feasible_history"][idx] = eval_result["feasible"]


def update_weights_if_segment_end(state, iteration, config):
    """At end of segment, update operator weights and log snapshot."""
    if (iteration + 1) % config["segment_length"] != 0:
        return
    rf = config["reaction_factor"]
    for key, w_key in [("destroy", "destroy_w"), ("repair", "repair_w"),
                       ("cross", "cross_w")]:
        s, c = state["scores"][key], state["counts"][key]
        _apply_weight_update(state, w_key, s, c, rf)
        s[:] = 0
        c[:] = 0
    log_weights_snapshot(state["log"], state["destroy_w"],
                         state["repair_w"])


def _apply_weight_update(state, w_key, scores, counts, reaction_factor):
    """Update weights using reaction factor formula."""
    w = state[w_key]
    for i in range(len(w)):
        avg_score = scores[i] / max(counts[i], 1)
        w[i] = w[i] * (1 - reaction_factor) + reaction_factor * avg_score
    total = w.sum()
    if total > 1e-10:
        w /= total
    else:
        w[:] = 1.0 / len(w)
