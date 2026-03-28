"""ALNS setup — init state, scoring, best tracking, weight updates."""
import numpy as np
from numba import njit

from src.data.constants import (
    EV_FITNESS, EV_COST, EV_MAKESPAN, EV_TOTAL_PENALTY, EV_FEASIBLE,
    STF_CURRENT_FITNESS, STF_BEST_FITNESS, STF_BEST_FEAS_FITNESS,
    STF_TEMPERATURE, STF_INITIAL_TEMP, STF_W3,
    STF_NO_IMPROVE_COUNT, STF_RESTART_COUNT, STF_HISTORY_BEST_COUNT,
    STF_LAST_D_IDX, STF_LAST_R_IDX, STF_LAST_CX_IDX,
    STF_ITERATION, STF_ACCEPTED, STF_REHEAT_DONE, STF_CROSS_USED,
    STF_HAS_BEST_FEASIBLE, STF_SIZE,
    CFG_SEGMENT_LENGTH, CFG_SIGMA_1, CFG_SIGMA_2, CFG_SIGMA_3,
    CFG_REACTION_FACTOR, CFG_MAX_ITERATIONS, CFG_DELTA_T, CFG_QL_ENABLED,
    N_DESTROY_OPS, N_REPAIR_OPS, N_CROSS_OPS,
    SOL_CUST_VEHICLE,
)
from src.alns.config import make_config
from src.alns.acceptance import compute_initial_temperature
from src.alns.penalty import init_penalty_weights
from src.alns.pareto import create_pareto_archive, update_pareto_archive
from src.alns.elite import create_elite_pool, maybe_add_to_pool
from src.alns.logger import create_logger, log_weights_snapshot
from src.solution.structure import copy_solution
from src.engine.fitness import evaluate_solution


def setup(customers, config_overrides, seed):
    """Config + rng."""
    n_cust = len(customers)
    config = make_config(n_cust, config_overrides)
    rng = np.random.default_rng(seed)
    return config, rng


@njit(cache=True)
def eval_sol(sol, customers, vehicles, dist_matrix, penalty_w, config):
    """Evaluate solution. No restricted. Returns i64 (EV_SIZE,)."""
    delta_t = int(config[CFG_DELTA_T])
    return evaluate_solution(sol, customers, vehicles, dist_matrix, delta_t, penalty_w)


def init_search_state(sol, customers, vehicles, dist_matrix, config, rng):
    """Init all state arrays. No restricted param. Returns dict."""
    penalty_w, w3 = init_penalty_weights(config)
    ev = eval_sol(sol, customers, vehicles, dist_matrix, penalty_w, config)
    temp = compute_initial_temperature(sol, customers, vehicles, dist_matrix, config, rng)

    state_f = np.zeros(STF_SIZE, dtype=np.float64)
    state_f[STF_CURRENT_FITNESS] = ev[EV_FITNESS]
    state_f[STF_BEST_FITNESS] = ev[EV_FITNESS]
    is_feas = ev[EV_FEASIBLE] > 0
    state_f[STF_BEST_FEAS_FITNESS] = ev[EV_FITNESS] if is_feas else np.inf
    state_f[STF_TEMPERATURE] = temp
    state_f[STF_INITIAL_TEMP] = temp
    state_f[STF_W3] = w3
    state_f[STF_HAS_BEST_FEASIBLE] = 1.0 if is_feas else 0.0

    nd, nr, nc = N_DESTROY_OPS, N_REPAIR_OPS, N_CROSS_OPS
    destroy_w = np.ones(nd, dtype=np.float64) / nd
    repair_w = np.ones(nr, dtype=np.float64) / nr
    cross_w = np.ones(nc, dtype=np.float64) / nc
    d_scores, d_counts = np.zeros(nd), np.zeros(nd)
    r_scores, r_counts = np.zeros(nr), np.zeros(nr)
    c_scores, c_counts = np.zeros(nc), np.zeros(nc)

    seg_len = int(config[CFG_SEGMENT_LENGTH])
    max_iter = int(config[CFG_MAX_ITERATIONS])
    log_data, wlog_d, wlog_r, n_logged, n_wsnap = create_logger(max_iter)

    n_cust = len(customers)
    state = {
        "state_f": state_f,
        "penalty_w": penalty_w,
        "destroy_w": destroy_w, "repair_w": repair_w, "cross_w": cross_w,
        "d_scores": d_scores, "d_counts": d_counts,
        "r_scores": r_scores, "r_counts": r_counts,
        "c_scores": c_scores, "c_counts": c_counts,
        "feasible_history": np.zeros(seg_len, dtype=bool),
        "history_freq": np.zeros(n_cust, dtype=np.int32),
        "best_sol": copy_solution(sol),
        "best_feasible_sol": copy_solution(sol) if is_feas else None,
        "archive": create_pareto_archive(),
        "elite_pool": create_elite_pool(),
        "log_data": log_data, "wlog_d": wlog_d, "wlog_r": wlog_r,
        "n_logged": n_logged, "n_wsnap": n_wsnap,
        "last_eval": ev,
    }

    maybe_add_to_pool(state["elite_pool"], sol, ev[EV_FITNESS], config)

    if config[CFG_QL_ENABLED] > 0.5:
        from src.alns.qlearning import init_qtable
        state["qtable"] = init_qtable(N_DESTROY_OPS, N_REPAIR_OPS)

    return state


@njit(cache=True)
def classify_result(accepted, new_ev, state_f, config):
    """Return score from config sigmas. Comparison threshold: < -1 (i64)."""
    if new_ev[EV_FITNESS] < state_f[STF_BEST_FITNESS] - 1:
        return config[CFG_SIGMA_3]
    if accepted and new_ev[EV_FITNESS] < state_f[STF_CURRENT_FITNESS] - 1:
        return config[CFG_SIGMA_2]
    if accepted:
        return config[CFG_SIGMA_1]
    return 0.0


def update_op_scores(state, state_f, score):
    """Add score to the operators used this iteration."""
    _update_op_scores_inner(
        state["d_scores"], state["d_counts"],
        state["r_scores"], state["r_counts"],
        state["c_scores"], state["c_counts"],
        state_f, score)


@njit(cache=True)
def _update_op_scores_inner(d_scores, d_counts, r_scores, r_counts,
                            c_scores, c_counts, state_f, score):
    d_idx = int(state_f[STF_LAST_D_IDX])
    r_idx = int(state_f[STF_LAST_R_IDX])
    d_scores[d_idx] += score
    d_counts[d_idx] += 1
    r_scores[r_idx] += score
    r_counts[r_idx] += 1
    if state_f[STF_CROSS_USED] > 0.5:
        cx_idx = int(state_f[STF_LAST_CX_IDX])
        c_scores[cx_idx] += score
        c_counts[cx_idx] += 1
        state_f[STF_CROSS_USED] = 0.0


def update_best(state, state_f, sol, new_ev, config):
    """Update best overall + best feasible + elite pool + history freq."""
    if new_ev[EV_FITNESS] < state_f[STF_BEST_FITNESS] - 1:
        state["best_sol"] = copy_solution(sol)
        state_f[STF_BEST_FITNESS] = new_ev[EV_FITNESS]
        state_f[STF_NO_IMPROVE_COUNT] = 0
        state_f[STF_REHEAT_DONE] = 0.0
        maybe_add_to_pool(state["elite_pool"], sol, new_ev[EV_FITNESS], config)
        _update_history_freq(state, sol)
    else:
        state_f[STF_NO_IMPROVE_COUNT] += 1

    if new_ev[EV_FEASIBLE] > 0 and new_ev[EV_FITNESS] < state_f[STF_BEST_FEAS_FITNESS] - 1:
        state["best_feasible_sol"] = copy_solution(sol)
        state_f[STF_BEST_FEAS_FITNESS] = new_ev[EV_FITNESS]
        state_f[STF_HAS_BEST_FEASIBLE] = 1.0


def _update_history_freq(state, sol):
    state_f = state["state_f"]
    state_f[STF_HISTORY_BEST_COUNT] += 1
    hf = state["history_freq"]
    hf[np.where(sol[SOL_CUST_VEHICLE][:len(hf)] >= 0)[0]] += 1

def update_pareto(state, ev, sol):
    update_pareto_archive(state["archive"], ev[EV_COST], ev[EV_MAKESPAN], sol)

def update_feasible_history(state, iteration, ev, config):
    idx = iteration % int(config[CFG_SEGMENT_LENGTH])
    state["feasible_history"][idx] = ev[EV_FEASIBLE] > 0


def update_weights_if_segment_end(state, state_f, iteration, config):
    seg = int(config[CFG_SEGMENT_LENGTH])
    if (iteration + 1) % seg != 0:
        return
    rf = config[CFG_REACTION_FACTOR]
    for s_key, c_key, w_key in [("d_scores", "d_counts", "destroy_w"),
                                 ("r_scores", "r_counts", "repair_w"),
                                 ("c_scores", "c_counts", "cross_w")]:
        _apply_weight_update(state[w_key], state[s_key], state[c_key], rf)
        state[s_key][:] = 0
        state[c_key][:] = 0
    state["n_wsnap"] = log_weights_snapshot(
        state["wlog_d"], state["wlog_r"], state["n_wsnap"],
        state["destroy_w"], state["repair_w"])


@njit(cache=True)
def _apply_weight_update(w, scores, counts, reaction_factor):
    for i in range(len(w)):
        avg = scores[i] / max(counts[i], 1.0)
        w[i] = w[i] * (1 - reaction_factor) + reaction_factor * avg
    np.maximum(w, 0.01, w)
    total = 0.0
    for i in range(len(w)):
        total += w[i]
    if total > 1e-10:
        for i in range(len(w)):
            w[i] /= total
    else:
        n = len(w)
        for i in range(n):
            w[i] = 1.0 / n
