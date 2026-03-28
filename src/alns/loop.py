"""Main ALNS loop — @njit core + thin Python wrapper for elite/pareto."""
import numpy as np
from numba import njit

from src.data.constants import (
    EV_FITNESS, EV_COST, EV_MAKESPAN, EV_TOTAL_PENALTY, EV_FEASIBLE,
    STF_CURRENT_FITNESS, STF_BEST_FITNESS, STF_TEMPERATURE,
    STF_W3, STF_LAST_D_IDX, STF_LAST_R_IDX, STF_LAST_CX_IDX,
    STF_ITERATION, STF_ACCEPTED, STF_CROSS_USED, STF_NO_IMPROVE_COUNT,
    STF_RESTART_COUNT, STF_BEST_FEAS_FITNESS, STF_HAS_BEST_FEASIBLE,
    STF_REHEAT_DONE, STF_HISTORY_BEST_COUNT, SOL_CUST_VEHICLE,
    CFG_MAX_ITERATIONS, CFG_NO_IMPROVE_LIMIT, CFG_LS_FREQUENCY,
    CFG_CROSS_FREQUENCY, CFG_SA_COOLING_RATE, CFG_QL_ENABLED,
    CFG_MAX_RESTARTS, CFG_SEGMENT_LENGTH, CFG_REACTION_FACTOR,
    CFG_DELTA_T,
)
from src.alns.acceptance import sa_accept, cool_temperature, maybe_reheat
from src.alns.penalty import adaptive_penalty_adjustment
from src.alns.logger import log_iteration, log_weights_snapshot
from src.alns.weights import select_operator
from src.alns.destroy import destroy_dispatch
from src.alns.repair import repair_dispatch
from src.alns.crosslayer import cross_dispatch
from src.alns.local_search import run_local_search
from src.alns.elite import maybe_restart_from_elite, maybe_add_to_pool
from src.alns.setup import (
    setup, eval_sol, init_search_state, classify_result,
    _apply_weight_update, _update_op_scores_inner,
)
from src.alns.pareto import update_pareto_archive
from src.init.builder import build_initial_solution
from src.solution.structure import copy_solution, copy_solution_into
from src.engine.fitness import evaluate_solution


def solve(customers, depot, vehicles, dist_matrix,
          n_trucks, n_bikes, config_overrides=None, seed=42, callback=None):
    """Main entry. Thin Python wrapper around @njit core loop."""
    config, rng = setup(customers, config_overrides, seed)
    sol = build_initial_solution(customers, depot, vehicles,
                                 dist_matrix, n_trucks, n_bikes, seed)
    state = init_search_state(sol, customers, vehicles, dist_matrix, config, rng)

    # Extract all arrays from state dict — no dict access in hot loop
    sf = state["state_f"]
    pw = state["penalty_w"]
    dw, rw, cw = state["destroy_w"], state["repair_w"], state["cross_w"]
    ds, dc = state["d_scores"], state["d_counts"]
    rs, rc = state["r_scores"], state["r_counts"]
    cs, cc = state["c_scores"], state["c_counts"]
    fh = state["feasible_history"]
    hf = state["history_freq"]
    log_data = state["log_data"]
    wlog_d, wlog_r = state["wlog_d"], state["wlog_r"]
    n_logged = state["n_logged"]
    n_wsnap = state["n_wsnap"]

    work_sol = copy_solution(sol)
    max_iter = int(config[CFG_MAX_ITERATIONS])

    # Generate all random seeds upfront (7 per iteration)
    all_seeds = rng.integers(0, 2**31, size=(max_iter, 7), dtype=np.int64)

    for iteration in range(max_iter):
        seeds = all_seeds[iteration]

        # @njit core: destroy, repair, LS, crosslayer, eval, accept, score, update
        (sol, new_ev, accepted, is_new_best, is_new_feas_best,
         n_logged, n_wsnap) = _iteration_njit(
            sol, work_sol, sf, pw, dw, rw, cw, ds, dc, rs, rc, cs, cc,
            fh, hf, customers, dist_matrix, vehicles, config,
            seeds, iteration, log_data, wlog_d, wlog_r,
            n_logged, n_wsnap, len(state["archive"]))

        # Python-only: elite pool + pareto (infrequent, use Python lists)
        if is_new_best:
            state["best_sol"] = copy_solution(sol)
            maybe_add_to_pool(state["elite_pool"], sol, new_ev[EV_FITNESS], config)
        if is_new_feas_best:
            state["best_feasible_sol"] = copy_solution(sol)

        update_pareto_archive(state["archive"], new_ev[EV_COST], new_ev[EV_MAKESPAN], sol)

        # Elite restart (infrequent)
        restarted = maybe_restart_from_elite(sf, state["elite_pool"], config, rng)
        if restarted is not None:
            sol = restarted
            rev = eval_sol(sol, customers, vehicles, dist_matrix, pw, config)
            sf[STF_CURRENT_FITNESS] = rev[EV_FITNESS]

        # Callback (optional, for viz)
        if callback is not None:
            from src.alns.logger import log_to_dict
            log_dict = log_to_dict(log_data, n_logged, wlog_d, wlog_r, n_wsnap)
            callback(iteration, state["best_sol"], new_ev, log_dict)

        if _should_stop(sf, config):
            break

    state["n_logged"] = n_logged
    state["n_wsnap"] = n_wsnap
    state["last_eval"] = new_ev
    return _finalize(sol, state, sf)


@njit(cache=True)
def _iteration_njit(sol, work_sol, sf, pw, dw, rw, cw, ds, dc, rs, rc, cs, cc,
                    fh, hf, customers, dist_matrix, vehicles, config,
                    seeds, iteration, log_data, wlog_d, wlog_r,
                    n_logged, n_wsnap, archive_size):
    """One full ALNS iteration. Fully @njit — zero Python overhead."""
    sf[STF_ITERATION] = iteration

    # --- Destroy & Repair ---
    copy_solution_into(sol, work_sol)
    d_idx = select_operator(dw, int(seeds[0]))
    r_idx = select_operator(rw, int(seeds[1]))
    removed = destroy_dispatch(d_idx, work_sol, customers, dist_matrix,
                               int(seeds[2]), config, hf)
    repair_dispatch(r_idx, work_sol, removed, customers, dist_matrix,
                    vehicles, int(seeds[3]), config)
    sf[STF_LAST_D_IDX] = d_idx
    sf[STF_LAST_R_IDX] = r_idx

    # --- Local Search ---
    ls_freq = int(config[CFG_LS_FREQUENCY])
    if ls_freq > 0 and iteration % ls_freq == 0:
        run_local_search(work_sol, dist_matrix, customers, vehicles)

    # --- Cross-Layer ---
    cx_freq = int(config[CFG_CROSS_FREQUENCY])
    if cx_freq > 0 and iteration % cx_freq == 0:
        cx_idx = select_operator(cw, int(seeds[4]))
        cross_dispatch(cx_idx, work_sol, customers, dist_matrix,
                       vehicles, np.int32(seeds[5]), config)
        sf[STF_LAST_CX_IDX] = cx_idx
        sf[STF_CROSS_USED] = 1.0

    # --- Evaluate ---
    delta_t = int(config[CFG_DELTA_T])
    new_ev = evaluate_solution(work_sol, customers, vehicles, dist_matrix, delta_t, pw)

    # --- Accept ---
    accepted = sa_accept(sf[STF_CURRENT_FITNESS], new_ev[EV_FITNESS],
                         sf[STF_TEMPERATURE], int(seeds[6]))

    # --- Score & Update Ops ---
    score = classify_result(accepted, new_ev, sf, config)
    _update_op_scores_inner(ds, dc, rs, rc, cs, cc, sf, score)

    if accepted:
        copy_solution_into(work_sol, sol)
        sf[STF_CURRENT_FITNESS] = new_ev[EV_FITNESS]

    # --- Update Best ---
    is_new_best = new_ev[EV_FITNESS] < sf[STF_BEST_FITNESS] - 1
    if is_new_best:
        sf[STF_BEST_FITNESS] = new_ev[EV_FITNESS]
        sf[STF_NO_IMPROVE_COUNT] = 0
        sf[STF_REHEAT_DONE] = 0.0
        # Update history freq
        sf[STF_HISTORY_BEST_COUNT] += 1
        cv = sol[SOL_CUST_VEHICLE]
        for i in range(min(len(hf), len(cv))):
            if cv[i] >= 0:
                hf[i] += 1
    else:
        sf[STF_NO_IMPROVE_COUNT] += 1

    is_new_feas_best = False
    if new_ev[EV_FEASIBLE] > 0 and new_ev[EV_FITNESS] < sf[STF_BEST_FEAS_FITNESS] - 1:
        sf[STF_BEST_FEAS_FITNESS] = new_ev[EV_FITNESS]
        sf[STF_HAS_BEST_FEASIBLE] = 1.0
        is_new_feas_best = True

    # --- Feasible History ---
    seg_len = int(config[CFG_SEGMENT_LENGTH])
    fh[iteration % seg_len] = new_ev[EV_FEASIBLE] > 0

    # --- Weight Update ---
    if seg_len > 0 and (iteration + 1) % seg_len == 0:
        rf = config[CFG_REACTION_FACTOR]
        _apply_weight_update(dw, ds, dc, rf)
        ds[:] = 0.0
        dc[:] = 0.0
        _apply_weight_update(rw, rs, rc, rf)
        rs[:] = 0.0
        rc[:] = 0.0
        _apply_weight_update(cw, cs, cc, rf)
        cs[:] = 0.0
        cc[:] = 0.0
        n_wsnap = log_weights_snapshot(wlog_d, wlog_r, n_wsnap, dw, rw)

    # --- Temperature ---
    sf[STF_TEMPERATURE] = cool_temperature(sf[STF_TEMPERATURE], config[CFG_SA_COOLING_RATE])
    maybe_reheat(sf, config)
    sf[STF_W3] = adaptive_penalty_adjustment(sf[STF_W3], fh, config)
    sf[STF_ACCEPTED] = 1.0 if accepted else 0.0

    # --- Log ---
    n_logged = log_iteration(
        log_data, n_logged, iteration,
        new_ev[EV_FITNESS], new_ev[EV_COST], new_ev[EV_MAKESPAN],
        new_ev[EV_TOTAL_PENALTY], new_ev[EV_FEASIBLE] > 0, accepted,
        sf[STF_TEMPERATURE], sf[STF_W3],
        int(sf[STF_LAST_D_IDX]), int(sf[STF_LAST_R_IDX]),
        sf[STF_BEST_FITNESS], archive_size)

    return (sol, new_ev, accepted, is_new_best, is_new_feas_best,
            n_logged, n_wsnap)


def _should_stop(sf, config):
    if sf[STF_NO_IMPROVE_COUNT] < config[CFG_NO_IMPROVE_LIMIT]:
        return False
    return sf[STF_RESTART_COUNT] >= config[CFG_MAX_RESTARTS]


def _finalize(sol, state, sf):
    if state["best_feasible_sol"] is not None:
        best_sol = state["best_feasible_sol"]
        best_fitness = sf[STF_BEST_FEAS_FITNESS]
    else:
        best_sol = state["best_sol"]
        best_fitness = sf[STF_BEST_FITNESS]
    from src.alns.logger import log_to_dict
    log_dict = log_to_dict(state["log_data"], state["n_logged"],
                           state["wlog_d"], state["wlog_r"], state["n_wsnap"])
    return best_sol, best_fitness, state["archive"], log_dict
