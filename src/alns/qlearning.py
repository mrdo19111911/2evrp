"""Q-Learning operator selection — replaces roulette wheel when enabled."""
import numpy as np


def init_qtable(n_destroy, n_repair, n_phases=5, n_feas_buckets=2):
    n_states = n_phases * n_feas_buckets
    n_actions = n_destroy * n_repair
    return {
        "q": np.zeros((n_states, n_actions), dtype=np.float64),
        "n_destroy": n_destroy,
        "n_repair": n_repair,
        "n_phases": n_phases,
        "n_feas_buckets": n_feas_buckets,
        "epsilon": 0.15,
        "prev_state": 0,
        "prev_action": 0,
    }


def compute_state(iteration, max_iterations, feasible_history, config, qt):
    n_phases = qt["n_phases"]
    phase = min(n_phases - 1, iteration * n_phases // max(max_iterations, 1))
    seg_len = config["segment_length"]
    n_total = min(iteration + 1, seg_len)
    n_feasible = int(np.sum(feasible_history[:n_total]))
    feas_bucket = 1 if n_feasible / max(n_total, 1) > 0.5 else 0
    return phase * qt["n_feas_buckets"] + feas_bucket


def select_action(qt, state, rng, epsilon):
    n_actions = qt["n_destroy"] * qt["n_repair"]
    if rng.random() < epsilon:
        action_flat = int(rng.integers(n_actions))
    else:
        q_row = qt["q"][state]
        max_q = np.max(q_row)
        candidates = np.where(np.abs(q_row - max_q) < 1e-10)[0]
        action_flat = int(rng.choice(candidates))
    d_idx = action_flat // qt["n_repair"]
    r_idx = action_flat % qt["n_repair"]
    return d_idx, r_idx, action_flat


def compute_reward(new_eval, state_dict):
    old_fitness = state_dict["current_fitness"]
    new_fitness = new_eval["fitness"]
    delta = old_fitness - new_fitness
    scale = max(abs(old_fitness), 1.0)
    return max(-1.0, min(1.0, delta / scale))


def update_qtable(qt, prev_state, prev_action, reward, new_state, alpha, gamma):
    best_next = np.max(qt["q"][new_state])
    qt["q"][prev_state, prev_action] = (
        (1 - alpha) * qt["q"][prev_state, prev_action]
        + alpha * (reward + gamma * best_next)
    )


def decay_epsilon(qt, config):
    qt["epsilon"] = max(
        config.get("ql_epsilon_min", 0.02),
        qt["epsilon"] * config.get("ql_epsilon_decay", 0.9999)
    )
