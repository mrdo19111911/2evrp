"""ALNS hyperparameters and scale-adaptive configuration."""

DEFAULT_CONFIG = {
    "max_iterations": 50000,
    "segment_length": 100,
    "no_improve_limit": 5000,
    "sa_initial_temp": 100.0,
    "sa_cooling_rate": 0.9997,
    "sa_min_temp": 0.01,
    "reaction_factor": 0.1,
    "sigma_1": 1,
    "sigma_2": 2,
    "sigma_3": 5,
    "penalty_w3_start": 10.0,
    "penalty_w3_end": 0.1,
    "penalty_decay_rate": 0.9999,
    "ls_frequency": 5,
    "ls_on_new_best": True,
    "cross_frequency": 10,
    "q_min": 3,
    "q_max_pct": 0.15,
    "shaw_randomness": 6.0,
    "worst_noise": 0.1,
    "zone_pct": 15,
    "regret_k": 3,
    "sat_threshold_km": 5.0,
}


def make_config(n_customers, overrides=None):
    """Scale-adaptive config. Returns merged dict."""
    config = DEFAULT_CONFIG.copy()

    # Scale-adaptive iteration limits
    if n_customers < 100:
        config["max_iterations"] = 20000
        config["no_improve_limit"] = 2000
    elif n_customers < 500:
        config["max_iterations"] = 50000
        config["no_improve_limit"] = 5000
    else:
        config["max_iterations"] = 100000
        config["no_improve_limit"] = 10000

    config["q_max"] = max(5, int(n_customers * config["q_max_pct"]))

    # Apply user overrides last
    if overrides:
        config.update(overrides)
    return config
