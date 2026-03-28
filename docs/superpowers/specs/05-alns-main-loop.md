# Phase 05 — ALNS Main Loop

## File structure

```
src/
  alns/
    acceptance.py     ← SA acceptance criterion
    penalty.py        ← adaptive penalty weight scheduling
    pareto.py         ← Pareto archive
    loop.py           ← main ALNS loop (orchestrator)
    config.py         ← hyperparameters, default configs
    logger.py         ← iteration logging cho debug + visualization
```

---

## config.py

### Hyperparameters

```python
DEFAULT_CONFIG = {
    # === Main loop ===
    "max_iterations": 50000,
    "segment_length": 100,        # cập nhật weights mỗi 100 iterations
    "no_improve_limit": 5000,     # dừng nếu 5000 iter không cải thiện

    # === Simulated Annealing ===
    "sa_initial_temp": 100.0,
    "sa_cooling_rate": 0.9997,
    "sa_min_temp": 0.01,

    # === Adaptive weights ===
    "reaction_factor": 0.1,       # tốc độ cập nhật operator weights
    "sigma_1": 1,                 # score: accepted worse
    "sigma_2": 2,                 # score: new local best
    "sigma_3": 5,                 # score: new global best

    # === Penalty scheduling ===
    "penalty_w3_start": 10.0,     # penalty weight ban đầu (cao → ép feasible)
    "penalty_w3_end": 0.1,        # penalty weight cuối
    "penalty_decay_rate": 0.9999, # decay per iteration

    # === Fitness weights ===
    "w1_cost": 0.6,
    "w2_makespan": 0.3,

    # === Local search ===
    "ls_frequency": 5,            # chạy local search mỗi 5 iterations
    "ls_on_new_best": True,       # luôn chạy LS khi tìm được best mới

    # === Cross-layer ===
    "cross_frequency": 10,        # thử cross-layer op mỗi 10 iterations

    # === Destroy params ===
    "q_min": 3,
    "q_max_pct": 0.15,           # q_max = 15% of N
    "shaw_randomness": 6.0,
    "worst_noise": 0.1,
    "zone_pct": 15,

    # === Repair params ===
    "regret_k": 3,
    "sat_threshold_km": 5.0,

    # === Scale-adaptive ===
    # Tự động điều chỉnh theo N:
    # N < 100:   max_iterations = 20000
    # N < 500:   max_iterations = 50000
    # N >= 500:  max_iterations = 100000
}
```

### make_config

- **Idea:** Tạo config với scale-adaptive adjustments.
- **Input:**
  - `n_customers: int`
  - `overrides: dict` — user muốn override giá trị nào
- **Output:**
  - `config: dict` — merged config
- **Logic:**
  ```
  config = DEFAULT_CONFIG.copy()

  # Scale-adaptive
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

  # Apply overrides
  config.update(overrides)
  return config
  ```
- **Test cases:**
  - TC1: N=50 → max_iterations=20000
  - TC2: N=1000 → max_iterations=100000, q_max=150
  - TC3: Override max_iterations=999 → config["max_iterations"]=999

---

## acceptance.py

### sa_accept

- **Idea:** Simulated Annealing acceptance. Luôn accept cải thiện, accept xấu hơn với xác suất giảm dần.
- **Input:**
  - `current_fitness: float`
  - `new_fitness: float`
  - `temperature: float`
  - `rng: Generator`
- **Output:**
  - `accept: bool`
- **Logic:**
  ```
  if new_fitness < current_fitness:
      return True  # cải thiện → luôn accept

  if temperature < 1e-10:
      return False  # đã nguội hoàn toàn

  delta = new_fitness - current_fitness
  prob = exp(-delta / temperature)
  return rng.random() < prob
  ```
- **Test cases:**
  - TC1: new < current → True (luôn)
  - TC2: new = current → True (delta=0, prob=1)
  - TC3: new >> current, temp cao → prob ~1 → có thể accept
  - TC4: new >> current, temp thấp → prob ~0 → reject
  - TC5: temp=0 → False (trừ khi cải thiện)

---

### cool_temperature

- **Idea:** Giảm nhiệt độ geometric.
- **Input:**
  - `temperature: float`
  - `cooling_rate: float`
- **Output:**
  - `new_temp: float`
- **Logic:** `max(temperature * cooling_rate, sa_min_temp)`
- **Test cases:**
  - TC1: 100 * 0.9997 = 99.97
  - TC2: Không giảm dưới min_temp

---

### compute_initial_temperature

- **Idea:** Tính nhiệt độ ban đầu tự động từ initial solution. Mục tiêu: ~80% worse solutions được accept ở iteration đầu.
- **Input:**
  - `sol: dict` — initial solution
  - `customers, dist_matrix, restricted, vehicles: ndarray`
  - `config: dict`
  - `rng: Generator`
  - `target_accept_rate: float` — default 0.8
  - `n_samples: int` — default 100
- **Output:**
  - `initial_temp: float`
- **Logic:**
  ```
  # Thử random perturbations, đo distribution of delta fitness
  deltas = []
  for _ in range(n_samples):
      sol_copy = copy_solution(sol)
      # Random destroy-repair
      destroy_op = random_removal
      removed = destroy_op(sol_copy, customers, dist_matrix, rng,
                           {"q_min": 1, "q_max": 3})
      repair_op = greedy_insertion
      repair_op(sol_copy, removed, customers, restricted, dist_matrix,
                vehicles, rng, {})

      old_fit = evaluate_solution(sol, ...)["fitness"]
      new_fit = evaluate_solution(sol_copy, ...)["fitness"]

      if new_fit > old_fit:
          deltas.append(new_fit - old_fit)

  if not deltas:
      return config["sa_initial_temp"]

  # T sao cho exp(-avg_delta / T) = target_accept_rate
  avg_delta = mean(deltas)
  T = -avg_delta / log(target_accept_rate)
  return T
  ```
- **Test cases:**
  - TC1: Deltas nhỏ → T nhỏ (solution gần optimal, không cần explore nhiều)
  - TC2: Deltas lớn → T lớn (solution kém, cần explore)
  - TC3: Không delta nào (mọi perturbation đều cải thiện) → return default

---

## penalty.py

### init_penalty_weights

- **Idea:** Khởi tạo penalty weights. Ban đầu cao (ép feasible), giảm dần.
- **Input:**
  - `config: dict`
- **Output:**
  - `penalty_weights: dict` — weights cho compute_penalties
  - `w3: float` — fitness penalty multiplier
- **Logic:**
  ```
  penalty_weights = {
      "unserved": PENALTY_UNSERVED,
      "duplicate": PENALTY_UNSERVED,
      "capacity": PENALTY_OVERLOAD,
      "time_window": PENALTY_LATE,
      "sync": PENALTY_SYNC_FAIL,
      "vehicle_restriction": PENALTY_RESTRICTION,
  }
  w3 = config["penalty_w3_start"]
  return penalty_weights, w3
  ```

---

### decay_penalty_weight

- **Idea:** Giảm w3 mỗi iteration.
- **Input:**
  - `w3: float`, `config: dict`
- **Output:**
  - `new_w3: float`
- **Logic:** `max(w3 * config["penalty_decay_rate"], config["penalty_w3_end"])`
- **Test cases:**
  - TC1: 10.0 * 0.9999 = 9.999
  - TC2: Không dưới penalty_w3_end

---

### adaptive_penalty_adjustment

- **Idea:** Nếu solution infeasible quá lâu → tăng w3 lại. Nếu feasible ổn định → giảm nhanh hơn. Feedback loop.
- **Input:**
  - `w3: float`
  - `feasible_history: ndarray bool` — lịch sử feasibility gần đây (vd 100 iter gần nhất)
  - `config: dict`
- **Output:**
  - `new_w3: float`
- **Logic:**
  ```
  feasible_pct = feasible_history.mean()

  if feasible_pct < 0.3:
      # Quá ít feasible → tăng penalty gấp đôi
      new_w3 = min(w3 * 2.0, config["penalty_w3_start"])
  elif feasible_pct > 0.8:
      # Đa số feasible → giảm nhanh hơn
      new_w3 = w3 * config["penalty_decay_rate"] ** 5
  else:
      # Bình thường
      new_w3 = w3 * config["penalty_decay_rate"]

  return max(new_w3, config["penalty_w3_end"])
  ```
- **Test cases:**
  - TC1: 10% feasible → w3 tăng gấp đôi
  - TC2: 90% feasible → w3 giảm nhanh
  - TC3: 50% feasible → w3 giảm bình thường

---

## pareto.py

### create_pareto_archive

- **Idea:** Tạo Pareto archive rỗng.
- **Output:**
  - `archive: list` — list of (cost, makespan, solution_copy)
  - max_size: int — giới hạn kích thước, default 50

---

### update_pareto_archive

- **Idea:** Thêm solution vào archive nếu nó non-dominated. Xoá solutions bị dominate.
- **Input:**
  - `archive: list`
  - `new_cost: float`, `new_makespan: float`
  - `new_sol: dict` — solution để lưu (sẽ bị copy)
  - `max_size: int`
- **Output:**
  - `updated: bool` — có thêm vào archive không
- **Logic:**
  ```
  # Check new bị dominate?
  for (c, m, _) in archive:
      if c <= new_cost and m <= new_makespan and (c < new_cost or m < new_makespan):
          return False  # bị dominate → skip

  # Xoá solutions bị new dominate
  archive[:] = [(c, m, s) for (c, m, s) in archive
                if not (new_cost <= c and new_makespan <= m
                        and (new_cost < c or new_makespan < m))]

  # Thêm new
  archive.append((new_cost, new_makespan, copy_solution(new_sol)))

  # Nếu archive quá lớn → loại solution ít diverse nhất (crowding distance)
  if len(archive) > max_size:
      _prune_archive(archive, max_size)

  return True
  ```
- **Test cases:**
  - TC1: First solution → luôn add
  - TC2: New dominate existing → existing bị xoá
  - TC3: New bị dominate → không add
  - TC4: Non-dominated (trade-off) → add
  - TC5: Archive full → prune bằng crowding distance

---

### get_pareto_front

- **Idea:** Trích Pareto front từ archive dưới dạng arrays (cho visualization).
- **Input:** `archive: list`
- **Output:**
  - `costs: ndarray float64 (P,)` — sorted by cost ascending
  - `makespans: ndarray float64 (P,)`
- **Logic:**
  ```
  if not archive:
      return empty(0), empty(0)
  data = array([(c, m) for c, m, _ in archive])
  order = argsort(data[:, 0])
  return data[order, 0], data[order, 1]
  ```

---

## logger.py

### create_logger

- **Idea:** Tạo logger lưu lịch sử mỗi iteration. Dùng cho visualization + debug.
- **Output:**
  - `log: dict` chứa:
    ```python
    {
        "iterations": [],       # iteration number
        "fitness": [],          # fitness value
        "cost": [],             # total cost
        "makespan": [],         # makespan
        "penalty": [],          # total penalty
        "feasible": [],         # bool
        "temperature": [],      # SA temperature
        "w3": [],               # penalty weight
        "destroy_op": [],       # destroy operator index used
        "repair_op": [],        # repair operator index used
        "accepted": [],         # bool: accepted or rejected
        "best_fitness": [],     # best fitness so far
        "destroy_weights": [],  # snapshot weights (mỗi segment)
        "repair_weights": [],
        "pareto_size": [],      # Pareto archive size
    }
    ```

---

### log_iteration

- **Idea:** Ghi 1 iteration vào log.
- **Input:**
  - `log: dict`
  - `iteration: int`
  - `fitness, cost, makespan, penalty: float`
  - `feasible, accepted: bool`
  - `temperature, w3: float`
  - `destroy_idx, repair_idx: int`
  - `best_fitness: float`
  - `pareto_size: int`
- **Output:** None (append to log)
- **Performance:** O(1)

---

### log_weights_snapshot

- **Idea:** Snapshot operator weights mỗi segment.
- **Input:**
  - `log: dict`
  - `destroy_weights, repair_weights: ndarray`
- **Output:** None

---

### save_log

- **Idea:** Lưu log ra JSON file.
- **Input:** `log: dict`, `filepath: str`
- **Output:** None
- **Test cases:**
  - TC1: Save → load → data identical

---

## loop.py — Main ALNS Loop (Short Orchestrator Pattern)

### solve — Orchestrator (10 dòng logic)

```python
def solve(customers, restricted, depot, vehicles, dist_matrix,
          n_trucks, n_bikes, config_overrides=None, seed=42, callback=None):

    config, rng = setup(customers, config_overrides, seed)
    sol = build_initial_solution(customers, restricted, depot, vehicles,
                                 dist_matrix, n_trucks, n_bikes, seed)
    state = init_search_state(sol, customers, restricted, vehicles,
                               dist_matrix, config)

    for iteration in range(config["max_iterations"]):
        new_sol = destroy_and_repair(sol, state, customers, restricted,
                                      dist_matrix, vehicles, rng, config)
        new_sol = maybe_local_search(new_sol, iteration, state, dist_matrix,
                                      customers, config)
        new_sol = maybe_cross_layer(new_sol, iteration, customers, restricted,
                                     dist_matrix, rng, config)
        sol, state = accept_and_update(sol, new_sol, state, rng, config)
        log_and_callback(state, iteration, callback)

        if should_stop(state, config):
            break

    return finalize(sol, state)
```

- **Input:**
  - `customers: ndarray float64 (N, F)`
  - `restricted: ndarray int8 (N,)`
  - `depot: ndarray float64 (2,)`
  - `vehicles: ndarray float64 (K, 4)`
  - `dist_matrix: ndarray float64 (N+1, N+1)`
  - `n_trucks, n_bikes: int`
  - `config_overrides: dict` — optional
  - `seed: int`
  - `callback: callable` — optional, signature: `callback(iteration, sol, eval_result, log)`
- **Output:**
  - `best_sol: dict`, `best_fitness: float`, `pareto_archive: list`, `log: dict`

---

### Sub-functions

#### setup

```python
def setup(customers, config_overrides, seed):
    config = make_config(len(customers), config_overrides or {})
    rng = np.random.default_rng(seed)
    return config, rng
```

---

#### init_search_state

```python
def init_search_state(sol, customers, restricted, vehicles, dist_matrix, config):
    eval_result = evaluate_solution(sol, ...)
    return {
        "current_fitness": eval_result["fitness"],
        "best_sol": copy_solution(sol),
        "best_fitness": eval_result["fitness"],
        "temperature": compute_initial_temperature(sol, ...),
        "w3": config["penalty_w3_start"],
        "penalty_weights": init_penalty_weights(config),
        "destroy_w": uniform_weights(len(DESTROY_OPS)),
        "repair_w": uniform_weights(len(REPAIR_OPS)),
        "cross_w": uniform_weights(len(CROSS_OPS)),
        "scores": init_scores(),
        "archive": create_pareto_archive(),
        "log": create_logger(),
        "feasible_history": zeros(config["segment_length"], dtype=bool),
        "no_improve_count": 0,
        "last_eval": eval_result,
    }
```

---

#### destroy_and_repair

```python
def destroy_and_repair(sol, state, customers, restricted, dist_matrix,
                        vehicles, rng, config):
    new_sol = copy_solution(sol)
    d_idx = select_operator(state["destroy_w"], rng)
    removed = DESTROY_OPS[d_idx](new_sol, customers, dist_matrix, rng, config)
    r_idx = select_operator(state["repair_w"], rng)
    REPAIR_OPS[r_idx](new_sol, removed, customers, restricted,
                       dist_matrix, vehicles, rng, config)
    state["last_d_idx"] = d_idx
    state["last_r_idx"] = r_idx
    return new_sol
```

---

#### maybe_local_search

```python
def maybe_local_search(sol, iteration, state, dist_matrix, customers, config):
    if iteration % config["ls_frequency"] != 0:
        return sol
    run_local_search(sol, dist_matrix, customers)
    return sol
```

#### run_local_search

```python
def run_local_search(sol, dist_matrix, customers):
    for vtype in [VEH_TRUCK, VEH_BIKE]:
        n_v = sol["n_trucks"] if vtype == VEH_TRUCK else sol["n_bikes"]
        for vid in range(n_v):
            two_opt(sol, vtype, vid, dist_matrix)
            or_opt(sol, vtype, vid, dist_matrix)
        relocate_inter_route(sol, vtype, dist_matrix, customers)
```

---

#### maybe_cross_layer

```python
def maybe_cross_layer(sol, iteration, customers, restricted, dist_matrix,
                       rng, config):
    if iteration % config["cross_frequency"] != 0:
        return sol
    cx_idx = select_operator(state["cross_w"], rng)
    CROSS_OPS[cx_idx](sol, customers, restricted, dist_matrix,
                       sol["satellites"], rng, {})
    return sol
```

---

#### accept_and_update

```python
def accept_and_update(sol, new_sol, state, rng, config):
    new_eval = evaluate_solution(new_sol, ...)
    accepted = sa_accept(state["current_fitness"], new_eval["fitness"],
                          state["temperature"], rng)

    score = classify_result(accepted, new_eval, state)
    update_operator_scores(state, score)

    if accepted:
        sol = new_sol
        state["current_fitness"] = new_eval["fitness"]

    update_best(state, sol, new_eval)
    update_pareto(state, new_eval, sol)
    update_weights_if_segment_end(state, config)
    state["temperature"] = cool_temperature(state["temperature"], config["sa_cooling_rate"])
    state["w3"] = adaptive_penalty_adjustment(state["w3"], state["feasible_history"], config)
    state["last_eval"] = new_eval

    return sol, state
```

---

#### log_and_callback

```python
def log_and_callback(state, iteration, callback):
    log_iteration(state["log"], iteration, state["last_eval"], state)
    if callback is not None:
        callback(iteration, state["best_sol"], state["last_eval"], state["log"])
```

---

#### should_stop

```python
def should_stop(state, config):
    return state["no_improve_count"] >= config["no_improve_limit"]
```

---

#### finalize

```python
def finalize(sol, state):
    best_sol = pick_best_feasible(state["best_sol"], state["archive"])
    save_log(state["log"], "alns_log.json")
    return best_sol, state["best_fitness"], state["archive"], state["log"]
```

---

- **Test cases:**
  - TC1: **Tiny instance (5 customers)** → converge, best feasible, pareto ≥ 1
  - TC2: **Deterministic** — cùng seed → cùng result
  - TC3: **best_fitness monotone non-increasing**
  - TC4: **Pareto non-dominated**
  - TC5: **Early stopping** khi no_improve_limit hit
  - TC6: **Callback called** mỗi iteration
  - TC7: **Log length == iterations run**
  - TC8: **N=100, 20k iter** → < 5 phút
  - TC9: **Infeasible initial → feasible best**
  - TC10: **best_sol satisfies all constraints** (nếu feasible)

---

## Tổng kết Phase 05

| File | Function | Dòng logic | Mục đích |
|------|----------|------------|----------|
| **config.py** | | | |
| | make_config | ~10 | Tạo config scale-adaptive |
| **acceptance.py** | | | |
| | sa_accept | 5 | SA acceptance criterion |
| | cool_temperature | 1 | Giảm nhiệt geometric |
| | compute_initial_temperature | ~15 | Auto-calibrate T₀ |
| **penalty.py** | | | |
| | init_penalty_weights | 5 | Khởi tạo penalty weights |
| | decay_penalty_weight | 1 | Giảm w3 |
| | adaptive_penalty_adjustment | 8 | Feedback: tăng/giảm w3 |
| **pareto.py** | | | |
| | create_pareto_archive | 1 | Tạo archive rỗng |
| | update_pareto_archive | ~10 | Thêm non-dominated solution |
| | get_pareto_front | 3 | Trích arrays |
| **logger.py** | | | |
| | create_logger | 3 | Tạo log structure |
| | log_iteration | 3 | Ghi 1 iteration |
| | log_weights_snapshot | 2 | Snapshot weights |
| | save_log | 2 | Lưu JSON |
| **loop.py** | | | |
| | **solve** | **10** | **Main orchestrator** |
| | setup | 3 | Config + rng |
| | init_search_state | 15 | Init tất cả state |
| | destroy_and_repair | 7 | Copy → destroy → repair |
| | maybe_local_search | 3 | Conditional LS |
| | run_local_search | 5 | LS all routes |
| | maybe_cross_layer | 4 | Conditional cross-layer |
| | accept_and_update | 10 | Accept → update state |
| | log_and_callback | 3 | Log + callback |
| | should_stop | 1 | Early stopping check |
| | finalize | 4 | Post-processing |

**Tổng: 24 functions.** Orchestrator `solve()` = 10 dòng. Mọi sub-function ≤ 15 dòng.

**Flow:**
```
solve:  setup → init → [destroy_repair → LS → cross → accept_update → log] → finalize
```
