# Interface Specification — Fixed Types for Numba

Single source of truth for ALL @njit function signatures.
Every input/output MUST match these exact dtypes. No exceptions.

## Units

| Quantity | Unit | Example |
|----------|------|---------|
| Distance | **meters** (int64) | 5km = 5000 |
| Time | **seconds** (int64) | 8 hours = 28800 |
| Weight | **grams** (int64) | 60kg = 60000 |
| Cost | **VND** (int64) | 12000 VND/km = 12 VND/m |
| Coordinates | **meters** (int64) | GPS projected to meters |
| Speed | pre-computed as **seconds per meter** (int64, ×1e6 fixed-point) | 25 km/h = 144 s/km = 144000 µs/m |

All quantities are **integer** to eliminate floating-point rounding.
Costs, penalties, fitness are all int64 VND (no fractional dong).

## Canonical Dtypes

| Name | Numpy dtype | Usage |
|------|-------------|-------|
| `i64` | `np.int64` | distances (m), times (s), costs (VND), weights (g), coordinates (m) |
| `i32` | `np.int32` | customer IDs, vehicle IDs, positions, counts |
| `i8` | `np.int8` | actions (0=DELIVER, 1=RELOAD, -1=PAD), vtype (0=TRUCK, 1=BIKE) |

No `float64` in hot path. Float only in: config array, penalty weights (tuning knobs), visualization.

## Solution Tuple — `sol`

15 elements, fixed order.

```
sol = (
    i32[n_trucks, max_route_len],     # [0]  truck_stops — customer indices, -1=pad
    i8[n_trucks, max_route_len],      # [1]  truck_actions
    i32[n_bikes, max_route_len],      # [2]  bike_stops
    i8[n_bikes, max_route_len],       # [3]  bike_actions
    i32[n_trucks],                     # [4]  truck_lengths
    i32[n_bikes],                      # [5]  bike_lengths
    i64[n_trucks],                     # [6]  truck_loads (grams)
    i64[n_bikes],                      # [7]  bike_loads (grams)
    i64[n_trucks],                     # [8]  truck_distances (meters)
    i64[n_bikes],                      # [9]  bike_distances (meters)
    i32[n_customers],                  # [10] cust_vehicle — global vehicle id, -1=unassigned
    i8[n_customers],                   # [11] cust_vtype
    i32[n_customers],                  # [12] cust_route_pos
    i64[MAX_SATELLITES, 5],            # [13] satellites — [cust, bike, truck, grams, time_s]
    i32[5],                            # [14] sol_meta — [n_trucks, n_bikes, max_route_len, n_customers, n_satellites]
)
```

## Common Arguments

| Arg name | Type | Shape | Unit |
|----------|------|-------|------|
| `sol` | tuple | 15 arrays as above | — |
| `customers` | `i64` | `(N, 7)` | [x_m, y_m, demand_g, tw_open_s, tw_close_s, service_s, restricted] |
| `dist_matrix` | `i64` | `(N+1, N+1)` | meters — row/col 0=depot |
| `vehicles` | `i64` | `(K, 4)` | [type, capacity_g, cost_per_m, speed_us_per_m] |
| `config` | `f64` | `(CFG_SIZE,)` | tuning knobs (float OK, not hot path) |
| `penalty_w` | `i64` | `(PW_SIZE,)` | VND per violation |
| `vtype` | `i32` | scalar | VEH_TRUCK=0, VEH_BIKE=1 |
| `vid` | `i32` | scalar | vehicle index within type |
| `pos` | `i32` | scalar | position in route |
| `customer` | `i32` | scalar | customer index |
| `seed` | `i32` | scalar | RNG seed |

Note: `restricted` is now column 6 of `customers` (i64), not a separate i8 array.
This reduces argument count in every function.

## @njit Functions

### solution/_helpers.py

```
get_route_arrays(sol, vtype: i32) -> (i32[:,:], i8[:,:], i32[:])
get_loads(sol, vtype: i32) -> i64[:]
get_distances(sol, vtype: i32) -> i64[:]
global_vid(vtype: i32, vid: i32, n_trucks: i32) -> i32
update_route_distance(sol, vtype: i32, vid: i32, dist_matrix: i64[:,:]) -> None
update_route_load(sol, vtype: i32, vid: i32, customers: i64[:,:]) -> None
```

### solution/route_ops.py

```
insert_stop(sol, vtype: i32, vid: i32, pos: i32, customer: i32, action: i32, dist_matrix: i64[:,:], customers: i64[:,:]) -> None
remove_stop(sol, vtype: i32, vid: i32, pos: i32, dist_matrix: i64[:,:], customers: i64[:,:]) -> (i32, i32)
swap_stops_within(sol, vtype: i32, vid: i32, pos_a: i32, pos_b: i32, dist_matrix: i64[:,:]) -> None
reverse_segment(sol, vtype: i32, vid: i32, start: i32, end: i32, dist_matrix: i64[:,:]) -> None
```

### solution/query.py

```
get_unassigned_customers(sol, n_customers: i32) -> i32[:]
get_assigned_customers(sol, n_customers: i32) -> i32[:]
get_route_customers_only(sol, vtype: i32, vid: i32) -> i32[:]
is_customer_assigned(sol, customer: i32) -> bool
get_customer_info(sol, customer: i32) -> (i32, i32, i32)
```

### solution/check.py

```
can_insert_customer(sol, vtype: i32, vid: i32, customer: i32, customers: i64[:,:], vehicle_capacity: i64) -> bool
can_swap_customers(sol, cust_a: i32, cust_b: i32, customers: i64[:,:]) -> bool
check_route_capacity_quick(sol, vtype: i32, vid: i32, vehicle_capacity: i64) -> bool
check_all_assigned(sol, n_customers: i32) -> bool
```

Note: `restricted` removed from check args — read from `customers[c, 6]`.

### solution/delta.py

```
insertion_cost_delta(sol, vtype: i32, vid: i32, pos: i32, customer: i32, dist_matrix: i64[:,:]) -> i64
removal_cost_delta(sol, vtype: i32, vid: i32, pos: i32, dist_matrix: i64[:,:]) -> i64
best_insertion_pos(sol, vtype: i32, vid: i32, customer: i32, dist_matrix: i64[:,:]) -> (i32, i64)
_estimate_arrival_at_pos(sol, vtype: i32, vid: i32, pos: i32, dist_matrix: i64[:,:], customers: i64[:,:], vehicles: i64[:,:]) -> (i64, i32)
_estimate_route_return_time(sol, vtype: i32, vid: i32, dist_matrix: i64[:,:], customers: i64[:,:], vehicles: i64[:,:]) -> i64
_check_tw_at_insertion(sol, vtype: i32, vid: i32, pos: i32, customer: i32, dist_matrix: i64[:,:], customers: i64[:,:], vehicles: i64[:,:]) -> bool
cascade_removal_value(sol, vtype: i32, vid: i32, pos: i32, dist_matrix: i64[:,:], customers: i64[:,:], vehicles: i64[:,:]) -> i64
find_best_insertion_all_routes(sol, vtype: i32, customer: i32, dist_matrix: i64[:,:], customers: i64[:,:], vehicles: i64[:,:]) -> (i32, i32, i64)
```

Note: `vehicle_capacity` and `speed` now come from `vehicles` array, not from global constants.

### solution/satellite_ops.py

```
add_satellite_event(sol, customer_idx: i32, bike_id: i32, truck_id: i32, transfer_g: i64, planned_time_s: i64) -> None
remove_satellite_by_idx(sol, sat_idx: i32) -> None
remove_satellites_for_customer(sol, customer_idx: i32) -> i32
remove_satellites_for_bike(sol, bike_id: i32) -> i32
update_satellite_time(sol, sat_idx: i32, new_time_s: i64) -> None
get_satellites(sol) -> i64[:,:]
```

### solution/structure.py

```
copy_solution_into(src: sol, dst: sol) -> None
```

### engine/simulate.py

```
simulate_route_into(out_state: i64[:,:], stops: i32[:], actions: i8[:], length: i32, vehicle_type: i32, vehicle_capacity: i64, speed_us_per_m: i64, customers: i64[:,:], dist_matrix: i64[:,:], satellites: i64[:,:], n_satellites: i32, vehicle_id: i32, delta_t_s: i64) -> (i64, i64, bool)
```

Returns: (return_time_s, total_dist_m, feasible).
`out_state` columns: [cust, action, arrive_s, wait_s, start_s, service_s, depart_s, load_before_g, load_after_g, feasible].

### engine/route_cost.py

```
compute_route_cost(total_dist_m: i64, total_time_s: i64, n_reloads: i32, vtype: i32, total_wait_s: i64, vehicles: i64[:,:], vid: i32) -> i64
compute_makespan(truck_return_times: i64[:], bike_return_times: i64[:]) -> i64
count_reloads_3d(sim_states: i64[:,:,:], vid: i32, length: i32) -> i32
total_wait_time_3d(sim_states: i64[:,:,:], vid: i32, length: i32) -> i64
```

### engine/reload.py

```
truck_reload_at_stop(customer_idx: i32, truck_id: i32, current_load_g: i64, truck_arrive_s: i64, satellites: i64[:,:], n_satellites: i32, reload_service_s: i64) -> (i64, i64, i64)
bike_reload_at_stop(customer_idx: i32, bike_id: i32, bike_arrive_s: i64, satellites: i64[:,:], n_satellites: i32, reload_service_s: i64) -> (i64, i64, i64)
```

### engine/validate.py

```
validate_capacity_3d(sim_states: i64[:,:,:], lengths: i32[:], vehicles: i64[:,:], vtype_offset: i32, n_vehicles: i32) -> (i64[:,:], i32)
validate_time_windows_3d(sim_states: i64[:,:,:], lengths: i32[:], n_vehicles: i32, customers: i64[:,:], vtype_label: i32) -> (i64[:,:], i32)
validate_sync_3d(truck_sim: i64[:,:,:], truck_lengths: i32[:], bike_sim: i64[:,:,:], bike_lengths: i32[:], satellites: i64[:,:], n_satellites: i32, delta_t_s: i64) -> (i64[:,:], i32)
_find_depart_3d(sim: i64[:,:,:], lengths: i32[:], vid: i32, cust: i32, action: i32) -> i64
_find_arrive_3d(sim: i64[:,:,:], lengths: i32[:], vid: i32, cust: i32, action: i32) -> i64
```

### engine/violations.py

```
calc_unserved_penalty(customer_idx: i32, customers: i64[:,:], dist_matrix: i64[:,:], max_dist_m: i64, vehicles: i64[:,:]) -> i64
compute_penalties(unserved: i32[:], n_unserved: i32, n_duplicates: i32, cap_viol: i64[:,:], n_cap: i32, tw_viol: i64[:,:], n_tw: i32, sync_viol: i64[:,:], n_sync: i32, n_vr: i32, return_times: i64[:], n_return_times: i32, penalty_w: i64[:], customers: i64[:,:], dist_matrix: i64[:,:], vehicles: i64[:,:]) -> (i64, i64[7])
```

### alns/ls_helpers.py

```
get_stops(sol, vtype: i32) -> i32[:,:]
get_actions(sol, vtype: i32) -> i8[:,:]
get_lengths(sol, vtype: i32) -> i32[:]
get_loads(sol, vtype: i32) -> i64[:]
get_distances(sol, vtype: i32) -> i64[:]
n_vehicles(sol, vtype: i32) -> i32
two_opt_delta(sol, vtype: i32, vid: i32, i: i32, j: i32, dist_matrix: i64[:,:]) -> i64
or_opt_delta(sol, vtype: i32, vid: i32, seg_start: i32, seg_len: i32, insert_pos: i32, dist_matrix: i64[:,:]) -> i64
do_or_opt_move(sol, vtype: i32, vid: i32, seg_start: i32, seg_len: i32, insert_pos: i32, dist_matrix: i64[:,:]) -> None
update_route_distance(sol, vtype: i32, vid: i32, dist_matrix: i64[:,:]) -> None
clear_tail_index(sol, vtype: i32, vid: i32, start: i32, end: i32) -> None
rebuild_route_index(sol, vtype: i32, vid: i32, L: i32) -> None
```

### alns/ls_intra.py

```
two_opt(sol, vtype: i32, vid: i32, dist_matrix: i64[:,:]) -> bool
or_opt(sol, vtype: i32, vid: i32, dist_matrix: i64[:,:]) -> bool
```

### alns/ls_inter.py

```
relocate_inter_route(sol, vtype: i32, dist_matrix: i64[:,:], customers: i64[:,:], vehicles: i64[:,:]) -> bool
exchange_inter_route(sol, vtype: i32, dist_matrix: i64[:,:], customers: i64[:,:]) -> bool
_swap_ins_delta(stops: i32[:,:], lengths: i32[:], vid: i32, pos: i32, new_cust: i32, dist_matrix: i64[:,:]) -> i64
```

### alns/ls_twooptstar.py

```
two_opt_star(sol, vtype: i32, dist_matrix: i64[:,:], customers: i64[:,:], vehicles: i64[:,:]) -> bool
_dm_node(stops: i32[:,:], vid: i32, pos: i32, L: i32) -> i32
_partial_load(stops: i32[:,:], actions: i8[:,:], vid: i32, start: i32, end: i32, customers: i64[:,:]) -> i64
_delta(stops: i32[:,:], va: i32, vb: i32, i: i32, j: i32, La: i32, Lb: i32, dm: i64[:,:]) -> i64
_execute(sol, vtype: i32, va: i32, vb: i32, i: i32, j: i32, La: i32, Lb: i32, dist_matrix: i64[:,:], customers: i64[:,:]) -> None
```

### alns/ls_advanced.py

```
swap_star(sol, vtype: i32, dist_matrix: i64[:,:], customers: i64[:,:]) -> bool
ejection_chain_3(sol, vtype: i32, dist_matrix: i64[:,:], customers: i64[:,:]) -> bool
```

### alns/destroy_helpers.py

```
sort_by_pos_desc(targets: i32[:], sol) -> i32[:]
remove_targets(sol, targets: i32[:], dist_matrix: i64[:,:], customers: i64[:,:]) -> i32[:]
shaw_relatedness(c: i32, targets: i32[:], customers: i64[:,:], dist_matrix: i64[:,:], sol, max_dist: i64, max_demand: i64) -> i64[:]
nonempty_routes(sol) -> i32[:,:]
```

### alns/destroy_basic.py

```
random_removal(sol, customers: i64[:,:], dist_matrix: i64[:,:], seed: i32, config: f64[:]) -> i32[:]
worst_cost_removal(sol, customers: i64[:,:], dist_matrix: i64[:,:], seed: i32, config: f64[:]) -> i32[:]
shaw_removal(sol, customers: i64[:,:], dist_matrix: i64[:,:], seed: i32, config: f64[:]) -> i32[:]
zone_removal(sol, customers: i64[:,:], dist_matrix: i64[:,:], seed: i32, config: f64[:]) -> i32[:]
```

### alns/repair_helpers.py

```
force_insert(sol, customer: i32, dist_matrix: i64[:,:], customers: i64[:,:], vehicles: i64[:,:]) -> None
greedy_insert_single(sol, customer: i32, customers: i64[:,:], dist_matrix: i64[:,:], vehicles: i64[:,:]) -> None
find_top_k_insertions(sol, customer: i32, k: i32, customers: i64[:,:], dist_matrix: i64[:,:], vehicles: i64[:,:]) -> (i64[:], i32[:], i32[:], i32[:], i32)
```

### alns/repair_ops.py

```
_regret_k_core(sol, remaining: i32[:], n_rem: i32, k: i32, noise_factor: f64, customers: i64[:,:], dist_matrix: i64[:,:], vehicles: i64[:,:], seed: i32) -> None
_blinks_core(sol, removed: i32[:], customers: i64[:,:], dist_matrix: i64[:,:], vehicles: i64[:,:], p_blink: f64, seed: i32) -> None
```

## Customer Array Columns (i64)

```
COL_X          = 0  # x coordinate (meters)
COL_Y          = 1  # y coordinate (meters)
COL_DEMAND     = 2  # demand (grams)
COL_TW_OPEN    = 3  # time window open (seconds from day start)
COL_TW_CLOSE   = 4  # time window close (seconds)
COL_SERVICE    = 5  # service time (seconds)
COL_RESTRICTED = 6  # 0=both vehicles, 1=bike only
```

## Vehicle Array Columns (i64)

```
VCOL_TYPE      = 0  # 0=truck, 1=bike
VCOL_CAPACITY  = 1  # capacity (grams)
VCOL_COST_M    = 2  # operating cost (VND per meter)
VCOL_SPEED     = 3  # microseconds per meter (pre-computed from km/h)
```

Speed encoding: `speed_us_per_m = 1_000_000 * 3600 / (speed_kmh * 1000)`
- Truck 25 km/h → 144000 µs/m
- Bike 20 km/h → 180000 µs/m

Travel time: `time_s = dist_m * speed_us_per_m / 1_000_000`

## Rules

1. ALL hot-path arrays are `int64` or `int32` or `int8`. No `float64` in compute.
2. `config` stays `float64` (SA temperature, decay rates — not hot path).
3. Tests MUST create arrays with exact dtypes from this spec.
4. NEVER pass `None` where array expected.
5. NEVER pass Python `list` where `ndarray` expected.
6. NEVER pass `np.random.default_rng()` to @njit — pass `int` seed.
7. `restricted` is column 6 of customers, not a separate array.
