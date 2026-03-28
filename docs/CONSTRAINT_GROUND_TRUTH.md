# 2E-VRP Constraint Ground Truth

**Purpose**: This document is the SINGLE SOURCE OF TRUTH for all problem constraints.
Every ALNS operator, repair heuristic, or solution modifier MUST respect these rules.
If code contradicts this document, the code is wrong.

**Last verified against codebase**: 2026-03-28

---

## 1. VEHICLE TYPES

### 1.1 Truck (VEH_TRUCK = 0)

| Parameter | Value | Source |
|-----------|-------|--------|
| Capacity | 2000 kg | `cost.py:TRUCK_CAPACITY` |
| Speed (urban) | 25 km/h | `cost.py:TRUCK_SPEED_URBAN` |
| Speed (suburban) | 40 km/h | `cost.py:TRUCK_SPEED_SUBURBAN` (not used in simulation) |
| Cost per km | 12,000 VND/km | `cost.py:TRUCK_TOTAL_KM` |
| Driver cost | 56,250 VND/hour | `cost.py:TRUCK_DRIVER_HOUR` |
| Fixed cost per day | 500,000 VND/day | `cost.py:TRUCK_FIXED_DAY` |
| Deploy cost per trip | 150,000 VND/trip | `cost.py:TRUCK_DEPLOY_COST` |
| Wait cost | 2,969 VND/min | `cost.py:TRUCK_WAIT_COST_MIN` |

**Simulation speed**: `vehicles[vid, VCOL_SPEED]` is read from the vehicles array. Typically set to `TRUCK_SPEED_URBAN = 25 km/h` for urban delivery.

### 1.2 Bike / Xe may (VEH_BIKE = 1)

| Parameter | Value | Source |
|-----------|-------|--------|
| Capacity | 60 kg | `cost.py:BIKE_CAPACITY` |
| Speed (urban) | 20 km/h | `cost.py:BIKE_SPEED_URBAN` |
| Speed (suburban) | 35 km/h | `cost.py:BIKE_SPEED_SUBURBAN` (not used in simulation) |
| Cost per km | 5,000 VND/km | `cost.py:BIKE_TOTAL_KM` |
| Driver cost | 37,500 VND/hour | `cost.py:BIKE_DRIVER_HOUR` |
| Fixed cost per day | 50,000 VND/day | `cost.py:BIKE_FIXED_DAY` |
| Deploy cost per trip | 30,000 VND/trip | `cost.py:BIKE_DEPLOY_COST` |
| Wait cost | 1,146 VND/min | `cost.py:BIKE_WAIT_COST_MIN` |

### 1.3 Vehicle Array Layout

```
vehicles: np.ndarray, shape (n_trucks + n_bikes, 4)
columns: [VCOL_TYPE=0, VCOL_CAPACITY=1, VCOL_COST_KM=2, VCOL_SPEED=3]

Rows 0..n_trucks-1   = trucks
Rows n_trucks..end    = bikes
```

Simulation uses `vehicles[vid, VCOL_CAPACITY]` and `vehicles[vid, VCOL_SPEED]` directly. For bikes, `vid = n_trucks + bike_index`.

---

## 2. CUSTOMER TYPES

### 2.1 Customer Data Layout

```
customers: np.ndarray, shape (N, 6+)
columns: [COL_X=0, COL_Y=1, COL_DEMAND=2, COL_TW_OPEN=3, COL_TW_CLOSE=4, COL_SERVICE_TIME=5]
```

Customer indices are 0-based. In the distance matrix, customer `c` maps to node `c + 1` (node 0 = depot).

### 2.2 Restriction Categories

The `restricted` array (`np.int8, shape (N,)`) determines vehicle eligibility:

| restricted[c] | Meaning | Truck can deliver? | Bike can deliver? |
|----------------|---------|-------------------|-------------------|
| 0 | Normal / big | YES | YES (if demand <= 60kg) |
| 1 | Restricted (alley) | NO | YES |

**Implicit rules derived from capacity**:
- If `demand > 60 kg` (BIKE_CAPACITY): only truck can deliver (bike physically cannot carry it)
- If `restricted[c] == 1`: only bike can deliver (truck cannot enter alley)
- If `restricted[c] == 1 AND demand > 60 kg`: INFEASIBLE customer (cannot be served by either vehicle -- should not exist in valid input)

### 2.3 Customer Type Summary

| Type | Condition | Vehicle assignment |
|------|-----------|-------------------|
| Truck-only | `demand > 60 kg AND restricted == 0` | Must go on truck route as DELIVER |
| Bike-only | `restricted == 1 AND demand <= 60 kg` | Must go on bike route as DELIVER |
| Flexible | `demand <= 60 kg AND restricted == 0` | Can go on either truck or bike |

**ALNS implication**: When moving a customer between vehicles, ALWAYS check both restriction AND capacity. A restricted customer must NEVER appear as a DELIVER on a truck route.

---

## 3. TIME CONSTRAINTS

### 3.1 Working Day

| Parameter | Value | Source |
|-----------|-------|--------|
| DAY_START | 0.0 min (7:00 AM) | `cost.py:DAY_START` |
| DAY_LENGTH | 480.0 min (8 hours) | `cost.py:DAY_LENGTH` |

All times are in **minutes from DAY_START**. A vehicle departing the depot at time 0 must return by time 480.

### 3.2 Time Windows

Each customer has `[tw_open, tw_close]` in minutes from DAY_START.

- Vehicle arrives **before** tw_open: waits (incurs wait cost, no penalty)
- Vehicle arrives **within** [tw_open, tw_close]: serves immediately
- Vehicle arrives **after** tw_close: TIME WINDOW VIOLATION (penalty)

**Enforcement in simulation** (`simulate.py` line 50-51):
```
if arrive > tw_close:
    feasible = False
```

Time windows apply ONLY to DELIVER actions, NOT to RELOAD actions. Reload stops use the customer's TW for wait calculation but TW violation is only checked for delivers in `validate_time_windows`.

### 3.3 Service Time

```
service_time = SERVICE_BASE + SERVICE_PER_100KG * (demand / 100.0)
             = 10.0 + 5.0 * (demand / 100.0)  minutes
```

Examples:
- 50 kg: 10 + 2.5 = 12.5 min
- 200 kg: 10 + 10 = 20 min
- 0 kg (reload): uses RELOAD_SERVICE_TIME = 5.0 min per transfer

### 3.4 Synchronization Tolerance

| Parameter | Value | Source |
|-----------|-------|--------|
| SYNC_DELTA_T | 15.0 min | `cost.py:SYNC_DELTA_T` |

Truck depart and bike arrive at a satellite must be within +/- 15 minutes. Beyond that, sync gap penalty applies.

---

## 4. CAPACITY CONSTRAINTS

### 4.1 Load Model

Load is **cumulative and increasing** for deliveries. Each vehicle starts at the depot with load = 0 and picks up goods to deliver:

```
DELIVER: current_load += demand   (load INCREASES -- vehicle is carrying goods TO deliver)
RELOAD (truck): current_load += load_change  (load_change is NEGATIVE -- truck gives goods to bikes)
RELOAD (bike): current_load += transfer_kg   (load INCREASES -- bike receives goods from truck)
```

**CRITICAL**: The load model is "load-on-departure-from-depot". The vehicle loads everything at the depot and the `current_load` tracks how much it's carrying. At each DELIVER, load increases because we're tracking cumulative demand served (NOT remaining goods). Wait -- re-reading the code:

Actually, re-reading `simulate.py` carefully:
- Load starts at 0
- At DELIVER: `current_load += demand` -- this means load tracks cumulative demand loaded onto the vehicle
- Feasibility check: `current_load > vehicle_capacity` means total demand on the trip cannot exceed capacity

This is equivalent to: the total demand of all deliveries on a single trip (between reloads or from depot) must not exceed the vehicle's capacity.

### 4.2 Capacity Validation

**Enforced in**: `simulate.py` line 57-58 and line 75-76, `validate.py:validate_capacity_all`

At EVERY stop: `0 <= load_after <= vehicle_capacity`

For trucks at reload: load_change is negative (giving goods away), so load decreases. This should never make load negative if satellite KG values are correct.

### 4.3 Per-Trip Capacity

Each trip (depot -> stops -> depot for single-trip model) has cumulative demand tracked. The total demand of all DELIVER stops between the start and any point in the route must not exceed capacity.

For bikes with RELOAD: after a RELOAD, the bike receives new goods. The load continues to accumulate. So the capacity constraint is: at any point in the route, `current_load <= capacity`.

---

## 5. RELOAD / SATELLITE MODEL

### 5.1 What is a Satellite?

A satellite is a **customer node** where a truck DELIVERS and a bike RELOADs. There are no dedicated satellite facilities -- any customer location can serve as a transfer point.

The satellite table tracks each transfer event:

```
satellites: np.ndarray, shape (S, 5)
columns: [SAT_CUST=0, SAT_BIKE=1, SAT_TRUCK=2, SAT_KG=3, SAT_TIME=4]
```

Each row = one transfer: bike `SAT_BIKE` receives `SAT_KG` kg from truck `SAT_TRUCK` at customer node `SAT_CUST` at planned time `SAT_TIME`.

### 5.2 Truck at Satellite

The truck's action at a satellite node is **ACT_DELIVER** (not ACT_RELOAD). The truck is delivering goods to the satellite location. In the truck's route, this is a normal delivery stop.

**Key insight**: The truck DELIVERS to the customer (the satellite is a real customer), and ALSO hands off goods to bikes. The truck's route shows ACT_DELIVER at the satellite node.

Wait -- re-reading `validate_sync` line 105: truck uses `ACT_DELIVER` at satellite nodes.
But re-reading `builder.py` line 147: in the giant tour, satellites are tagged as `type="satellite"` and get `ACT_RELOAD` action.

Let me reconcile: In `split.py` line 27: `action = ACT_RELOAD if stop["type"] == "satellite" else ACT_DELIVER`. So in the TRUCK route, satellite stops get ACT_RELOAD.

But `validate_sync` line 105 checks: `t_state[:, ST_ACTION] == ACT_DELIVER` for the truck at the satellite.

**BUG / DISCREPANCY DETECTED**:
- `split.py` line 27 and 147: truck satellite stops get `ACT_RELOAD`
- `validate.py:validate_sync` line 105: expects `t_state[:, ST_ACTION] == ACT_DELIVER`
- `sync_cost.py:_find_truck_depart` line 56: expects `t_state[:, ST_ACTION] == ACT_DELIVER`

The builder tags truck satellite stops as RELOAD, but the validator/sync cost look for DELIVER. This means sync validation will ALWAYS fail to find the truck at the satellite, triggering `truck_not_at_node` violations and PENALTY_MISSING_RELOAD.

**GROUND TRUTH for ALNS operators**: Follow what `validate_sync` and `sync_cost.py` expect:
- Truck at satellite: `ACT_DELIVER` (truck delivers goods, some of which get transferred to bikes)
- Bike at satellite: `ACT_RELOAD` (bike receives goods)

**ACTION NEEDED**: Fix `split.py` to assign `ACT_DELIVER` to truck satellite stops, OR fix `validate_sync` and `sync_cost.py` to look for `ACT_RELOAD`. The former is more semantically correct (truck IS delivering at the satellite location).

### 5.3 Bike at Satellite

The bike's action at a satellite is `ACT_RELOAD`. During simulation (`simulate.py` line 62-66):
1. Look up satellite events where `SAT_CUST == cust AND SAT_BIKE == bike_id`
2. `transfer_kg` = amount to receive
3. `current_load += transfer_kg`
4. Service time = `RELOAD_SERVICE_TIME` = 5.0 min
5. `sync_wait` = time bike must wait for the planned transfer time

### 5.4 Truck Serving Multiple Bikes

When a truck stops at a satellite, it may serve N bikes sequentially (`reload.py:truck_reload_at_stop`):
1. All satellite events at this `(customer_idx, truck_id)` are gathered
2. Sorted by `SAT_TIME`
3. Each transfer takes `RELOAD_SERVICE_TIME` = 5.0 min
4. Transfer[i] starts at `max(cursor, event[i].SAT_TIME)`
5. Truck must stay for the full duration of all transfers
6. `load_change` = negative sum of all `SAT_KG` for these events

### 5.5 Reload Costs

| Parameter | Value | Source |
|-----------|-------|--------|
| RELOAD_SERVICE_TIME | 5.0 min/transfer | `cost.py` |
| RELOAD_HANDLING_COST | 5,000 VND/transfer | `cost.py` |

---

## 6. MULTI-TRIP RULES

### 6.1 Current Model: Single Trip per Vehicle

From `cost.py` comment (lines 84-85):
```
# Moi xe chi chay 1 chuyen/ngay: depot -> customers -> depot
# Tong thoi gian phai <= DAY_LENGTH. Khong reload tai depot.
```

**Each vehicle does exactly ONE trip per day**: depot -> stops -> depot.

From `split.py:group_trips_to_trucks` (line 75):
```
"""1 trip = 1 vehicle. No multi-trip. No reload at depot."""
```

### 6.2 Bikes with Reload Stops

Although bikes do a single trip (depot -> ... -> depot), they CAN include RELOAD stops along the way. This is NOT multi-trip -- it's a single continuous trip with intermediate reloads at satellite nodes.

A bike route might look like:
```
depot -> [RELOAD at sat1] -> [DELIVER c1] -> [DELIVER c2] -> [RELOAD at sat2] -> [DELIVER c3] -> depot
```

This is ONE trip. The bike never returns to the depot mid-route.

### 6.3 Trucks with Satellite Stops

Similarly, a truck route is one trip that may include satellite stops (where it hands off goods to bikes) mixed with normal delivery stops:
```
depot -> [DELIVER c10] -> [DELIVER/RELOAD at sat_node] -> [DELIVER c11] -> depot
```

### 6.4 ALNS Implication

- NEVER create a route that returns to depot mid-trip and goes out again
- A vehicle's route is a single sequence: depot -> stops -> depot
- Reload stops are intermediate stops within the single trip, not trip boundaries

---

## 7. OBJECTIVE FUNCTION

### 7.1 Fitness Formula

```python
fitness = total_cost + sync_cost + total_penalty - n_feasible_routes * FEASIBLE_ROUTE_BONUS
```

**Lower is better.**

Source: `fitness.py:compute_fitness`

| Component | Description |
|-----------|-------------|
| `total_cost` | Sum of operating costs across ALL vehicles (VND) |
| `sync_cost` | Synchronization penalty for truck-bike timing gaps (VND) |
| `total_penalty` | Sum of constraint violation penalties (VND) |
| `n_feasible_routes * 2,000,000` | Bonus subtracted for each fully feasible route |

### 7.2 Operating Cost per Route

```python
total = distance_cost + time_cost + wait_cost + fixed_cost + deploy_cost + reload_cost
```

Source: `route_cost.py:compute_route_cost`

| Component | Formula |
|-----------|---------|
| distance_cost | `total_distance_km * cost_per_km` |
| time_cost | `(total_time_min / 60) * driver_hour_rate` |
| wait_cost | `total_wait_min * wait_cost_per_min` |
| fixed_cost | `FIXED_DAY` (per vehicle, always charged if route non-empty) |
| deploy_cost | `DEPLOY_COST` (per trip) |
| reload_cost | `n_reloads * RELOAD_HANDLING_COST` |

### 7.3 Makespan

```python
makespan = max(return_time across all trucks and bikes)
```

Source: `route_cost.py:compute_makespan`

Makespan is computed but NOT directly in the fitness formula in the current implementation. It's tracked for reporting/Pareto purposes.

### 7.4 Feasible Route Bonus

`FEASIBLE_ROUTE_BONUS = 2,000,000 VND` per route.

A route is "fully feasible" if:
1. No time window violations in that route
2. Return time <= DAY_LENGTH
3. Route is non-empty (return_time > 0)

Source: `fitness.py` lines 60-69

---

## 8. PENALTY STRUCTURE

### 8.1 Violation Types and Their Penalties

| Violation | Type | Base Penalty | How Applied | Source |
|-----------|------|-------------|-------------|--------|
| Unserved customer | SOFT | Dynamic (distance-weighted) | Per customer, scaled by distance from depot | `violations.py:calc_unserved_penalty` |
| Duplicate delivery | SOFT | 500,000 VND/occurrence | Per duplicate customer | `violations.py` line 53 |
| Capacity overload | SOFT | `overload_pct * fixed_cost * 10` | Per stop where load > capacity | `violations.py` lines 55-62 |
| Time window late | SOFT | 10,000 VND/min late | Sum of (arrive - tw_close) for all late stops | `violations.py` line 66 |
| Sync failure | SOFT | 100,000 VND/event | Per failed sync event | `violations.py` line 69 |
| Vehicle restriction | SOFT | 200,000 VND/violation | Per restricted customer on truck | `violations.py` line 70 |
| Overtime | SOFT | 10,000 VND/min (same as TW rate) | Per minute beyond DAY_LENGTH for each vehicle | `violations.py` lines 74-78 |

### 8.2 Unserved Customer Penalty (Special)

The unserved penalty is **distance-weighted** and **dynamic**:

```python
base_cost = round_trip_bike_cost(depot_to_customer)
distance_multiplier = (dist_ratio)^2 * 9 + 1   # range [1, 10]
penalty = PENALTY_UNSERVED_MULTIPLIER(5.0) * base_cost * distance_multiplier
```

Far customers: up to 50x the base cost of a bike round trip.
Near customers: 5x the base cost.

This ensures ALNS strongly prefers serving far customers (they're expensive to leave unserved).

### 8.3 Capacity Overload Penalty

```python
overload_pct = (load - capacity) / capacity * 100
penalty = overload_pct * fixed_cost_per_day * PENALTY_OVERLOAD_PER_PCT(10)
```

For a truck: 1% overload = 500,000 * 10 = 5,000,000 VND penalty per stop.
For a bike: 1% overload = 50,000 * 10 = 500,000 VND penalty per stop.

### 8.4 Sync Cost (Separate from Penalties)

Computed in `sync_cost.py:compute_sync_cost` and added to fitness SEPARATELY from penalties:

| Condition | Cost |
|-----------|------|
| Truck or bike missing at satellite | 1,000,000 VND (`PENALTY_MISSING_RELOAD`) |
| Sync gap > delta_t (15 min) | 20,000 VND/min beyond delta_t (`PENALTY_SYNC_GAP_MIN`) |
| Sync gap <= delta_t | 0 (OK) |

### 8.5 Adaptive Penalty Weight (w3)

The ALNS uses an adaptive multiplier `w3` on total_penalty:
- Starts high (e.g., 10.0) to enforce feasibility early
- Decays toward 0.1 as feasible solutions accumulate
- Feedback: if < 30% feasible solutions recently, w3 doubles; if > 80% feasible, w3 decays faster

Source: `alns/penalty.py:adaptive_penalty_adjustment`

---

## 9. SOLUTION REPRESENTATION

### 9.1 Data Structure

```python
sol = {
    # Routes: padded 2D arrays
    "truck_stops":    np.int32,  shape (n_trucks, max_route_len)   # customer indices, -1 = pad
    "truck_actions":  np.int8,   shape (n_trucks, max_route_len)   # ACT_DELIVER=0, ACT_RELOAD=1, ACT_PAD=-1
    "bike_stops":     np.int32,  shape (n_bikes, max_route_len)    # same
    "bike_actions":   np.int8,   shape (n_bikes, max_route_len)    # same

    # Satellite events
    "satellites":     np.float64, shape (S, 5)    # [cust, bike_id, truck_id, kg, time]

    # Cached route metadata
    "truck_lengths":  np.int32,   shape (n_trucks,)  # actual stop count per truck
    "bike_lengths":   np.int32,   shape (n_bikes,)   # actual stop count per bike
    "truck_loads":    np.float64, shape (n_trucks,)  # total demand per truck route
    "bike_loads":     np.float64, shape (n_bikes,)   # total demand per bike route
    "truck_distances": np.float64, shape (n_trucks,) # total distance per truck route
    "bike_distances": np.float64, shape (n_bikes,)   # total distance per bike route

    # Customer index (reverse lookup)
    "cust_vehicle":   np.int32,  shape (n_customers,)  # which vehicle serves this customer (-1 = unserved)
    "cust_vtype":     np.int8,   shape (n_customers,)  # VEH_TRUCK=0 or VEH_BIKE=1 (-1 = unserved)
    "cust_route_pos": np.int32,  shape (n_customers,)  # position in route (-1 = unserved)

    # Metadata
    "n_trucks":       int
    "n_bikes":        int
    "max_route_len":  int (default 100)
}
```

### 9.2 Indexing Convention

- Customer indices: 0-based (0 to N-1)
- Distance matrix node: customer `c` is at node `c + 1`; depot is node `0`
- Vehicle ID: trucks are 0..n_trucks-1; bikes are 0..n_bikes-1 within their arrays, but n_trucks+bike_id in the vehicles array
- `cust_vehicle[c]`: global vehicle ID (trucks: 0..n_trucks-1, bikes: n_trucks..n_trucks+n_bikes-1)

### 9.3 Route Reading Convention

For truck `t`:
```python
length = sol["truck_lengths"][t]
stops  = sol["truck_stops"][t, :length]    # only first `length` entries are valid
actions = sol["truck_actions"][t, :length]
# Everything at index >= length is padding (-1 / ACT_PAD)
```

### 9.4 ALNS Operator Obligations After Modification

After ANY modification to a solution, the operator MUST:

1. **Update lengths**: `sol["truck_lengths"][vid]` or `sol["bike_lengths"][vid]`
2. **Update loads**: `sol["truck_loads"][vid]` or `sol["bike_loads"][vid]` (or call `update_route_load`)
3. **Update distances**: call `update_route_distance(sol, vtype, vid, dist_matrix)`
4. **Update customer index**: `sol["cust_vehicle"][c]`, `sol["cust_vtype"][c]`, `sol["cust_route_pos"][c]` for every affected customer (or call `rebuild_index`)
5. **Pad removed slots**: set removed stop slots to -1 and action slots to ACT_PAD
6. **Update satellites**: if any satellite-related customer is moved, update or rebuild the satellites array
7. **Compact route**: ensure no gaps -- active stops must be contiguous from index 0

---

## 10. FEASIBILITY DEFINITION

### 10.1 Hard vs Soft Constraints

**ALL constraints in this model are SOFT** -- violations incur penalties rather than making the solution invalid. The solver uses penalty-weighted fitness to guide search toward feasibility.

However, the `feasible` flag in evaluation results distinguishes:

| Check | Returns `feasible=False` if | Penalty mechanism |
|-------|---------------------------|-------------------|
| Delivery uniqueness | Any customer unserved or duplicate | Unserved/duplicate penalty |
| Vehicle restrictions | Restricted customer on truck | Restriction penalty |
| Capacity | Load exceeds capacity at any stop | Overload penalty |
| Time windows | Arrival after tw_close at any DELIVER | Late penalty |
| Sync | Truck-bike timing gap > delta_t | Sync cost |
| Overtime | Any vehicle returns after DAY_LENGTH | Overtime penalty (same rate as TW) |

A solution is `feasible = True` only when ALL of the above pass with zero violations.

### 10.2 What Makes a Solution "Valid" (validate_all)

`validate.py:validate_all` checks these five conditions:
1. `validate_delivery_uniqueness` -- each customer has exactly 1 DELIVER across all routes
2. `validate_vehicle_restrictions` -- no restricted customer on truck DELIVER
3. `validate_capacity_all` -- load at every stop in [0, capacity]
4. `validate_time_windows` -- arrive <= tw_close for every DELIVER
5. `validate_sync` -- truck-bike meet within +/- delta_t at every satellite

### 10.3 Simulation Feasibility vs Validation Feasibility

The simulation (`simulate.py`) sets a local `feasible` flag per route during forward pass:
- Set to False if arrive > tw_close at a DELIVER
- Set to False if current_load > capacity at any point

The validation (`validate.py`) is a post-simulation global check across all routes.

Both must pass for `feasible=True` in the evaluation result.

---

## 11. CONSTRAINT ENFORCEMENT SUMMARY FOR ALNS

### C1: Every Customer Served Exactly Once

- **Type**: SOFT (unserved penalty, duplicate penalty)
- **Enforced in**: `validate.py:validate_delivery_uniqueness`
- **ALNS rule**: After destroy, repair MUST attempt to re-insert all removed customers. Unserved customers remain in a "request bank" with high penalty pressure.
- **What happens on violation**: Distance-weighted penalty per unserved customer (up to 50x bike round-trip cost for far customers). Duplicate delivery gets flat 500,000 VND penalty.

### C2: Vehicle Restriction (Restricted = Bike Only)

- **Type**: SOFT (200,000 VND per violation)
- **Enforced in**: `validate.py:validate_vehicle_restrictions`
- **ALNS rule**: NEVER insert a restricted customer (restricted[c]==1) into a truck route as DELIVER. Check BEFORE insertion.
- **What happens on violation**: 200,000 VND penalty per occurrence. Solution marked infeasible.

### C3: Capacity

- **Type**: SOFT (exponential penalty based on overload %)
- **Enforced in**: `simulate.py` (local flag), `validate.py:validate_capacity_all`
- **ALNS rule**: Before inserting a customer, check that `route_load + demand <= capacity`. For trucks, capacity = 2000 kg. For bikes, capacity = 60 kg.
- **What happens on violation**: `overload_pct * fixed_cost * 10` penalty PER STOP where overloaded. For trucks, 1% overload = 5M VND. Extremely expensive.

### C4: Time Windows

- **Type**: SOFT (10,000 VND per minute late)
- **Enforced in**: `simulate.py` (local flag), `validate.py:validate_time_windows`
- **ALNS rule**: Prefer insertions that arrive within [tw_open, tw_close]. Check arrival time BEFORE insertion if possible (greedy check).
- **What happens on violation**: Linear penalty = minutes_late * 10,000 VND. Solution marked infeasible.

### C5: Day Length (Overtime)

- **Type**: SOFT (10,000 VND per minute overtime)
- **Enforced in**: `violations.py:compute_penalties` (overtime section)
- **ALNS rule**: Route total time (depot -> stops -> depot) should fit within 480 min. Check return time BEFORE committing insertion.
- **What happens on violation**: Linear penalty = overtime_minutes * 10,000 VND. Not counted by validate_all but penalized in fitness.

### C6: Truck-Bike Synchronization

- **Type**: SOFT (sync cost computed separately)
- **Enforced in**: `validate.py:validate_sync`, `sync_cost.py:compute_sync_cost`
- **ALNS rule**: If a bike has a RELOAD at node X, there MUST be a matching satellite event AND the truck must DELIVER at node X. The timing gap must be within +/- 15 min.
- **What happens on violation**:
  - Missing truck/bike at satellite: 1,000,000 VND per event
  - Gap > 15 min: 20,000 VND per minute beyond 15 min
  - Sync violations: 100,000 VND per event (separate from sync_cost)

### C7: Satellite Consistency

- **Type**: Structural (broken satellites = broken solution)
- **Enforced in**: `reload.py` (during simulation), `sync_cost.py`, `validate.py:validate_sync`
- **ALNS rule**: When modifying any route that touches a satellite node:
  1. If removing a truck's DELIVER at a satellite node: MUST also remove or reassign ALL bike RELOAD events at that satellite
  2. If removing a bike's RELOAD at a satellite node: MUST remove the corresponding satellite event
  3. If moving a customer that is a satellite node: MUST update the satellites array
  4. Satellites array must stay consistent with actual route stops
- **What happens on violation**: PENALTY_MISSING_RELOAD = 1,000,000 VND per broken satellite link. Simulation may produce incorrect load values.

### C8: Route Contiguity

- **Type**: HARD (structural -- code will crash or produce garbage)
- **Enforced in**: `simulate.py` (reads stops[0..length-1] sequentially)
- **ALNS rule**: Active stops MUST be contiguous from index 0. No gaps allowed. After removing a stop, compact the array by shifting subsequent stops left.
- **What happens on violation**: Simulation reads -1 as a customer index, causing array-out-of-bounds or incorrect behavior.

### C9: Padding Consistency

- **Type**: HARD (structural)
- **Enforced in**: `simulate.py` line 26 (`l_actual = np.sum(stops >= 0)`)
- **ALNS rule**: All slots at index >= length must have stops=-1 and actions=ACT_PAD(-1). The simulation counts non-negative stops to determine route length.
- **What happens on violation**: Phantom stops processed, garbage results.

### C10: Customer Index Consistency

- **Type**: Structural (affects operator decisions, not fitness)
- **Enforced in**: `structure.py:rebuild_index`
- **ALNS rule**: After modifications, `cust_vehicle`, `cust_vtype`, `cust_route_pos` must accurately reflect current assignments. Call `rebuild_index` if unsure.
- **What happens on violation**: ALNS operators make wrong decisions (e.g., try to remove a customer that's already removed, or skip a customer that needs insertion).

---

## 12. DISTANCE MATRIX CONVENTION

```
dist_matrix: np.float64, shape (N+1, N+1)
  - Row/col 0 = depot
  - Row/col c+1 = customer c (0-indexed)
  - dist_matrix[i, j] = distance in km from node i to node j
```

Travel time = `distance / speed * 60.0` (converting hours to minutes).

---

## 13. ACTION CODES

| Code | Constant | Meaning |
|------|----------|---------|
| 0 | ACT_DELIVER | Deliver goods to customer |
| 1 | ACT_RELOAD | Transfer goods at satellite |
| -1 | ACT_PAD | Padding (empty slot) |

---

## 14. INVARIANTS THAT MUST ALWAYS HOLD

These are conditions that must be true at ALL times in a valid solution. Any ALNS operator that violates these produces a corrupted solution:

1. **Stops and actions arrays have same shape** and are synchronized: `stops[v, i]` and `actions[v, i]` describe the same event.

2. **Length consistency**: `sol["truck_lengths"][t]` equals the number of non-pad entries at the start of `sol["truck_stops"][t]`.

3. **No duplicate DELIVER**: Each customer appears as ACT_DELIVER in at most one route across all trucks and bikes.

4. **Satellite array matches routes**: Every row in `satellites` must correspond to an actual RELOAD stop in the bike route and a presence of the truck at that customer node.

5. **Load cache approximately correct**: `sol["truck_loads"][t]` should equal the sum of demands of all DELIVER stops in truck t's route. (Approximation OK during search; recomputed at evaluation.)

6. **Distance cache approximately correct**: `sol["truck_distances"][t]` should approximate the actual route distance. (Recomputed at evaluation.)

7. **Customer index consistent**: `sol["cust_vehicle"][c] != -1` implies customer c appears as DELIVER in the indicated route, and vice versa.

---

## 15. QUICK REFERENCE: WHAT ALNS OPERATORS MUST CHECK

### Before inserting customer `c` into vehicle route `v`:

1. Is `restricted[c] == 1`? If yes, `v` must be a bike.
2. Is `demand[c] > BIKE_CAPACITY(60)`? If yes, `v` must be a truck.
3. Will `route_load + demand > vehicle_capacity`? If yes, reject or accept with penalty awareness.
4. Will insertion cause time window violations at `c` or downstream customers?
5. Will insertion cause overtime (return > 480 min)?
6. Is customer `c` already assigned elsewhere? If yes, remove first (no duplicates).

### Before removing customer `c` from route:

1. Is `c` a satellite node (appears in satellites array as SAT_CUST)? If yes, handle satellite cleanup.
2. Update route length, load, distance caches.
3. Compact the route array (shift left).
4. Mark `cust_vehicle[c] = -1`, `cust_vtype[c] = -1`, `cust_route_pos[c] = -1`.
5. Add `c` to the unserved/request bank.

### After any route modification:

1. Compact route (no gaps).
2. Pad remaining slots with -1/ACT_PAD.
3. Update `lengths[v]`.
4. Update `loads[v]` (sum of demands of DELIVER stops).
5. Update `distances[v]` (call `update_route_distance`).
6. Update customer index for all affected customers.
7. If satellites affected, rebuild/update satellites array.
