# 2-Echelon VRP with Synchronization — Design Spec

## Problem Definition

Two-echelon vehicle routing problem (2E-VRP) with temporal synchronization constraints. Two vehicle types serve customers from a single depot, with reload capability at customer nodes.

### Vehicles

| Type | Capacity | Role |
|------|----------|------|
| Truck (oto) | 2000 kg | Deliver directly + serve as mobile depot for bikes |
| Bike (xe may) | 60 kg | Deliver directly + reload from trucks at customer nodes |

Both vehicle types depart from and return to a single depot.

### Customers

- Demand range: 3 kg to 2000 kg
- Each customer has a time window [tw_open, tw_close]
- Multi-dimensional constraints (size, type, etc.)
- Some customers have vehicle-type restrictions (e.g., alley = bike only)
- Every customer must be served exactly once (one DELIVER action)

### Reload Mechanism

- Bikes can meet trucks at **customer nodes** (not arbitrary locations) to reload
- A customer node can serve as a reload point (satellite) while also being a delivery destination
- One truck at a satellite can serve multiple bikes (sequential transfers)
- No limit on number of reloads per bike (each costs service time)
- Synchronization tolerance: truck and bike must meet within +-delta_t

### Objectives (Multi-objective)

1. Minimize total cost: sum of (distance x cost_per_km) for all vehicles
2. Minimize makespan: latest vehicle return time to depot

---

## Approach: Two-Phase Decomposition + ALNS

### Architecture: 3-Layer Iterative

```
Layer 1: ASSIGNMENT
  - Customer -> truck deliver / bike deliver / satellite candidate
  - Forced: demand > 60kg -> truck, restricted -> bike
  - Flexible: heuristic assignment

Layer 2: TRUCK ROUTING (ALNS)
  - Input: truck-deliver customers + satellite nodes
  - Output: truck routes with schedule (arrival/departure at every stop)
  - Determines: satellite supply (what kg, when, where)

Layer 3: BIKE ROUTING (ALNS)
  - Input: bike-deliver customers + satellites + truck schedule from Layer 2
  - Output: bike routes with reload schedule
  - Constrained by: truck presence at satellites +-delta_t, available supply

FEEDBACK LOOP:
  - ALNS operators can modify any layer
  - Bike infeasible/expensive -> adjust assignment or truck route -> re-evaluate
```

Truck routes are built first each evaluation (supply side), bike routes second (demand side). But ALNS can modify truck routes based on bike feedback — not locked.

---

## Data Model — Pure Numpy

All data stored as numpy arrays. No classes, no OOP. Functions take arrays in, return arrays out.

### Input Data

```python
# Customers: shape (N, num_features)
# columns: [x, y, demand, tw_open, tw_close, service_time, dim1, dim2, ...]
customers: np.ndarray  # float64

# Vehicle-type restriction: 0 = both allowed, 1 = bike only
restricted: np.ndarray  # int8, shape (N,)

# Depot coordinates
depot: np.ndarray  # float64, shape (2,)

# Vehicles: shape (K, 4)
# columns: [type (0=truck, 1=bike), capacity, cost_per_km, speed]
vehicles: np.ndarray  # float64

# Precomputed distance matrix: shape (N+1, N+1), index 0 = depot
dist_matrix: np.ndarray  # float64
```

### Solution Encoding

```python
# Route stops: customer indices, -1 = padding
truck_stops: np.ndarray   # int32, shape (num_trucks, max_route_len)
bike_stops: np.ndarray    # int32, shape (num_bikes, max_route_len)

# Action per stop: 0 = DELIVER, 1 = RELOAD, -1 = padding
truck_actions: np.ndarray  # int8, shape (num_trucks, max_route_len)
bike_actions: np.ndarray   # int8, shape (num_bikes, max_route_len)

# Satellite reload events: shape (num_reloads, 5)
# columns: [customer_idx, bike_id, truck_id, transfer_kg, planned_time]
satellites: np.ndarray  # float64
```

A customer node can appear multiple times across routes (as RELOAD in one, DELIVER in another), but exactly ONE DELIVER action per customer across all routes.

Column indices defined as module-level constants (COL_X = 0, COL_Y = 1, etc.).

---

## Route Simulation Engine

### State Tracking Per Stop

Each route simulation produces a state array:

```python
# shape: (num_stops, 10)
# columns:
COL_CUST     = 0   # customer index
COL_ACTION   = 1   # 0=deliver, 1=reload
COL_ARRIVE   = 2   # arrival time
COL_WAIT     = 3   # wait time (early arrival or sync wait)
COL_START    = 4   # service start = arrive + wait
COL_SERVICE  = 5   # service duration
COL_DEPART   = 6   # departure = start + service
COL_LOAD_BEF = 7   # load before action
COL_LOAD_AFT = 8   # load after action
COL_FEASIBLE = 9   # 1=ok, 0=violation
```

### Sequential Simulation

Each route is simulated stop-by-stop (state depends on previous stop). Multiple routes can be simulated in parallel.

For each stop:
1. Compute travel time from previous stop
2. Check time window feasibility
3. Compute wait time (arrive early -> wait for TW open, or wait for sync partner)
4. Execute action:
   - DELIVER: load decreases by customer demand
   - RELOAD (truck): load decreases by total transfer to bikes at this stop; dwell time covers all sequential bike transfers
   - RELOAD (bike): load increases by transfer amount; may wait for truck
5. Check capacity: load >= 0 and load <= vehicle capacity at all times
6. Record full state

### Truck at RELOAD — Serving Multiple Bikes

When a truck stops at a satellite to serve N bikes:
- Bikes arrive at different times
- Transfers are sequential, each taking reload_service_time
- Transfer[i] starts at max(end_of_previous_transfer, bike_i_arrival)
- Truck must stay from first bike arrival through last transfer completion
- Total dwell time counts toward truck route time

### Synchronization Validation

After simulating all routes independently, validate every reload event:
- Truck must be present at the customer node when bike arrives (within +-delta_t)
- Specifically: bike_arrive in [truck_arrive - delta_t, truck_depart + delta_t]
- Truck must have sufficient remaining load for the transfer at that point in its route

### Full Solution Evaluation Flow

```
evaluate_solution(solution):
  1. Simulate each truck route  -> truck_states, truck_return_times
  2. Simulate each bike route   -> bike_states, bike_return_times
  3. Validate delivery uniqueness (each customer has exactly 1 DELIVER)
  4. Validate synchronization (truck-bike meet within +-delta_t)
  5. Validate capacity at every stop (non-negative, within limits)
  6. Validate time windows at every DELIVER stop
  7. Validate vehicle restrictions (alley -> bike only)
  8. Compute objectives:
       cost = sum(distance * cost_per_km) for all vehicles
       makespan = max(return_time) across all vehicles
  9. Compute penalty for each violation
       fitness = w1 * cost + w2 * makespan + w3 * total_penalty
```

---

## ALNS Operators

### Destroy Operators

| Operator | Description |
|----------|-------------|
| random_removal | Remove q random customers |
| worst_cost_removal | Remove q most expensive customers (highest insertion cost) |
| shaw_removal | Remove q similar customers (close distance, similar demand/TW) |
| route_removal | Remove an entire route |
| satellite_removal | Remove a satellite + all dependent bike customers |
| zone_removal | Remove all customers in a geographic zone |

### Repair Operators

| Operator | Description |
|----------|-------------|
| greedy_insertion | Insert each customer at cheapest position |
| regret_k_insertion | Insert hardest-to-place customer first (highest regret) |
| satellite_aware_insertion | Prefer inserting near active satellites for bike clusters |

### Cross-layer Operators

| Operator | Description |
|----------|-------------|
| swap_assignment | Move customer between truck and bike (create/use satellite) |
| relocate_satellite | Move a satellite to a different customer node |

### Adaptive Weight Update

Every segment (e.g., 100 iterations), update operator selection probabilities based on performance. Operators that found improving solutions get higher weight. Normalize weights after update.

---

## Multi-Objective Handling

### Primary: Weighted Sum

```
fitness = w1 * cost + w2 * makespan + w3 * penalties
```

Default weights: w1=0.6, w2=0.3, w3=0.1. These are final-stage weights. During search, penalty weight (w3) starts high (e.g., 10.0) to enforce feasibility early, then decays toward 0.1 as the search progresses and feasible solutions dominate. w1 and w2 are re-normalized accordingly at each stage.

### Secondary: Pareto Archive

Maintain a set of non-dominated solutions throughout the search. A solution enters the archive if no existing solution is better in both cost AND makespan. Archive runs passively alongside the main ALNS search.

Final output: Pareto front of trade-off solutions for user to choose from.

### Acceptance Criterion: Simulated Annealing

- Better solution: always accept
- Worse solution: accept with probability exp(-delta / temperature)
- Temperature cools geometrically: T = T * cooling_rate (e.g., 0.9997)

---

## Initial Solution Construction

Greedy 3-step approach. Prioritizes feasibility over optimality.

1. **Assignment**: forced assignments (demand > 60kg -> truck, restricted -> bike), flexible customers assigned by proximity heuristic
2. **Truck routing**: nearest-neighbor through truck-deliver customers + selected satellite nodes
3. **Bike routing**: cluster bike customers by nearest satellite, build nearest-neighbor routes per cluster with reload stops

---

## Implementation

- Language: Python
- Style: purely functional/procedural — no OOP, no class hierarchies
- Data: pure numpy arrays
- Functions: take arrays in, return arrays out
- Constants: module-level column indices

---

## Scale Target

Designed to handle 50 to 1000+ customers. Decomposition enables scaling: Layer 1 assignment reduces sub-problem sizes for Layers 2 and 3.
