# Phase 02 — Simulation Engine

## File structure

```
src/
  engine/
    simulate.py       ← route simulation (core)
    reload.py         ← truck/bike reload logic
    validate.py       ← constraint validation
    fitness.py        ← objective + penalty calculation
```

---

## simulate.py

### simulate_route

- **Idea:** Mô phỏng 1 xe chạy qua route từng stop một. Track thời gian, tải trọng, feasibility tại mỗi stop. Đây là hàm CORE — mọi thứ khác đều gọi hàm này.
- **Input:**
  - `stops: ndarray int32 (L,)` — customer indices, -1 = padding
  - `actions: ndarray int8 (L,)` — ACT_DELIVER / ACT_RELOAD / ACT_PAD
  - `vehicle_type: int` — VEH_TRUCK hoặc VEH_BIKE
  - `vehicle_capacity: float` — 2000.0 hoặc 60.0
  - `vehicle_speed: float` — km/h
  - `initial_load: float` — hàng chở từ kho
  - `customers: ndarray float64 (N, F)`
  - `dist_matrix: ndarray float64 (N+1, N+1)`
  - `satellites: ndarray float64 (S, 5)` — bảng reload events
  - `vehicle_id: int` — ID của xe đang simulate
  - `delta_t: float` — sync tolerance (phút)
- **Output:**
  - `state: ndarray float64 (L_actual, ST_COLS)` — state tại mỗi stop (L_actual = số stop thực)
  - `return_time: float` — thời điểm xe về tới kho
  - `total_distance: float` — tổng quãng đường
  - `feasible: bool` — toàn route có khả thi không
- **Logic:**
  ```
  L_actual = count stops >= 0
  state = zeros(L_actual, ST_COLS)
  prev_node = 0  (depot)
  prev_depart = 0.0
  current_load = initial_load
  total_dist = 0.0
  feasible = True

  for i in 0..L_actual-1:
      cust = stops[i]
      cust_node = cust + 1  (dist_matrix offset)
      action = actions[i]

      # Travel
      d = dist_matrix[prev_node, cust_node]
      total_dist += d
      travel_time = d / vehicle_speed * 60  (convert to minutes)
      arrive = prev_depart + travel_time

      # Time window
      tw_open = customers[cust, COL_TW_OPEN]
      tw_close = customers[cust, COL_TW_CLOSE]
      tw_wait = max(0, tw_open - arrive)

      if action == ACT_DELIVER:
          if arrive > tw_close:
              feasible = False
          start = arrive + tw_wait
          service = customers[cust, COL_SERVICE_TIME]
          load_before = current_load
          current_load -= customers[cust, COL_DEMAND]
          if current_load < 0:
              feasible = False

      elif action == ACT_RELOAD:
          if vehicle_type == VEH_TRUCK:
              load_change, service, sync_wait = truck_reload_at_stop(...)
              current_load += load_change  (negative = giving away)
          else:
              load_change, service, sync_wait = bike_reload_at_stop(...)
              current_load += load_change  (positive = receiving)

          load_before = current_load - load_change
          tw_wait += sync_wait
          start = arrive + tw_wait

          if current_load < 0 or current_load > vehicle_capacity:
              feasible = False

      depart = start + service
      state[i] = [cust, action, arrive, tw_wait+sync_wait, start,
                   service, depart, load_before, current_load, feasible]
      prev_node = cust_node
      prev_depart = depart

  # Return to depot
  d_back = dist_matrix[prev_node, 0]
  total_dist += d_back
  return_time = prev_depart + d_back / vehicle_speed * 60

  return state, return_time, total_dist, feasible
  ```
- **Test cases:**
  - TC1: **Route rỗng** — stops = [-1,-1], return 0 stops, return_time=0, feasible=True
  - TC2: **1 stop DELIVER** — depot(0,0), customer(3,4), demand=10, speed=60km/h → travel=5km, travel_time=5min, load: 100→90, feasible
  - TC3: **Capacity violation** — initial_load=5, demand=10 → current_load=-5 → feasible=False
  - TC4: **TW violation** — arrive=100, tw_close=80 → feasible=False
  - TC5: **TW wait** — arrive=50, tw_open=70 → wait=20, start=70
  - TC6: **Round trip distance** — depot→A→depot, verify total_distance = dist(depot,A)*2
  - TC7: **Multiple stops** — 3 DELIVER stops, verify cumulative load, cumulative time, total distance

---

### simulate_all_routes

- **Idea:** Simulate tất cả truck routes và bike routes. Wrapper gọi simulate_route cho mỗi xe.
- **Input:**
  - `truck_stops, truck_actions: ndarray` — tất cả truck routes
  - `bike_stops, bike_actions: ndarray` — tất cả bike routes
  - `vehicles: ndarray float64 (K, 4)`
  - `customers, dist_matrix, satellites, delta_t`
  - `truck_initial_loads: ndarray float64 (T,)` — mỗi truck chở bao nhiêu từ kho
  - `bike_initial_loads: ndarray float64 (B,)` — mỗi bike chở bao nhiêu từ kho
- **Output:**
  - `truck_states: list[ndarray]` — mỗi phần tử = state array của 1 truck
  - `bike_states: list[ndarray]` — mỗi phần tử = state array của 1 bike
  - `truck_return_times: ndarray float64 (T,)`
  - `bike_return_times: ndarray float64 (B,)`
  - `truck_distances: ndarray float64 (T,)`
  - `bike_distances: ndarray float64 (B,)`
  - `all_feasible: bool`
- **Logic:**
  1. Loop trucks: simulate_route cho mỗi truck
  2. Loop bikes: simulate_route cho mỗi bike
  3. Aggregate results
- **Test cases:**
  - TC1: 2 trucks rỗng, 3 bikes rỗng → all_feasible=True, all return_times=0
  - TC2: 1 truck 1 stop + 1 bike 1 stop → verify từng state riêng
  - TC3: Mixed feasible/infeasible → all_feasible=False

---

## reload.py

### truck_reload_at_stop

- **Idea:** Tính toán khi truck đậu tại 1 customer node để phục vụ N bikes. Trả về: load thay đổi, thời gian phục vụ, thời gian chờ thêm.
- **Input:**
  - `customer_idx: int` — node mà truck đậu
  - `truck_id: int`
  - `current_load: float` — hàng truck đang chở
  - `truck_arrive_time: float` — thời điểm truck đến
  - `satellites: ndarray float64 (S, 5)`
  - `reload_service_time: float` — thời gian cho 1 lần transfer (phút), default 5.0
- **Output:**
  - `load_change: float` — luôn <= 0 (truck cho hàng đi)
  - `total_service: float` — tổng thời gian truck phải đậu (phút)
  - `sync_wait: float` — thời gian chờ bike đầu tiên nếu truck đến sớm
- **Logic:**
  ```
  # Lọc reload events tại (customer_idx, truck_id)
  mask = (satellites[:, SAT_CUST] == customer_idx) &
         (satellites[:, SAT_TRUCK] == truck_id)
  events = satellites[mask]

  if len(events) == 0:
      return 0.0, 0.0, 0.0

  # Sort theo planned_time
  events = events[np.argsort(events[:, SAT_TIME])]

  # Tổng hàng cho đi
  total_kg = events[:, SAT_KG].sum()
  load_change = -total_kg

  # Timeline: phục vụ từng bike tuần tự
  first_bike_time = events[0, SAT_TIME]
  sync_wait = max(0, first_bike_time - truck_arrive_time)

  cursor = truck_arrive_time + sync_wait  # thời điểm bắt đầu phục vụ
  for j in range(len(events)):
      bike_time = events[j, SAT_TIME]
      transfer_start = max(cursor, bike_time)
      cursor = transfer_start + reload_service_time

  total_service = cursor - (truck_arrive_time + sync_wait)

  return load_change, total_service, sync_wait
  ```
- **Test cases:**
  - TC1: **Không có bike nào** → (0, 0, 0)
  - TC2: **1 bike** — truck arrive=100, bike planned=110, svc=5 → sync_wait=10, total_service=5, load_change=-30
  - TC3: **3 bikes cùng lúc** — planned_time=[100,100,100], svc=5 → total_service=15
  - TC4: **3 bikes lần lượt** — planned=[100,110,120], svc=5 →
    - bike_0: start=100, end=105
    - bike_1: start=max(105,110)=110, end=115
    - bike_2: start=max(115,120)=120, end=125
    - total_service=25
  - TC5: **Truck đến sau bike đầu** — truck_arrive=120, bike_planned=100 → sync_wait=0 (truck không chờ, nhưng bike đã phải chờ — check ở phía bike)
  - TC6: **Load check** — 3 bikes nhận [20, 25, 15] = 60kg → load_change = -60

---

### bike_reload_at_stop

- **Idea:** Tính toán khi bike đến 1 customer node để nhận hàng từ truck. Trả về load thay đổi và thời gian.
- **Input:**
  - `customer_idx: int`
  - `bike_id: int`
  - `bike_arrive_time: float`
  - `satellites: ndarray float64 (S, 5)`
  - `reload_service_time: float` — default 5.0
- **Output:**
  - `load_change: float` — luôn >= 0 (bike nhận hàng)
  - `service_time: float`
  - `sync_wait: float` — thời gian bike chờ truck
- **Logic:**
  ```
  mask = (satellites[:, SAT_CUST] == customer_idx) &
         (satellites[:, SAT_BIKE] == bike_id)
  events = satellites[mask]

  if len(events) == 0:
      return 0.0, 0.0, 0.0

  # Giả sử 1 bike chỉ reload 1 lần tại 1 customer
  event = events[0]
  transfer_kg = event[SAT_KG]
  planned_time = event[SAT_TIME]

  # Bike chờ nếu đến sớm hơn planned
  sync_wait = max(0, planned_time - bike_arrive_time)

  return transfer_kg, reload_service_time, sync_wait
  ```
- **Test cases:**
  - TC1: **Không có event** → (0, 0, 0)
  - TC2: **Bike đến đúng giờ** — arrive=100, planned=100 → sync_wait=0, load_change=45, service=5
  - TC3: **Bike đến sớm** — arrive=90, planned=100 → sync_wait=10
  - TC4: **Bike đến trễ** — arrive=110, planned=100 → sync_wait=0 (bike không chờ, nhưng có thể truck đã đi — check ở validate_sync)

---

## validate.py

### validate_delivery_uniqueness

- **Idea:** Kiểm tra mỗi customer được DELIVER đúng 1 lần across tất cả routes.
- **Input:**
  - `truck_stops: ndarray int32 (T, L)`
  - `truck_actions: ndarray int8 (T, L)`
  - `bike_stops: ndarray int32 (B, L)`
  - `bike_actions: ndarray int8 (B, L)`
  - `n_customers: int`
- **Output:**
  - `valid: bool`
  - `unserved: ndarray int32` — customers chưa được giao
  - `duplicates: ndarray int32` — customers bị giao > 1 lần
- **Logic:**
  ```
  count = zeros(n_customers, int)

  # Đếm DELIVER từ trucks
  mask = (truck_actions == ACT_DELIVER) & (truck_stops >= 0)
  np.add.at(count, truck_stops[mask], 1)

  # Đếm DELIVER từ bikes
  mask = (bike_actions == ACT_DELIVER) & (bike_stops >= 0)
  np.add.at(count, bike_stops[mask], 1)

  unserved = where(count == 0)[0]
  duplicates = where(count > 1)[0]
  valid = len(unserved) == 0 and len(duplicates) == 0
  ```
- **Test cases:**
  - TC1: **5 customers, all delivered once** → valid=True, unserved=[], duplicates=[]
  - TC2: **Customer 3 not delivered** → valid=False, unserved=[3]
  - TC3: **Customer 2 delivered by truck AND bike** → valid=False, duplicates=[2]
  - TC4: **Customer appears as RELOAD in truck + DELIVER in bike** → valid=True (RELOAD không đếm)
  - TC5: **Empty routes** → valid=False, unserved = all customers

---

### validate_vehicle_restrictions

- **Idea:** Kiểm tra customer restricted (bike only) không bị assign cho truck DELIVER.
- **Input:**
  - `truck_stops: ndarray int32 (T, L)`
  - `truck_actions: ndarray int8 (T, L)`
  - `restricted: ndarray int8 (N,)`
- **Output:**
  - `valid: bool`
  - `violations: ndarray int32` — restricted customers bị truck DELIVER
- **Logic:**
  ```
  mask = (truck_actions == ACT_DELIVER) & (truck_stops >= 0)
  truck_delivers = truck_stops[mask]
  violations = truck_delivers[restricted[truck_delivers] == 1]
  valid = len(violations) == 0
  ```
- **Test cases:**
  - TC1: **Truck DELIVER customer không restricted** → valid=True
  - TC2: **Truck DELIVER customer restricted** → valid=False, violations=[customer_id]
  - TC3: **Truck RELOAD tại customer restricted** → valid=True (RELOAD ở hẻm OK, truck chỉ đậu ngoài đầu hẻm / hoặc ta cho phép)

---

### validate_capacity_all

- **Idea:** Kiểm tra load không bao giờ < 0 hoặc > capacity tại bất kỳ stop nào, across tất cả routes.
- **Input:**
  - `truck_states: list[ndarray]` — output từ simulate_all_routes
  - `bike_states: list[ndarray]`
  - `vehicles: ndarray float64 (K, 4)`
- **Output:**
  - `valid: bool`
  - `violations: list[tuple(vehicle_type, vehicle_id, stop_idx, load)]` — danh sách vi phạm
- **Logic:**
  ```
  violations = []
  for i, state in enumerate(truck_states):
      if len(state) == 0: continue
      cap = vehicles[truck_vehicle_ids[i], VCOL_CAPACITY]
      load_after = state[:, ST_LOAD_AFT]
      bad = where((load_after < 0) | (load_after > cap))[0]
      for j in bad:
          violations.append(("truck", i, j, load_after[j]))

  # tương tự cho bikes
  valid = len(violations) == 0
  ```
- **Test cases:**
  - TC1: **Tất cả load hợp lệ** → valid=True
  - TC2: **Truck load âm** → violation reported
  - TC3: **Bike load > 60** sau reload → violation reported
  - TC4: **Bike load = 60.0 exactly** → valid=True (boundary OK)

---

### validate_time_windows

- **Idea:** Kiểm tra DELIVER stops không vi phạm time window.
- **Input:**
  - `truck_states: list[ndarray]`
  - `bike_states: list[ndarray]`
  - `customers: ndarray float64 (N, F)`
- **Output:**
  - `valid: bool`
  - `violations: list[tuple(vehicle_type, vehicle_id, customer_idx, arrive_time, tw_close)]`
- **Logic:**
  ```
  violations = []
  for all states (truck + bike):
      for each stop where action == ACT_DELIVER:
          cust = state[i, ST_CUST]
          arrive = state[i, ST_ARRIVE]
          tw_close = customers[cust, COL_TW_CLOSE]
          if arrive > tw_close:
              violations.append(...)
  valid = len(violations) == 0
  ```
- **Test cases:**
  - TC1: **All on time** → valid=True
  - TC2: **Arrive 1 phút sau tw_close** → violation
  - TC3: **Arrive đúng tw_close** → valid=True (boundary)
  - TC4: **RELOAD stop ngoài TW** → không tính violation (TW chỉ áp dụng DELIVER)

---

### validate_sync

- **Idea:** Kiểm tra mỗi reload event: truck và bike thực sự gặp nhau trong khoảng +-delta_t.
- **Input:**
  - `truck_states: list[ndarray]` — state arrays từ simulation
  - `bike_states: list[ndarray]`
  - `satellites: ndarray float64 (S, 5)`
  - `delta_t: float`
- **Output:**
  - `valid: bool`
  - `violations: list[tuple(sat_idx, reason_str, details)]`
- **Logic:**
  ```
  violations = []
  for s in range(len(satellites)):
      cust = satellites[s, SAT_CUST]
      bike_id = satellites[s, SAT_BIKE]
      truck_id = satellites[s, SAT_TRUCK]

      # Tìm truck state tại customer cust với action RELOAD
      t_state = truck_states[truck_id]
      t_mask = (t_state[:, ST_CUST] == cust) & (t_state[:, ST_ACTION] == ACT_RELOAD)
      if not any(t_mask):
          violations.append((s, "truck_not_at_node", {}))
          continue
      truck_arrive = t_state[t_mask][0, ST_ARRIVE]
      truck_depart = t_state[t_mask][0, ST_DEPART]

      # Tìm bike state tại customer cust với action RELOAD
      b_state = bike_states[bike_id]
      b_mask = (b_state[:, ST_CUST] == cust) & (b_state[:, ST_ACTION] == ACT_RELOAD)
      if not any(b_mask):
          violations.append((s, "bike_not_at_node", {}))
          continue
      bike_arrive = b_state[b_mask][0, ST_ARRIVE]

      # Check overlap within delta_t
      if bike_arrive < truck_arrive - delta_t:
          violations.append((s, "bike_too_early",
              {"bike_arrive": bike_arrive, "truck_arrive": truck_arrive, "gap": truck_arrive - bike_arrive}))
      elif bike_arrive > truck_depart + delta_t:
          violations.append((s, "bike_too_late",
              {"bike_arrive": bike_arrive, "truck_depart": truck_depart, "gap": bike_arrive - truck_depart}))

  valid = len(violations) == 0
  ```
- **Test cases:**
  - TC1: **Perfect sync** — bike arrive = truck arrive → valid
  - TC2: **Bike 5 phút sớm, delta_t=10** → valid
  - TC3: **Bike 15 phút sớm, delta_t=10** → violation "bike_too_early"
  - TC4: **Bike đến sau truck đã đi** — bike_arrive > truck_depart + delta_t → violation "bike_too_late"
  - TC5: **Truck không ghé customer** → violation "truck_not_at_node"
  - TC6: **Bike không ghé customer** → violation "bike_not_at_node"
  - TC7: **Nhiều bikes cùng truck, 1 sync OK, 1 fail** → 1 violation

---

### validate_all

- **Idea:** Chạy tất cả validations, trả về tổng hợp.
- **Input:** Tất cả arrays cần thiết
- **Output:**
  - `valid: bool` — tất cả pass
  - `report: dict` — kết quả từng validator
    ```python
    {
        "delivery_uniqueness": {"valid": bool, "unserved": [...], "duplicates": [...]},
        "vehicle_restrictions": {"valid": bool, "violations": [...]},
        "capacity": {"valid": bool, "violations": [...]},
        "time_windows": {"valid": bool, "violations": [...]},
        "sync": {"valid": bool, "violations": [...]},
    }
    ```
- **Test cases:**
  - TC1: **Solution hoàn toàn hợp lệ** → valid=True, tất cả sub-validators pass
  - TC2: **Nhiều loại violation cùng lúc** → valid=False, report chứa tất cả violations
  - TC3: **Solution rỗng** → delivery_uniqueness fail (unserved = all)

---

## fitness.py

### compute_cost

- **Idea:** Tính tổng chi phí vận hành (distance * cost_per_km) cho tất cả xe.
- **Input:**
  - `truck_distances: ndarray float64 (T,)`
  - `bike_distances: ndarray float64 (B,)`
  - `vehicles: ndarray float64 (K, 4)`
- **Output:**
  - `total_cost: float`
  - `truck_costs: ndarray float64 (T,)` — chi phí từng truck
  - `bike_costs: ndarray float64 (B,)` — chi phí từng bike
- **Logic:**
  ```
  truck_costs = truck_distances * vehicles[truck_ids, VCOL_COST_KM]
  bike_costs = bike_distances * vehicles[bike_ids, VCOL_COST_KM]
  total_cost = truck_costs.sum() + bike_costs.sum()
  ```
- **Test cases:**
  - TC1: 1 truck chạy 100km, cost_km=5 → cost=500
  - TC2: Không xe nào chạy → cost=0
  - TC3: 2 trucks + 3 bikes → tổng đúng

---

### compute_makespan

- **Idea:** Thời gian xe cuối cùng về tới kho.
- **Input:**
  - `truck_return_times: ndarray float64 (T,)`
  - `bike_return_times: ndarray float64 (B,)`
- **Output:**
  - `makespan: float`
- **Logic:** `max(truck_return_times.max(), bike_return_times.max())` — xử lý trường hợp array rỗng
- **Test cases:**
  - TC1: trucks=[100, 200], bikes=[150, 180] → makespan=200
  - TC2: Tất cả return_time=0 (routes rỗng) → makespan=0

---

### compute_penalties

- **Idea:** Tính tổng penalty từ tất cả violations. Mỗi loại violation có hệ số penalty riêng.
- **Input:**
  - `report: dict` — output từ validate_all
  - `penalty_weights: dict` — hệ số cho mỗi loại violation
    ```python
    {
        "unserved": 1000.0,      # per customer
        "duplicate": 1000.0,     # per customer
        "capacity": 100.0,       # per violation
        "time_window": 50.0,     # per minute late
        "sync": 200.0,           # per failed sync
        "vehicle_restriction": 500.0,  # per violation
    }
    ```
- **Output:**
  - `total_penalty: float`
  - `penalty_breakdown: dict` — penalty từng loại
- **Logic:**
  ```
  p = 0
  breakdown = {}

  breakdown["unserved"] = len(report["delivery_uniqueness"]["unserved"]) * weights["unserved"]
  breakdown["duplicate"] = len(report["delivery_uniqueness"]["duplicates"]) * weights["duplicate"]
  breakdown["capacity"] = len(report["capacity"]["violations"]) * weights["capacity"]

  # TW: penalty tỉ lệ với số phút trễ
  tw_violations = report["time_windows"]["violations"]
  tw_penalty = sum(arrive - tw_close for (_, _, _, arrive, tw_close) in tw_violations) * weights["time_window"]
  breakdown["time_window"] = tw_penalty

  breakdown["sync"] = len(report["sync"]["violations"]) * weights["sync"]
  breakdown["vehicle_restriction"] = len(report["vehicle_restrictions"]["violations"]) * weights["vehicle_restriction"]

  total_penalty = sum(breakdown.values())
  ```
- **Test cases:**
  - TC1: **Không violation** → penalty=0, breakdown tất cả = 0
  - TC2: **2 unserved customers** → penalty = 2 * 1000 = 2000
  - TC3: **TW violation 10 phút trễ** → penalty = 10 * 50 = 500
  - TC4: **Mixed violations** → tổng các loại

---

### compute_fitness

- **Idea:** Hàm fitness tổng hợp = weighted sum of cost + makespan + penalties.
- **Input:**
  - `total_cost: float`
  - `makespan: float`
  - `total_penalty: float`
  - `w1: float` — weight cost, default 0.6
  - `w2: float` — weight makespan, default 0.3
  - `w3: float` — weight penalty, default 0.1 (nhưng thay đổi theo iteration)
- **Output:**
  - `fitness: float` — càng nhỏ càng tốt
- **Logic:** `fitness = w1 * cost + w2 * makespan + w3 * penalty`
- **Test cases:**
  - TC1: cost=1000, makespan=200, penalty=0, w=(0.6, 0.3, 0.1) → 0.6*1000 + 0.3*200 + 0 = 660
  - TC2: penalty=0 → fitness chỉ phụ thuộc cost + makespan
  - TC3: cost=0, makespan=0, penalty=5000 → fitness = 0.1*5000 = 500

---

### evaluate_solution

- **Idea:** One-stop function: simulate tất cả routes → validate → tính fitness. Đây là hàm được gọi nhiều nhất trong ALNS loop.
- **Input:**
  - `truck_stops, truck_actions, bike_stops, bike_actions, satellites` — solution encoding
  - `customers, restricted, vehicles, dist_matrix` — problem data
  - `truck_initial_loads, bike_initial_loads: ndarray float64`
  - `delta_t: float`
  - `penalty_weights: dict`
  - `fitness_weights: tuple (w1, w2, w3)`
- **Output:**
  - `fitness: float`
  - `cost: float`
  - `makespan: float`
  - `total_penalty: float`
  - `feasible: bool`
  - `report: dict` — full validation report (cho debug)
  - `truck_states, bike_states: list[ndarray]` — full state (cho visualization)
- **Logic:**
  ```
  1. simulate_all_routes → states, return_times, distances, sim_feasible
  2. validate_all → valid, report
  3. compute_cost → cost
  4. compute_makespan → makespan
  5. compute_penalties → penalty
  6. compute_fitness → fitness
  7. feasible = sim_feasible and valid
  8. return everything
  ```
- **Test cases:**
  - TC1: **Solution hợp lệ hoàn chỉnh** → feasible=True, penalty=0, fitness > 0
  - TC2: **Solution rỗng** → feasible=False (unserved customers), penalty rất lớn
  - TC3: **Solution 1 truck serves all** → check cost, makespan hợp lý
  - TC4: **Gọi 2 lần cùng input** → cùng output (deterministic)

---

## Tổng kết Phase 02

| File | Function | Mục đích |
|------|----------|----------|
| simulate.py | simulate_route | Mô phỏng 1 route từng stop |
| simulate.py | simulate_all_routes | Mô phỏng tất cả routes |
| reload.py | truck_reload_at_stop | Truck phục vụ N bikes tại 1 stop |
| reload.py | bike_reload_at_stop | Bike nhận hàng từ truck tại 1 stop |
| validate.py | validate_delivery_uniqueness | Mỗi customer đúng 1 DELIVER |
| validate.py | validate_vehicle_restrictions | Restricted → không truck DELIVER |
| validate.py | validate_capacity_all | Load không < 0 hoặc > capacity |
| validate.py | validate_time_windows | DELIVER không trễ TW |
| validate.py | validate_sync | Truck-bike gặp nhau đúng +-delta_t |
| validate.py | validate_all | Wrapper tất cả validators |
| fitness.py | compute_cost | Tổng chi phí distance * cost/km |
| fitness.py | compute_makespan | Thời gian xe cuối cùng về kho |
| fitness.py | compute_penalties | Tổng penalty từ violations |
| fitness.py | compute_fitness | Weighted sum: cost + makespan + penalty |
| fitness.py | evaluate_solution | One-stop: simulate → validate → fitness |
