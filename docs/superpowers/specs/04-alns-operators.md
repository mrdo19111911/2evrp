# Phase 04 — ALNS Operators

## File structure

```
src/
  alns/
    destroy.py        ← destroy operators
    repair.py         ← repair operators
    crosslayer.py     ← cross-layer operators (assignment changes)
    local_search.py   ← local search improvement operators
    weights.py        ← adaptive weight management
```

Mỗi operator là 1 function cùng signature. Destroy trả về removed customers, Repair nhận removed customers và chèn lại.

---

## destroy.py

### Signature chung

```python
def destroy_xxx(sol, customers, dist_matrix, rng, params) -> ndarray int32:
    """
    Input:
      sol: dict         — solution hiện tại (SẼ BỊ MODIFY in-place)
      customers: ndarray — problem data
      dist_matrix: ndarray
      rng: Generator     — numpy random
      params: dict       — hyperparameters
    Output:
      removed: ndarray int32 — danh sách customer indices đã bị loại ra
    """
```

Mỗi destroy operator:
1. Chọn customers để loại
2. Gọi `remove_stop()` cho mỗi customer
3. Xoá satellite events liên quan
4. Trả về danh sách đã loại

---

### random_removal

- **Idea:** Loại q customers ngẫu nhiên. Đơn giản nhất, đảm bảo diversity.
- **Params:**
  - `q_min: int` — số customer tối thiểu loại, default 3
  - `q_max: int` — tối đa, default min(15, N//5)
- **Logic:**
  ```
  q = rng.integers(params["q_min"], params["q_max"] + 1)
  assigned = get_assigned_customers(sol, N)
  q = min(q, len(assigned))
  targets = rng.choice(assigned, size=q, replace=False)

  removed = []
  for c in targets:
      vtype, vid, pos = get_customer_info(sol, c)
      remove_stop(sol, vtype, vid, pos, dist_matrix, customers)
      remove_satellites_for_customer(sol, c)
      removed.append(c)

  return array(removed, int32)
  ```
- **Test cases:**
  - TC1: Solution 10 customers → remove q ∈ [3, 5] → len(removed) ∈ [3, 5]
  - TC2: Sau remove, removed customers unassigned
  - TC3: Deterministic với cùng rng seed
  - TC4: q > assigned → remove tất cả assigned

---

### worst_cost_removal

- **Idea:** Loại q customers có chi phí phục vụ CAO nhất. Chi phí = removal_cost_delta (nếu bỏ customer ra tiết kiệm nhiều → customer đó đắt).
- **Params:**
  - `q_min, q_max: int`
  - `noise: float` — thêm random noise vào cost để diversify, default 0.1
- **Logic:**
  ```
  assigned = get_assigned_customers(sol, N)
  costs = zeros(len(assigned))

  for i, c in enumerate(assigned):
      vtype, vid, pos = get_customer_info(sol, c)
      # delta âm → bỏ ra tiết kiệm delta. cost = -delta = savings
      costs[i] = -removal_cost_delta(sol, vtype, vid, pos, dist_matrix)
      # Thêm noise
      costs[i] += rng.uniform(0, noise * costs[i])

  q = rng.integers(params["q_min"], params["q_max"] + 1)
  q = min(q, len(assigned))

  # Chọn q đắt nhất
  targets = assigned[argsort(costs)[-q:]]

  removed = []
  # Remove theo thứ tự pos giảm dần (tránh shift ảnh hưởng)
  for c in _sort_by_pos_desc(targets, sol):
      vtype, vid, pos = get_customer_info(sol, c)
      remove_stop(sol, vtype, vid, pos, dist_matrix, customers)
      remove_satellites_for_customer(sol, c)
      removed.append(c)

  return array(removed, int32)
  ```
- **Lưu ý quan trọng:** Khi remove nhiều customers từ cùng route, phải remove từ **pos lớn nhất trước** → tránh shift index ảnh hưởng các pos nhỏ hơn.
- **Test cases:**
  - TC1: Customer nằm xa route (detour lớn) → bị loại trước
  - TC2: Customer nằm trên đường (detour nhỏ) → ít bị loại
  - TC3: Noise=0 → deterministic (sorted by cost)
  - TC4: Remove thứ tự pos desc → index không bị lệch

---

### shaw_removal

- **Idea:** Loại q customers **tương tự nhau** (gần nhau + demand gần + TW gần). Tạo "lỗ trống" coherent → dễ repair tốt hơn.
- **Params:**
  - `q_min, q_max: int`
  - `phi_dist: float` — weight khoảng cách, default 0.4
  - `phi_demand: float` — weight demand, default 0.3
  - `phi_tw: float` — weight time window, default 0.3
  - `randomness: float` — determinism, default 6.0 (cao → chọn giống nhất)
- **Logic:**
  ```
  assigned = get_assigned_customers(sol, N)
  seed = rng.choice(assigned)

  # Tính relatedness giữa seed và mọi assigned customer
  relatedness = zeros(len(assigned))
  for i, c in enumerate(assigned):
      d = dist_matrix[seed + 1, c + 1]
      demand_diff = abs(customers[seed, COL_DEMAND] - customers[c, COL_DEMAND])
      tw_diff = abs(customers[seed, COL_TW_OPEN] - customers[c, COL_TW_OPEN])

      # Normalize
      d_norm = d / dist_matrix.max()
      dem_norm = demand_diff / customers[:, COL_DEMAND].max()
      tw_norm = tw_diff / DAY_LENGTH

      relatedness[i] = phi_dist * d_norm + phi_demand * dem_norm + phi_tw * tw_norm

  q = rng.integers(params["q_min"], params["q_max"] + 1)
  q = min(q, len(assigned))

  # Chọn q customers giống seed nhất (relatedness thấp nhất)
  # Dùng randomized selection: chọn theo xác suất tỉ lệ nghịch với relatedness
  targets = [seed]
  remaining = list(assigned[assigned != seed])

  while len(targets) < q and remaining:
      # relatedness tới TẬP targets hiện tại (min relatedness tới bất kỳ target nào)
      scores = []
      for c in remaining:
          min_rel = min(dist_matrix[c + 1, t + 1] for t in targets)
          scores.append(min_rel)
      scores = array(scores)

      # Chọn theo rank: pos = floor(random^randomness * len)
      pos = int(rng.random() ** randomness * len(remaining))
      sorted_idx = argsort(scores)
      chosen = remaining[sorted_idx[pos]]
      targets.append(chosen)
      remaining.remove(chosen)

  # Remove
  removed = []
  for c in _sort_by_pos_desc(targets, sol):
      vtype, vid, pos = get_customer_info(sol, c)
      remove_stop(sol, vtype, vid, pos, dist_matrix, customers)
      remove_satellites_for_customer(sol, c)
      removed.append(c)

  return array(removed, int32)
  ```
- **Test cases:**
  - TC1: Seed ở cụm → loại cả cụm customers gần nhau
  - TC2: randomness cao → gần như deterministic (chọn giống nhất)
  - TC3: randomness thấp → random hơn
  - TC4: Seed customer luôn nằm trong removed

---

### route_removal

- **Idea:** Loại toàn bộ 1 route — phá cấu trúc lớn, cho phép re-route hoàn toàn.
- **Params:**
  - `allow_truck: bool` — cho phép loại truck route, default True
  - `allow_bike: bool` — default True
- **Logic:**
  ```
  # Thu thập các route không rỗng
  nonempty = []
  if allow_truck:
      for t in range(sol["n_trucks"]):
          if sol["truck_lengths"][t] > 0:
              nonempty.append((VEH_TRUCK, t))
  if allow_bike:
      for b in range(sol["n_bikes"]):
          if sol["bike_lengths"][b] > 0:
              nonempty.append((VEH_BIKE, b))

  if not nonempty:
      return empty(0, int32)

  vtype, vid = nonempty[rng.integers(len(nonempty))]
  removed_pairs = clear_route(sol, vtype, vid)

  # Xoá satellites liên quan
  removed = []
  for c, action in removed_pairs:
      if action == ACT_DELIVER:
          removed.append(c)
      remove_satellites_for_customer(sol, c)

  return array(removed, int32)
  ```
- **Test cases:**
  - TC1: 1 route 5 stops (3 DELIVER, 2 RELOAD) → removed = 3 customers
  - TC2: Tất cả routes rỗng → removed empty
  - TC3: RELOAD stops bị xoá nhưng KHÔNG nằm trong removed (chỉ DELIVER)

---

### satellite_removal

- **Idea:** Chọn 1 satellite, loại TẤT CẢ bike customers phụ thuộc vào satellite đó. Phá 1 cluster bike hoàn toàn.
- **Params:** (không có params đặc biệt)
- **Logic:**
  ```
  sats = sol["satellites"]
  if len(sats) == 0:
      return empty(0, int32)

  # Tìm unique satellite customer nodes
  unique_sat_custs = unique(sats[:, SAT_CUST].astype(int))
  sat_cust = rng.choice(unique_sat_custs)

  # Tìm tất cả bikes reload tại satellite này
  mask = sats[:, SAT_CUST] == sat_cust
  bike_ids = unique(sats[mask, SAT_BIKE].astype(int))

  # Loại tất cả DELIVER customers trong routes của các bikes này
  removed = []
  for bid in bike_ids:
      deliver_custs = get_route_customers_only(sol, VEH_BIKE, bid)
      for c in _sort_by_pos_desc(deliver_custs, sol):
          vtype, vid, pos = get_customer_info(sol, c)
          remove_stop(sol, vtype, vid, pos, dist_matrix, customers)
          removed.append(c)
      # Clear toàn bộ RELOAD stops còn lại trong bike route
      clear_route(sol, VEH_BIKE, bid)
      remove_satellites_for_bike(sol, bid)

  return array(removed, int32)
  ```
- **Test cases:**
  - TC1: Satellite có 3 bikes, mỗi bike 4 DELIVER → removed = 12
  - TC2: Satellite không có bike nào → removed empty
  - TC3: Sau remove, bike routes cleared, satellite events xoá

---

### zone_removal

- **Idea:** Loại tất cả customers trong 1 vùng tròn xung quanh 1 customer. Phá cấu trúc spatial.
- **Params:**
  - `zone_pct: float` — % distance dùng làm radius, default 15
- **Logic:**
  ```
  assigned = get_assigned_customers(sol, N)
  if len(assigned) == 0:
      return empty(0, int32)

  center = rng.choice(assigned)
  cx, cy = customers[center, COL_X], customers[center, COL_Y]

  dists_to_center = sqrt(
      (customers[assigned, COL_X] - cx)**2 +
      (customers[assigned, COL_Y] - cy)**2
  )

  radius = percentile(dists_to_center, zone_pct)
  targets = assigned[dists_to_center <= radius]

  removed = []
  for c in _sort_by_pos_desc(targets, sol):
      vtype, vid, pos = get_customer_info(sol, c)
      remove_stop(sol, vtype, vid, pos, dist_matrix, customers)
      remove_satellites_for_customer(sol, c)
      removed.append(c)

  return array(removed, int32)
  ```
- **Test cases:**
  - TC1: zone_pct=15 → ~15% customers bị loại
  - TC2: zone_pct=100 → loại tất cả
  - TC3: Customers bị loại nằm gần nhau (trong vùng tròn)

---

## repair.py

### Signature chung

```python
def repair_xxx(sol, removed, customers, restricted, dist_matrix, vehicles, rng, params) -> None:
    """
    Input:
      sol: dict         — solution (SẼ BỊ MODIFY in-place)
      removed: ndarray int32 — customers cần chèn lại
      customers, restricted, dist_matrix, vehicles: problem data
      rng: Generator
      params: dict
    Output:
      None (modify sol in-place, tất cả removed phải được insert lại)
    """
```

---

### greedy_insertion

- **Idea:** Chèn từng customer vào vị trí rẻ nhất (chi phí tăng ít nhất). Thứ tự chèn: random.
- **Params:** (không có)
- **Logic:**
  ```
  remaining = list(rng.permutation(removed))

  for c in remaining:
      best_cost = inf
      best_move = None  # (vtype, vid, pos)

      demand = customers[c, COL_DEMAND]

      # Thử truck routes
      if restricted[c] != 1:  # không bị restrict
          vid, pos, delta = find_best_insertion_all_routes(
              sol, VEH_TRUCK, c, dist_matrix, customers, TRUCK_CAPACITY)
          if delta < best_cost:
              best_cost = delta
              best_move = (VEH_TRUCK, vid, pos)

      # Thử bike routes
      if demand <= BIKE_CAPACITY:
          vid, pos, delta = find_best_insertion_all_routes(
              sol, VEH_BIKE, c, dist_matrix, customers, BIKE_CAPACITY)
          if delta < best_cost:
              best_cost = delta
              best_move = (VEH_BIKE, vid, pos)

      if best_move is not None:
          vtype, vid, pos = best_move
          insert_stop(sol, vtype, vid, pos, c, ACT_DELIVER, dist_matrix, customers)
      else:
          # Không có chỗ → chèn vào route ít tải nhất (sẽ infeasible, penalty xử lý)
          _force_insert(sol, c, dist_matrix, customers)
  ```
- **Test cases:**
  - TC1: 3 removed, 2 routes có chỗ → tất cả insert thành công
  - TC2: Customer restricted → chỉ insert vào bike routes
  - TC3: Tất cả routes đầy → force insert (infeasible nhưng không crash)
  - TC4: Sau repair, tất cả removed đều assigned

---

### regret_k_insertion

- **Idea:** Chèn customer có **regret cao nhất** trước. Regret = cost(2nd best) - cost(best). Customer khó chèn (ít chỗ tốt) → ưu tiên chèn trước.
- **Params:**
  - `k: int` — regret order, default 3
- **Logic:**
  ```
  remaining = list(removed)

  while remaining:
      best_regret = -inf
      best_cust = None
      best_move = None

      for c in remaining:
          # Tìm top-k insertion positions
          top_k = _find_top_k_insertions(sol, c, k, customers, restricted,
                                          dist_matrix)
          # top_k = list of (cost, vtype, vid, pos), sorted ascending

          if len(top_k) == 0:
              # Không có chỗ → regret = rất lớn (ưu tiên chèn trước)
              regret = inf
              move = None
          elif len(top_k) == 1:
              regret = inf  # chỉ 1 chỗ → phải chèn ngay
              move = top_k[0]
          else:
              regret = sum(top_k[i][0] - top_k[0][0] for i in range(1, len(top_k)))
              move = top_k[0]

          if regret > best_regret:
              best_regret = regret
              best_cust = c
              best_move = move

      if best_move is not None:
          cost, vtype, vid, pos = best_move
          insert_stop(sol, vtype, vid, pos, best_cust, ACT_DELIVER,
                      dist_matrix, customers)
      else:
          _force_insert(sol, best_cust, dist_matrix, customers)

      remaining.remove(best_cust)
  ```
- **Test cases:**
  - TC1: Customer với chỉ 1 vị trí feasible → regret=inf → insert đầu tiên
  - TC2: Customer với nhiều vị trí tốt ngang nhau → regret thấp → insert sau
  - TC3: k=2 vs k=3 → thứ tự insert có thể khác
  - TC4: Tất cả removed insert thành công

---

### satellite_aware_insertion

- **Idea:** Ưu tiên chèn customer gần satellite đang active → bike. Tận dụng satellite có sẵn, giảm truck detour.
- **Params:**
  - `sat_threshold_km: float` — ngưỡng khoảng cách tới satellite, default 5.0
- **Logic:**
  ```
  # Tìm active satellites
  active_sats = unique(sol["satellites"][:, SAT_CUST].astype(int))

  # Sort removed: gần satellite nhất trước
  if len(active_sats) > 0:
      min_dist_to_sat = array([
          dist_matrix[c + 1, active_sats + 1].min() for c in removed
      ])
      order = argsort(min_dist_to_sat)
  else:
      order = rng.permutation(len(removed))

  for idx in order:
      c = removed[idx]
      demand = customers[c, COL_DEMAND]

      # Nếu gần satellite + demand phù hợp bike → chèn vào bike route reload ở satellite đó
      inserted = False
      if len(active_sats) > 0 and demand <= BIKE_CAPACITY:
          nearest_sat = active_sats[argmin(dist_matrix[c + 1, active_sats + 1])]
          dist_to_sat = dist_matrix[c + 1, nearest_sat + 1]

          if dist_to_sat < sat_threshold_km:
              # Tìm bike route đã reload ở satellite này + có chỗ
              for b in range(sol["n_bikes"]):
                  if not can_insert_customer(sol, VEH_BIKE, b, c, customers,
                                              restricted, BIKE_CAPACITY):
                      continue
                  # Check bike route đã có reload tại nearest_sat
                  route = get_route_as_list(sol, VEH_BIKE, b)
                  has_sat = any(node == nearest_sat and act == ACT_RELOAD
                               for node, act in route)
                  if has_sat:
                      pos, delta = best_insertion_pos(sol, VEH_BIKE, b, c, dist_matrix)
                      insert_stop(sol, VEH_BIKE, b, pos, c, ACT_DELIVER,
                                  dist_matrix, customers)
                      inserted = True
                      break

      if not inserted:
          # Fallback: greedy insert (cả truck + bike)
          _greedy_insert_single(sol, c, customers, restricted, dist_matrix)
  ```
- **Test cases:**
  - TC1: Customer 2km từ active satellite → insert vào bike route đúng satellite
  - TC2: Customer xa mọi satellite → fallback greedy
  - TC3: Không có active satellites → pure greedy
  - TC4: Bike route tại satellite đầy → fallback

---

## crosslayer.py

### Signature chung

```python
def cross_xxx(sol, customers, restricted, dist_matrix, satellites_arr, rng, params) -> bool:
    """
    Cross-layer operators thay đổi ASSIGNMENT (truck ↔ bike).
    Return True nếu thay đổi thành công, False nếu không tìm được move.
    """
```

---

### swap_assignment

- **Idea:** Chuyển 1 customer từ truck → bike hoặc ngược lại. Nếu chuyển sang bike → cần đảm bảo có satellite hoặc tạo satellite mới.
- **Logic:**
  ```
  # Chọn random customer flexible (không forced truck/bike)
  assigned = get_assigned_customers(sol, N)
  flexible = [c for c in assigned
              if customers[c, COL_DEMAND] <= BIKE_CAPACITY and restricted[c] != 1]

  if not flexible:
      return False

  c = rng.choice(flexible)
  vtype, vid, pos = get_customer_info(sol, c)

  if vtype == VEH_TRUCK:
      # Truck → Bike: tìm bike route tốt nhất
      if not can_insert_customer(sol, VEH_BIKE, ANY_ROUTE, c, ...):
          return False
      remove_stop(sol, VEH_TRUCK, vid, pos, dist_matrix, customers)
      best_vid, best_pos, _ = find_best_insertion_all_routes(
          sol, VEH_BIKE, c, dist_matrix, customers, BIKE_CAPACITY)
      if best_vid >= 0:
          insert_stop(sol, VEH_BIKE, best_vid, best_pos, c, ACT_DELIVER, ...)
          # Có thể cần thêm RELOAD event nếu bike cần reload cho c
          _ensure_satellite_for_bike(sol, best_vid, c, ...)
          return True
      else:
          # Rollback: insert lại vào truck
          insert_stop(sol, VEH_TRUCK, vid, pos, c, ACT_DELIVER, ...)
          return False

  else:
      # Bike → Truck: đơn giản hơn, không cần satellite
      remove_stop(sol, VEH_BIKE, vid, pos, dist_matrix, customers)
      remove_satellites_for_customer(sol, c)
      best_vid, best_pos, _ = find_best_insertion_all_routes(
          sol, VEH_TRUCK, c, dist_matrix, customers, TRUCK_CAPACITY)
      if best_vid >= 0:
          insert_stop(sol, VEH_TRUCK, best_vid, best_pos, c, ACT_DELIVER, ...)
          return True
      else:
          insert_stop(sol, VEH_BIKE, vid, pos, c, ACT_DELIVER, ...)
          return False
  ```
- **Test cases:**
  - TC1: Customer demand=30 in truck → move to bike → success
  - TC2: Customer demand=80 → cannot move to bike (>60) → False
  - TC3: Customer restricted → cannot move to truck → False
  - TC4: No feasible bike route → rollback, return False
  - TC5: Satellite created/updated khi move truck→bike

---

### relocate_satellite

- **Idea:** Di chuyển 1 satellite sang customer node khác. Tất cả bike reload events tại satellite cũ → di chuyển sang satellite mới.
- **Logic:**
  ```
  sats = sol["satellites"]
  if len(sats) == 0:
      return False

  # Chọn random satellite hiện tại
  unique_sats = unique(sats[:, SAT_CUST].astype(int))
  old_sat = rng.choice(unique_sats)

  # Tìm candidates: customer nodes gần old_sat
  assigned = get_assigned_customers(sol, N)
  dists = dist_matrix[old_sat + 1, assigned + 1]
  nearby = assigned[argsort(dists)[:10]]  # top 10 gần nhất

  # Chọn new_sat = candidate tốt nhất (gần nhiều bike customers nhất)
  best_new = None
  best_score = inf
  for cand in nearby:
      if cand == old_sat:
          continue
      # Score = tổng distance từ candidate tới bike customers đang reload ở old_sat
      mask = sats[:, SAT_CUST] == old_sat
      bike_ids = sats[mask, SAT_BIKE].astype(int)
      # Tìm bike customers
      score = 0
      for bid in bike_ids:
          bike_custs = get_route_customers_only(sol, VEH_BIKE, bid)
          score += dist_matrix[cand + 1, bike_custs + 1].sum()
      if score < best_score:
          best_score = score
          best_new = cand

  if best_new is None:
      return False

  # Migrate: update satellite events
  mask = sats[:, SAT_CUST] == old_sat
  sol["satellites"][mask, SAT_CUST] = best_new

  # Update routes: thay old_sat → new_sat trong truck và bike RELOAD stops
  _replace_reload_node(sol, old_sat, best_new)

  return True
  ```
- **Test cases:**
  - TC1: Satellite di chuyển gần hơn bike customers → tổng distance giảm
  - TC2: Không có candidate tốt hơn → return False
  - TC3: Routes updated: RELOAD stops trỏ sang node mới

---

### create_new_satellite

- **Idea:** Chọn 1 customer node chưa phải satellite, biến nó thành satellite mới. Tìm bikes gần đó chuyển reload sang satellite mới.
- **Logic:**
  ```
  # Tìm bike routes có total distance dài (kém hiệu quả)
  bike_dists = sol["bike_distances"]
  if bike_dists.max() == 0:
      return False

  # Chọn bike route dài nhất
  worst_bike = argmax(bike_dists)
  bike_custs = get_route_customers_only(sol, VEH_BIKE, worst_bike)
  if len(bike_custs) == 0:
      return False

  # Tìm customer node trung tâm của bike route → candidate satellite
  cx = customers[bike_custs, COL_X].mean()
  cy = customers[bike_custs, COL_Y].mean()
  assigned = get_assigned_customers(sol, N)
  dists_to_center = sqrt(
      (customers[assigned, COL_X] - cx)**2 +
      (customers[assigned, COL_Y] - cy)**2
  )
  new_sat = assigned[argmin(dists_to_center)]

  # Tìm truck route gần new_sat nhất → thêm RELOAD stop
  best_truck = None
  best_delta = inf
  for t in range(sol["n_trucks"]):
      pos, delta = best_insertion_pos(sol, VEH_TRUCK, t, new_sat, dist_matrix)
      if delta < best_delta:
          best_delta = delta
          best_truck = (t, pos)

  if best_truck is None:
      return False

  t, pos = best_truck
  insert_stop(sol, VEH_TRUCK, t, pos, new_sat, ACT_RELOAD, dist_matrix, customers)

  # Thêm RELOAD stop vào bike route
  bike_pos, _ = best_insertion_pos(sol, VEH_BIKE, worst_bike, new_sat, dist_matrix)
  insert_stop(sol, VEH_BIKE, worst_bike, bike_pos, new_sat, ACT_RELOAD,
              dist_matrix, customers)

  # Create satellite event
  transfer_kg = _estimate_transfer_kg(sol, worst_bike, bike_pos, customers)
  planned_time = _estimate_time(sol, worst_bike, bike_pos, dist_matrix, customers)
  add_satellite_event(sol, new_sat, worst_bike, t, transfer_kg, planned_time)

  return True
  ```
- **Test cases:**
  - TC1: Bike route dài → satellite tạo ở giữa → distance giảm
  - TC2: Không có truck route gần → return False
  - TC3: Satellite event created với transfer_kg hợp lý

---

### remove_satellite_node

- **Idea:** Xoá 1 satellite hoàn toàn — bike không reload ở đó nữa, truck không ghé nữa. Ngược lại với create_new_satellite.
- **Logic:**
  ```
  sats = sol["satellites"]
  if len(sats) == 0:
      return False

  unique_sats = unique(sats[:, SAT_CUST].astype(int))
  target_sat = rng.choice(unique_sats)

  # Xoá satellite events
  remove_satellites_for_customer(sol, target_sat)

  # Xoá RELOAD stops tại target_sat trong tất cả routes
  _remove_all_reload_at_node(sol, target_sat, dist_matrix, customers)

  return True
  ```
- **Test cases:**
  - TC1: Satellite xoá → RELOAD stops xoá → routes ngắn hơn
  - TC2: Bike routes không còn reload → phải về kho lấy hàng (có thể infeasible)

---

## local_search.py

### two_opt

- **Idea:** Cải thiện 1 route bằng cách đảo ngược 1 đoạn. Cổ điển, hiệu quả cho TSP-like.
- **Input:**
  - `sol: dict`, `vtype: int`, `vid: int`, `dist_matrix: ndarray`
- **Output:** `improved: bool`
- **Logic:**
  ```
  L = _get_lengths(sol, vtype)[vid]
  if L < 3:
      return False

  improved = False
  for i in range(L - 1):
      for j in range(i + 2, L):
          # Delta nếu reverse segment [i, j]
          delta = _two_opt_delta(sol, vtype, vid, i, j, dist_matrix)
          if delta < -1e-6:
              reverse_segment(sol, vtype, vid, i, j, dist_matrix)
              improved = True

  return improved
  ```
- **Performance:** O(route_len^2) — chấp nhận được vì route_len thường < 50
- **Test cases:**
  - TC1: Route có crossing → 2-opt uncross → distance giảm
  - TC2: Route đã optimal → không thay đổi
  - TC3: Route 2 stops → skip (< 3)

---

### or_opt

- **Idea:** Di chuyển 1 segment (1-3 stops liên tiếp) tới vị trí khác trong cùng route.
- **Input:** `sol: dict`, `vtype: int`, `vid: int`, `dist_matrix: ndarray`
- **Output:** `improved: bool`
- **Logic:**
  ```
  L = _get_lengths(sol, vtype)[vid]
  improved = False

  for seg_len in [1, 2, 3]:
      for i in range(L - seg_len + 1):
          for j in range(L + 1):
              if j >= i and j <= i + seg_len:
                  continue  # skip overlap
              delta = _or_opt_delta(sol, vtype, vid, i, seg_len, j, dist_matrix)
              if delta < -1e-6:
                  _do_or_opt_move(sol, vtype, vid, i, seg_len, j, dist_matrix)
                  improved = True
                  break  # restart inner loop

  return improved
  ```
- **Test cases:**
  - TC1: 1 stop ở vị trí tệ → move tới vị trí tốt hơn
  - TC2: Segment 3 stops → move cả block
  - TC3: Đã optimal → no change

---

### relocate_inter_route

- **Idea:** Di chuyển 1 customer từ route A sang route B (cross-route). Thử mọi cặp route cùng loại xe.
- **Input:**
  - `sol: dict`, `vtype: int`, `dist_matrix, customers: ndarray`
- **Output:** `improved: bool`
- **Logic:**
  ```
  n_vehicles = sol["n_trucks"] if vtype == VEH_TRUCK else sol["n_bikes"]
  improved = False

  for va in range(n_vehicles):
      La = _get_lengths(sol, vtype)[va]
      for pos_a in range(La):
          c = _get_stop(sol, vtype, va, pos_a)
          if _get_action(sol, vtype, va, pos_a) != ACT_DELIVER:
              continue

          removal_delta = removal_cost_delta(sol, vtype, va, pos_a, dist_matrix)

          for vb in range(n_vehicles):
              if vb == va:
                  continue
              if not can_insert_customer(sol, vtype, vb, c, customers, restricted, capacity):
                  continue

              ins_pos, ins_delta = best_insertion_pos(sol, vtype, vb, c, dist_matrix)
              total_delta = removal_delta + ins_delta

              if total_delta < -1e-6:
                  move_stop(sol, vtype, va, pos_a, vtype, vb, ins_pos,
                           dist_matrix, customers)
                  improved = True
                  break  # restart

  return improved
  ```
- **Test cases:**
  - TC1: Customer ở route A xa → move sang route B gần hơn
  - TC2: Capacity constraint violated → skip route B
  - TC3: No improvement possible → return False

---

## weights.py

### init_weights

- **Idea:** Khởi tạo uniform weights cho tất cả operators.
- **Input:**
  - `n_destroy: int`, `n_repair: int`, `n_cross: int`
- **Output:**
  - `destroy_weights: ndarray float64 (D,)` — uniform, sum=1
  - `repair_weights: ndarray float64 (R,)` — uniform, sum=1
  - `cross_weights: ndarray float64 (C,)` — uniform, sum=1
  - `destroy_scores: ndarray float64 (D,)` — zeros
  - `repair_scores, cross_scores: ndarray` — zeros
  - `destroy_counts, repair_counts, cross_counts: ndarray int32` — zeros
- **Test cases:**
  - TC1: 6 destroy ops → weights = [1/6, 1/6, ...], sum=1

---

### select_operator

- **Idea:** Chọn operator theo xác suất (roulette wheel selection).
- **Input:**
  - `weights: ndarray float64`
  - `rng: Generator`
- **Output:**
  - `idx: int` — index operator được chọn
- **Logic:** `rng.choice(len(weights), p=weights)`
- **Performance:** O(1)
- **Test cases:**
  - TC1: weights=[0.5, 0.3, 0.2] → operator 0 được chọn ~50% (statistically)
  - TC2: weights=[1, 0, 0] → luôn chọn operator 0

---

### update_scores

- **Idea:** Cập nhật score cho operator vừa dùng dựa trên kết quả.
- **Input:**
  - `scores: ndarray`, `counts: ndarray`
  - `op_idx: int` — operator vừa dùng
  - `result: int` — 0=worse, 1=accepted, 2=new_best, 3=new_global_best
  - `score_values: tuple` — (sigma_1, sigma_2, sigma_3) default (1, 2, 5)
- **Logic:**
  ```
  scores[op_idx] += score_values[result] if result > 0 else 0
  counts[op_idx] += 1
  ```
- **Test cases:**
  - TC1: result=3 (new global best) → score += 5
  - TC2: result=0 (worse not accepted) → score += 0, count += 1

---

### update_weights

- **Idea:** Cuối mỗi segment, cập nhật weights dựa trên performance (score/count).
- **Input:**
  - `weights, scores, counts: ndarray`
  - `reaction_factor: float` — default 0.1 (0=giữ nguyên, 1=theo performance hoàn toàn)
- **Output:** None (modify in-place)
- **Logic:**
  ```
  for i in range(len(weights)):
      if counts[i] > 0:
          performance = scores[i] / counts[i]
          weights[i] = (1 - reaction_factor) * weights[i] + reaction_factor * performance

  # Normalize + minimum weight (tránh operator bị loại hoàn toàn)
  weights = maximum(weights, 0.01)
  weights /= weights.sum()

  # Reset scores, counts
  scores[:] = 0
  counts[:] = 0
  ```
- **Test cases:**
  - TC1: Operator hiệu quả → weight tăng
  - TC2: Operator tệ → weight giảm nhưng >= 0.01
  - TC3: Sum weights = 1 sau update
  - TC4: reaction_factor=0 → weights không đổi

---

## Internal Helpers

```python
def _sort_by_pos_desc(customers_list, sol):
    """Sort customers theo route position GIẢM DẦN.
    Quan trọng: remove từ cuối route trước → tránh index shift."""
    infos = [(c, *get_customer_info(sol, c)) for c in customers_list]
    infos.sort(key=lambda x: (x[1], x[2], -x[3]))  # vtype, vid, -pos
    # Group by (vtype, vid), sort pos desc within group
    result = []
    from itertools import groupby
    for key, group in groupby(infos, key=lambda x: (x[1], x[2])):
        group_list = sorted(group, key=lambda x: -x[3])
        result.extend([g[0] for g in group_list])
    return result

def _force_insert(sol, customer, dist_matrix, customers):
    """Insert customer bất kỳ đâu (infeasible OK). Dùng khi không còn chỗ feasible."""
    # Tìm route ngắn nhất → chèn cuối
    ...

def _greedy_insert_single(sol, customer, customers, restricted, dist_matrix):
    """Greedy insert 1 customer vào vị trí rẻ nhất across cả truck + bike."""
    ...

def _ensure_satellite_for_bike(sol, bike_id, customer, ...):
    """Đảm bảo bike route có RELOAD stop khi cần."""
    ...

def _replace_reload_node(sol, old_node, new_node):
    """Thay tất cả RELOAD stops tại old_node → new_node trong routes."""
    ...

def _remove_all_reload_at_node(sol, node, dist_matrix, customers):
    """Xoá tất cả RELOAD stops tại node trong routes."""
    ...

def _two_opt_delta(sol, vtype, vid, i, j, dist_matrix):
    """Tính delta distance nếu reverse [i,j]. O(1)."""
    ...

def _or_opt_delta(sol, vtype, vid, i, seg_len, j, dist_matrix):
    """Tính delta distance nếu move segment. O(1)."""
    ...

def _estimate_transfer_kg(sol, bike_id, reload_pos, customers):
    """Ước tính kg cần transfer tại reload position."""
    ...

def _estimate_time(sol, vehicle_id, pos, dist_matrix, customers):
    """Ước tính thời gian arrive tại position."""
    ...
```

---

## Tổng kết Phase 04

| File | Function | Loại | Mục đích |
|------|----------|------|----------|
| **destroy.py** | | | |
| | random_removal | Destroy | Loại q random customers |
| | worst_cost_removal | Destroy | Loại q đắt nhất |
| | shaw_removal | Destroy | Loại q tương tự nhau |
| | route_removal | Destroy | Loại cả 1 route |
| | satellite_removal | Destroy | Loại 1 satellite + bikes phụ thuộc |
| | zone_removal | Destroy | Loại vùng địa lý |
| **repair.py** | | | |
| | greedy_insertion | Repair | Chèn vào chỗ rẻ nhất |
| | regret_k_insertion | Repair | Ưu tiên customer khó chèn |
| | satellite_aware_insertion | Repair | Ưu tiên chèn gần satellite |
| **crosslayer.py** | | | |
| | swap_assignment | Cross | Chuyển truck ↔ bike |
| | relocate_satellite | Cross | Di chuyển satellite node |
| | create_new_satellite | Cross | Tạo satellite mới |
| | remove_satellite_node | Cross | Xoá satellite |
| **local_search.py** | | | |
| | two_opt | LS | Đảo segment trong route |
| | or_opt | LS | Di chuyển segment trong route |
| | relocate_inter_route | LS | Move customer cross-route |
| **weights.py** | | | |
| | init_weights | Mgmt | Khởi tạo uniform weights |
| | select_operator | Mgmt | Roulette wheel selection |
| | update_scores | Mgmt | Cập nhật score sau dùng operator |
| | update_weights | Mgmt | Cập nhật weights cuối segment |

**Tổng: 20 functions** (6 destroy + 3 repair + 4 cross-layer + 3 local search + 4 weight management)
