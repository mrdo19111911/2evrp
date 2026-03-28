# Phase 01b — Solution Operations (Speed-Critical)

## Tại sao cần phase này?

ALNS loop gọi insert/remove/swap hàng **triệu lần**. Nếu mỗi lần thao tác chậm → solver chậm. Phase này định nghĩa:

1. **Solution structure** rõ ràng — "solution là gì, chứa gì"
2. **Manipulation functions** — insert, remove, swap, create, delete routes
3. **Delta evaluation** — tính nhanh thay đổi cost mà KHÔNG cần re-simulate toàn bộ
4. **Query functions** — tra cứu nhanh thông tin solution

Tất cả phải **O(1) hoặc O(route_len)**, không bao giờ O(N) full re-evaluate.

## File structure

```
src/
  solution/
    structure.py      ← solution structure definition + creation
    route_ops.py      ← thao tác trên 1 route (insert, remove, swap stops)
    solution_ops.py   ← thao tác cross-route (move, transfer, create/delete route)
    satellite_ops.py  ← thao tác trên satellite events
    delta.py          ← delta cost evaluation (speed-critical)
    query.py          ← tra cứu nhanh: ai ở đâu, route nào, assigned chưa
```

---

## structure.py

### Solution Structure

Không phải class. Solution = 1 dict chứa tất cả arrays + precomputed cache.

```python
def create_solution(n_trucks, n_bikes, max_route_len=100):
    """
    Returns dict:
    {
        # === Core arrays ===
        "truck_stops":    ndarray int32 (T, L),   # -1 = padding
        "truck_actions":  ndarray int8  (T, L),   # ACT_DELIVER/RELOAD/PAD
        "bike_stops":     ndarray int32 (B, L),
        "bike_actions":   ndarray int8  (B, L),
        "satellites":     ndarray float64 (S, 5), # dynamic size

        # === Route metadata (precomputed, updated incrementally) ===
        "truck_lengths":  ndarray int32 (T,),     # actual stops per route (non -1)
        "bike_lengths":   ndarray int32 (B,),
        "truck_loads":    ndarray float64 (T,),   # total demand assigned to truck
        "bike_loads":     ndarray float64 (B,),   # total demand assigned to bike

        # === Customer index (O(1) lookup: customer → ai phục vụ?) ===
        "cust_vehicle":   ndarray int32 (N,),     # vehicle_id, -1 = unassigned
        "cust_vtype":     ndarray int8  (N,),     # VEH_TRUCK / VEH_BIKE / -1
        "cust_route_pos": ndarray int32 (N,),     # position in route, -1 = unassigned

        # === Precomputed route costs (updated incrementally) ===
        "truck_distances": ndarray float64 (T,),  # total distance per truck route
        "bike_distances":  ndarray float64 (B,),

        # === Config ===
        "n_trucks": int,
        "n_bikes": int,
        "max_route_len": int,
    }
    """
```

- **Idea:** Dict thay vì class. Truyền `sol` vào mọi function. Customer index cho O(1) lookup "customer X đang ở route nào, vị trí nào".
- **Test cases:**
  - TC1: create_solution(3, 5) → tất cả arrays đúng shape, fills = -1, lengths = 0
  - TC2: Truy cập sol["truck_stops"] hoạt động bình thường

---

### copy_solution

- **Idea:** Deep copy toàn bộ solution dict. ALNS cần copy trước khi thử modify.
- **Input:** `sol: dict`
- **Output:** `new_sol: dict` — bản copy độc lập
- **Logic:** `{k: v.copy() if isinstance(v, ndarray) else v for k, v in sol.items()}`
- **Performance:** ~0.01ms cho N=1000 (chỉ copy metadata arrays, không copy customer data)
- **Test cases:**
  - TC1: Modify copy → original không đổi
  - TC2: sol["cust_vehicle"] in copy vs original → different memory

---

### rebuild_index

- **Idea:** Rebuild customer index từ routes. Gọi khi index bị out-of-sync (debug/safety).
- **Input:** `sol: dict`, `n_customers: int`
- **Output:** None (modify sol in-place)
- **Logic:**
  ```
  sol["cust_vehicle"][:] = -1
  sol["cust_vtype"][:] = -1
  sol["cust_route_pos"][:] = -1

  for t in range(n_trucks):
      for i in range(sol["truck_lengths"][t]):
          c = sol["truck_stops"][t, i]
          if sol["truck_actions"][t, i] == ACT_DELIVER:
              sol["cust_vehicle"][c] = t
              sol["cust_vtype"][c] = VEH_TRUCK
              sol["cust_route_pos"][c] = i

  # tương tự cho bikes (vehicle_id offset: n_trucks + b)
  for b in range(n_bikes):
      for i in range(sol["bike_lengths"][b]):
          c = sol["bike_stops"][b, i]
          if sol["bike_actions"][b, i] == ACT_DELIVER:
              sol["cust_vehicle"][c] = n_trucks + b
              sol["cust_vtype"][c] = VEH_BIKE
              sol["cust_route_pos"][c] = i
  ```
- **Test cases:**
  - TC1: Sau insert/remove nhiều lần → rebuild_index → index khớp với routes
  - TC2: Solution rỗng → tất cả = -1

---

## route_ops.py

### insert_stop

- **Idea:** Chèn 1 stop vào route tại vị trí pos. Shift phần sau sang phải. Cập nhật index + metadata.
- **Input:**
  - `sol: dict`
  - `vtype: int` — VEH_TRUCK / VEH_BIKE
  - `vid: int` — vehicle index (0-based trong loại xe đó)
  - `pos: int` — vị trí chèn
  - `customer: int`
  - `action: int` — ACT_DELIVER / ACT_RELOAD
  - `dist_matrix: ndarray float64 (N+1, N+1)` — để update distance cache
  - `customers: ndarray float64 (N, F)` — để update load cache
- **Output:** None (modify sol in-place)
- **Logic:**
  ```
  stops, actions, lengths = _get_route_arrays(sol, vtype, vid)
  L = lengths[vid]
  assert L < sol["max_route_len"]

  # Shift right
  stops[vid, pos+1:L+1] = stops[vid, pos:L]
  actions[vid, pos+1:L+1] = actions[vid, pos:L]

  # Insert
  stops[vid, pos] = customer
  actions[vid, pos] = action
  lengths[vid] += 1

  # Update customer index (nếu DELIVER)
  if action == ACT_DELIVER:
      sol["cust_vehicle"][customer] = _global_vid(vtype, vid, sol["n_trucks"])
      sol["cust_vtype"][customer] = vtype
      sol["cust_route_pos"][customer] = pos
      # Shift route_pos cho customers SAU pos
      for j in range(pos + 1, lengths[vid]):
          c = stops[vid, j]
          if actions[vid, j] == ACT_DELIVER:
              sol["cust_route_pos"][c] = j

  # Update distance cache (incremental)
  _update_route_distance(sol, vtype, vid, dist_matrix)

  # Update load cache
  if action == ACT_DELIVER:
      demand = customers[customer, COL_DEMAND]
      _get_loads(sol, vtype)[vid] += demand
  ```
- **Performance:** O(route_len) — shift + index update
- **Test cases:**
  - TC1: Insert vào route rỗng pos=0 → length=1, customer ở pos 0
  - TC2: Insert ở giữa → shift đúng, length tăng 1
  - TC3: Insert DELIVER → cust_vehicle, cust_route_pos cập nhật đúng
  - TC4: Insert RELOAD → cust_vehicle KHÔNG thay đổi
  - TC5: Insert vào route đầy → assert error
  - TC6: Customers sau pos → route_pos tăng 1

---

### remove_stop

- **Idea:** Xoá stop ở vị trí pos. Shift trái. Cập nhật index + metadata.
- **Input:**
  - `sol: dict`
  - `vtype: int`, `vid: int`, `pos: int`
  - `dist_matrix: ndarray`, `customers: ndarray`
- **Output:**
  - `removed_customer: int`
  - `removed_action: int`
- **Logic:**
  ```
  stops, actions, lengths = _get_route_arrays(sol, vtype, vid)
  L = lengths[vid]
  removed_customer = stops[vid, pos]
  removed_action = actions[vid, pos]

  # Clear index nếu DELIVER
  if removed_action == ACT_DELIVER:
      sol["cust_vehicle"][removed_customer] = -1
      sol["cust_vtype"][removed_customer] = -1
      sol["cust_route_pos"][removed_customer] = -1

  # Shift left
  stops[vid, pos:L-1] = stops[vid, pos+1:L]
  actions[vid, pos:L-1] = actions[vid, pos+1:L]
  stops[vid, L-1] = -1
  actions[vid, L-1] = ACT_PAD
  lengths[vid] -= 1

  # Update route_pos cho customers SAU pos
  for j in range(pos, lengths[vid]):
      c = stops[vid, j]
      if actions[vid, j] == ACT_DELIVER:
          sol["cust_route_pos"][c] = j

  # Update distance cache
  _update_route_distance(sol, vtype, vid, dist_matrix)

  # Update load cache
  if removed_action == ACT_DELIVER:
      demand = customers[removed_customer, COL_DEMAND]
      _get_loads(sol, vtype)[vid] -= demand

  return removed_customer, removed_action
  ```
- **Performance:** O(route_len)
- **Test cases:**
  - TC1: Remove từ route 1 stop → route rỗng, length=0
  - TC2: Remove ở giữa → shift đúng, length giảm 1
  - TC3: Remove DELIVER → cust_vehicle = -1
  - TC4: Remove RELOAD → cust_vehicle không đổi
  - TC5: Remove từ route rỗng → error
  - TC6: Customers sau pos → route_pos giảm 1

---

### swap_stops_within

- **Idea:** Đổi chỗ 2 stops trong cùng 1 route. Nhanh hơn remove+insert.
- **Input:**
  - `sol: dict`, `vtype: int`, `vid: int`
  - `pos_a: int`, `pos_b: int`
  - `dist_matrix: ndarray`
- **Output:** None (in-place)
- **Logic:**
  ```
  stops, actions, _ = _get_route_arrays(sol, vtype, vid)

  # Swap stops
  stops[vid, pos_a], stops[vid, pos_b] = stops[vid, pos_b], stops[vid, pos_a]
  actions[vid, pos_a], actions[vid, pos_b] = actions[vid, pos_b], actions[vid, pos_a]

  # Update route_pos index
  ca, cb = stops[vid, pos_a], stops[vid, pos_b]
  if actions[vid, pos_a] == ACT_DELIVER:
      sol["cust_route_pos"][ca] = pos_a
  if actions[vid, pos_b] == ACT_DELIVER:
      sol["cust_route_pos"][cb] = pos_b

  _update_route_distance(sol, vtype, vid, dist_matrix)
  ```
- **Performance:** O(1) swap + O(route_len) distance update
- **Test cases:**
  - TC1: Swap pos 0 và 2 trong route [A, B, C] → [C, B, A]
  - TC2: Swap cùng vị trí → không đổi
  - TC3: Route_pos index cập nhật đúng

---

### reverse_segment

- **Idea:** Đảo ngược đoạn [start, end] trong route. Dùng cho 2-opt.
- **Input:**
  - `sol: dict`, `vtype: int`, `vid: int`
  - `start: int`, `end: int` — inclusive
  - `dist_matrix: ndarray`
- **Output:** None (in-place)
- **Logic:**
  ```
  stops, actions, _ = _get_route_arrays(sol, vtype, vid)

  # Reverse segment
  stops[vid, start:end+1] = stops[vid, start:end+1][::-1]
  actions[vid, start:end+1] = actions[vid, start:end+1][::-1]

  # Update route_pos index cho tất cả DELIVER trong segment
  for j in range(start, end + 1):
      c = stops[vid, j]
      if actions[vid, j] == ACT_DELIVER:
          sol["cust_route_pos"][c] = j

  _update_route_distance(sol, vtype, vid, dist_matrix)
  ```
- **Performance:** O(segment_len)
- **Test cases:**
  - TC1: Route [A,B,C,D,E], reverse [1,3] → [A,D,C,B,E]
  - TC2: Reverse toàn route → OK
  - TC3: Reverse 1 phần tử → không đổi

---

## solution_ops.py

### move_stop

- **Idea:** Di chuyển 1 stop từ route này sang route khác (có thể khác loại xe). = remove + insert nhưng atomic.
- **Input:**
  - `sol: dict`
  - `from_vtype, from_vid, from_pos: int`
  - `to_vtype, to_vid, to_pos: int`
  - `dist_matrix, customers: ndarray`
- **Output:** None (in-place)
- **Logic:**
  ```
  cust, action = remove_stop(sol, from_vtype, from_vid, from_pos, dist_matrix, customers)
  insert_stop(sol, to_vtype, to_vid, to_pos, cust, action, dist_matrix, customers)
  ```
- **Performance:** O(route_len) × 2
- **Test cases:**
  - TC1: Move customer từ truck route sang bike route
  - TC2: Move trong cùng route nhưng vị trí khác
  - TC3: Move RELOAD stop → satellite index cần update (riêng)
  - TC4: Sau move, customer index chỉ đúng về vehicle mới

---

### swap_stops_between

- **Idea:** Đổi chỗ 2 stops ở 2 routes khác nhau. = 2 remove + 2 insert nhưng atomic.
- **Input:**
  - `sol: dict`
  - `vtype_a, vid_a, pos_a: int`
  - `vtype_b, vid_b, pos_b: int`
  - `dist_matrix, customers: ndarray`
- **Output:** None (in-place)
- **Logic:**
  ```
  cust_a, act_a = remove_stop(sol, vtype_a, vid_a, pos_a, ...)
  cust_b, act_b = remove_stop(sol, vtype_b, vid_b, pos_b, ...)
  # pos có thể shift do remove → điều chỉnh
  insert_stop(sol, vtype_a, vid_a, pos_a, cust_b, act_b, ...)
  insert_stop(sol, vtype_b, vid_b, pos_b, cust_a, act_a, ...)
  ```
- **Performance:** O(route_len) × 4
- **Test cases:**
  - TC1: Swap giữa truck và bike → actions cũng swap
  - TC2: Swap giữa 2 trucks → OK
  - TC3: Customer index cập nhật đúng cho cả 2

---

### clear_route

- **Idea:** Xoá toàn bộ stops trong 1 route. Trả về danh sách customers đã remove.
- **Input:**
  - `sol: dict`, `vtype: int`, `vid: int`
- **Output:**
  - `removed: list[tuple(int, int)]` — [(customer, action), ...]
- **Logic:**
  ```
  stops, actions, lengths = _get_route_arrays(sol, vtype, vid)
  L = lengths[vid]
  removed = []

  for i in range(L):
      c = stops[vid, i]
      a = actions[vid, i]
      removed.append((c, a))
      if a == ACT_DELIVER:
          sol["cust_vehicle"][c] = -1
          sol["cust_vtype"][c] = -1
          sol["cust_route_pos"][c] = -1

  stops[vid, :] = -1
  actions[vid, :] = ACT_PAD
  lengths[vid] = 0
  _get_loads(sol, vtype)[vid] = 0.0
  _get_distances(sol, vtype)[vid] = 0.0

  return removed
  ```
- **Performance:** O(route_len)
- **Test cases:**
  - TC1: Clear route 5 stops → length=0, returns 5 items
  - TC2: Clear route rỗng → returns []
  - TC3: Sau clear, customers unassigned

---

### create_route_from_list

- **Idea:** Tạo route mới từ danh sách (customer, action) pairs. Dùng khi repair operator muốn build route nhanh.
- **Input:**
  - `sol: dict`, `vtype: int`, `vid: int`
  - `stop_list: list[tuple(int, int)]` — [(customer, action), ...]
  - `dist_matrix, customers: ndarray`
- **Output:** None (in-place, ghi đè route vid)
- **Logic:**
  ```
  # Clear existing
  clear_route(sol, vtype, vid)

  stops, actions, lengths = _get_route_arrays(sol, vtype, vid)
  for i, (c, a) in enumerate(stop_list):
      stops[vid, i] = c
      actions[vid, i] = a
      if a == ACT_DELIVER:
          sol["cust_vehicle"][c] = _global_vid(vtype, vid, sol["n_trucks"])
          sol["cust_vtype"][c] = vtype
          sol["cust_route_pos"][c] = i

  lengths[vid] = len(stop_list)
  _update_route_distance(sol, vtype, vid, dist_matrix)
  _update_route_load(sol, vtype, vid, customers)
  ```
- **Performance:** O(len(stop_list))
- **Test cases:**
  - TC1: Tạo route 3 stops → length=3, stops đúng, index đúng
  - TC2: Tạo route rỗng → length=0
  - TC3: Ghi đè route có sẵn → route cũ bị clear trước

---

## satellite_ops.py

### add_satellite_event

- **Idea:** Thêm 1 reload event vào bảng satellites.
- **Input:**
  - `sol: dict`
  - `customer_idx, bike_id, truck_id: int`
  - `transfer_kg, planned_time: float`
- **Output:** None (modify sol in-place, grow satellites array)
- **Logic:**
  ```
  new_row = array([[customer_idx, bike_id, truck_id, transfer_kg, planned_time]])
  sol["satellites"] = vstack([sol["satellites"], new_row])
  ```
- **Performance:** O(S) do vstack copy — xem note tối ưu bên dưới
- **Test cases:**
  - TC1: Add vào empty satellites → shape (1, 5)
  - TC2: Add 3 lần → shape (3, 5)
  - TC3: Giá trị lưu đúng

---

### remove_satellite_event

- **Idea:** Xoá reload event theo index hoặc theo (customer, bike, truck) match.
- **Input:**
  - `sol: dict`
  - `sat_idx: int` — index trong satellites array, HOẶC
  - `customer_idx, bike_id, truck_id: int` — match criteria (nếu sat_idx = -1)
- **Output:**
  - `removed: ndarray float64 (5,)` — event đã xoá
- **Logic:**
  ```
  if sat_idx >= 0:
      removed = sol["satellites"][sat_idx].copy()
      sol["satellites"] = delete(sol["satellites"], sat_idx, axis=0)
  else:
      mask = (sol["satellites"][:, SAT_CUST] == customer_idx) & \
             (sol["satellites"][:, SAT_BIKE] == bike_id) & \
             (sol["satellites"][:, SAT_TRUCK] == truck_id)
      idx = where(mask)[0]
      if len(idx) > 0:
          removed = sol["satellites"][idx[0]].copy()
          sol["satellites"] = delete(sol["satellites"], idx[0], axis=0)
  ```
- **Test cases:**
  - TC1: Remove by index → satellites shrinks by 1
  - TC2: Remove by match → đúng event bị xoá
  - TC3: Remove from empty → no crash, return None

---

### remove_satellites_for_customer

- **Idea:** Xoá TẤT CẢ reload events liên quan tới 1 customer node. Dùng khi customer bị remove khỏi route.
- **Input:**
  - `sol: dict`, `customer_idx: int`
- **Output:**
  - `removed_count: int`
- **Logic:**
  ```
  mask = sol["satellites"][:, SAT_CUST] != customer_idx
  removed_count = sum(~mask)
  sol["satellites"] = sol["satellites"][mask]
  ```
- **Test cases:**
  - TC1: 3 events tại customer 7 → xoá cả 3, count=3
  - TC2: Không event nào tại customer → count=0, array không đổi

---

### remove_satellites_for_bike

- **Idea:** Xoá tất cả reload events của 1 bike. Dùng khi clear bike route.
- **Input:** `sol: dict`, `bike_id: int`
- **Output:** `removed_count: int`
- **Logic:** Tương tự, filter by SAT_BIKE
- **Test cases:**
  - TC1: Bike có 2 reload events → xoá cả 2

---

### update_satellite_time

- **Idea:** Cập nhật planned_time cho 1 satellite event. Dùng sau khi route thay đổi → timing shift.
- **Input:**
  - `sol: dict`, `sat_idx: int`, `new_time: float`
- **Output:** None
- **Logic:** `sol["satellites"][sat_idx, SAT_TIME] = new_time`
- **Test cases:**
  - TC1: Update time → verify changed

---

## delta.py (Speed-Critical)

### Ý tưởng chung

Thay vì evaluate toàn bộ solution mỗi lần thay đổi 1 stop, ta tính **delta** — sự thay đổi cost. O(1) thay vì O(total_stops).

Distance delta chỉ phụ thuộc vào 3 node: trước, tại, sau.

```
Route: ... → A → B → C → ...
Remove B:  delta = dist(A,C) - dist(A,B) - dist(B,C)
Insert X between A and C:  delta = dist(A,X) + dist(X,C) - dist(A,C)
```

---

### insertion_cost_delta

- **Idea:** Tính delta distance nếu chèn customer X vào vị trí pos trong route. KHÔNG modify solution.
- **Input:**
  - `sol: dict`
  - `vtype: int`, `vid: int`, `pos: int`
  - `customer: int`
  - `dist_matrix: ndarray float64 (N+1, N+1)`
- **Output:**
  - `delta_dist: float` — thay đổi distance (dương = tăng, âm = giảm)
- **Logic:**
  ```
  stops, _, lengths = _get_route_arrays(sol, vtype, vid)
  L = lengths[vid]
  cust_node = customer + 1

  if L == 0:
      # Route rỗng: depot → customer → depot
      return dist_matrix[0, cust_node] + dist_matrix[cust_node, 0]

  if pos == 0:
      # Chèn đầu: depot → X → old_first → ...
      old_first = stops[vid, 0] + 1
      delta = dist_matrix[0, cust_node] + dist_matrix[cust_node, old_first] \
              - dist_matrix[0, old_first]

  elif pos == L:
      # Chèn cuối: ... → old_last → X → depot
      old_last = stops[vid, L-1] + 1
      delta = dist_matrix[old_last, cust_node] + dist_matrix[cust_node, 0] \
              - dist_matrix[old_last, 0]

  else:
      # Chèn giữa A và B: ... → A → X → B → ...
      a = stops[vid, pos-1] + 1
      b = stops[vid, pos] + 1
      delta = dist_matrix[a, cust_node] + dist_matrix[cust_node, b] \
              - dist_matrix[a, b]

  return delta
  ```
- **Performance:** **O(1)** — 3 lookups trong dist_matrix
- **Test cases:**
  - TC1: Route rỗng → delta = dist(depot, X) + dist(X, depot)
  - TC2: Chèn giữa A, B gần nhau → delta nhỏ
  - TC3: Chèn giữa A, B xa nhau mà X nằm giữa → delta có thể âm (tiết kiệm)
  - TC4: Consistent với full re-evaluate: `old_total + delta == new_total`

---

### removal_cost_delta

- **Idea:** Tính delta distance nếu xoá stop ở pos. KHÔNG modify solution.
- **Input:**
  - `sol: dict`, `vtype: int`, `vid: int`, `pos: int`
  - `dist_matrix: ndarray`
- **Output:**
  - `delta_dist: float` — luôn <= 0 (xoá stop = tiết kiệm hoặc bằng)
- **Logic:**
  ```
  stops, _, lengths = _get_route_arrays(sol, vtype, vid)
  L = lengths[vid]
  cust_node = stops[vid, pos] + 1

  if L == 1:
      # Xoá stop cuối cùng: route thành rỗng
      return -(dist_matrix[0, cust_node] + dist_matrix[cust_node, 0])

  if pos == 0:
      next_node = stops[vid, 1] + 1
      delta = dist_matrix[0, next_node] \
              - dist_matrix[0, cust_node] - dist_matrix[cust_node, next_node]

  elif pos == L - 1:
      prev_node = stops[vid, L-2] + 1
      delta = dist_matrix[prev_node, 0] \
              - dist_matrix[prev_node, cust_node] - dist_matrix[cust_node, 0]

  else:
      prev_node = stops[vid, pos-1] + 1
      next_node = stops[vid, pos+1] + 1
      delta = dist_matrix[prev_node, next_node] \
              - dist_matrix[prev_node, cust_node] - dist_matrix[cust_node, next_node]

  return delta
  ```
- **Performance:** **O(1)**
- **Test cases:**
  - TC1: Route 1 stop → delta = -(route cost)
  - TC2: Remove middle node → delta = shortcut savings
  - TC3: Consistent: `old_total + delta == new_total`
  - TC4: delta luôn <= 0

---

### best_insertion_pos

- **Idea:** Tìm vị trí tốt nhất (delta nhỏ nhất) để chèn customer X vào 1 route.
- **Input:**
  - `sol: dict`, `vtype: int`, `vid: int`
  - `customer: int`
  - `dist_matrix: ndarray`
- **Output:**
  - `best_pos: int`
  - `best_delta: float`
- **Logic:**
  ```
  L = _get_lengths(sol, vtype)[vid]
  best_pos = 0
  best_delta = inf

  for pos in range(L + 1):
      delta = insertion_cost_delta(sol, vtype, vid, pos, customer, dist_matrix)
      if delta < best_delta:
          best_delta = delta
          best_pos = pos

  return best_pos, best_delta
  ```
- **Performance:** O(route_len) — gọi O(1) delta cho mỗi position
- **Test cases:**
  - TC1: Route rỗng → pos=0
  - TC2: Customer nằm trên đường A→B → pos giữa A và B, delta nhỏ
  - TC3: Customer xa mọi node → delta lớn ở mọi pos

---

### find_best_insertion_all_routes

- **Idea:** Tìm route + position tốt nhất để chèn customer, xét TẤT CẢ routes thuộc 1 loại xe.
- **Input:**
  - `sol: dict`, `vtype: int`
  - `customer: int`
  - `dist_matrix: ndarray`
  - `customers: ndarray` — để check capacity feasibility
  - `vehicle_capacity: float`
- **Output:**
  - `best_vid: int`, `best_pos: int`, `best_delta: float`
  - Returns (-1, -1, inf) nếu không vị trí nào feasible
- **Logic:**
  ```
  best = (-1, -1, inf)
  demand = customers[customer, COL_DEMAND]

  n_vehicles = sol["n_trucks"] if vtype == VEH_TRUCK else sol["n_bikes"]
  loads = _get_loads(sol, vtype)

  for vid in range(n_vehicles):
      if loads[vid] + demand > vehicle_capacity:
          continue  # không vừa
      pos, delta = best_insertion_pos(sol, vtype, vid, customer, dist_matrix)
      if delta < best[2]:
          best = (vid, pos, delta)

  return best
  ```
- **Performance:** O(n_vehicles × route_len)
- **Test cases:**
  - TC1: 1 route có chỗ → trả về route đó
  - TC2: Tất cả routes đầy → returns (-1, -1, inf)
  - TC3: Nhiều routes → chọn delta nhỏ nhất

---

## query.py

### get_unassigned_customers

- **Idea:** Danh sách customers chưa được assign (DELIVER) cho xe nào.
- **Input:** `sol: dict`, `n_customers: int`
- **Output:** `ndarray int32`
- **Logic:** `where(sol["cust_vehicle"] == -1)[0]`
- **Performance:** O(N) — nhưng vectorized
- **Test cases:**
  - TC1: Solution rỗng → all customers
  - TC2: Solution hoàn chỉnh → empty array

---

### get_assigned_customers

- **Idea:** Danh sách customers đã assign.
- **Input:** `sol: dict`, `n_customers: int`
- **Output:** `ndarray int32`
- **Logic:** `where(sol["cust_vehicle"] >= 0)[0]`

---

### get_route_as_list

- **Idea:** Trích route thành list[(customer, action)] — dễ đọc, dễ debug.
- **Input:** `sol: dict`, `vtype: int`, `vid: int`
- **Output:** `list[tuple(int, int)]`
- **Logic:**
  ```
  stops, actions, lengths = _get_route_arrays(sol, vtype, vid)
  L = lengths[vid]
  return [(stops[vid, i], actions[vid, i]) for i in range(L)]
  ```
- **Test cases:**
  - TC1: Route [3,7,12] → [(3, ACT_DELIVER), (7, ACT_RELOAD), (12, ACT_DELIVER)]
  - TC2: Route rỗng → []

---

### get_route_customers_only

- **Idea:** Chỉ trả DELIVER customers trong 1 route (bỏ RELOAD stops).
- **Input:** `sol: dict`, `vtype: int`, `vid: int`
- **Output:** `ndarray int32`
- **Logic:**
  ```
  stops, actions, lengths = _get_route_arrays(sol, vtype, vid)
  L = lengths[vid]
  mask = actions[vid, :L] == ACT_DELIVER
  return stops[vid, :L][mask]
  ```

---

### is_customer_assigned

- **Idea:** O(1) check: customer X đã assign chưa?
- **Input:** `sol: dict`, `customer: int`
- **Output:** `bool`
- **Logic:** `sol["cust_vehicle"][customer] >= 0`
- **Performance:** **O(1)**

---

### get_customer_info

- **Idea:** O(1) lookup: customer X ở route nào, vị trí nào, loại xe nào?
- **Input:** `sol: dict`, `customer: int`
- **Output:** `tuple(vtype: int, vid: int, pos: int)` — hoặc (-1, -1, -1) nếu unassigned
- **Logic:**
  ```
  veh = sol["cust_vehicle"][customer]
  if veh == -1:
      return (-1, -1, -1)
  vtype = sol["cust_vtype"][customer]
  pos = sol["cust_route_pos"][customer]
  vid = veh if vtype == VEH_TRUCK else veh - sol["n_trucks"]
  return (vtype, vid, pos)
  ```
- **Performance:** **O(1)**
- **Test cases:**
  - TC1: Customer in truck route 2, pos 3 → (VEH_TRUCK, 2, 3)
  - TC2: Unassigned → (-1, -1, -1)

---

## Internal Helpers

```python
def _get_route_arrays(sol, vtype, vid):
    """Return (stops_2d, actions_2d, lengths_1d) cho đúng loại xe."""
    if vtype == VEH_TRUCK:
        return sol["truck_stops"], sol["truck_actions"], sol["truck_lengths"]
    else:
        return sol["bike_stops"], sol["bike_actions"], sol["bike_lengths"]

def _get_loads(sol, vtype):
    return sol["truck_loads"] if vtype == VEH_TRUCK else sol["bike_loads"]

def _get_distances(sol, vtype):
    return sol["truck_distances"] if vtype == VEH_TRUCK else sol["bike_distances"]

def _global_vid(vtype, vid, n_trucks):
    """Vehicle ID toàn cục: trucks = 0..T-1, bikes = T..T+B-1."""
    return vid if vtype == VEH_TRUCK else n_trucks + vid

def _update_route_distance(sol, vtype, vid, dist_matrix):
    """Recompute total distance cho 1 route. O(route_len)."""
    stops, _, lengths = _get_route_arrays(sol, vtype, vid)
    L = lengths[vid]
    if L == 0:
        _get_distances(sol, vtype)[vid] = 0.0
        return
    total = dist_matrix[0, stops[vid, 0] + 1]  # depot → first
    for i in range(L - 1):
        total += dist_matrix[stops[vid, i] + 1, stops[vid, i+1] + 1]
    total += dist_matrix[stops[vid, L-1] + 1, 0]  # last → depot
    _get_distances(sol, vtype)[vid] = total

def _update_route_load(sol, vtype, vid, customers):
    """Recompute total demand cho 1 route. O(route_len)."""
    stops, actions, lengths = _get_route_arrays(sol, vtype, vid)
    L = lengths[vid]
    total = 0.0
    for i in range(L):
        if actions[vid, i] == ACT_DELIVER:
            total += customers[stops[vid, i], COL_DEMAND]
    _get_loads(sol, vtype)[vid] = total
```

---

## cost.py — Chi phí vận hành thực tế (Việt Nam 2025-2026)

### Cost Model

Dữ liệu thu thập từ thực tế thị trường Việt Nam:

```python
# === XE TẢI 2 TẤN ===
TRUCK_FUEL_CONSUMPTION = 12.0    # lít dầu / 100 km (đô thị, có tải)
TRUCK_FUEL_PRICE       = 21000   # VND / lít (diesel 2025)
TRUCK_FUEL_COST_KM     = 2520    # VND / km (= 12 * 21000 / 100)
TRUCK_MAINTENANCE_KM   = 800     # VND / km (bảo trì, lốp, dầu máy)
TRUCK_DEPRECIATION_KM  = 1200    # VND / km (khấu hao xe ~500M / 400k km)
TRUCK_DRIVER_HOUR      = 37500   # VND / giờ (~6M/tháng / 160h)
TRUCK_FIXED_DAY        = 200000  # VND / ngày (bảo hiểm, phí đường bộ, đậu xe)
TRUCK_TOTAL_KM         = 4520    # VND / km (fuel + maintenance + depreciation)
TRUCK_SPEED_URBAN      = 25.0    # km/h (đô thị, giao hàng)
TRUCK_SPEED_SUBURBAN   = 40.0    # km/h (ngoại thành)
TRUCK_CAPACITY         = 2000.0  # kg

# === XE MÁY GIAO HÀNG ===
BIKE_FUEL_CONSUMPTION  = 2.0     # lít xăng / 100 km
BIKE_FUEL_PRICE        = 25000   # VND / lít (RON 95, 2025)
BIKE_FUEL_COST_KM      = 500     # VND / km (= 2.0 * 25000 / 100)
BIKE_MAINTENANCE_KM    = 200     # VND / km
BIKE_DEPRECIATION_KM   = 150     # VND / km (~30M / 200k km)
BIKE_DRIVER_HOUR       = 25000   # VND / giờ (~4M/tháng / 160h)
BIKE_FIXED_DAY         = 50000   # VND / ngày (bảo hiểm)
BIKE_TOTAL_KM          = 850     # VND / km (fuel + maintenance + depreciation)
BIKE_SPEED_URBAN       = 20.0    # km/h (đô thị, giao hàng, vào hẻm)
BIKE_SPEED_SUBURBAN    = 35.0    # km/h
BIKE_CAPACITY          = 60.0    # kg

# === CHI PHÍ RELOAD ===
RELOAD_SERVICE_TIME    = 5.0     # phút / lần transfer
RELOAD_HANDLING_COST   = 5000    # VND / lần (công bốc dỡ)

# === TIME ===
DAY_START              = 0.0     # phút (7:00 AM → minute 0)
DAY_LENGTH             = 480.0   # phút (8 giờ làm việc)
SYNC_DELTA_T           = 15.0    # phút (tolerance đồng bộ)

# === PENALTY (dùng trong fitness, đơn vị VND) ===
PENALTY_UNSERVED       = 500000  # VND / customer bỏ sót
PENALTY_LATE           = 10000   # VND / phút trễ deadline
PENALTY_OVERLOAD       = 50000   # VND / kg vượt tải
PENALTY_SYNC_FAIL      = 100000  # VND / lần sync thất bại
PENALTY_RESTRICTION    = 200000  # VND / lần vi phạm xe vào hẻm
```

Sources:
- Xe tải: [otophucuong.vn](https://otophucuong.vn/dinh-muc-tieu-hao-nguyen-lieu-xe-tai/), [vtruck.vn](https://vtruck.vn/xe-tai-2.5t-an-bao-nhieu-lit-dau)
- Xe máy: [ICCT 2025 Report](https://theicct.org/wp-content/uploads/2025/02/ID-250-%E2%80%93-Vietnam-two-wheelers_report_final.pdf), [rentabikevn.com](https://rentabikevn.com/petrol-in-vietnam-all-you-need-to-know/)
- Cước vận tải: [allship.vn](https://allship.vn/bang-gia-cuoc-van-chuyen-xe-tai-63-tinh-thanh/)

---

### compute_route_cost

- **Idea:** Tính TỔNG chi phí 1 route = distance cost + time cost + fixed cost + reload handling.
- **Input:**
  - `total_distance: float` — km
  - `total_time: float` — phút (từ xuất phát tới về kho)
  - `n_reloads: int` — số lần reload trong route
  - `vtype: int` — VEH_TRUCK / VEH_BIKE
- **Output:**
  - `cost: float` — VND
  - `cost_breakdown: dict` — {"distance": ..., "time": ..., "fixed": ..., "reload": ...}
- **Logic:**
  ```
  if vtype == VEH_TRUCK:
      dist_cost = total_distance * TRUCK_TOTAL_KM
      time_cost = (total_time / 60) * TRUCK_DRIVER_HOUR
      fixed = TRUCK_FIXED_DAY
  else:
      dist_cost = total_distance * BIKE_TOTAL_KM
      time_cost = (total_time / 60) * BIKE_DRIVER_HOUR
      fixed = BIKE_FIXED_DAY

  reload_cost = n_reloads * RELOAD_HANDLING_COST
  total = dist_cost + time_cost + fixed + reload_cost

  return total, {"distance": dist_cost, "time": time_cost,
                  "fixed": fixed, "reload": reload_cost}
  ```
- **Test cases:**
  - TC1: Truck chạy 50km, 3 giờ, 2 reloads → 50*4520 + 3*37500 + 200000 + 2*5000 = 548,500 VND
  - TC2: Bike chạy 30km, 4 giờ, 1 reload → 30*850 + 4*25000 + 50000 + 5000 = 180,500 VND
  - TC3: Route rỗng (0km, 0min) → chỉ fixed cost

---

### compute_solution_cost

- **Idea:** Tổng chi phí toàn bộ solution.
- **Input:**
  - `truck_distances: ndarray float64 (T,)`
  - `truck_times: ndarray float64 (T,)` — total time per truck
  - `bike_distances: ndarray float64 (B,)`
  - `bike_times: ndarray float64 (B,)`
  - `satellites: ndarray float64 (S, 5)`
- **Output:**
  - `total_cost: float` — VND
  - `truck_costs: ndarray float64 (T,)`
  - `bike_costs: ndarray float64 (B,)`
- **Test cases:**
  - TC1: Tổng = sum(truck_costs) + sum(bike_costs)
  - TC2: Không xe nào chạy → chỉ fixed costs cho xe được assign

---

### insertion_cost_delta_full

- **Idea:** Delta cost ĐẦY ĐỦ (không chỉ distance, mà cả time + reload). O(1) estimate cho ALNS.
- **Input:**
  - `sol: dict`, `vtype, vid, pos, customer: int`
  - `dist_matrix, customers: ndarray`
- **Output:**
  - `delta_cost: float` — VND
- **Logic:**
  ```
  delta_dist = insertion_cost_delta(sol, vtype, vid, pos, customer, dist_matrix)  # O(1)

  if vtype == VEH_TRUCK:
      cost_km = TRUCK_TOTAL_KM
      speed = TRUCK_SPEED_URBAN
      driver_min = TRUCK_DRIVER_HOUR / 60
  else:
      cost_km = BIKE_TOTAL_KM
      speed = BIKE_SPEED_URBAN
      driver_min = BIKE_DRIVER_HOUR / 60

  delta_time = (delta_dist / speed) * 60  # phút thêm travel
  delta_time += customers[customer, COL_SERVICE_TIME]  # service time

  delta_cost = delta_dist * cost_km + delta_time * driver_min
  return delta_cost
  ```
- **Performance:** **O(1)**
- **Test cases:**
  - TC1: Insert customer gần → delta nhỏ
  - TC2: Insert customer xa + service_time dài → delta lớn
  - TC3: Delta bao gồm cả distance cost lẫn time cost

---

## check.py — Quick Constraint Checks (cho ALNS)

Khác với validate.py ở Phase 02 (full validation sau simulate), đây là **checks nhanh** gọi TRƯỚC khi modify solution — để ALNS biết move có feasible không mà không cần re-simulate.

---

### can_insert_customer

- **Idea:** Check nhanh: có thể chèn customer vào route (vtype, vid) không? Kiểm tra capacity + vehicle restriction. KHÔNG check TW (cần simulate).
- **Input:**
  - `sol: dict`, `vtype: int`, `vid: int`
  - `customer: int`
  - `customers: ndarray`, `restricted: ndarray`
  - `vehicle_capacity: float`
- **Output:** `bool`
- **Logic:**
  ```
  # Check restriction
  if vtype == VEH_TRUCK and restricted[customer] == 1:
      return False  # hẻm, chỉ bike

  # Check capacity
  demand = customers[customer, COL_DEMAND]
  current_load = _get_loads(sol, vtype)[vid]
  if current_load + demand > vehicle_capacity:
      return False

  # Check route not full
  if _get_lengths(sol, vtype)[vid] >= sol["max_route_len"] - 1:
      return False

  return True
  ```
- **Performance:** **O(1)**
- **Test cases:**
  - TC1: Truck route có chỗ, customer không restricted → True
  - TC2: Customer restricted, vtype=truck → False
  - TC3: Demand + current_load > capacity → False
  - TC4: Route đầy → False

---

### can_move_customer

- **Idea:** Check nhanh: có thể move customer từ route A sang route B không?
- **Input:**
  - `sol: dict`
  - `customer: int`
  - `to_vtype, to_vid: int`
  - `customers, restricted: ndarray`
  - `vehicle_capacity: float`
- **Output:** `bool`
- **Logic:**
  ```
  # Customer phải đang assigned
  if not is_customer_assigned(sol, customer):
      return False

  # Check target route feasibility
  return can_insert_customer(sol, to_vtype, to_vid, customer,
                              customers, restricted, vehicle_capacity)
  ```
- **Performance:** **O(1)**
- **Test cases:**
  - TC1: Customer in truck, move to bike with space → True
  - TC2: Customer restricted, move to truck → False
  - TC3: Target route full → False

---

### can_swap_customers

- **Idea:** Check nhanh: có thể swap 2 customers giữa 2 routes không? Check capacity feasibility sau swap.
- **Input:**
  - `sol: dict`
  - `cust_a, cust_b: int`
  - `customers, restricted: ndarray`
- **Output:** `bool`
- **Logic:**
  ```
  vtype_a, vid_a, _ = get_customer_info(sol, cust_a)
  vtype_b, vid_b, _ = get_customer_info(sol, cust_b)

  demand_a = customers[cust_a, COL_DEMAND]
  demand_b = customers[cust_b, COL_DEMAND]

  # Route A: mất demand_a, nhận demand_b
  load_a = _get_loads(sol, vtype_a)[vid_a] - demand_a + demand_b
  cap_a = TRUCK_CAPACITY if vtype_a == VEH_TRUCK else BIKE_CAPACITY
  if load_a > cap_a:
      return False

  # Route B: mất demand_b, nhận demand_a
  load_b = _get_loads(sol, vtype_b)[vid_b] - demand_b + demand_a
  cap_b = TRUCK_CAPACITY if vtype_b == VEH_TRUCK else BIKE_CAPACITY
  if load_b > cap_b:
      return False

  # Restriction check
  if vtype_b == VEH_TRUCK and restricted[cust_a] == 1:
      return False
  if vtype_a == VEH_TRUCK and restricted[cust_b] == 1:
      return False

  return True
  ```
- **Performance:** **O(1)**
- **Test cases:**
  - TC1: Swap 2 customers cùng demand → True (load không đổi)
  - TC2: Swap demand lớn vào route nhỏ → False (overload)
  - TC3: Swap restricted customer vào truck → False

---

### check_route_capacity_quick

- **Idea:** Check tổng demand route <= capacity. Nhanh vì dùng cached load.
- **Input:** `sol: dict`, `vtype: int`, `vid: int`, `vehicle_capacity: float`
- **Output:** `bool`
- **Logic:** `_get_loads(sol, vtype)[vid] <= vehicle_capacity`
- **Performance:** **O(1)**

---

### check_all_assigned

- **Idea:** Check tất cả customers đã được assign chưa.
- **Input:** `sol: dict`, `n_customers: int`
- **Output:** `bool`
- **Logic:** `np.all(sol["cust_vehicle"] >= 0)`
- **Performance:** O(N) vectorized

---

## Performance Note: Satellites Array Growth

`vstack` copy toàn bộ array mỗi lần add — O(S). Nếu satellites lớn, dùng **pre-allocated buffer**:

```python
# Trong create_solution:
"satellites": empty((SAT_BUFFER_SIZE, 5), float64),  # pre-allocate 1000 rows
"sat_count": 0,  # actual count

# add: sol["satellites"][sol["sat_count"]] = new_row; sol["sat_count"] += 1
# remove: swap with last, decrement count
# grow: if sat_count == buffer_size → double buffer
```

Quyết định khi implement. Spec giữ interface đơn giản, tối ưu internal sau.

---

## Tổng kết Phase 01b

| File | Function | Performance | Mục đích |
|------|----------|-------------|----------|
| **structure.py** | | | |
| | create_solution | O(1) | Tạo solution rỗng với index |
| | copy_solution | O(N) | Deep copy |
| | rebuild_index | O(total_stops) | Rebuild customer index |
| **route_ops.py** | | | |
| | insert_stop | O(route_len) | Chèn stop + update index |
| | remove_stop | O(route_len) | Xoá stop + update index |
| | swap_stops_within | O(route_len) | Đổi 2 stops cùng route |
| | reverse_segment | O(segment_len) | Đảo ngược đoạn (2-opt) |
| **solution_ops.py** | | | |
| | move_stop | O(route_len) | Di chuyển stop cross-route |
| | swap_stops_between | O(route_len) | Đổi stops cross-route |
| | clear_route | O(route_len) | Xoá toàn bộ route |
| | create_route_from_list | O(list_len) | Build route từ list |
| **satellite_ops.py** | | | |
| | add_satellite_event | O(S) | Thêm reload event |
| | remove_satellite_event | O(S) | Xoá reload event |
| | remove_satellites_for_customer | O(S) | Xoá events theo customer |
| | remove_satellites_for_bike | O(S) | Xoá events theo bike |
| | update_satellite_time | O(1) | Cập nhật thời gian |
| **delta.py** | | | |
| | insertion_cost_delta | **O(1)** | Delta distance chèn |
| | removal_cost_delta | **O(1)** | Delta distance xoá |
| | insertion_cost_delta_full | **O(1)** | Delta cost đầy đủ (VND) |
| | best_insertion_pos | O(route_len) | Vị trí chèn tốt nhất 1 route |
| | find_best_insertion_all_routes | O(V × route_len) | Vị trí chèn tốt nhất mọi route |
| **cost.py** | | | |
| | (constants) | — | Chi phí thực tế VN 2025 |
| | compute_route_cost | O(1) | Tổng chi phí 1 route (VND) |
| | compute_solution_cost | O(T+B) | Tổng chi phí solution (VND) |
| **check.py** | | | |
| | can_insert_customer | **O(1)** | Feasibility check trước insert |
| | can_move_customer | **O(1)** | Feasibility check trước move |
| | can_swap_customers | **O(1)** | Feasibility check trước swap |
| | check_route_capacity_quick | **O(1)** | Capacity check nhanh |
| | check_all_assigned | O(N) | Tất cả customers assigned? |
| **query.py** | | | |
| | get_unassigned_customers | O(N) | Customers chưa assign |
| | get_assigned_customers | O(N) | Customers đã assign |
| | get_route_as_list | O(route_len) | Route → list (debug) |
| | get_route_customers_only | O(route_len) | Chỉ DELIVER customers |
| | is_customer_assigned | **O(1)** | Check assigned |
| | get_customer_info | **O(1)** | Lookup: route, pos, vtype |

**Tổng: 33 functions** — 9 hàm O(1), đảm bảo ALNS inner loop nhanh.
