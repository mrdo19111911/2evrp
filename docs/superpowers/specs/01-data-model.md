# Phase 01 — Data Model

## File structure

```
src/
  data/
    constants.py      ← column indices, action codes, vehicle types
    io.py             ← load/save data
    distance.py       ← distance matrix computation
    generator.py      ← random test instance generation
    solution.py       ← solution encoding helpers
```

---

## constants.py

### Ý tưởng

Một chỗ duy nhất định nghĩa tất cả magic numbers. Import từ đây, không hardcode ở đâu khác.

```python
# Customer columns
COL_X           = 0
COL_Y           = 1
COL_DEMAND      = 2
COL_TW_OPEN     = 3
COL_TW_CLOSE    = 4
COL_SERVICE_TIME= 5
# thêm dimension columns từ index 6 trở đi

# Vehicle columns
VCOL_TYPE       = 0   # 0=truck, 1=bike
VCOL_CAPACITY   = 1
VCOL_COST_KM    = 2
VCOL_SPEED      = 3

# Action codes
ACT_DELIVER     = 0
ACT_RELOAD      = 1
ACT_PAD         = -1

# Vehicle types
VEH_TRUCK       = 0
VEH_BIKE        = 1

# Satellite columns
SAT_CUST        = 0
SAT_BIKE        = 1
SAT_TRUCK       = 2
SAT_KG          = 3
SAT_TIME        = 4

# Route state columns (output of simulation)
ST_CUST         = 0
ST_ACTION       = 1
ST_ARRIVE       = 2
ST_WAIT         = 3
ST_START        = 4
ST_SERVICE      = 5
ST_DEPART       = 6
ST_LOAD_BEF     = 7
ST_LOAD_AFT     = 8
ST_FEASIBLE     = 9
ST_COLS         = 10  # total columns in state array
```

---

## io.py

### load_instance

- **Idea:** Đọc file CSV/JSON chứa data bài toán, trả về numpy arrays chuẩn.
- **Input:**
  - `filepath: str` — đường dẫn file
- **Output:**
  - `customers: ndarray float64 (N, F)` — N customers, F features
  - `restricted: ndarray int8 (N,)` — 0=both, 1=bike_only
  - `depot: ndarray float64 (2,)` — tọa độ kho
  - `vehicles: ndarray float64 (K, 4)` — K vehicles
- **Logic:**
  1. Đọc file (csv hoặc json)
  2. Parse customers → array (N, F), validate: demand > 0, tw_open < tw_close
  3. Parse depot → array (2,)
  4. Parse vehicles → array (K, 4)
  5. Parse restricted flags
  6. Return tuple
- **Test cases:**
  - TC1: File CSV hợp lệ 5 customers, 2 trucks, 3 bikes → shapes đúng
  - TC2: File có customer demand = 0 → raise ValueError
  - TC3: File có tw_open > tw_close → raise ValueError
  - TC4: File không tồn tại → raise FileNotFoundError

---

### save_solution

- **Idea:** Lưu solution ra file JSON để reload sau.
- **Input:**
  - `filepath: str`
  - `truck_stops: ndarray int32 (T, L)`
  - `truck_actions: ndarray int8 (T, L)`
  - `bike_stops: ndarray int32 (B, L)`
  - `bike_actions: ndarray int8 (B, L)`
  - `satellites: ndarray float64 (S, 5)`
- **Output:** None (ghi file)
- **Logic:**
  1. Convert arrays to lists (JSON serializable)
  2. Pack vào dict
  3. json.dump
- **Test cases:**
  - TC1: Save rồi load lại → arrays identical
  - TC2: Solution rỗng (không route nào) → file hợp lệ, load lại OK

---

### load_solution

- **Idea:** Đọc solution từ file JSON đã save.
- **Input:**
  - `filepath: str`
- **Output:**
  - `truck_stops, truck_actions, bike_stops, bike_actions, satellites` — cùng types như save
- **Test cases:**
  - TC1: Round-trip save → load → so sánh np.array_equal
  - TC2: File corrupt → raise ValueError

---

## distance.py

### compute_dist_matrix

- **Idea:** Tính ma trận khoảng cách Euclidean giữa tất cả nodes (depot + customers). Tính 1 lần, dùng mãi.
- **Input:**
  - `depot: ndarray float64 (2,)`
  - `customers: ndarray float64 (N, F)` — chỉ dùng cột COL_X, COL_Y
- **Output:**
  - `dist_matrix: ndarray float64 (N+1, N+1)` — index 0 = depot, 1..N = customers
- **Logic:**
  1. Ghép depot vào đầu: `coords = vstack([depot, customers[:, :2]])` → shape (N+1, 2)
  2. Tính pairwise distance: `dist[i,j] = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)`
  3. Vectorized: dùng broadcasting `diff = coords[:, None, :] - coords[None, :, :]` → `norm`
  4. Đường chéo = 0
- **Test cases:**
  - TC1: Depot (0,0), 1 customer (3,4) → dist[0,1] = dist[1,0] = 5.0
  - TC2: 2 customers cùng vị trí → dist = 0
  - TC3: Symmetry: dist[i,j] == dist[j,i] cho mọi i,j
  - TC4: Tam giác: dist[i,j] <= dist[i,k] + dist[k,j]
  - TC5: N=1000 → chạy < 1 giây, shape = (1001, 1001)

---

## generator.py

### generate_random_instance

- **Idea:** Tạo instance ngẫu nhiên để test. Configurable: số customers, tỉ lệ truck/bike, density, v.v.
- **Input:**
  - `n_customers: int` — số customers
  - `n_trucks: int`
  - `n_bikes: int`
  - `area_size: float` — kích thước vùng (km), default 50.0
  - `demand_range: tuple (float, float)` — default (3.0, 120.0)
  - `tw_width_range: tuple (float, float)` — độ rộng time window (phút), default (30.0, 180.0)
  - `day_length: float` — độ dài ngày làm việc (phút), default 480.0 (8h)
  - `restrict_pct: float` — % customers chỉ bike, default 0.15
  - `seed: int` — random seed
- **Output:**
  - `customers: ndarray float64 (N, 6)` — [x, y, demand, tw_open, tw_close, service_time]
  - `restricted: ndarray int8 (N,)`
  - `depot: ndarray float64 (2,)`
  - `vehicles: ndarray float64 (K, 4)`
- **Logic:**
  1. `rng = np.random.default_rng(seed)`
  2. Depot tại trung tâm: `[area_size/2, area_size/2]`
  3. Customer positions: `rng.uniform(0, area_size, (n_customers, 2))`
  4. Demand: `rng.uniform(*demand_range, n_customers)` rồi clip
  5. TW: mỗi customer có tw_open random trong ngày, tw_close = tw_open + width random
  6. Service time: tỉ lệ demand (nặng hơn → lâu hơn), range 5-30 phút
  7. Restricted: random `restrict_pct` customers → bike only
  8. Vehicles: trucks (type=0, cap=2000, cost=5.0/km, speed=40 km/h), bikes (type=1, cap=60, cost=1.0/km, speed=25 km/h)
- **Test cases:**
  - TC1: n=100, seed=42 → chạy 2 lần cho kết quả giống nhau (deterministic)
  - TC2: Tất cả demands ∈ [demand_range]
  - TC3: Tất cả tw_open < tw_close
  - TC4: tw_close <= day_length
  - TC5: Số restricted customers ≈ restrict_pct * n (±10%)
  - TC6: n=0 → trả arrays rỗng, không crash

---

### generate_clustered_instance

- **Idea:** Tạo instance với customers phân bố theo cụm — gần thực tế hơn (khu dân cư).
- **Input:** Giống `generate_random_instance` + thêm:
  - `n_clusters: int` — số cụm, default 5
  - `cluster_std: float` — độ phân tán trong cụm (km), default 3.0
- **Output:** Giống `generate_random_instance`
- **Logic:**
  1. Tạo n_clusters tâm cụm random trong area
  2. Mỗi customer: chọn 1 cụm random, position = tâm_cụm + noise gaussian(0, cluster_std)
  3. Clip vào area bounds
  4. Phần còn lại giống generate_random_instance
- **Test cases:**
  - TC1: n=100, n_clusters=3 → customers phân bố quanh 3 vùng (visual check)
  - TC2: cluster_std=0 → tất cả customers trong 1 cụm nằm cùng 1 điểm
  - TC3: Deterministic với cùng seed

---

## solution.py

### create_empty_solution

- **Idea:** Tạo solution rỗng (chưa assign customer nào).
- **Input:**
  - `n_trucks: int`
  - `n_bikes: int`
  - `max_route_len: int` — default 100
- **Output:**
  - `truck_stops: ndarray int32 (T, L)` — filled with -1
  - `truck_actions: ndarray int8 (T, L)` — filled with -1
  - `bike_stops: ndarray int32 (B, L)` — filled with -1
  - `bike_actions: ndarray int8 (B, L)` — filled with -1
  - `satellites: ndarray float64 (0, 5)` — empty
- **Test cases:**
  - TC1: 3 trucks, 5 bikes → shapes (3, 100), (5, 100), satellites shape (0, 5)
  - TC2: Tất cả giá trị = -1 (trừ satellites)

---

### copy_solution

- **Idea:** Deep copy solution (ALNS cần copy trước khi modify).
- **Input:** truck_stops, truck_actions, bike_stops, bike_actions, satellites
- **Output:** bản copy mới, modify không ảnh hưởng bản gốc
- **Logic:** `np.copy()` cho mỗi array
- **Test cases:**
  - TC1: Copy → modify copy → original không đổi
  - TC2: Copy solution rỗng → OK

---

### get_route_length

- **Idea:** Đếm số stops thực (khác -1) trong 1 route.
- **Input:**
  - `stops: ndarray int32 (L,)` — 1 route
- **Output:**
  - `length: int`
- **Logic:** `np.sum(stops >= 0)`
- **Test cases:**
  - TC1: `[3, 7, 12, -1, -1]` → 3
  - TC2: `[-1, -1, -1]` → 0
  - TC3: `[0, 1, 2, 3, 4]` → 5 (no padding)

---

### insert_stop

- **Idea:** Chèn 1 customer vào vị trí pos trong route, đẩy phần còn lại sang phải.
- **Input:**
  - `stops: ndarray int32 (L,)` — mutable, sẽ bị modify
  - `actions: ndarray int8 (L,)` — mutable
  - `pos: int` — vị trí chèn (0-indexed)
  - `customer: int`
  - `action: int` — ACT_DELIVER hoặc ACT_RELOAD
- **Output:** None (modify in-place)
- **Logic:**
  1. Tìm route_len hiện tại
  2. Assert route_len < L (còn chỗ)
  3. Shift stops[pos:route_len] sang phải 1
  4. stops[pos] = customer, actions[pos] = action
- **Test cases:**
  - TC1: Route `[3, 7, -1]`, insert customer 5 at pos=1 → `[3, 5, 7]`
  - TC2: Insert at pos=0 (đầu route) → OK
  - TC3: Insert vào route đầy → raise error
  - TC4: Insert RELOAD action → actions[pos] == ACT_RELOAD

---

### remove_stop

- **Idea:** Xoá stop ở vị trí pos, kéo phần còn lại sang trái, pad -1 ở cuối.
- **Input:**
  - `stops: ndarray int32 (L,)` — mutable
  - `actions: ndarray int8 (L,)` — mutable
  - `pos: int`
- **Output:**
  - `removed_customer: int`
  - `removed_action: int`
- **Logic:**
  1. Lưu stops[pos], actions[pos]
  2. Shift stops[pos+1:] sang trái 1
  3. Pad -1 ở cuối
  4. Return removed values
- **Test cases:**
  - TC1: Route `[3, 5, 7, -1]`, remove pos=1 → `[3, 7, -1, -1]`, returns (5, 0)
  - TC2: Remove pos=0 (đầu) → OK
  - TC3: Remove từ route rỗng → raise error

---

## Tổng kết Phase 01

| File | Function | Mục đích |
|------|----------|----------|
| constants.py | (constants only) | Column indices, action codes |
| io.py | load_instance | Đọc data bài toán từ file |
| io.py | save_solution | Lưu solution ra JSON |
| io.py | load_solution | Đọc solution từ JSON |
| distance.py | compute_dist_matrix | Ma trận khoảng cách Euclidean |
| generator.py | generate_random_instance | Tạo test instance ngẫu nhiên |
| generator.py | generate_clustered_instance | Tạo test instance dạng cụm |
| solution.py | create_empty_solution | Solution rỗng |
| solution.py | copy_solution | Deep copy solution |
| solution.py | get_route_length | Đếm stops trong route |
| solution.py | insert_stop | Chèn stop vào route |
| solution.py | remove_stop | Xoá stop khỏi route |
