# Phase 03 — Initial Solution Construction

## Code Philosophy: Short Orchestrators

**Quy tắc bắt buộc cho toàn project:**
- Hàm chính (orchestrator) tối đa **10-15 dòng logic**
- Mỗi dòng = 1 function call có tên tự giải thích
- Đọc hàm chính = đọc pseudocode, hiểu ngay flow
- Logic chi tiết nằm trong sub-functions
- Áp dụng cho TẤT CẢ phases, không chỉ Phase 03

## File structure

```
src/
  init/
    clustering.py     ← phân cụm customers
    giant_tour.py     ← tạo giant tour qua big nodes + clusters
    split.py          ← split giant tour → vehicle routes
    bike_routing.py   ← tạo bike routes + reload
    sync.py           ← đồng bộ thời gian truck-bike
    builder.py        ← orchestrator (5 dòng)
```

---

## builder.py — Orchestrator

### build_initial_solution

```python
def build_initial_solution(customers, restricted, depot, vehicles,
                           dist_matrix, n_trucks, n_bikes, seed=42):
    rng = np.random.default_rng(seed)

    clusters = cluster_customers(customers, restricted, dist_matrix, rng)
    clusters = rebalance_clusters(clusters, customers, TRUCK_CAPACITY)
    giant_tour = build_giant_tour(clusters, customers, depot, dist_matrix)
    truck_sol = split_to_truck_routes(giant_tour, clusters, customers,
                                      dist_matrix, n_trucks, TRUCK_CAPACITY)
    bike_sol = build_bike_routes(clusters, truck_sol, customers,
                                 dist_matrix, n_bikes, BIKE_CAPACITY)
    satellites = synchronize_times(truck_sol, bike_sol, customers,
                                   dist_matrix, vehicles)

    return pack_solution(truck_sol, bike_sol, satellites, n_trucks, n_bikes)
```

**7 dòng logic. Mỗi dòng = 1 step rõ ràng.** Dưới đây spec chi tiết từng sub-function.

---

## clustering.py

### cluster_customers

- **Idea:** Phân customers thành cụm địa lý. Mỗi cụm sẽ được 1 truck phục vụ (deliver hoặc reload).
- **Input:**
  - `customers: ndarray (N, F)`
  - `restricted: ndarray int8 (N,)`
  - `dist_matrix: ndarray (N+1, N+1)`
  - `rng: Generator`
- **Output:**
  - `clusters: list[dict]` — mỗi cluster:
    ```python
    {
        "members": ndarray int32,    # customer indices
        "big_nodes": ndarray int32,  # demand > BIKE_CAPACITY (truck-only)
        "bike_nodes": ndarray int32, # demand <= BIKE_CAPACITY
        "restricted": ndarray int32, # bike-only (hẻm)
        "total_demand": float,
        "centroid": ndarray float64 (2,),
    }
    ```
- **Logic:**
  ```
  coords = customers[:, [COL_X, COL_Y]]
  n_clusters = estimate_n_clusters(customers, n_trucks, TRUCK_CAPACITY)
  labels = kmeans(coords, n_clusters, rng)
  clusters = group_by_label(labels, customers, restricted)
  return clusters
  ```
- **Sub-functions:**
  - `estimate_n_clusters(customers, n_trucks, capacity)` → int: ước lượng dựa trên total_demand / capacity
  - `kmeans(coords, k, rng)` → labels: simple k-means (numpy thuần, không sklearn)
  - `group_by_label(labels, customers, restricted)` → list[dict]: phân loại members trong mỗi cluster
- **Test cases:**
  - TC1: 100 customers, 3 clusters → mỗi cluster ~33 customers
  - TC2: Tất cả customers cùng vị trí → 1 cluster
  - TC3: Deterministic với cùng seed

---

### rebalance_clusters

- **Idea:** Nếu 1 cụm có total_demand > TRUCK_CAPACITY/2, chia nhỏ hơn. NGOẠI TRỪ: nếu cụm chỉ có 1 customer.demand > TRUCK_CAPACITY/2 → không chia nữa (customer đó bắt buộc 1 truck riêng).
- **Input:**
  - `clusters: list[dict]`
  - `customers: ndarray (N, F)`
  - `truck_capacity: float` — 2000 kg
- **Output:**
  - `clusters: list[dict]` — rebalanced, có thể nhiều hơn input
- **Logic:**
  ```
  result = []
  for cluster in clusters:
      if cluster["total_demand"] <= truck_capacity / 2:
          result.append(cluster)
          continue

      # Có customer nặng (>capacity/2) mà đứng 1 mình?
      heavy = cluster["members"][customers[cluster["members"], COL_DEMAND] > truck_capacity / 2]
      if len(heavy) > 0 and len(cluster["members"]) == len(heavy):
          # Toàn heavy → không chia nữa, mỗi customer = 1 cluster
          for c in heavy:
              result.append(make_single_cluster(c, customers))
          continue

      # Chia cluster: tách heavy ra riêng, phần còn lại chia đôi
      if len(heavy) > 0:
          for c in heavy:
              result.append(make_single_cluster(c, customers))
          remaining = setdiff1d(cluster["members"], heavy)
      else:
          remaining = cluster["members"]

      # Chia remaining thành 2 sub-clusters bằng k-means(k=2)
      sub_clusters = split_cluster(remaining, customers, k=2)
      result.extend(sub_clusters)

  return result
  ```
- **Test cases:**
  - TC1: Cluster demand=500 (< 1000) → giữ nguyên
  - TC2: Cluster demand=1500, no heavy customer → chia 2
  - TC3: Cluster 1 customer demand=1200 → 1 cluster riêng, không chia
  - TC4: Cluster 3 customers [1200, 30, 40] → 1200 tách riêng, [30,40] gom lại

---

## giant_tour.py

### build_giant_tour

- **Idea:** Tạo 1 tour lớn đi qua: (1) big nodes (demand > BIKE_CAPACITY, truck phải deliver trực tiếp), và (2) centroid/đại diện mỗi cụm (truck ghé để reload cho bikes). Mỗi cụm chỉ ghé 1 lần. Tour = thứ tự tối ưu bằng nearest-neighbor.
- **Input:**
  - `clusters: list[dict]`
  - `customers: ndarray (N, F)`
  - `depot: ndarray (2,)`
  - `dist_matrix: ndarray (N+1, N+1)`
- **Output:**
  - `giant_tour: list[dict]` — ordered list of stops:
    ```python
    [
        {"node": 15, "type": "deliver", "demand": 200.0, "cluster_idx": -1},
        {"node": 7,  "type": "satellite", "demand": 180.0, "cluster_idx": 2},
        {"node": 42, "type": "deliver", "demand": 150.0, "cluster_idx": -1},
        ...
    ]
    ```
- **Logic:**
  ```
  stops = []

  for i, cluster in enumerate(clusters):
      # Big nodes: truck deliver trực tiếp
      for c in cluster["big_nodes"]:
          stops.append({"node": c, "type": "deliver",
                        "demand": customers[c, COL_DEMAND], "cluster_idx": -1})

      # Cluster có bike nodes → cần satellite (chọn node gần centroid nhất)
      if len(cluster["bike_nodes"]) > 0:
          sat_node = find_cluster_satellite(cluster, customers, dist_matrix)
          sat_demand = customers[cluster["bike_nodes"], COL_DEMAND].sum()
          stops.append({"node": sat_node, "type": "satellite",
                        "demand": sat_demand, "cluster_idx": i})

  # Order stops bằng nearest-neighbor từ depot
  giant_tour = nearest_neighbor_order(stops, depot, dist_matrix)
  return giant_tour
  ```
- **Sub-functions:**
  - `find_cluster_satellite(cluster, customers, dist_matrix)` → int: node trong cluster gần centroid nhất
  - `nearest_neighbor_order(stops, depot, dist_matrix)` → list[dict]: sắp xếp NN
- **Test cases:**
  - TC1: 3 clusters, mỗi cluster 1 big + bike nodes → tour có 3 delivers + 3 satellites
  - TC2: Cluster chỉ big nodes → chỉ "deliver", không satellite
  - TC3: Cluster chỉ bike nodes → chỉ "satellite"
  - TC4: Tour bắt đầu và kết thúc tại depot (implied)

---

## split.py

### split_to_truck_routes

- **Idea:** Split giant tour thành nhiều truck routes sao cho mỗi route không vượt capacity. Classic "split" algorithm cho VRP.
- **Input:**
  - `giant_tour: list[dict]`
  - `clusters: list[dict]`
  - `customers: ndarray (N, F)`
  - `dist_matrix: ndarray (N+1, N+1)`
  - `n_trucks: int`
  - `truck_capacity: float`
- **Output:**
  - `truck_sol: list[dict]` — mỗi route:
    ```python
    {
        "stops": ndarray int32,     # customer indices
        "actions": ndarray int8,    # ACT_DELIVER / ACT_RELOAD
        "total_demand": float,
        "total_distance": float,
    }
    ```
- **Logic:**
  ```
  # Optimal split: shortest path on auxiliary graph
  # Node i → j: cost = distance(depot→i→i+1→...→j→depot)
  # Edge exists only if sum(demand[i..j]) <= capacity

  n = len(giant_tour)
  cost = full(n + 1, inf)    # cost[i] = min cost to serve tour[0..i-1]
  pred = full(n + 1, -1, int)  # predecessor for backtracking
  cost[0] = 0.0

  for i in range(n):
      load = 0.0
      dist = dist_matrix[0, giant_tour[i]["node"] + 1]  # depot → first

      for j in range(i, n):
          load += giant_tour[j]["demand"]
          if load > truck_capacity:
              break

          if j > i:
              dist += dist_matrix[giant_tour[j-1]["node"]+1, giant_tour[j]["node"]+1]
          dist_with_return = dist + dist_matrix[giant_tour[j]["node"]+1, 0]

          if cost[i] + dist_with_return < cost[j + 1]:
              cost[j + 1] = cost[i] + dist_with_return
              pred[j + 1] = i

  # Backtrack → routes
  routes = backtrack_split(pred, n, giant_tour)

  # Nếu routes > n_trucks → merge shortest routes
  routes = merge_if_needed(routes, n_trucks, dist_matrix, truck_capacity)

  return routes
  ```
- **Sub-functions:**
  - `backtrack_split(pred, n, giant_tour)` → list[dict]: tạo routes từ pred array
  - `merge_if_needed(routes, n_trucks, dist_matrix, capacity)` → list[dict]: gộp routes nếu quá nhiều
- **Test cases:**
  - TC1: Giant tour 5 stops, total demand < capacity → 1 route
  - TC2: Giant tour 10 stops, total demand = 3x capacity → 3 routes
  - TC3: Mỗi route.total_demand <= capacity
  - TC4: Nếu n_routes > n_trucks → merge thành n_trucks routes
  - TC5: Split tối ưu (shortest path) → cost <= nearest-neighbor split

---

## bike_routing.py

### build_bike_routes

- **Idea:** Cho mỗi cluster có bike_nodes, tạo bike routes. Bike xuất phát từ kho → giao → reload tại satellite (big node trong cluster hoặc truck stop) → giao tiếp → kho.
- **Input:**
  - `clusters: list[dict]`
  - `truck_sol: list[dict]` — truck routes (biết satellite ở đâu)
  - `customers: ndarray (N, F)`
  - `dist_matrix: ndarray (N+1, N+1)`
  - `n_bikes: int`
  - `bike_capacity: float` — 60 kg
- **Output:**
  - `bike_sol: list[dict]` — mỗi route:
    ```python
    {
        "stops": ndarray int32,
        "actions": ndarray int8,
        "initial_load": float,
        "satellite_node": int,     # reload tại node nào
        "cluster_idx": int,
    }
    ```
- **Logic:**
  ```
  all_routes = []

  for cluster in clusters:
      if len(cluster["bike_nodes"]) == 0:
          continue

      sat_node = find_satellite_in_truck_sol(cluster, truck_sol)
      routes = route_cluster_bikes(cluster["bike_nodes"], sat_node,
                                    customers, dist_matrix, bike_capacity)
      all_routes.extend(routes)

  # Assign routes to bikes (1 bike có thể chạy nhiều routes nếu đủ thời gian)
  bike_sol = assign_routes_to_bikes(all_routes, n_bikes)
  return bike_sol
  ```
- **Sub-functions:**
  - `find_satellite_in_truck_sol(cluster, truck_sol)` → int: tìm node satellite trong truck route
  - `route_cluster_bikes(nodes, sat, customers, dist, capacity)` → list[dict]: NN routing với reload
  - `assign_routes_to_bikes(routes, n_bikes)` → list[dict]: phân bổ routes cho bikes

### route_cluster_bikes

- **Idea:** Routing bike qua bike_nodes của 1 cluster. Khi load đầy → ghé satellite reload → tiếp.
- **Input:**
  - `bike_nodes: ndarray int32`
  - `sat_node: int`
  - `customers, dist_matrix: ndarray`
  - `bike_capacity: float`
- **Output:**
  - `routes: list[dict]`
- **Logic:**
  ```
  sorted_nodes = sort_by_nearest_neighbor(bike_nodes, sat_node, dist_matrix)
  routes = []
  current_route = []
  current_load = 0.0

  for node in sorted_nodes:
      demand = customers[node, COL_DEMAND]

      if current_load + demand > bike_capacity:
          # Insert reload tại satellite
          current_route.append((sat_node, ACT_RELOAD))
          current_load = 0.0

      current_route.append((node, ACT_DELIVER))
      current_load += demand

  if current_route:
      routes.append(make_bike_route(current_route, customers, bike_capacity))

  return routes
  ```
- **Test cases:**
  - TC1: 3 nodes, total=50 (< 60) → 1 route, no reload
  - TC2: 5 nodes, total=100 → 1 route with reload in middle
  - TC3: 10 nodes, total=300 → multiple reloads
  - TC4: 1 node, demand=55 → 1 route, no reload

---

## sync.py

### synchronize_times

- **Idea:** Sau khi có truck routes + bike routes, ước tính thời gian arrival tại mỗi stop → tạo satellite events với planned_time hợp lý.
- **Input:**
  - `truck_sol: list[dict]`
  - `bike_sol: list[dict]`
  - `customers: ndarray (N, F)`
  - `dist_matrix: ndarray (N+1, N+1)`
  - `vehicles: ndarray (K, 4)`
- **Output:**
  - `satellites: ndarray float64 (S, 5)` — [cust, bike, truck, kg, time]
- **Logic:**
  ```
  truck_times = estimate_all_arrival_times(truck_sol, dist_matrix, TRUCK_SPEED)
  bike_times = estimate_all_arrival_times(bike_sol, dist_matrix, BIKE_SPEED)
  satellites = match_reload_events(truck_sol, bike_sol, truck_times, bike_times, customers)
  satellites = adjust_sync_times(satellites, truck_times, bike_times, SYNC_DELTA_T)
  return satellites
  ```
- **Sub-functions:**
  - `estimate_all_arrival_times(routes, dist_matrix, speed)` → list[ndarray]: thời gian tại mỗi stop
  - `match_reload_events(truck_sol, bike_sol, t_times, b_times, customers)` → ndarray: pair truck-bike tại satellites
  - `adjust_sync_times(satellites, t_times, b_times, delta_t)` → ndarray: điều chỉnh planned_time để minimize waiting
- **Test cases:**
  - TC1: 1 truck, 1 bike, 1 satellite → 1 event, planned_time hợp lý
  - TC2: 1 truck, 3 bikes cùng satellite → 3 events, planned_times cách nhau
  - TC3: Truck arrive trước bike → planned_time = bike arrive time
  - TC4: Adjust: |truck_time - bike_time| <= delta_t (hoặc gần nhất có thể)

---

## builder.py — pack_solution

### pack_solution

- **Idea:** Convert list[dict] routes → numpy arrays format cho solution dict.
- **Input:**
  - `truck_sol: list[dict]`, `bike_sol: list[dict]`, `satellites: ndarray`
  - `n_trucks, n_bikes: int`
- **Output:**
  - `sol: dict` — full solution dict (Phase 01b format)
- **Logic:**
  ```
  sol = create_solution(n_trucks, n_bikes)
  fill_truck_routes(sol, truck_sol)
  fill_bike_routes(sol, bike_sol)
  sol["satellites"] = satellites
  rebuild_index(sol, n_customers)
  return sol
  ```
- **Test cases:**
  - TC1: Pack → evaluate_solution không crash
  - TC2: Route data khớp với list[dict] input

---

## Áp dụng Short Orchestrator pattern cho Phase 05

Phase 05 `solve()` cũng phải ngắn. Refactor:

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

**10 dòng logic trong loop.** Chi tiết trong sub-functions:

### Sub-functions cho solve

```python
def setup(customers, config_overrides, seed):
    """Return (config, rng)."""

def init_search_state(sol, customers, restricted, vehicles, dist_matrix, config):
    """Return state dict: fitness, best_sol, temperature, weights, log, archive, etc."""

def destroy_and_repair(sol, state, customers, restricted, dist_matrix, vehicles, rng, config):
    """Copy → destroy → repair → return new_sol."""

def maybe_local_search(sol, iteration, state, dist_matrix, customers, config):
    """Chạy LS nếu đúng frequency. Return sol (modified or not)."""

def maybe_cross_layer(sol, iteration, customers, restricted, dist_matrix, rng, config):
    """Chạy cross-layer op nếu đúng frequency. Return sol."""

def accept_and_update(sol, new_sol, state, rng, config):
    """SA accept → update best → update weights → cool temp. Return (sol, state)."""

def log_and_callback(state, iteration, callback):
    """Log iteration + call callback nếu có."""

def should_stop(state, config):
    """Return True nếu no_improve_limit hoặc condition khác."""

def finalize(sol, state):
    """Post-processing → return (best_sol, fitness, archive, log)."""
```

---

## Tổng kết Phase 03 (revised)

| File | Function | Dòng logic | Mục đích |
|------|----------|------------|----------|
| **builder.py** | | | |
| | build_initial_solution | **7** | Orchestrator |
| | pack_solution | **5** | Convert → numpy |
| **clustering.py** | | | |
| | cluster_customers | ~5 | K-means clustering |
| | rebalance_clusters | ~15 | Chia cluster nặng |
| | estimate_n_clusters | ~3 | Ước lượng k |
| | kmeans | ~20 | K-means numpy thuần |
| | group_by_label | ~10 | Group members |
| | split_cluster | ~5 | Chia 1 cluster thành k |
| | make_single_cluster | ~3 | 1 customer = 1 cluster |
| **giant_tour.py** | | | |
| | build_giant_tour | ~10 | Giant tour qua big nodes + clusters |
| | find_cluster_satellite | ~5 | Node gần centroid nhất |
| | nearest_neighbor_order | ~10 | NN ordering |
| **split.py** | | | |
| | split_to_truck_routes | ~15 | Optimal split (shortest path) |
| | backtrack_split | ~10 | Backtrack pred → routes |
| | merge_if_needed | ~10 | Gộp nếu > n_trucks |
| **bike_routing.py** | | | |
| | build_bike_routes | **5** | Orchestrator per cluster |
| | find_satellite_in_truck_sol | ~5 | Tìm satellite |
| | route_cluster_bikes | ~15 | NN + reload insertion |
| | assign_routes_to_bikes | ~10 | Phân routes cho bikes |
| **sync.py** | | | |
| | synchronize_times | **4** | Orchestrator sync |
| | estimate_all_arrival_times | ~10 | Ước tính thời gian |
| | match_reload_events | ~10 | Pair truck-bike |
| | adjust_sync_times | ~10 | Minimize wait time |

**Orchestrators: 5-7 dòng.** Sub-functions: 3-20 dòng mỗi hàm.
