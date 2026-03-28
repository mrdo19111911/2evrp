# Phase 06 — Visualization & Debug Dashboard

## File structure

```
src/
  viz/
    route_map.py          ← bản đồ routes trên mặt phẳng 2D
    timeline.py           ← Gantt chart: thời gian từng xe
    convergence.py        ← fitness/cost/makespan theo iteration
    operator_stats.py     ← operator weights + performance
    pareto_plot.py        ← Pareto front visualization
    solution_inspector.py ← chi tiết 1 solution: routes, loads, violations
    dashboard.py          ← tổng hợp: multi-panel dashboard
    live.py               ← live update dashboard (callback từ ALNS)
    export.py             ← export hình ảnh, HTML report
```

**Library:** matplotlib cho static plots, plotly cho interactive dashboard.

---

## Module Interfaces

Mỗi module viz có cùng pattern:

```python
# 1. Hàm tạo figure (trả về fig, ax — KHÔNG show)
def plot_xxx(...) -> tuple[Figure, Axes]:
    """Tạo figure, trả về để caller quyết định show/save/embed."""

# 2. Hàm update (cho live dashboard — update data trên figure có sẵn)
def update_xxx(fig, ax, ...) -> None:
    """Update data trên figure đã tạo. Dùng cho animation/live."""

# 3. Hàm convenience (tạo + show luôn, cho quick debug)
def show_xxx(...) -> None:
    """plot_xxx + plt.show(). Tiện dùng khi debug."""
```

---

## route_map.py — Bản đồ Routes

### Interface

```python
def plot_route_map(sol, customers, depot, dist_matrix,
                   truck_states=None, bike_states=None,
                   highlight_violations=True,
                   show_satellites=True,
                   show_labels=True,
                   show_demand=False,
                   title=None,
                   figsize=(14, 10)) -> tuple[Figure, Axes]:
    """Vẽ bản đồ 2D: depot, customers, routes truck/bike, satellites."""

def update_route_map(fig, ax, sol, customers, depot, **kwargs) -> None:
    """Clear + redraw. Cho live dashboard."""

def show_route_map(sol, customers, depot, dist_matrix, **kwargs) -> None:
    """plot + show."""
```

### plot_route_map

- **Idea:** Bản đồ 2D hiển thị toàn bộ solution: vị trí depot, customers, routes, satellites, violations.
- **Input:**
  - `sol: dict` — solution
  - `customers: ndarray (N, F)` — positions + data
  - `depot: ndarray (2,)` — vị trí kho
  - `dist_matrix: ndarray` — cho distance labels
  - `truck_states: list[ndarray]` — optional, từ simulate (cho violation highlighting)
  - `bike_states: list[ndarray]` — optional
  - `highlight_violations: bool` — đánh dấu đỏ nodes vi phạm
  - `show_satellites: bool` — vẽ satellite markers
  - `show_labels: bool` — ID labels trên nodes
  - `show_demand: bool` — hiện demand size
  - `title: str`, `figsize: tuple`
- **Output:** `(fig, ax)`
- **Visual elements:**
  ```
  Depot:          ★ đen, kích thước lớn
  Customer nodes: ● với size tỉ lệ demand
    - Truck-assigned:  màu xanh dương
    - Bike-assigned:   màu xanh lá
    - Unassigned:      màu xám
    - Restricted:      viền đỏ (bike-only)
    - Violation:       ✕ đỏ chồng lên

  Truck routes:   ─── nét liền, đậm, mỗi truck 1 màu (palette tab10)
    - Mũi tên chỉ hướng
    - DELIVER stops: ● tròn
    - RELOAD stops:  ◆ diamond

  Bike routes:    --- nét đứt, mảnh, mỗi bike 1 màu (palette pastel)
    - Mũi tên chỉ hướng
    - DELIVER stops: ● tròn
    - RELOAD stops:  ◆ diamond

  Satellites:     ◆ vàng lớn, viền đen, label "SAT"
    - Đường nối truck↔bike tại satellite: nét chấm vàng

  Legend:         góc trên phải, liệt kê tất cả vehicles + thống kê
  Title:          "2E-VRP Solution — Cost: X VND, Makespan: Y min, Feasible: ✓/✗"
  ```
- **Test cases:**
  - TC1: Solution 10 customers, 2 trucks, 3 bikes → figure hiển thị đủ routes
  - TC2: highlight_violations=True + có TW violation → node đỏ
  - TC3: show_satellites=True → diamond vàng tại satellite nodes
  - TC4: show_labels=False → không text clutter
  - TC5: Empty solution → chỉ depot + unassigned nodes

---

### plot_single_route

- **Idea:** Zoom vào 1 route cụ thể. Hiển thị chi tiết: thứ tự stops, load tại mỗi stop, thời gian.
- **Input:**
  - `sol: dict`
  - `vtype: int`, `vid: int` — route cần vẽ
  - `customers, depot, dist_matrix: ndarray`
  - `state: ndarray` — route state từ simulate (optional)
  - `figsize: tuple`
- **Output:** `(fig, axes)` — 2 subplot: map + load/time profile
- **Visual elements:**
  ```
  Subplot 1 (map):
    - Chỉ vẽ route này + depot
    - Mỗi stop có label: "#{id} D={demand}kg"
    - Mũi tên + thứ tự số (1, 2, 3, ...)

  Subplot 2 (profile):
    - X axis = stop index
    - Y1 (trái) = load (kg), line chart, threshold capacity = nét đứt đỏ
    - Y2 (phải) = cumulative time (min)
    - Bar color: xanh = OK, đỏ = violation
  ```
- **Test cases:**
  - TC1: Truck route 5 stops → map + profile đúng 5 points
  - TC2: Load xuống dưới 0 → bar đỏ
  - TC3: Route rỗng → empty figure

---

## timeline.py — Gantt Chart

### Interface

```python
def plot_timeline(sol, truck_states, bike_states,
                  customers, vehicles,
                  show_sync=True,
                  show_tw=True,
                  figsize=(16, 8)) -> tuple[Figure, Axes]:
    """Gantt chart: mỗi xe 1 hàng, thời gian trải ngang."""

def update_timeline(fig, ax, sol, truck_states, bike_states, **kwargs) -> None:
def show_timeline(sol, truck_states, bike_states, customers, vehicles, **kwargs) -> None:
```

### plot_timeline

- **Idea:** Gantt chart hiển thị timeline tất cả xe. Thấy rõ: xe nào đang làm gì lúc nào, sync đúng không, idle time ở đâu.
- **Input:**
  - `sol: dict`
  - `truck_states: list[ndarray]` — state arrays từ simulate
  - `bike_states: list[ndarray]`
  - `customers: ndarray`
  - `vehicles: ndarray`
  - `show_sync: bool` — vẽ đường nối truck-bike tại sync points
  - `show_tw: bool` — vẽ time window brackets
- **Output:** `(fig, ax)`
- **Visual elements:**
  ```
  Y axis: mỗi xe = 1 hàng
    - Trucks ở trên (Truck 0, Truck 1, ...)
    - Bikes ở dưới (Bike 0, Bike 1, ...)
    - Đường kẻ phân cách truck/bike

  X axis: thời gian (phút từ 0)

  Mỗi xe = 1 dải ngang chia thành segments:
    ░░░ Travel (xám nhạt)
    ▓▓▓ Service/Deliver (xanh dương cho truck, xanh lá cho bike)
    ◆◆◆ Reload (vàng)
    ··· Wait/Idle (trắng, viền chấm)

  Sync lines:  ─── nối dọc giữa truck và bike tại reload events
    - Xanh nếu sync OK (within delta_t)
    - Đỏ nếu sync violation

  Time windows: [ ] brackets trên mỗi DELIVER segment
    - Xanh nếu đúng TW
    - Đỏ nếu trễ

  Annotations trên mỗi segment: "C#{id}" (customer ID)

  Bottom bar: makespan indicator (thời gian xe cuối cùng về kho)
  ```
- **Test cases:**
  - TC1: 2 trucks, 3 bikes → 5 hàng
  - TC2: Sync lines hiển thị đúng tại reload events
  - TC3: TW violation → red bracket
  - TC4: show_sync=False → không có nét nối
  - TC5: Tất cả xe idle (empty routes) → chỉ hiện hàng trống

---

## convergence.py — Fitness Convergence

### Interface

```python
def plot_convergence(log,
                     show_cost=True,
                     show_makespan=True,
                     show_penalty=True,
                     show_temperature=True,
                     show_feasibility=True,
                     figsize=(14, 10)) -> tuple[Figure, list[Axes]]:
    """Multi-panel convergence plot từ ALNS log."""

def update_convergence(fig, axes, log) -> None:
def show_convergence(log, **kwargs) -> None:
```

### plot_convergence

- **Idea:** Multi-panel chart theo iterations. Thấy rõ quá trình hội tụ, temperature, penalties, operator behavior.
- **Input:**
  - `log: dict` — output từ logger (Phase 05)
  - Toggle flags cho từng panel
- **Output:** `(fig, axes)` — list of subplots
- **Visual elements:**
  ```
  Panel 1 — Fitness + Best Fitness:
    - Y: fitness value
    - Line xám nhạt: fitness mỗi iteration (all, mờ)
    - Line xanh đậm: accepted fitness (connected)
    - Line đỏ: best fitness so far (step function, giảm dần)
    - Dots xanh: accepted improvements
    - Dots đỏ nhỏ: rejected

  Panel 2 — Cost + Makespan (dual Y-axis):
    - Y1 (trái): total cost (VND) — line xanh
    - Y2 (phải): makespan (min) — line cam
    - Chỉ vẽ accepted solutions

  Panel 3 — Penalty + Feasibility:
    - Y1 (trái): total penalty — line đỏ
    - Y2 (phải): feasibility rate (rolling window 100) — line xanh lá
    - Horizontal line: 100% feasible target
    - Background shading: xanh nhạt khi feasible > 80%, hồng khi < 30%

  Panel 4 — Temperature + w3:
    - Y1 (trái): SA temperature — line cam, log scale
    - Y2 (phải): penalty weight w3 — line tím

  Panel 5 — Feasibility scatter (optional):
    - X: iteration, Y: 0 (infeasible) / 1 (feasible)
    - Dot color: red/green
    - Rolling average line
  ```
- **Test cases:**
  - TC1: Log 10000 iterations → 5 panels, đầy đủ data
  - TC2: Best fitness non-increasing (monotone)
  - TC3: Temperature decreasing (monotone)
  - TC4: Empty log → empty figure

---

## operator_stats.py — Operator Analysis

### Interface

```python
def plot_operator_weights(log,
                          figsize=(12, 6)) -> tuple[Figure, list[Axes]]:
    """Stacked area chart: operator weights theo thời gian."""

def plot_operator_performance(log,
                              figsize=(12, 8)) -> tuple[Figure, list[Axes]]:
    """Bar chart: score/count cho mỗi operator."""

def show_operator_stats(log, **kwargs) -> None:
```

### plot_operator_weights

- **Idea:** Thấy rõ operator nào đang dominant, sự chuyển đổi qua thời gian.
- **Input:** `log: dict` — chứa destroy_weights, repair_weights snapshots
- **Output:** `(fig, axes)` — 2 subplots (destroy + repair)
- **Visual elements:**
  ```
  Subplot 1 — Destroy operator weights:
    - Stacked area chart
    - X: segment number
    - Y: weight (0 to 1)
    - Mỗi operator 1 màu, legend: operator name
    - Operator dominant → vùng lớn

  Subplot 2 — Repair operator weights:
    - Tương tự
  ```
- **Test cases:**
  - TC1: Weights sum = 1 tại mỗi snapshot
  - TC2: Operator hiệu quả → vùng mở rộng dần

---

### plot_operator_performance

- **Idea:** Bar chart so sánh hiệu suất (score/count) từng operator.
- **Visual elements:**
  ```
  Grouped bar chart:
    - X: operator name
    - Y1: total score (xanh)
    - Y2: total count (xám)
    - Y3: avg performance = score/count (đỏ, secondary axis)
  ```

---

## pareto_plot.py — Pareto Front

### Interface

```python
def plot_pareto_front(archive,
                      highlight_best=True,
                      show_dominated=False,
                      figsize=(8, 6)) -> tuple[Figure, Axes]:
    """Scatter plot: cost vs makespan, Pareto front connected."""

def update_pareto_front(fig, ax, archive) -> None:
def show_pareto_front(archive, **kwargs) -> None:
```

### plot_pareto_front

- **Idea:** Visualize trade-off giữa cost và makespan.
- **Input:**
  - `archive: list` — Pareto archive
  - `highlight_best: bool` — highlight solution tốt nhất
  - `show_dominated: bool` — vẽ cả solutions bị dominate (từ log)
- **Output:** `(fig, ax)`
- **Visual elements:**
  ```
  Pareto front: ● connected by line, sorted by cost
    - Mỗi point = 1 non-dominated solution
    - Size tỉ lệ feasibility score
    - Hover label (plotly): "Cost: X VND, Makespan: Y min"

  Dominated solutions (nếu show): ○ mờ, nhỏ

  Best cost: ★ xanh, label
  Best makespan: ★ cam, label
  Knee point: ◆ đỏ (point gần gốc nhất normalized)

  Axes: X = Cost (VND), Y = Makespan (min)
  Ideal point: ✕ ở (min_cost, min_makespan) — reference
  ```
- **Test cases:**
  - TC1: Archive 10 solutions → 10 points on front
  - TC2: Archive rỗng → empty plot
  - TC3: Pareto front convex (hoặc concave) — visual check

---

## solution_inspector.py — Chi tiết Solution

### Interface

```python
def print_solution_summary(sol, eval_result, customers, vehicles) -> str:
    """Text summary: routes, costs, violations. Return formatted string."""

def plot_solution_detail(sol, eval_result, customers, vehicles, depot,
                         dist_matrix,
                         figsize=(16, 12)) -> tuple[Figure, list[Axes]]:
    """Multi-panel detail view: routes table + load profiles + violations."""

def show_solution_detail(sol, eval_result, customers, vehicles, depot,
                         dist_matrix) -> None:
```

### print_solution_summary

- **Idea:** Text report dạng bảng, in ra console. Debug nhanh.
- **Input:**
  - `sol: dict`, `eval_result: dict` (từ evaluate_solution)
  - `customers, vehicles: ndarray`
- **Output:** `str` — formatted multi-line text
- **Format:**
  ```
  ═══════════════════════════════════════════════════════════
  2E-VRP SOLUTION SUMMARY
  ═══════════════════════════════════════════════════════════
  Feasible: ✓ (hoặc ✗ + violation count)
  Total Cost: 1,234,500 VND
  Makespan:   385.2 min
  Fitness:    742.3

  ── TRUCKS ─────────────────────────────────────────────────
  Truck 0: depot → C3(D,200kg) → C7(R,130kg) → C12(D,170kg) → depot
           Distance: 45.3 km | Time: 180 min | Load: 500 kg | Cost: 312,000 VND
  Truck 1: depot → C15(D,80kg) → depot
           Distance: 12.1 km | Time: 45 min | Load: 80 kg | Cost: 98,000 VND

  ── BIKES ──────────────────────────────────────────────────
  Bike 0:  depot → C5(D,20kg) → C7(R,+45kg) → C9(D,25kg) → C11(D,15kg) → depot
           Distance: 22.7 km | Time: 160 min | Load: 60 kg | Cost: 89,000 VND
           Reload: C7 from Truck 0, 45kg @ t=210
  Bike 1:  (empty)
  ...

  ── SATELLITES ─────────────────────────────────────────────
  C7:  Truck 0 serves [Bike 0 (45kg @ 210), Bike 2 (30kg @ 215)]
       Truck dwell: 10 min

  ── VIOLATIONS ─────────────────────────────────────────────
  (none)
  hoặc:
  [TW]  C9 arrive=320, tw_close=300 (20 min late)
  [CAP] Bike 0 load=75 at C7 (>60kg)
  [SYNC] Sat C7: bike 2 arrive=250, truck depart=230 (gap=20 > delta_t=15)

  ═══════════════════════════════════════════════════════════
  ```
- **Test cases:**
  - TC1: Feasible solution → "✓", no violations section
  - TC2: Violations → listed with detail
  - TC3: Empty routes → "(empty)"
  - TC4: String length > 0

---

### plot_solution_detail

- **Idea:** Multi-panel graphic: tổng quan solution + load profiles.
- **Visual elements:**
  ```
  Panel 1 (large, top): Route map (reuse plot_route_map)

  Panel 2 (bottom-left): Load profile tất cả trucks
    - X: stop index
    - Y: load (kg)
    - Mỗi truck 1 line
    - Threshold 2000 kg = nét đứt

  Panel 3 (bottom-center): Load profile tất cả bikes
    - Threshold 60 kg = nét đứt đỏ
    - Highlight vùng > 60 kg (shading đỏ)

  Panel 4 (bottom-right): Summary stats table
    - Total cost, makespan, # vehicles used, # satellites, # violations
  ```
- **Test cases:**
  - TC1: 4 panels rendered, không overlap
  - TC2: Bike load > 60 → red shading visible

---

## dashboard.py — Tổng hợp Dashboard

### Interface

```python
def create_dashboard(sol, eval_result, log, archive,
                     customers, vehicles, depot, dist_matrix,
                     truck_states, bike_states,
                     figsize=(20, 16)) -> tuple[Figure, list[Axes]]:
    """Full dashboard: 6 panels, tất cả thông tin cần debug."""

def save_dashboard(fig, filepath="dashboard.png", dpi=150) -> None:
    """Save static image."""

def show_dashboard(sol, eval_result, log, archive,
                   customers, vehicles, depot, dist_matrix,
                   truck_states, bike_states) -> None:
```

### create_dashboard

- **Idea:** 1 figure lớn, 6 panels, nhìn là hiểu toàn bộ trạng thái solver.
- **Input:** Tất cả data từ solver
- **Output:** `(fig, axes)`
- **Layout:**
  ```
  ┌─────────────────────────┬───────────────────────┐
  │                         │                       │
  │   Route Map (large)     │   Convergence         │
  │   plot_route_map()      │   plot_convergence()  │
  │                         │   (fitness + best)    │
  │                         │                       │
  ├────────────┬────────────┼───────────────────────┤
  │            │            │                       │
  │  Timeline  │  Pareto    │  Operator Weights     │
  │  (Gantt)   │  Front     │  (stacked area)       │
  │            │            │                       │
  ├────────────┴────────────┼───────────────────────┤
  │                         │                       │
  │  Solution Summary       │  Violation Detail     │
  │  (text/table)           │  (text/table)         │
  │                         │                       │
  └─────────────────────────┴───────────────────────┘
  ```
- **Test cases:**
  - TC1: Tất cả 6 panels rendered
  - TC2: Save as PNG → file readable
  - TC3: figsize tự scale nếu nhiều data

---

## live.py — Live Dashboard

### Interface

```python
def create_live_dashboard(customers, depot, vehicles,
                          update_interval_ms=500,
                          figsize=(20, 12)) -> dict:
    """Tạo live dashboard figure + state. Return dashboard handle."""

def live_callback(dashboard, iteration, sol, eval_result, log) -> None:
    """Callback function truyền vào solve(). Update dashboard mỗi iteration."""

def start_live_dashboard(customers, depot, vehicles) -> dict:
    """create + plt.ion() + show. Return handle."""

def stop_live_dashboard(dashboard) -> None:
    """plt.ioff() + final render."""
```

### create_live_dashboard

- **Idea:** Tạo figure matplotlib với `plt.ion()` (interactive mode). ALNS main loop gọi `live_callback` mỗi N iterations → figure tự update.
- **Input:**
  - `customers, depot, vehicles: ndarray`
  - `update_interval_ms: int` — minimum time giữa 2 lần redraw (tránh quá nhanh)
- **Output:**
  - `dashboard: dict` chứa:
    ```python
    {
        "fig": Figure,
        "axes": dict of Axes,  # keyed by panel name
        "last_update_time": float,
        "update_interval": float,  # seconds
        "iteration_count": int,
    }
    ```
- **Layout giống create_dashboard nhưng:**
  - Bớt panel (4 thay vì 6) cho performance
  - Route map (top-left), Convergence (top-right), Timeline (bottom-left), Stats text (bottom-right)

---

### live_callback

- **Idea:** Hàm callback truyền vào `solve(callback=live_callback)`. Mỗi iteration kiểm tra: đã đủ thời gian update chưa → nếu có thì redraw.
- **Input:**
  - `dashboard: dict` — handle từ create_live_dashboard
  - `iteration: int`
  - `sol: dict`
  - `eval_result: dict`
  - `log: dict`
- **Output:** None (update figure in-place)
- **Logic:**
  ```
  now = time.time()
  if now - dashboard["last_update_time"] < dashboard["update_interval"]:
      return  # quá sớm, skip

  # Update route map
  update_route_map(dashboard["fig"], dashboard["axes"]["map"], sol, ...)

  # Update convergence (chỉ append data mới)
  _append_convergence_data(dashboard["axes"]["conv"], log)

  # Update stats text
  _update_stats_text(dashboard["axes"]["stats"], eval_result)

  # Redraw
  dashboard["fig"].canvas.draw_idle()
  dashboard["fig"].canvas.flush_events()

  dashboard["last_update_time"] = now
  dashboard["iteration_count"] = iteration
  ```
- **Performance:** Skip redraw nếu interval chưa đủ → ALNS loop không bị slow down
- **Test cases:**
  - TC1: update_interval=0.5s, ALNS chạy 1000 iter/s → ~2 redraws/s (không 1000)
  - TC2: Dashboard hiện route map đúng solution hiện tại
  - TC3: Convergence plot grow theo thời gian

---

## export.py — Export

### Interface

```python
def export_dashboard_png(sol, eval_result, log, archive,
                         customers, vehicles, depot, dist_matrix,
                         truck_states, bike_states,
                         filepath="dashboard.png", dpi=150) -> str:
    """Create dashboard + save as PNG. Return filepath."""

def export_html_report(sol, eval_result, log, archive,
                       customers, vehicles, depot, dist_matrix,
                       truck_states, bike_states,
                       filepath="report.html") -> str:
    """Generate interactive HTML report with plotly. Return filepath."""

def export_solution_json(sol, eval_result, filepath="solution.json") -> str:
    """Save solution + evaluation as JSON."""

def export_pareto_csv(archive, filepath="pareto.csv") -> str:
    """Save Pareto front as CSV: cost, makespan."""
```

### export_html_report

- **Idea:** HTML file self-contained với plotly charts. Mở trên browser, interactive: zoom, hover, pan.
- **Input:** Tất cả data
- **Output:** `filepath: str`
- **Nội dung HTML:**
  ```html
  <h1>2E-VRP Solution Report</h1>
  <div id="summary">
    Feasible: ✓ | Cost: 1,234,500 VND | Makespan: 385 min
  </div>

  <h2>Route Map</h2>
  <div id="route-map"> <!-- plotly scatter, interactive --> </div>

  <h2>Timeline</h2>
  <div id="timeline"> <!-- plotly gantt --> </div>

  <h2>Convergence</h2>
  <div id="convergence"> <!-- plotly line chart --> </div>

  <h2>Pareto Front</h2>
  <div id="pareto"> <!-- plotly scatter, hover shows details --> </div>

  <h2>Operator Analysis</h2>
  <div id="operators"> <!-- plotly stacked area --> </div>

  <h2>Route Details</h2>
  <table> <!-- per-route breakdown --> </table>

  <h2>Violations</h2>
  <table> <!-- violation list --> </table>
  ```
- **Test cases:**
  - TC1: File tạo thành công, size > 0
  - TC2: Mở trên browser → tất cả charts render
  - TC3: Interactive: hover trên route map hiện customer info

---

### export_solution_json

- **Idea:** JSON chứa solution + evaluation result. Load lại bằng load_solution + visualize.
- **Format:**
  ```json
  {
    "solution": {
      "truck_stops": [[3, 7, 12, -1], ...],
      "truck_actions": [[0, 1, 0, -1], ...],
      "bike_stops": [...],
      "bike_actions": [...],
      "satellites": [[7, 0, 0, 45.0, 630], ...]
    },
    "evaluation": {
      "fitness": 742.3,
      "cost": 1234500,
      "makespan": 385.2,
      "penalty": 0,
      "feasible": true
    },
    "metadata": {
      "n_customers": 100,
      "n_trucks": 3,
      "n_bikes": 8,
      "iterations": 20000,
      "seed": 42,
      "timestamp": "2026-03-28T10:30:00"
    }
  }
  ```

---

## Tổng kết Phase 06

| File | Function | Loại | Mục đích |
|------|----------|------|----------|
| **route_map.py** | | | |
| | plot_route_map | Static | Bản đồ 2D routes + satellites + violations |
| | update_route_map | Live | Update route map |
| | show_route_map | Convenience | plot + show |
| | plot_single_route | Static | Zoom 1 route: map + load profile |
| **timeline.py** | | | |
| | plot_timeline | Static | Gantt chart tất cả xe |
| | update_timeline | Live | Update timeline |
| | show_timeline | Convenience | plot + show |
| **convergence.py** | | | |
| | plot_convergence | Static | Multi-panel fitness/cost/temp/penalty |
| | update_convergence | Live | Append data |
| | show_convergence | Convenience | plot + show |
| **operator_stats.py** | | | |
| | plot_operator_weights | Static | Stacked area: weights theo thời gian |
| | plot_operator_performance | Static | Bar chart: score/count |
| | show_operator_stats | Convenience | plot + show |
| **pareto_plot.py** | | | |
| | plot_pareto_front | Static | Cost vs makespan scatter |
| | update_pareto_front | Live | Update scatter |
| | show_pareto_front | Convenience | plot + show |
| **solution_inspector.py** | | | |
| | print_solution_summary | Text | Console text report |
| | plot_solution_detail | Static | 4-panel detail view |
| | show_solution_detail | Convenience | plot + show |
| **dashboard.py** | | | |
| | create_dashboard | Static | 6-panel full dashboard |
| | save_dashboard | Export | Save PNG |
| | show_dashboard | Convenience | create + show |
| **live.py** | | | |
| | create_live_dashboard | Live | Tạo interactive figure |
| | live_callback | Live | ALNS callback → update dashboard |
| | start_live_dashboard | Live | create + ion + show |
| | stop_live_dashboard | Live | ioff + final render |
| **export.py** | | | |
| | export_dashboard_png | Export | Static PNG |
| | export_html_report | Export | Interactive HTML (plotly) |
| | export_solution_json | Export | Solution + eval JSON |
| | export_pareto_csv | Export | Pareto front CSV |

**Tổng: 29 functions**

**Interface pattern cho mỗi viz module:**
```
plot_xxx()    → (fig, ax)           # tạo, không show
update_xxx()  → None                # update in-place (live)
show_xxx()    → None                # tạo + show (debug)
```
