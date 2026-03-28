# 2E-VRP with Synchronization — Master Index

## Bài toán

Two-echelon VRP: xe ô tô (2 tấn) + xe máy (60 kg) phối hợp giao hàng từ 1 kho.
Xe máy có thể reload hàng từ ô tô tại các customer nodes. Đồng bộ thời gian +-delta_t.

## Mục tiêu

Multi-objective: tối thiểu tổng chi phí VÀ makespan (thời gian xe cuối cùng về kho).

## Giải pháp

Two-Phase Decomposition + Adaptive Large Neighborhood Search (ALNS).
3 layer: Assignment → Truck Routing → Bike Routing, với feedback loop.

## Implementation

- Python, pure numpy, functional/procedural (không OOP)
- Visualization dashboard để debug chi tiết

## Spec Documents

| Phase | File | Nội dung | Functions |
|-------|------|----------|-----------|
| 01 | [01-data-model.md](01-data-model.md) | Data structures, constants, loading, test data generation | 11 |
| 01b | [01b-solution-ops.md](01b-solution-ops.md) | Solution structure, route ops, cost model, constraint checks, delta eval | 33 |
| 02 | [02-simulation-engine.md](02-simulation-engine.md) | Route simulation, constraint validation, fitness evaluation | 15 |
| 03 | [03-initial-solution.md](03-initial-solution.md) | Assignment heuristic, nearest-neighbor routing, satellite selection | 9 |
| 04 | [04-alns-operators.md](04-alns-operators.md) | Destroy, repair, cross-layer, local search, weights | 20 |
| 05 | [05-alns-main-loop.md](05-alns-main-loop.md) | SA acceptance, penalty schedule, Pareto archive, main loop | 15 |
| 06 | [06-visualization.md](06-visualization.md) | Route map, timeline, convergence, dashboard, live, export | 29 |

## Dependency Graph

```
Phase 01 (Data) → Phase 01b (Solution Ops, Cost, Checks)
  └→ Phase 02 (Simulation)
       └→ Phase 03 (Initial Solution)
            └→ Phase 04 (ALNS Operators)
                 └→ Phase 05 (ALNS Main Loop)
  └→ Phase 06 (Visualization) — dùng được ngay từ Phase 01b trở đi
```

## Quy ước chung

- Mỗi function: pure function, nhận numpy arrays, trả numpy arrays
- Column indices: module-level constants (COL_X = 0, COL_Y = 1, ...)
- Không side effect, không global state
- Random: luôn truyền `rng` (np.random.Generator) làm parameter
- Distance matrix index: 0 = depot, 1..N = customers
