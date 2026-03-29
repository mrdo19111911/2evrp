# 2E-VRP Solver Dashboard — Design Spec

## Problem

Current visualization is matplotlib static PNGs: ugly, not interactive, insufficient information, unusable for debugging or presenting. Need a real-time, interactive dashboard that shows full algorithm state at a glance.

## Users

1. **Developer/researcher** (primary): debug solver, tune operators, verify correctness, profile performance
2. **Presenter** (secondary): show results to advisors, team, customers. Must look professional.

## Tech Stack

- **Frontend:** React 18 + Vite + TypeScript
- **Map:** Deck.gl (TripsLayer animation, ScatterplotLayer customers, PathLayer routes)
- **Charts:** Recharts (convergence, operators, dimensions)
- **Gantt:** Custom SVG (lightweight)
- **Backend:** FastAPI (Python) thin wrapper around solver
- **Live updates:** WebSocket from FastAPI to React
- **Data flow:** Solver writes iteration log → FastAPI serves via WebSocket → Dashboard renders
- **Styling:** Tailwind CSS, dark theme, JetBrains Mono monospace font

## Architecture

```
┌─────────────────┐     WebSocket      ┌──────────────────────┐
│  Python Solver   │ ─────────────────> │  React Dashboard     │
│  (FastAPI wrap)  │     JSON stream    │  (Vite dev server)   │
│                  │ <───────────────── │                      │
│  - solve()       │     HTTP (config)  │  - 7 tab views       │
│  - evaluate()    │                    │  - Deck.gl map       │
│  - export JSON   │                    │  - Recharts          │
└─────────────────┘                     └──────────────────────┘
```

Data contract: solver emits JSON messages via WebSocket:

```python
# Iteration update (every N iterations)
{"type": "iteration", "iter": 1000, "fitness": 847302, "cost": 523100,
 "makespan": 234200, "penalty": 0, "feasible": true, "temperature": 45.2,
 "best_fitness": 847302, "operator_weights": {...}}

# Solution snapshot (on improvement or on demand)
{"type": "solution", "truck_routes": [...], "bike_routes": [...],
 "satellites": [...], "eval": {...}, "states": {...}}

# Status
{"type": "status", "state": "running"|"done"|"error", "iter": ..., "elapsed_s": ...}
```

## Layout

### Control Bar (always visible, top)

- Instance selector (file picker or dropdown of data/*.json)
- Config: iterations, trucks, bikes, seed (editable)
- RUN / STOP buttons
- Progress: iteration count, progress bar, ETA, current best fitness
- LOAD (import solution JSON) / EXPORT (save solution + log)

### 7 Tabs

#### Tab 1: Overview

Map (left 60%) + panels (right 40%). One glance = full state.

**Map features:**
- Customer dots with IDs visible at zoom level
- Truck routes (solid blue lines, different shades per truck)
- Bike routes (dashed orange lines)
- Satellite diamonds (purple)
- Depot square (red)
- Hover tooltip: customer ID, demand, TW, served by, arrival time, status
- Click route → highlight on Gantt
- Zoom/pan controls

**Right panels:**
- Quick stats grid: fitness, feasible, cost, makespan, trucks used, bikes used, satellites, customers served
- Violations summary: TW / capacity / sync / unserved counts (green=0, red=N)
- Mini convergence sparkline
- Mini operator bars (top 5)

#### Tab 2: Route Inspector

Per-vehicle deep dive. Vehicle selector (T0, T1... B0, B1...) at top.

**Route Schedule Chart** (inspired by PyVRP `plot_route_schedule`):
- X axis: cumulative distance along route
- Y axis: time
- Solid line: time trajectory (arrival at each stop)
- Background fill: remaining load vs capacity (blue gradient fading as load decreases)
- Grey vertical bars: time windows at each customer
- Capacity line (red dashed): max capacity
- Wait time dots: where vehicle waits for TW to open
- Customer IDs along X axis at each stop position

**Stop Table:**
- Columns: #, Customer, Action (DEL/RLD), Demand, Load After, TW, Arrive, Depart, Wait, Status (OK/LATE/SYNC)
- Sortable, highlight violations in red
- Click row → zoom map to that customer

**Gantt Chart** (all vehicles):
- One row per vehicle (trucks then bikes)
- Horizontal bars: driving+service time
- Yellow marks: wait time
- Purple marks: sync events
- Click vehicle → select in vehicle selector above

#### Tab 3: Dimensions

Fleet-wide analysis. Three panels:

**Load vs Capacity (horizontal bars):**
- One bar per vehicle: current total load / capacity
- Color: blue (OK) → yellow (>80%) → red (>95% or overloaded)
- Numeric label: "1700/2000 kg"

**Time Window Slack (table):**
- All customers sorted by slack (tightest first)
- Columns: Customer ID, Arrive, Window, Slack
- Color: green (plenty of slack) → yellow (<30 min) → red (LATE)
- Click → zoom to customer on map

**Route Distance Distribution (bar chart):**
- One bar per vehicle, sorted by distance
- Shows balance across fleet
- Average line

#### Tab 4: Giant Tour

Visualize initial solution construction:

- Step-by-step cheapest insertion animation (play/pause/step)
- Current partial tour on map
- Split points marked when split algorithm runs
- Before/after split comparison (side by side)
- Cluster assignment coloring
- Data: requires solver to export GT construction log (new feature)

#### Tab 5: Convergence

Full-size analysis charts:

**Main chart (large):**
- Fitness over iterations (current + best)
- Temperature curve (secondary Y axis)
- Penalty weight evolution
- Restart/reheat vertical markers
- Feasibility timeline (background: red when infeasible, green when feasible)
- Brush to zoom time range

**Operator weight chart:**
- Stacked area or multi-line: weight of each operator over iterations
- Grouped by type: destroy / repair / crosslayer

**Accept/reject chart:**
- Per-segment: accepted %, improved %, rejected %
- Shows SA acceptance behavior over time

#### Tab 6: Pareto

Interactive cost vs makespan scatter:

- All archive solutions as dots
- Pareto front connected with line
- Current best highlighted
- Click point → load that solution into all other tabs
- Archive size counter
- Dominated region shading

#### Tab 7: Animation

Deck.gl TripsLayer vehicle animation:

- Animated vehicle icons moving along routes on map
- Play / pause / speed (1x, 2x, 5x, 10x) controls
- Time scrubber with current clock display
- Vehicle trails with fading tails
- Sync events flash (purple pulse) when truck-bike meet
- Load indicator bar on each vehicle icon
- Customer dots change color when served (grey → green)

## New Solver Output Required

The dashboard needs data the solver doesn't currently export:

1. **Iteration log** (per-iteration or per-segment): fitness, cost, makespan, penalty, temperature, feasible, operator used, accepted
2. **Operator weights** per segment
3. **Solution snapshots** at improvement milestones
4. **Pareto archive** with full solutions
5. **GT construction log** (for Giant Tour tab): insertion order, split decisions
6. **Route simulation states** (per-stop: arrive, depart, wait, load) — already exists in engine/simulate.py

## File Structure

```
dashboard/
  package.json
  vite.config.ts
  tsconfig.json
  tailwind.config.ts
  src/
    main.tsx
    App.tsx
    api/
      solver.ts          — WebSocket client + REST calls
      types.ts           — TypeScript interfaces matching solver JSON
    components/
      ControlBar.tsx
      TabBar.tsx
      overview/
        OverviewTab.tsx
        RouteMap.tsx      — Deck.gl map component
        QuickStats.tsx
        ViolationsSummary.tsx
        MiniConvergence.tsx
        MiniOperators.tsx
      inspector/
        InspectorTab.tsx
        VehicleSelector.tsx
        RouteScheduleChart.tsx   — PyVRP-style chart
        StopTable.tsx
        GanttChart.tsx
      dimensions/
        DimensionsTab.tsx
        LoadCapacityBars.tsx
        TWSlackTable.tsx
        DistanceDistribution.tsx
      giant-tour/
        GiantTourTab.tsx
        GTAnimation.tsx
      convergence/
        ConvergenceTab.tsx
        FitnessChart.tsx
        OperatorWeights.tsx
        AcceptRejectChart.tsx
      pareto/
        ParetoTab.tsx
        ParetoScatter.tsx
      animation/
        AnimationTab.tsx
        TripAnimation.tsx    — Deck.gl TripsLayer
    hooks/
      useSolver.ts       — WebSocket connection + state
      useAnimation.ts    — animation frame loop
    utils/
      colors.ts          — truck/bike/satellite color palettes
      format.ts          — number/time formatting

server/
  app.py               — FastAPI main
  solver_runner.py     — run solver in background thread, emit WS
  routes.py            — REST endpoints (load instance, get solution, export)
```

## Color System

| Element | Color | Hex |
|---------|-------|-----|
| Truck routes | Blue shades | #3b82f6, #60a5fa, #2563eb, #1d4ed8 |
| Bike routes | Orange shades | #f97316, #fb923c, #ea580c, #c2410c |
| Satellites | Purple | #a855f7 |
| Depot | Red | #ef4444 |
| OK/feasible | Green | #10b981 |
| Warning | Yellow/amber | #f59e0b |
| Violation/error | Red | #ef4444 |
| Fitness | Amber | #f59e0b |
| Temperature | Red faded | #ef4444 opacity 0.5 |
| Background | Dark | #0f1117, #1a1d27 |
| Text primary | Light | #e0e0e0 |
| Text secondary | Grey | #6b7280, #9ca3af |

## Interactions

- **Map click route** → highlight in Gantt, select in Inspector
- **Map hover customer** → tooltip with ID, demand, TW, vehicle, status
- **Gantt click vehicle** → select in Inspector, highlight route on map
- **Stop table click row** → zoom map to customer
- **Pareto click point** → load solution across all tabs
- **Convergence brush** → zoom to iteration range
- **Animation scrub** → jump to time point

## Non-Goals (v1)

- Real street maps (use coordinate grid, not OSM/Mapbox tiles)
- 3D visualization (Cesium)
- Multi-instance comparison side-by-side
- User-defined custom operators from UI
- Mobile responsive design
