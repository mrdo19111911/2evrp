# 2E-VRP Solver Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interactive web dashboard for the 2E-VRP solver — run solver, view routes/dimensions/convergence in real-time, inspect per-vehicle details, animate solutions.

**Architecture:** FastAPI backend wraps the existing Python solver, streams iteration data over WebSocket. React frontend with Deck.gl maps, Recharts, and custom SVG renders 7 tab views. Dark theme, dense information, presentation-ready.

**Tech Stack:** React 18, Vite, TypeScript, Tailwind CSS, Deck.gl, Recharts, FastAPI, WebSocket, uvicorn

---

## File Structure

```
dashboard/
  package.json
  vite.config.ts
  tsconfig.json
  tailwind.config.ts
  index.html
  src/
    main.tsx                    — React entry point
    App.tsx                     — Layout: ControlBar + TabBar + active tab
    types.ts                    — All TypeScript interfaces
    api/
      solver.ts                 — WebSocket client + REST
    hooks/
      useSolver.ts              — Connection + state management
    components/
      ControlBar.tsx            — Instance selector, config, run/stop, progress
      TabBar.tsx                — 7 tab buttons
      overview/
        OverviewTab.tsx         — Map + panels layout
        RouteMap.tsx            — Deck.gl map with routes, customers, satellites
        QuickStats.tsx          — 2x4 grid of key metrics
        ViolationsSummary.tsx   — TW/capacity/sync/unserved counts
        MiniConvergence.tsx     — Sparkline of fitness
        MiniOperators.tsx       — Top-5 operator bars
      inspector/
        InspectorTab.tsx        — Vehicle selector + chart + table + gantt
        VehicleSelector.tsx     — T0 T1... B0 B1... buttons
        RouteScheduleChart.tsx  — PyVRP-style: distance vs time vs load
        StopTable.tsx           — Per-stop detail table
        GanttChart.tsx          — All vehicles timeline
      dimensions/
        DimensionsTab.tsx       — 3-panel layout
        LoadCapacityBars.tsx    — Load vs capacity per vehicle
        TWSlackTable.tsx        — Time window slack sorted by urgency
        DistanceDistribution.tsx — Bar chart of route distances
      convergence/
        ConvergenceTab.tsx      — Full charts layout
        FitnessChart.tsx        — Fitness + best + temperature over iterations
        OperatorWeights.tsx     — Stacked area: weights over time
        AcceptRejectChart.tsx   — Per-segment accept/reject ratio
      pareto/
        ParetoTab.tsx           — Scatter + archive info
        ParetoScatter.tsx       — Cost vs makespan interactive
      giant-tour/
        GiantTourTab.tsx        — Step-by-step GT visualization
      animation/
        AnimationTab.tsx        — Deck.gl TripsLayer + controls

server/
  app.py                        — FastAPI main, CORS, WebSocket
  solver_runner.py              — Background thread solver + WS emitter
  routes.py                     — REST: list instances, get solution, export
  solution_serializer.py        — Convert numpy solution tuple → JSON dict
```

---

## Phase 1: Server + Data Contract

### Task 1: Solution Serializer

**Files:**
- Create: `server/solution_serializer.py`
- Create: `server/__init__.py`
- Test: `tests/test_server/test_serializer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_server/__init__.py
# (empty)

# tests/test_server/test_serializer.py
import numpy as np
from src.data.io import load_instance
from src.data.distance import compute_dist_matrix
from src.init.builder import build_initial_solution
from server.solution_serializer import serialize_solution, serialize_eval

def test_serialize_solution_has_required_keys():
    customers, depot, vehicles = load_instance("data/grid_20x20.json")
    dist_matrix = compute_dist_matrix(depot, customers)
    sol = build_initial_solution(customers, depot, vehicles, dist_matrix, 5, 15, seed=42)
    result = serialize_solution(sol, customers, depot)
    assert "truck_routes" in result
    assert "bike_routes" in result
    assert "satellites" in result
    assert "summary" in result
    assert isinstance(result["truck_routes"], list)
    assert len(result["truck_routes"]) == 5

def test_truck_route_has_stops_with_ids():
    customers, depot, vehicles = load_instance("data/grid_20x20.json")
    dist_matrix = compute_dist_matrix(depot, customers)
    sol = build_initial_solution(customers, depot, vehicles, dist_matrix, 5, 15, seed=42)
    result = serialize_solution(sol, customers, depot)
    route = result["truck_routes"][0]
    assert "vehicle_id" in route
    assert "stops" in route
    assert "total_distance_m" in route
    if len(route["stops"]) > 0:
        stop = route["stops"][0]
        assert "customer_id" in stop
        assert "action" in stop
        assert "x" in stop
        assert "y" in stop
        assert "demand_g" in stop

def test_serialize_eval_returns_dict():
    ev = np.zeros(20, dtype=np.int64)
    ev[0] = 847302  # fitness
    ev[1] = 523100  # cost
    ev[3] = 234200  # makespan
    result = serialize_eval(ev)
    assert result["fitness"] == 847302
    assert result["cost"] == 523100
```

- [ ] **Step 2: Run test — verify fails**

```bash
cd e:/2evrp && python -m pytest tests/test_server/test_serializer.py -v
```

Expected: ModuleNotFoundError for `server.solution_serializer`

- [ ] **Step 3: Implement serializer**

```python
# server/__init__.py
# (empty)

# server/solution_serializer.py
"""Convert solver numpy arrays to JSON-serializable dicts."""
import numpy as np
from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, ACT_PAD, VEH_TRUCK, VEH_BIKE,
    COL_X, COL_Y, COL_DEMAND, COL_TW_OPEN, COL_TW_CLOSE, COL_SERVICE,
    COL_RESTRICTED,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_TRUCK_LENGTHS,
    SOL_BIKE_STOPS, SOL_BIKE_ACTIONS, SOL_BIKE_LENGTHS,
    SOL_TRUCK_LOADS, SOL_BIKE_LOADS,
    SOL_TRUCK_DISTANCES, SOL_BIKE_DISTANCES,
    SOL_SATELLITES, SOL_META,
    META_N_TRUCKS, META_N_BIKES, META_N_SATELLITES,
    SAT_CUST, SAT_BIKE, SAT_TRUCK, SAT_KG, SAT_TIME,
    EV_FITNESS, EV_COST, EV_MAKESPAN, EV_TOTAL_PENALTY,
    EV_FEASIBLE, EV_SYNC_COST, EV_TOTAL_WAIT,
)


def _action_str(code):
    if code == ACT_DELIVER:
        return "deliver"
    if code == ACT_RELOAD:
        return "reload"
    return "pad"


def _serialize_route(stops_arr, actions_arr, length, load, distance,
                     vehicle_id, vtype_str, customers):
    """Serialize one route to dict."""
    stops = []
    for i in range(length):
        c = int(stops_arr[i])
        stops.append({
            "customer_id": c,
            "action": _action_str(int(actions_arr[i])),
            "x": float(customers[c, COL_X]),
            "y": float(customers[c, COL_Y]),
            "demand_g": int(customers[c, COL_DEMAND]),
            "tw_open_s": int(customers[c, COL_TW_OPEN]),
            "tw_close_s": int(customers[c, COL_TW_CLOSE]),
            "service_s": int(customers[c, COL_SERVICE]),
            "restricted": bool(customers[c, COL_RESTRICTED]),
        })
    return {
        "vehicle_id": vehicle_id,
        "vehicle_type": vtype_str,
        "n_stops": length,
        "stops": stops,
        "total_load_g": int(load),
        "total_distance_m": int(distance),
    }


def serialize_solution(sol, customers, depot):
    """Convert solution tuple + customers to JSON-safe dict."""
    meta = sol[SOL_META]
    n_trucks = int(meta[META_N_TRUCKS])
    n_bikes = int(meta[META_N_BIKES])
    n_sats = int(meta[META_N_SATELLITES])

    truck_routes = []
    for t in range(n_trucks):
        L = int(sol[SOL_TRUCK_LENGTHS][t])
        truck_routes.append(_serialize_route(
            sol[SOL_TRUCK_STOPS][t], sol[SOL_TRUCK_ACTIONS][t], L,
            sol[SOL_TRUCK_LOADS][t], sol[SOL_TRUCK_DISTANCES][t],
            t, "truck", customers))

    bike_routes = []
    for b in range(n_bikes):
        L = int(sol[SOL_BIKE_LENGTHS][b])
        bike_routes.append(_serialize_route(
            sol[SOL_BIKE_STOPS][b], sol[SOL_BIKE_ACTIONS][b], L,
            sol[SOL_BIKE_LOADS][b], sol[SOL_BIKE_DISTANCES][b],
            b, "bike", customers))

    sats_arr = sol[SOL_SATELLITES]
    satellites = []
    for i in range(n_sats):
        satellites.append({
            "customer_id": int(sats_arr[i, SAT_CUST]),
            "bike_id": int(sats_arr[i, SAT_BIKE]),
            "truck_id": int(sats_arr[i, SAT_TRUCK]),
            "transfer_g": int(sats_arr[i, SAT_KG]),
            "planned_time_s": int(sats_arr[i, SAT_TIME]),
        })

    n_served = sum(1 for r in truck_routes + bike_routes
                   for s in r["stops"] if s["action"] == "deliver")

    return {
        "truck_routes": truck_routes,
        "bike_routes": bike_routes,
        "satellites": satellites,
        "depot": {"x": float(depot[0]), "y": float(depot[1])},
        "summary": {
            "n_trucks_used": sum(1 for r in truck_routes if r["n_stops"] > 0),
            "n_bikes_used": sum(1 for r in bike_routes if r["n_stops"] > 0),
            "n_satellites": n_sats,
            "n_served": n_served,
            "n_customers": len(customers),
        },
    }


def serialize_eval(ev):
    """Convert eval array (i64) to dict."""
    return {
        "fitness": int(ev[EV_FITNESS]),
        "cost": int(ev[EV_COST]),
        "sync_cost": int(ev[EV_SYNC_COST]),
        "makespan": int(ev[EV_MAKESPAN]),
        "total_penalty": int(ev[EV_TOTAL_PENALTY]),
        "total_wait": int(ev[EV_TOTAL_WAIT]),
        "feasible": bool(ev[EV_FEASIBLE] > 0),
    }
```

- [ ] **Step 4: Run tests — verify passes**

```bash
cd e:/2evrp && python -m pytest tests/test_server/test_serializer.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add server/ tests/test_server/
git commit -m "feat(dashboard): add solution serializer for JSON export"
```

---

### Task 2: FastAPI Server + WebSocket

**Files:**
- Create: `server/app.py`
- Create: `server/solver_runner.py`
- Create: `server/routes.py`

- [ ] **Step 1: Write app.py — FastAPI + WebSocket + CORS**

```python
# server/app.py
"""FastAPI server: REST for config, WebSocket for live solver updates."""
import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="2E-VRP Solver Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared state
clients: list[WebSocket] = []
solver_task = None


async def broadcast(msg: dict):
    """Send JSON to all connected WebSocket clients."""
    text = json.dumps(msg)
    disconnected = []
    for ws in clients:
        try:
            await ws.send_text(text)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        clients.remove(ws)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            await ws.receive_text()  # keep alive, ignore client messages
    except WebSocketDisconnect:
        if ws in clients:
            clients.remove(ws)


@app.get("/api/instances")
def list_instances():
    """List available instance JSON files."""
    data_dir = Path("data")
    files = sorted(data_dir.glob("*.json"))
    return [{"name": f.name, "path": str(f)} for f in files]


# Import routes after app is created
from server.routes import router
app.include_router(router, prefix="/api")
```

- [ ] **Step 2: Write solver_runner.py — background solver with WS emit**

```python
# server/solver_runner.py
"""Run solver in background thread, emit progress over WebSocket."""
import asyncio
import time
import threading
import numpy as np

from src.data.io import load_instance
from src.data.distance import compute_dist_matrix
from src.alns.loop import solve
from src.alns.logger import log_to_dict
from server.solution_serializer import serialize_solution, serialize_eval


class SolverRunner:
    """Manages one solver run in a background thread."""

    def __init__(self, broadcast_fn, loop):
        self.broadcast = broadcast_fn
        self.loop = loop
        self.running = False
        self.thread = None
        self.customers = None
        self.depot = None
        self.vehicles = None
        self.dist_matrix = None

    def start(self, instance_path, n_trucks, n_bikes, max_iterations, seed):
        if self.running:
            return {"error": "solver already running"}
        self.running = True

        self.customers, self.depot, self.vehicles = load_instance(instance_path)
        self.dist_matrix = compute_dist_matrix(self.depot, self.customers)

        self.thread = threading.Thread(
            target=self._run,
            args=(n_trucks, n_bikes, max_iterations, seed),
            daemon=True,
        )
        self.thread.start()
        return {"status": "started"}

    def stop(self):
        self.running = False

    def _emit(self, msg):
        asyncio.run_coroutine_threadsafe(self.broadcast(msg), self.loop)

    def _run(self, n_trucks, n_bikes, max_iterations, seed):
        start_time = time.time()
        last_emit = 0

        def callback(iteration, best_sol, eval_arr, log_dict):
            nonlocal last_emit
            if not self.running:
                raise StopIteration("stopped by user")

            now = time.time()
            if now - last_emit < 0.1:  # throttle to 10 updates/s
                return
            last_emit = now

            self._emit({
                "type": "iteration",
                "iter": int(iteration),
                "max_iter": max_iterations,
                "elapsed_s": round(now - start_time, 2),
                "eval": serialize_eval(eval_arr),
                "log_tail": {
                    "fitness": log_dict["fitness"][-100:] if log_dict["fitness"] else [],
                    "best_fitness": log_dict["best_fitness"][-100:] if log_dict["best_fitness"] else [],
                    "temperature": log_dict["temperature"][-100:] if log_dict["temperature"] else [],
                },
            })

        try:
            from src.data.constants import CFG_MAX_ITERATIONS
            best_sol, best_fitness, archive, log = solve(
                self.customers, self.depot, self.vehicles, self.dist_matrix,
                n_trucks, n_bikes,
                config_overrides={CFG_MAX_ITERATIONS: max_iterations},
                seed=seed,
                callback=callback,
            )

            solution_data = serialize_solution(best_sol, self.customers, self.depot)
            self._emit({
                "type": "done",
                "solution": solution_data,
                "fitness": int(best_fitness),
                "elapsed_s": round(time.time() - start_time, 2),
                "log": log,
                "archive_size": len(archive),
            })
        except StopIteration:
            self._emit({"type": "stopped"})
        except Exception as e:
            self._emit({"type": "error", "message": str(e)})
        finally:
            self.running = False
```

- [ ] **Step 3: Write routes.py — REST endpoints**

```python
# server/routes.py
"""REST endpoints: start/stop solver, get status."""
import asyncio
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# Lazy init — runner created when first request comes in
_runner = None


def _get_runner():
    global _runner
    if _runner is None:
        from server.app import broadcast
        loop = asyncio.get_event_loop()
        from server.solver_runner import SolverRunner
        _runner = SolverRunner(broadcast, loop)
    return _runner


class SolveRequest(BaseModel):
    instance_path: str = "data/grid_20x20.json"
    n_trucks: int = 5
    n_bikes: int = 15
    max_iterations: int = 50000
    seed: int = 42


@router.post("/solve")
def start_solve(req: SolveRequest):
    runner = _get_runner()
    return runner.start(req.instance_path, req.n_trucks, req.n_bikes,
                        req.max_iterations, req.seed)


@router.post("/stop")
def stop_solve():
    runner = _get_runner()
    runner.stop()
    return {"status": "stopping"}
```

- [ ] **Step 4: Test server manually**

```bash
cd e:/2evrp && pip install fastapi uvicorn pydantic websockets
cd e:/2evrp && python -m uvicorn server.app:app --reload --port 8000
```

In another terminal:
```bash
curl http://localhost:8000/api/instances
# Expected: [{"name": "grid_20x20.json", "path": "data/grid_20x20.json"}]
```

- [ ] **Step 5: Commit**

```bash
git add server/
git commit -m "feat(dashboard): FastAPI server with WebSocket solver streaming"
```

---

## Phase 2: React Dashboard Shell + Overview Tab

### Task 3: Scaffold React App

**Files:**
- Create: `dashboard/package.json`
- Create: `dashboard/vite.config.ts`
- Create: `dashboard/tsconfig.json`
- Create: `dashboard/tailwind.config.ts`
- Create: `dashboard/index.html`
- Create: `dashboard/src/main.tsx`
- Create: `dashboard/src/App.tsx`
- Create: `dashboard/src/types.ts`

- [ ] **Step 1: Create Vite React project**

```bash
cd e:/2evrp && npm create vite@latest dashboard -- --template react-ts
cd e:/2evrp/dashboard && npm install
npm install tailwindcss @tailwindcss/vite
npm install recharts deck.gl @deck.gl/core @deck.gl/layers @deck.gl/react @luma.gl/core
npm install @types/react @types/react-dom
```

- [ ] **Step 2: Configure Tailwind + dark theme**

```typescript
// dashboard/vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
})
```

```css
/* dashboard/src/index.css */
@import "tailwindcss";

:root {
  --bg-primary: #0f1117;
  --bg-panel: #1a1d27;
  --bg-input: #374151;
  --text-primary: #e0e0e0;
  --text-secondary: #9ca3af;
  --text-dim: #6b7280;
  --truck: #3b82f6;
  --bike: #f97316;
  --satellite: #a855f7;
  --depot: #ef4444;
  --ok: #10b981;
  --warn: #f59e0b;
  --error: #ef4444;
  --fitness: #f59e0b;
}

body {
  margin: 0;
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
```

- [ ] **Step 3: Write types.ts — all interfaces**

```typescript
// dashboard/src/types.ts
export interface Stop {
  customer_id: number
  action: 'deliver' | 'reload'
  x: number
  y: number
  demand_g: number
  tw_open_s: number
  tw_close_s: number
  service_s: number
  restricted: boolean
}

export interface Route {
  vehicle_id: number
  vehicle_type: 'truck' | 'bike'
  n_stops: number
  stops: Stop[]
  total_load_g: number
  total_distance_m: number
}

export interface Satellite {
  customer_id: number
  bike_id: number
  truck_id: number
  transfer_g: number
  planned_time_s: number
}

export interface EvalResult {
  fitness: number
  cost: number
  sync_cost: number
  makespan: number
  total_penalty: number
  total_wait: number
  feasible: boolean
}

export interface Solution {
  truck_routes: Route[]
  bike_routes: Route[]
  satellites: Satellite[]
  depot: { x: number; y: number }
  summary: {
    n_trucks_used: number
    n_bikes_used: number
    n_satellites: number
    n_served: number
    n_customers: number
  }
}

export interface IterationUpdate {
  type: 'iteration'
  iter: number
  max_iter: number
  elapsed_s: number
  eval: EvalResult
  log_tail: {
    fitness: number[]
    best_fitness: number[]
    temperature: number[]
  }
}

export interface DoneUpdate {
  type: 'done'
  solution: Solution
  fitness: number
  elapsed_s: number
  log: Record<string, number[]>
  archive_size: number
}

export type SolverMessage = IterationUpdate | DoneUpdate | { type: 'stopped' } | { type: 'error'; message: string }

export type SolverState = 'idle' | 'running' | 'done' | 'error'

export interface SolverConfig {
  instance_path: string
  n_trucks: number
  n_bikes: number
  max_iterations: number
  seed: number
}
```

- [ ] **Step 4: Write App.tsx shell — ControlBar + TabBar + tab content**

```tsx
// dashboard/src/App.tsx
import { useState } from 'react'
import { useSolver } from './hooks/useSolver'
import { ControlBar } from './components/ControlBar'
import { TabBar } from './components/TabBar'
import { OverviewTab } from './components/overview/OverviewTab'

const TABS = ['Overview', 'Inspector', 'Dimensions', 'Giant Tour', 'Convergence', 'Pareto', 'Animation'] as const
type Tab = typeof TABS[number]

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('Overview')
  const solver = useSolver()

  return (
    <div className="h-screen flex flex-col bg-[var(--bg-primary)]">
      <ControlBar solver={solver} />
      <TabBar tabs={TABS} active={activeTab} onSelect={setActiveTab} />
      <div className="flex-1 overflow-hidden p-2">
        {activeTab === 'Overview' && <OverviewTab solver={solver} />}
        {activeTab !== 'Overview' && (
          <div className="flex items-center justify-center h-full text-[var(--text-dim)]">
            {activeTab} — coming soon
          </div>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Write useSolver hook**

```typescript
// dashboard/src/hooks/useSolver.ts
import { useState, useRef, useCallback, useEffect } from 'react'
import type { SolverState, SolverConfig, Solution, EvalResult, SolverMessage } from '../types'

const DEFAULT_CONFIG: SolverConfig = {
  instance_path: 'data/grid_20x20.json',
  n_trucks: 5,
  n_bikes: 15,
  max_iterations: 50000,
  seed: 42,
}

export function useSolver() {
  const [state, setState] = useState<SolverState>('idle')
  const [config, setConfig] = useState<SolverConfig>(DEFAULT_CONFIG)
  const [iter, setIter] = useState(0)
  const [maxIter, setMaxIter] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [eval_, setEval] = useState<EvalResult | null>(null)
  const [solution, setSolution] = useState<Solution | null>(null)
  const [log, setLog] = useState<Record<string, number[]>>({})
  const [logTail, setLogTail] = useState<{ fitness: number[]; best_fitness: number[]; temperature: number[] }>({
    fitness: [], best_fitness: [], temperature: [],
  })
  const wsRef = useRef<WebSocket | null>(null)

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`)
    ws.onmessage = (event) => {
      const msg: SolverMessage = JSON.parse(event.data)
      if (msg.type === 'iteration') {
        setIter(msg.iter)
        setMaxIter(msg.max_iter)
        setElapsed(msg.elapsed_s)
        setEval(msg.eval)
        setLogTail(msg.log_tail)
      } else if (msg.type === 'done') {
        setState('done')
        setSolution(msg.solution)
        setLog(msg.log)
        setEval({ fitness: msg.fitness, cost: 0, sync_cost: 0, makespan: 0, total_penalty: 0, total_wait: 0, feasible: true })
      } else if (msg.type === 'stopped') {
        setState('idle')
      } else if (msg.type === 'error') {
        setState('error')
      }
    }
    ws.onclose = () => setTimeout(connect, 2000)
    wsRef.current = ws
  }, [])

  useEffect(() => { connect(); return () => wsRef.current?.close() }, [connect])

  const start = useCallback(async () => {
    setState('running')
    setIter(0)
    setSolution(null)
    setLog({})
    await fetch('/api/solve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
  }, [config])

  const stop = useCallback(async () => {
    await fetch('/api/stop', { method: 'POST' })
  }, [])

  return { state, config, setConfig, iter, maxIter, elapsed, eval: eval_, solution, log, logTail, start, stop }
}

export type UseSolver = ReturnType<typeof useSolver>
```

- [ ] **Step 6: Write ControlBar + TabBar**

```tsx
// dashboard/src/components/ControlBar.tsx
import type { UseSolver } from '../hooks/useSolver'

export function ControlBar({ solver }: { solver: UseSolver }) {
  const { state, config, setConfig, iter, maxIter, elapsed, eval: ev, start, stop } = solver
  const pct = maxIter > 0 ? Math.round((iter / maxIter) * 100) : 0

  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-[var(--bg-panel)] border-b border-[var(--bg-input)] text-xs flex-wrap">
      <span className="text-sm font-bold text-white">2E-VRP</span>

      <label className="flex items-center gap-1 bg-[var(--bg-input)] px-2 py-1 rounded">
        <span className="text-[var(--text-dim)] text-[9px]">ITER</span>
        <input type="number" value={config.max_iterations} onChange={e => setConfig(c => ({...c, max_iterations: +e.target.value}))}
          className="bg-transparent text-[var(--text-primary)] w-16 outline-none" />
      </label>
      <label className="flex items-center gap-1 bg-[var(--bg-input)] px-2 py-1 rounded">
        <span className="text-[var(--text-dim)] text-[9px]">TRUCKS</span>
        <input type="number" value={config.n_trucks} onChange={e => setConfig(c => ({...c, n_trucks: +e.target.value}))}
          className="bg-transparent text-[var(--truck)] w-8 outline-none" />
      </label>
      <label className="flex items-center gap-1 bg-[var(--bg-input)] px-2 py-1 rounded">
        <span className="text-[var(--text-dim)] text-[9px]">BIKES</span>
        <input type="number" value={config.n_bikes} onChange={e => setConfig(c => ({...c, n_bikes: +e.target.value}))}
          className="bg-transparent text-[var(--bike)] w-8 outline-none" />
      </label>
      <label className="flex items-center gap-1 bg-[var(--bg-input)] px-2 py-1 rounded">
        <span className="text-[var(--text-dim)] text-[9px]">SEED</span>
        <input type="number" value={config.seed} onChange={e => setConfig(c => ({...c, seed: +e.target.value}))}
          className="bg-transparent text-[var(--text-primary)] w-12 outline-none" />
      </label>

      {state !== 'running' ? (
        <button onClick={start} className="bg-[var(--ok)] text-black px-4 py-1 rounded font-bold hover:brightness-110">RUN</button>
      ) : (
        <button onClick={stop} className="bg-[var(--error)] text-white px-4 py-1 rounded font-bold hover:brightness-110">STOP</button>
      )}

      <div className="flex items-center gap-2 ml-auto">
        {state === 'running' && <div className="w-2 h-2 rounded-full bg-[var(--ok)] animate-pulse" />}
        <span className="text-[var(--text-dim)]">{iter.toLocaleString()} / {maxIter.toLocaleString()}</span>
        <div className="w-24 h-1 bg-[var(--bg-input)] rounded">
          <div className="h-full bg-[var(--ok)] rounded transition-all" style={{ width: `${pct}%` }} />
        </div>
        {ev && <span className="text-[var(--fitness)] font-semibold">{ev.fitness.toLocaleString()}</span>}
        {ev && <span className={ev.feasible ? 'text-[var(--ok)]' : 'text-[var(--error)]'}>{ev.feasible ? 'feasible' : 'infeasible'}</span>}
        <span className="text-[var(--text-dim)]">{elapsed}s</span>
      </div>
    </div>
  )
}
```

```tsx
// dashboard/src/components/TabBar.tsx
export function TabBar<T extends string>({ tabs, active, onSelect }: { tabs: readonly T[]; active: T; onSelect: (t: T) => void }) {
  return (
    <div className="flex gap-0.5 px-2 py-1 bg-[var(--bg-panel)]">
      {tabs.map(tab => (
        <button key={tab} onClick={() => onSelect(tab)}
          className={`px-4 py-1.5 rounded text-xs transition-colors ${
            tab === active ? 'bg-[var(--truck)] text-white font-semibold' : 'text-[var(--text-dim)] hover:text-[var(--text-primary)]'
          }`}>
          {tab}
        </button>
      ))}
    </div>
  )
}
```

- [ ] **Step 7: Write Overview tab — QuickStats + ViolationsSummary + MiniConvergence + MiniOperators (placeholders for RouteMap)**

```tsx
// dashboard/src/components/overview/OverviewTab.tsx
import type { UseSolver } from '../../hooks/useSolver'
import { QuickStats } from './QuickStats'
import { ViolationsSummary } from './ViolationsSummary'
import { MiniConvergence } from './MiniConvergence'

export function OverviewTab({ solver }: { solver: UseSolver }) {
  return (
    <div className="h-full grid grid-cols-[3fr_2fr] gap-2">
      {/* Map placeholder */}
      <div className="bg-[#111827] rounded-lg flex items-center justify-center text-[var(--text-dim)]">
        Route Map — Task 4
      </div>
      {/* Right panels */}
      <div className="flex flex-col gap-2 overflow-auto">
        <QuickStats solver={solver} />
        <ViolationsSummary eval_={solver.eval} />
        <MiniConvergence logTail={solver.logTail} />
      </div>
    </div>
  )
}
```

```tsx
// dashboard/src/components/overview/QuickStats.tsx
import type { UseSolver } from '../../hooks/useSolver'

export function QuickStats({ solver }: { solver: UseSolver }) {
  const { eval: ev, solution } = solver
  const s = solution?.summary
  return (
    <div className="bg-[var(--bg-panel)] rounded-lg p-3 grid grid-cols-2 gap-2 text-xs">
      <div><span className="text-[var(--text-dim)]">Fitness</span><br/><span className="text-[var(--fitness)] text-lg font-bold">{ev?.fitness.toLocaleString() ?? '—'}</span></div>
      <div><span className="text-[var(--text-dim)]">Feasible</span><br/><span className={`text-lg font-bold ${ev?.feasible ? 'text-[var(--ok)]' : 'text-[var(--error)]'}`}>{ev ? (ev.feasible ? 'YES' : 'NO') : '—'}</span></div>
      <div><span className="text-[var(--text-dim)]">Cost</span><br/><span className="text-[var(--truck)] font-semibold">{ev ? `${(ev.cost / 1000).toFixed(1)} km` : '—'}</span></div>
      <div><span className="text-[var(--text-dim)]">Makespan</span><br/><span className="text-[var(--bike)] font-semibold">{ev ? `${Math.floor(ev.makespan / 3600)}h ${Math.floor((ev.makespan % 3600) / 60)}m` : '—'}</span></div>
      <div><span className="text-[var(--text-dim)]">Trucks</span><br/><span className="text-[var(--truck)]">{s ? `${s.n_trucks_used}` : '—'}</span></div>
      <div><span className="text-[var(--text-dim)]">Bikes</span><br/><span className="text-[var(--bike)]">{s ? `${s.n_bikes_used}` : '—'}</span></div>
      <div><span className="text-[var(--text-dim)]">Satellites</span><br/><span className="text-[var(--satellite)]">{s?.n_satellites ?? '—'}</span></div>
      <div><span className="text-[var(--text-dim)]">Customers</span><br/><span>{s ? `${s.n_served}/${s.n_customers}` : '—'}</span></div>
    </div>
  )
}
```

```tsx
// dashboard/src/components/overview/ViolationsSummary.tsx
import type { EvalResult } from '../../types'

export function ViolationsSummary({ eval_ }: { eval_: EvalResult | null }) {
  const penalty = eval_?.total_penalty ?? 0
  const ok = penalty === 0
  return (
    <div className="bg-[var(--bg-panel)] rounded-lg p-3 text-xs">
      <div className="text-[var(--text-dim)] mb-1 text-[9px]">VIOLATIONS</div>
      <div className={`text-sm font-bold ${ok ? 'text-[var(--ok)]' : 'text-[var(--error)]'}`}>
        {ok ? 'All clear' : `Penalty: ${penalty.toLocaleString()}`}
      </div>
    </div>
  )
}
```

```tsx
// dashboard/src/components/overview/MiniConvergence.tsx
import { LineChart, Line, ResponsiveContainer } from 'recharts'

interface Props {
  logTail: { fitness: number[]; best_fitness: number[] }
}

export function MiniConvergence({ logTail }: Props) {
  const data = logTail.fitness.map((f, i) => ({ f, b: logTail.best_fitness[i] ?? f }))
  return (
    <div className="bg-[var(--bg-panel)] rounded-lg p-3 flex-1">
      <div className="text-[var(--text-dim)] text-[9px] mb-1">CONVERGENCE</div>
      <ResponsiveContainer width="100%" height={80}>
        <LineChart data={data}>
          <Line type="monotone" dataKey="f" stroke="var(--fitness)" dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey="b" stroke="var(--ok)" dot={false} strokeWidth={1} strokeDasharray="4 2" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
```

- [ ] **Step 8: Write main.tsx entry point**

```tsx
// dashboard/src/main.tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
)
```

- [ ] **Step 9: Verify dashboard renders**

```bash
cd e:/2evrp/dashboard && npm run dev
```

Open http://localhost:5173 — should see dark dashboard with control bar, tabs, and overview tab with stats panels.

- [ ] **Step 10: Commit**

```bash
cd e:/2evrp && git add dashboard/
git commit -m "feat(dashboard): React app shell with control bar, tabs, overview panels"
```

---

### Task 4: Route Map with Deck.gl

**Files:**
- Create: `dashboard/src/components/overview/RouteMap.tsx`
- Modify: `dashboard/src/components/overview/OverviewTab.tsx`

- [ ] **Step 1: Write RouteMap component**

```tsx
// dashboard/src/components/overview/RouteMap.tsx
import { useMemo } from 'react'
import DeckGL from '@deck.gl/react'
import { OrthographicView } from '@deck.gl/core'
import { PathLayer, ScatterplotLayer, TextLayer } from '@deck.gl/layers'
import type { Solution } from '../../types'

const TRUCK_COLORS = [
  [59, 130, 246], [96, 165, 250], [37, 99, 235], [29, 78, 216],
  [147, 197, 253], [191, 219, 254], [59, 130, 246], [96, 165, 250],
] as const

const BIKE_COLORS = [
  [249, 115, 22], [251, 146, 60], [234, 88, 12], [194, 65, 12],
  [253, 186, 116], [254, 215, 170], [249, 115, 22], [251, 146, 60],
] as const

const SAT_COLOR = [168, 85, 247]
const DEPOT_COLOR = [239, 68, 68]

interface Props {
  solution: Solution | null
}

export function RouteMap({ solution }: Props) {
  const layers = useMemo(() => {
    if (!solution) return []
    const { truck_routes, bike_routes, satellites, depot } = solution

    const truckPaths = truck_routes
      .filter(r => r.n_stops > 0)
      .map((r, i) => ({
        path: [
          [depot.x, depot.y],
          ...r.stops.map(s => [s.x, s.y]),
          [depot.x, depot.y],
        ],
        color: TRUCK_COLORS[i % TRUCK_COLORS.length],
      }))

    const bikePaths = bike_routes
      .filter(r => r.n_stops > 0)
      .map((r, i) => ({
        path: [
          [depot.x, depot.y],
          ...r.stops.map(s => [s.x, s.y]),
          [depot.x, depot.y],
        ],
        color: BIKE_COLORS[i % BIKE_COLORS.length],
      }))

    const allStops = [...truck_routes, ...bike_routes].flatMap(r =>
      r.stops.filter(s => s.action === 'deliver').map(s => ({
        position: [s.x, s.y],
        id: s.customer_id,
        demand: s.demand_g,
        tw: `${Math.floor(s.tw_open_s / 60)}-${Math.floor(s.tw_close_s / 60)}`,
      }))
    )

    const satPoints = satellites.map(s => {
      const stop = allStops.find(st => st.id === s.customer_id)
      return { position: stop ? stop.position : [0, 0], ...s }
    })

    return [
      new PathLayer({
        id: 'truck-routes',
        data: truckPaths,
        getPath: d => d.path,
        getColor: d => [...d.color, 180],
        getWidth: 0.3,
        widthMinPixels: 2,
      }),
      new PathLayer({
        id: 'bike-routes',
        data: bikePaths,
        getPath: d => d.path,
        getColor: d => [...d.color, 200],
        getWidth: 0.15,
        widthMinPixels: 1,
        getDashArray: [4, 2],
        dashJustified: true,
        extensions: [],
      }),
      new ScatterplotLayer({
        id: 'customers',
        data: allStops,
        getPosition: d => d.position,
        getRadius: 0.3,
        getFillColor: [107, 114, 128],
        getLineColor: [156, 163, 175],
        stroked: true,
        lineWidthMinPixels: 1,
        radiusMinPixels: 4,
        pickable: true,
      }),
      new ScatterplotLayer({
        id: 'satellites',
        data: satPoints,
        getPosition: d => d.position,
        getRadius: 0.4,
        getFillColor: SAT_COLOR,
        radiusMinPixels: 6,
      }),
      new ScatterplotLayer({
        id: 'depot',
        data: [{ position: [depot.x, depot.y] }],
        getPosition: d => d.position,
        getRadius: 0.5,
        getFillColor: DEPOT_COLOR,
        radiusMinPixels: 8,
      }),
      new TextLayer({
        id: 'customer-labels',
        data: allStops,
        getPosition: d => d.position,
        getText: d => String(d.id),
        getSize: 10,
        getColor: [200, 200, 200],
        getTextAnchor: 'middle',
        getAlignmentBaseline: 'bottom',
        getPixelOffset: [0, -8],
      }),
    ]
  }, [solution])

  return (
    <div className="bg-[#111827] rounded-lg relative overflow-hidden h-full">
      {solution ? (
        <DeckGL
          views={new OrthographicView()}
          initialViewState={{ target: [solution.depot.x, solution.depot.y, 0], zoom: 3 }}
          controller={true}
          layers={layers}
          getTooltip={({ object }: any) => object?.id !== undefined ? `#${object.id} | ${object.demand}g | TW: ${object.tw}` : null}
        />
      ) : (
        <div className="flex items-center justify-center h-full text-[var(--text-dim)]">
          Run solver to see routes
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Wire RouteMap into OverviewTab**

Replace the map placeholder in `OverviewTab.tsx`:

```tsx
// dashboard/src/components/overview/OverviewTab.tsx
import type { UseSolver } from '../../hooks/useSolver'
import { RouteMap } from './RouteMap'
import { QuickStats } from './QuickStats'
import { ViolationsSummary } from './ViolationsSummary'
import { MiniConvergence } from './MiniConvergence'

export function OverviewTab({ solver }: { solver: UseSolver }) {
  return (
    <div className="h-full grid grid-cols-[3fr_2fr] gap-2">
      <RouteMap solution={solver.solution} />
      <div className="flex flex-col gap-2 overflow-auto">
        <QuickStats solver={solver} />
        <ViolationsSummary eval_={solver.eval} />
        <MiniConvergence logTail={solver.logTail} />
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Verify map renders with solver output**

Start server + dashboard, run solver, verify:
- Customer dots with ID labels appear
- Truck routes (solid blue) and bike routes (dashed orange) render
- Depot (red) and satellites (purple) visible
- Hover shows tooltip with customer info
- Zoom/pan works

- [ ] **Step 4: Commit**

```bash
cd e:/2evrp && git add dashboard/
git commit -m "feat(dashboard): Deck.gl route map with customer IDs, tooltips, route layers"
```

---

## Phase 3: Route Inspector + Dimensions (separate plan)

Tasks 5-8: VehicleSelector, RouteScheduleChart (PyVRP-style), StopTable, GanttChart, LoadCapacityBars, TWSlackTable, DistanceDistribution.

## Phase 4: Convergence + Operators (separate plan)

Tasks 9-11: FitnessChart, OperatorWeights, AcceptRejectChart.

## Phase 5: Pareto + Giant Tour + Animation (separate plan)

Tasks 12-14: ParetoScatter, GiantTourTab, TripAnimation (Deck.gl TripsLayer).

---

Each phase ships independently. Phase 1+2 gives you a working dashboard with live solver, route map, stats, and convergence. Phases 3-5 add depth.
