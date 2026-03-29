export interface Stop {
  customer_id: number
  action: string
  x: number
  y: number
  demand_g: number
}

export interface Route {
  vehicle_id: number
  stops: Stop[]
  load_g: number
  distance_m: number
}

export interface Satellite {
  customer_id: number
  bike_id: number
  truck_id: number
  load_g: number
  time_s: number
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
    n_customers: number
  }
}

export interface CustomerPoint {
  id: number
  x: number
  y: number
  demand_g: number
  tw_open_s: number
  tw_close_s: number
  service_s: number
  restricted: boolean
}

export interface InstanceData {
  type: 'instance_loaded'
  depot: { x: number; y: number }
  customers: CustomerPoint[]
  n_customers: number
}

export interface IterationUpdate {
  type: 'iteration'
  iter: number
  max_iter: number
  pct: number
  elapsed_s: number
  eval: EvalResult
  log_tail: {
    fitness: number[]
    best_fitness: number[]
    temperature: number[]
  }
}

export interface SolutionUpdate {
  type: 'solution_update'
  iter: number
  max_iter: number
  pct: number
  elapsed_s: number
  eval: EvalResult
  solution: Solution
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

export interface StatusUpdate {
  type: 'status'
  state: string
  message: string
}

export type SolverMessage =
  | InstanceData
  | IterationUpdate
  | SolutionUpdate
  | DoneUpdate
  | StatusUpdate
  | { type: 'stopped' }
  | { type: 'error'; message: string }

export type SolverState = 'idle' | 'loading' | 'running' | 'done' | 'error'

export interface SolverConfig {
  instance_path: string
  n_trucks: number
  n_bikes: number
  max_iterations: number
  seed: number
}
