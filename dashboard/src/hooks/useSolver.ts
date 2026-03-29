import { useState, useRef, useCallback, useEffect } from 'react'
import type { SolverState, SolverConfig, Solution, EvalResult, SolverMessage, CustomerPoint } from '../types'

const DEFAULT_CONFIG: SolverConfig = {
  instance_path: 'data/grid_20x20.json',
  n_trucks: 5,
  n_bikes: 15,
  max_iterations: 50000,
  seed: 42,
}

export function useSolver() {
  const [state, setState] = useState<SolverState>('idle')
  const [statusMessage, setStatusMessage] = useState('')
  const [config, setConfig] = useState<SolverConfig>(DEFAULT_CONFIG)
  const [iter, setIter] = useState(0)
  const [maxIter, setMaxIter] = useState(0)
  const [pct, setPct] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [eval_, setEval] = useState<EvalResult | null>(null)
  const [solution, setSolution] = useState<Solution | null>(null)
  const [instance, setInstance] = useState<{ depot: { x: number; y: number }; customers: CustomerPoint[] } | null>(null)
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

      if (msg.type === 'instance_loaded') {
        setInstance({ depot: msg.depot, customers: msg.customers })
        setStatusMessage(`Loaded ${msg.n_customers} customers`)
      } else if (msg.type === 'status') {
        setState('loading')
        setStatusMessage(msg.message)
      } else if (msg.type === 'iteration') {
        setState('running')
        setIter(msg.iter)
        setMaxIter(msg.max_iter)
        setPct(msg.pct)
        setElapsed(msg.elapsed_s)
        setEval(msg.eval)
        setLogTail(msg.log_tail)
      } else if (msg.type === 'solution_update') {
        setState('running')
        setIter(msg.iter)
        setMaxIter(msg.max_iter)
        setPct(msg.pct)
        setElapsed(msg.elapsed_s)
        setEval(msg.eval)
        setSolution(msg.solution)
        setLogTail(msg.log_tail)
      } else if (msg.type === 'done') {
        setState('done')
        setSolution(msg.solution)
        setLog(msg.log)
        setPct(100)
        setStatusMessage('Done')
      } else if (msg.type === 'stopped') {
        setState('idle')
        setStatusMessage('Stopped')
      } else if (msg.type === 'error') {
        setState('error')
        setStatusMessage(msg.message)
      }
    }
    ws.onclose = () => setTimeout(connect, 2000)
    wsRef.current = ws
  }, [])

  useEffect(() => { connect(); return () => wsRef.current?.close() }, [connect])

  // Load instance on mount so map shows customers before RUN
  const loadInstance = useCallback(async (path: string) => {
    try {
      const resp = await fetch(`/api/instance?path=${encodeURIComponent(path)}`)
      if (!resp.ok) return
      const data = await resp.json()
      setInstance({ depot: data.depot, customers: data.customers })
      setStatusMessage(`${data.n_customers} customers loaded`)
    } catch { /* server not running yet, ignore */ }
  }, [])

  useEffect(() => { loadInstance(config.instance_path) }, [config.instance_path, loadInstance])

  const start = useCallback(async () => {
    setState('loading')
    setIter(0)
    setPct(0)
    setSolution(null)
    setLog({})
    setStatusMessage('Loading instance...')
    await fetch('/api/solve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
  }, [config])

  const stop = useCallback(async () => {
    await fetch('/api/stop', { method: 'POST' })
  }, [])

  return {
    state, statusMessage, config, setConfig,
    iter, maxIter, pct, elapsed,
    eval: eval_, solution, instance, log, logTail,
    start, stop,
  }
}

export type UseSolver = ReturnType<typeof useSolver>
