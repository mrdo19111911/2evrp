import type { UseSolver } from '../hooks/useSolver'

export function ControlBar({ solver }: { solver: UseSolver }) {
  const { state, statusMessage, config, setConfig, iter, maxIter, pct, elapsed, eval: ev, start, stop } = solver

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

      {state === 'idle' || state === 'done' || state === 'error' ? (
        <button onClick={start} className="bg-[var(--ok)] text-black px-4 py-1 rounded font-bold hover:brightness-110">RUN</button>
      ) : (
        <button onClick={stop} className="bg-[var(--error)] text-white px-4 py-1 rounded font-bold hover:brightness-110">STOP</button>
      )}

      <div className="flex items-center gap-2 ml-auto">
        {state === 'loading' && (
          <>
            <div className="w-2 h-2 rounded-full bg-[var(--warn)] animate-pulse" />
            <span className="text-[var(--warn)]">{statusMessage}</span>
          </>
        )}
        {state === 'running' && (
          <>
            <div className="w-2 h-2 rounded-full bg-[var(--ok)] animate-pulse" />
            <span className="text-[var(--text-dim)]">{iter.toLocaleString()} / {maxIter.toLocaleString()}</span>
            <span className="text-[var(--ok)] font-bold">{pct}%</span>
            <div className="w-32 h-1.5 bg-[var(--bg-input)] rounded overflow-hidden">
              <div className="h-full bg-[var(--ok)] rounded transition-all duration-300" style={{ width: `${pct}%` }} />
            </div>
          </>
        )}
        {state === 'done' && (
          <>
            <div className="w-2 h-2 rounded-full bg-[var(--ok)]" />
            <span className="text-[var(--ok)]">Done</span>
          </>
        )}
        {ev && <span className="text-[var(--fitness)] font-semibold">{ev.fitness.toLocaleString()}</span>}
        {ev && <span className={ev.feasible ? 'text-[var(--ok)]' : 'text-[var(--error)]'}>{ev.feasible ? 'feasible' : 'infeasible'}</span>}
        {elapsed > 0 && <span className="text-[var(--text-dim)]">{elapsed}s</span>}
      </div>
    </div>
  )
}
