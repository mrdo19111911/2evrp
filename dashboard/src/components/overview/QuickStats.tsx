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
      <div><span className="text-[var(--text-dim)]">Customers</span><br/><span>{s ? `${s.n_customers}` : '—'}</span></div>
    </div>
  )
}
