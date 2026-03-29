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
