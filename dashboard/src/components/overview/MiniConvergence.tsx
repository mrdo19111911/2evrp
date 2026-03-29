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
