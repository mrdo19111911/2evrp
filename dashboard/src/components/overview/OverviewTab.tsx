import type { UseSolver } from '../../hooks/useSolver'
import { QuickStats } from './QuickStats'
import { ViolationsSummary } from './ViolationsSummary'
import { MiniConvergence } from './MiniConvergence'
import { RouteMap } from './RouteMap'

export function OverviewTab({ solver }: { solver: UseSolver }) {
  return (
    <div className="h-full grid grid-cols-[3fr_2fr] gap-2">
      <RouteMap solution={solver.solution} instance={solver.instance} />
      {/* Right panels */}
      <div className="flex flex-col gap-2 overflow-auto">
        <QuickStats solver={solver} />
        <ViolationsSummary eval_={solver.eval} />
        <MiniConvergence logTail={solver.logTail} />
      </div>
    </div>
  )
}
