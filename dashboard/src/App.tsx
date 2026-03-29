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
