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
