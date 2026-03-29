import { useMemo } from 'react'
import DeckGL from '@deck.gl/react'
import { OrthographicView } from '@deck.gl/core'
import { PathLayer, ScatterplotLayer, TextLayer } from '@deck.gl/layers'
import type { Solution, Stop, CustomerPoint } from '../../types'

const TRUCK_COLORS: [number, number, number][] = [
  [59, 130, 246], [96, 165, 250], [37, 99, 235], [29, 78, 216], [147, 197, 253],
]
const BIKE_COLORS: [number, number, number][] = [
  [249, 115, 22], [251, 146, 60], [234, 88, 12], [194, 65, 12], [253, 186, 116],
]
const SATELLITE_COLOR: [number, number, number] = [168, 85, 247]
const DEPOT_COLOR: [number, number, number] = [239, 68, 68]
const CUSTOMER_COLOR: [number, number, number] = [107, 114, 128]
const CUSTOMER_SERVED: [number, number, number] = [156, 163, 175]

/** Layers for instance-only view (customers + depot, no routes yet) */
function buildInstanceLayers(depot: { x: number; y: number }, customers: CustomerPoint[]) {
  return [
    new ScatterplotLayer({
      id: 'customers',
      data: customers,
      getPosition: (d: CustomerPoint) => [d.x, d.y],
      getFillColor: CUSTOMER_COLOR,
      getRadius: 0.35,
      radiusUnits: 'common' as const,
      pickable: true,
    }),
    new ScatterplotLayer({
      id: 'depot',
      data: [depot],
      getPosition: (d: { x: number; y: number }) => [d.x, d.y],
      getFillColor: DEPOT_COLOR,
      getRadius: 0.8,
      radiusUnits: 'common' as const,
    }),
    new TextLayer({
      id: 'customer-labels',
      data: customers,
      getPosition: (d: CustomerPoint) => [d.x, d.y],
      getText: (d: CustomerPoint) => String(d.id),
      getSize: 10,
      getColor: [180, 180, 180],
      getPixelOffset: [0, -12],
      fontFamily: 'monospace',
    }),
    new TextLayer({
      id: 'depot-label',
      data: [depot],
      getPosition: (d: { x: number; y: number }) => [d.x, d.y],
      getText: () => 'DEPOT',
      getSize: 11,
      getColor: DEPOT_COLOR,
      getPixelOffset: [0, 14],
      fontFamily: 'monospace',
      fontWeight: 700,
    }),
  ]
}

/** Full layers with routes + customers */
function buildSolutionLayers(sol: Solution) {
  const depot = sol.depot

  const customers: Stop[] = []
  const seen = new Set<number>()
  for (const r of [...sol.truck_routes, ...sol.bike_routes]) {
    for (const s of r.stops) {
      if (s.action === 'deliver' && !seen.has(s.customer_id)) {
        seen.add(s.customer_id)
        customers.push(s)
      }
    }
  }

  const satMap = new Map<number, { x: number; y: number }>()
  for (const r of [...sol.truck_routes, ...sol.bike_routes]) {
    for (const s of r.stops) {
      if (s.action === 'reload' && !satMap.has(s.customer_id)) {
        satMap.set(s.customer_id, { x: s.x, y: s.y })
      }
    }
  }
  const sats = [...satMap.entries()].map(([id, c]) => ({ id, ...c }))

  const truckPaths = sol.truck_routes
    .filter(r => r.stops.length > 0)
    .map((r, i) => ({
      path: [[depot.x, depot.y], ...r.stops.map(s => [s.x, s.y]), [depot.x, depot.y]],
      color: TRUCK_COLORS[i % TRUCK_COLORS.length],
    }))

  const bikePaths = sol.bike_routes
    .filter(r => r.stops.length > 0)
    .map((r, i) => ({
      path: [[depot.x, depot.y], ...r.stops.map(s => [s.x, s.y]), [depot.x, depot.y]],
      color: BIKE_COLORS[i % BIKE_COLORS.length],
    }))

  return [
    new PathLayer({
      id: 'truck-routes',
      data: truckPaths,
      getPath: d => d.path,
      getColor: d => d.color,
      getWidth: 0.3,
      widthUnits: 'common' as const,
      jointRounded: true,
    }),
    new PathLayer({
      id: 'bike-routes',
      data: bikePaths,
      getPath: d => d.path,
      getColor: d => d.color,
      getWidth: 0.2,
      widthUnits: 'common' as const,
      jointRounded: true,
    }),
    new ScatterplotLayer({
      id: 'customers',
      data: customers,
      getPosition: (d: Stop) => [d.x, d.y],
      getFillColor: CUSTOMER_SERVED,
      getRadius: 0.35,
      radiusUnits: 'common' as const,
      pickable: true,
    }),
    new ScatterplotLayer({
      id: 'satellites',
      data: sats,
      getPosition: (d: { x: number; y: number }) => [d.x, d.y],
      getFillColor: SATELLITE_COLOR,
      getRadius: 0.6,
      radiusUnits: 'common' as const,
    }),
    new ScatterplotLayer({
      id: 'depot',
      data: [depot],
      getPosition: (d: { x: number; y: number }) => [d.x, d.y],
      getFillColor: DEPOT_COLOR,
      getRadius: 0.8,
      radiusUnits: 'common' as const,
    }),
    new TextLayer({
      id: 'customer-labels',
      data: customers,
      getPosition: (d: Stop) => [d.x, d.y],
      getText: (d: Stop) => String(d.customer_id),
      getSize: 10,
      getColor: [255, 255, 255],
      getPixelOffset: [0, -12],
      fontFamily: 'monospace',
    }),
  ]
}

interface Props {
  solution: Solution | null
  instance: { depot: { x: number; y: number }; customers: CustomerPoint[] } | null
}

export function RouteMap({ solution, instance }: Props) {
  const layers = useMemo(() => {
    if (solution) return buildSolutionLayers(solution)
    if (instance) return buildInstanceLayers(instance.depot, instance.customers)
    return []
  }, [solution, instance])

  const center = solution?.depot ?? instance?.depot

  // Compute zoom from data extent
  const zoom = useMemo(() => {
    const pts = instance?.customers ?? []
    if (pts.length === 0) return -4
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
    for (const p of pts) {
      if (p.x < minX) minX = p.x
      if (p.x > maxX) maxX = p.x
      if (p.y < minY) minY = p.y
      if (p.y > maxY) maxY = p.y
    }
    const range = Math.max(maxX - minX, maxY - minY, 1)
    // OrthographicView zoom: 2^zoom pixels per unit. We want ~600px for the range.
    return Math.log2(600 / range)
  }, [instance])

  if (!center) {
    return (
      <div className="bg-[#111827] rounded-lg flex items-center justify-center h-full text-[var(--text-dim)]">
        Press RUN to load instance
      </div>
    )
  }

  return (
    <div className="bg-[#111827] rounded-lg h-full relative overflow-hidden">
      <DeckGL
        views={new OrthographicView({ id: 'ortho' })}
        initialViewState={{
          target: [center.x, center.y, 0],
          zoom,
        }}
        controller={true}
        layers={layers}
        getTooltip={({ object }: { object?: Record<string, unknown> }) => {
          if (!object) return null
          if ('customer_id' in object) {
            const s = object as unknown as Stop
            return `#${s.customer_id} | ${s.demand_g}g`
          }
          if ('id' in object) {
            const c = object as unknown as CustomerPoint
            const twOpen = Math.floor(c.tw_open_s / 60)
            const twClose = Math.floor(c.tw_close_s / 60)
            return `#${c.id} | ${c.demand_g}g | TW: ${twOpen}-${twClose}min${c.restricted ? ' | BIKE ONLY' : ''}`
          }
          return null
        }}
        style={{ position: 'absolute', top: '0', left: '0', right: '0', bottom: '0' }}
      />
    </div>
  )
}
