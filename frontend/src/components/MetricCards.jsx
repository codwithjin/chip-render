import useSwingStore from '../store/useSwingStore'

const RESULT_STYLES = {
  PASS:          { color: '#22c55e', label: 'PASS' },
  FAIL:          { color: '#ef4444', label: 'FAIL' },
  LOW_CONFIDENCE:{ color: '#f59e0b', label: 'LOW' },
  CAMERA_LIMIT:  { color: '#6b7280', label: 'CAM' },
  'GEMINI FLASH':{ color: '#8b5cf6', label: 'AI' },
}

// Hardcoded benchmarks from swing audit
const BENCHMARKS = {
  P1: [
    { name: 'Spine Angle',       unit: '°',  lo: 38, hi: 42  },
    { name: 'Trail Elbow',       unit: '°',  lo: 160, hi: 170 },
    { name: 'Trail Knee Flex',   unit: '°',  lo: 20, hi: 25  },
    { name: 'Lead Knee Flex',    unit: '°',  lo: 20, hi: 25  },
  ],
  P3: [
    { name: 'Shoulder Turn',     unit: '°',  lo: 50, hi: 56  },
    { name: 'Hip Turn',          unit: '°',  lo: 25, hi: 30  },
    { name: 'X-Factor',          unit: '°',  lo: 25, hi: 35  },
    { name: 'Trail Elbow',       unit: '°',  lo: 100, hi: 110 },
  ],
  P4: [
    { name: 'Shoulder Turn',     unit: '°',  lo: 90, hi: 100 },
    { name: 'Hip Turn',          unit: '°',  lo: 45, hi: 55  },
    { name: 'X-Factor',          unit: '°',  lo: 45, hi: 55  },
    { name: 'Trail Elbow',       unit: '°',  lo: 85, hi: 100 },
  ],
  P5: [
    { name: 'Hip Turn',          unit: '°',  lo: 55, hi: 65  },
    { name: 'Shoulder Turn',     unit: '°',  lo: 80, hi: 90  },
  ],
  P7: [
    { name: 'Hip Turn',          unit: '°',  lo: 35, hi: 45  },
    { name: 'Shoulder Turn',     unit: '°',  lo: 20, hi: 35  },
  ],
}

export default function MetricCards() {
  const { activePhase, metrics } = useSwingStore()
  const phaseMetrics = metrics[activePhase] || BENCHMARKS[activePhase] || []

  if (!phaseMetrics.length) {
    return (
      <div className="p-3 text-xs text-gray-600 text-center">
        No metrics for {activePhase}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-1 p-2 overflow-y-auto">
      {phaseMetrics.map((m, i) => {
        const style  = RESULT_STYLES[m.result] || RESULT_STYLES.LOW_CONFIDENCE
        const hasVal = m.value !== null && m.value !== undefined
        return (
          <div key={i}
            className="flex items-center justify-between px-2 py-1.5 rounded border border-gray-800 bg-[#0f1410]"
            style={{ borderLeft: `3px solid ${style.color}` }}
          >
            <div className="flex flex-col">
              <span className="text-xs text-gray-300">{m.name}</span>
              {hasVal && (
                <span className="text-xs font-mono text-gray-500">
                  {m.value}{m.unit}
                  {m.lo !== undefined && (
                    <span className="text-gray-700 ml-1">({m.lo}–{m.hi})</span>
                  )}
                </span>
              )}
            </div>
            <span className="text-xs font-bold" style={{ color: style.color }}>
              {style.label}
            </span>
          </div>
        )
      })}
    </div>
  )
}
