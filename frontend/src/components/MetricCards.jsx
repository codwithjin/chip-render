import useSwingStore from '../store/useSwingStore'

// Metrics to show per phase, with pro benchmarks
const PHASE_METRICS = {
  P1: [
    { key: 'forward_bend', name: 'Forward Bend', unit: '°', lo: 32, hi: 45 },
    { key: 'shaft_angle', name: 'Shaft Angle', unit: '°', lo: 55,  hi: 65  },
    { key: 'lead_knee',   name: 'Lead Knee',   unit: '°', lo: 145, hi: 165 },
    { key: 'trail_elbow', name: 'Trail Elbow', unit: '°', lo: 155, hi: 175 },
  ],
  P2: [
    { key: 'shoulder_turn', name: 'Shoulder Turn',  unit: '°', lo: 10,  hi: 25  },
    { key: 'hip_turn',      name: 'Hip Turn',       unit: '°', lo: 5,   hi: 15  },
    { key: 'trail_elbow',   name: 'Trail Elbow',    unit: '°', lo: 155, hi: 175 },
  ],
  P3: [
    { key: 'shoulder_turn', name: 'Shoulder Turn',  unit: '°', lo: 45,  hi: 60  },
    { key: 'hip_turn',      name: 'Hip Turn',       unit: '°', lo: 20,  hi: 35  },
    { key: 'x_factor',      name: 'X-Factor',       unit: '°', lo: 20,  hi: 35  },
    { key: 'trail_elbow',   name: 'Trail Elbow',    unit: '°', lo: 85,  hi: 110 },
  ],
  P4: [
    { key: 'shoulder_turn', name: 'Shoulder Turn',  unit: '°', lo: 85,  hi: 110 },
    { key: 'hip_turn',      name: 'Hip Turn',       unit: '°', lo: 40,  hi: 55  },
    { key: 'x_factor',      name: 'X-Factor',       unit: '°', lo: 40,  hi: 55  },
    { key: 'trail_elbow',   name: 'Trail Elbow',    unit: '°', lo: 80,  hi: 100 },
  ],
  P5: [
    { key: 'hip_turn',      name: 'Hip Turn',       unit: '°', lo: 50,  hi: 65  },
    { key: 'shoulder_turn', name: 'Shoulder Turn',  unit: '°', lo: 75,  hi: 95  },
    { key: 'x_factor',      name: 'X-Factor',       unit: '°', lo: 15,  hi: 35  },
  ],
  P6: [
    { key: 'hip_turn',      name: 'Hip Turn',       unit: '°', lo: 50,  hi: 65  },
    { key: 'shoulder_turn', name: 'Shoulder Turn',  unit: '°', lo: 45,  hi: 65  },
    { key: 'lead_elbow',    name: 'Lead Elbow',     unit: '°', lo: 155, hi: 175 },
  ],
  P7: [
    { key: 'hip_turn',      name: 'Hip Turn',       unit: '°', lo: 30,  hi: 50  },
    { key: 'shoulder_turn', name: 'Shoulder Turn',  unit: '°', lo: 15,  hi: 35  },
    { key: 'lead_knee',     name: 'Lead Knee',      unit: '°', lo: 150, hi: 175 },
  ],
}

function classify(value, lo, hi) {
  if (value === null || value === undefined) return 'none'
  return value >= lo && value <= hi ? 'pass' : 'fail'
}

const STATUS = {
  pass: { color: '#22c55e', label: 'PASS' },
  fail: { color: '#ef4444', label: 'FAIL' },
  none: { color: '#4b5563', label: '—'    },
}

export default function MetricCards() {
  const { activePhase, metrics } = useSwingStore()
  const defs    = PHASE_METRICS[activePhase] || []
  const values  = metrics?.[activePhase] || {}

  if (!defs.length) {
    return (
      <div className="p-3 text-xs text-gray-600 text-center">
        No metrics for {activePhase}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-1 p-2">
      {defs.map(({ key, name, unit, lo, hi }) => {
        const val    = values[key]
        const status = classify(val, lo, hi)
        const { color, label } = STATUS[status]
        return (
          <div
            key={key}
            className="flex items-center justify-between px-2 py-1.5 rounded border border-gray-800 bg-[#0f1410]"
            style={{ borderLeft: `3px solid ${color}` }}
          >
            <div className="flex flex-col min-w-0">
              <span className="text-xs text-gray-300 truncate">{name}</span>
              <span className="text-xs font-mono text-gray-500">
                {val !== undefined && val !== null
                  ? <>{val}{unit} <span className="text-gray-700">({lo}–{hi})</span></>
                  : <span className="text-gray-700">{lo}–{hi}{unit}</span>
                }
              </span>
            </div>
            <span className="text-xs font-bold ml-2 shrink-0" style={{ color }}>
              {label}
            </span>
          </div>
        )
      })}
    </div>
  )
}
