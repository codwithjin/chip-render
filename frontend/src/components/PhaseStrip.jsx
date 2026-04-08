import useSwingStore from '../store/useSwingStore'

const PHASE_ORDER  = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']
const PHASE_LABELS = {
  P1: 'Address', P2: 'Takeaway', P3: 'Halfway',
  P4: 'Top', P5: 'Transition', P6: 'Pre-Impact', P7: 'Impact',
}
const PHASE_COLORS = {
  P1: '#888899', P2: '#009999', P3: '#996600',
  P4: '#6644bb', P5: '#aa0099', P6: '#998800', P7: '#009944',
}
const FREEZE_PHASES = new Set(['P1', 'P3', 'P4', 'P5', 'P7'])

export default function PhaseStrip({ videoRef }) {
  const { phases, activePhase, setActivePhase } = useSwingStore()

  const jumpToPhase = (key) => {
    const phase = phases[key]
    const video = videoRef?.current
    if (!phase?.detected || !video) return
    video.currentTime = phase.time_s
    video.pause()
    setActivePhase(key)
  }

  return (
    <div className="flex border-t border-gray-800">
      {PHASE_ORDER.map(key => {
        const phase    = phases[key]
        const detected = phase?.detected
        const active   = key === activePhase
        const color    = PHASE_COLORS[key]
        const hasFree  = FREEZE_PHASES.has(key)

        return (
          <button
            key={key}
            onClick={() => jumpToPhase(key)}
            disabled={!detected}
            className="flex-1 py-2 text-center text-xs transition-colors border-r border-gray-800 last:border-r-0"
            style={{
              background:  active ? color : 'transparent',
              color:       active ? '#fff' : detected ? color : '#444',
              opacity:     detected ? 1 : 0.35,
              borderBottom: active ? `2px solid ${color}` : '2px solid transparent',
            }}
          >
            <div className="font-bold">{key}</div>
            <div style={{ fontSize: 8, opacity: 0.7 }}>{PHASE_LABELS[key]}</div>
            {detected && (
              <div style={{ fontSize: 7, opacity: 0.5 }}>
                {phase.time_s?.toFixed(2)}s
              </div>
            )}
          </button>
        )
      })}
    </div>
  )
}
