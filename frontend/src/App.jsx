import { useRef } from 'react'
import VideoCanvas from './components/VideoCanvas'
import VideoUpload from './components/VideoUpload'
import PhaseStrip from './components/PhaseStrip'
import PlaybackControls from './components/PlaybackControls'
import MetricCards from './components/MetricCards'
import NotesPanel from './components/NotesPanel'
import SessionList from './components/SessionList'
import useSwingStore from './store/useSwingStore'

export default function App() {
  // Single shared videoRef — passed down so VideoCanvas owns the element
  const videoRef = useRef(null)
  const { poseFrameCount, activePhase, phases } = useSwingStore()

  const phaseDetected = phases[activePhase]?.detected

  return (
    <div className="flex h-screen w-screen bg-[#0a0e0b] text-[#d4d0c8] overflow-hidden font-mono">

      {/* Hidden video element — owned here, shared via ref */}
      <video
        ref={videoRef}
        style={{ position: 'absolute', width: 1, height: 1, opacity: 0, pointerEvents: 'none' }}
        preload="metadata"
        playsInline
      />

      {/* ── LEFT PANEL ── */}
      <div className="flex flex-col w-56 border-r border-gray-800 shrink-0">
        {/* Logo */}
        <div className="px-3 py-2 border-b border-gray-800 text-xs text-gray-600 tracking-widest uppercase">
          CHIP · Swing Analysis
        </div>

        {/* Upload */}
        <VideoUpload videoRef={videoRef} />

        {/* Session list */}
        <div className="flex-1 overflow-y-auto border-t border-gray-800">
          <SessionList />
        </div>

        {/* Stats */}
        <div className="border-t border-gray-800 p-3 text-xs text-gray-700 space-y-1">
          <div>Pose frames: <span className="text-gray-500">{poseFrameCount || '—'}</span></div>
          <div>Active phase: <span className="text-gray-500">{activePhase}</span></div>
          {phaseDetected && (
            <div>Time: <span className="text-gray-500">{phases[activePhase]?.time_s?.toFixed(3)}s</span></div>
          )}
        </div>
      </div>

      {/* ── CENTER PANEL ── */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Video canvas — takes all remaining space */}
        <div className="flex-1 min-h-0 overflow-hidden">
          <VideoCanvas videoRef={videoRef} />
        </div>

        {/* Phase strip */}
        <PhaseStrip videoRef={videoRef} />

        {/* Playback controls */}
        <PlaybackControls videoRef={videoRef} />
      </div>

      {/* ── RIGHT PANEL ── */}
      <div className="flex flex-col w-64 border-l border-gray-800 shrink-0">
        {/* Phase header */}
        <div className="px-3 py-2 border-b border-gray-800 text-xs text-gray-600 tracking-widest uppercase flex justify-between items-center">
          <span>Metrics · {activePhase}</span>
          {phaseDetected && (
            <span className="text-green-700">DETECTED</span>
          )}
        </div>

        {/* Metrics */}
        <div className="flex-1 overflow-y-auto">
          <MetricCards />
        </div>

        {/* Notes */}
        <NotesPanel />
      </div>
    </div>
  )
}
