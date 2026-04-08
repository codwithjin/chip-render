import { useState, useEffect } from 'react'
import useSwingStore from '../store/useSwingStore'

const SPEEDS = [0.25, 0.5, 1, 2]

export default function PlaybackControls({ videoRef }) {
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [speed, setSpeed] = useState(1)
  const { videoFps, landmarksOn, toggleLandmarks } = useSwingStore()

  useEffect(() => {
    const video = videoRef?.current
    if (!video) return
    const onPlay  = () => setPlaying(true)
    const onPause = () => setPlaying(false)
    const onTime  = () => setCurrentTime(video.currentTime)
    const onMeta  = () => setDuration(video.duration || 0)
    video.addEventListener('play',            onPlay)
    video.addEventListener('pause',           onPause)
    video.addEventListener('timeupdate',      onTime)
    video.addEventListener('loadedmetadata',  onMeta)
    return () => {
      video.removeEventListener('play',           onPlay)
      video.removeEventListener('pause',          onPause)
      video.removeEventListener('timeupdate',     onTime)
      video.removeEventListener('loadedmetadata', onMeta)
    }
  }, [videoRef])

  const togglePlay = () => {
    const v = videoRef?.current
    if (!v) return
    playing ? v.pause() : v.play()
  }

  const stepFrame = (dir) => {
    const v = videoRef?.current
    if (!v) return
    v.pause()
    v.currentTime = Math.max(0, Math.min(duration, v.currentTime + dir / videoFps))
  }

  const setVideoSpeed = (s) => {
    const v = videoRef?.current
    if (!v) return
    v.playbackRate = s
    setSpeed(s)
  }

  const fmtTime = t => {
    if (!isFinite(t)) return '0:00.0'
    const m = Math.floor(t / 60)
    const s = (t % 60).toFixed(1).padStart(4, '0')
    return `${m}:${s}`
  }

  const frame = Math.round(currentTime * videoFps)

  return (
    <div className="flex flex-col gap-1 p-2 border-t border-gray-800 bg-[#0a0e0b]">
      {/* Scrub bar */}
      <input
        type="range" min={0} max={duration || 1} step={0.001}
        value={currentTime}
        onChange={e => {
          const v = videoRef?.current
          if (v) v.currentTime = parseFloat(e.target.value)
        }}
        className="w-full accent-orange-500 h-1"
      />

      <div className="flex items-center gap-2 text-xs text-gray-400">
        {/* Frame step back */}
        <button onClick={() => stepFrame(-1)}
          className="px-2 py-1 rounded border border-gray-700 hover:bg-gray-800">
          ◀
        </button>

        {/* Play/pause */}
        <button onClick={togglePlay}
          className="px-3 py-1 rounded border border-gray-600 text-gray-200 hover:bg-gray-800 min-w-12 text-center">
          {playing ? '⏸' : '▶'}
        </button>

        {/* Frame step forward */}
        <button onClick={() => stepFrame(1)}
          className="px-2 py-1 rounded border border-gray-700 hover:bg-gray-800">
          ▶
        </button>

        {/* Time / frame */}
        <span className="font-mono ml-1">
          {fmtTime(currentTime)} / {fmtTime(duration)}
        </span>
        <span className="text-gray-600 font-mono">F{frame}</span>

        {/* Speed */}
        <div className="flex gap-1 ml-auto">
          {SPEEDS.map(s => (
            <button key={s} onClick={() => setVideoSpeed(s)}
              className={`px-1.5 py-0.5 rounded text-xs border ${
                speed === s
                  ? 'border-orange-500 text-orange-400'
                  : 'border-gray-700 text-gray-500 hover:bg-gray-800'
              }`}>
              {s}x
            </button>
          ))}
        </div>

        {/* Landmarks toggle */}
        <button onClick={toggleLandmarks}
          className={`px-2 py-1 rounded border text-xs ${
            landmarksOn
              ? 'border-orange-500 text-orange-400'
              : 'border-gray-700 text-gray-600'
          }`}>
          LM
        </button>
      </div>
    </div>
  )
}
