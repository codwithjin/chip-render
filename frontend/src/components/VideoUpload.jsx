import { useRef, useState } from 'react'
import useSwingStore from '../store/useSwingStore'
import {
  uploadVideo as apiUpload, pollProgress, fetchResult,
  detectPhases, createSession,
} from '../api/client'

const sleep = ms => new Promise(r => setTimeout(r, ms))

function buildFrameLookup(data) {
  const frameLookup = {}
  let poseFrameCount = 0

  // First pass — collect raw poses
  data.frames.forEach(f => {
    if (f.poses?.length > 0) {
      frameLookup[f.frame] = f.poses[0]
      poseFrameCount++
    }
  })

  // World-coord bounding box
  const MARGIN = 0.07
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity
  for (const pose of Object.values(frameLookup)) {
    const raw = pose.landmarks || {}
    for (const lm of Object.values(raw)) {
      if ((lm.visibility || 0) >= 0.15) {
        if (lm.x < xMin) xMin = lm.x; if (lm.x > xMax) xMax = lm.x
        if (lm.y < yMin) yMin = lm.y; if (lm.y > yMax) yMax = lm.y
      }
    }
  }

  // Second pass — normalize to [MARGIN, 1-MARGIN]
  for (const pose of Object.values(frameLookup)) {
    const raw = pose.landmarks || {}
    const lm2d = {}
    for (const [id, lm] of Object.entries(raw)) {
      lm2d[id] = {
        x: MARGIN + ((lm.x - xMin) / (xMax - xMin || 1)) * (1 - 2 * MARGIN),
        y: MARGIN + ((lm.y - yMin) / (yMax - yMin || 1)) * (1 - 2 * MARGIN),
        z: lm.z,
        visibility: lm.visibility,
      }
    }
    pose.landmarks_2d = lm2d
  }

  return { frameLookup, poseFrameCount }
}

export default function VideoUpload({ videoRef }) {
  const [dragOver, setDragOver] = useState(false)
  const [status, setStatus]     = useState('')
  const [progress, setProgress] = useState(0)
  const fileInputRef = useRef(null)

  const {
    resetSession, setVideoUrl, setProcessing, setProcessingDone,
    setFrameLookup, setPhases, setSessionId, processing, videoFile,
    setVideoFile,
  } = useSwingStore()

  const handleFile = async (file) => {
    if (!file || !/\.(mp4|mov|avi|webm)$/i.test(file.name)) {
      setStatus('Please upload a video file (.mp4, .mov, .avi, .webm)')
      return
    }
    if (file.size > 500 * 1024 * 1024) {
      setStatus('File too large — max 500 MB')
      return
    }

    resetSession()
    setVideoFile(file)

    // Load video immediately for preview
    const previewUrl = URL.createObjectURL(file)
    setVideoUrl(previewUrl)

    // Upload pipeline
    try {
      setStatus('Uploading...')
      setProcessing('Uploading...', 0, 100)

      const jobId = await apiUpload(file, pct => {
        setProgress(pct)
        setProcessing(`Uploading ${pct}%`, pct, 100)
      })

      setStatus('Processing with MediaPipe...')

      // Poll until done
      while (true) {
        await sleep(1000)
        const prog = await pollProgress(jobId)
        if (prog.status === 'error') throw new Error(prog.error || 'Processing failed')
        const pct = prog.total > 0 ? Math.round(prog.progress / prog.total * 100) : 0
        setProcessing(`Extracting landmarks: ${prog.progress}/${prog.total}`, pct, 100)
        setProgress(pct)
        if (prog.status === 'done') break
      }

      setStatus('Fetching landmark data...')
      const result = await fetchResult(jobId)
      if (!result?.frames) throw new Error('Invalid landmark data')

      const { frameLookup, poseFrameCount } = buildFrameLookup(result)
      setFrameLookup(frameLookup, poseFrameCount, result.fps || 30)

      setStatus('Detecting swing phases...')
      const phaseData = await detectPhases(result.frames, result.fps || 30)
      setPhases(phaseData)

      // Save session to DB
      try {
        const session = await createSession({
          golfer_name: '',
          video_filename: file.name,
          fps: result.fps || 30,
          total_frames: result.total_frames || 0,
          phases: phaseData,
          metrics: {},
        })
        setSessionId(session.id)
      } catch (_) {
        // DB optional — don't break if unavailable
      }

      setProcessingDone()
      setStatus(`Done — ${poseFrameCount} pose frames`)
      setProgress(100)
    } catch (err) {
      setProcessingDone()
      setStatus(`Error: ${err.message}`)
    }
  }

  const onDrop = e => {
    e.preventDefault(); setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  return (
    <div className="flex flex-col gap-3 p-3">
      {/* Drop zone */}
      <div
        className={`border-2 border-dashed rounded p-4 text-center cursor-pointer transition-colors ${
          dragOver
            ? 'border-orange-500 bg-orange-950'
            : 'border-gray-700 hover:border-gray-500'
        }`}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="text-gray-500 text-xs uppercase tracking-widest">
          {videoFile ? videoFile.name : 'Drop video or click to upload'}
        </div>
        <div className="text-gray-700 text-xs mt-1">.mp4 · .mov · max 500 MB</div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={e => { if (e.target.files[0]) handleFile(e.target.files[0]); e.target.value = '' }}
      />

      {/* Progress */}
      {processing && (
        <div className="flex flex-col gap-1">
          <div className="w-full bg-gray-800 rounded h-1">
            <div
              className="bg-orange-500 h-1 rounded transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="text-xs text-gray-500">{status}</div>
        </div>
      )}

      {!processing && status && (
        <div className="text-xs text-gray-500">{status}</div>
      )}
    </div>
  )
}
