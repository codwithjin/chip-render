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

  // Server (Tasks API) already provides landmarks_2d and landmarks_3d directly.
  // No normalization needed — just index by frame number.
  data.frames.forEach(f => {
    if (f.poses?.length > 0) {
      frameLookup[f.frame] = {
        ...f.poses[0],
        club_head:   f.club_head   ?? null,
        club_handle: f.club_handle ?? null,
        golf_ball:   f.golf_ball   ?? null,
      }
      poseFrameCount++
    }
  })

  console.log('[Upload] buildFrameLookup done. poseFrameCount:', poseFrameCount)
  const sample = frameLookup[Object.keys(frameLookup)[0]]
  console.log('[Upload] sample pose keys:', sample ? Object.keys(sample) : 'none')
  console.log('[Upload] sample landmarks_2d keys:', sample?.landmarks_2d ? Object.keys(sample.landmarks_2d).slice(0, 5) : 'MISSING')

  return { frameLookup, poseFrameCount }
}

export default function VideoUpload({ videoRef }) {
  const [dragOver, setDragOver] = useState(false)
  const [status, setStatus]     = useState('')
  const [progress, setProgress] = useState(0)
  const fileInputRef = useRef(null)

  const {
    resetSession, setVideoUrl, setProcessing, setProcessingDone,
    setFrameLookup, setPhases, setMetrics, setSessionId, processing, videoFile,
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
      console.log('[Upload] setFrameLookup called. size:', Object.keys(frameLookup).length, 'fps:', result.fps)
      console.log('[Upload] sample frame 0:', frameLookup[0])
      console.log('[Upload] videoUrl (previewUrl):', previewUrl)

      setStatus('Detecting swing phases...')
      const phaseData = await detectPhases(result.frames, result.fps || 30)
      setPhases(phaseData)
      if (phaseData.metrics) setMetrics(phaseData.metrics)

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
