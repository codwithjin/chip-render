import { useEffect, useRef, useCallback } from 'react'
import useSwingStore from '../store/useSwingStore'

const JOINT_COLORS = {
  0:  '#c8c0b0',
  11: '#4a8eff', 13: '#4a8eff', 15: '#4a8eff',
  23: '#4a8eff', 25: '#4a8eff', 27: '#4a8eff',
  12: '#b7410e', 14: '#b7410e', 16: '#b7410e',
  24: '#b7410e', 26: '#b7410e', 28: '#b7410e',
}

const BONES = [
  [11, 12, '#c8c0b0'], [23, 24, '#c8c0b0'],
  [11, 23, '#c8c0b0'], [12, 24, '#c8c0b0'],
  [0,  11, '#c8c0b0'], [0,  12, '#c8c0b0'],
  [11, 13, '#4a8eff'], [13, 15, '#4a8eff'],
  [12, 14, '#b7410e'], [14, 16, '#b7410e'],
  [23, 25, '#4a8eff'], [25, 27, '#4a8eff'],
  [24, 26, '#b7410e'], [26, 28, '#b7410e'],
]

const JOINT_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
const FREEZE_DURATION = 3000

export default function VideoCanvas({ videoRef }) {
  const canvasRef = useRef(null)
  const rafRef = useRef(null)
  const freezeTimeoutRef = useRef(null)

  const {
    videoUrl, videoFps, frameLookup, landmarksOn,
    phases, freezeFrames, triggeredFrames, freezeActive,
    setVideoReady, setActivePhase, setFreezeActive,
    markFrameTriggered,
  } = useSwingStore()

  // RAF draw loop
  const draw = useCallback(() => {
    const video  = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) { rafRef.current = requestAnimationFrame(draw); return }

    const ctx = canvas.getContext('2d')
    const container = canvas.parentElement
    canvas.width  = container.clientWidth  || 800
    canvas.height = container.clientHeight || 600
    const W = canvas.width, H = canvas.height

    if (video.readyState >= 2 && video.videoWidth > 0) {
      const natAsp = video.videoWidth / video.videoHeight
      const canAsp = W / H
      let rW, rH, oX, oY
      if (natAsp > canAsp) {
        rW = W; rH = W / natAsp; oX = 0; oY = (H - rH) / 2
      } else {
        rH = H; rW = H * natAsp; oX = (W - rW) / 2; oY = 0
      }

      try { ctx.drawImage(video, oX, oY, rW, rH) }
      catch (_) { ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H) }

      if (landmarksOn && Object.keys(frameLookup).length) {
        const currentFrame = Math.round(video.currentTime * videoFps)
        console.log('[Canvas] currentFrame:', currentFrame,
                    '| frameLookup size:', Object.keys(frameLookup).length,
                    '| videoFps:', videoFps,
                    '| readyState:', video.readyState,
                    '| videoWidth:', video.videoWidth)
        let pose = frameLookup[currentFrame]
        if (!pose) {
          for (let d = 1; d <= 3; d++) {
            pose = frameLookup[currentFrame + d] || frameLookup[currentFrame - d]
            if (pose) break
          }
        }
        if (pose?.landmarks_2d) {
          console.log('[Canvas] Drawing landmarks for frame', currentFrame)
          const lm2d = pose.landmarks_2d

          const lm3d = pose.landmarks_3d
          const toPixel = lm => ({ x: oX + lm.x * rW, y: oY + lm.y * rH })
          const zRadius = id => {
            const l3 = lm3d?.[String(id)]
            if (!l3) return 5
            return Math.round(3 + Math.max(0, Math.min(1, (l3.z + 0.3) / 0.6)) * 7)
          }

          // Bones
          BONES.forEach(([a, b, color]) => {
            const la = lm2d[String(a)], lb = lm2d[String(b)]
            if (!la || !lb) return
            const vis = Math.min(la.visibility, lb.visibility)
            if (vis < 0.15) return
            const pa = toPixel(la), pb = toPixel(lb)
            ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y)
            ctx.strokeStyle = color; ctx.lineWidth = 2
            ctx.globalAlpha = 0.2 + vis * 0.8; ctx.stroke(); ctx.globalAlpha = 1
          })

          // Joints
          JOINT_IDS.forEach(id => {
            const lm = lm2d[String(id)]
            if (!lm || lm.visibility < 0.15) return
            const p = toPixel(lm), r = zRadius(id)
            const color = JOINT_COLORS[id] || '#fff'
            ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2)
            if (lm.visibility >= 0.4) {
              ctx.fillStyle = color
              ctx.globalAlpha = 0.4 + lm.visibility * 0.6; ctx.fill()
            } else {
              ctx.strokeStyle = color; ctx.lineWidth = 1.5
              ctx.globalAlpha = 0.4; ctx.stroke()
            }
            ctx.globalAlpha = 1
          })

        } else {
          console.log('[Canvas] No landmarks_2d for frame', currentFrame,
                      '| pose exists:', !!pose,
                      '| pose keys:', pose ? Object.keys(pose) : 'none',
                      '| nearest frameLookup keys:', Object.keys(frameLookup).slice(0, 5))
        }
      }

      // ── Club tracking visualization ──
      const currentFrame = Math.round(video.currentTime * videoFps)
      const chFrame = frameLookup[currentFrame]
        || frameLookup[currentFrame + 1]
        || frameLookup[currentFrame - 1]

      if (chFrame) {
        const { club_head, club_handle, golf_ball } = chFrame

        // Draw shaft line from handle to head
        if (club_head && club_handle) {
          const hx = oX + club_handle.x * rW
          const hy = oY + club_handle.y * rH
          const cx = oX + club_head.x * rW
          const cy = oY + club_head.y * rH
          ctx.beginPath()
          ctx.moveTo(hx, hy)
          ctx.lineTo(cx, cy)
          ctx.strokeStyle = '#ffffff'
          ctx.lineWidth = 2
          ctx.globalAlpha = 0.7
          ctx.stroke()
          ctx.globalAlpha = 1.0
        }

        // Red circle — club handle (grip end)
        if (club_handle && club_handle.conf > 0.25) {
          const hx = oX + club_handle.x * rW
          const hy = oY + club_handle.y * rH
          ctx.beginPath()
          ctx.arc(hx, hy, 8, 0, Math.PI * 2)
          ctx.fillStyle = '#ff2222'
          ctx.globalAlpha = 0.85
          ctx.fill()
          ctx.globalAlpha = 1.0
        }

        // White circle — middle of shaft
        if (club_head && club_handle) {
          const mx = oX + ((club_head.x + club_handle.x) / 2) * rW
          const my = oY + ((club_head.y + club_handle.y) / 2) * rH
          ctx.beginPath()
          ctx.arc(mx, my, 5, 0, Math.PI * 2)
          ctx.fillStyle = '#ffffff'
          ctx.globalAlpha = 0.6
          ctx.fill()
          ctx.globalAlpha = 1.0
        }

        // Yellow circle — club head
        if (club_head && club_head.conf > 0.25) {
          const cx = oX + club_head.x * rW
          const cy = oY + club_head.y * rH
          ctx.beginPath()
          ctx.arc(cx, cy, 10, 0, Math.PI * 2)
          ctx.fillStyle = '#ffdd00'
          ctx.globalAlpha = 0.9
          ctx.fill()
          ctx.globalAlpha = 1.0
        }

        // Green circle — golf ball
        if (golf_ball && golf_ball.conf > 0.25) {
          const bx = oX + golf_ball.x * rW
          const by = oY + golf_ball.y * rH
          ctx.beginPath()
          ctx.arc(bx, by, 8, 0, Math.PI * 2)
          ctx.fillStyle = '#00ff44'
          ctx.globalAlpha = 0.9
          ctx.fill()
          ctx.globalAlpha = 1.0
        }
      }
    } else {
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H)
    }

    rafRef.current = requestAnimationFrame(draw)
  }, [videoFps, frameLookup, landmarksOn, videoRef])

  // Start RAF loop
  useEffect(() => {
    rafRef.current = requestAnimationFrame(draw)
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [draw])

  // Load video when URL changes
  useEffect(() => {
    const video = videoRef.current
    if (!video || !videoUrl) return
    video.src = videoUrl
    video.load()
    const onMeta = () => setVideoReady(true)
    video.addEventListener('loadedmetadata', onMeta, { once: true })
    return () => video.removeEventListener('loadedmetadata', onMeta)
  }, [videoUrl, videoRef, setVideoReady])

  // timeupdate — freeze + active phase tracking
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handleTimeUpdate = () => {
      if (freezeActive || video.paused) return
      const currentFrame = Math.round(video.currentTime * videoFps)
      console.log('[Freeze] currentFrame:', currentFrame,
                  '| freezeFrames:', [...freezeFrames],
                  '| freezeActive:', freezeActive,
                  '| phases:', JSON.stringify(phases).substring(0, 200))

      // Update active phase
      const PHASE_ORDER = ['P1', 'P3', 'P4', 'P5', 'P7']
      let currentPhase = 'P1'
      for (let i = PHASE_ORDER.length - 1; i >= 0; i--) {
        const key = PHASE_ORDER[i]
        if (phases[key]?.frame !== null &&
            phases[key]?.frame !== undefined &&
            currentFrame >= phases[key].frame) {
          currentPhase = key; break
        }
      }
      setActivePhase(currentPhase)

      // Freeze check
      if (freezeFrames.size === 0) return
      for (const ff of freezeFrames) {
        if (Math.abs(currentFrame - ff) <= 1 && !triggeredFrames.has(ff)) {
          markFrameTriggered(ff)
          setFreezeActive(true)
          video.pause()

          let phaseKey = ''
          for (const [key, data] of Object.entries(phases)) {
            if (data.frame === ff) { phaseKey = key; break }
          }
          if (phaseKey) setActivePhase(phaseKey)

          freezeTimeoutRef.current = setTimeout(() => {
            setFreezeActive(false)
            video.play()
          }, FREEZE_DURATION)
          break
        }
      }
    }

    video.addEventListener('timeupdate', handleTimeUpdate)
    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate)
      if (freezeTimeoutRef.current) clearTimeout(freezeTimeoutRef.current)
    }
  }, [videoFps, freezeFrames, triggeredFrames, phases, freezeActive,
      setActivePhase, setFreezeActive, markFrameTriggered, videoRef])

  return (
    <div className="relative w-full h-full bg-black overflow-hidden">
      <canvas ref={canvasRef} className="w-full h-full block" />
    </div>
  )
}
