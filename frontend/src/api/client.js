import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL || ''

export const uploadVideo = async (file, onUploadProgress) => {
  const fd = new FormData()
  fd.append('video', file)
  const res = await axios.post(`${BASE}/process`, fd, {
    onUploadProgress: e => {
      if (onUploadProgress && e.total) {
        onUploadProgress(Math.round(e.loaded / e.total * 100))
      }
    },
  })
  return res.data.job_id
}

export const pollProgress = async (jobId) => {
  const res = await axios.get(`${BASE}/progress/${jobId}`)
  return res.data
}

export const fetchResult = async (jobId) => {
  const res = await axios.get(`${BASE}/result/${jobId}`)
  return res.data
}

export const detectPhases = async (frames, fps) => {
  const res = await axios.post(`${BASE}/phases`, { frames, fps })
  return res.data
}

export const createSession = async (data) => {
  const res = await axios.post(`${BASE}/api/sessions`, data)
  return res.data
}

export const getSessions = async () => {
  const res = await axios.get(`${BASE}/api/sessions`)
  return res.data
}

export const saveNote = async (sessionId, phaseKey, text, type, screenshotUrl) => {
  const res = await axios.post(`${BASE}/api/notes`, {
    session_id: sessionId,
    phase_key: phaseKey,
    note_text: text,
    note_type: type,
    screenshot_url: screenshotUrl || null,
  })
  return res.data
}

export const getNotes = async (sessionId) => {
  const res = await axios.get(`${BASE}/api/notes/${sessionId}`)
  return res.data
}
