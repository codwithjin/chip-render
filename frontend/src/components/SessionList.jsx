import { useEffect } from 'react'
import useSwingStore from '../store/useSwingStore'
import { getSessions } from '../api/client'

export default function SessionList() {
  const { sessions, setSessions } = useSwingStore()

  useEffect(() => {
    getSessions().then(setSessions).catch(() => {})
  }, [])

  if (!sessions.length) {
    return (
      <div className="p-3 text-xs text-gray-700 text-center">No saved sessions</div>
    )
  }

  return (
    <div className="flex flex-col gap-1 p-2 overflow-y-auto max-h-48">
      <div className="text-xs text-gray-600 uppercase tracking-widest mb-1">Sessions</div>
      {sessions.map(s => (
        <div key={s.id}
          className="flex flex-col px-2 py-1.5 rounded border border-gray-800 bg-[#0f1410] cursor-pointer hover:border-gray-600">
          <span className="text-xs text-gray-300">{s.golfer_name || 'Unnamed'}</span>
          <span className="text-xs text-gray-600">{s.video_filename}</span>
          <span className="text-xs text-gray-700">{new Date(s.created_at).toLocaleString()}</span>
        </div>
      ))}
    </div>
  )
}
