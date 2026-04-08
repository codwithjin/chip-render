import { useState, useRef } from 'react'
import useSwingStore from '../store/useSwingStore'
import { saveNote } from '../api/client'

export default function NotesPanel() {
  const { activePhase, sessionId, notes, addNote } = useSwingStore()
  const [text, setText]         = useState('')
  const [listening, setListening] = useState(false)
  const recognitionRef = useRef(null)
  const phaseNotes = notes[activePhase] || []

  const handleSaveText = async () => {
    if (!text.trim()) return
    const noteData = {
      note_type: 'text',
      note_text: text,
      phase_key: activePhase,
    }
    if (sessionId) {
      try {
        const saved = await saveNote(sessionId, activePhase, text, 'text', null)
        addNote(activePhase, saved)
      } catch (_) {
        addNote(activePhase, { ...noteData, id: Date.now(), created_at: new Date().toISOString() })
      }
    } else {
      addNote(activePhase, { ...noteData, id: Date.now(), created_at: new Date().toISOString() })
    }
    setText('')
  }

  const handleVoice = () => {
    if (listening) {
      recognitionRef.current?.stop()
      setListening(false)
      return
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) {
      alert('Speech recognition not supported in this browser')
      return
    }
    const rec = new SpeechRecognition()
    rec.continuous = true
    rec.interimResults = true
    rec.lang = 'en-US'
    rec.onresult = e => {
      const transcript = Array.from(e.results).map(r => r[0].transcript).join('')
      setText(transcript)
    }
    rec.onend = () => setListening(false)
    recognitionRef.current = rec
    rec.start()
    setListening(true)
  }

  const handleScreenshot = async () => {
    const canvas = document.querySelector('canvas')
    if (!canvas) return
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8)
    const noteText = text.trim() || `Screenshot at ${activePhase}`
    const noteData = {
      note_type: 'screenshot',
      note_text: noteText,
      screenshot_url: dataUrl,
      phase_key: activePhase,
    }
    if (sessionId) {
      try {
        const saved = await saveNote(sessionId, activePhase, noteText, 'screenshot', dataUrl)
        addNote(activePhase, saved)
      } catch (_) {
        addNote(activePhase, { ...noteData, id: Date.now(), created_at: new Date().toISOString() })
      }
    } else {
      addNote(activePhase, { ...noteData, id: Date.now(), created_at: new Date().toISOString() })
    }
    setText('')
  }

  return (
    <div className="flex flex-col gap-2 p-3 border-t border-gray-800">
      <div className="text-xs text-gray-600 uppercase tracking-widest">
        Notes — {activePhase}
      </div>

      {/* Existing notes */}
      <div className="flex flex-col gap-1.5 max-h-36 overflow-y-auto">
        {phaseNotes.map((note, i) => (
          <div key={note.id || i} className="text-xs bg-gray-900 rounded p-2">
            {note.note_type === 'screenshot' && note.screenshot_url && (
              <img src={note.screenshot_url} className="w-full rounded mb-1" alt="screenshot" />
            )}
            <span className="text-gray-300">{note.note_text}</span>
            <span className="text-gray-600 ml-2">
              {note.note_type === 'voice' ? '🎙' : note.note_type === 'screenshot' ? '📷' : '✍️'}
            </span>
          </div>
        ))}
      </div>

      <textarea
        className="w-full bg-gray-900 text-gray-200 text-xs rounded p-2 border border-gray-700 resize-none h-14"
        placeholder={`Add note for ${activePhase}...`}
        value={text}
        onChange={e => setText(e.target.value)}
        onKeyDown={e => { if (e.key === 'Enter' && e.metaKey) handleSaveText() }}
      />

      <div className="flex gap-1.5">
        <button onClick={handleSaveText}
          className="flex-1 text-xs py-1 rounded border border-gray-600 text-gray-300 hover:bg-gray-800">
          Save
        </button>
        <button onClick={handleVoice}
          className={`px-2.5 py-1 rounded text-xs border ${
            listening
              ? 'border-red-500 text-red-400 bg-red-950'
              : 'border-gray-600 text-gray-400 hover:bg-gray-800'
          }`}>
          {listening ? '⏹' : '🎙'}
        </button>
        <button onClick={handleScreenshot}
          className="px-2.5 py-1 rounded text-xs border border-gray-600 text-gray-400 hover:bg-gray-800">
          📷
        </button>
      </div>
    </div>
  )
}
