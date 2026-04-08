import { create } from 'zustand'

const useSwingStore = create((set, get) => ({
  // Video
  videoFile: null,
  videoUrl: null,
  videoFps: 30,
  videoReady: false,

  // Landmarks
  frameLookup: {},
  poseFrameCount: 0,

  // Processing
  jobId: null,
  processing: false,
  processingProgress: 0,
  processingTotal: 0,
  processingStatus: '',

  // Phases
  phases: {},
  activePhase: 'P1',
  freezeFrames: new Set(),
  triggeredFrames: new Set(),
  freezeActive: false,

  // Metrics
  metrics: {},

  // Session
  sessionId: null,
  sessions: [],
  golferName: '',

  // Notes keyed by phase_key
  notes: {},

  // UI
  landmarksOn: true,

  // Actions
  setVideoFile: (file) => set({ videoFile: file }),
  setVideoUrl: (url) => set({ videoUrl: url }),
  setVideoReady: (ready) => set({ videoReady: ready }),
  setFrameLookup: (lookup, count, fps) => set({
    frameLookup: lookup,
    poseFrameCount: count,
    videoFps: fps,
  }),
  setProcessing: (status, progress, total) => set({
    processing: true,
    processingStatus: status,
    processingProgress: progress,
    processingTotal: total,
  }),
  setProcessingDone: () => set({ processing: false }),
  setPhases: (phases) => {
    const freezeFrames = new Set()
    const FREEZE_PHASES = ['P1', 'P3', 'P4', 'P5', 'P7']
    FREEZE_PHASES.forEach(key => {
      if (phases[key]?.frame !== null && phases[key]?.frame !== undefined) {
        freezeFrames.add(phases[key].frame)
      }
    })
    set({ phases, freezeFrames, triggeredFrames: new Set() })
  },
  setActivePhase: (phase) => set({ activePhase: phase }),
  setFreezeActive: (active) => set({ freezeActive: active }),
  markFrameTriggered: (frame) => set(state => ({
    triggeredFrames: new Set([...state.triggeredFrames, frame]),
  })),
  setMetrics: (metrics) => set({ metrics }),
  setSessionId: (id) => set({ sessionId: id }),
  setSessions: (sessions) => set({ sessions }),
  setGolferName: (name) => set({ golferName: name }),
  addNote: (phaseKey, note) => set(state => ({
    notes: {
      ...state.notes,
      [phaseKey]: [...(state.notes[phaseKey] || []), note],
    },
  })),
  setNotes: (notes) => set({ notes }),
  toggleLandmarks: () => set(state => ({ landmarksOn: !state.landmarksOn })),
  resetSession: () => set({
    videoFile: null, videoUrl: null, videoReady: false,
    frameLookup: {}, poseFrameCount: 0,
    jobId: null, processing: false,
    processingProgress: 0, processingTotal: 0, processingStatus: '',
    phases: {}, activePhase: 'P1',
    freezeFrames: new Set(), triggeredFrames: new Set(),
    freezeActive: false, metrics: {},
    sessionId: null, notes: {},
  }),
}))

export default useSwingStore
