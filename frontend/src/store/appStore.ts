import { create } from 'zustand'

interface Score {
  ctr_score: number
  halflife_days: number
  confidence: number
}

interface AppState {
  uploadId: string | null
  vertical: string
  imageUrl: string | null
  score: Score | null
  heatmapB64: string | null
  isScoring: boolean
  isHeatmapping: boolean
  modelWarm: boolean
  setUploadId: (id: string) => void
  setVertical: (v: string) => void
  setImageUrl: (url: string) => void
  setScore: (score: Score) => void
  setHeatmap: (b64: string) => void
  setIsScoring: (v: boolean) => void
  setModelWarm: (v: boolean) => void
  reset: () => void
}

const initialState = {
  uploadId: null,
  vertical: 'gaming',
  imageUrl: null,
  score: null,
  heatmapB64: null,
  isScoring: false,
  isHeatmapping: false,
  modelWarm: true,
}

export const useAppStore = create<AppState>((set) => ({
  ...initialState,
  setUploadId: (id) => set({ uploadId: id }),
  setVertical: (v) => set({ vertical: v }),
  setImageUrl: (url) => set({ imageUrl: url }),
  setScore: (score) => set({ score }),
  setHeatmap: (b64) => set({ heatmapB64: b64 }),
  setIsScoring: (v) => set({ isScoring: v }),
  setModelWarm: (v) => set({ modelWarm: v }),
  reset: () => set(initialState),
}))
