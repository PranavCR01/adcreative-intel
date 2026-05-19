import { create } from 'zustand'

interface ToolCall {
  tool_call_id: string   // uuid, generated from tool name + timestamp
  tool: string
  inputs: Record<string, unknown>
  output: Record<string, unknown>
  duration_ms: number
  status: 'pending' | 'complete' | 'error'
}

interface TraceStore {
  // Map keyed by tool_call_id — O(1) upserts, no re-render storms
  calls: Map<string, ToolCall>
  addCall: (call: ToolCall) => void
  updateCall: (id: string, partial: Partial<ToolCall>) => void
  clearTrace: () => void
}

export const useTraceStore = create<TraceStore>((set) => ({
  calls: new Map(),
  addCall: (call) => set((state) => {
    const next = new Map(state.calls)
    next.set(call.tool_call_id, call)
    return { calls: next }
  }),
  updateCall: (id, partial) => set((state) => {
    const existing = state.calls.get(id)
    if (!existing) return state
    const next = new Map(state.calls)
    // Merge logic — only update defined fields, never overwrite with undefined
    next.set(id, { ...existing, ...Object.fromEntries(
      Object.entries(partial).filter(([_, v]) => v !== undefined)
    )})
    return { calls: next }
  }),
  clearTrace: () => set({ calls: new Map() }),
}))
