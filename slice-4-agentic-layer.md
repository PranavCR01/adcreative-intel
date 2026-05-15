# Slice 4 — Agentic Layer

## Goal
LLM agent with grounded tool calls producing real, non-hallucinated explanations.
By end of this slice: /chat endpoint working locally, agent calls correct tools,
every numerical claim traces to a tool result, Zustand trace store designed.

## Architecture decisions locked in this slice
- Agent runs on LOCAL machine or HF Spaces — Railway free tier has no GPU
- Qwen2.5-1.5B loaded in 4-bit with bitsandbytes — fits 1650 Ti (3.5GB VRAM)
- smolagents CodeAgent is primary. If tool-call parsing fails after 4hrs debugging,
  fall back to manual ReAct loop — document the switch in README
- Agent response is a single JSON object returned over HTTP — NO SSE streaming
- Reasoning trace is captured server-side and returned with the final answer
- max_steps=5 on the agent — prevents infinite tool-call loops
- Zustand store uses Map keyed by tool_call_id — not array (prevents re-render storms)
- Every tool call is logged: tool_name, inputs, raw_output, duration_ms

## File structure for this slice
```
agent/
├── tools.py          ← 4 tool functions wrapping HF Spaces API
├── agent.py          ← smolagents CodeAgent setup + run loop
└── prompts.py        ← system prompt and few-shot examples
api/
├── routes/
│   └── chat.py       ← /chat POST endpoint
└── main.py           ← updated to include chat router
frontend/src/store/
└── traceStore.ts     ← Zustand Map for reasoning trace (designed, not built)
```

## The 4 tools — tools.py

### Tool design rule
Every tool must:
1. Call HF Spaces via hf_client.py (already written in Slice 3)
2. Return structured data the agent can reason over
3. Include error handling — return `{"error": "..."}` never raise inside a tool
4. Log: tool name, inputs, output, duration_ms to a list for the trace

```python
from smolagents import tool
from api.hf_client import score_image, get_heatmap
import httpx, time, os

TRACE_LOG = []  # module-level, reset per request

@tool
def get_creative_score(image_id: str) -> dict:
    """
    Returns predicted CTR score and fatigue halflife for an uploaded creative.
    Use this first when asked about creative performance.
    Args:
        image_id: the upload UUID for the creative being analyzed
    Returns:
        ctr_score (float 0-1), halflife_days (float), confidence (float 0-1)
    """
    start = time.time()
    try:
        # Fetch from Supabase scores table (image already scored at upload time)
        from api.db import get_db
        db = get_db()
        result = (db.table("cia_scores")
                   .select("ctr_score, halflife_days, confidence")
                   .eq("upload_id", image_id)
                   .single()
                   .execute())
        data = result.data
        TRACE_LOG.append({
            "tool": "get_creative_score",
            "inputs": {"image_id": image_id},
            "output": data,
            "duration_ms": int((time.time() - start) * 1000)
        })
        return data
    except Exception as e:
        return {"error": f"Could not fetch score: {str(e)}"}

@tool
def get_heatmap_regions(image_id: str) -> dict:
    """
    Returns attention regions from GradCAM analysis of the creative.
    Use this when asked why specific elements are working or not.
    Args:
        image_id: the upload UUID for the creative being analyzed
    Returns:
        high_attention (list of region labels), low_attention (list of region labels)
    """
    # Fetch heatmap analysis cached at upload time from cia_scores JSONB field
    # or re-request from HF Spaces if not cached
    ...

@tool
def get_benchmark(vertical: str) -> dict:
    """
    Returns industry benchmark CTR and fatigue halflife for a given vertical.
    Always call this to give context before comparing a score.
    Args:
        vertical: one of "gaming", "ecommerce", "finance", "other"
    Returns:
        median_ctr (float), median_halflife (float), sample_size (int)
    """
    ...

@tool
def get_improvement_suggestions(image_id: str) -> dict:
    """
    Returns 3 concrete creative improvement suggestions based on score and heatmap.
    Call this after get_creative_score and get_heatmap_regions.
    Args:
        image_id: the upload UUID for the creative being analyzed
    Returns:
        suggestions: list of 3 strings, each a concrete actionable change
    """
    # This tool is LLM-generated internally but grounded in score + heatmap data
    # It calls a separate lightweight prompt, NOT the main agent
    # So suggestions are always derived from real tool outputs, never hallucinated
    ...
```

## agent.py — smolagents setup

```python
from smolagents import CodeAgent, HfApiModel
from agent.tools import (get_creative_score, get_heatmap_regions,
                          get_benchmark, get_improvement_suggestions, TRACE_LOG)
from agent.prompts import SYSTEM_PROMPT
import os

def create_agent():
    model = HfApiModel(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        token=os.getenv("HF_TOKEN"),
    )
    # Use HF Inference API — avoids loading model locally on Railway
    # For local dev: replace HfApiModel with TransformersModel (4-bit local)
    agent = CodeAgent(
        tools=[get_creative_score, get_heatmap_regions,
               get_benchmark, get_improvement_suggestions],
        model=model,
        max_steps=5,            # hard cap — prevents infinite loops
        system_prompt=SYSTEM_PROMPT,
    )
    return agent

def run_agent(image_id: str, user_message: str) -> dict:
    TRACE_LOG.clear()           # reset trace for this request
    agent = create_agent()
    full_prompt = f"[Creative ID: {image_id}]\n\nUser: {user_message}"
    try:
        answer = agent.run(full_prompt)
        return {
            "answer": str(answer),
            "trace": list(TRACE_LOG),   # copy before next request clears it
            "image_id": image_id,
        }
    except Exception as e:
        return {
            "answer": "I encountered an error analyzing this creative. Please try again.",
            "trace": list(TRACE_LOG),
            "image_id": image_id,
            "error": str(e)     # logged server-side, not shown to user
        }
```

## prompts.py — system prompt
```python
SYSTEM_PROMPT = """
You are a creative performance analyst for mobile advertising.
You have access to tools that analyze ad creatives using a trained vision model.

CRITICAL RULES — follow these exactly:
1. You MUST call get_creative_score before making any claim about CTR or fatigue.
2. You MUST call get_benchmark before comparing a score to "industry average."
3. You MUST call get_heatmap_regions before claiming any visual element is a problem.
4. Never state a number that did not come from a tool call result.
5. If a tool returns an error, say so clearly. Do not guess or substitute a number.

Your answers should be:
- Specific: reference actual numbers from tool results
- Actionable: end with concrete changes the advertiser can make
- Brief: 3-5 sentences maximum for the final answer

Example of a correct response:
"Your creative has a CTR score of 0.31 and a predicted fatigue halflife of
4.2 days (get_creative_score result). The gaming vertical median is 0.48 CTR
and 8.3 days halflife (get_benchmark result). The heatmap shows your CTA
button is in a low-attention bottom-left region (get_heatmap_regions result).
Moving the CTA to center-right and reducing text density are the two changes
most likely to extend performance."
"""
```

## Manual ReAct fallback (use if smolagents fails)
If smolagents tool-call parsing fails after 4 hours of debugging, implement
a manual ReAct loop instead. Document the switch in README.

```python
# Manual ReAct pattern — pure Python, no framework dependency
import json, re

REACT_PROMPT = """You are analyzing ad creatives. Available tools:
- get_creative_score(image_id) → {ctr_score, halflife_days, confidence}
- get_benchmark(vertical) → {median_ctr, median_halflife}
- get_heatmap_regions(image_id) → {high_attention, low_attention}
- get_improvement_suggestions(image_id) → {suggestions}

Respond with: Thought: [reasoning] Action: tool_name(arg) Result: [tool output]
Repeat until you have enough data. Then respond: Final Answer: [your answer]
"""

TOOL_MAP = {
    "get_creative_score": get_creative_score_fn,
    "get_benchmark": get_benchmark_fn,
    "get_heatmap_regions": get_heatmap_regions_fn,
    "get_improvement_suggestions": get_improvement_suggestions_fn,
}

def react_loop(image_id: str, question: str, max_steps: int = 5) -> dict:
    messages = [{"role": "system", "content": REACT_PROMPT}]
    messages.append({"role": "user", "content": f"[{image_id}] {question}"})
    trace = []
    for step in range(max_steps):
        response = call_llm(messages)   # call Qwen via HF Inference API
        # Parse Action: tool_name(arg) from response
        action_match = re.search(r'Action: (\w+)\(([^)]*)\)', response)
        if not action_match:
            break   # no more actions — extract Final Answer
        tool_name, arg = action_match.group(1), action_match.group(2).strip('"\'')
        result = TOOL_MAP[tool_name](arg)
        trace.append({"tool": tool_name, "input": arg, "output": result})
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Result: {json.dumps(result)}"})
    final = re.search(r'Final Answer: (.+)', response, re.DOTALL)
    return {"answer": final.group(1).strip() if final else response, "trace": trace}
```

## api/routes/chat.py — /chat endpoint
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agent.agent import run_agent

router = APIRouter()

class ChatRequest(BaseModel):
    image_id: str
    message: str
    session_id: str = ""    # optional, default prevents 422

class ChatResponse(BaseModel):
    answer: str
    trace: list
    image_id: str

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = run_agent(request.image_id, request.message)
        return ChatResponse(**result)
    except Exception as e:
        # Log error server-side, return human-readable to client
        print(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Please try again."
        )
```

## Zustand trace store design — traceStore.ts
Not built yet (frontend is Slice 5) but design it now to prevent re-render storms.

```typescript
// frontend/src/store/traceStore.ts
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
```

## Using HF Inference API vs local model
smolagents `HfApiModel` calls Qwen2.5-1.5B via HF Inference API (free tier).
This means no local GPU needed for the agent — the model runs on HF servers.
Free tier allows ~1000 requests/day which is plenty for portfolio demos.
For local dev/testing, swap to `TransformersModel` with 4-bit quantization.

## Bug prevention checklist for this slice
- [ ] TRACE_LOG.clear() at start of every run_agent call — never leaks between requests
- [ ] max_steps=5 set on CodeAgent — test that it stops after 5 steps with a complex query
- [ ] ChatRequest.session_id has default "" — prevents 422 when frontend omits it
- [ ] run_agent wraps everything in try/except — agent errors never propagate as 500 unhandled
- [ ] Tools return {"error": "..."} dict on failure — never raise inside a tool function
- [ ] Test: ask agent about a nonexistent image_id — confirm it returns error message, not hallucinated score
- [ ] Test: ask a question that requires 3 tool calls — confirm all 3 appear in trace
- [ ] lsof -i :8000 before starting local uvicorn

## Start prompt for Claude Code
```
Starting Slice 4 of Creative Intelligence Agent. Read CLAUDE.md first,
then slice-4-agentic-layer.md in full.

Goals this session:
1. Write agent/tools.py with all 4 tools — each wraps HF Spaces or Supabase,
   returns structured dict, handles errors without raising, logs to TRACE_LOG
2. Write agent/prompts.py with the system prompt from the slice doc
3. Write agent/agent.py with smolagents CodeAgent, HfApiModel for Qwen2.5-1.5B,
   max_steps=5, and run_agent() that clears TRACE_LOG on each call
4. Write api/routes/chat.py with /chat POST endpoint
5. Update api/main.py to include the chat router
6. Write frontend/src/store/traceStore.ts exactly as designed in the slice doc

Use /plan first. Primary: smolagents CodeAgent. If parsing fails after
debugging, implement the manual ReAct fallback from the slice doc instead.
```

## Done when
- [ ] POST /chat with a real image_id returns answer + trace in < 30 seconds
- [ ] Trace contains at least 1 tool call for any non-trivial question
- [ ] Agent never returns a number not present in trace tool outputs (test 5 questions)
- [ ] Asking about nonexistent image_id returns error message not hallucinated score
- [ ] max_steps respected: agent stops after 5 steps maximum
- [ ] traceStore.ts file exists with correct Map-based design

## Next slice
Slice 5 — Full-Stack Product. Come back to claude.ai first.
