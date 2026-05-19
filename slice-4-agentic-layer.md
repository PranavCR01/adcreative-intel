# Slice 4 — Agentic Layer

## Goal
LLM agent with grounded tool calls producing real, non-hallucinated explanations.
By end of this slice: /chat endpoint working locally, agent calls correct tools,
every numerical claim traces to a tool result, Zustand trace store designed.

## Architecture decisions locked in this slice
- Anthropic Python SDK (0.97.0) + claude-haiku-4-5 — API-based, no local GPU needed
- Manual ReAct loop is the primary approach — Claude responds in natural language,
  Python parses Action: / Final Answer: lines, no framework dependency
- ANTHROPIC_API_KEY via os.environ — add to Railway env vars in Slice 5
- Agent response is a single JSON object returned over HTTP — NO SSE streaming
- Reasoning trace is captured server-side and returned with the final answer
- max_steps=5 (range(5) loop in run_agent) — prevents infinite tool-call loops
- Zustand store uses Map keyed by tool_call_id — not array (prevents re-render storms)
- Every tool call is logged: tool_name, inputs, raw_output, duration_ms

## File structure for this slice
```
agent/
├── tools.py          ← 4 tool functions wrapping HF Spaces API
├── agent.py          ← Anthropic SDK client + manual ReAct loop
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
import time

from api.db import get_db

TRACE_LOG = []  # module-level, reset per request in run_agent()


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

## agent.py — Anthropic SDK + manual ReAct loop

```python
import json
import os
import re

import anthropic

from agent.tools import (
    get_creative_score, get_heatmap_regions,
    get_benchmark, get_improvement_suggestions, TRACE_LOG,
)
from agent.prompts import SYSTEM_PROMPT

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


TOOL_MAP = {
    "get_creative_score": get_creative_score,
    "get_benchmark": get_benchmark,
    "get_heatmap_regions": get_heatmap_regions,
    "get_improvement_suggestions": get_improvement_suggestions,
}


def run_agent(image_id: str, user_message: str) -> dict:
    TRACE_LOG.clear()           # reset trace for this request
    messages = [{"role": "user", "content": f"[Creative ID: {image_id}]\n\n{user_message}"}]
    response_text = ""
    try:
        for _ in range(5):      # max_steps=5 — hard cap prevents infinite loops
            resp = _get_client().messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            response_text = resp.content[0].text
            action_match = re.search(r'Action:\s*(\w+)\(([^)]*)\)', response_text)
            if not action_match:
                break           # no more actions — extract Final Answer
            tool_name = action_match.group(1)
            arg = action_match.group(2).strip("\"'")
            if tool_name in TOOL_MAP:
                result = TOOL_MAP[tool_name](arg)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": f"Result: {json.dumps(result)}"})
        final = re.search(r'Final Answer:\s*(.+)', response_text, re.DOTALL)
        return {
            "answer": final.group(1).strip() if final else response_text,
            "trace": list(TRACE_LOG),   # copy before next request clears it
            "image_id": image_id,
        }
    except Exception as e:
        return {
            "answer": "I encountered an error analyzing this creative. Please try again.",
            "trace": list(TRACE_LOG),
            "image_id": image_id,
            "error": str(e),    # logged server-side, not shown to user
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

To call a tool, respond with:
Thought: [your reasoning about what to do next]
Action: tool_name(argument)

When you receive a Result, continue reasoning. When you have enough data, respond with:
Final Answer: [your answer — specific, actionable, 3-5 sentences]

Example of a correct response sequence:
Thought: I need to get the CTR score for this creative before making any claims.
Action: get_creative_score(abc-123)
[after receiving result]
Thought: Now I need the gaming benchmark to contextualize this score.
Action: get_benchmark(gaming)
[after receiving result]
Final Answer: Your creative scores 0.31 CTR with a predicted fatigue halflife of
4.2 days — below the gaming median of 0.48 CTR and 8.3 days halflife. The heatmap
shows your CTA in the low-attention bottom-left region. Moving the CTA to
center-right and reducing text density are the two changes most likely to extend
performance.
"""
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

## Environment variables for this slice
```
ANTHROPIC_API_KEY=your_key    # add to Railway env vars in Slice 5
```
Haiku pricing: ~$0.001 per agent run (3-5 API calls × ~200 tokens each).
Plenty for portfolio demos with no GPU requirement.

## Bug prevention checklist for this slice
- [ ] TRACE_LOG.clear() at start of every run_agent call — never leaks between requests
- [ ] max_steps=5 (range(5) loop) — test that it stops after 5 steps with a complex query
- [ ] ANTHROPIC_API_KEY in os.environ — run_agent will raise KeyError at startup if missing
- [ ] ChatRequest.session_id has default "" — prevents 422 when frontend omits it
- [ ] run_agent wraps everything in try/except — agent errors never propagate as 500 unhandled
- [ ] Tools are plain functions returning dicts — never raise inside a tool function
- [ ] Test: ask agent about a nonexistent image_id — confirm it returns error message, not hallucinated score
- [ ] Test: ask a question that requires 3 tool calls — confirm all 3 appear in trace
- [ ] lsof -i :8000 before starting local uvicorn

## Start prompt for Claude Code
```
Starting Slice 4 of Creative Intelligence Agent. Read CLAUDE.md first,
then slice-4-agentic-layer.md in full.

Goals this session:
1. Write agent/tools.py with all 4 tools — plain Python functions, each returns
   structured dict, handles errors without raising, logs to TRACE_LOG
2. Write agent/prompts.py with the SYSTEM_PROMPT from the slice doc
3. Write agent/agent.py with Anthropic SDK client, TOOL_MAP, and run_agent()
   implementing the manual ReAct loop — max_steps=5, TRACE_LOG.clear() on each call
4. Write api/routes/chat.py with /chat POST endpoint
5. Update api/main.py to include the chat router
6. Write frontend/src/store/traceStore.ts exactly as designed in the slice doc

Use /plan first.
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
