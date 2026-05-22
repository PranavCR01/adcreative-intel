import json
import os

import anthropic

from agent.prompts import SYSTEM_PROMPT
from agent.tools import (
    TRACE_LOG,
    get_benchmark,
    get_creative_score,
    get_heatmap_regions,
    get_improvement_suggestions,
)

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

TOOL_SCHEMAS = [
    {
        "name": "get_creative_score",
        "description": "Returns predicted CTR score and fatigue halflife for an uploaded creative. Call this first.",
        "input_schema": {"type": "object", "properties": {"image_id": {"type": "string"}}, "required": ["image_id"]},
    },
    {
        "name": "get_heatmap_regions",
        "description": "Returns GradCAM attention regions (high/low) for the creative.",
        "input_schema": {"type": "object", "properties": {"image_id": {"type": "string"}}, "required": ["image_id"]},
    },
    {
        "name": "get_benchmark",
        "description": "Returns industry benchmark CTR and halflife for a vertical. Call before comparing scores.",
        "input_schema": {"type": "object", "properties": {"vertical": {"type": "string", "enum": ["gaming", "ecommerce", "finance", "other"]}}, "required": ["vertical"]},
    },
    {
        "name": "get_improvement_suggestions",
        "description": "Returns 3 concrete improvement suggestions. Call after get_creative_score and get_heatmap_regions.",
        "input_schema": {"type": "object", "properties": {"image_id": {"type": "string"}}, "required": ["image_id"]},
    },
]


def run_agent(image_id: str, user_message: str) -> dict:
    TRACE_LOG.clear()           # reset trace for this request
    messages = [{"role": "user", "content": f"[Creative ID: {image_id}]\n\n{user_message}"}]
    try:
        for _ in range(5):      # max_steps=5 — hard cap prevents infinite loops
            resp = _get_client().messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            if resp.stop_reason == "end_turn":
                answer = next(
                    (block.text for block in resp.content if hasattr(block, "text")), ""
                )
                return {"answer": answer, "trace": list(TRACE_LOG), "image_id": image_id}

            # stop_reason == "tool_use" — execute every tool block in this turn
            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                fn = TOOL_MAP.get(block.name)
                arg = list(block.input.values())[0] if block.input else ""
                result = fn(arg) if fn else {"error": f"Unknown tool: {block.name}"}
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user",      "content": tool_results})

        return {"answer": "Analysis incomplete — too many steps.", "trace": list(TRACE_LOG), "image_id": image_id}
    except Exception as e:
        return {
            "answer": "I encountered an error analyzing this creative. Please try again.",
            "trace": list(TRACE_LOG),
            "image_id": image_id,
            "error": str(e),    # logged server-side, not shown to user
        }
