import json
import os
import re

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
