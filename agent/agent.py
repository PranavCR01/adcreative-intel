import json

from agent.prompts import SYSTEM_PROMPT
from agent.tools import (
    _get_client,
    get_benchmark,
    get_creative_score,
    get_fatigue_projection,
    get_heatmap_regions,
    get_improvement_suggestions,
)

TOOL_MAP = {
    "get_creative_score":          get_creative_score,
    "get_benchmark":               get_benchmark,
    "get_fatigue_projection":      get_fatigue_projection,
    "get_heatmap_regions":         get_heatmap_regions,
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
    {
        "name": "get_fatigue_projection",
        "description": (
            "Computes ad fatigue retention at day 7, 14, and 21 from a halflife value. "
            "Call after get_creative_score when the user asks about longevity, "
            "refresh cadence, or how long to run the ad."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "halflife_days": {
                    "type": "number",
                    "description": "Predicted fatigue halflife in days",
                }
            },
            "required": ["halflife_days"],
        },
    },
]


def run_agent(image_id: str, user_message: str, vertical: str = "other", history: list[dict] = [], vertical_confidence: float = 1.0) -> dict:
    trace: list[dict] = []
    messages = []
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    if history:
        messages.append({
            "role": "user",
            "content": f"[Reminder: Creative ID is {image_id}, Vertical is {vertical} (confidence: {vertical_confidence:.0%})]"
        })
        messages.append({
            "role": "assistant",
            "content": "Understood, I'll use that creative ID for any tool calls."
        })
    if not history:
        current_content = f"[Creative ID: {image_id}] [Vertical: {vertical} (confidence: {vertical_confidence:.0%})]\n\n{user_message}"
    else:
        current_content = user_message
    messages.append({"role": "user", "content": current_content})
    try:
        for _ in range(5):      # max_steps=5 — hard cap prevents infinite loops
            print(f"[agent] turn={_} messages={json.dumps(messages[-2:], default=str, indent=2)}", flush=True)
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
                return {"answer": answer, "trace": trace, "image_id": image_id}

            # stop_reason == "tool_use" — execute every tool block in this turn
            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                fn = TOOL_MAP.get(block.name)
                result = fn(**block.input, trace=trace) if fn else {"error": f"Unknown tool: {block.name}"}
                print(f"[agent] tool={block.name} input={block.input} result={result}", flush=True)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user",      "content": tool_results})

        return {"answer": "Analysis incomplete — too many steps.", "trace": trace, "image_id": image_id}
    except Exception as e:
        return {
            "answer": "I encountered an error analyzing this creative. Please try again.",
            "trace": trace,
            "image_id": image_id,
            "error": str(e),    # logged server-side, not shown to user
        }
