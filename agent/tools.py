import json
import os
import time

import anthropic

from api.db import get_db

_anthropic_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


_BENCHMARKS = {
    "gaming":    {"median_ctr": 0.119, "median_halflife": 10.7, "sample_size": 7485},
    "ecommerce": {"median_ctr": 0.125, "median_halflife": 11.2, "sample_size": 7078},
    "finance":   {"median_ctr": 0.111, "median_halflife": 10.0, "sample_size": 6520},
    "other":     {"median_ctr": 0.118, "median_halflife": 10.6, "sample_size": 1000},
}


def get_creative_score(image_id: str, trace: list[dict]) -> dict:
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
        db = get_db()
        result = (
            db.table("cia_scores")
            .select("ctr_score, halflife_days, confidence")
            .eq("upload_id", image_id)
            .single()
            .execute()
        )
        data = result.data
        trace.append({
            "tool": "get_creative_score",
            "inputs": {"image_id": image_id},
            "output": data,
            "duration_ms": int((time.time() - start) * 1000),
        })
        return data
    except Exception as e:
        return {"error": f"Could not fetch score: {str(e)}"}


def get_heatmap_regions(image_id: str, trace: list[dict]) -> dict:
    """
    Returns attention regions from GradCAM analysis of the creative.
    Use this when asked why specific elements are working or not.
    Args:
        image_id: the upload UUID for the creative being analyzed
    Returns:
        high_attention (list of region labels), low_attention (list of region labels)
    """
    start = time.time()
    try:
        db = get_db()
        result = (
            db.table("cia_scores")
            .select("heatmap_regions")
            .eq("upload_id", image_id)
            .single()
            .execute()
        )
        regions = result.data.get("heatmap_regions") if result.data else None
        if regions is None:
            data = {
                "high_attention": [],
                "low_attention": [],
                "note": "Heatmap not yet scored.",
            }
        else:
            data = regions
        trace.append({
            "tool": "get_heatmap_regions",
            "inputs": {"image_id": image_id},
            "output": data,
            "duration_ms": int((time.time() - start) * 1000),
        })
        return data
    except Exception as e:
        return {"error": f"Could not fetch heatmap regions: {str(e)}"}


def get_benchmark(vertical: str, trace: list[dict]) -> dict:
    """
    Returns industry benchmark CTR and fatigue halflife for a given vertical.
    Always call this to give context before comparing a score.
    Args:
        vertical: one of "gaming", "ecommerce", "finance", "other"
    Returns:
        median_ctr (float), median_halflife (float), sample_size (int)
    """
    start = time.time()
    key = vertical.lower()
    if key not in _BENCHMARKS:
        return {"error": f"Unknown vertical '{vertical}'. Valid: {list(_BENCHMARKS)}"}
    data = {"vertical": key, **_BENCHMARKS[key]}
    trace.append({
        "tool": "get_benchmark",
        "inputs": {"vertical": vertical},
        "output": data,
        "duration_ms": int((time.time() - start) * 1000),
    })
    return data


def get_improvement_suggestions(image_id: str, trace: list[dict]) -> dict:
    """
    Returns 3 concrete creative improvement suggestions based on score and heatmap.
    Call this after get_creative_score and get_heatmap_regions.
    Args:
        image_id: the upload UUID for the creative being analyzed
    Returns:
        suggestions: list of 3 strings, each a concrete actionable change
    """
    start = time.time()
    try:
        db = get_db()
        result = (
            db.table("cia_scores")
            .select("ctr_score, halflife_days, confidence, heatmap_regions")
            .eq("upload_id", image_id)
            .single()
            .execute()
        )
        score_data = result.data or {}

        ctr = score_data.get("ctr_score", "unknown")
        halflife = score_data.get("halflife_days", "unknown")
        heatmap = score_data.get("heatmap_regions") or {}
        high = heatmap.get("high_attention", [])
        low = heatmap.get("low_attention", [])

        grounding = (
            f"Creative metrics:\n"
            f"- CTR score: {ctr}\n"
            f"- Predicted fatigue halflife: {halflife} days\n"
            f"- High-attention regions: {high}\n"
            f"- Low-attention regions: {low}\n"
        )

        resp = _get_client().messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            system=(
                "You are a mobile ad creative expert. "
                "Given the performance metrics below, return exactly 3 concrete, "
                "actionable improvement suggestions as a JSON array of strings. "
                "Each suggestion must reference a specific metric from the data. "
                "Respond only with the JSON array, no other text."
            ),
            messages=[{"role": "user", "content": grounding}],
        )

        raw = resp.content[0].text.strip()
        try:
            parsed = json.loads(raw)
            suggestions = parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            suggestions = []
        if not suggestions:
            suggestions = [s.strip("- ").strip() for s in raw.split("\n") if s.strip()][:3]

        data = {"suggestions": suggestions[:3]}
        trace.append({
            "tool": "get_improvement_suggestions",
            "inputs": {"image_id": image_id},
            "output": data,
            "duration_ms": int((time.time() - start) * 1000),
        })
        return data
    except Exception as e:
        return {"error": f"Could not generate suggestions: {str(e)}"}
