from fastapi import APIRouter, HTTPException, UploadFile, Form

from api.db import get_db, insert_upload, upsert_score
from api.hf_client import get_heatmap, score_image

router = APIRouter()


@router.post("")
async def upload_creative(
    image: UploadFile,
    vertical: str = Form(...),
    upload_id: str = Form(...),    # client-generated UUID — exists before DB row
    session_id: str = Form(""),   # optional, default prevents 422
):
    image_bytes = await image.read()

    # 1. Upload to Supabase Storage bucket "creatives"
    storage_path = f"creatives/{upload_id}.jpg"
    try:
        db = get_db()
        db.storage.from_("creatives").upload(
            path=storage_path,
            file=image_bytes,
            file_options={"content-type": "image/jpeg"},
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Upload storage failed.")

    # 2. Insert into Supabase DB (storage_path stored in r2_key column)
    await insert_upload(upload_id, storage_path, vertical, session_id or upload_id)

    # 3. Score via HF Spaces
    try:
        score_result = await score_image(image_bytes, vertical)
    except Exception:
        raise HTTPException(status_code=503, detail="Model is warming up. Please retry.")

    # 4. Get heatmap (non-fatal — returns empty on failure)
    try:
        heatmap_result = await get_heatmap(image_bytes)
    except Exception:
        heatmap_result = {"heatmap_b64": "", "high_attention": [], "low_attention": []}

    # 5. Persist score
    await upsert_score(
        upload_id,
        score_result["ctr_score"],
        score_result.get("halflife_days"),
        score_result["confidence"],
    )

    return {
        "upload_id": upload_id,
        **score_result,
        "heatmap_b64": heatmap_result.get("heatmap_b64", ""),
        "high_attention": heatmap_result.get("high_attention", []),
        "low_attention": heatmap_result.get("low_attention", []),
    }
