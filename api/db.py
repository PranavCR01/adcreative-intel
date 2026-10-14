import os
import threading

from supabase import Client, create_client

_client: Client | None = None
_client_lock = threading.Lock()


def get_db() -> Client:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = create_client(
                    os.environ["SUPABASE_URL"],
                    os.environ["SUPABASE_ANON_KEY"],
                )
    return _client


async def insert_upload(
    upload_id: str,
    r2_key: str,
    vertical: str,
    session_id: str,
) -> dict:
    result = (
        get_db()
        .table("cia_uploads")
        .insert({
            "id": upload_id,
            "r2_key": r2_key,
            "vertical": vertical,
            "user_session": session_id,
        })
        .execute()
    )
    if not result.data:
        raise RuntimeError(f"Supabase returned no data for insert_upload({upload_id})")
    return result.data[0]


async def upsert_score(
    upload_id: str,
    ctr_score: float,
    halflife_days: float | None,
    confidence: float,
    high_attention: list = [],
    low_attention: list = [],
) -> dict:
    result = (
        get_db()
        .table("cia_scores")
        .upsert({
            "upload_id": upload_id,
            "ctr_score": ctr_score,
            "halflife_days": halflife_days,
            "confidence": confidence,
            "heatmap_regions": {"high_attention": high_attention, "low_attention": low_attention},
        })
        .execute()
    )
    if not result.data:
        raise RuntimeError(f"Supabase returned no data for upsert_score({upload_id})")
    return result.data[0]
