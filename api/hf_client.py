import os

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed

HF_SPACES_URL = os.environ["HF_SPACES_URL"]
API_TOKEN = os.environ["API_TOKEN"]
_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
async def score_image(image_bytes: bytes, vertical: str) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{HF_SPACES_URL}/score",
            headers=_HEADERS,
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={"vertical": vertical},
        )
        resp.raise_for_status()
        return resp.json()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
async def get_heatmap(image_bytes: bytes) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{HF_SPACES_URL}/heatmap",
            headers=_HEADERS,
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
        )
        resp.raise_for_status()
        return resp.json()
