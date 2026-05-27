import os

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

HF_SPACES_URL = os.environ["HF_SPACES_URL"]
API_TOKEN = os.environ["API_TOKEN"]
_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=60.0)
    return _http_client


def _is_server_error(exc: BaseException) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


@retry(stop=stop_after_attempt(3), wait=wait_fixed(3), retry=retry_if_exception(_is_server_error))
async def score_image(image_bytes: bytes, vertical: str) -> dict:
    client = get_http_client()
    resp = await client.post(
        f"{HF_SPACES_URL}/score",
        headers=_HEADERS,
        files={"image": ("image.jpg", image_bytes, "image/jpeg")},
        data={"vertical": vertical},
    )
    resp.raise_for_status()
    return resp.json()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(3), retry=retry_if_exception(_is_server_error))
async def get_heatmap(image_bytes: bytes) -> dict:
    client = get_http_client()
    resp = await client.post(
        f"{HF_SPACES_URL}/heatmap",
        headers=_HEADERS,
        files={"image": ("image.jpg", image_bytes, "image/jpeg")},
    )
    resp.raise_for_status()
    return resp.json()
