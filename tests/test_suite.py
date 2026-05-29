"""
tests/test_suite.py — Stress test suite for adcreative-intel
=============================================================

Covers:
  Unit tests   1-7   : HF Spaces endpoints, Backend endpoints, Supabase tables
  Load tests   8-10  : Concurrent uploads, concurrent chats, sequential HF scoring
  Integration  11    : Full upload → score → chat → grounded-answer flow

Dependencies (add to dev requirements):
    pip install pytest pytest-asyncio httpx Pillow

Run (from repo root):
    pytest tests/test_suite.py -v --asyncio-mode=auto -s

Required env vars:
    API_TOKEN          — Bearer token for HF Spaces /score and /heatmap
    SUPABASE_URL       — Supabase project URL       (optional; DB tests skip if absent)
    SUPABASE_ANON_KEY  — Supabase anon key          (optional; DB tests skip if absent)

Endpoints under test:
    HF Spaces : https://pcr12-creative-intelligence-scorer.hf.space
    Backend   : https://adcreative-intel.onrender.com
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import httpx
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND = "https://adcreative-intel.onrender.com"
HF      = "https://pcr12-creative-intelligence-scorer.hf.space"

API_TOKEN    = os.environ.get("API_TOKEN", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

HF_AUTH = {"Authorization": f"Bearer {API_TOKEN}"}

# Per-request timeout.  HF Spaces cold-start can take ~45 s; backend /upload
# calls HF internally, so it inherits that latency.
T_UNIT = 90.0

# ---------------------------------------------------------------------------
# Pytest skip markers
# ---------------------------------------------------------------------------

needs_token = pytest.mark.skipif(
    not API_TOKEN,
    reason="API_TOKEN env var not set — skipping authenticated HF Spaces tests",
)
needs_supabase = pytest.mark.skipif(
    not (SUPABASE_URL and SUPABASE_KEY),
    reason="SUPABASE_URL / SUPABASE_ANON_KEY not set — skipping DB tests",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(w: int = 224, h: int = 224) -> bytes:
    """Return a minimal synthetic cornflower-blue JPEG (no real ad data needed)."""
    img = Image.new("RGB", (w, h), color=(100, 149, 237))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# Single in-memory image reused across all tests to avoid repeated allocation.
_TEST_IMAGE: bytes = _make_image()


def _percentile(data: list[float], p: float) -> float:
    """Linear-interpolation percentile (matches numpy default)."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Result collector + terminal report
# ---------------------------------------------------------------------------

@dataclass
class _Result:
    name: str
    passed: bool
    latency_ms: Optional[float] = None
    detail: str = ""


_RESULTS: list[_Result] = []


def _record(
    name: str,
    passed: bool,
    latency_ms: float | None = None,
    detail: str = "",
) -> None:
    _RESULTS.append(_Result(name=name, passed=passed, latency_ms=latency_ms, detail=detail))


def _print_report() -> None:
    SEP  = "=" * 74
    SEP2 = "-" * 74
    GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"

    print(f"\n{SEP}")
    print("  ADCREATIVE-INTEL  STRESS TEST REPORT")
    print(SEP)

    n_pass = sum(1 for r in _RESULTS if r.passed)
    n_fail = len(_RESULTS) - n_pass
    print(f"  PASS: {n_pass}   FAIL: {n_fail}   TOTAL: {len(_RESULTS)}")
    print(SEP2)

    for r in _RESULTS:
        colour = GREEN if r.passed else RED
        tag    = f"{colour}{'PASS' if r.passed else 'FAIL'}{RESET}"
        lat    = f"  {r.latency_ms:>8.0f} ms" if r.latency_ms is not None else "             "
        note   = f"  │  {r.detail}" if r.detail else ""
        print(f"  [{tag}]{lat}  {r.name}{note}")

    print(SEP)


# ---------------------------------------------------------------------------
# Session-level shared state
#   _STATE["upload_id"] — a pre-uploaded image ID available to load / chat tests
# ---------------------------------------------------------------------------

_STATE: dict = {}


@pytest.fixture(scope="session", autouse=True)
def _session_bootstrap_and_report():
    """
    1. Pre-upload one image so chat-load and integration tests have a real DB row.
    2. Print the summary report after all tests finish.
    """
    async def _preflight_upload() -> Optional[str]:
        uid = str(uuid.uuid4())
        try:
            async with httpx.AsyncClient(timeout=T_UNIT) as c:
                r = await c.post(
                    f"{BACKEND}/upload",
                    files={"image": ("preflight.jpg", _TEST_IMAGE, "image/jpeg")},
                    data={"vertical": "gaming", "upload_id": uid},
                )
                if r.status_code == 200:
                    return uid
        except Exception:
            pass
        return None  # load tests fall back to a synthetic UUID if this fails

    _STATE["upload_id"] = asyncio.run(_preflight_upload())
    yield
    _print_report()


# ---------------------------------------------------------------------------
# ── UNIT TESTS ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# 1 ─ HF Spaces /health ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hf_health():
    """GET /health → {"status": "ok", "model_loaded": true}"""
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=T_UNIT) as c:
        r = await c.get(f"{HF}/health")
    ms = (time.monotonic() - t0) * 1000
    body = r.json()

    ok_status  = r.status_code == 200
    ok_payload = body.get("status") == "ok" and body.get("model_loaded") is True

    _record(
        "1  hf /health — status=ok model_loaded=true",
        ok_status and ok_payload,
        ms,
        detail="" if (ok_status and ok_payload) else f"HTTP {r.status_code} body={body}",
    )

    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert body.get("status") == "ok", f"status field wrong: {body}"
    assert body.get("model_loaded") is True, f"model_loaded not True: {body}"


# 2 ─ HF Spaces /score ──────────────────────────────────────────────────────

@needs_token
@pytest.mark.asyncio
async def test_hf_score():
    """POST /score → ctr_score ∈ [0,1], halflife_days > 0, confidence ∈ [0,1]"""
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=T_UNIT) as c:
        r = await c.post(
            f"{HF}/score",
            headers=HF_AUTH,
            files={"image": ("test.jpg", _TEST_IMAGE, "image/jpeg")},
            data={"vertical": "gaming"},
        )
    ms = (time.monotonic() - t0) * 1000
    body = r.json()

    passed = (
        r.status_code == 200
        and isinstance(body.get("ctr_score"), float)
        and isinstance(body.get("halflife_days"), float)
        and isinstance(body.get("confidence"), float)
        and 0.0 <= body["ctr_score"] <= 1.0
        and body["halflife_days"] > 0.0
        and 0.0 <= body["confidence"] <= 1.0
    )
    _record(
        "2  hf /score — ctr_score, halflife_days, confidence in range",
        passed,
        ms,
        detail=(
            f"ctr={body.get('ctr_score'):.4f} hl={body.get('halflife_days'):.1f}d"
            if passed else f"HTTP {r.status_code} body={body}"
        ),
    )

    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {body}"
    assert isinstance(body.get("ctr_score"), float), f"ctr_score missing or wrong type: {body}"
    assert 0.0 <= body["ctr_score"] <= 1.0, f"ctr_score out of range: {body['ctr_score']}"
    assert body["halflife_days"] > 0.0, f"halflife_days not positive: {body}"
    assert 0.0 <= body["confidence"] <= 1.0, f"confidence out of range: {body}"


# 2b ─ HF Spaces /heatmap (verifies heatmap_b64) ────────────────────────────
#      /score does NOT return heatmap_b64 — that lives on /heatmap.
#      The backend /upload combines both; this test exercises the raw HF endpoint.

@needs_token
@pytest.mark.asyncio
async def test_hf_heatmap():
    """POST /heatmap → heatmap_b64 non-empty, high/low_attention lists present"""
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=T_UNIT) as c:
        r = await c.post(
            f"{HF}/heatmap",
            headers=HF_AUTH,
            files={"image": ("test.jpg", _TEST_IMAGE, "image/jpeg")},
        )
    ms = (time.monotonic() - t0) * 1000
    body = r.json()
    b64_len = len(body.get("heatmap_b64", ""))

    passed = (
        r.status_code == 200
        and b64_len > 100
        and isinstance(body.get("high_attention"), list)
        and isinstance(body.get("low_attention"), list)
        and len(body["high_attention"]) > 0
        and len(body["low_attention"]) > 0
    )
    _record(
        "2b hf /heatmap — heatmap_b64 non-empty, attention regions present",
        passed,
        ms,
        detail=(
            f"b64_len={b64_len} high={body.get('high_attention')} low={body.get('low_attention')}"
            if passed else f"HTTP {r.status_code} b64_len={b64_len}"
        ),
    )

    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {body}"
    assert b64_len > 100, f"heatmap_b64 suspiciously short ({b64_len} chars)"
    assert isinstance(body.get("high_attention"), list) and len(body["high_attention"]) > 0
    assert isinstance(body.get("low_attention"), list)  and len(body["low_attention"])  > 0


# 3 ─ HF Spaces /benchmark ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hf_benchmark():
    """GET /benchmark?vertical=gaming → median_ctr, median_halflife, sample_size"""
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=T_UNIT) as c:
        r = await c.get(f"{HF}/benchmark", params={"vertical": "gaming"})
    ms = (time.monotonic() - t0) * 1000
    body = r.json()

    passed = (
        r.status_code == 200
        and body.get("vertical") == "gaming"
        and isinstance(body.get("median_ctr"), float)
        and isinstance(body.get("median_halflife"), float)
        and isinstance(body.get("sample_size"), int)
        and body["median_ctr"] > 0
        and body["sample_size"] > 0
    )
    _record(
        "3  hf /benchmark — median_ctr, median_halflife, sample_size for gaming",
        passed,
        ms,
        detail=(
            f"ctr={body.get('median_ctr')} hl={body.get('median_halflife')} n={body.get('sample_size')}"
            if passed else f"HTTP {r.status_code} body={body}"
        ),
    )

    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {body}"
    assert body.get("vertical") == "gaming"
    assert isinstance(body.get("median_ctr"), float)    and body["median_ctr"] > 0
    assert isinstance(body.get("median_halflife"), float)
    assert isinstance(body.get("sample_size"), int)     and body["sample_size"] > 0


# 4 ─ Backend /health ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_backend_health():
    """GET /health → {"status": "ok"}"""
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.get(f"{BACKEND}/health")
    ms = (time.monotonic() - t0) * 1000
    body = r.json()

    passed = r.status_code == 200 and body.get("status") == "ok"
    _record(
        "4  backend /health — status=ok",
        passed,
        ms,
        detail="" if passed else f"HTTP {r.status_code} body={body}",
    )

    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert body.get("status") == "ok", f"Unexpected body: {body}"


# 5 ─ Backend /upload ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_backend_upload():
    """POST /upload → upload_id echoed, ctr_score, confidence, heatmap_b64 present"""
    uid = str(uuid.uuid4())
    t0  = time.monotonic()
    async with httpx.AsyncClient(timeout=T_UNIT) as c:
        r = await c.post(
            f"{BACKEND}/upload",
            files={"image": ("test.jpg", _TEST_IMAGE, "image/jpeg")},
            data={"vertical": "gaming", "upload_id": uid},
        )
    ms   = (time.monotonic() - t0) * 1000
    body = r.json()

    passed = (
        r.status_code == 200
        and body.get("upload_id") == uid
        and isinstance(body.get("ctr_score"), float)
        and isinstance(body.get("confidence"), float)
        and isinstance(body.get("heatmap_b64"), str)
        and 0.0 <= body["ctr_score"] <= 1.0
        and 0.0 <= body["confidence"] <= 1.0
    )
    _record(
        "5  backend /upload — upload_id, ctr_score, confidence, heatmap_b64",
        passed,
        ms,
        detail=(
            f"ctr={body.get('ctr_score'):.4f} conf={body.get('confidence'):.4f} "
            f"hl={body.get('halflife_days')} b64_len={len(body.get('heatmap_b64',''))}"
            if passed else f"HTTP {r.status_code} body={body}"
        ),
    )

    assert r.status_code == 200, f"Upload failed ({r.status_code}): {body}"
    assert body.get("upload_id") == uid,             "upload_id not echoed"
    assert isinstance(body.get("ctr_score"), float), "ctr_score missing"
    assert 0.0 <= body["ctr_score"] <= 1.0,          "ctr_score out of [0,1]"
    assert isinstance(body.get("confidence"), float), "confidence missing"
    assert isinstance(body.get("heatmap_b64"), str),  "heatmap_b64 missing"


# 6 ─ Backend /chat ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_backend_chat():
    """POST /chat → answer str, trace list, image_id echoed"""
    image_id = _STATE.get("upload_id") or str(uuid.uuid4())
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=T_UNIT) as c:
        r = await c.post(
            f"{BACKEND}/chat",
            json={
                "image_id": image_id,
                "message": "What is the CTR score for this creative?",
                "vertical": "gaming",
            },
        )
    ms   = (time.monotonic() - t0) * 1000
    body = r.json()

    answer = body.get("answer", "")
    passed = (
        r.status_code == 200
        and isinstance(answer, str)
        and len(answer) >= 20
        and body.get("image_id") == image_id
        and isinstance(body.get("trace"), list)
    )
    _record(
        "6  backend /chat — answer returned, trace list, image_id echoed",
        passed,
        ms,
        detail=(
            f"answer_len={len(answer)} trace_steps={len(body.get('trace',[]))}"
            if passed else f"HTTP {r.status_code} body={body}"
        ),
    )

    assert r.status_code == 200, f"Chat failed ({r.status_code}): {body}"
    assert isinstance(answer, str) and len(answer) >= 20, f"Answer too short: {answer!r}"
    assert body.get("image_id") == image_id,  "image_id not echoed"
    assert isinstance(body.get("trace"), list), "trace not a list"


# 7 ─ Supabase table reachability ────────────────────────────────────────────

@needs_supabase
@pytest.mark.asyncio
async def test_supabase_tables():
    """
    Verify cia_uploads and cia_scores tables are reachable via the Supabase REST API.
    A 200/206 response (even empty rows) confirms the table exists and the anon key works.
    """
    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept":        "application/json",
    }
    base     = SUPABASE_URL.rstrip("/")
    tables   = ("cia_uploads", "cia_scores")
    details  = []
    all_ok   = True

    async with httpx.AsyncClient(timeout=20.0) as c:
        for table in tables:
            t0 = time.monotonic()
            r  = await c.get(
                f"{base}/rest/v1/{table}",
                headers=headers,
                params={"limit": "1", "select": "*"},
            )
            ms = (time.monotonic() - t0) * 1000
            ok = r.status_code in (200, 206)
            details.append(f"{table}={'OK' if ok else f'HTTP {r.status_code}'} ({ms:.0f}ms)")
            if not ok:
                all_ok = False

    _record(
        "7  supabase — cia_uploads + cia_scores reachable via REST API",
        all_ok,
        detail=", ".join(details),
    )
    assert all_ok, f"Supabase table check failed: {details}"


# ---------------------------------------------------------------------------
# ── LOAD TESTS ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# 8 ─ 10 simultaneous /upload requests ──────────────────────────────────────

@pytest.mark.asyncio
async def test_load_upload_concurrent():
    """
    Fire 10 simultaneous POST /upload requests.
    SLA: all complete within 60 s wall time, zero 5xx responses.
    """
    N       = 10
    SLA_S   = 60.0

    async def _one_upload(idx: int) -> tuple[int, float]:
        uid = str(uuid.uuid4())
        t0  = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=T_UNIT) as c:
                r = await c.post(
                    f"{BACKEND}/upload",
                    files={"image": ("load.jpg", _TEST_IMAGE, "image/jpeg")},
                    data={"vertical": "gaming", "upload_id": uid},
                )
            return r.status_code, (time.monotonic() - t0) * 1000
        except Exception:
            return 0, (time.monotonic() - t0) * 1000

    wall_t0 = time.monotonic()
    outcomes = await asyncio.gather(*[_one_upload(i) for i in range(N)])
    wall_s   = time.monotonic() - wall_t0

    statuses  = [s for s, _ in outcomes]
    latencies = [ms for _, ms in outcomes]
    success   = sum(1 for s in statuses if s == 200)
    err_5xx   = sum(1 for s in statuses if s >= 500)
    err_other = sum(1 for s in statuses if 0 < s < 500 and s != 200)

    passed = err_5xx == 0 and wall_s <= SLA_S
    detail = (
        f"{success}/{N} ok  5xx={err_5xx}  other_err={err_other}  "
        f"wall={wall_s:.1f}s  "
        f"p50={_percentile(latencies,50):.0f}ms  "
        f"p95={_percentile(latencies,95):.0f}ms"
    )
    _record(
        f"8  load /upload {N}x concurrent — no 5xx, wall≤{SLA_S:.0f}s",
        passed,
        wall_s * 1000,
        detail=detail,
    )

    print(f"\n  Load /upload ({N} concurrent):")
    print(f"    statuses : {statuses}")
    print(f"    wall     : {wall_s:.1f}s  (SLA: {SLA_S}s)")
    print(f"    p50      : {_percentile(latencies,50):.0f}ms")
    print(f"    p95      : {_percentile(latencies,95):.0f}ms")

    assert err_5xx == 0, f"{err_5xx}/{N} requests returned 5xx: {statuses}"
    assert wall_s <= SLA_S, f"Wall time {wall_s:.1f}s exceeded {SLA_S}s SLA"


# 9 ─ 10 simultaneous /chat requests ────────────────────────────────────────

@pytest.mark.asyncio
async def test_load_chat_concurrent():
    """
    Fire 10 simultaneous POST /chat requests against the same image_id.
    SLA: all complete within 30 s wall time, zero 5xx responses.
    Note: the agent may return a graceful error message if the upload_id has
    no DB row; what we're stress-testing is the thread-pool + Anthropic client
    under concurrency, not data freshness.
    """
    N       = 10
    SLA_S   = 30.0
    image_id = _STATE.get("upload_id") or str(uuid.uuid4())

    async def _one_chat(idx: int) -> tuple[int, float]:
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=T_UNIT) as c:
                r = await c.post(
                    f"{BACKEND}/chat",
                    json={
                        "image_id": image_id,
                        "message": f"Give me a one-line performance summary. (req #{idx})",
                        "vertical": "gaming",
                    },
                )
            return r.status_code, (time.monotonic() - t0) * 1000
        except Exception:
            return 0, (time.monotonic() - t0) * 1000

    wall_t0  = time.monotonic()
    outcomes = await asyncio.gather(*[_one_chat(i) for i in range(N)])
    wall_s   = time.monotonic() - wall_t0

    statuses  = [s for s, _ in outcomes]
    latencies = [ms for _, ms in outcomes]
    success   = sum(1 for s in statuses if s == 200)
    err_5xx   = sum(1 for s in statuses if s >= 500)

    passed = err_5xx == 0 and wall_s <= SLA_S
    detail = (
        f"{success}/{N} ok  5xx={err_5xx}  "
        f"wall={wall_s:.1f}s  "
        f"p50={_percentile(latencies,50):.0f}ms  "
        f"p95={_percentile(latencies,95):.0f}ms"
    )
    _record(
        f"9  load /chat {N}x concurrent — no 5xx, wall≤{SLA_S:.0f}s",
        passed,
        wall_s * 1000,
        detail=detail,
    )

    print(f"\n  Load /chat ({N} concurrent):")
    print(f"    statuses : {statuses}")
    print(f"    wall     : {wall_s:.1f}s  (SLA: {SLA_S}s)")
    print(f"    p50      : {_percentile(latencies,50):.0f}ms")
    print(f"    p95      : {_percentile(latencies,95):.0f}ms")

    assert err_5xx == 0, f"{err_5xx}/{N} requests returned 5xx: {statuses}"
    assert wall_s <= SLA_S, f"Wall time {wall_s:.1f}s exceeded {SLA_S}s SLA"


# 10 ─ 20 sequential HF Spaces /score requests ──────────────────────────────

@needs_token
@pytest.mark.asyncio
async def test_load_hf_score_sequential():
    """
    Send 20 sequential POST /score requests to HF Spaces and measure latency
    distribution (p50 / p95 / p99).  Uses a single persistent AsyncClient to
    exercise keep-alive reuse.  All requests must succeed (0 errors).
    """
    N         = 20
    latencies: list[float] = []
    errors    = 0

    async with httpx.AsyncClient(timeout=T_UNIT) as c:
        for i in range(N):
            t0 = time.monotonic()
            try:
                r = await c.post(
                    f"{HF}/score",
                    headers=HF_AUTH,
                    files={"image": ("seq.jpg", _TEST_IMAGE, "image/jpeg")},
                    data={"vertical": "gaming"},
                )
                if r.status_code != 200:
                    errors += 1
            except Exception:
                errors += 1
            latencies.append((time.monotonic() - t0) * 1000)

    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)
    mn  = min(latencies)
    mx  = max(latencies)

    passed = errors == 0
    detail = (
        f"n={N}  errors={errors}  "
        f"p50={p50:.0f}ms  p95={p95:.0f}ms  p99={p99:.0f}ms  "
        f"min={mn:.0f}ms  max={mx:.0f}ms"
    )
    _record(
        f"10 load hf /score {N}x sequential — p50/p95/p99 latency",
        passed,
        sum(latencies),
        detail=detail,
    )

    print(f"\n  Sequential HF /score latency ({N} requests, single keep-alive client):")
    print(f"    p50  = {p50:>7.0f} ms")
    print(f"    p95  = {p95:>7.0f} ms")
    print(f"    p99  = {p99:>7.0f} ms")
    print(f"    min  = {mn:>7.0f} ms")
    print(f"    max  = {mx:>7.0f} ms")
    print(f"    errors = {errors}/{N}")

    assert errors == 0, f"{errors} out of {N} sequential /score requests failed"


# ---------------------------------------------------------------------------
# ── INTEGRATION TEST ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# 11 ─ Full flow: upload → score already in DB → chat → grounded answer ──────

@pytest.mark.asyncio
async def test_integration_full_flow():
    """
    End-to-end flow:
      1. Upload a fresh image via Backend /upload  → capture ctr_score
      2. POST /chat asking about the creative's performance
      3. Assert the agent called get_creative_score (trace verification)
      4. Assert the answer contains at least one numeric claim (grounded)
      5. Assert the answer is substantive (≥ 50 chars)
      6. Assert the answer is NOT the generic error fallback
    """
    uid = str(uuid.uuid4())

    # ── Step 1: Upload ──────────────────────────────────────────────────────
    t_up = time.monotonic()
    async with httpx.AsyncClient(timeout=T_UNIT) as c:
        up_r = await c.post(
            f"{BACKEND}/upload",
            files={"image": ("integ.jpg", _TEST_IMAGE, "image/jpeg")},
            data={"vertical": "gaming", "upload_id": uid},
        )
    upload_ms = (time.monotonic() - t_up) * 1000

    assert up_r.status_code == 200, (
        f"Integration upload failed ({up_r.status_code}): {up_r.text}"
    )
    up_body   = up_r.json()
    ctr_score = up_body.get("ctr_score")
    halflife  = up_body.get("halflife_days")
    assert isinstance(ctr_score, float), f"ctr_score missing from upload: {up_body}"

    # ── Step 2: Chat ────────────────────────────────────────────────────────
    t_ch = time.monotonic()
    async with httpx.AsyncClient(timeout=T_UNIT) as c:
        ch_r = await c.post(
            f"{BACKEND}/chat",
            json={
                "image_id": uid,
                "message": (
                    "What is the CTR score for this creative, "
                    "and how does it compare to the gaming benchmark?"
                ),
                "vertical": "gaming",
            },
        )
    chat_ms = (time.monotonic() - t_ch) * 1000

    assert ch_r.status_code == 200, (
        f"Integration chat failed ({ch_r.status_code}): {ch_r.text}"
    )
    ch_body = ch_r.json()
    answer  = ch_body.get("answer", "")
    trace   = ch_body.get("trace", [])

    # ── Step 3: Verify agent used get_creative_score ─────────────────────────
    tool_names = [step.get("tool") for step in trace if isinstance(step, dict)]
    assert "get_creative_score" in tool_names, (
        f"Agent did not call get_creative_score. Trace tools: {tool_names}\n"
        f"Answer: {answer!r}"
    )

    # ── Step 4: Verify answer contains a numeric claim ───────────────────────
    # The agent may express the score as 0.1234, 12.34 %, "12%", etc.
    has_number = bool(re.search(r"\d+(?:\.\d+)?", answer))
    assert has_number, f"Answer contains no numerical claims: {answer!r}"

    # ── Step 5: Answer is substantive ────────────────────────────────────────
    assert len(answer) >= 50, f"Answer suspiciously short ({len(answer)} chars): {answer!r}"

    # ── Step 6: Not a generic error fallback ─────────────────────────────────
    error_phrases = [
        "encountered an error",
        "analysis incomplete",
        "please try again",
    ]
    for phrase in error_phrases:
        assert phrase not in answer.lower(), (
            f"Answer looks like an error fallback: {answer!r}"
        )

    total_ms = upload_ms + chat_ms
    detail   = (
        f"uid={uid[:8]}…  ctr={ctr_score:.4f}  hl={halflife}d  "
        f"tools={tool_names}  answer_len={len(answer)}  "
        f"upload={upload_ms:.0f}ms  chat={chat_ms:.0f}ms"
    )
    _record("11 integration — upload→score→chat grounded answer", True, total_ms, detail=detail)
