# Slice 3 — Model Serving on HF Spaces

## Goal
Live inference endpoint on HF Spaces, callable from anywhere.
By end of this slice: three endpoints responding correctly, cold start
< 60s, GitHub Actions keepalive running, Supabase tables created.

## Architecture decisions locked in this slice
- HF Spaces runs FastAPI (not Gradio) — more professional, easier to call from Railway
- Model loads from HF Hub on startup — never bundle weights into the Space
- All inference is CPU-only on HF Spaces free tier — use map_location='cpu'
- Frontend NEVER calls HF Spaces directly — all traffic goes Railway → HF Spaces
- Supabase tables are created this slice using the migration SQL from Slice 1 doc
- Every Supabase write uses .select().throwOnError() — silent failures are not allowed
- Bearer token auth: single shared token stored in HF Space secrets + Railway env vars

## File structure for this slice
```
spaces/
├── app.py              ← FastAPI inference server
├── model_loader.py     ← loads model from HF Hub, singleton pattern
├── requirements.txt    ← spaces-specific dependencies
└── README.md           ← HF Spaces config (YAML frontmatter required)
api/
├── __init__.py
├── main.py             ← FastAPI app skeleton (routes added Slice 5)
├── hf_client.py        ← calls HF Spaces from Railway backend
└── db.py               ← Supabase client setup
.github/
└── workflows/
    └── keepalive.yml   ← GitHub Actions cron to ping HF Spaces
```

## spaces/README.md — required HF Spaces config
```yaml
---
title: Creative Intelligence Scorer
emoji: 🎨
colorFrom: purple
colorTo: teal
sdk: docker
pinned: false
---
```
HF Spaces with FastAPI requires Docker SDK, not Gradio. Include a Dockerfile.

## spaces/Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```
HF Spaces Docker apps must run on port 7860.

## spaces/model_loader.py — singleton model loading
```python
# Singleton pattern — model loads once on startup, reused across requests
# Critical: map_location='cpu' — HF Spaces free tier has no GPU
import torch
from transformers import CLIPProcessor
from model.clip_head import CreativeScorer  # copy model/ into spaces/
import os

_model = None
_processor = None

def get_model():
    global _model, _processor
    if _model is None:
        hf_repo = os.getenv("HF_MODEL_REPO")  # e.g. "pranav/creative-intelligence-scorer"
        # Load processor
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # Load model weights from HF Hub
        _model = CreativeScorer()
        state_dict = torch.hub.load_state_dict_from_url(
            f"https://huggingface.co/{hf_repo}/resolve/main/pytorch_model.bin",
            map_location="cpu"   # CRITICAL — never remove this
        )
        _model.load_state_dict(state_dict)
        _model.eval()
    return _model, _processor
```

## spaces/app.py — FastAPI inference server
### Endpoints

**POST /score**
```
Input:  multipart/form-data { image: file, vertical: str }
Output: {
  ad_id: str,          # UUID generated server-side
  ctr_score: float,    # 0.0–1.0
  halflife_days: float,# predicted fatigue halflife
  confidence: float,   # 1 - prediction std across augmentations (simple proxy)
  inference_ms: int    # latency for monitoring
}
```

**POST /heatmap**
```
Input:  multipart/form-data { image: file }
Output: {
  heatmap_b64: str,    # base64-encoded PNG of GradCAM overlay
  high_attention: list[str],  # ["top-left", "center"] — region labels
  low_attention: list[str]
}
```
Region labeling: divide 224x224 into 3x3 grid, label top-3 high/low cells
by their GradCAM intensity. Map grid positions to human labels:
["top-left", "top-center", "top-right", "mid-left", "center",
 "mid-right", "bot-left", "bot-center", "bot-right"]

**GET /benchmark**
```
Input:  ?vertical=gaming (query param)
Output: {
  vertical: str,
  median_ctr: float,
  median_halflife: float,
  sample_size: int
}
```
Benchmarks are hardcoded from your training data statistics — computed
once in evaluate.py and hardcoded here. Not a DB call. Fast, reliable.

```python
BENCHMARKS = {
    "gaming":    {"median_ctr": 0.52, "median_halflife": 8.3,  "sample_size": 15000},
    "ecommerce": {"median_ctr": 0.48, "median_halflife": 11.2, "sample_size": 18000},
    "finance":   {"median_ctr": 0.41, "median_halflife": 14.7, "sample_size": 9000},
    "other":     {"median_ctr": 0.45, "median_halflife": 10.1, "sample_size": 8000},
}
# Replace with real values from evaluate.py after training completes
```

**GET /health**
```
Output: { "status": "ok", "model_loaded": bool }
```
Used by GitHub Actions keepalive and Railway health checks.

### Auth
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security, HTTPException

security = HTTPBearer()
BEARER_TOKEN = os.getenv("API_TOKEN")  # set in HF Space secrets

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
```
Apply `verify_token` to /score and /heatmap. /health and /benchmark are public.

### Error handling — no raw errors to clients
```python
# Every endpoint wraps logic in try/except
# Never return str(e) — always return a human-readable message
ERROR_MESSAGES = {
    "model_not_loaded": "Model is warming up. Please retry in 30 seconds.",
    "invalid_image": "Could not process image. Please upload a JPG or PNG.",
    "inference_failed": "Scoring failed. Please try again.",
}
```

## api/hf_client.py — Railway calls HF Spaces
```python
import httpx, base64, os
from tenacity import retry, stop_after_attempt, wait_fixed

HF_SPACES_URL = os.getenv("HF_SPACES_URL")
API_TOKEN = os.getenv("API_TOKEN")
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
async def score_image(image_bytes: bytes, vertical: str) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{HF_SPACES_URL}/score",
            headers=HEADERS,
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={"vertical": vertical}
        )
        resp.raise_for_status()
        return resp.json()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
async def get_heatmap(image_bytes: bytes) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{HF_SPACES_URL}/heatmap",
            headers=HEADERS,
            files={"image": ("image.jpg", image_bytes, "image/jpeg")}
        )
        resp.raise_for_status()
        return resp.json()
```
Note: `tenacity` handles the retry logic with 3s delay — equivalent to
`fetchWithRetry` on the frontend. Always 3 retries, always 3s wait.

## api/db.py — Supabase client
```python
from supabase import create_client, Client
import os

_client: Client | None = None

def get_db() -> Client:
    global _client
    if _client is None:
        _client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_ANON_KEY")
        )
    return _client

async def insert_upload(upload_id: str, r2_key: str, vertical: str, session_id: str):
    db = get_db()
    result = (db.table("cia_uploads")
               .insert({
                   "id": upload_id,    # client-generated UUID
                   "r2_key": r2_key,
                   "vertical": vertical,
                   "user_session": session_id
               })
               .select()              # surfaces errors immediately
               .execute())
    return result.data[0]

async def upsert_score(upload_id: str, ctr_score: float,
                       halflife_days: float, confidence: float):
    db = get_db()
    result = (db.table("cia_scores")
               .upsert({               # safe because UNIQUE(upload_id) exists
                   "upload_id": upload_id,
                   "ctr_score": ctr_score,
                   "halflife_days": halflife_days,
                   "confidence": confidence,
               })
               .select()
               .execute())
    return result.data[0]
```

## Supabase migration — run this in Slice 3, not later
Copy the SQL from slice-1-data-pipeline.md and run it in Supabase SQL editor.
Confirm tables exist before writing any code that touches them.
Check: `UNIQUE(upload_id)` constraint on cia_scores must be visible in
Table Editor → cia_scores → Indexes. If missing, upsert will fail with 42P10.

## GitHub Actions keepalive — .github/workflows/keepalive.yml
```yaml
name: Keep HF Space Warm
on:
  schedule:
    - cron: '*/10 * * * *'   # every 10 minutes
  workflow_dispatch:           # manual trigger for testing

jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - name: Ping health endpoint
        run: |
          response=$(curl -s -o /dev/null -w "%{http_code}" \
            ${{ secrets.HF_SPACES_URL }}/health)
          echo "Health check response: $response"
          if [ "$response" != "200" ]; then
            echo "Warning: Space returned $response"
          fi
```
Add `HF_SPACES_URL` to GitHub repo secrets (Settings → Secrets → Actions).
This runs every 10 minutes. HF Spaces sleeps after 15 minutes of inactivity.

## Environment variables for this slice
```
# HF Spaces secrets (set in Space settings UI)
HF_MODEL_REPO=your-username/creative-intelligence-scorer
API_TOKEN=generate-with-python-secrets-token-hex-32

# Railway env vars
HF_SPACES_URL=https://your-username-creative-intelligence-scorer.hf.space
API_TOKEN=same-token-as-above
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_ANON_KEY=eyJxxx...

# GitHub repo secrets
HF_SPACES_URL=same-as-railway
```

## Bug prevention checklist for this slice
- [ ] Run migration SQL before writing any db.py code — confirm tables in Supabase UI
- [ ] Confirm UNIQUE(upload_id) index exists on cia_scores before first upsert
- [ ] map_location='cpu' in model_loader.py — never remove, HF Spaces has no GPU
- [ ] All Supabase writes use .select().execute() — not .execute() alone
- [ ] Pydantic models: every optional field has a default value
- [ ] HF Spaces port: must be 7860 in Dockerfile CMD — not 8000
- [ ] API_TOKEN: same value in HF Spaces secrets AND Railway env vars
- [ ] Test keepalive manually: go to Actions tab, click "Run workflow"
- [ ] Cold start test: let Space sleep 20 minutes, then hit /score, measure time
- [ ] lsof -i :8000 before starting local uvicorn

## Start prompt for Claude Code
```
Starting Slice 3 of Creative Intelligence Agent. Read CLAUDE.md first,
then slice-3-model-serving.md in full.

Goals this session:
1. Create Supabase tables using migration SQL from slice-1 doc — do this first
   before writing any other code. Confirm tables exist before continuing.
2. Write spaces/app.py with /score, /heatmap, /benchmark, /health endpoints
   exactly as specified in the slice doc
3. Write spaces/model_loader.py with singleton pattern and map_location='cpu'
4. Write spaces/Dockerfile and spaces/README.md with correct HF Spaces config
5. Write api/hf_client.py with tenacity retry (3 attempts, 3s wait)
6. Write api/db.py with insert_upload and upsert_score — both using .select()
7. Write .github/workflows/keepalive.yml

Use /plan first. Confirm Supabase table creation before approving plan step 2+.
```

## Done when
- [ ] HF Spaces deployment is live and green (check Space logs)
- [ ] curl /health returns {"status": "ok", "model_loaded": true}
- [ ] curl /score with a test image returns valid JSON with ctr_score and halflife_days
- [ ] curl /heatmap returns a base64 string that renders as an image
- [ ] curl /benchmark?vertical=gaming returns the hardcoded values
- [ ] Cold start < 60 seconds (test: sleep Space, then hit /score, time it)
- [ ] GitHub Actions keepalive shows green in Actions tab
- [ ] Supabase: insert one row manually into cia_uploads, confirm it appears
- [ ] Supabase: upsert into cia_scores twice with same upload_id — confirm no duplicate

## Next slice
Slice 4 — Agentic Layer. Come back to claude.ai first.
