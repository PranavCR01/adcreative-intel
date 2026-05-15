# Slice 5 — Full-Stack Product

## Goal
Live deployed app, shareable URL, demo-ready in 60 seconds without explanation.
By end of this slice: Vercel URL works, full flow end-to-end, 3 demo creatives
preloaded, mobile layout correct at 375px, no raw errors shown to users.

## Architecture decisions locked in this slice
- upload_id generated CLIENT-SIDE with crypto.randomUUID() before upload starts
  (avoids null key bug from previous projects — DB row created with known ID)
- fetchWithRetry wraps every backend call — never bare fetch()
- vercel.json SPA rewrite in frontend/ subdirectory — not project root
- Error boundary uses ERROR_MESSAGES lookup — never String(e) or error.message
- Health check fires 8s after mount, 10s timeout, 2 consecutive failures for banner
- Event delegation for any list of interactive items (suggestion cards, trace items)
- Demo creatives scored at build time — their scores hardcoded, no cold-start wait
- Recharts NOT used — custom SVG for score gauge, CSS bar for benchmark comparison
- Image resized to 224x224 client-side before upload (saves R2 bandwidth)

## File structure for this slice
```
frontend/
├── public/
│   └── demo/
│       ├── gaming-ad.jpg       ← pre-scored demo creative
│       ├── ecommerce-ad.jpg    ← pre-scored demo creative
│       └── finance-ad.jpg      ← pre-scored demo creative
├── src/
│   ├── components/
│   │   ├── UploadZone.tsx
│   │   ├── ScoreCard.tsx
│   │   ├── HeatmapViewer.tsx
│   │   ├── ChatPanel.tsx
│   │   ├── ReasoningTrace.tsx
│   │   ├── BenchmarkBar.tsx
│   │   ├── DemoSelector.tsx
│   │   └── HealthBanner.tsx
│   ├── store/
│   │   ├── traceStore.ts       ← from Slice 4
│   │   └── appStore.ts         ← global app state
│   ├── lib/
│   │   ├── api.ts              ← fetchWithRetry + all API calls
│   │   ├── errors.ts           ← ERROR_MESSAGES lookup
│   │   └── imageUtils.ts       ← client-side resize to 224x224
│   └── App.tsx
├── vercel.json                 ← SPA rewrite — MUST be in frontend/ not root
└── package.json
api/
└── routes/
    └── upload.py               ← /upload endpoint (new this slice)
```

## Client-side ID generation — lib/api.ts
```typescript
// upload_id generated before any network call
// This prevents the null-key deduplication bug from previous projects
export async function uploadCreative(file: File, vertical: string) {
  const uploadId = crypto.randomUUID()   // exists before DB row
  const resized = await resizeImage(file, 224)  // imageUtils.ts
  const formData = new FormData()
  formData.append('image', resized)
  formData.append('vertical', vertical)
  formData.append('upload_id', uploadId)  // pass to backend
  const result = await fetchWithRetry('/api/upload', {
    method: 'POST',
    body: formData,
  })
  return { uploadId, ...result }
}
```

## fetchWithRetry — lib/api.ts
```typescript
const BACKEND_URL = import.meta.env.VITE_API_URL  // Railway URL

export async function fetchWithRetry(
  path: string,
  options: RequestInit = {},
  retries = 3,
  delay = 3000
): Promise<Response> {
  const url = path.startsWith('http') ? path : `${BACKEND_URL}${path}`
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const res = await fetch(url, options)
      if (res.ok) return res
      // Don't retry 4xx errors — only 5xx and network failures
      if (res.status >= 400 && res.status < 500) throw new APIError(res.status)
      if (attempt < retries - 1) await sleep(delay)
    } catch (e) {
      if (e instanceof APIError) throw e
      if (attempt === retries - 1) throw e
      await sleep(delay)
    }
  }
  throw new Error('Max retries exceeded')
}

// Never expose raw error to users
export function getUserMessage(error: unknown): string {
  if (error instanceof APIError) {
    return ERROR_MESSAGES[error.status] ?? ERROR_MESSAGES.default
  }
  return ERROR_MESSAGES.default
}
```

## Error messages lookup — lib/errors.ts
```typescript
export const ERROR_MESSAGES: Record<string | number, string> = {
  401: "Authentication failed. Please refresh the page.",
  404: "Creative not found. Please upload again.",
  422: "Invalid request. Please check your input.",
  500: "Something went wrong on our end. Please try again.",
  503: "The model is warming up. Please wait 30 seconds and try again.",
  default: "Something went wrong. Please try again.",
  model_warming: "Model is warming up (~30 seconds). Hang tight.",
  upload_failed: "Upload failed. Please try a JPG or PNG under 5MB.",
}
// Never use String(error) or error.message directly in UI
```

## imageUtils.ts — client-side resize
```typescript
export async function resizeImage(file: File, size: number): Promise<Blob> {
  return new Promise((resolve) => {
    const img = new Image()
    const url = URL.createObjectURL(file)
    img.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = size
      canvas.height = size
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(img, 0, 0, size, size)
      canvas.toBlob((blob) => {
        URL.revokeObjectURL(url)
        resolve(blob!)
      }, 'image/jpeg', 0.92)
    }
    img.src = url
  })
}
```

## vercel.json — MUST be in frontend/ directory
```json
{
  "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }]
}
```
This file goes in `frontend/vercel.json` — not the project root.
Without it: direct URL navigation returns 404 in production.
Verify this is the first thing added when starting Slice 5.

## appStore.ts — global Zustand state
```typescript
import { create } from 'zustand'

interface Score {
  ctr_score: number
  halflife_days: number
  confidence: number
}

interface AppState {
  uploadId: string | null
  vertical: string
  imageUrl: string | null      // object URL for preview
  score: Score | null
  heatmapB64: string | null
  isScoring: boolean
  isHeatmapping: boolean
  modelWarm: boolean           // false = show warming banner
  // Zustand merge rule — only update defined fields
  setScore: (score: Score) => void
  setHeatmap: (b64: string) => void
  reset: () => void
}

// Merge pattern from previous projects — never overwrite with undefined
const mergeUpdate = <T>(existing: T, update: Partial<T>): T => ({
  ...existing,
  ...Object.fromEntries(Object.entries(update).filter(([_, v]) => v !== undefined))
} as T)
```

## UploadZone.tsx — key requirements
- Drag and drop + click to browse
- Accepts: image/jpeg, image/png only — validate before upload, show error otherwise
- Shows image preview immediately after selection (before upload completes)
- Resize happens client-side before FormData (imageUtils.resizeImage)
- upload_id generated with crypto.randomUUID() before FormData built
- Loading state: spinner + "Uploading and scoring..." message
- Error state: getUserMessage(error) from errors.ts — never raw error

## ScoreCard.tsx — CTR gauge + fatigue display
- CTR gauge: custom SVG arc (not Recharts) — avoids CSS var resolution issues
  - Arc fills from 0 to ctr_score * 180 degrees
  - Color: green (#22c55e) if > benchmark, amber (#f59e0b) if within 20%, red (#ef4444) if below
  - Use hex colors directly in SVG — never CSS vars inside SVG
- Fatigue halflife: plain text "X.X days" with a small benchmark comparison inline
- Confidence: muted small text below score
- Numbers displayed with .toFixed(2) — never raw float

## HeatmapViewer.tsx — overlay toggle
- Two states: original image | heatmap overlay
- Toggle button between them (not a tab — avoid display:none)
- Heatmap is the base64 PNG from /heatmap endpoint overlaid at 40% opacity
- High/low attention regions listed as text below the image
- Image dimensions fixed at 224x224px — no reflow when toggling

## ChatPanel.tsx + ReasoningTrace.tsx
- ChatPanel: message bubbles, input at bottom, submit on Enter
- User messages: right-aligned, light background
- Agent messages: left-aligned, white card
- ReasoningTrace: collapsible accordion below each agent message
  - Shows tool calls from traceStore (Map → Array.from for render)
  - Each trace item: tool name, duration_ms, key output values
  - Collapsed by default — click to expand
  - Event delegation: one click handler on the trace container,
    not individual handlers on each item (avoids re-attachment bug)

## HealthBanner.tsx — cold start warning
```typescript
// Fires 8s after mount — not immediately
// 10s timeout on health check
// Shows banner only after 2 consecutive failures
// Dismisses automatically when health check succeeds
useEffect(() => {
  let failures = 0
  const check = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/health`, { signal: AbortSignal.timeout(10000) })
      if (res.ok) {
        setModelWarm(true)
        failures = 0
      } else {
        failures++
        if (failures >= 2) setModelWarm(false)
      }
    } catch {
      failures++
      if (failures >= 2) setModelWarm(false)
    }
  }
  const timer = setTimeout(() => {
    check()
    const interval = setInterval(check, 30000)
    return () => clearInterval(interval)
  }, 8000)   // 8s delay before first check
  return () => clearTimeout(timer)
}, [])
```

## DemoSelector.tsx — 3 preloaded creatives
```typescript
const DEMO_CREATIVES = [
  {
    id: "demo-gaming",
    label: "Gaming — mobile RPG",
    imagePath: "/demo/gaming-ad.jpg",
    vertical: "gaming",
    // Pre-scored — no API call needed for demos
    score: { ctr_score: 0.41, halflife_days: 6.2, confidence: 0.81 },
    heatmap_b64: "...",  // pre-generated base64
  },
  {
    id: "demo-ecommerce",
    label: "E-commerce — fashion",
    imagePath: "/demo/ecommerce-ad.jpg",
    vertical: "ecommerce",
    score: { ctr_score: 0.56, halflife_days: 12.4, confidence: 0.87 },
    heatmap_b64: "...",
  },
  {
    id: "demo-finance",
    label: "Finance — trading app",
    imagePath: "/demo/finance-ad.jpg",
    vertical: "finance",
    score: { ctr_score: 0.33, halflife_days: 8.9, confidence: 0.79 },
    heatmap_b64: "...",
  },
]
// Selecting a demo immediately populates the full UI without any API call
// So even if HF Spaces is cold, hiring managers can see the full experience
```

## api/routes/upload.py — /upload endpoint
```python
from fastapi import APIRouter, UploadFile, Form, HTTPException
from api.hf_client import score_image, get_heatmap
from api.db import insert_upload, upsert_score
import boto3, os

router = APIRouter()
r2 = boto3.client('s3',
    endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
    aws_access_key_id=os.getenv('R2_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('R2_SECRET_KEY'),
)

@router.post("/upload")
async def upload_creative(
    image: UploadFile,
    vertical: str = Form(...),
    upload_id: str = Form(...),    # client-generated UUID
    session_id: str = Form(""),   # optional — default prevents 422
):
    image_bytes = await image.read()
    # 1. Upload to R2
    r2_key = f"creatives/{upload_id}.jpg"
    try:
        r2.put_object(Bucket=os.getenv('R2_BUCKET'), Key=r2_key, Body=image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Upload storage failed.")
    # 2. Insert into Supabase (upload_id from client — no null key risk)
    await insert_upload(upload_id, r2_key, vertical, session_id or upload_id)
    # 3. Score via HF Spaces
    try:
        score_result = await score_image(image_bytes, vertical)
    except Exception:
        raise HTTPException(status_code=503, detail="Model is warming up. Please retry.")
    # 4. Upsert score (UNIQUE constraint prevents duplicates)
    await upsert_score(
        upload_id,
        score_result["ctr_score"],
        score_result.get("halflife_days"),
        score_result["confidence"]
    )
    return {"upload_id": upload_id, **score_result}
```

## R2 CORS configuration
Set this in Cloudflare R2 bucket settings → CORS policy:
```json
[{
  "AllowedOrigins": ["https://your-app.vercel.app", "http://localhost:5173"],
  "AllowedMethods": ["GET", "PUT", "POST"],
  "AllowedHeaders": ["*"]
}]
```
Do this before any frontend work — CORS errors in production are hard to debug.

## Environment variables for this slice
```
# Frontend (.env) — all must start with VITE_
VITE_API_URL=https://your-railway-app.railway.app

# Railway (existing + new)
R2_ACCOUNT_ID=
R2_ACCESS_KEY=
R2_SECRET_KEY=
R2_BUCKET=creative-intelligence-agent
```

## Bug prevention checklist for this slice
- [ ] vercel.json in frontend/ — check this is the FIRST thing committed
- [ ] crypto.randomUUID() for upload_id on client before FormData
- [ ] fetchWithRetry wraps every API call — grep for bare fetch() and replace
- [ ] ERROR_MESSAGES used everywhere — grep for .message and String(e) in JSX
- [ ] ScoreCard SVG uses hex colors — grep for var(-- inside SVG elements
- [ ] Health check: setTimeout 8000ms, not 0. Two failures before banner.
- [ ] DemoSelector: pre-scored scores hardcoded — no API call on demo selection
- [ ] R2 CORS policy set before first frontend upload test
- [ ] session_id = Form("") in upload.py — not required
- [ ] lsof -i :8000 before starting local uvicorn

## Start prompt for Claude Code
```
Starting Slice 5 of Creative Intelligence Agent. Read CLAUDE.md, then
slice-5-fullstack-product.md in full before writing anything.

Goals this session — do in this order:
1. Create frontend/vercel.json with SPA rewrite — do this FIRST
2. Write frontend/src/lib/api.ts with fetchWithRetry and uploadCreative
3. Write frontend/src/lib/errors.ts with ERROR_MESSAGES lookup
4. Write frontend/src/lib/imageUtils.ts with resizeImage
5. Write frontend/src/store/appStore.ts
6. Write all components: UploadZone, ScoreCard, HeatmapViewer,
   ChatPanel, ReasoningTrace, BenchmarkBar, DemoSelector, HealthBanner
7. Write api/routes/upload.py
8. Update api/main.py to include upload router

Use /plan first. vercel.json must be step 1 in the plan — flag if it isn't.
Mobile-first layout, 375px minimum. No Recharts — custom SVG for gauge.
```

## Done when
- [ ] vercel.json in frontend/ — direct URL navigation works in production
- [ ] Full flow: select demo creative → scores appear without any API call
- [ ] Full flow: upload new image → score → heatmap → chat question → grounded answer
- [ ] Health banner: appears if Railway is cold, dismisses when warm
- [ ] Error states: upload failure, scoring failure, chat failure all show human messages
- [ ] Mobile: layout correct at 375px (test in Chrome DevTools)
- [ ] No raw error strings visible anywhere in the UI
- [ ] R2 CORS: image upload works from Vercel domain (not just localhost)

## Next slice
Slice 6 — Polish and Pitch. Come back to claude.ai first.
