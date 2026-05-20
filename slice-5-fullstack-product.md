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
- Image resized to 224x224 client-side before upload (saves storage bandwidth)
- Image storage: Supabase Storage bucket "creatives" (NOT R2/Cloudflare)
- Backend host: Render at https://adcreative-intel.onrender.com (NOT Railway)
- Agent LLM: Claude Haiku via Anthropic SDK manual ReAct loop (NOT smolagents/Qwen)

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
│   │   ├── icons.tsx           ← from design/icons.jsx
│   │   └── HealthBanner.tsx
│   ├── pages/
│   │   ├── Landing.tsx         ← from design/landing.jsx
│   │   ├── Analyzer.tsx        ← from design/analyzer.jsx
│   │   └── Benchmark.tsx       ← from design/benchmark.jsx
│   ├── store/
│   │   ├── traceStore.ts       ← from Slice 4
│   │   └── appStore.ts         ← global app state
│   ├── lib/
│   │   ├── api.ts              ← fetchWithRetry + all API calls
│   │   ├── errors.ts           ← ERROR_MESSAGES lookup
│   │   └── imageUtils.ts       ← client-side resize to 224x224
│   ├── styles.css              ← from design/styles.css (light theme)
│   └── App.tsx                 ← from design/app.jsx (hash router)
├── .env                        ← VITE_API_URL
├── vercel.json                 ← SPA rewrite — MUST be in frontend/ not root
└── package.json
api/
└── routes/
    └── upload.py               ← /upload endpoint (new this slice)
```

## Design files — use these as implementation reference
Claude Design files are in design/ folder. Convert them to proper
React+TypeScript components. Keep all component logic exactly as written.
Convert window.XXX exports to proper TS named exports.

## Demo creatives — REAL MODEL SCORES (do not use placeholder values)
```typescript
const DEMO_CREATIVES = {
  gaming: {
    ad_id: "apify_f321af3f4e59",
    label: "Gaming · Mobile RPG",
    vertical: "gaming",
    ctr_score: 0.3179,
    halflife_days: 33.04,
    confidence: 0.3641,
    dtype: "wear-out",
    // Copy file from data/raw/gaming/apify_f321af3f4e59.jpg
    // → frontend/public/demo/gaming-ad.jpg
  },
  ecommerce: {
    ad_id: "apify_6d945f5a1b9d",
    label: "Ecommerce · DTC",
    vertical: "ecommerce",
    ctr_score: 0.5063,
    halflife_days: 47.99,
    confidence: 0.0126,   // NOTE: very low confidence — show indicator in UI
    dtype: "wear-out",
    // Copy file from data/raw/ecommerce/apify_6d945f5a1b9d.jpg
    // → frontend/public/demo/ecommerce-ad.jpg
  },
  finance: {
    ad_id: "synthetic_591690dcf2f2",
    label: "Finance · Neobank",
    vertical: "finance",
    ctr_score: 0.0993,
    halflife_days: 12.0,
    confidence: 0.8014,
    dtype: "wear-out",
    // Copy file from data/synthetic/finance/synthetic_591690dcf2f2.jpg
    // → frontend/public/demo/finance-ad.jpg
  },
}
// Selecting a demo immediately populates the full UI without any API call
// So even if HF Spaces is cold, hiring managers can see the full experience
```

## Benchmark values — from real training data
```typescript
const BENCHMARKS = {
  gaming:    { median_ctr: 0.119, median_halflife: 10.7, sample_size: 7485 },
  ecommerce: { median_ctr: 0.125, median_halflife: 11.2, sample_size: 7078 },
  finance:   { median_ctr: 0.111, median_halflife: 10.0, sample_size: 6520 },
}
```

## Client-side ID generation — lib/api.ts
```typescript
const BACKEND_URL = import.meta.env.VITE_API_URL  // Render URL

export async function uploadCreative(file: File, vertical: string) {
  const uploadId = crypto.randomUUID()   // exists before DB row
  const resized = await resizeImage(file, 224)  // imageUtils.ts
  const formData = new FormData()
  formData.append('image', resized)
  formData.append('vertical', vertical)
  formData.append('upload_id', uploadId)
  const res = await fetchWithRetry(`${BACKEND_URL}/upload`, {
    method: 'POST',
    body: formData,
  })
  const result = await res.json()
  return { uploadId, ...result }
}

export async function sendChat(imageId: string, message: string) {
  const res = await fetchWithRetry(`${BACKEND_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: imageId, message }),
  })
  return res.json()
}
```

## fetchWithRetry — lib/api.ts
```typescript
export async function fetchWithRetry(
  url: string,
  options: RequestInit = {},
  retries = 3,
  delay = 3000
): Promise<Response> {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const res = await fetch(url, options)
      if (res.ok) return res
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

export function getUserMessage(error: unknown): string {
  if (error instanceof APIError) {
    return ERROR_MESSAGES[error.status] ?? ERROR_MESSAGES.default
  }
  return ERROR_MESSAGES.default
}

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))

class APIError extends Error {
  constructor(public status: number) { super(`API error ${status}`) }
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
This is the FIRST file created in this slice.

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
  imageUrl: string | null
  score: Score | null
  heatmapB64: string | null
  isScoring: boolean
  isHeatmapping: boolean
  modelWarm: boolean
  setScore: (score: Score) => void
  setHeatmap: (b64: string) => void
  reset: () => void
}
```

## HealthBanner.tsx — cold start warning
```typescript
useEffect(() => {
  let failures = 0
  const BACKEND_URL = import.meta.env.VITE_API_URL
  const check = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/health`, {
        signal: AbortSignal.timeout(10000)
      })
      if (res.ok) { setModelWarm(true); failures = 0 }
      else { failures++; if (failures >= 2) setModelWarm(false) }
    } catch {
      failures++
      if (failures >= 2) setModelWarm(false)
    }
  }
  const timer = setTimeout(() => {
    check()
    const interval = setInterval(check, 30000)
    return () => clearInterval(interval)
  }, 8000)
  return () => clearTimeout(timer)
}, [])
```

## api/routes/upload.py — /upload endpoint
Uses Supabase Storage (NOT R2/Cloudflare).

```python
from fastapi import APIRouter, UploadFile, Form, HTTPException
from api.hf_client import score_image, get_heatmap
from api.db import get_db, insert_upload, upsert_score
import os, uuid

router = APIRouter()

@router.post("/upload")
async def upload_creative(
    image: UploadFile,
    vertical: str = Form(...),
    upload_id: str = Form(...),    # client-generated UUID
    session_id: str = Form(""),   # optional — default prevents 422
):
    image_bytes = await image.read()

    # 1. Upload to Supabase Storage bucket "creatives"
    storage_path = f"creatives/{upload_id}.jpg"
    try:
        db = get_db()
        db.storage.from_("creatives").upload(
            path=storage_path,
            file=image_bytes,
            file_options={"content-type": "image/jpeg"}
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Upload storage failed.")

    # 2. Insert into Supabase DB
    await insert_upload(upload_id, storage_path, vertical, session_id or upload_id)

    # 3. Score via HF Spaces
    try:
        score_result = await score_image(image_bytes, vertical)
    except Exception:
        raise HTTPException(status_code=503, detail="Model is warming up. Please retry.")

    # 4. Get heatmap via HF Spaces
    try:
        heatmap_result = await get_heatmap(image_bytes)
    except Exception:
        heatmap_result = {"heatmap_b64": "", "high_attention": [], "low_attention": []}

    # 5. Upsert score
    await upsert_score(
        upload_id,
        score_result["ctr_score"],
        score_result.get("halflife_days"),
        score_result["confidence"]
    )

    return {
        "upload_id": upload_id,
        **score_result,
        "heatmap_b64": heatmap_result.get("heatmap_b64", ""),
        "high_attention": heatmap_result.get("high_attention", []),
        "low_attention": heatmap_result.get("low_attention", []),
    }
```

## Wiring chat to real backend
Replace the mock `buildReply()` function in Analyzer.tsx with a real API call:

```typescript
// In Analyzer.tsx — replace buildReply() call with:
const send = async (text: string) => {
  if (!text.trim() || busy) return
  const userMsg = { role: 'user', time: new Date().toLocaleTimeString(), text }
  setMessages(prev => [...prev, userMsg])
  setText('')
  setBusy(true)
  try {
    const result = await sendChat(currentImageId, text)
    const agentMsg = {
      role: 'agent',
      time: new Date().toLocaleTimeString(),
      latency: result.trace?.reduce((s: number, t: any) => s + (t.duration_ms || 0), 0),
      text: result.answer,
      trace: result.trace,
      open: false,
    }
    setMessages(prev => [...prev, agentMsg])
  } catch (err) {
    const errMsg = {
      role: 'agent',
      time: new Date().toLocaleTimeString(),
      text: getUserMessage(err),
      trace: [],
      open: false,
    }
    setMessages(prev => [...prev, errMsg])
  } finally {
    setBusy(false)
  }
}
```

For demo creatives, use the demo ad_id as the imageId.
For uploaded creatives, use the upload_id returned from /upload.

## Environment variables
```
# frontend/.env
VITE_API_URL=https://adcreative-intel.onrender.com

# Render environment variables (add these in Render dashboard)
SUPABASE_URL=https://ecgglpptvhjkpjqtgoa.supabase.co
SUPABASE_ANON_KEY=your_anon_key
HF_SPACES_URL=https://pcr12-creative-intelligence-scorer.hf.space
API_TOKEN=your_api_token
ANTHROPIC_API_KEY=your_anthropic_key
```

## Bug prevention checklist for this slice
- [ ] vercel.json in frontend/ — FIRST thing committed
- [ ] crypto.randomUUID() for upload_id on client before FormData
- [ ] fetchWithRetry wraps every API call — no bare fetch()
- [ ] ERROR_MESSAGES used everywhere — no String(e) or error.message in UI
- [ ] ScoreCard SVG uses hex colors — no CSS vars inside SVG elements
- [ ] Health check: setTimeout 8000ms. Two failures before banner shows.
- [ ] Demo creatives load without any API call (cold-start proof)
- [ ] Ecommerce demo shows low-confidence indicator (confidence=0.0126)
- [ ] Architecture section in Landing.tsx: "Claude Haiku · ReAct agent" not Qwen/smolagents
- [ ] session_id = Form("") in upload.py — not required field
- [ ] Demo image files copied to frontend/public/demo/

## Done when
- [ ] vercel.json in frontend/ — direct URL navigation works in production
- [ ] Full flow: select demo creative → scores appear without any API call
- [ ] Full flow: upload new image → score → heatmap → chat → grounded answer
- [ ] Health banner: appears if Render is cold, dismisses when warm
- [ ] Error states: upload, scoring, chat failures all show human-readable messages
- [ ] Mobile: layout correct at 375px
- [ ] No raw error strings visible anywhere in the UI
- [ ] Deployed to Vercel — live URL working

## Next slice
Slice 6 — Polish and Pitch. Come back to claude.ai first.
