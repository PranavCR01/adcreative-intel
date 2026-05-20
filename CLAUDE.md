# Creative Intelligence Agent — Session Context

## What this project is
Multi-Task Creative Lifespan Prediction — a full-stack AI tool that predicts
ad creative CTR and fatigue half-life from raw images, with an agentic
explanation layer. Pitched to Moloco, AppLovin, Liftoff as "the open-source
creative scoring layer." Mirrors Kitada et al. (2022) SOTA architecture.

## Stack
- Frontend: React + TypeScript + Tailwind + shadcn/ui → Vercel
- Backend: FastAPI (Python) → Railway free tier
- Model serving: HF Spaces (FastAPI, 16GB RAM free tier)
- Storage: Cloudflare R2 (uploaded creatives)
- DB: Supabase (reuse existing free account, prefix tables with cia_)
- Vision model: CLIP-ViT-B/32 frozen + trainable multi-task head (PyTorch)
- Agentic layer: Qwen2.5-1.5B via smolagents + HF Inference API
- Training: Hugging Face Trainer + PEFT

## Repo structure
```
adcreative-intel/
├── data/
│   ├── raw/{vertical}/          ← Apify real ads
│   ├── synthetic/{vertical}/    ← PIL generated ads
│   ├── apify_raw/               ← raw JSON from Apify runs
│   └── labels_clean.csv         ← final merged dataset
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   └── dataset_research.md      ← ML Intern research output
├── scripts/
│   ├── apify_downloader.py
│   ├── synthetic_gen.py
│   └── build_labels.py
├── model/
│   ├── clip_head.py             ← YOU write this (ML depth signal)
│   ├── train.py
│   ├── survival.py              ← YOU write this (ML depth signal)
│   ├── dataset.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── push_to_hub.py
├── agent/
│   ├── tools.py
│   ├── agent.py
│   └── prompts.py
├── api/
│   ├── main.py
│   ├── hf_client.py
│   ├── db.py
│   └── routes/
│       ├── upload.py
│       └── chat.py
├── spaces/
│   ├── app.py
│   ├── model_loader.py
│   ├── Dockerfile
│   └── README.md
└── frontend/src/
```

## Model architecture (do not change without asking)
Input → CLIP-ViT-B/32 (FROZEN, no grad) → 768-dim embedding (pooler_output)
→ Projection (768→256, ReLU, Dropout 0.2)
→ CTR head (256→1, Sigmoid) + Fatigue head (256→2, Weibull params)
Loss = 0.5 * BCELoss(ctr) + 0.5 * WeibullNLLLoss(fatigue)
Fatigue labels are right-censored (ad still running = censored=True)
Weibull clamp: log inputs clamped to [-10, 10] — never remove this

## Research-validated design decisions (from Kitada 2022, Chen 2025)
- Frozen backbone: confirmed by literature for datasets <100K images
- Survival loss: Kitada 2022 uses hazard-function loss for fatigue — Weibull is equivalent
- Two discontinuation types: halflife_days < 7 = "cut-out", >= 7 = "wear-out"
  Add discontinuation_type column to labels_clean.csv
- Eval: Spearman rank correlation (primary, r > 0.30) + MAE (secondary, < 0.15)
  CTR AUC removed — binarizing continuous proxy labels at median is meaningless
- Face presence: add to synthetic label rules (face detected → halflife *= 1.2)
- Portfolio framing: "Multi-Task Creative Lifespan Prediction" — use this exact phrase

## Data schema — labels_clean.csv (locked)
ad_id, image_path, ctr_score (0-1), halflife_days (float, null if censored),
censored (bool), vertical (gaming|ecommerce|finance|other),
source (apify|synthetic), discontinuation_type (cut-out|wear-out|censored),
scraped_at (ISO 8601)

## Agent tools (API contract — do not change)
get_creative_score(image_id) → {ctr_score, halflife_days, confidence}
get_heatmap_regions(image_id) → {high_attention: [regions], low_attention: [regions]}
get_benchmark(vertical) → {median_ctr, median_halflife, sample_size}
get_improvement_suggestions(image_id) → {suggestions: [str x3]}

## Supabase tables
cia_uploads: id, created_at, r2_key, vertical, user_session
cia_scores: upload_id (UNIQUE), ctr_score, halflife_days, confidence, scored_at
cia_sessions: id, created_at, messages (jsonb)

## Environment variables needed
SUPABASE_URL, SUPABASE_ANON_KEY
R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET
HF_SPACES_URL, HF_TOKEN, API_TOKEN

## Current slice
SLICE: 5 — Full-Stack Product
STATUS: Code complete, zero TS errors. Ready for build + deploy.
NEXT ACTION:
1. cd frontend && npm run build — verify production build passes
2. git add -A && git commit -m "Slice 5: full-stack frontend + upload endpoint"
3. git push origin master
4. Deploy frontend/ to Vercel — set root directory to frontend/
5. Add to Render env vars: SUPABASE_URL, SUPABASE_ANON_KEY,
   HF_SPACES_URL, API_TOKEN, ANTHROPIC_API_KEY
6. Set VITE_API_URL=https://adcreative-intel.onrender.com in Vercel
   environment variables
7. Test full flow: demo creative loads → scores show → chat works

MODEL: model/best_alpha/ — Spearman r=0.2542, ALPHA=0.9
HF HUB: https://huggingface.co/pcr12/creative-intelligence-scorer

COMPLETED:
- Slice 1: data pipeline, 22,248 rows, labels_clean.csv
- Slice 2: training complete, best_alpha checkpoint, pushed to HF Hub
- Slice 3 files written: spaces/, api/hf_client.py,
  api/db.py, .github/workflows/keepalive.yml
- Slice 4: agentic layer — tools, ReAct loop, chat endpoint, traceStore
- Slice 5 code complete:
  - Pre-work: 3 demo images copied to frontend/public/demo/
  - frontend/vercel.json (SPA rewrite)
  - Vite+React+TS scaffolded, zustand installed, styles.css imported
  - frontend/src/lib/errors.ts, imageUtils.ts, api.ts
  - frontend/src/store/appStore.ts
  - frontend/src/components/icons.tsx, HealthBanner.tsx
  - frontend/src/App.tsx (hash router)
  - frontend/src/pages/Landing.tsx, Analyzer.tsx, Benchmark.tsx
  - frontend/.env (VITE_API_URL=https://adcreative-intel.onrender.com)
  - api/routes/upload.py (Supabase Storage)
  - api/main.py updated with upload router

## Apify data status
- Run 1 complete: 589 ads, gaming/local mix, NO start/stop dates (active_status=active)
- Run 2-4 pending: use active_status=all to get inactive ads WITH dates
- Corrected URL format:
  https://www.facebook.com/ads/library/?active_status=all&ad_type=all&country=US&q=KEYWORD&search_type=keyword_unordered

## Hard constraints
- CLIP backbone ALWAYS frozen — never set requires_grad=True on it
- clip_head.py and survival.py written by YOU, never ML Intern
- Never store image pixel data in Supabase — R2 only
- All scores stored in cia_scores, never recomputed on the fly
- Batch size for CLIP extraction: 64 (reduce to 32 if OOM on 1650 Ti)
- Frontend never calls HF Spaces directly — all inference via Railway API
- Use /plan before every implementation prompt
- /compact every ~45 minutes in Claude Code sessions
- Supabase writes always use .select().execute() — never silent failures
- vercel.json goes in frontend/ not project root
- upload_id generated client-side with crypto.randomUUID()
- fetchWithRetry wraps every API call — never bare fetch()
- Error boundaries use ERROR_MESSAGES lookup — never String(e) or error.message
