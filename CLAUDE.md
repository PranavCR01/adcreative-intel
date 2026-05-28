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
- Vision model: SigLIP 2 (google/siglip2-base-patch16-224) frozen + trainable multi-task head (PyTorch)
- Agentic layer: Claude Haiku 4.5 via Anthropic API (native tool use)
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
Input → SigLIP 2 (google/siglip2-base-patch16-224, FROZEN, no grad) → 768-dim embedding (pooler_output)
→ Projection (768→256, ReLU, Dropout 0.2)
→ CTR head (256→1, Sigmoid) + Fatigue head (256→2, Weibull params)
Loss = 0.5 * BCELoss(ctr) + 0.5 * WeibullNLLLoss(fatigue)
Fatigue labels are right-censored (ad still running = censored=True)
Weibull clamp: log inputs clamped to [-10, 10] — never remove this
Backbone attribute: self.backbone (was self.clip — never revert)

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
get_fatigue_projection(halflife_days) → {day_7, day_14, day_21, recommendation}

## Vertical classifier
- Artifact: model/vertical_classifier.pkl (LogisticRegression, 768-dim SigLIP 2 embeddings)
- CV accuracy: 79.7% (4-class: gaming/ecommerce/finance/other), 5-fold stratified
- HF Hub: vertical_classifier.pkl uploaded to pcr12/creative-intelligence-scorer
- Loaded at HF Spaces startup via get_classifier() — non-fatal if pkl unavailable
- /score endpoint now returns predicted_vertical + vertical_confidence (float 0-1)
- Frontend auto-selects detected vertical on upload; shows indigo chip badge; clears on manual override or reset

## Supabase tables
cia_uploads: id, created_at, r2_key, vertical, user_session
cia_scores: upload_id (UNIQUE), ctr_score, halflife_days, confidence, heatmap_regions (jsonb), scored_at
cia_sessions: id, created_at, messages (jsonb)

## Environment variables needed
SUPABASE_URL, SUPABASE_ANON_KEY
R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET
HF_SPACES_URL, HF_TOKEN, API_TOKEN

## Deployments
- Frontend: https://adcreative-intel.vercel.app (live, Vercel)
- Backend: https://adcreative-intel.onrender.com (live, Render)

## Current slice
SLICE: 6 — Launch
STATUS: Pending

NEXT ACTIONS:
1. README — project overview, architecture diagram, demo GIF
2. Demo video
3. LinkedIn post
4. Pitch email to Moloco

## Open bugs
1. Blank trace steps for uploaded images in chat UI (low priority)

## Pending
- Stress test suite: tests/test_suite.py written — not yet run against live services
- Embedding cache stale: data/clip_embeddings.pt built with CLIP — delete and regenerate with SigLIP 2 before next training run
- Render env var: set ALLOWED_ORIGINS=https://adcreative-intel.vercel.app,http://localhost:5173 in Render dashboard

MODEL: model_siglip2_best.pt — Spearman r=0.645 (SigLIP 2, epoch 18/30)
HF HUB: https://huggingface.co/pcr12/creative-intelligence-scorer

COMPLETED:
- Slice 1: data pipeline, 22,248 rows, labels_clean.csv
- Slice 2: training complete, best_alpha checkpoint, pushed to HF Hub
- Slice 3 files written: spaces/, api/hf_client.py,
  api/db.py, .github/workflows/keepalive.yml
- Slice 4: agentic layer — tools, ReAct loop, chat endpoint, traceStore
- Slice 5: full-stack product — COMPLETE and deployed
  - Frontend: React+TS+Tailwind, hash router, zustand store
  - Pages: Landing, Analyzer, Benchmark
  - Upload endpoint, Supabase Storage, R2 integration
  - Demo images live at frontend/public/demo/
- Architecture audit: all HIGH/MEDIUM issues resolved
  - TRACE_LOG race condition fixed (request-local trace list)
  - heatmap_regions now saved to cia_scores on every upload
  - Persistent httpx.AsyncClient (no per-request open/close)
  - Single Anthropic client in tools.py (_get_client())
  - Supabase singleton double-init guarded with threading.Lock
  - CORS origins from env var ALLOWED_ORIGINS
  - tenacity retry only on 5xx (not 4xx)
- SigLIP 2 swap: spaces/, model/, frontend/ all updated to google/siglip2-base-patch16-224
- Slice 5a AI upgrades (Features 1-3):
  - Feature 1: Vertical classifier — LogisticRegression on SigLIP 2 embeddings, 79.7% CV acc
    pkl on HF Hub; /score returns predicted_vertical + vertical_confidence
    Frontend auto-selects + shows badge; clears on reset or manual override
  - Feature 2: Confidence-aware agent — CONFIDENCE CALIBRATION block in system prompt
    LOW (<0.15): flags explicitly; MODERATE (0.15-0.40): hedged; HIGH (>0.40): authoritative
  - Feature 3: Fatigue projection tool — get_fatigue_projection(halflife_days)
    Weibull retention at day 7/14/21, shape=1.5; recommendation string

## Apify data status
- Run 1 complete: 589 ads, gaming/local mix, NO start/stop dates (active_status=active)
- Run 2-4 pending: use active_status=all to get inactive ads WITH dates
- Corrected URL format:
  https://www.facebook.com/ads/library/?active_status=all&ad_type=all&country=US&q=KEYWORD&search_type=keyword_unordered

## Hard constraints
- SigLIP 2 backbone ALWAYS frozen — never set requires_grad=True on self.backbone
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
