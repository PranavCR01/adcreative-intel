# Creative Intelligence Agent

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-SigLIP%202-FFD21E?style=flat&logo=huggingface&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)
![Claude API](https://img.shields.io/badge/Claude-Haiku%204.5-6B48FF?style=flat&logo=anthropic&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?style=flat&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-5.x-646CFF?style=flat&logo=vite&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind%20CSS-3.x-06B6D4?style=flat&logo=tailwindcss&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=flat&logo=supabase&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-Frontend-000000?style=flat&logo=vercel&logoColor=white)
![Render](https://img.shields.io/badge/Render-Backend-46E3B7?style=flat&logo=render&logoColor=white)
![HF Spaces](https://img.shields.io/badge/HF%20Spaces-Inference-FFD21E?style=flat&logo=huggingface&logoColor=black)

**Multi-task vision model that predicts ad creative CTR and fatigue half-life from raw images, with an agentic explanation layer powered by Claude.**

---

## Links

| | |
|---|---|
| Live Demo | [adcreative-intel.vercel.app](https://adcreative-intel.vercel.app) |
| Demo Video | [Youtube Link](https://youtu.be/S3rk1Sr2sWU) |
| Research Notes | [CIA Research & Architecture Notes](https://www.notion.so/CIA-Research-Architecture-Notes-36ea47a2b4cf80d693b8ef5a6aa32209) |
| Dev Journal | [CIA Project Arc](https://www.notion.so/CIA-Project-Arc-36ea47a2b4cf80aabc94c1ee6ece0570) |
| GitHub | [PranavCR01/adcreative-intel](https://github.com/PranavCR01/adcreative-intel) |
| Model Hub | [pcr12/creative-intelligence-scorer](https://huggingface.co/pcr12/creative-intelligence-scorer) |

---

## The Problem

Performance marketers make creative decisions mostly on intuition. A gaming ad goes live, burns through budget for two weeks, CTR collapses — and by the time the data is in, the spend is gone. The standard workflow is to run creatives until they die, then post-mortem. There is no pre-flight signal that tells you how long a creative will last or whether its click-through rate will be above or below vertical median.

The deeper problem is that "CTR" is only half the picture. A creative with a strong CTR but a three-day fatigue half-life is worse than a moderate creative that runs for a month without degrading. Fatigue is a survival problem — it has a different mathematical structure than click prediction — and almost no tooling treats it that way. Existing creative analytics platforms score aesthetics or brand compliance; none model the time-to-discontinuation distribution directly from the image.

This project is an attempt to close that gap: a vision model trained jointly on CTR regression and Weibull survival loss, grounded in Kitada et al. (2022) and deployed as a full-stack product with an agentic layer that can explain its own predictions, compare against vertical benchmarks, and project fatigue curves out to day 21.

---

## What CIA Does

**Step 1 — Upload**
Drop any ad creative (PNG or JPEG, up to 8MB). The image is uploaded to Cloudflare R2, a vertical classifier auto-detects whether it is gaming, ecommerce, or finance, and the image is forwarded to the inference endpoint on HuggingFace Spaces.

**Step 2 — Score**
The frozen SigLIP 2 backbone extracts a 768-dim pooler embedding. The multi-task head produces two outputs simultaneously: a predicted CTR score (0–1, sigmoid) and two Weibull parameters (lambda, k) that define the fatigue survival curve. A GradCAM pass over layer 11 generates the attention heatmap. All results are written to Supabase and returned to the frontend in a single response.

**Step 3 — Explain**
Claude Haiku 4.5 runs a ReAct loop with five grounded tools: `get_creative_score`, `get_heatmap_regions`, `get_benchmark`, `get_improvement_suggestions`, and `get_fatigue_projection`. The agent cannot fabricate numbers — every claim it makes must come from a tool call result. It compares the creative against vertical benchmarks, projects retention at day 7/14/21, and suggests concrete improvements. Conversation history is maintained across turns so follow-up questions work naturally.

---

## Architecture

### System Overview

```
Browser
  │
  ├─ Upload image ──────────────────────────────────────────────────────┐
  │                                                                       │
  ▼                                                                       ▼
Vercel (React + TS)                                              Cloudflare R2
  │                                                              (raw image store)
  │  POST /upload, POST /chat
  ▼
Render (FastAPI backend)
  │
  ├─ /upload ──► HF Spaces (SigLIP 2 inference) ──► score + heatmap_b64
  │                    │
  │              vertical_classifier.pkl (LogisticRegression, 768-dim)
  │
  ├─ /chat ───► run_agent(image_id, message, vertical, history)
  │                    │
  │              Claude Haiku 4.5
  │              ReAct loop (max 5 steps)
  │              ├─ get_creative_score      (reads cia_scores)
  │              ├─ get_heatmap_regions     (reads cia_scores)
  │              ├─ get_benchmark           (hardcoded vertical stats)
  │              ├─ get_improvement_suggestions  (Claude sub-call)
  │              └─ get_fatigue_projection  (Weibull math, no I/O)
  │
  └─ Supabase
       ├─ cia_uploads   (metadata, r2_key, vertical, session)
       └─ cia_scores    (ctr_score, halflife_days, confidence, heatmap_regions jsonb)
```

### Model Design

The model is a frozen SigLIP 2 backbone (`google/siglip2-base-patch16-224`) with a trainable multi-task head. The backbone is never updated — it provides 768-dim `pooler_output` embeddings and is treated as a fixed feature extractor throughout training and inference.

The head is a two-branch architecture: a shared projection layer (768→256, ReLU, Dropout 0.2) feeds into a CTR branch (256→1, Sigmoid) and a fatigue branch (256→2, producing Weibull lambda and k parameters). Total loss is `0.5 * BCELoss(ctr) + 0.5 * WeibullNLLLoss(fatigue)`, where the Weibull loss is a right-censored negative log-likelihood — creatives that were still running at scrape time are marked `censored=True` and contribute only to the survival term, not the event term.

The decision to freeze the backbone follows Kitada et al. (2022), who showed that fine-tuning large vision encoders on datasets below ~100K images degrades performance through overfitting. The Weibull parameterization is mathematically equivalent to the hazard-function loss used in Kitada's fatigue model and has the advantage of producing interpretable shape (k) and scale (lambda) parameters that map directly to a survival curve. The final model (SigLIP 2, epoch 18/30) achieves Spearman r=0.645 on a real-ads-only held-out test set.

### Agent Design

The agent is a five-tool ReAct loop running Claude Haiku 4.5 with a hard cap of five steps. The system prompt encodes eleven rules that enforce grounding: the agent cannot state a CTR number without first calling `get_creative_score`, cannot compare to a benchmark without calling `get_benchmark`, and cannot project fatigue without calling `get_fatigue_projection`. Rule 10 prohibits re-calling any tool whose result already appears in conversation history, preventing redundant API calls on follow-up turns.

Conversation memory works by replaying the full `history[]` array on each turn, then prepending a synthetic two-message exchange that re-injects `[Creative ID: X] [Vertical: Y (confidence: Z%)]` so the agent retains context across turns without the frontend re-sending raw scores. The vertical confidence feeds the agent's calibration rules: below 70% it adds a one-sentence caveat; above 40% model confidence it speaks with authority; below 15% it flags the prediction as low-confidence and withholds strong recommendations.

### Vertical Classifier

A LogisticRegression classifier (scikit-learn) trained on 768-dim SigLIP 2 embeddings from real scraped ads. Four-class output: gaming / ecommerce / finance / other. 5-fold stratified cross-validation accuracy: **79.7%**. The classifier runs at HF Spaces startup and is loaded once via a singleton pattern; if the `.pkl` is unavailable the `/score` endpoint degrades gracefully by returning `predicted_vertical: null`. The artifact is stored on HuggingFace Hub at `pcr12/creative-intelligence-scorer`.

---

## Model Results

| Model | Description | Spearman r | Notes |
|---|---|---|---|
| Random baseline | Uniform random predictions | 0.018 | Floor |
| Duration-only | Linear regression on runtime features | 0.062 | Barely above random |
| CLIP baseline | CLIP-ViT-B/32 + BCE + Weibull, mixed real/synthetic eval | 0.254 | First meaningful signal |
| Exp1 · drop sentinels | CLIP, sentinel rows removed | 0.203 | Worse — removing sentinels hurt generalization |
| Exp2 · balanced | CLIP, 60/40 real/synthetic, small test set | 0.594* | *Unreliable — test set too small to trust |
| **SigLIP 2 (ours)** | **SigLIP2-base + BCE + Weibull, real-only eval** | **0.645** | **Canonical result, epoch 18/30** |

Evaluation metric: Spearman rank correlation on a held-out real-ads-only test set. CTR AUC was removed as a metric — binarizing a continuous proxy label at the median is statistically meaningless and inflates apparent performance.

---

## Data Model

### Supabase Tables

| Table | Purpose |
|---|---|
| `cia_uploads` | Upload metadata: R2 key, vertical label, user session ID, created timestamp |
| `cia_scores` | Model outputs per upload: CTR score, halflife (days), confidence, heatmap regions (jsonb), scored timestamp |
| `creatives` (Storage) | Raw image files — Supabase Storage bucket, files never stored in DB rows |

### Training Data

| Source | Count | Notes |
|---|---|---|
| Apify (Meta Ad Library) | 2,542 real ads | Gaming, ecommerce, finance verticals |
| Synthetic (PIL-generated) | 18,746 images | Rule-based labels, face detection multiplier |
| **Total** | **22,248 rows** | `data/labels_clean.csv` (gitignored — use HF Hub) |

Schema: `ad_id, image_path, ctr_score, halflife_days, censored, vertical, source, discontinuation_type, scraped_at`

---

## Design Decisions

1. **Frozen backbone, always.** SigLIP 2 is never fine-tuned. Literature (Kitada 2022) confirms that for datasets below ~100K images, frozen encoders outperform fine-tuned ones due to overfitting risk. The 768-dim pooler embedding is treated as a fixed compressed representation of the image.

2. **Weibull survival loss, not just CTR regression.** CTR is a point prediction. Fatigue is a time-to-event distribution. Using Weibull NLL loss with right-censored labels correctly handles ads that were still running at scrape time — a regression loss would treat censored observations as ground truth zeros, which is wrong.

3. **Two discontinuation types.** Half-life < 7 days = "cut-out" (sudden performance drop, usually creative fatigue or budget exhaustion). Half-life ≥ 7 days = "wear-out" (gradual saturation). The distinction is shown in the UI and informs rotation recommendations.

4. **Real-only evaluation.** Synthetic data is used only for training augmentation. All held-out test evaluation is on real scraped ads only. Mixing synthetic into the test set inflates Spearman r by evaluating on images generated from the same label rules the model was trained on — not a valid measure of generalization.

5. **Agent, not classifier.** The explanation layer could have been a fixed template. Instead it is a ReAct agent with five grounded tools. This allows it to answer arbitrary follow-up questions, compare to benchmarks on demand, and project fatigue curves — none of which a template can do. The tradeoff is latency (~1–2s per turn).

6. **No markdown in agent responses.** The chat UI renders plain text. Markdown symbols passed through `dangerouslySetInnerHTML` with a minimal renderer. The system prompt explicitly prohibits `##`, `**`, `-` bullets, and numbered lists to prevent raw symbols appearing in the UI.

7. **Confidence calibration in three tiers.** Model confidence below 0.15 triggers an explicit low-confidence flag and withholds strong recommendations. 0.15–0.40 uses hedged language ("suggests", "likely"). Above 0.40 the agent speaks with authority. This prevents the agent from making definitive claims about creatives the model has never seen anything like.

8. **Frontend never calls HF Spaces directly.** All inference goes through the Railway/Render backend. This allows token injection, rate limiting, error handling, and Supabase writes to happen in one place, and keeps HF Spaces credentials off the client.

---

## Local Development

### Environment Variables

Create a `.env` file at the project root (see `.env.example`):

```
SUPABASE_URL=
SUPABASE_ANON_KEY=
R2_ACCOUNT_ID=
R2_ACCESS_KEY=
R2_SECRET_KEY=
R2_BUCKET=
HF_SPACES_URL=
HF_TOKEN=
ANTHROPIC_API_KEY=
API_TOKEN=
ALLOWED_ORIGINS=http://localhost:5173
```

### Backend (FastAPI)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

### Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

The frontend expects the backend at `http://localhost:8000`. Override with `VITE_API_URL` in `frontend/.env.local`.

### Model Training

```bash
pip install -r requirements-dev.txt

# Generate synthetic data
python scripts/synthetic_gen.py

# Download real ads (requires Apify API key)
python scripts/apify_downloader.py

# Build labels_clean.csv
python scripts/build_labels.py

# Train
python model/train.py

# Push to HF Hub
python model/push_to_hub.py
```

### HuggingFace Spaces (Inference)

The `spaces/` directory is a self-contained FastAPI app deployed to HF Spaces. It loads the model checkpoint and vertical classifier from the Hub at startup and exposes a `/score` endpoint. Deploy by pushing `spaces/` to a HF Space repo.

---

## Project Structure

```
adcreative-intel/
│
├── agent/                      # Agentic explanation layer
│   ├── agent.py                # ReAct loop, run_agent(), 5-step cap
│   ├── tools.py                # 5 tool implementations (score, heatmap, benchmark, suggestions, fatigue)
│   └── prompts.py              # System prompt with 11 grounding rules
│
├── api/                        # FastAPI backend (Render)
│   ├── main.py                 # App entrypoint, CORS, health check
│   ├── hf_client.py            # Persistent httpx client for HF Spaces inference
│   ├── db.py                   # Supabase singleton + R2 storage client
│   ├── requirements.txt        # Backend-only deps (deployed separately)
│   └── routes/
│       ├── upload.py           # POST /upload — R2 write, HF inference, Supabase write
│       └── chat.py             # POST /chat — run_agent() wrapper
│
├── model/                      # Vision model + training code
│   ├── clip_head.py            # Multi-task head (projection + CTR + Weibull fatigue branches)
│   ├── survival.py             # Weibull NLL loss with right-censored support
│   ├── dataset.py              # PyTorch Dataset for labels_clean.csv
│   ├── train.py                # HF Trainer training loop
│   ├── evaluate.py             # Spearman r + MAE evaluation
│   ├── gradcam.py              # GradCAM heatmap generation (layer 11)
│   ├── push_to_hub.py          # Upload checkpoint + classifier to HF Hub
│   ├── train_vertical_classifier.py  # LogisticRegression on SigLIP 2 embeddings
│   ├── alpha_search.py         # Loss weight (alpha) grid search
│   ├── finetune_real.py        # Real-data fine-tuning experiments
│   └── score_demos.py          # Score demo images for frontend
│
├── scripts/                    # Data pipeline
│   ├── apify_downloader.py     # Download real ads from Meta Ad Library via Apify
│   ├── synthetic_gen.py        # PIL-based synthetic ad generator with rule-based labels
│   ├── build_labels.py         # Merge real + synthetic into labels_clean.csv
│   ├── reclassify_verticals.py # Re-label verticals using SigLIP 2 embeddings
│   └── analyze_real_ads.py     # EDA on scraped ad metadata
│
├── spaces/                     # HuggingFace Spaces inference app
│   ├── app.py                  # FastAPI app with /score endpoint
│   ├── model_loader.py         # Load checkpoint + classifier from HF Hub at startup
│   ├── clip_head.py            # Model head (mirror of model/clip_head.py)
│   ├── gradcam.py              # GradCAM (mirror of model/gradcam.py)
│   ├── survival.py             # Weibull loss (mirror of model/survival.py)
│   ├── Dockerfile              # HF Spaces container definition
│   └── requirements.txt        # Inference-only deps
│
├── frontend/                   # React + TypeScript app (Vercel)
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.tsx     # Marketing landing page with architecture diagram
│   │   │   ├── Analyzer.tsx    # Main tool: upload, score cards, heatmap, chat
│   │   │   └── Benchmark.tsx   # Model results, dataset stats, vertical breakdown
│   │   ├── components/
│   │   │   ├── icons.tsx       # SVG icon set
│   │   │   └── HealthBanner.tsx
│   │   ├── lib/
│   │   │   ├── api.ts          # fetchWithRetry wrappers for all backend calls
│   │   │   ├── errors.ts       # ERROR_MESSAGES lookup (never String(e))
│   │   │   └── imageUtils.ts
│   │   ├── store/
│   │   │   ├── appStore.ts     # Zustand store: score, imageUrl, heatmapB64, uploadId
│   │   │   └── traceStore.ts   # Agent trace storage
│   │   ├── App.tsx             # Hash router, nav, page dispatch
│   │   ├── main.tsx            # Vite entrypoint
│   │   └── styles.css          # Design system (warm stone neutrals, violet accent)
│   ├── public/
│   │   ├── demo/               # Demo ad images loaded at startup
│   │   └── favicon.svg         # Aperture C mark
│   └── vercel.json             # Vercel SPA routing config
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   └── 02_training.ipynb       # Training experiments notebook
│
├── tests/
│   ├── test_suite.py           # End-to-end stress tests against live services
│   └── conftest.py             # Pytest fixtures
│
├── design/                     # Pre-build design prototypes (reference only)
│
├── .github/workflows/
│   └── keepalive.yml           # Cron ping to prevent Render cold starts
│
├── CLAUDE.md                   # AI session context (architecture constraints, schema)
├── requirements.txt            # Root Python deps (training + scripts)
├── requirements-dev.txt        # Dev deps (pytest, black, etc.)
├── Procfile                    # Render process definition
└── .env.example                # Environment variable template
```

---

## Built By

**Pranav CR**
AI Engineer · [prc4@illinois.edu](mailto:prc4@illinois.edu) · [GitHub](https://github.com/PranavCR01)
