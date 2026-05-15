# ML Intern Prompts — Creative Intelligence Agent

## How to use
1. Install: `git clone https://github.com/huggingface/ml-intern.git`
   `cd ml-intern && uv sync && uv tool install -e .`
2. Set env vars:
   `export HF_TOKEN=your_hf_token`
   `export ANTHROPIC_API_KEY=your_anthropic_key`
3. Start: `ml-intern --model anthropic/claude-sonnet-4-5`
4. Paste the prompt. ML Intern plans, writes code, runs it on your local GPU, iterates.
5. You review outputs, then continue with Claude Code for the rest of the slice.

## Rule: never let ML Intern write clip_head.py or survival.py
Those two files are yours. They are your ML depth signal in interviews.
Everything else ML Intern can touch.

---

## SLICE 1 — Data Pipeline

### Prompt 1A: Dataset research (run immediately — no data needed yet)
```
I'm building a multi-task vision model to predict ad creative CTR score
and fatigue half-life from raw images for a portfolio project targeting
Moloco, AppLovin, and Liftoff (mobile advertising companies).

Important context on my data strategy:
- The Meta Ad Library official API only provides political/social issue ads in EU.
  I cannot use it for commercial gaming/ecommerce/finance ads.
- I am using Apify's Facebook Ad Library web scraper for real ads (~10-15K)
- I am using PIL-based synthetic image generation for the rest (~35-40K)
- Total target: ~50K images with CTR proxy labels and fatigue halflife labels
- CTR proxy = normalized run duration (days run / 90, capped at 1.0)
- Fatigue halflife = days until ad was discontinued (survival label)
- Verticals: gaming, ecommerce, finance

Please do the following:
1. Search HF Hub for any existing ad creative datasets, CTR prediction
   datasets, or image-performance datasets I could supplement with.
   Specifically look for datasets that include actual image files
   (not just feature vectors). List with size, license, HF dataset ID.

2. Search arxiv for papers on ad creative performance prediction from
   raw images, 2022-2026. Read the top 3 most relevant and summarize:
   - What datasets did they use?
   - Did they use frozen or fine-tuned vision encoders?
   - What label proxies did they use when real CTR wasn't available?
   - Did any of them use survival/time-to-event models for fatigue?

3. Validate my CTR proxy approach: is "run duration / 90 days" a
   reasonable proxy for CTR? What are the known biases? What do
   papers suggest as better alternatives given only public data?

4. Given my synthetic label rules (text density → faster decay,
   central CTA → better performance), are these directionally
   consistent with what the literature says about creative elements?

Output a 1-page research summary I can paste into my README dataset section.
Do NOT write any code yet.
```

### Prompt 1B: Dataset audit (run AFTER apify_downloader.py has collected 5K+ rows)
```
I have a dataset at data/labels.csv from two sources:
- Apify-scraped real Meta Ad Library ads (gaming, ecommerce, finance)
- PIL-generated synthetic ads on Open Images V7

Schema: ad_id, image_path, ctr_score, halflife_days, censored, vertical, source, scraped_at

Please do the following in adcreative-intel/ working directory:
1. Load the CSV and run a full data quality audit:
   - Missing values per column
   - ctr_score distribution (histogram)
   - halflife_days distribution by source (apify vs synthetic)
   - Censoring rate (% where censored=True) — expected higher for apify
   - Count per vertical and source
   - Image existence check: % of image_path values that point to real files

2. Flag problem rows:
   - ctr_score outside [0, 1]
   - halflife_days < 0
   - censored=False but halflife_days is null (contradiction)
   - image_path pointing to missing file

3. Write cleaned data to data/labels_clean.csv
   Remove all flagged rows. Print: how many removed and why.

4. Check for distribution mismatch between apify and synthetic:
   - Do they have similar ctr_score distributions?
   - If very different, flag it — this could hurt training.

Run locally using Python. No GPU needed.
```

### Prompt 1C: Synthetic quality check (run AFTER 5K synthetic images generated)
```
I've generated synthetic ad images using PIL in adcreative-intel/data/synthetic/

Please do the following:
1. Sample 20 random synthetic images (across all 3 verticals)
   and describe what you see — do they look like real ads?
   Are the text overlays readable? CTAs visible?

2. Compute image statistics on the synthetic set:
   - Mean brightness per vertical
   - Mean edge density (proxy for visual complexity)
   - Mean text area percentage (rough estimate)
   Compare these to the same stats on the Apify real ads in data/raw/

3. If the synthetic images look clearly artificial or the statistics
   are very different from real ads, suggest ONE specific PIL change
   to make them more realistic. Implement it and regenerate 100 samples.
   Show before/after statistics.

4. Confirm the label distribution makes sense:
   - synthetic ads should have halflife_days between 1-60
   - gaming should have shorter halflife than finance on average
   - no censored synthetic rows (source="synthetic")

Working directory: adcreative-intel/
This is CPU work — no GPU needed.
```

---

## SLICE 2 — Vision Model Training

### Prompt 2A: Architecture validation (run BEFORE writing training code)
```
I'm building a multi-task vision model for ad creative scoring.
Please validate my architecture and suggest any improvements.

My architecture (DO NOT change these core design decisions —
they are intentional for the portfolio project):
- Frozen CLIP-ViT-B/32 backbone (openai/clip-vit-base-patch32)
- Projection: Linear(512, 256) → ReLU → Dropout(0.2)
- CTR head: Linear(256, 1) → Sigmoid, BCELoss
- Fatigue head: Linear(256, 2) → Weibull log_scale + log_shape
- Fatigue loss: WeibullNLLLoss with right-censoring support
- Multi-task loss: 0.5 * BCELoss + 0.5 * WeibullNLLLoss
- HF Trainer, fp16, batch 64, lr 1e-3, 15 epochs
- GPU: NVIDIA GTX 1650 Ti, 4GB VRAM

My dataset: ~50K images, ~30% censored.
Labels are proxy (run duration), not real CTR.

Please:
1. Search arxiv — do papers freeze CLIP for ad/image tasks or fine-tune?
   What's the consensus for small datasets (<100K)?

2. Estimate VRAM usage: frozen CLIP-ViT-B/32 + projection + 2 heads
   + batch 64 + fp16. Safe for 4GB?

3. Is WeibullNLLLoss the right choice for right-censored survival data
   or should I consider Cox proportional hazards instead?
   What are the tradeoffs at this dataset size?

4. My labels are noisy (proxy not real CTR). What regularization
   strategies help most with noisy labels in multi-task settings?
   Can I add anything to the current architecture without changing the core?

Output: validation summary + max 2 concrete suggestions I can add
WITHOUT changing clip_head.py or survival.py architecture.
Do NOT write any code yet.
```

### Prompt 2B: Baseline experiments (run WHILE main model trains on GPU)
```
I need 3 baseline models for my README results table.
My main model is training on the GPU — run these baselines in parallel.

Dataset: adcreative-intel/data/labels_clean.csv
Split rule: 80/10/10 by index (rows 0-80%, 80-90%, 90-100%)
This must match the main model's split exactly.

Baseline 1 — Random:
- Predict ctr_score = random.uniform(0, 1) for every test sample
- Compute CTR AUC (sklearn.metrics.roc_auc_score)

Baseline 2 — Duration-only (no image):
- Features: vertical (one-hot, 4 classes), censored (bool as int)
- Target: ctr_score
- Model: sklearn LogisticRegression
- Report CTR AUC on test set

Baseline 3 — CTR-only (single task, no fatigue head):
- Load frozen CLIP-ViT-B/32
- Same projection layer as main model
- Only CTR head (no fatigue head, no Weibull loss)
- Train with same HF Trainer args as main model
- Report CTR AUC on test set

Save all results to:
adcreative-intel/notebooks/baselines_results.json
Format: {
  "random": {"ctr_auc": X.XX},
  "duration_only": {"ctr_auc": X.XX},
  "ctr_only": {"ctr_auc": X.XX}
}

Working directory: adcreative-intel/
Baseline 3 uses GPU. Baselines 1 and 2 are CPU only.
```

### Prompt 2C: Hyperparameter iteration (run AFTER first training run completes)
```
My main model training just finished. Results:

[PASTE YOUR ACTUAL METRICS HERE]
CTR AUC on test set: X.XX
D-calibration bins within expected: X/10
Loss curve: [describe — converged / still dropping / unstable]

Targets: CTR AUC > 0.65, D-calibration > 7/10 bins

If targets not met:
1. Diagnose the most likely cause from these metrics
2. Suggest ONE change (not multiple)
3. Implement it in model/train.py ONLY
   Do NOT touch model/clip_head.py or model/survival.py
4. Re-run training
5. Report new metrics vs previous run

Changes allowed (pick the most likely fix):
- Alpha weight: currently 0.5, try 0.3 or 0.7
- pos_weight in BCELoss: try 2.0 if ctr_score is imbalanced toward low values
- Learning rate: currently 1e-3, try 5e-4
- Dropout: currently 0.2, try 0.3
- Epochs: add 5 more if loss was still dropping

Hard limit: max 3 iterations. If CTR AUC < 0.65 after 3 tries,
stop and report — I'll review data quality instead of continuing to tune.

Working directory: adcreative-intel/
Run on local GPU.
```

### Prompt 2D: GradCAM validation (run AFTER training meets targets)
```
My model training is complete and meets the eval targets.
Model checkpoint: adcreative-intel/model/checkpoints/best_model/

Please:
1. Load the trained model (CPU is fine for this)

2. Run GradCAM on 10 test set images:
   - 2-3 images with ctr_score > 0.6 (high performers)
   - 2-3 images with ctr_score < 0.3 (low performers)
   - 2-3 images with ctr_score 0.3-0.6 (average)
   Use the GradCAM implementation in model/gradcam.py

3. Save heatmap overlays to adcreative-intel/notebooks/heatmaps/
   Filename format: {ad_id}_{ctr_score:.2f}_heatmap.png

4. For each image, print:
   - Predicted ctr_score and halflife_days
   - Top 3 high-attention grid regions (from 3x3 grid)
   - Top 3 low-attention grid regions

5. Assess: do heatmaps correlate with visual features?
   - High scorers: do faces, CTAs, bright elements get attention?
   - Low scorers: do cluttered/text-heavy regions get flagged?
   Be honest. If they look random, say so — we document the limitation.

IMPORTANT: If heatmaps look random, do NOT try to fix the architecture.
Our GradCAM on the projection layer is an approximation (not exact spatial
GradCAM). Document this limitation honestly. We note it in the README.

Working directory: adcreative-intel/
```

---

## Parallel workflow across all slices

```
DAY 1 — Setup
  Browser:     apify.com → run Facebook Ad Library Scraper
  ML Intern:   Prompt 1A (dataset research) — runs immediately
  You:         Set up repo, read Weibull survival analysis docs

DAY 2 — Data collection
  Terminal 1:  python scripts/apify_downloader.py (once JSON downloaded)
  Terminal 2:  python scripts/synthetic_gen.py (~3-4 hrs)
  You:         Write model/clip_head.py (YOUR work, not ML Intern)
               Write model/survival.py (YOUR work, not ML Intern)

DAY 3 — Data ready
  ML Intern:   Prompt 1B (dataset audit) + Prompt 1C (quality check)
  You:         Write model/train.py, model/dataset.py

DAY 4 — Training starts
  Terminal 1:  python model/train.py (GPU, ~4-6 hrs)
  ML Intern:   Prompt 2A (architecture validation) — runs simultaneously
  You:         Write spaces/app.py (Slice 3 work — no GPU needed)

DAY 5 — Training running
  ML Intern:   Prompt 2B (CPU baselines while GPU trains main model)
  You:         Write api/hf_client.py, api/db.py, GitHub Actions keepalive

DAY 6 — Training done
  ML Intern:   Prompt 2C (iterate if targets missed)
  You:         Deploy HF Spaces (Slice 3)

DAY 7
  ML Intern:   Prompt 2D (GradCAM validation)
  You:         Start Slice 4 agent setup
```
