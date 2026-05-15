# Slice 6 — Polish and Pitch

## Goal
Portfolio-ready artifact that gets you interviews.
No Claude Code needed this slice — this is writing, recording, and sending.

## README.md — written like an AdKDD applied paper

### Structure (follow exactly)
```
# Creative Intelligence Agent

## Abstract
One paragraph, 3 sentences:
1. What problem you're solving and why it matters
2. What you built (model + agent)
3. Key result (CTR AUC, demo link)

## Problem
- Ad creative fatigue costs UA teams X% of budget (cite Moloco casino study)
- Current solutions: manual tagging (slow), rule-based (no signal), black-box DSP (no explanation)
- Gap: no open tool that predicts fatigue from raw creatives AND explains why

## Dataset
Table:
| Source | Count | Verticals | Label type |
|---|---|---|---|
| Meta Ad Library | ~40K | gaming, ecommerce, finance | Run duration proxy |
| Synthetic (PIL) | ~15K | all | Rule-based simulator |
| Total | ~55K | 3 verticals | CTR score + halflife |

Stats: median halflife by vertical, censoring rate, ctr_score distribution

## Model Architecture
Include the architecture diagram (ASCII or PNG):
Input → CLIP-ViT-B/32 (frozen) → Projection → CTR head + Fatigue head

Table of parameters:
| Component | Trainable params | Frozen params |
|---|---|---|
| CLIP backbone | 0 | 86M |
| Projection + heads | ~132K | 0 |

## Results
Table — your model vs baselines:
| Model | CTR AUC | Fatigue D-cal | Notes |
|---|---|---|---|
| Random baseline | 0.500 | — | Coin flip |
| Duration-only | 0.61 | — | No image signal |
| CTR head only | 0.XX | — | Single task |
| Full model (ours) | 0.XX | X/10 bins | Multi-task Weibull |

Include: training curves (loss over epochs), D-calibration plot as image

## Agentic Layer
Explain: 4 tools, grounded reasoning, anti-hallucination architecture
Show one example conversation with trace (screenshot or code block)

## Limitations
- CTR proxy (run duration) ≠ real CTR — we approximate, not measure
- GradCAM on projection layer is a spatial approximation, not exact
- Meta Ad Library scrape may over-represent certain verticals
- Agent response time ~20s — not suitable for real-time use cases

## Future Work
- Real CTR labels via A/B test integration
- Video frame analysis (extend to MP4)
- Fine-tuned agent on ad-tech conversations
- SKAN postback integration for mobile-specific fatigue signals

## Live Demo
[URL] — try with the preloaded gaming/ecommerce/finance creatives

## Model on HF Hub
[URL] — public weights, model card, usage example

## Tech Stack
List: CLIP, PyTorch, Weibull survival loss, smolagents, Qwen2.5-1.5B,
FastAPI, React, Tailwind, Supabase, Cloudflare R2, HF Spaces, Railway, Vercel
```

## Results table — fill in your actual numbers
Run evaluate.py on the test set, record:
- CTR AUC (roc_auc_score)
- D-calibration bins within expected range (count out of 10)
- Baseline comparisons (train single-task CTR model, duration-only model)
These go in the README results table. Do not fabricate numbers.

## Demo video — 2 minutes max, no commentary needed
Record with Loom (loom.com, free):
```
00:00 — Open the app URL
00:05 — Click "Gaming" demo creative, scores appear instantly
00:25 — Toggle to heatmap overlay, point at high-attention regions
00:40 — Type in chat: "Why is this creative underperforming?"
01:10 — Agent answer appears, open reasoning trace accordion
01:30 — Click "Ecommerce" demo creative, repeat quickly
01:50 — Type: "What should I change first?"
02:00 — End
```
No voiceover needed — the UI speaks for itself.
Add the Loom URL to the README.

## Portfolio site update — PranavCR01.github.io
Add project card with:
- Title: "Creative Intelligence Agent"
- Tags: RAG & Retrieval, Agents, Computer Vision, Ad-Tech
- One-line description: "Vision model + agentic AI for ad creative fatigue prediction"
- Live demo link + GitHub link
- Tech icons: Python, PyTorch, React, Hugging Face

## The pitch email — for your Moloco contact
```
Subject: Built the creative scoring layer Moloco Studios doesn't have yet

Hi [Name],

I built a project I think your team would find interesting — a multi-task
vision model that predicts ad creative CTR and fatigue half-life from raw
images, trained on Meta Ad Library data using a Weibull survival loss
(the same time-to-event modeling approach used in production DSPs).

On top of the model I built an agentic explanation layer that tells
advertisers exactly which visual elements are hurting performance and
proposes concrete variants — every claim sourced from the model output,
never hallucinated.

It's essentially the ML upgrade to the manual creative tagging approach
from Moloco's casino creatives study — a working demo of what a creative
intelligence pipeline could look like.

Live demo: [URL]
Model on HF Hub: [URL]
Code + write-up: [GitHub URL]

Happy to walk through the architecture or the training approach if useful.

[Your name]
```

## LinkedIn post (optional but high leverage)
```
Built a project targeting a real gap in the mobile ad-tech stack.

Ad creatives fail silently — UA managers discover fatigue 7-14 days
after ROAS already dropped, and nobody can tell them which visual element
caused it.

I built a vision model (CLIP + Weibull survival loss) that predicts
creative fatigue BEFORE it happens, plus an agentic explanation layer
that grounds every claim in model output — no hallucination.

Trained on 55K Meta Ad Library ads across gaming, ecommerce, and finance.
CTR AUC: [X] on held-out test set.

Live demo → [URL]
Model on HF Hub → [URL]

Built with: PyTorch, smolagents, FastAPI, React, HF Spaces, Supabase.

#adtech #machinelearning #AIengineer #programmatic #portfolioproject
```

## Resume bullet (final version)
```
Built Creative Intelligence Agent: fine-tuned CLIP-ViT-B/32 with a multi-task
Weibull survival head on 55K Meta Ad Library creatives (CTR AUC: X.XX);
added a smolagents agentic layer with grounded tool-call reasoning for
creative diagnosis; deployed as a full-stack app (React + FastAPI + HF Spaces).
Public model on Hugging Face Hub.
```
Fill in your actual CTR AUC number before pasting onto resume.

## Done when
- [ ] README passes the "AdKDD paper" test — abstract, dataset table, architecture, results table, limitations, future work
- [ ] Results table has real numbers from evaluate.py — not placeholders
- [ ] Demo video is 2 minutes or less, Loom URL in README
- [ ] Portfolio site updated with project card and live link
- [ ] Pitch email drafted, reviewed by your Moloco contact before sending
- [ ] LinkedIn post drafted (send after first interview request, not before)
- [ ] Resume bullet updated with real CTR AUC number

## Final checklist before pitching
- [ ] Live demo URL works on mobile (test on your phone)
- [ ] HF Hub model page has a proper model card (not empty)
- [ ] GitHub repo is public
- [ ] No TODO comments or debug print statements visible in code
- [ ] .env file is in .gitignore — API tokens not committed
- [ ] All 3 demo creatives load without any API call (cold-start proof)

## Project complete.
Archive CLAUDE.md: update STATUS to COMPLETE, add lessons learned section.
Log final session in Notion Dev Journal with the six-section template.
