# Slice 5A — AI Architecture Upgrades

**Status:** Planned  
**Portfolio framing:** Multi-Task Creative Lifespan Prediction  
**Goal:** Four targeted upgrades that make the model smarter, the agent more honest,
and the predictions more credible — without touching the SigLIP 2 backbone or
retraining the scoring head.

Implementation order matters: Feature 1 unlocks the embedding pipeline that
Feature 4 depends on. Features 2 and 3 are independent and can be done in parallel
after Feature 1.

---

## Feature 1 — Vertical Classifier

### Problem
Users must manually select gaming / ecommerce / finance on upload.
A wrong selection produces a meaningless benchmark comparison —
a gaming ad scored against finance norms fails in ways the user can't detect.

### Implementation

**Training (`model/train_vertical_classifier.py` — new file)**
- Load `data/labels_clean.csv`, filter `source=apify` (3,502 rows with ground-truth verticals)
- For each image, extract frozen SigLIP 2 `pooler_output` (768-dim) using the same
  `AutoProcessor` + `SiglipVisionModel` pipeline already used in training
- Fit a `sklearn.linear_model.LogisticRegression(max_iter=1000, C=1.0)` on those embeddings
- Evaluate with 5-fold cross-validation; log accuracy per vertical
- Save to `model/vertical_classifier.pkl` via `joblib.dump`

**Serving (`spaces/app.py`)**
- Load `vertical_classifier.pkl` at startup alongside the scoring model
- In `POST /score`: after extracting `pooler_output`, run `clf.predict` + `clf.predict_proba`
- Add to response: `predicted_vertical` (str) and `vertical_confidence` (float 0–1)
- The `vertical` form field becomes optional; if omitted, use `predicted_vertical` for
  benchmark lookup inside the Spaces endpoint

**API propagation (`api/routes/upload.py`)**
- Read `predicted_vertical` and `vertical_confidence` from HF Spaces `/score` response
- Return both fields in the `/upload` response JSON (no DB change needed)

**Frontend (`frontend/src/lib/api.ts`, `frontend/src/pages/Analyzer.tsx`)**
- On upload response, read `predicted_vertical`
- Pre-select the vertical dropdown to `predicted_vertical`
- Show a small badge: `"Auto-detected: Gaming (87%)"` next to the dropdown
- User can still override manually; override updates the displayed benchmark

### Acceptance Criteria
- [ ] `POST /score` returns `predicted_vertical` and `vertical_confidence` on every call
- [ ] `/upload` propagates both fields to the frontend
- [ ] Vertical dropdown is pre-selected on image upload with no manual action required
- [ ] Auto-detected badge shows vertical name and confidence percentage
- [ ] 5-fold CV accuracy ≥ 70 % across gaming / ecommerce / finance
  (logistic regression on frozen embeddings from a balanced 3.5K corpus)

### What NOT to Build
- Do not retrain the backbone or scoring head on vertical labels
- Do not build a multi-label classifier (one vertical per ad only)
- Do not surface `vertical_confidence` in the chat UI — it is display-only in the analyzer
- Do not fall back to user-selected vertical if the classifier is absent;
  default to `"other"` instead and surface a warning

### Dependencies
None — this is the foundation feature. Feature 4 reuses the embedding extraction
path introduced here.

---

## Feature 2 — Confidence-Aware Agent

### Problem
The agent speaks with identical authority whether `confidence = 0.06` or
`confidence = 0.82`. Low-confidence predictions arise when the creative falls
outside the training distribution; presenting them without caveat is misleading.

### Implementation

**Prompt update (`agent/prompts.py` — only file changed)**

Add a confidence-reasoning section to `SYSTEM_PROMPT`:

```
CONFIDENCE CALIBRATION RULES (apply whenever get_creative_score is called):

- confidence < 0.15  → LOW CONFIDENCE
  Explicitly flag: "Note: this prediction has low confidence (X%) —
  the creative may fall outside the training distribution. Treat these
  numbers as directional, not definitive."
  Do not make strong recommendations.

- confidence 0.15–0.40 → MODERATE CONFIDENCE
  Normal analytical tone. Mention confidence in passing if asked.
  Hedging is appropriate but not required.

- confidence > 0.40  → HIGH CONFIDENCE
  Speak with authority. Lead with the score and benchmark comparison.
  No confidence caveat needed unless the user asks.
```

The agent already receives `confidence` as part of the `get_creative_score()` tool
result — no schema changes, no new tools, no backend changes.

### Acceptance Criteria
- [ ] For a synthetic low-confidence creative (confidence < 0.15), the agent response
  contains explicit uncertainty language ("low confidence", "treat with caution",
  or equivalent)
- [ ] For a high-confidence creative (confidence > 0.40), no uncertainty caveat appears
- [ ] The word "confidence" or its synonym appears in the answer when the score is low
- [ ] Existing integration test (test 11 in `tests/test_suite.py`) still passes —
  the answer must still contain numerical claims

### What NOT to Build
- Do not add a new tool to fetch confidence separately — it is already in
  `get_creative_score()` output
- Do not surface confidence as a UI element (already shown in the score card)
- Do not change the confidence computation in `spaces/app.py` — the formula
  (`1 - 2 * |ctr - 0.5|`) stays as-is

### Dependencies
None — purely a prompt change, fully independent.

---

## Feature 3 — Fatigue Projection Tool

### Problem
The agent reports `halflife_days` as a single number.
"Your ad has a halflife of 5.2 days" is technically correct but operationally useless.
Users need to know: what will CTR look like at day 7? Day 14? Should I refresh now?

### Implementation

**New tool (`agent/tools.py`)**

```python
def get_fatigue_projection(halflife_days: float, trace: list[dict]) -> dict:
    """
    Computes Weibull retention curve at fixed checkpoints.
    Formula: retention(t) = exp(-(t / halflife_days) ** 1.5)
    shape=1.5 matches the Weibull head trained in clip_head.py.
    """
    def retention(t):
        return round(math.exp(-((t / halflife_days) ** 1.5)), 3)

    d7  = retention(7)
    d14 = retention(14)
    d21 = retention(21)

    if d7 < 0.5:
        recommendation = "rotate creative by day 5"
    elif d14 < 0.5:
        recommendation = "plan creative refresh for week 2"
    else:
        recommendation = "strong longevity, can run 3+ weeks"

    return {
        "day_7":  d7,
        "day_14": d14,
        "day_21": d21,
        "recommendation": recommendation,
    }
```

**Wire into agent (`agent/agent.py`)**

Add to `TOOL_MAP`:
```python
"get_fatigue_projection": get_fatigue_projection,
```

Add to `TOOL_SCHEMAS`:
```python
{
    "name": "get_fatigue_projection",
    "description": (
        "Computes ad fatigue retention at day 7, 14, and 21 from a halflife value. "
        "Call after get_creative_score when the user asks about longevity, "
        "refresh cadence, or how long to run the ad."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "halflife_days": {"type": "number", "description": "Predicted fatigue halflife in days"}
        },
        "required": ["halflife_days"],
    },
}
```

Note: this tool takes `halflife_days` (a number), not `image_id`. The agent extracts
`halflife_days` from the `get_creative_score()` result and passes it directly.

### Acceptance Criteria
- [ ] `get_fatigue_projection` appears in `TOOL_SCHEMAS` and `TOOL_MAP`
- [ ] Tool returns `day_7`, `day_14`, `day_21` as floats between 0 and 1
- [ ] `recommendation` string matches one of the three defined branches
- [ ] For `halflife_days=5.0`: `day_7 < 0.5` → recommendation is "rotate creative by day 5"
- [ ] For `halflife_days=15.0`: `day_7 > 0.5`, `day_14 ≈ 0.5` → recommendation is
  "plan creative refresh for week 2"
- [ ] For `halflife_days=25.0`: `day_21 > 0.5` → recommendation is "strong longevity"
- [ ] Agent calls this tool when asked "how long should I run this ad?" or similar

### What NOT to Build
- Do not build a UI chart of the retention curve — text output from the agent only
- Do not expose this as a standalone backend endpoint — agent tool only
- Do not vary the Weibull shape parameter (shape=1.5 is fixed, matches training)
- Do not add `image_id` to the tool input — the tool is pure math, no DB reads

### Dependencies
None — pure computation, fully independent. Can be implemented in parallel with
Feature 2 after Feature 1 is merged.

---

## Feature 4 — Cross-Creative Similarity Search (pgvector)

### Problem
"Your CTR is 0.08" is an assertion. "Ads visually similar to yours had a median CTR
of 0.09 in our 3,502-ad corpus" is evidence. Similarity search makes predictions
auditable and grounds the agent's claims in real data.

### Implementation

#### 4A — Supabase schema migration (one-time, manual)
```sql
-- Run in Supabase SQL editor
CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE cia_scores ADD COLUMN IF NOT EXISTS embedding vector(768);
CREATE INDEX ON cia_scores USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

#### 4B — Embedding extraction in HF Spaces (`spaces/app.py`)
- In `POST /score`, after computing `pooler_output`, include the raw 768-dim tensor
  in the response as a Python list: `"embedding": pooler_output.squeeze().tolist()`
- This is the only place the backbone runs, so extraction is zero extra cost

#### 4C — Save embedding on upload (`api/routes/upload.py`, `api/db.py`)
- `upload.py`: read `embedding` list from HF Spaces `/score` response;
  pass to `upsert_score()`
- `db.py` — `upsert_score()`: add `embedding: list[float] | None = None` param;
  include in upsert dict when present (Supabase handles `vector` type via JSON array)

#### 4D — Corpus population script (`model/embed_corpus.py` — new file)
- Iterates all 3,502 apify images in `data/raw/`
- For each image: runs `SiglipVisionModel` forward pass, extracts `pooler_output`
- Upserts `(upload_id, embedding)` into `cia_scores` in batches of 64
- Estimated runtime: ~2 hrs on CPU (GTX 1650 Ti would be ~15 min)
- Run once after schema migration; re-run only if backbone changes

#### 4E — Similarity tool (`agent/tools.py`)

```python
def get_similar_creatives(image_id: str, trace: list[dict]) -> dict:
    """
    Finds the 5 most visually similar ads in the corpus via pgvector cosine search.
    Returns their upload_ids, ctr_scores, verticals, and similarity scores.
    """
    db = get_db()
    # Fetch this creative's embedding
    row = (
        db.table("cia_scores")
        .select("embedding")
        .eq("upload_id", image_id)
        .single()
        .execute()
    )
    if not row.data or not row.data.get("embedding"):
        return {"error": "No embedding found for this creative. Upload may predate corpus indexing."}

    embedding = row.data["embedding"]  # list[float] len=768

    # pgvector cosine similarity via Supabase RPC
    results = db.rpc(
        "match_creatives",
        {"query_embedding": embedding, "match_count": 5, "exclude_id": image_id},
    ).execute()

    similar = results.data or []
    if not similar:
        return {"similar_creatives": [], "note": "No similar ads found in corpus."}

    ctr_values = [r["ctr_score"] for r in similar if r.get("ctr_score") is not None]
    median_ctr = sorted(ctr_values)[len(ctr_values) // 2] if ctr_values else None

    return {
        "similar_creatives": similar,   # [{upload_id, ctr_score, vertical, similarity}]
        "corpus_median_ctr": median_ctr,
        "sample_size": len(similar),
    }
```

**Supabase RPC function (run in SQL editor once):**
```sql
CREATE OR REPLACE FUNCTION match_creatives(
    query_embedding vector(768),
    match_count     int,
    exclude_id      uuid
)
RETURNS TABLE (
    upload_id  uuid,
    ctr_score  float,
    vertical   text,
    similarity float
)
LANGUAGE sql STABLE AS $$
    SELECT
        s.upload_id,
        s.ctr_score,
        u.vertical,
        1 - (s.embedding <=> query_embedding) AS similarity
    FROM cia_scores s
    JOIN cia_uploads u ON u.id = s.upload_id
    WHERE s.embedding IS NOT NULL
      AND s.upload_id != exclude_id
    ORDER BY s.embedding <=> query_embedding
    LIMIT match_count;
$$;
```

**Wire into agent (`agent/agent.py`)**
- Add `get_similar_creatives` to `TOOL_MAP` and `TOOL_SCHEMAS`
- Input schema: `{image_id: string}` — same pattern as existing tools
- Description: "Returns the 5 most visually similar ads in the corpus with their
  actual CTR values. Call this to ground predictions in real comparable data."

### Acceptance Criteria
- [ ] `pgvector` extension enabled; `cia_scores.embedding` column exists (`vector(768)`)
- [ ] `match_creatives` RPC function deployed in Supabase
- [ ] `POST /score` on HF Spaces returns an `embedding` field (list of 768 floats)
- [ ] `POST /upload` on backend saves embedding to `cia_scores.embedding`
- [ ] `model/embed_corpus.py` runs to completion on all 3,502 apify images
  with no silent failures (logs skipped images to stderr)
- [ ] `get_similar_creatives` tool returns ≥ 1 result for any uploaded creative
  once corpus is populated
- [ ] Agent calls this tool when asked "what ads are similar to mine?" or
  "how does this compare to real data?"
- [ ] `corpus_median_ctr` in the tool result is a float, not null, for populated corpus

### What NOT to Build
- Do not build a frontend UI for similarity results — agent text only in this slice
- Do not store embeddings in R2 or any external store — Supabase `cia_scores` only
- Do not expose a raw `/similar` HTTP endpoint — agent tool only
- Do not embed synthetic ads in the corpus — apify real ads only (3,502 rows)
- Do not run the corpus embedding script as part of the upload flow — offline only

### Dependencies
- **Requires Feature 1**: Feature 1 adds the embedding extraction step in `spaces/app.py`
  (`pooler_output` is already computed for the vertical classifier).
  Feature 4B reuses this — do not duplicate the forward pass.
- **Requires Supabase migration** (4A) before any code that writes embeddings will work.
- Features 2 and 3 are independent of Feature 4.

---

## Dependency Graph

```
Feature 1 (Vertical Classifier)
    │
    └──► Feature 4 (Similarity Search)   [shares embedding extraction from /score]

Feature 2 (Confidence-Aware Agent)       [independent — prompt only]

Feature 3 (Fatigue Projection Tool)      [independent — pure math tool]
```

Recommended implementation order:
1. Feature 1 — establishes embedding pipeline used by Feature 4
2. Features 2 + 3 — can be parallelised, no blocking dependency
3. Feature 4 — after Feature 1 is merged and Supabase migration is applied

---

## Risk Register

| Feature | Risk | Likelihood | Mitigation |
|---|---|---|---|
| 1 | Classifier accuracy < 70% on 3-class problem | Low | Logistic regression on SigLIP 2 embeddings is well above chance; worst case fall back to user selection |
| 2 | Prompt change degrades general answer quality | Low | Integration test (test 11) catches regressions; revert is a one-line change |
| 3 | Agent passes wrong `halflife_days` value to tool | Low | Tool is pure math — wrong input → wrong output, but no crash; add unit test |
| 4 | pgvector query timeout on large corpus | Medium | IVFFlat index with lists=100 keeps query < 100ms at 3.5K rows; re-tune if corpus grows past 50K |
| 4 | Corpus embedding run fails midway | Medium | Script uses upsert (idempotent) — safe to re-run from any point |
| 4 | Supabase free tier storage limit | Medium | 768 floats × 4 bytes × 3,502 rows ≈ 10 MB — well within free tier limits |

---

## Files Changed Summary

| File | Feature | Change type |
|---|---|---|
| `model/train_vertical_classifier.py` | 1 | New |
| `model/vertical_classifier.pkl` | 1 | New (generated artifact) |
| `model/embed_corpus.py` | 4 | New |
| `spaces/app.py` | 1, 4 | Modified |
| `agent/prompts.py` | 2 | Modified |
| `agent/tools.py` | 3, 4 | Modified |
| `agent/agent.py` | 3, 4 | Modified |
| `api/routes/upload.py` | 4 | Modified |
| `api/db.py` | 4 | Modified |
| `frontend/src/lib/api.ts` | 1 | Modified |
| `frontend/src/pages/Analyzer.tsx` | 1 | Modified |
| Supabase SQL editor | 4 | Migration (manual, one-time) |
