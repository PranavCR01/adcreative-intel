/* Landing page (route: #/) */
const { useState, useEffect } = React;

function Landing({ navigate }) {
  return (
    <>
      <Hero navigate={navigate} />
      <ProblemSection />
      <HowItWorks />
      <Architecture />
      <Stats />
      <Footer />
    </>
  );
}

function Hero({ navigate }) {
  return (
    <section className="hero">
      <div className="hero-inner">
        <div className="hero-copy">
          <div className="eyebrow" style={{ marginBottom: 22 }}>
            <span className="bullet"></span>
            <span>Multi-Task Creative Lifespan Prediction</span>
          </div>
          <h1 className="h1">
            Know why your ad is failing.
            <br />
            <span className="stress">Before you waste budget.</span>
          </h1>
          <p className="hero-sub">
            Vision model plus agentic AI for ad creative fatigue prediction. Drop in a
            creative, get a CTR estimate, a half-life in days, and an explanation grounded
            in the regions of the image the model actually looked at.
          </p>
          <div className="hero-ctas">
            <button className="btn btn-primary" onClick={() => navigate("/app")}>
              Analyze a Creative <IconArrow size={14} />
            </button>
            <button className="btn" onClick={() => navigate("/benchmark")}>
              See model results
            </button>
          </div>
          <div className="hero-meta">
            <span><span className="dot"></span>CLIP-ViT-B/32 backbone</span>
            <span><span className="dot"></span>Weibull survival head</span>
            <span><span className="dot"></span>Trained on 22,248 ads</span>
          </div>
        </div>
        <HeroViz />
      </div>
    </section>
  );
}

function HeroViz() {
  const [heat, setHeat] = useState(true);
  return (
    <div className="hero-viz">
      <div className="hero-viz-head">
        <span>creative_score_v1.ckpt</span>
        <span style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span className="status-dot"></span> live
        </span>
      </div>
      <div className="hero-viz-row">
        <div className="viz-ad">
          {heat && <div className="heat"></div>}
          drop creative here
        </div>
        <div className="viz-scores">
          <div className="viz-score-box">
            <div className="viz-score-label">CTR · gaming</div>
            <div className="viz-score-val">2.84<span className="unit">%</span></div>
            <div className="viz-bar">
              <i style={{ width: "62%" }}></i>
              <span className="mark" style={{ left: "44%" }}></span>
            </div>
            <div className="viz-score-sub">+0.91 vs vertical median</div>
          </div>
          <div className="viz-score-box">
            <div className="viz-score-label">Half-life</div>
            <div className="viz-score-val">7.2<span className="unit">d</span></div>
            <div className="viz-bar">
              <i style={{ width: "48%", background: "var(--warn)" }}></i>
              <span className="mark" style={{ left: "62%" }}></span>
            </div>
            <div className="viz-score-sub">Type: <span style={{ color: "var(--warn)" }}>wear-out</span></div>
          </div>
        </div>
      </div>
      <div className="viz-trace">
        <div className="ln"><span className="gutter">01</span><span><span className="kw">tool</span>: get_creative_score(<span className="str">"id_94c2"</span>) <span style={{ color: "var(--text-4)"}}>// 142ms</span></span></div>
        <div className="ln"><span className="gutter">02</span><span><span className="kw">tool</span>: get_heatmap_regions() <span style={{ color: "var(--text-4)"}}>// 208ms</span></span></div>
        <div className="ln"><span className="gutter">03</span><span><span className="kw">tool</span>: get_benchmark(<span className="str">"gaming"</span>) <span style={{ color: "var(--text-4)"}}>// 91ms</span></span></div>
        <div className="ln"><span className="gutter">04</span><span><span className="kw">agent</span>: high attention on <span className="str">"face + CTA"</span></span></div>
        <div className="ln"><span className="gutter">05</span><span><span className="kw">agent</span>: predicts <span className="num-tok">7.2d</span> halflife → <span className="str">wear-out</span></span></div>
        <div className="ln"><span className="gutter">06</span><span>→ done <span style={{ color: "var(--text-4)"}}>(441ms total)</span></span></div>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-3)"}}>
        <button className="toggle on" onClick={() => setHeat(!heat)} style={{ fontSize: 11 }}>
          <span className="sw"></span> attention overlay
        </button>
        <span>confidence 0.71</span>
      </div>
    </div>
  );
}

function ProblemSection() {
  const cards = [
    {
      idx: "01",
      title: "Manual creative tagging doesn't scale",
      body: "Performance teams hand-label every new asset by hue, theme, character — then re-label it next sprint. The taxonomy is stale before the campaign launches.",
      chips: [{ t: "today", cls: "alert" }, { t: "human-in-the-loop", cls: "" }]
    },
    {
      idx: "02",
      title: "DSPs grade your creative as a black box",
      body: "Bidders give you a creative score with no provenance. You learn it's bad after the budget is spent. There's no rubric to iterate against.",
      chips: [{ t: "opaque", cls: "alert" }, { t: "no rubric", cls: "" }]
    },
    {
      idx: "03",
      title: "Nobody can tell you why",
      body: "Even when fatigue is detected, the signal is a number. Was it the headline? The face? The lack of motion? Without explanation, the next creative is a guess.",
      chips: [{ t: "no explanation", cls: "alert" }, { t: "no learning loop", cls: "" }]
    }
  ];
  return (
    <section className="section">
      <div className="section-inner">
        <div className="section-head">
          <span className="eyebrow"><span className="bullet"></span>The gap</span>
          <h2 className="h2">Three things the current creative stack can't do.</h2>
          <p style={{ color: "var(--text-3)", fontSize: 16, margin: 0, lineHeight: 1.6 }}>
            Predict fatigue before launch. Explain a score in plain language. Tell you which pixel
            of the creative is doing the work.
          </p>
        </div>
        <div className="three-col">
          {cards.map((c) => (
            <div className="problem-card" key={c.idx}>
              <span className="num-idx">{c.idx}</span>
              <h4>{c.title}</h4>
              <p>{c.body}</p>
              <div className="tag-row">
                {c.chips.map((ch, i) => (
                  <span key={i} className={"chip " + ch.cls}>{ch.t}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function HowItWorks() {
  const steps = [
    {
      n: "Step 01",
      h: "Upload",
      p: "Drop a static image creative — gaming, ecommerce, or finance vertical. PNG or JPEG, up to 8MB.",
      v: "→ POST /upload · ~120ms"
    },
    {
      n: "Step 02",
      h: "Score",
      p: "CLIP-ViT extracts a 768-dim embedding. A multi-task head returns a CTR estimate and a Weibull half-life distribution.",
      v: "→ ctr_score · halflife_days · confidence"
    },
    {
      n: "Step 03",
      h: "Explain",
      p: "An agent calls Grad-CAM, benchmark, and improvement tools to ground the score in the regions of the image that drove it.",
      v: "→ agent.run() · 4 tools · streaming"
    }
  ];
  return (
    <section className="section">
      <div className="section-inner">
        <div className="section-head">
          <span className="eyebrow"><span className="bullet"></span>How it works</span>
          <h2 className="h2">Upload, score, explain.</h2>
        </div>
        <div className="flow">
          {steps.map((s, i) => (
            <div className="flow-step" key={i}>
              <span className="step-n">{s.n}</span>
              <h4>{s.h}</h4>
              <p>{s.p}</p>
              <div className="visual">{s.v}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Architecture() {
  return (
    <section className="section">
      <div className="section-inner">
        <div className="section-head">
          <span className="eyebrow"><span className="bullet"></span>Architecture</span>
          <h2 className="h2">Frozen vision backbone. Trainable head. Agent on top.</h2>
          <p style={{ color: "var(--text-3)", fontSize: 16, margin: 0, lineHeight: 1.6 }}>
            Literature is clear on datasets under 100K images: freeze the backbone, train a
            light head, and let the agent earn its keep on the explanation layer — not on
            re-discovering CLIP.
          </p>
        </div>

        <div className="arch">
          <div className="arch-arrows"></div>
          <div className="arch-node">
            <div className="lbl">Input</div>
            <div className="name">Creative</div>
            <div className="desc">PNG · 4:5 · ≤8MB</div>
          </div>
          <div className="arch-node">
            <div className="lbl">Vision</div>
            <div className="name">CLIP-ViT-B/32</div>
            <div className="desc">frozen · 768-d emb</div>
          </div>
          <div className="arch-node accent">
            <div className="lbl">Head</div>
            <div className="name">Multi-task MLP</div>
            <div className="desc">CTR + Weibull(α, β)</div>
          </div>
          <div className="arch-node">
            <div className="lbl">Agent</div>
            <div className="name">Qwen-2.5 · smolagents</div>
            <div className="desc">4 tools · streaming</div>
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 16 }} className="arch-detail">
          <div className="card">
            <div className="card-header"><div className="card-title">Multi-task loss <span className="kbd">loss.py</span></div></div>
            <div className="card-body" style={{ fontFamily: "var(--font-mono)", fontSize: 12.5, lineHeight: 1.7, color: "var(--text-2)"}}>
              <div><span style={{ color: "var(--text-3)" }}>L =</span> 0.5 · <span style={{ color: "var(--indigo-2)"}}>BCE</span>(ctr) + 0.5 · <span style={{ color: "var(--indigo-2)"}}>WeibullNLL</span>(α, β, t, c)</div>
              <div style={{ color: "var(--text-3)", marginTop: 8 }}>// censored=True when ad still running at scrape time</div>
              <div style={{ color: "var(--text-3)" }}>// clamp log inputs to [-10, 10] to stabilize tail</div>
            </div>
          </div>
          <div className="card">
            <div className="card-header"><div className="card-title">Agent toolbelt <span className="kbd">tools.py</span></div></div>
            <div className="card-body" style={{ fontFamily: "var(--font-mono)", fontSize: 12, lineHeight: 1.85, color: "var(--text-2)"}}>
              <div><span style={{ color: "var(--indigo-2)" }}>get_creative_score</span>(image_id)</div>
              <div><span style={{ color: "var(--indigo-2)" }}>get_heatmap_regions</span>(image_id)</div>
              <div><span style={{ color: "var(--indigo-2)" }}>get_benchmark</span>(vertical)</div>
              <div><span style={{ color: "var(--indigo-2)" }}>get_improvement_suggestions</span>(image_id)</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function Stats() {
  const items = [
    { v: "22,248", u: "", l: "training creatives" },
    { v: "3", u: "", l: "verticals · gaming · ecom · finance" },
    { v: "0.254", u: "", l: "Spearman r (test set)" },
    { v: "Weibull", u: "", l: "right-censored survival" }
  ];
  return (
    <section className="section" style={{ paddingTop: 0, borderBottom: "1px solid var(--border)" }}>
      <div className="section-inner">
        <div className="stats">
          {items.map((s, i) => (
            <div className="stat" key={i}>
              <div className="stat-val">
                {s.v}
                {s.u && <span className="unit">{s.u}</span>}
              </div>
              <div className="stat-lbl">{s.l}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div className="brand-mark" style={{ width: 18, height: 18 }}></div>
        <span style={{ fontSize: 13, color: "var(--text-3)"}}>Creative Intelligence Agent</span>
        <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-4)", fontSize: 11.5 }}>v1.0 · research preview</span>
      </div>
      <div className="footer-links">
        <a href="#"><IconLinkedin size={14} /> &nbsp;LinkedIn</a>
        <a href="#"><IconGithub size={14} /> &nbsp;GitHub</a>
        <a href="#"><IconUser size={14} /> &nbsp;Portfolio</a>
      </div>
      <div className="footer-meta">© 2026 · pcr12 · built on Hugging Face</div>
    </footer>
  );
}

Object.assign(window, { Landing });
