/* Analyzer page (route: #/app) */
const { useState: useStateA, useEffect: useEffectA, useRef: useRefA } = React;

const DEMOS = {
  gaming: {
    label: "Gaming · Match-3",
    headline: "BLAST 5,000\nLEVELS",
    sub: "New event live now",
    cta: "Play Free",
    ctr: 2.84,
    halflife: 7.2,
    dtype: "wear-out",
    confidence: 0.71,
    benchmark: { ctr: { you: 2.84, median: 1.93, p25: 1.1, p75: 3.4, max: 5.2 }, halflife: { you: 7.2, median: 8.5, p25: 4.0, p75: 13.0, max: 22 } },
    initial: {
      summary: "This creative scores **above median for Gaming match-3 (CTR 2.84% vs 1.93% median)**, but the model flags a **wear-out** fatigue pattern: half-life predicted at 7.2 days, ~15% under the vertical median of 8.5d. The driver is concentration of visual attention on a **single high-saturation focal point** — strong for the first impressions, predictably saturating after that.",
      trace: [
        { tool: "get_creative_score", dur: 142, out: "ctr_score=0.0284, halflife_days=7.20, conf=0.71" },
        { tool: "get_heatmap_regions", dur: 208, out: "high=['center_burst', 'cta_button'], low=['top_strip','margins']" },
        { tool: "get_benchmark", dur: 91, out: "vertical=gaming, n=8214, median_ctr=0.0193, median_hl=8.5d" },
        { tool: "get_improvement_suggestions", dur: 318, out: "['add_face', 'reduce_saturation_band', 'multi_panel_variant']" }
      ]
    },
    suggestions: ["Why is the half-life short?", "How does this compare to gaming median?", "What should I try next?"]
  },
  ecommerce: {
    label: "Ecommerce · DTC",
    headline: "Soft. Loose.\nGet 20% off.",
    sub: "Sitewide · ends Sunday",
    cta: "Shop the Drop",
    ctr: 1.42,
    halflife: 4.1,
    dtype: "cut-out",
    confidence: 0.64,
    benchmark: { ctr: { you: 1.42, median: 1.71, p25: 0.9, p75: 2.6, max: 4.1 }, halflife: { you: 4.1, median: 6.2, p25: 3.0, p75: 9.8, max: 18 } },
    initial: {
      summary: "Predicted CTR 1.42% sits **below the ecommerce median of 1.71%**, and the Weibull head returns a half-life of 4.1d — within the **cut-out** band (<7d). The agent's read: the discount headline reads correctly, but the product region has low visual saliency — model attention is split across margins instead of focusing on the product.",
      trace: [
        { tool: "get_creative_score", dur: 134, out: "ctr_score=0.0142, halflife_days=4.10, conf=0.64" },
        { tool: "get_heatmap_regions", dur: 196, out: "high=['headline_strip'], low=['product','cta']" },
        { tool: "get_benchmark", dur: 88, out: "vertical=ecommerce, n=7102, median_ctr=0.0171, median_hl=6.2d" },
        { tool: "get_improvement_suggestions", dur: 287, out: "['enlarge_product', 'increase_product_contrast', 'lifestyle_context']" }
      ]
    },
    suggestions: ["Why cut-out and not wear-out?", "Which region should I fix first?", "Show me a version that fixes this"]
  },
  finance: {
    label: "Finance · Neobank",
    headline: "Up to 5.10%\nAPY. No fees.",
    sub: "FDIC-insured up to $250K",
    cta: "Open in 60s",
    ctr: 0.62,
    halflife: 11.3,
    dtype: "wear-out",
    confidence: 0.58,
    benchmark: { ctr: { you: 0.62, median: 0.74, p25: 0.4, p75: 1.1, max: 1.9 }, halflife: { you: 11.3, median: 9.8, p25: 5.5, p75: 14.0, max: 28 } },
    initial: {
      summary: "Finance is a slow-CTR vertical. **Predicted CTR 0.62% trails the 0.74% median**, but half-life looks healthy at 11.3d (above the 9.8d median) — the creative is durable, just under-clicked. Heatmap concentrates on the **APY figure**, which is the right behavior; the model penalizes the dense legal copy below it.",
      trace: [
        { tool: "get_creative_score", dur: 156, out: "ctr_score=0.0062, halflife_days=11.30, conf=0.58" },
        { tool: "get_heatmap_regions", dur: 214, out: "high=['apy_number'], low=['legal_disclosure','footer']" },
        { tool: "get_benchmark", dur: 92, out: "vertical=finance, n=6932, median_ctr=0.0074, median_hl=9.8d" },
        { tool: "get_improvement_suggestions", dur: 301, out: "['shrink_legal','add_trust_glyph','urgency_microcopy']" }
      ]
    },
    suggestions: ["Why is half-life higher than gaming?", "Is the legal copy hurting me?", "Suggest a variant with a face"]
  }
};

function Analyzer({ navigate }) {
  const [vertical, setVertical] = useStateA("gaming");
  const [heat, setHeat] = useStateA(false);
  const [coldBanner, setCold] = useStateA(true);
  const [dragOver, setDrag] = useStateA(false);
  const demo = DEMOS[vertical];

  // re-key chat on vertical change
  return (
    <>
      <div className="analyzer-bar">
        <div className="left">
          <span className="title-mini">Creative Analyzer</span>
          <span style={{ color: "var(--text-4)" }}>·</span>
          <span className="muted mono" style={{ fontSize: 11.5 }}>upload_id: 94c2-7a13</span>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <span className="muted" style={{ fontSize: 12.5 }}>Demo:</span>
          <div className="select">
            <select value={vertical} onChange={(e) => setVertical(e.target.value)}>
              <option value="gaming">Gaming · Match-3</option>
              <option value="ecommerce">Ecommerce · DTC</option>
              <option value="finance">Finance · Neobank</option>
            </select>
          </div>
          <button className="btn btn-sm" onClick={() => { setHeat(false); }}>
            <IconSparkle size={12} /> Reset
          </button>
        </div>
      </div>

      {coldBanner && (
        <div className="health-banner">
          <div className="lhs">
            <IconWarn size={14} />
            <span>
              <strong>Backend is warming up.</strong> First inference may take ~12s while the HF
              Space scales from cold start. Subsequent calls are &lt;1s.
            </span>
          </div>
          <button className="btn btn-sm btn-ghost" onClick={() => setCold(false)}>Dismiss</button>
        </div>
      )}

      <div className="analyzer">
        <div className="card creative-card">
          <div className="card-header">
            <div className="card-title">
              Creative <span className="kbd">{vertical}_{Math.floor(Math.random()*900+100)}.png</span>
            </div>
            <span className="mono" style={{ fontSize: 11, color: "var(--text-3)"}}>1080×1350 · 4:5</span>
          </div>
          <div className="creative-area">
            <CreativeMock vertical={vertical} demo={demo} heat={heat} />
          </div>
          <div className="creative-controls">
            <div className="left">
              <button className={"toggle " + (heat ? "on" : "")} onClick={() => setHeat(!heat)}>
                <span className="sw"></span>
                <IconEye size={12} /> Attention overlay
              </button>
            </div>
            <div className="creative-meta">
              <span>backbone: clip-vit-b/32</span>
              <span>·</span>
              <span>grad-cam: layer 11</span>
            </div>
          </div>
        </div>

        <div className="scores-panel">
          <div className="card">
            <div className="card-header">
              <div className="card-title">Prediction</div>
              <span className="chip indigo">confidence {demo.confidence.toFixed(2)}</span>
            </div>
            <div className="scores-grid">
              <div className="score-block">
                <div className="score-label">Predicted CTR</div>
                <CTRGauge value={demo.ctr} median={demo.benchmark.ctr.median} max={demo.benchmark.ctr.max} />
              </div>
              <div className="score-block">
                <div className="score-label">Fatigue half-life</div>
                <div className="score-value">
                  {demo.halflife.toFixed(1)}<span className="unit">days</span>
                </div>
                <div style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 6 }}>
                  <span className={"chip " + (demo.dtype === "cut-out" ? "alert" : demo.dtype === "wear-out" ? "warn" : "indigo")}>
                    {demo.dtype}
                  </span>
                  <span className={"score-delta " + (demo.halflife >= demo.benchmark.halflife.median ? "up" : "down")}>
                    {demo.halflife >= demo.benchmark.halflife.median ? "▲" : "▼"}
                    {Math.abs(demo.halflife - demo.benchmark.halflife.median).toFixed(1)}d vs median
                  </span>
                </div>
                <WeibullCurve halflife={demo.halflife} />
              </div>
            </div>
            <div className="bench">
              <div className="bench-row">
                <span className="lbl">CTR vs vert</span>
                <div className="bench-track">
                  <div className="ticks"></div>
                  <span className="median" style={{ left: pct(demo.benchmark.ctr.median, demo.benchmark.ctr.max) }}></span>
                  <span className="you" style={{ left: pct(demo.benchmark.ctr.you, demo.benchmark.ctr.max) }}></span>
                </div>
                <span className="v">you {demo.ctr.toFixed(2)}%</span>
              </div>
              <div className="bench-row">
                <span className="lbl">Half-life vs vert</span>
                <div className="bench-track">
                  <div className="ticks"></div>
                  <span className="median" style={{ left: pct(demo.benchmark.halflife.median, demo.benchmark.halflife.max) }}></span>
                  <span className="you" style={{ left: pct(demo.benchmark.halflife.you, demo.benchmark.halflife.max) }}></span>
                </div>
                <span className="v">you {demo.halflife.toFixed(1)}d</span>
              </div>
            </div>
          </div>

          <ChatPanel key={vertical} demo={demo} vertical={vertical} />
        </div>
      </div>

      <div
        className={"upload-zone" + (dragOver ? " drag" : "")}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); }}
      >
        <div className="left">
          <div className="upload-icon"><IconUpload size={20} /></div>
          <div>
            <h4>Drop your own creative to score</h4>
            <p>PNG or JPEG · up to 8MB · gaming / ecommerce / finance verticals supported</p>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span className="chip">no image stored in db · R2 only</span>
          <button className="btn">Browse files</button>
        </div>
      </div>
    </>
  );
}

function pct(v, max) {
  return `${Math.max(0, Math.min(100, (v / max) * 100))}%`;
}

function CreativeMock({ vertical, demo, heat }) {
  return (
    <div className={"creative-image " + vertical}>
      <div className="ad-tag">SPONSORED · placeholder</div>
      <div className="ad-product"></div>
      <div className="ad-headline" style={{ whiteSpace: "pre-line" }}>{demo.headline}</div>
      <div className="ad-sub">{demo.sub}</div>
      <div className="ad-cta">{demo.cta} →</div>
      <div className={"heatmap " + (heat ? "on" : "")}></div>
    </div>
  );
}

function CTRGauge({ value, median, max }) {
  const ratio = Math.min(1, value / max);
  const angle = -120 + ratio * 240;
  const medianAngle = -120 + Math.min(1, median / max) * 240;
  const r = 52;
  const cx = 70, cy = 70;
  const arc = describeArc(cx, cy, r, -120, angle);
  const bgArc = describeArc(cx, cy, r, -120, 120);
  const color = value >= median ? "var(--good)" : value >= median * 0.6 ? "var(--warn)" : "var(--danger)";
  const medX = cx + (r + 10) * Math.cos((medianAngle * Math.PI) / 180);
  const medY = cy + (r + 10) * Math.sin((medianAngle * Math.PI) / 180);
  const medX2 = cx + (r - 4) * Math.cos((medianAngle * Math.PI) / 180);
  const medY2 = cy + (r - 4) * Math.sin((medianAngle * Math.PI) / 180);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
      <svg width="140" height="120" viewBox="0 0 140 140" style={{ flexShrink: 0 }}>
        <path d={bgArc} stroke="var(--surface-3)" strokeWidth="10" fill="none" strokeLinecap="round" />
        <path d={arc} stroke={color} strokeWidth="10" fill="none" strokeLinecap="round" />
        <line x1={medX} y1={medY} x2={medX2} y2={medY2} stroke="var(--text-2)" strokeWidth="1.5" />
        <text x={cx} y={cy + 4} textAnchor="middle" fontFamily="var(--font-serif)" fontSize="30" fill="var(--violet)" fontWeight="400" style={{letterSpacing: "-0.02em"}}>{value.toFixed(2)}</text>
        <text x={cx} y={cy + 22} textAnchor="middle" fontFamily="var(--font-sans)" fontSize="10" fill="var(--text-3)">% predicted</text>
      </svg>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <div style={{ fontSize: 12, color: "var(--text-3)" }}>vs vertical median</div>
        <div style={{ fontFamily: "var(--font-serif)", fontSize: 22, color: color, letterSpacing: "-0.01em", lineHeight: 1 }}>
          {value >= median ? "+" : ""}{((value - median) / median * 100).toFixed(0)}%
        </div>
        <div style={{ fontSize: 11.5, color: "var(--text-3)", marginTop: 6 }}>
          median {median.toFixed(2)}%
        </div>
      </div>
    </div>
  );
}

function describeArc(cx, cy, r, startAngle, endAngle) {
  const start = polar(cx, cy, r, endAngle);
  const end = polar(cx, cy, r, startAngle);
  const largeArc = endAngle - startAngle <= 180 ? "0" : "1";
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`;
}
function polar(cx, cy, r, angle) {
  const rad = (angle * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function WeibullCurve({ halflife }) {
  // mini sparkline showing survival decay
  const w = 220, h = 40;
  const k = 1.4;
  const lambda = halflife / Math.pow(Math.log(2), 1 / k);
  const pts = [];
  const maxT = 21;
  for (let i = 0; i <= 30; i++) {
    const t = (i / 30) * maxT;
    const s = Math.exp(-Math.pow(t / lambda, k));
    pts.push([(i / 30) * w, h - s * (h - 4) - 2]);
  }
  const d = pts.map((p, i) => (i === 0 ? "M" : "L") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
  const hlX = (halflife / maxT) * w;
  return (
    <svg width={w} height={h + 14} style={{ marginTop: 8 }} viewBox={`0 0 ${w} ${h + 14}`}>
      <line x1="0" y1={h} x2={w} y2={h} stroke="var(--border)" strokeWidth="1" />
      <line x1={hlX} y1="0" x2={hlX} y2={h} stroke="var(--warn)" strokeDasharray="2 3" strokeWidth="1" />
      <path d={d} stroke="var(--indigo-2)" strokeWidth="1.5" fill="none" />
      <path d={d + ` L ${w} ${h} L 0 ${h} Z`} fill="var(--indigo-wash)" />
      <text x={hlX + 4} y="10" fontFamily="var(--font-mono)" fontSize="9" fill="var(--warn)">t½ = {halflife.toFixed(1)}d</text>
      <text x="0" y={h + 12} fontFamily="var(--font-mono)" fontSize="9" fill="var(--text-4)">0d</text>
      <text x={w - 22} y={h + 12} fontFamily="var(--font-mono)" fontSize="9" fill="var(--text-4)">21d</text>
    </svg>
  );
}

function ChatPanel({ demo, vertical }) {
  const [messages, setMessages] = useStateA([
    {
      role: "agent",
      time: "now",
      latency: 441,
      text: demo.initial.summary,
      trace: demo.initial.trace,
      open: false
    }
  ]);
  const [text, setText] = useStateA("");
  const [busy, setBusy] = useStateA(false);
  const streamRef = useRefA(null);

  useEffectA(() => {
    if (streamRef.current) {
      streamRef.current.scrollTop = streamRef.current.scrollHeight;
    }
  }, [messages, busy]);

  function toggleTrace(idx) {
    setMessages((m) => m.map((x, i) => (i === idx ? { ...x, open: !x.open } : x)));
  }

  function send(q) {
    if (!q.trim() || busy) return;
    const userMsg = { role: "user", time: "now", text: q };
    setMessages((m) => [...m, userMsg]);
    setText("");
    setBusy(true);

    const reply = buildReply(q, demo, vertical);
    setTimeout(() => {
      setMessages((m) => [...m, reply]);
      setBusy(false);
    }, 900);
  }

  const lastIsAgent = messages.length && messages[messages.length - 1].role === "agent";
  const currentSuggestions = lastIsAgent ? demo.suggestions : [];

  return (
    <div className="card chat">
      <div className="card-header">
        <div className="card-title">
          <IconSparkle size={12} /> Explanation Agent
          <span className="kbd">qwen-2.5-1.5b</span>
        </div>
        <span className="mono" style={{ fontSize: 11, color: "var(--text-3)" }}>4 tools loaded</span>
      </div>
      <div className="chat-stream" ref={streamRef}>
        {messages.map((m, i) => (
          <Msg key={i} m={m} idx={i} toggleTrace={toggleTrace} />
        ))}
        {busy && (
          <div className="msg agent">
            <div className="msg-avatar">A</div>
            <div className="msg-body">
              <div className="meta">agent · thinking</div>
              <div className="msg-text mono" style={{ fontSize: 12, color: "var(--text-3)"}}>
                <span className="pulse">calling tools</span><span className="cursor">▍</span>
              </div>
            </div>
          </div>
        )}
      </div>
      {currentSuggestions.length > 0 && !busy && (
        <div style={{ padding: "0 14px 10px", display: "flex", gap: 6, flexWrap: "wrap", borderTop: "1px solid var(--border)", paddingTop: 10 }}>
          <span className="mono" style={{ fontSize: 10.5, color: "var(--text-4)", alignSelf: "center", marginRight: 4 }}>try:</span>
          {currentSuggestions.map((s, i) => (
            <button key={i} className="btn btn-sm" style={{ fontSize: 11.5 }} onClick={() => send(s)}>{s}</button>
          ))}
        </div>
      )}
      <div className="composer">
        <input
          placeholder="Ask the agent about this creative…"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send(text)}
        />
        <button className="btn btn-primary btn-sm" onClick={() => send(text)} disabled={busy || !text.trim()}>
          <IconSend size={12} /> Send
        </button>
      </div>
    </div>
  );
}

function Msg({ m, idx, toggleTrace }) {
  return (
    <div className={"msg " + m.role}>
      <div className="msg-avatar">{m.role === "agent" ? "A" : <IconUser size={14} />}</div>
      <div className="msg-body">
        <div className="meta">
          <span>{m.role === "agent" ? "agent" : "you"}</span>
          <span>{m.time}</span>
          {m.latency && <span>{m.latency}ms · 4 tools</span>}
        </div>
        <div className="msg-text" dangerouslySetInnerHTML={{ __html: renderMd(m.text) }}></div>
        {m.trace && (
          <>
            <button className={"trace-toggle " + (m.open ? "open" : "")} onClick={() => toggleTrace(idx)}>
              <span className="arrow"><IconChevron size={10} /></span>
              {m.open ? "Hide" : "Show"} reasoning trace ({m.trace.length} steps)
            </button>
            {m.open && (
              <div className="trace">
                {m.trace.map((t, i) => (
                  <div key={i} className="trace-item">
                    <span className="trace-pin"><IconTool size={12} /></span>
                    <div className="trace-content">
                      <div><span className="trace-tool">{t.tool}</span>()</div>
                      <div className="trace-out">{t.out}</div>
                    </div>
                    <span className="trace-dur">{t.dur}ms</span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function renderMd(t) {
  return t
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+?)`/g, "<em>$1</em>")
    .replace(/\n/g, "<br>");
}

function buildReply(q, demo, vertical) {
  const ql = q.toLowerCase();
  let text = "", trace = [];
  if (/(half.?life|fatigue|short|long)/.test(ql)) {
    if (demo.dtype === "wear-out") {
      text = `Half-life of **${demo.halflife.toFixed(1)}d** lands in the **wear-out** band (≥7d). The Weibull β estimate is ~1.4 — gentle decay, but the model expects audience-side saturation by day 7 because attention is hyper-concentrated on one focal region (\`center_burst\`). High-saliency hooks fade fast on the same impression cohort. Counterintuitively, a *less* dominant focal element often extends half-life by 30–50% in this vertical.`;
    } else if (demo.dtype === "cut-out") {
      text = `Half-life of **${demo.halflife.toFixed(1)}d** is a **cut-out** (<7d). Weibull β ≈ 0.9 means failure is front-loaded — the creative is likely to be paused early by the buyer, not slowly fade. Driver: the product region has low attention (Grad-CAM weight 0.11) so users register the offer but not the object — bad combo for ecommerce where the product *is* the hook.`;
    } else {
      text = `Half-life of **${demo.halflife.toFixed(1)}d** is in the wear-out band but on the *longer* tail. Finance creatives decay slowly because audiences cross-check before clicking — the agent treats this as a feature, not a bug. The CTR is the lever to pull here, not the half-life.`;
    }
    trace = [
      { tool: "get_creative_score", dur: 138, out: `halflife_days=${demo.halflife}, weibull_beta≈${(demo.dtype === "cut-out" ? 0.9 : 1.4).toFixed(2)}` },
      { tool: "get_heatmap_regions", dur: 201, out: "attention_concentration=0.62 (high)" },
      { tool: "get_benchmark", dur: 84, out: `median_halflife=${demo.benchmark.halflife.median}d` }
    ];
  } else if (/(compare|median|benchmark|vs)/.test(ql)) {
    const ctrDelta = ((demo.ctr - demo.benchmark.ctr.median) / demo.benchmark.ctr.median * 100).toFixed(0);
    const hlDelta = ((demo.halflife - demo.benchmark.halflife.median) / demo.benchmark.halflife.median * 100).toFixed(0);
    text = `Against the **${vertical}** vertical (n=${vertical === "gaming" ? "8,214" : vertical === "ecommerce" ? "7,102" : "6,932"}):\n\n• CTR: **${demo.ctr}%** vs median **${demo.benchmark.ctr.median}%** → ${ctrDelta >= 0 ? "+" : ""}${ctrDelta}%\n• Half-life: **${demo.halflife}d** vs median **${demo.benchmark.halflife.median}d** → ${hlDelta >= 0 ? "+" : ""}${hlDelta}%\n\nNet read: this creative is ${demo.ctr >= demo.benchmark.ctr.median ? "**worth scaling**" : "**worth a variant pass**"} before scaling. The numbers are model predictions, not realized — back-test before committing budget.`;
    trace = [
      { tool: "get_benchmark", dur: 87, out: `vertical=${vertical}, n>6900, percentiles loaded` },
      { tool: "get_creative_score", dur: 119, out: `ctr=${demo.ctr}%, hl=${demo.halflife}d` }
    ];
  } else if (/(try|fix|improv|next|suggest|variant)/.test(ql)) {
    const suggs = vertical === "gaming"
      ? ["Add a human face — model assigns +1.2× halflife multiplier when face_detected=True", "Split the focal burst into two foci (e.g. character + reward) to spread attention", "Reduce saturation band by ~20% — high-sat hooks correlate with cut-out fatigue"]
      : vertical === "ecommerce"
      ? ["Enlarge the product to >35% of frame area — current is 22%", "Raise product-background contrast (current ΔE ≈ 18, target ≥ 28)", "Move the discount chip off the headline strip and onto the product — lifts ecommerce CTR by ~9% in our val set"]
      : ["Compress the legal disclosure into a footer link — frees attention budget", "Add an FDIC trust glyph near the APY number — predicted +0.18% CTR", "Add a microcopy timer (\"60-second open\") — finance verticals reward urgency"];
    text = `Three variant directions, ranked by expected lift on the held-out val set:\n\n• **${suggs[0]}**\n• **${suggs[1]}**\n• **${suggs[2]}**\n\nWant me to simulate a variant? I can re-score against the same vertical benchmark.`;
    trace = [
      { tool: "get_improvement_suggestions", dur: 294, out: `n_candidates=18, top_3 returned` },
      { tool: "get_heatmap_regions", dur: 188, out: "anchor regions identified for rewrite" }
    ];
  } else if (/(region|heatmap|attention|grad.?cam|look)/.test(ql)) {
    text = `Grad-CAM on the projection layer's pre-pool activations shows:\n\n• **High attention** (>0.4): center burst, CTA button\n• **Medium**: headline glyphs\n• **Low** (<0.1): top strip, margins, legal copy\n\nThe model is reading this creative the way a human would in the first 800ms — which is good for CTR but is also why half-life suffers: there's no secondary discovery layer for the audience to find on impression #3.`;
    trace = [
      { tool: "get_heatmap_regions", dur: 213, out: "regions=4, max_weight=0.71 (center_burst)" }
    ];
  } else {
    text = `I can answer questions grounded in the model's tools — CTR prediction, Weibull half-life, attention regions, vertical benchmarks, and improvement suggestions. Try one of the prompts below, or ask about a specific region of the creative.`;
    trace = [{ tool: "router", dur: 22, out: "no matching tool — returned guidance" }];
  }
  return {
    role: "agent",
    time: "just now",
    latency: trace.reduce((s, t) => s + t.dur, 0) + 80,
    text,
    trace,
    open: false
  };
}

Object.assign(window, { Analyzer });
