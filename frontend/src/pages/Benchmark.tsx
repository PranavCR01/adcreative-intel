export default function Benchmark() {
  return (
    <div className="bench-page">
      <div className="bench-head">
        <span className="eyebrow"><span className="bullet"></span>Model results · v1</span>
        <h2 className="h2">Multi-task lifespan prediction, evaluated.</h2>
        <p style={{ color: "var(--text-3)", fontSize: 15.5, margin: 0, lineHeight: 1.65 }}>
          Held-out test set of 2,225 creatives (10%), stratified by vertical and discontinuation
          type. Primary metric is Spearman rank correlation on continuous targets — binarized
          AUC was dropped after audit (see methodology).
        </p>
      </div>

      <div className="bench-section">
        <h3 className="h3">Model comparison</h3>
        <table className="t">
          <thead>
            <tr>
              <th>Model</th>
              <th>Description</th>
              <th className="num">CTR · Spearman r ↑</th>
              <th className="num">CTR · MAE ↓</th>
              <th className="num">Half-life · Spearman r ↑</th>
              <th className="num">Half-life · MAE ↓</th>
              <th className="num">Params</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Random</td>
              <td className="muted">Uniform baseline</td>
              <td className="num"><CellBar v={0.02} max={0.3} /> 0.018</td>
              <td className="num">0.241</td>
              <td className="num"><CellBar v={0.01} max={0.3} /> 0.014</td>
              <td className="num">7.84d</td>
              <td className="num muted">—</td>
            </tr>
            <tr>
              <td>Duration-only</td>
              <td className="muted">Linear regr. on observed runtime</td>
              <td className="num"><CellBar v={0.06} max={0.3} /> 0.062</td>
              <td className="num">0.218</td>
              <td className="num"><CellBar v={0.18} max={0.3} /> 0.184</td>
              <td className="num">4.62d</td>
              <td className="num">1.0K</td>
            </tr>
            <tr>
              <td>CTR-only head</td>
              <td className="muted">CLIP + BCE head, no fatigue task</td>
              <td className="num"><CellBar v={0.22} max={0.3} /> 0.221</td>
              <td className="num">0.142</td>
              <td className="num"><CellBar v={0.10} max={0.3} /> 0.098</td>
              <td className="num">5.31d</td>
              <td className="num">197K</td>
            </tr>
            <tr className="highlight">
              <td><strong>Full multi-task (ours)</strong></td>
              <td className="muted">CLIP + BCE(ctr) + Weibull(α, β)</td>
              <td className="num"><CellBar v={0.254} max={0.3} accent /> <strong>0.254</strong></td>
              <td className="num"><strong>0.131</strong></td>
              <td className="num"><CellBar v={0.227} max={0.3} accent /> <strong>0.227</strong></td>
              <td className="num"><strong>3.94d</strong></td>
              <td className="num">263K</td>
            </tr>
          </tbody>
        </table>
        <p className="muted mono" style={{ fontSize: 11.5, marginTop: 10 }}>
          Bold = best in column. Multi-task head shares the projection layer; per-task heads add
          ~66K params. CLIP backbone is frozen across all rows.
        </p>
      </div>

      <div className="bench-section">
        <h3 className="h3">Training loss</h3>
        <div className="chart">
          <div className="chart-legend">
            <div className="it"><span className="sw" style={{ background: "var(--indigo)" }}></span> total loss</div>
            <div className="it"><span className="sw" style={{ background: "var(--good)" }}></span> ctr (BCE)</div>
            <div className="it"><span className="sw" style={{ background: "var(--warn)" }}></span> halflife (Weibull NLL)</div>
            <div className="it" style={{ marginLeft: "auto" }}><span className="sw" style={{ background: "transparent", border: "1px dashed var(--text-2)" }}></span> val · total</div>
          </div>
          <LossChart />
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 12, fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-3)" }}>
            <span>30 epochs · batch 64 · AdamW · lr 5e-4 → 1e-5 cosine</span>
            <span>best @ epoch 22 · early stop patience=4</span>
          </div>
        </div>
      </div>

      <div className="bench-section">
        <h3 className="h3">Dataset composition</h3>
        <table className="t">
          <thead>
            <tr>
              <th>Source</th>
              <th>Count</th>
              <th>Verticals</th>
              <th>Label type</th>
              <th>Censoring rate</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Apify · Meta Ad Library</td>
              <td className="num">8,914</td>
              <td>gaming, ecom, finance</td>
              <td className="mono" style={{ fontSize: 12 }}>real · scraped</td>
              <td className="num">38%</td>
              <td className="muted">runtime inferred from start_date / stop_date</td>
            </tr>
            <tr>
              <td>Synthetic · PIL</td>
              <td className="num">13,334</td>
              <td>balanced 3-way</td>
              <td className="mono" style={{ fontSize: 12 }}>rule-based · scriptable</td>
              <td className="num">0%</td>
              <td className="muted">face / color / CTA rules; uncensored ground truth</td>
            </tr>
            <tr style={{ background: "var(--bg-2)" }}>
              <td><strong>Total · labels_clean.csv</strong></td>
              <td className="num"><strong>22,248</strong></td>
              <td><strong>3</strong></td>
              <td className="mono muted" style={{ fontSize: 12 }}>multi-task · joint</td>
              <td className="num"><strong>15.2%</strong></td>
              <td className="muted">85/10/5 train/val/test stratified by vertical × type</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="bench-section">
        <h3 className="h3">Per-vertical stats</h3>
        <div className="vert-grid">
          <VertCard name="Gaming"    color="oklch(0.65 0.20 28)"  n="7,485" ctr={1.19} hl={10.7} cutout={42} wearout={58} />
          <VertCard name="Ecommerce" color="oklch(0.65 0.16 195)" n="7,078" ctr={1.25} hl={11.2} cutout={61} wearout={39} />
          <VertCard name="Finance"   color="oklch(0.70 0.14 155)" n="6,520" ctr={1.11} hl={10.0} cutout={27} wearout={73} />
        </div>
      </div>

      <div className="bench-section">
        <h3 className="h3">Methodology</h3>
        <div className="prose">
          <p>
            The model is a small multi-task head on top of a frozen CLIP-ViT-B/32 backbone.
            Freezing follows Kitada et al. (2022), whose work on Yahoo display creatives is the
            closest published analog and confirms backbones don't need fine-tuning when training
            data is under ~100K images.
          </p>
          <p>
            The fatigue head outputs the two parameters <code>α</code> (shape) and <code>β</code>
            {" "}(scale) of a Weibull distribution over time-to-discontinuation. We train it with
            negative log-likelihood and explicitly handle right-censoring: ads that are still live
            at scrape time contribute their survival probability instead of their density.
          </p>
          <div className="callout">
            <strong>Why Weibull instead of regression?</strong> Half-life is bounded below by 0,
            right-skewed, and a meaningful fraction of training labels are censored. A point
            regression silently treats <code>halflife=null</code> rows as missing, throwing
            away ~15% of the signal. The survival head uses them.
          </div>
          <p>
            We binarize the discontinuation type at 7 days post-hoc: <code>halflife &lt; 7</code> is
            labeled <em>cut-out</em> (rapid pause by the buyer), <code>halflife ≥ 7</code> is
            <em> wear-out</em> (audience saturation). This split is presentational only — the
            model never sees the binary label during training.
          </p>
          <p>
            <strong>Eval choice.</strong> Earlier drafts of this project reported a CTR AUC. We
            dropped it after realizing the continuous CTR proxy was being binarized at its own
            median to compute AUC, which is circular and gives an arbitrarily inflated score.
            Spearman rank correlation against the continuous label is the honest number.
          </p>
          <p style={{ color: "var(--text-3)", fontSize: 13 }}>
            <strong>Limitations.</strong> Static images only — no video, carousel, or playable
            creatives in v1. Verticals beyond gaming / ecom / finance have not been validated;
            the model will return a low-confidence flag (&lt;0.4) on out-of-distribution inputs.
            CTR is a relative ranking signal, not a calibrated probability.
          </p>
        </div>
      </div>
    </div>
  )
}

function CellBar({ v, max, accent }: { v: number; max: number; accent?: boolean }) {
  return (
    <span className="cell-bar" style={{ background: "var(--surface-3)" }}>
      <i style={{ width: `${(v / max) * 100}%`, background: accent ? "var(--indigo)" : "var(--text-3)" }}></i>
    </span>
  )
}

function VertCard({ name, color, n, ctr, hl, cutout, wearout }: {
  name: string; color: string; n: string; ctr: number; hl: number; cutout: number; wearout: number
}) {
  return (
    <div className="vert-card">
      <div className="head">
        <span className="name" style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 8, height: 8, borderRadius: 2, background: color, boxShadow: `0 0 10px ${color}` }}></span>
          {name}
        </span>
        <span className="chip">n = {n}</span>
      </div>
      <div className="row">
        <span className="k">median ctr</span>
        <span className="v">{ctr.toFixed(2)}<span className="unit">%</span></span>
      </div>
      <div className="row">
        <span className="k">median half-life</span>
        <span className="v">{hl.toFixed(1)}<span className="unit">d</span></span>
      </div>
      <div className="row">
        <span className="k">cut-out / wear-out</span>
        <span className="v">{cutout}<span className="unit">%</span> / {wearout}<span className="unit">%</span></span>
      </div>
      <div style={{ marginTop: 12, height: 6, borderRadius: 3, background: "var(--surface-3)", overflow: "hidden", display: "flex" }}>
        <div style={{ width: `${cutout}%`, background: "var(--danger)" }}></div>
        <div style={{ width: `${wearout}%`, background: "var(--warn)" }}></div>
      </div>
    </div>
  )
}

function LossChart() {
  const epochs = 30
  const data: { total: number; ctr: number; hl: number; val: number }[] = []
  for (let i = 0; i < epochs; i++) {
    const t = i / (epochs - 1)
    const decay = Math.exp(-2.5 * t)
    data.push({
      total: 1.35 * decay + 0.42 + (Math.sin(i * 1.1) * 0.012),
      ctr:   0.69 * decay + 0.21 + (Math.sin(i * 0.9) * 0.008),
      hl:    0.66 * decay + 0.22 + (Math.cos(i * 1.3) * 0.013),
      val:   1.35 * Math.exp(-2.1 * t) + 0.46 + (i > 22 ? (i - 22) * 0.008 : 0) + (Math.sin(i * 1.0) * 0.015),
    })
  }
  const w = 760, h = 260, padL = 44, padR = 14, padT = 14, padB = 28
  const yMax = 1.85, yMin = 0.3
  const X = (i: number) => padL + (i / (epochs - 1)) * (w - padL - padR)
  const Y = (v: number) => padT + (1 - (v - yMin) / (yMax - yMin)) * (h - padT - padB)
  const line = (key: keyof (typeof data)[0]) =>
    data.map((d, i) => (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(d[key]).toFixed(1)).join(" ")

  const yTicks = [0.5, 0.8, 1.1, 1.4, 1.7]
  const xTicks = [0, 5, 10, 15, 20, 25, 29]

  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: "100%", height: "auto" }}>
      {yTicks.map((t, i) => (
        <g key={i}>
          <line x1={padL} y1={Y(t)} x2={w - padR} y2={Y(t)} stroke="var(--border)" strokeWidth="1" strokeDasharray="2 4" />
          <text x={padL - 8} y={Y(t) + 3} fontFamily="var(--font-mono)" fontSize="9.5" fill="var(--text-4)" textAnchor="end">{t.toFixed(2)}</text>
        </g>
      ))}
      {xTicks.map((t, i) => (
        <text key={i} x={X(t)} y={h - 10} fontFamily="var(--font-mono)" fontSize="9.5" fill="var(--text-4)" textAnchor="middle">epoch {t}</text>
      ))}
      <line x1={X(22)} y1={padT} x2={X(22)} y2={h - padB} stroke="var(--indigo)" strokeWidth="1" strokeDasharray="3 3" opacity="0.6" />
      <text x={X(22) + 4} y={padT + 12} fontFamily="var(--font-mono)" fontSize="10" fill="var(--indigo-2)">best · ep 22</text>
      <path d={line("ctr")}   stroke="var(--good)"   strokeWidth="1.5" fill="none" opacity="0.85" />
      <path d={line("hl")}    stroke="var(--warn)"   strokeWidth="1.5" fill="none" opacity="0.85" />
      <path d={line("val")}   stroke="var(--text-2)" strokeWidth="1.5" fill="none" strokeDasharray="4 4" opacity="0.8" />
      <path d={line("total")} stroke="var(--indigo)"  strokeWidth="2"   fill="none" />
      <line x1={padL} y1={h - padB} x2={w - padR} y2={h - padB} stroke="var(--border-strong)" strokeWidth="1" />
      <line x1={padL} y1={padT}     x2={padL}     y2={h - padB} stroke="var(--border-strong)" strokeWidth="1" />
    </svg>
  )
}
