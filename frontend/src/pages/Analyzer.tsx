import { useState, useEffect, useRef } from 'react'
import {
  IconSparkle, IconEye, IconWarn, IconUpload,
  IconChevron, IconTool, IconUser, IconSend,
} from '../components/icons'
import { uploadCreative, sendChat, getUserMessage } from '../lib/api'
import { useAppStore } from '../store/appStore'

// ── hex colours for SVG (CSS vars are unreliable inside SVG across browsers) ──
const HEX = {
  good:       '#2d9e6d',
  warn:       '#c87d24',
  danger:     '#c44021',
  violet:     '#7c33e0',
  violetWash: '#7c33e014',
  surface3:   '#eeece9',
  text2:      '#5a5248',
  text3:      '#8a827b',
  text4:      '#a89e98',
  border:     '#e5e2de',
}

interface BenchRef { median: number; max: number }
interface Demo {
  ad_id: string; label: string; ctr: number; halflife: number
  confidence: number; dtype: string; imageSrc: string
  benchmark: { ctr: BenchRef; halflife: BenchRef }
  initial: { summary: string; trace: { tool: string; dur: number; out: string }[] }
  suggestions: string[]
}

const VERTICAL_MEDIANS: Record<string, number> = {
  gaming:    0.119,
  ecommerce: 0.125,
  finance:   0.111,
  other:     0.118,
}

const DEMOS: Record<string, Demo> = {
  gaming: {
    ad_id: 'apify_f321af3f4e59',
    label: 'Gaming · Mobile RPG',
    ctr: 0.3179, halflife: 33.04, confidence: 0.3641, dtype: 'wear-out',
    imageSrc: '/demo/gaming-ad.jpg',
    benchmark: { ctr: { median: 0.119, max: 0.80 }, halflife: { median: 10.7, max: 60 } },
    initial: {
      summary: 'This creative scores **above median for Gaming (CTR 0.32 vs 0.12 median)**, and the model predicts a **wear-out** fatigue pattern: half-life of 33.0 days — well above the vertical median of 10.7d. High confidence (0.36) on a real scraped ad. The driver is strong visual saliency concentrated on the central focal region.',
      trace: [
        { tool: 'get_creative_score',        dur: 142, out: 'ctr_score=0.3179, halflife_days=33.04, conf=0.3641' },
        { tool: 'get_heatmap_regions',        dur: 208, out: "high=['center_burst','cta_button'], low=['top_strip','margins']" },
        { tool: 'get_benchmark',              dur:  91, out: 'vertical=gaming, n=7485, median_ctr=0.119, median_hl=10.7d' },
        { tool: 'get_improvement_suggestions',dur: 318, out: "['add_face','reduce_saturation_band','multi_panel_variant']" },
      ],
    },
    suggestions: ['Why is the half-life so long?', 'How does this compare to gaming median?', 'What should I try next?'],
  },
  ecommerce: {
    ad_id: 'apify_6d945f5a1b9d',
    label: 'Ecommerce · DTC',
    ctr: 0.5063, halflife: 47.99, confidence: 0.0126, dtype: 'wear-out',
    imageSrc: '/demo/ecommerce-ad.jpg',
    benchmark: { ctr: { median: 0.125, max: 0.80 }, halflife: { median: 11.2, max: 60 } },
    initial: {
      summary: 'Predicted CTR 0.51 is **well above the ecommerce median of 0.12**, and half-life of 48.0d is strong. **Note: confidence is very low (0.01)** — the model is uncertain on this creative. Treat these scores as a rough signal only. The low confidence flag triggers when the embedding is far from the training distribution.',
      trace: [
        { tool: 'get_creative_score',        dur: 134, out: 'ctr_score=0.5063, halflife_days=47.99, conf=0.0126 ⚠ LOW' },
        { tool: 'get_heatmap_regions',        dur: 196, out: "high=['headline_strip'], low=['product','cta']" },
        { tool: 'get_benchmark',              dur:  88, out: 'vertical=ecommerce, n=7078, median_ctr=0.125, median_hl=11.2d' },
        { tool: 'get_improvement_suggestions',dur: 287, out: "['enlarge_product','increase_product_contrast','lifestyle_context']" },
      ],
    },
    suggestions: ['Why is confidence so low?', 'Which region should I fix first?', 'Show me a version that fixes this'],
  },
  finance: {
    ad_id: 'synthetic_591690dcf2f2',
    label: 'Finance · Neobank',
    ctr: 0.0993, halflife: 12.0, confidence: 0.8014, dtype: 'wear-out',
    imageSrc: '/demo/finance-ad.jpg',
    benchmark: { ctr: { median: 0.111, max: 0.80 }, halflife: { median: 10.0, max: 60 } },
    initial: {
      summary: 'Finance is a slow-CTR vertical. **Predicted CTR 0.10 is near the 0.11 median**, but half-life looks healthy at 12.0d (above the 10.0d median) — the creative is durable. High confidence (0.80) on this synthetic creative. Heatmap concentrates on the APY figure, which is correct behavior for a neobank ad.',
      trace: [
        { tool: 'get_creative_score',        dur: 156, out: 'ctr_score=0.0993, halflife_days=12.00, conf=0.8014' },
        { tool: 'get_heatmap_regions',        dur: 214, out: "high=['apy_number'], low=['legal_disclosure','footer']" },
        { tool: 'get_benchmark',              dur:  92, out: 'vertical=finance, n=6520, median_ctr=0.111, median_hl=10.0d' },
        { tool: 'get_improvement_suggestions',dur: 301, out: "['shrink_legal','add_trust_glyph','urgency_microcopy']" },
      ],
    },
    suggestions: ['Why is half-life higher than ecommerce?', 'Is the legal copy hurting me?', 'Suggest a variant with a face'],
  },
}

function pct(v: number, max: number) {
  return `${Math.max(0, Math.min(100, (v / max) * 100))}%`
}

export default function Analyzer({ navigate: _navigate }: { navigate: (p: string) => void }) {
  const [vertical, setVertical] = useState('gaming')
  const [dailyBudget, setDailyBudget] = useState(100)
  const [heat, setHeat] = useState(false)
  const [isResetting, setIsResetting] = useState(false)
  const [chatKey, setChatKey] = useState(0)
  const [dragOver, setDrag] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [detectedVertical, setDetectedVertical] = useState<string | null>(null)
  const [detectedConfidence, setDetectedConfidence] = useState<number>(0)
  const { isScoring, setIsScoring, setScore, setUploadId, score, uploadId } = useAppStore()
  const fileRef = useRef<HTMLInputElement>(null)

  const demo = DEMOS[vertical]
  // For demo mode use the demo object's scores; for upload use store score
  const displayCtr       = score && uploadId ? score.ctr_score   : demo.ctr
  const displayHalflife  = score && uploadId ? score.halflife_days : demo.halflife
  const displayConf      = score && uploadId ? score.confidence    : demo.confidence
  const currentImageId   = uploadId ?? demo.ad_id
  const displayImageSrc  = useAppStore(s => s.imageUrl) ?? demo.imageSrc

  async function handleFile(file: File) {
    if (!file) return
    setUploadError(null)
    setIsScoring(true)
    try {
      const result = await uploadCreative(file, vertical)
      setUploadId(result.uploadId)
      setScore({ ctr_score: result.ctr_score, halflife_days: result.halflife_days, confidence: result.confidence })
      useAppStore.getState().setImageUrl(URL.createObjectURL(file))
      if (result.heatmap_b64) useAppStore.getState().setHeatmap(result.heatmap_b64)
      if (result.predicted_vertical) {
        const validVerticals = Object.keys(DEMOS)  // ['gaming', 'ecommerce', 'finance']
        if (validVerticals.includes(result.predicted_vertical)) {
          setVertical(result.predicted_vertical)
        }
        setDetectedVertical(result.predicted_vertical)
        setDetectedConfidence(result.vertical_confidence ?? 0)
      }
      setChatKey(k => k + 1)
    } catch (err) {
      setUploadError(getUserMessage(err))
    } finally {
      setIsScoring(false)
    }
  }

  const dtype = score && uploadId
    ? (displayHalflife < 7 ? 'cut-out' : 'wear-out')
    : demo.dtype

  const lowConfidence       = displayConf < 0.2
  const medianCtr           = VERTICAL_MEDIANS[vertical] ?? 0.118
  const isAboveMedian       = displayCtr >= medianCtr
  const dailyWaste          = isAboveMedian ? 0 : ((medianCtr - displayCtr) / medianCtr) * dailyBudget
  const totalWaste          = dailyWaste * displayHalflife
  const optimalRotation     = Math.round(displayHalflife * Math.pow(-Math.log(0.5), 1 / 1.5))

  return (
    <>
      <div className="analyzer-bar">
        <div className="left">
          <span className="title-mini">Creative Analyzer</span>
          <span style={{ color: 'var(--text-4)' }}>·</span>
          <span className="muted mono" style={{ fontSize: 11.5 }}>
            upload_id: {uploadId ? uploadId.slice(0, 9) : '—'}
          </span>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <span className="muted" style={{ fontSize: 12.5 }}>Demo:</span>
          <div className="select">
            <select value={vertical} onChange={(e) => { setVertical(e.target.value); setDetectedVertical(null); setDetectedConfidence(0); useAppStore.getState().reset(); setChatKey(k => k + 1) }}>
              <option value="gaming">Gaming · Mobile RPG</option>
              <option value="ecommerce">Ecommerce · DTC</option>
              <option value="finance">Finance · Neobank</option>
            </select>
          </div>
          {detectedVertical && (
            <span className="chip indigo" style={{ fontSize: 11 }}>
              Auto-detected: {detectedVertical.charAt(0).toUpperCase() + detectedVertical.slice(1)} ({Math.round(detectedConfidence * 100)}%)
            </span>
          )}
          <button className="btn btn-sm" onClick={() => { setIsResetting(true); setHeat(false); setDetectedVertical(null); setDetectedConfidence(0); useAppStore.getState().reset(); setIsResetting(false) }}>
            <span className={isResetting ? 'spin' : ''}><IconSparkle size={12} /></span> Reset
          </button>
        </div>
      </div>

      <div className="analyzer">
        <div className="card creative-card">
          <div className="card-header">
            <div className="card-title">
              Creative <span className="kbd">{vertical}_demo.jpg</span>
            </div>
            <span className="mono" style={{ fontSize: 11, color: 'var(--text-3)' }}>224×224 · JPEG</span>
          </div>
          <div className="creative-area" style={{ position: 'relative', overflow: 'hidden' }}>
            <img
              src={displayImageSrc}
              alt={demo.label}
              style={{ position: 'relative', width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
            />
            {heat && useAppStore.getState().heatmapB64 && (
              <img
                src={`data:image/png;base64,${useAppStore.getState().heatmapB64}`}
                alt="heatmap"
                style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain', opacity: 0.4, pointerEvents: 'none' }}
              />
            )}
          </div>
          <div className="creative-controls">
            <div className="left">
              <button className={'toggle ' + (heat ? 'on' : '')} onClick={() => setHeat(!heat)}>
                <span className="sw"></span>
                <IconEye size={12} /> Attention overlay
              </button>
            </div>
            <div className="creative-meta">
              <span>backbone: siglip2-base-patch16-224</span>
              <span>·</span>
              <span>grad-cam: layer 11</span>
            </div>
          </div>
        </div>

        <div className="scores-panel">
          <div className="card">
            <div className="card-header">
              <div className="card-title">Prediction</div>
              <span className={'chip ' + (lowConfidence ? 'alert' : 'indigo')}>
                {lowConfidence && <IconWarn size={11} />} confidence {displayConf.toFixed(4)}
              </span>
            </div>
            {isScoring ? (
              <div style={{ padding: '32px 24px', textAlign: 'center', color: 'var(--text-2)', fontFamily: 'var(--font-mono)', fontSize: 13 }}>
                Scoring your creative… this may take up to 30s on first load<span className="cursor">▍</span>
              </div>
            ) : (
              <>
                <div className="scores-grid">
                  <div className="score-block">
                    <div className="score-label">Predicted CTR</div>
                    <CTRGauge value={displayCtr} median={demo.benchmark.ctr.median} max={demo.benchmark.ctr.max} />
                  </div>
                  <div className="score-block">
                    <div className="score-label">Fatigue half-life</div>
                    <div className="score-value">
                      {displayHalflife.toFixed(1)}<span className="unit">days</span>
                    </div>
                    <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 6 }}>
                      <span className={'chip ' + (dtype === 'cut-out' ? 'alert' : 'warn')}>{dtype}</span>
                      <span className={'score-delta ' + (displayHalflife >= demo.benchmark.halflife.median ? 'up' : 'down')}>
                        {displayHalflife >= demo.benchmark.halflife.median ? '▲' : '▼'}
                        {Math.abs(displayHalflife - demo.benchmark.halflife.median).toFixed(1)}d vs median
                      </span>
                    </div>
                    <WeibullCurve halflife={displayHalflife} />
                  </div>
                </div>
                <div className="bench">
                  <div className="bench-row">
                    <span className="lbl">CTR vs vert</span>
                    <div className="bench-track">
                      <div className="ticks"></div>
                      <span className="median" style={{ left: pct(demo.benchmark.ctr.median, demo.benchmark.ctr.max) }}></span>
                      <span className="you" style={{ left: pct(displayCtr, demo.benchmark.ctr.max) }}></span>
                    </div>
                    <span className="v">you {displayCtr.toFixed(2)}</span>
                  </div>
                  <div className="bench-row">
                    <span className="lbl">Half-life vs vert</span>
                    <div className="bench-track">
                      <div className="ticks"></div>
                      <span className="median" style={{ left: pct(demo.benchmark.halflife.median, demo.benchmark.halflife.max) }}></span>
                      <span className="you" style={{ left: pct(displayHalflife, demo.benchmark.halflife.max) }}></span>
                    </div>
                    <span className="v">you {displayHalflife.toFixed(1)}d</span>
                  </div>
                </div>
              </>
            )}
          </div>

          {score && uploadId && (
            <div className="card">
              <div className="card-header">
                <div className="card-title"><IconSparkle size={12} /> Budget Impact</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontSize: 12, color: 'var(--text-3)' }}>Daily budget</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <span style={{ fontSize: 12, color: 'var(--text-2)' }}>$</span>
                    <input
                      type="number"
                      min={1}
                      value={dailyBudget}
                      onChange={e => setDailyBudget(Math.max(1, Number(e.target.value)))}
                      style={{
                        width: 60, fontSize: 12, fontFamily: 'var(--font-mono)',
                        border: '1px solid var(--border)', borderRadius: 4,
                        padding: '2px 6px', textAlign: 'right',
                        background: 'var(--surface-2)', color: 'var(--text)',
                      }}
                    />
                  </div>
                </div>
              </div>
              {isAboveMedian ? (
                <div className="bench" style={{ background: 'var(--good-wash)', borderTop: 'none' }}>
                  <div className="bench-row">
                    <span className="lbl" style={{ flex: 1 }}>CTR advantage</span>
                    <span className="v" style={{ fontFamily: 'var(--font-mono)', color: HEX.good }}>+{(((displayCtr - medianCtr) / medianCtr) * 100).toFixed(0)}% vs median</span>
                  </div>
                  <div className="bench-row">
                    <span className="lbl" style={{ flex: 1 }}>Rotate by</span>
                    <span className="v" style={{ fontFamily: 'var(--font-mono)' }}>Day {optimalRotation}</span>
                  </div>
                  <div className="bench-row">
                    <span className="lbl" style={{ flex: 1 }}>Analysis cost</span>
                    <span className="v" style={{ fontFamily: 'var(--font-mono)', color: HEX.good }}>&lt; $0.01</span>
                  </div>
                </div>
              ) : (
                <div className="bench" style={{ borderTop: 'none' }}>
                  <div className="bench-row">
                    <span className="lbl" style={{ flex: 1 }}>Daily waste</span>
                    <span className="v" style={{ fontFamily: 'var(--font-mono)', color: HEX.danger }}>${Math.round(dailyWaste)}</span>
                  </div>
                  <div className="bench-row">
                    <span className="lbl" style={{ flex: 1 }}>Campaign waste</span>
                    <span className="v" style={{ fontFamily: 'var(--font-mono)', color: HEX.danger }}>${Math.round(totalWaste)}</span>
                  </div>
                  <div className="bench-row">
                    <span className="lbl" style={{ flex: 1 }}>Rotate by</span>
                    <span className="v" style={{ fontFamily: 'var(--font-mono)', color: HEX.warn }}>Day {optimalRotation}</span>
                  </div>
                  <div className="bench-row">
                    <span className="lbl" style={{ flex: 1 }}>Analysis cost</span>
                    <span className="v" style={{ fontFamily: 'var(--font-mono)', color: HEX.good }}>&lt; $0.01</span>
                  </div>
                </div>
              )}
            </div>
          )}

          <ChatPanel key={chatKey} demo={demo} imageId={currentImageId} vertical={vertical} isDemo={!uploadId} />
        </div>
      </div>

      <div
        className={'upload-zone' + (dragOver ? ' drag' : '')}
        onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f) }}
        onClick={() => fileRef.current?.click()}
      >
        <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f) }} />
        <div className="left">
          <div className="upload-icon"><IconUpload size={20} /></div>
          <div>
            <h4>Drop your own creative to score</h4>
            <p>PNG or JPEG · up to 8MB · gaming / ecommerce / finance verticals supported</p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span className="chip">Supabase Storage · creatives bucket</span>
          <button className="btn" onClick={(e) => { e.stopPropagation(); fileRef.current?.click() }}>Browse files</button>
        </div>
      </div>
      {uploadError && (
        <p style={{ marginTop: 8, fontSize: 12, color: '#c44021', fontFamily: 'var(--font-mono)' }}>
          {uploadError}
        </p>
      )}
    </>
  )
}

function CTRGauge({ value, median, max }: { value: number; median: number; max: number }) {
  const ratio = Math.min(1, value / max)
  const angle = -120 + ratio * 240
  const medianAngle = -120 + Math.min(1, median / max) * 240
  const r = 52, cx = 70, cy = 70
  const arc = describeArc(cx, cy, r, -120, angle)
  const bgArc = describeArc(cx, cy, r, -120, 120)
  const color = value >= median ? HEX.good : value >= median * 0.6 ? HEX.warn : HEX.danger
  const medX  = cx + (r + 10) * Math.cos((medianAngle * Math.PI) / 180)
  const medY  = cy + (r + 10) * Math.sin((medianAngle * Math.PI) / 180)
  const medX2 = cx + (r - 4)  * Math.cos((medianAngle * Math.PI) / 180)
  const medY2 = cy + (r - 4)  * Math.sin((medianAngle * Math.PI) / 180)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
      <svg width="140" height="120" viewBox="0 0 140 140" style={{ flexShrink: 0 }}>
        <path d={bgArc} stroke={HEX.surface3} strokeWidth="10" fill="none" strokeLinecap="round" />
        <path d={arc}   stroke={color}        strokeWidth="10" fill="none" strokeLinecap="round" />
        <line x1={medX} y1={medY} x2={medX2} y2={medY2} stroke={HEX.text2} strokeWidth="1.5" />
        <text x={cx} y={cy + 4}  textAnchor="middle" fontFamily="var(--font-serif)" fontSize="30" fill={HEX.violet} fontWeight="400" style={{ letterSpacing: '-0.02em' }}>{value.toFixed(2)}</text>
        <text x={cx} y={cy + 22} textAnchor="middle" fontFamily="var(--font-sans)"  fontSize="10" fill={HEX.text3}>predicted</text>
      </svg>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <div style={{ fontSize: 12, color: 'var(--text-3)' }}>vs vertical median</div>
        <div style={{ fontFamily: 'var(--font-serif)', fontSize: 22, color, letterSpacing: '-0.01em', lineHeight: 1 }}>
          {value >= median ? '+' : ''}{((value - median) / median * 100).toFixed(0)}%
        </div>
        <div style={{ fontSize: 11.5, color: 'var(--text-3)', marginTop: 6 }}>
          median {median.toFixed(3)}
        </div>
      </div>
    </div>
  )
}

function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number) {
  const start = polar(cx, cy, r, endAngle)
  const end   = polar(cx, cy, r, startAngle)
  const largeArc = endAngle - startAngle <= 180 ? '0' : '1'
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`
}
function polar(cx: number, cy: number, r: number, angle: number) {
  const rad = (angle * Math.PI) / 180
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
}

function WeibullCurve({ halflife }: { halflife: number }) {
  const w = 220, h = 40, k = 1.4
  const lambda = halflife / Math.pow(Math.log(2), 1 / k)
  const pts: [number, number][] = []
  const maxT = Math.max(21, halflife * 0.8)
  for (let i = 0; i <= 30; i++) {
    const t = (i / 30) * maxT
    const s = Math.exp(-Math.pow(t / lambda, k))
    pts.push([(i / 30) * w, h - s * (h - 4) - 2])
  }
  const d = pts.map((p, i) => (i === 0 ? 'M' : 'L') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ')
  const hlX = Math.min((halflife / maxT) * w, w - 2)
  return (
    <svg width={w} height={h + 14} style={{ marginTop: 8 }} viewBox={`0 0 ${w} ${h + 14}`}>
      <line x1="0" y1={h} x2={w} y2={h} stroke={HEX.border} strokeWidth="1" />
      <line x1={hlX} y1="0" x2={hlX} y2={h} stroke={HEX.warn} strokeDasharray="2 3" strokeWidth="1" />
      <path d={d} stroke={HEX.violet} strokeWidth="1.5" fill="none" />
      <path d={d + ` L ${w} ${h} L 0 ${h} Z`} fill={HEX.violetWash} />
      <text x={hlX + 4} y="10" fontFamily="var(--font-mono)" fontSize="9" fill={HEX.warn}>t½={halflife.toFixed(1)}d</text>
      <text x="0"     y={h + 12} fontFamily="var(--font-mono)" fontSize="9" fill={HEX.text4}>0d</text>
      <text x={w - 28} y={h + 12} fontFamily="var(--font-mono)" fontSize="9" fill={HEX.text4}>{maxT.toFixed(0)}d</text>
    </svg>
  )
}

interface TraceItem { tool: string; dur?: number; out?: string; duration_ms?: number; output?: unknown }
interface Message {
  role: 'user' | 'agent'
  time: string
  latency?: number
  text: string
  trace?: TraceItem[]
  open?: boolean
}

function ChatPanel({ demo, imageId, vertical, isDemo }: { demo: Demo; imageId: string; vertical: string; isDemo: boolean }) {
  const [messages, setMessages] = useState<Message[]>(
    isDemo
      ? [{ role: 'agent', time: 'now', latency: demo.initial.trace.reduce((s, t) => s + (t.dur ?? 0), 0),
           text: demo.initial.summary, trace: demo.initial.trace, open: false }]
      : []
  )
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const streamRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (streamRef.current) streamRef.current.scrollTop = streamRef.current.scrollHeight
  }, [messages, busy])

  useEffect(() => {
    if (!isDemo) send('Analyze this creative and give me the key performance insights.')
  }, [])

  function toggleTrace(idx: number) {
    setMessages(m => m.map((x, i) => i === idx ? { ...x, open: !x.open } : x))
  }

  async function send(q: string) {
    if (!q.trim() || busy) return
    const userMsg: Message = { role: 'user', time: new Date().toLocaleTimeString(), text: q }
    const history = messages.map(m => ({
      role: m.role === 'agent' ? 'assistant' : 'user',
      content: m.text,
    }))
    setMessages(m => [...m, userMsg])
    setText('')
    setBusy(true)
    try {
      const result = await sendChat(imageId, q, vertical, history)
      const agentMsg: Message = {
        role: 'agent',
        time: new Date().toLocaleTimeString(),
        latency: result.trace?.reduce((s: number, t: TraceItem) => s + (t.dur || 0), 0),
        text: result.answer,
        trace: result.trace,
        open: false,
      }
      setMessages(m => [...m, agentMsg])
    } catch (err) {
      setMessages(m => [...m, {
        role: 'agent', time: new Date().toLocaleTimeString(),
        text: getUserMessage(err), trace: [], open: false,
      }])
    } finally {
      setBusy(false)
    }
  }

  const lastIsAgent = messages.length > 0 && messages[messages.length - 1].role === 'agent'

  return (
    <div className="card chat">
      <div className="card-header">
        <div className="card-title">
          <span className={busy ? 'spin' : ''}><IconSparkle size={12} /></span> Explanation Agent
          <span className="kbd">claude-haiku-4-5</span>
        </div>
        <span className="mono" style={{ fontSize: 11, color: 'var(--text-3)' }}>5 tools loaded</span>
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
              <div className="msg-text mono" style={{ fontSize: 12, color: 'var(--text-3)' }}>
                <span className="pulse">calling tools</span><span className="cursor">▍</span>
              </div>
            </div>
          </div>
        )}
      </div>
      {lastIsAgent && !busy && demo.suggestions.length > 0 && (
        <div style={{ padding: '0 14px 10px', display: 'flex', gap: 6, flexWrap: 'wrap', borderTop: '1px solid var(--border)', paddingTop: 10 }}>
          <span className="mono" style={{ fontSize: 10.5, color: 'var(--text-4)', alignSelf: 'center', marginRight: 4 }}>try:</span>
          {demo.suggestions.map((s, i) => (
            <button key={i} className="btn btn-sm" style={{ fontSize: 11.5 }} onClick={() => send(s)}>{s}</button>
          ))}
        </div>
      )}
      <div className="composer">
        <input
          placeholder="Ask the agent about this creative…"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send(text)}
        />
        <button className="btn btn-primary btn-sm" onClick={() => send(text)} disabled={busy || !text.trim()}>
          <IconSend size={12} /> Send
        </button>
      </div>
    </div>
  )
}

function Msg({ m, idx, toggleTrace }: { m: Message; idx: number; toggleTrace: (i: number) => void }) {
  return (
    <div className={'msg ' + m.role}>
      <div className="msg-avatar">{m.role === 'agent' ? 'A' : <IconUser size={14} />}</div>
      <div className="msg-body">
        <div className="meta">
          <span>{m.role === 'agent' ? 'agent' : 'you'}</span>
          <span>{m.time}</span>
          {m.latency != null && <span>{m.latency}ms · 4 tools</span>}
        </div>
        <div className="msg-text" dangerouslySetInnerHTML={{ __html: renderMd(m.text) }}></div>
        {m.trace && m.trace.length > 0 && (
          <>
            <button className={'trace-toggle ' + (m.open ? 'open' : '')} onClick={() => toggleTrace(idx)}>
              <span className="arrow"><IconChevron size={10} /></span>
              {m.open ? 'Hide' : 'Show'} reasoning trace ({m.trace.length} steps)
            </button>
            {m.open && (
              <div className="trace">
                {m.trace.map((t, i) => (
                  <div key={i} className="trace-item">
                    <span className="trace-pin"><IconTool size={12} /></span>
                    <div className="trace-content">
                      <div><span className="trace-tool">{t.tool}</span>()</div>
                      <div className="trace-out">{typeof t.output === 'string' ? t.output : t.out ?? (t.output != null ? JSON.stringify(t.output) : '')}</div>
                    </div>
                    <span className="trace-dur">{t.duration_ms ?? t.dur}ms</span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function renderMd(t: string): string {
  return t
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+?)`/g, '<em>$1</em>')
    .replace(/\n/g, '<br>')
}
