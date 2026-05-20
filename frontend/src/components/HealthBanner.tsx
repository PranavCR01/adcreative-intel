import { useEffect, useState } from 'react'
import { useAppStore } from '../store/appStore'
import { IconWarn } from './icons'

export default function HealthBanner() {
  const modelWarm = useAppStore(s => s.modelWarm)
  const setModelWarm = useAppStore(s => s.setModelWarm)
  const [dismissed, setDismissed] = useState(false)
  const [checked, setChecked] = useState(false)

  useEffect(() => {
    let failures = 0
    let interval: ReturnType<typeof setInterval> | null = null
    const BACKEND_URL = import.meta.env.VITE_API_URL

    const check = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/health`, {
          signal: AbortSignal.timeout(10000),
        })
        if (res.ok) {
          setModelWarm(true)
          failures = 0
        } else {
          failures++
          if (failures >= 2) setModelWarm(false)
        }
      } catch {
        failures++
        if (failures >= 2) setModelWarm(false)
      }
      setChecked(true)
    }

    const timer = setTimeout(() => {
      check()
      interval = setInterval(check, 30000)
    }, 8000)

    return () => {
      clearTimeout(timer)
      if (interval) clearInterval(interval)
    }
  }, [setModelWarm])

  if (!checked || modelWarm || dismissed) return null

  return (
    <div className="health-banner">
      <div className="lhs">
        <IconWarn size={14} />
        <span>
          <strong>Backend is warming up.</strong> First inference may take ~30s while the
          Render service scales from cold start. Subsequent calls are &lt;1s.
        </span>
      </div>
      <button className="btn btn-sm btn-ghost" onClick={() => setDismissed(true)}>Dismiss</button>
    </div>
  )
}
