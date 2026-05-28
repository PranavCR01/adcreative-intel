import { useState, useEffect } from 'react'
import HealthBanner from './components/HealthBanner'
import { IconGithub } from './components/icons'
import Landing from './pages/Landing'
import Analyzer from './pages/Analyzer'
import Benchmark from './pages/Benchmark'

function parseHash(): string {
  const h = window.location.hash.replace(/^#/, '')
  if (h === '/app') return '/app'
  if (h === '/benchmark') return '/benchmark'
  return '/'
}

function useHashRoute(): [string, (path: string) => void] {
  const [route, setRoute] = useState(() => parseHash())
  useEffect(() => {
    const onHash = () => setRoute(parseHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  function navigate(path: string) {
    window.location.hash = '#' + path
    window.scrollTo({ top: 0, behavior: 'instant' as ScrollBehavior })
  }
  return [route, navigate]
}

function Nav({ route, navigate }: { route: string; navigate: (p: string) => void }) {
  const link = (path: string, label: string) => (
    <span
      className={'nav-link ' + (route === path ? 'active' : '')}
      onClick={() => navigate(path)}
    >
      {label}
    </span>
  )
  return (
    <nav className="nav" data-screen-label={
      route === '/' ? '01 Landing' : route === '/app' ? '02 Analyzer' : '03 Benchmark'
    }>
      <div className="nav-left">
        <div className="brand" onClick={() => navigate('/')}>
          <img src="/favicon.svg" style={{ width: 24, height: 24, verticalAlign: 'middle', marginRight: 2 }} alt="" />
          <span className="brand-text">
            Creative Intelligence Agent <span className="tag mono">v1</span>
          </span>
        </div>
        <div className="nav-links">
          {link('/', 'Overview')}
          {link('/app', 'Analyzer')}
          {link('/benchmark', 'Benchmark')}
        </div>
      </div>
      <div className="nav-right">
        <span className="status-pill">
          <span className="status-dot"></span> api · live
        </span>
        <a className="btn btn-sm" href="https://github.com/PranavCR01/adcreative-intel" target="_blank" rel="noopener noreferrer">
          <IconGithub size={13} /> Source
        </a>
      </div>
    </nav>
  )
}

export default function App() {
  const [route, navigate] = useHashRoute()

  let page
  if (route === '/app') page = <Analyzer navigate={navigate} />
  else if (route === '/benchmark') page = <Benchmark />
  else page = <Landing navigate={navigate} />

  return (
    <div className="app-shell">
      <Nav route={route} navigate={navigate} />
      <HealthBanner />
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>{page}</main>
    </div>
  )
}
