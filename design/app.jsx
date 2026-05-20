/* App shell + hash router */
const { useState: useStateR, useEffect: useEffectR } = React;

function useHashRoute() {
  const [route, setRoute] = useStateR(() => parseHash());
  useEffectR(() => {
    const onHash = () => setRoute(parseHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  function navigate(path) {
    window.location.hash = "#" + path;
    window.scrollTo({ top: 0, behavior: "instant" });
  }
  return [route, navigate];
}
function parseHash() {
  const h = window.location.hash.replace(/^#/, "");
  if (h === "" || h === "/") return "/";
  if (h === "/app") return "/app";
  if (h === "/benchmark") return "/benchmark";
  return "/";
}

function Nav({ route, navigate }) {
  const link = (path, label) => (
    <span
      className={"nav-link " + (route === path ? "active" : "")}
      onClick={() => navigate(path)}
    >
      {label}
    </span>
  );
  return (
    <nav className="nav" data-screen-label={
      route === "/" ? "01 Landing" : route === "/app" ? "02 Analyzer" : "03 Benchmark"
    }>
      <div className="nav-left">
        <div className="brand" onClick={() => navigate("/")}>
          <div className="brand-mark"></div>
          <span className="brand-text">
            Creative Intelligence Agent <span className="tag mono">v1</span>
          </span>
        </div>
        <div className="nav-links">
          {link("/", "Overview")}
          {link("/app", "Analyzer")}
          {link("/benchmark", "Benchmark")}
        </div>
      </div>
      <div className="nav-right">
        <span className="status-pill">
          <span className="status-dot"></span> api · live
        </span>
        <button className="btn btn-sm">
          <IconGithub size={13} /> Source
        </button>
      </div>
    </nav>
  );
}

function App() {
  const [route, navigate] = useHashRoute();

  let page;
  if (route === "/app") page = <Analyzer navigate={navigate} />;
  else if (route === "/benchmark") page = <Benchmark navigate={navigate} />;
  else page = <Landing navigate={navigate} />;

  return (
    <div className="app-shell">
      <Nav route={route} navigate={navigate} />
      <main style={{ flex: 1, display: "flex", flexDirection: "column" }}>{page}</main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
