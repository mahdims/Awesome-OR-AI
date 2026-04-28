// App shell — header, nav, routing

const App = () => {
  const saved = (() => {
    try { return JSON.parse(localStorage.getItem('ri_nav') || 'null'); } catch { return null; }
  })();
  const [tab, setTab] = React.useState(saved?.tab || 'today');
  const [sdId, setSdId] = React.useState(saved?.sdId || window.DEFAULT_SDID || null);
  const [paper, setPaper] = React.useState(null);

  React.useEffect(() => {
    localStorage.setItem('ri_nav', JSON.stringify({tab, sdId}));
  }, [tab, sdId]);

  const nav = (t, arg) => {
    setTab(t);
    if (t === 'subdomain' && arg) setSdId(arg);
    window.scrollTo({top:0, behavior:'instant'});
  };

  const total = window.PAPERS.length;
  const mr = window.PAPERS.filter(p=>p.must_read).length;
  const gapTotal = Object.values(window.GAPS).flat().length;

  return (
    <div className="shell">
      <header className="header">
        <div className="header-top">
          <div className="brand">
            <div className="mark">Research <em>Intelligence</em></div>
            <div className="sub">OR × AI · v2</div>
          </div>
          <div className="header-meta">
            <div className="search-wrap" style={{width: 240}}>
              <Icon name="search" size={14}/>
              <input className="input" placeholder="Search corpus…" style={{padding:'6px 12px 6px 34px', fontSize: 12}}/>
            </div>
            <span><span className="pulse"/>Updated <b>2d ago</b></span>
            <span><b>{total}</b> papers · <b>{mr}</b> must-read · <b>{gapTotal}</b> gaps</span>
            <span className="mono" style={{fontSize:11}}>{window.ME?.email || 'guest'}</span>
          </div>
        </div>
        <nav className="nav">
          {window.NAV.map(n => (
            <button key={n.id} className={`nav-btn ${tab===n.id ? 'active' : ''}`} onClick={() => nav(n.id)}>
              {n.label}
              {n.id === 'subdomains' && <span className="badge">{Object.keys(window.SUBDOMAINS).length}</span>}
              {n.id === 'gaps' && <span className="badge">{gapTotal}</span>}
              {n.id === 'queue' && <span className="badge">{mr}</span>}
            </button>
          ))}
        </nav>
      </header>

      <main style={{flex:1}}>
        {tab === 'today' && <window.Today onNav={nav} onOpenPaper={setPaper}/>}
        {tab === 'subdomains' && <window.SubdomainsHome onNav={nav}/>}
        {tab === 'subdomain' && <window.SubdomainPage sdId={sdId} onOpenPaper={setPaper} onNav={nav}/>}
        {tab === 'feed' && <window.Feed onOpenPaper={setPaper}/>}
        {tab === 'novelty' && <window.NoveltyCheck onOpenPaper={setPaper}/>}
        {tab === 'notebook' && <window.NotebookBuilder/>}
        {tab === 'gaps' && <window.GapExplorer onOpenPaper={setPaper} onNav={nav}/>}
        {tab === 'queue' && <window.MyQueue onOpenPaper={setPaper}/>}
      </main>

      <footer style={{padding:'32px 0 48px', borderTop:'1px solid var(--hair)', textAlign:'center'}}>
        <div className="container">
          <div className="mono" style={{fontSize:11, color:'var(--ink-3)', textTransform:'uppercase', letterSpacing:'0.1em'}}>
            Research Intelligence · curated taxonomy over {total} papers · generated Apr 21 2026
          </div>
        </div>
      </footer>

      {paper && <window.PaperDrawer paper={paper} onClose={()=>setPaper(null)}/>}
      <window.TweaksPanel/>
    </div>
  );
};

// Wait for /api/init before mounting so the App component's first render
// already has window.PAPERS / SUBDOMAINS / ME populated.
const _mount = () => ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
if (window.__ri_init_promise && typeof window.__ri_init_promise.then === 'function') {
  window.__ri_init_promise.then(_mount);
} else {
  _mount();
}
