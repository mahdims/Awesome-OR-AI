// Notebook Builder

const NotebookBuilder = () => {
  const [q, setQ] = React.useState('self-instrumenting agent diagnostics');
  const [results, setResults] = React.useState(
    window.PAPERS.slice(0, 8).map((p,i) => ({ paper: p, checked: i < 5, reason: i<3 ? 'embedding similarity' : i<5 ? 'shared method' : 'keyword match' }))
  );

  const toggle = (id) => setResults(results.map(r => r.paper.id===id ? {...r, checked: !r.checked} : r));
  const n = results.filter(r=>r.checked).length;

  return (
    <div className="container" style={{paddingTop: 40, paddingBottom: 96}}>
      <div style={{maxWidth: 820, marginBottom: 28}}>
        <div className="eyebrow" style={{marginBottom: 14}}>Notebook builder</div>
        <h1 className="h-display">Curate a <em>shareable</em> paper set.</h1>
        <p style={{marginTop: 16, fontSize: 16, color: 'var(--ink-2)', maxWidth: 680}}>
          Query → ranked candidates with reasons → check off what you want → spawn a NotebookLM session.
          Saved sets persist as team artifacts.
        </p>
      </div>

      <div className="card" style={{padding: 20, marginBottom: 24}}>
        <div className="search-wrap" style={{marginBottom: 14}}>
          <Icon name="search" size={16}/>
          <input className="input" value={q} onChange={e=>setQ(e.target.value)} placeholder="What's this notebook about?"/>
        </div>
        <div className="between">
          <div style={{fontSize: 13, color:'var(--ink-3)'}}>{results.length} candidates · {n} selected</div>
          <div className="row" style={{gap:8}}>
            <button className="btn ghost sm">Search within results</button>
            <button className="btn ghost sm"><Icon name="plus" size={12} stroke={2}/> Add more</button>
          </div>
        </div>
      </div>

      <div className="card">
        {results.map((r,i) => (
          <div key={r.paper.id} className="paper-row" style={{gridTemplateColumns:'auto auto 1fr auto'}} onClick={()=>toggle(r.paper.id)}>
            <div style={{paddingTop:4}}>
              <div style={{
                width: 20, height: 20, borderRadius: 4,
                border: r.checked ? 'none' : '1.5px solid var(--hair-2)',
                background: r.checked ? 'var(--ink)' : 'transparent',
                display:'flex', alignItems:'center', justifyContent:'center',
                color: 'var(--canvas)'
              }}>
                {r.checked && <Icon name="check" size={14} stroke={2.5}/>}
              </div>
            </div>
            <ScoreRing value={Math.round(r.paper.priority||0)} label="priority"/>
            <div className="grow">
              <div style={{fontSize:14, fontWeight: 600, marginBottom: 4}}>{r.paper.title}</div>
              <div className="meta"><AuthorLine paper={r.paper}/></div>
              <div className="row" style={{marginTop: 8, gap: 10}}>
                <span className="mono" style={{fontSize:11, color:'var(--accent)'}}>↳ {r.reason}</span>
                {(r.paper.methods||[]).slice(0,2).map(m => <Chip key={m} mono>{m}</Chip>)}
              </div>
            </div>
            <div className="col" style={{alignItems:'flex-end'}}>
              <PriorityBadge score={r.paper.priority}/>
            </div>
          </div>
        ))}
      </div>

      <div className="row" style={{marginTop: 24, gap: 10, justifyContent:'flex-end'}}>
        <button className="btn ghost">Save list only</button>
        <button className="btn"><Icon name="headphones" size={14}/> Save + generate audio</button>
        <button className="btn primary"><Icon name="external" size={14}/> Spawn NotebookLM ({n})</button>
      </div>
    </div>
  );
};

window.NotebookBuilder = NotebookBuilder;
