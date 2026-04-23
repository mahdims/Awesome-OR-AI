// Gap Explorer — cross-subdomain

const GapExplorer = ({ onOpenPaper, onNav }) => {
  const [sdFilter, setSdFilter] = React.useState('all');
  const [kindFilter, setKindFilter] = React.useState('all');
  const [expanded, setExpanded] = React.useState(null);
  const [hoverCell, setHoverCell] = React.useState(null);

  // Build a benchmark × method coverage matrix for the evo_llm_search subdomain
  const sd = 'evo_llm_search';
  const papers = window.papersIn(sd);
  const topBench = window.benchmarksFor(sd).slice(0, 10).map(b => b.name);
  const methodCounts = {};
  papers.forEach(p => (p.methods||[]).forEach(m => methodCounts[m] = (methodCounts[m]||0)+1));
  const topMeth = Object.entries(methodCounts).sort((a,b)=>b[1]-a[1]).slice(0, 8).map(([m])=>m);

  const cellCount = (bench, meth) =>
    papers.filter(p => (p.benchmarks||[]).includes(bench) && (p.methods||[]).includes(meth)).length;

  const allGaps = Object.entries(window.GAPS).flatMap(([sdId, gs]) => gs.map(g => ({...g, sdId, sdName: window.SUBDOMAINS[sdId].name})));
  const filtered = allGaps.filter(g => (sdFilter==='all' || g.sdId===sdFilter) && (kindFilter==='all' || g.kind===kindFilter));

  return (
    <div className="container" style={{paddingTop: 40, paddingBottom: 96}}>
      <div style={{maxWidth: 820, marginBottom: 36}}>
        <div className="eyebrow" style={{marginBottom: 14}}>Cross-subdomain gap detection</div>
        <h1 className="h-display">Where <em>no-one</em> has looked yet.</h1>
        <p style={{marginTop: 16, fontSize: 16, color: 'var(--ink-2)', maxWidth: 680}}>
          Benchmark×method sparsity, problem-property coverage holes, and cross-subdomain method transfer.
          Candidate gaps, not authoritative claims — always click through to the evidence.
        </p>
      </div>

      {/* Coverage matrix */}
      <div className="card" style={{padding: 28, marginBottom: 32}}>
        <div className="between" style={{marginBottom: 20, alignItems:'baseline'}}>
          <div>
            <div className="eyebrow" style={{marginBottom: 6}}>Coverage matrix · LLM-driven evolutionary search</div>
            <div className="serif" style={{fontSize: 22, letterSpacing:'-0.01em'}}>Benchmark × method — darker = more papers, striped = zero</div>
          </div>
          <select className="select" style={{width: 260}} defaultValue={sd}>
            {Object.entries(window.SUBDOMAINS).map(([id,s]) => <option key={id} value={id}>{s.name}</option>)}
          </select>
        </div>
        <div style={{display:'grid', gridTemplateColumns:`200px 1fr 60px`, gap: 14, alignItems:'start'}}>
          <div/>
          <div style={{display:'grid', gridTemplateColumns:`repeat(${topMeth.length}, 1fr)`, gap: 2}}>
            {topMeth.map(m => (
              <div key={m} style={{
                fontSize:10, color:'var(--ink-3)', fontFamily:'var(--mono)', padding:'4px 4px',
                writingMode:'vertical-rl', transform:'rotate(180deg)', height: 120, display:'flex', alignItems:'flex-end',
                lineHeight:1.1
              }}>{m.replace(/_/g,' ')}</div>
            ))}
          </div>
          <div/>

          {topBench.map((b, bi) => (
            <React.Fragment key={b}>
              <div style={{fontSize: 12, color: 'var(--ink-2)', padding:'6px 0', textAlign:'right', paddingRight: 12, fontWeight: 500}}>{b}</div>
              <div style={{display:'grid', gridTemplateColumns:`repeat(${topMeth.length}, 1fr)`, gap: 2}}>
                {topMeth.map(m => {
                  const c = cellCount(b, m);
                  const cls = c === 0 ? 'gap-cell' : c === 1 ? 'c1' : c === 2 ? 'c2' : c <= 4 ? 'c3' : 'c4';
                  const hoverKey = `${b}||${m}`;
                  return (
                    <div key={m} className={`matrix-cell ${cls}`}
                      onMouseEnter={()=>setHoverCell({b,m,c})}
                      onMouseLeave={()=>setHoverCell(null)}
                      title={`${c} papers at ${b} × ${m}`}>
                      {c === 0 ? '·' : c}
                    </div>
                  );
                })}
              </div>
              <div style={{fontSize:10, color:'var(--ink-3)', fontFamily:'var(--mono)', paddingTop:8, textAlign:'right', paddingRight: 6}}>
                {window.benchmarksFor(sd).find(x=>x.name===b)?.count || 0}×
              </div>
            </React.Fragment>
          ))}
        </div>

        {/* Hover readout */}
        <div style={{marginTop: 18, paddingTop: 18, borderTop: '1px solid var(--hair)', minHeight: 54}}>
          {hoverCell ? (
            <div className="row" style={{gap: 16}}>
              <div>
                <div className="eyebrow" style={{marginBottom: 4}}>
                  {hoverCell.c === 0 ? 'Candidate gap' : `${hoverCell.c} paper${hoverCell.c!==1?'s':''}`}
                </div>
                <div style={{fontSize: 14}}>
                  <b>{hoverCell.b}</b> <span style={{color:'var(--ink-3)'}}>×</span> <span className="mono" style={{fontSize:12}}>{hoverCell.m}</span>
                </div>
              </div>
              {hoverCell.c === 0 && (
                <button className="btn sm accent" style={{marginLeft:'auto'}}>Flag as gap</button>
              )}
            </div>
          ) : (
            <div style={{fontSize: 12, color: 'var(--ink-3)', fontFamily:'var(--mono)'}}>
              Hover a cell for details. Striped cells have both axes present elsewhere but never together.
            </div>
          )}
        </div>
      </div>

      {/* Gap list */}
      <div className="filters" style={{border:'none', padding:'0 0 16px 0'}}>
        <div className="row gap-12">
          <span className="mono" style={{fontSize:11, color:'var(--ink-3)', textTransform:'uppercase', letterSpacing:'0.08em'}}>Subdomain</span>
          <button className={`filter-pill ${sdFilter==='all'?'on':''}`} onClick={()=>setSdFilter('all')}>All</button>
          {Object.entries(window.SUBDOMAINS).filter(([id])=>window.GAPS[id]).map(([id, s]) => (
            <button key={id} className={`filter-pill ${sdFilter===id?'on':''}`} onClick={()=>setSdFilter(id)}>
              {s.name} <span className="mono" style={{fontSize:10, opacity:0.7}}>{window.GAPS[id].length}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="filters" style={{border:'none', padding:'0 0 20px 0'}}>
        <div className="row gap-12">
          <span className="mono" style={{fontSize:11, color:'var(--ink-3)', textTransform:'uppercase', letterSpacing:'0.08em'}}>Kind</span>
          {['all','benchmark×method','problem-coverage','cross-subdomain','unreplicated','under-benchmarked'].map(k => (
            <button key={k} className={`filter-pill ${kindFilter===k?'on':''}`} onClick={()=>setKindFilter(k)}>{k}</button>
          ))}
        </div>
      </div>

      <div className="card">
        {filtered.map(g => {
          const open = expanded === `${g.sdId}:${g.id}`;
          return (
            <div key={`${g.sdId}:${g.id}`} className="gap" onClick={()=>setExpanded(open?null:`${g.sdId}:${g.id}`)}>
              <div className="row" style={{justifyContent:'space-between', alignItems:'flex-start', gap:16}}>
                <div style={{flex:1}}>
                  <div className="row" style={{gap: 10, marginBottom: 6}}>
                    <Chip mono kind={g.severity==='high'?'accent':'ghost'}>{g.kind}</Chip>
                    <span className="mono" style={{fontSize:11, color:'var(--ink-3)'}}>{g.sdName}</span>
                    {g.severity === 'high' && <span className="mono" style={{fontSize:10, color:'var(--neg)', fontWeight:600, textTransform:'uppercase', letterSpacing:'0.08em'}}>High</span>}
                  </div>
                  <div className="g-title">{g.title}</div>
                  <div className="g-evidence">{g.evidence}</div>
                </div>
                <Icon name={open?'chevronDown':'chevron'} size={14}/>
              </div>
              {open && (
                <div className="g-expand">
                  <div className="eyebrow" style={{marginBottom: 10}}>Evidence papers</div>
                  <div className="col" style={{gap: 8}}>
                    {(g.papers||[]).map(pid => {
                      const p = window.getPaper(pid);
                      if (!p) return null;
                      return (
                        <div key={pid} className="row" style={{padding:'8px 12px', background:'var(--surface-sunk)', borderRadius:'var(--radius-sm)', gap:10}}>
                          <PriorityBadge score={p.priority}/>
                          <span style={{fontSize:13, fontWeight:500, flex:1}}>{p.title}</span>
                          <button className="btn ghost sm" onClick={(e)=>{e.stopPropagation(); onOpenPaper(p);}}>Open</button>
                        </div>
                      );
                    })}
                  </div>
                  <div className="row" style={{marginTop: 14, gap: 8}}>
                    <button className="btn sm primary">Novelty check this gap</button>
                    <button className="btn sm ghost">Open notebook</button>
                    <button className="btn sm ghost" onClick={(e)=>{e.stopPropagation(); onNav('subdomain', g.sdId);}}>See subdomain →</button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

window.GapExplorer = GapExplorer;
