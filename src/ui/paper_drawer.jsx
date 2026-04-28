// Paper drawer + Tweaks panel

const PaperDrawer = ({ paper, onClose }) => {
  if (!paper) return <div className="drawer-scrim"/>;
  // Initial status: whatever's already in the user's queue, otherwise 'unread'.
  const initialStatus = (window.QUEUE || []).find(q => q.paper_id === paper.id)?.status || 'unread';
  const [state, _setState] = React.useState(initialStatus);
  const setState = (next) => {
    _setState(next);
    if (window.api && window.ME) {
      window.api.setPaperState(paper.id, next, null).catch(err => {
        console.error('failed to persist paper state', err);
      });
    }
  };

  return (
    <>
      <div className="drawer-scrim open" onClick={onClose}/>
      <aside className="drawer open" onClick={e=>e.stopPropagation()}>
        <button className="drawer-close" onClick={onClose}><Icon name="close" size={18}/></button>
        <div className="drawer-header">
          <div className="row" style={{gap: 10, marginBottom: 12, flexWrap:'wrap'}}>
            <Chip mono>{paper.category}</Chip>
            <Chip mono>{window.SUBDOMAINS[paper.subdomain]?.name || paper.subdomain}</Chip>
            <span className="mono" style={{fontSize:11, color:'var(--ink-3)'}}>arXiv:{paper.id}</span>
            <span className="mono" style={{fontSize:11, color:'var(--ink-3)'}}>{paper.date}</span>
          </div>
          <h2 className="h-2" style={{lineHeight:1.25, marginBottom: 10}}>{paper.title}</h2>
          <div style={{fontSize:13, color:'var(--ink-2)'}}>
            <b>{Array.isArray(paper.authors)?paper.authors[0]:'—'} et al.</b> · {paper.affiliations}
          </div>
          <div className="row" style={{marginTop: 16, gap: 8, flexWrap:'wrap'}}>
            <SigMarkers paper={paper}/>
            <PriorityBadge score={paper.priority}/>
            <ConfidenceDot level={paper.confidence_results}/>
            <span style={{fontSize:11, color:'var(--ink-3)'}}>conf.results</span>
          </div>
          <div className="row" style={{marginTop: 16, gap: 6}}>
            <div className="seg">
              {['unread','reading','read','discarded'].map(s => (
                <button key={s} className={state===s?'on':''} onClick={()=>setState(s)}>{s}</button>
              ))}
            </div>
            <button className="btn ghost sm" style={{marginLeft:'auto'}}><Icon name="external" size={12}/> arXiv</button>
            {paper.code_url && <button className="btn ghost sm"><Icon name="external" size={12}/> Code</button>}
            <button className="btn sm"><Icon name="plus" size={12} stroke={2}/> Add to notebook</button>
          </div>
        </div>
        <div className="drawer-body">
          {paper.reasoning && (
            <div className="drawer-section">
              <div className="eyebrow" style={{marginBottom:8}}>Why this is significant</div>
              <div style={{fontSize:14, lineHeight:1.55, color:'var(--ink)'}}>{paper.reasoning}</div>
            </div>
          )}
          {paper.brief && (
            <div className="drawer-section">
              <div className="eyebrow" style={{marginBottom:8}}>Brief</div>
              <div style={{fontSize:14, lineHeight:1.6, color:'var(--ink-2)'}}>{paper.brief}</div>
            </div>
          )}

          <div className="grid-2 drawer-section" style={{paddingBottom: 20, borderBottom:'1px solid var(--hair)', marginBottom: 20}}>
            <div>
              <div className="eyebrow" style={{marginBottom:8}}>Relevance</div>
              <div className="col" style={{gap:8}}>
                {[['methodological','M'],['problem','P'],['inspirational','I']].map(([k,label]) => {
                  const v = paper.relevance?.[k] || 0;
                  return (
                    <div key={k} className="row" style={{gap:10}}>
                      <span className="mono" style={{fontSize:11, color:'var(--ink-3)', width:20}}>{label}</span>
                      <div className="bar" style={{flex:1, height: 8}}><span style={{width: `${v*10}%`}}/></div>
                      <span className="mono" style={{fontSize:12, minWidth:24, textAlign:'right'}}>{v}/10</span>
                    </div>
                  );
                })}
              </div>
            </div>
            <div>
              <div className="eyebrow" style={{marginBottom:8}}>Methodology</div>
              <div className="col" style={{gap:6}}>
                {paper.llm_model && <div style={{fontSize:13}}><span style={{color:'var(--ink-3)'}}>Model · </span><span className="mono">{paper.llm_model}</span></div>}
                {paper.framework_lineage && <div style={{fontSize:13}}><span style={{color:'var(--ink-3)'}}>Lineage · </span><span className="mono">{paper.framework_lineage}</span></div>}
                {paper.novelty_type && <div style={{fontSize:13}}><span style={{color:'var(--ink-3)'}}>Novelty · </span><span className="mono">{paper.novelty_type}</span></div>}
              </div>
            </div>
          </div>

          {paper.benchmarks?.length > 0 && (
            <div className="drawer-section">
              <div className="eyebrow" style={{marginBottom:10}}>Benchmarks · {paper.benchmarks.length}</div>
              <div className="tag-list">{paper.benchmarks.map(b => <Chip key={b} mono>{b}</Chip>)}</div>
            </div>
          )}

          {paper.baselines?.length > 0 && (
            <div className="drawer-section">
              <div className="eyebrow" style={{marginBottom:10}}>Baselines</div>
              <div className="tag-list">{paper.baselines.map(b => <Chip key={b} mono>{b}</Chip>)}</div>
            </div>
          )}

          {Object.keys(paper.vs_baselines||{}).length > 0 && (
            <div className="drawer-section">
              <div className="eyebrow" style={{marginBottom:10}}>Results vs baselines</div>
              <div className="col" style={{gap: 6}}>
                {Object.entries(paper.vs_baselines).slice(0, 6).map(([b,v]) => (
                  <div key={b} className="row" style={{padding:'8px 12px', background:'var(--surface-sunk)', borderRadius:'var(--radius-sm)', gap: 12, alignItems:'flex-start'}}>
                    <span className="mono" style={{fontSize:11, color:'var(--ink-3)', minWidth:140, paddingTop: 1}}>{b}</span>
                    <span style={{fontSize:13, flex:1, color:'var(--ink)'}}>{typeof v === 'string' ? v : JSON.stringify(v).slice(0,200)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {(paper.methods?.length || paper.problems?.length) && (
            <div className="drawer-section">
              <div className="eyebrow" style={{marginBottom: 10}}>Tags</div>
              {paper.methods?.length > 0 && (
                <div style={{marginBottom: 10}}>
                  <div style={{fontSize:11, color:'var(--ink-3)', marginBottom: 6}}>Methods</div>
                  <div className="tag-list">{paper.methods.map(m => <Chip key={m} mono>{m}</Chip>)}</div>
                </div>
              )}
              {paper.problems?.length > 0 && (
                <div>
                  <div style={{fontSize:11, color:'var(--ink-3)', marginBottom: 6}}>Problems</div>
                  <div className="tag-list">{paper.problems.map(m => <Chip key={m} mono>{m}</Chip>)}</div>
                </div>
              )}
            </div>
          )}

          <div className="drawer-section">
            <div className="eyebrow" style={{marginBottom:10}}>Subdomain assignment</div>
            <div className="row" style={{gap: 10}}>
              <Chip kind="ghost" mono>{window.SUBDOMAINS[paper.subdomain]?.name}</Chip>
              <span className="mono" style={{fontSize:11, color:'var(--ink-3)'}}>conf 0.86</span>
              <button className="btn ghost sm" style={{marginLeft:'auto'}}>Reassign</button>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};

window.PaperDrawer = PaperDrawer;
