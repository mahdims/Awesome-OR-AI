// Topic detail page (hero)

const SubdomainPage = ({ sdId, onOpenPaper, onNav }) => {
  const sd = window.SUBDOMAINS[sdId];
  if (!sd) return <div className="container" style={{padding: 60}}>Unknown subdomain.</div>;

  const papers = window.papersIn(sdId);
  const sota = window.sotaFor(sdId);
  const benches = window.benchmarksFor(sdId);
  const baselines = window.baselinesFor(sdId);
  const labs = window.labsFor(sdId);
  const gaps = window.GAPS[sdId] || [];
  const signals = window.SIGNALS[sdId] || [];
  const mustReadCount = papers.filter(p => p.must_read).length;

  const [filter, setFilter] = React.useState('all'); // all | must-read | 7d | 30d
  const [sort, setSort] = React.useState('date'); // date | priority
  const [expandedGap, setExpandedGap] = React.useState(null);

  const filtered = papers.filter(p => {
    if (filter === 'must-read') return p.must_read;
    if (filter === '30d') return (new Date(p.date)) > new Date(Date.now() - 30*86400000);
    return true;
  }).sort((a,b) => {
    if (sort === 'priority') return (b.priority||0) - (a.priority||0);
    return (b.date||'').localeCompare(a.date||'');
  });

  const extractClaim = (p) => {
    // Try to pull "+X%" or best result line from vs_baselines
    const vb = p.vs_baselines || {};
    const ks = Object.keys(vb);
    if (ks.length) {
      const best = ks.find(k => /\+\d+/.test(vb[k])) || ks[0];
      return { claim: vb[best], baseline: best };
    }
    // fallback: first brief sentence
    return { claim: (p.brief||'').split('.')[0], baseline: null };
  };

  return (
    <div className="container" style={{paddingTop: 24, paddingBottom: 96}}>
      {/* Breadcrumb */}
      <div className="row" style={{fontSize:13, color:'var(--ink-3)', marginBottom: 8}}>
        <button className="btn ghost sm" onClick={() => onNav('subdomains')} style={{marginLeft:-10}}>
          <Icon name="chevron" size={12} stroke={2}/> <span style={{transform:'rotate(180deg)', display:'inline-block'}}>← </span>All topics
        </button>
        <span className="divider"/>
        <span>{sd.category}</span>
      </div>

      {/* Hero */}
      <div className="sd-hero">
        <div className="eyebrow-row">
          <span className="eyebrow">Topic Detail</span>
          <span className="divider"/>
          <span className="mono">sd_id: {sdId}</span>
          <span className="divider"/>
          <span>Updated 2 days ago</span>
        </div>
        <h1>{sd.name}</h1>
        <p className="sd-lede">{sd.tagline}</p>

        <div className="sd-stats">
          {[
            {l: 'Papers', v: papers.length},
            {l: 'Must-read', v: mustReadCount},
            {l: 'Last 7 days', v: sd.weekly},
            {l: 'Benchmarks', v: benches.length},
            {l: 'Open gaps', v: gaps.length},
          ].map((k,i) => (
            <div key={i} className="sd-stat">
              <div className="v">{k.v}</div>
              <div className="l">{k.l}</div>
            </div>
          ))}
        </div>

        <div className="row" style={{marginTop: 24, gap: 10}}>
          <button className="btn primary"><Icon name="headphones" size={14}/> This week's audio</button>
          <button className="btn"><Icon name="book" size={14}/> Spawn notebook</button>
          <button className="btn ghost"><Icon name="bookmark" size={14}/> Follow subdomain</button>
        </div>
      </div>

      {/* Two-column layout: main + sidebar */}
      <div className="grid-3">
        <div>
          {/* Roughly SOTA */}
          <div className="section" style={{marginTop: 0}}>
            <SectionHeader
              title={<span>Roughly <em className="italic">SOTA</em></span>}
              hint="Top papers by priority. Claims are extracted from results.vs_baselines; expert judgment required."
              right={<button className="btn ghost sm">See all {papers.length} →</button>}
            />
            <div className="card">
              {sota.map((p, i) => {
                const c = extractClaim(p);
                return (
                  <div key={p.id} className="sota" onClick={() => onOpenPaper(p)}>
                    <div className="rank">{String(i+1).padStart(2, '0')}</div>
                    <div>
                      <div className="sota-title">{p.title}</div>
                      <div className="sota-claim">
                        {c.baseline && <><em>vs {c.baseline}:</em> </>}
                        {c.claim ? c.claim : (p.brief||'').split('.')[0]}.
                      </div>
                      <div className="sota-meta">
                        <AuthorLine paper={p}/>
                        <span className="divider"/>
                        <ConfidenceDot level={p.confidence_results}/>
                        <span style={{fontSize:11, color:'var(--ink-3)', marginLeft:-4}}>conf.results</span>
                        {p.must_read && <><span className="divider"/><Chip kind="accent" mono>must-read</Chip></>}
                        {p.framework_lineage && <><span className="divider"/><span className="mono" style={{fontSize:11, color:'var(--ink-3)'}}>lineage:{p.framework_lineage}</span></>}
                      </div>
                    </div>
                    <div className="col" style={{alignItems:'flex-end', gap: 8, minWidth: 60}}>
                      <div className="priority">p{(p.priority||0).toFixed(1)}</div>
                      <Icon name="chevron" size={14}/>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Canonical Benchmarks */}
          <div className="section">
            <SectionHeader
              title="Canonical benchmarks"
              hint="Ranked by how many papers in this subdomain report on them."
            />
            <div className="card" style={{padding: '4px 0'}}>
              <table className="table">
                <thead>
                  <tr>
                    <th>Benchmark</th>
                    <th style={{width: 180}}>Frequency</th>
                    <th>Latest result</th>
                    <th style={{width: 100}}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {benches.slice(0, 6).map(b => {
                    const lp = window.getPaper(b.latestPaper);
                    const lvb = lp?.vs_baselines ? Object.values(lp.vs_baselines)[0] : '—';
                    return (
                      <tr key={b.name} className="rowish" onClick={() => lp && onOpenPaper(lp)}>
                        <td>
                          <div style={{fontWeight: 500}}>{b.name}</div>
                          <div className="mono" style={{fontSize:11, color:'var(--ink-3)', marginTop:2}}>
                            {b.count} paper{b.count!==1?'s':''}
                          </div>
                        </td>
                        <td>
                          <div className="bar" style={{marginTop:10, width:150}}>
                            <span style={{width: `${Math.min(100, b.count/benches[0].count*100)}%`}}/>
                          </div>
                        </td>
                        <td style={{fontSize:13, color:'var(--ink-2)', maxWidth: 280}}>
                          {lp ? (
                            <div>
                              <span style={{color:'var(--ink)'}}>{lp.title.split(':')[0]}</span>
                              <span className="mono" style={{fontSize:11, color:'var(--ink-3)', marginLeft:6}}>{typeof lvb === 'string' ? lvb.slice(0,60) : '—'}</span>
                            </div>
                          ) : '—'}
                        </td>
                        <td className="mono" style={{fontSize:12, color:'var(--ink-3)'}}>{b.latestDate || '—'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Candidate Gaps */}
          <div className="section">
            <SectionHeader
              title="Candidate gaps"
              hint="Precomputed by matching benchmarks × methods, problem-property coverage, and cross-subdomain transfer. Not authoritative — check the evidence."
              right={<button className="btn ghost sm" onClick={() => onNav('gaps')}>Open Gap Explorer →</button>}
            />
            <div className="card">
              {gaps.map(g => {
                const open = expandedGap === g.id;
                return (
                  <div key={g.id} className="gap" onClick={() => setExpandedGap(open?null:g.id)}>
                    <div className="row" style={{justifyContent:'space-between', alignItems:'flex-start'}}>
                      <div style={{flex:1}}>
                        <div className="row" style={{gap: 10, marginBottom: 4}}>
                          <Chip kind={g.severity==='high'?'accent':g.severity==='med'?'ghost':'ghost'} mono>{g.kind}</Chip>
                          {g.severity === 'high' && <span className="mono" style={{fontSize:10, color:'var(--neg)', fontWeight:600, textTransform:'uppercase', letterSpacing:'0.08em'}}>High priority</span>}
                        </div>
                        <div className="g-title">{g.title}</div>
                        <div className="g-evidence">{g.evidence}</div>
                      </div>
                      <Icon name={open?'chevronDown':'chevron'} size={14}/>
                    </div>
                    {open && (
                      <div className="g-expand">
                        <div className="eyebrow" style={{marginBottom: 10}}>Derived from</div>
                        <div className="col" style={{gap: 8}}>
                          {(g.papers||[]).map(pid => {
                            const p = window.getPaper(pid);
                            if (!p) return <div key={pid} className="mono" style={{fontSize:12, color:'var(--ink-3)'}}>arXiv:{pid} (outside current corpus slice)</div>;
                            return (
                              <div key={pid} className="row" style={{padding:'8px 12px', background:'var(--surface-sunk)', borderRadius:'var(--radius-sm)'}}>
                                <PriorityBadge score={p.priority}/>
                                <span style={{fontSize:13, fontWeight:500, flex:1}}>{p.title}</span>
                                <button className="btn ghost sm" onClick={(e)=>{e.stopPropagation(); onOpenPaper(p);}}>Open</button>
                              </div>
                            );
                          })}
                          {!(g.papers||[]).length && <div className="mono" style={{fontSize:12, color:'var(--ink-3)'}}>Evidence is statistical (counts across corpus).</div>}
                        </div>
                        <div className="row" style={{marginTop: 14, gap: 8}}>
                          <button className="btn sm primary">Run novelty check on this gap →</button>
                          <button className="btn sm ghost">Mark as explored</button>
                          <button className="btn sm ghost">Dismiss</button>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Recent activity */}
          <div className="section">
            <SectionHeader
              title="Recent activity"
              hint="Reverse-chronological by default. Filters compose with sort."
            />
            <div className="filters">
              <div className="row gap-12">
                <div className="row" style={{gap:6}}>
                  <span className="mono" style={{fontSize:11, color:'var(--ink-3)', textTransform:'uppercase', letterSpacing:'0.08em'}}>Filter</span>
                  {[
                    {id:'all', label:'All'},
                    {id:'must-read', label: `Must-read (${mustReadCount})`},
                    {id:'30d', label: 'Last 30d'},
                  ].map(f => (
                    <button key={f.id} className={`filter-pill ${filter===f.id?'on':''}`} onClick={()=>setFilter(f.id)}>
                      {f.label}
                    </button>
                  ))}
                </div>
                <div className="row" style={{gap:6, marginLeft: 16}}>
                  <span className="mono" style={{fontSize:11, color:'var(--ink-3)', textTransform:'uppercase', letterSpacing:'0.08em'}}>Sort</span>
                  <div className="seg">
                    <button className={sort==='date'?'on':''} onClick={()=>setSort('date')}>Date</button>
                    <button className={sort==='priority'?'on':''} onClick={()=>setSort('priority')}>Priority</button>
                  </div>
                </div>
              </div>
              <div style={{marginLeft:'auto', fontSize:12, color:'var(--ink-3)'}}>
                Showing <b style={{color:'var(--ink)'}}>{filtered.length}</b> of {papers.length}
              </div>
            </div>
            <div>
              {filtered.slice(0, 12).map(p => <PaperRow key={p.id} paper={p} onOpen={onOpenPaper}/>)}
              {filtered.length > 12 && (
                <div style={{padding:'16px 18px', textAlign:'center', borderTop:'1px solid var(--hair)'}}>
                  <button className="btn ghost sm">Load {filtered.length - 12} more</button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <aside>
          {/* Emerging signals */}
          <div className="card" style={{padding: '22px 24px', marginBottom: 20}}>
            <div className="eyebrow" style={{marginBottom: 14, color:'var(--accent-ink)'}}>Emerging signals</div>
            <div>
              {signals.map((s, i) => (
                <div key={i} className="signal">
                  <div className="kind">{s.kind}</div>
                  <div className="body">
                    {s.body}
                    <div style={{marginTop:4}}>
                      <span className="trend">{s.trend}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div style={{marginTop: 14, paddingTop: 14, borderTop:'1px solid var(--hair)', fontSize:11, color:'var(--ink-3)', fontFamily:'var(--mono)'}}>
              Each signal is an independent counting check. None claim more than they can deliver.
            </div>
          </div>

          {/* Canonical baselines */}
          <div className="card" style={{padding: '22px 24px', marginBottom: 20}}>
            <div className="between" style={{marginBottom: 14, alignItems:'baseline'}}>
              <div className="eyebrow">Canonical baselines</div>
              <span className="mono" style={{fontSize:11, color:'var(--ink-3)'}}>top {Math.min(baselines.length,6)}</span>
            </div>
            {baselines.slice(0, 6).map(b => (
              <div key={b.name} className="row" style={{padding:'8px 0', borderTop:'1px solid var(--hair)', gap: 10}}>
                <span style={{flex:1, fontSize:13, fontWeight: 500}}>{b.name}</span>
                <span className="mono" style={{fontSize:11, color:'var(--ink-3)'}}>{b.count}×</span>
              </div>
            ))}
          </div>

          {/* Active labs */}
          <div className="card" style={{padding: '22px 24px', marginBottom: 20}}>
            <div className="between" style={{marginBottom: 14, alignItems:'baseline'}}>
              <div className="eyebrow">Active labs · 90d</div>
              <span className="mono" style={{fontSize:11, color:'var(--ink-3)'}}>affiliation counts</span>
            </div>
            {labs.slice(0, 7).map(l => (
              <div key={l.name} className="row" style={{padding:'7px 0', borderTop:'1px solid var(--hair)', gap: 10}}>
                <span style={{flex:1, fontSize:13}}>{l.name}</span>
                <div className="bar" style={{width: 40}}>
                  <span style={{width: `${Math.min(100, l.count/labs[0].count*100)}%`}}/>
                </div>
                <span className="mono" style={{fontSize:11, color:'var(--ink-3)', minWidth:18, textAlign:'right'}}>{l.count}</span>
              </div>
            ))}
          </div>

          {/* Notebooks & audio */}
          <div className="card" style={{padding: '22px 24px'}}>
            <div className="eyebrow" style={{marginBottom: 14}}>Team artifacts</div>
            <div className="col" style={{gap: 10}}>
              <div className="q-card">
                <div className="row" style={{gap:10}}>
                  <Icon name="headphones" size={16}/>
                  <div style={{flex:1}}>
                    <div style={{fontSize:13, fontWeight:500}}>Weekly audio · Apr 21</div>
                    <div style={{fontSize:11, color:'var(--ink-3)'}}>18 min · 5 new papers</div>
                  </div>
                  <button className="btn sm ghost">Play</button>
                </div>
              </div>
              <div className="q-card">
                <div className="row" style={{gap:10}}>
                  <Icon name="book" size={16}/>
                  <div style={{flex:1}}>
                    <div style={{fontSize:13, fontWeight:500}}>AlgoEvo reading list</div>
                    <div style={{fontSize:11, color:'var(--ink-3)'}}>Mahdi · 12 papers · updated 3d ago</div>
                  </div>
                  <button className="btn sm ghost">Open</button>
                </div>
              </div>
              <div className="q-card">
                <div className="row" style={{gap:10}}>
                  <Icon name="quote" size={16}/>
                  <div style={{flex:1}}>
                    <div style={{fontSize:13, fontWeight:500}}>Self-instrumenting deep-dive</div>
                    <div style={{fontSize:11, color:'var(--ink-3)'}}>Team · 3 papers · novelty report</div>
                  </div>
                  <button className="btn sm ghost">Open</button>
                </div>
              </div>
              <button className="btn ghost sm" style={{alignSelf:'flex-start', marginTop: 4}}>
                <Icon name="plus" size={12} stroke={2}/> New artifact
              </button>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
};

window.SubdomainPage = SubdomainPage;
