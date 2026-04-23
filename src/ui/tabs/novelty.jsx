// Novelty Check — verdict-first

const NoveltyCheck = ({ onOpenPaper }) => {
  const [idea, setIdea] = React.useState("Combine Elo-tournament selection with self-instrumenting agents to train a code-evolving LLM on IndustryOR, using solver execution as the sole reward signal.");
  const [sd, setSd] = React.useState('evo_llm_search');
  const [result, setResult] = React.useState(null);
  const [running, setRunning] = React.useState(false);

  const run = () => {
    setRunning(true);
    setTimeout(() => {
      const papers = window.papersIn(sd);
      const novel = papers.filter(p => (p.methods||[]).some(m=>/elo|instrumenting|tournament/i.test(m))).slice(0,2);
      const related = papers.sort((a,b)=>(b.priority||0)-(a.priority||0)).slice(0, 5);
      setResult({
        score: 62,
        novel: ['Combining Elo-tournament selection with solver-execution rewards (no paper does both)',
                'Self-instrumenting diagnostics applied to OR code rather than general RL agents'],
        not_novel: [
          { point: 'Outcome-only reward from solver feedback', paper: window.getPaper('2604.00442') },
          { point: 'Elo rating for validation-free evolution', paper: window.getPaper('2604.04347') },
          { point: 'Self-instrumenting agents in evolutionary search', paper: window.getPaper('2604.04347') },
        ],
        one_step: related.slice(0, 3),
        closest: related.slice(0, 5),
        baselines_needed: ['ORLM (SFT)', 'EVOM (outcome-only RL)', 'RoboPhD (Elo on Text2SQL)', 'AutoOR (GRPO + curriculum)'],
        benchmarks_needed: ['IndustryOR', 'OptiBench', 'MAMO-Complex', 'Pump-NLP (stretch)'],
      });
      setRunning(false);
    }, 900);
  };

  return (
    <div className="container" style={{paddingTop: 40, paddingBottom: 96}}>
      <div style={{maxWidth: 820, marginBottom: 36}}>
        <div className="eyebrow" style={{marginBottom: 14}}>Novelty check · user-initiated query</div>
        <h1 className="h-display">
          Is this <em>idea</em> new?
        </h1>
        <p style={{marginTop: 16, fontSize: 16, color: 'var(--ink-2)', maxWidth: 680}}>
          Describe a research idea. We find the closest prior work across {window.PAPERS.length} curated papers,
          score what's novel and what isn't, and tell you which baselines and benchmarks you'd need to compare against.
        </p>
      </div>

      {/* Input */}
      <div className="card" style={{padding: 24, marginBottom: 32}}>
        <div style={{display:'grid', gridTemplateColumns:'1fr auto', gap: 20}}>
          <div>
            <div className="eyebrow" style={{marginBottom:8}}>Describe the idea</div>
            <textarea className="textarea" value={idea} onChange={e=>setIdea(e.target.value)}
              rows={4} placeholder="One or two paragraphs about what you want to try..."/>
          </div>
          <div style={{width: 280}}>
            <div className="eyebrow" style={{marginBottom:8}}>Target subdomain</div>
            <select className="select" value={sd} onChange={e=>setSd(e.target.value)}>
              {Object.entries(window.SUBDOMAINS).map(([id, s]) => (
                <option key={id} value={id}>{s.name}</option>
              ))}
            </select>
            <div style={{fontSize:11, color:'var(--ink-3)', marginTop: 8}}>
              Scopes baselines/benchmarks to this subdomain's canonical set.
            </div>
            <button className="btn accent w-full" style={{marginTop: 18, justifyContent:'center', padding:'12px 16px'}} onClick={run} disabled={running}>
              {running ? 'Running…' : <><Icon name="spark" size={14}/> Run novelty check</>}
            </button>
          </div>
        </div>
      </div>

      {!result && (
        <div className="placeholder">
          <h4 className="serif" style={{fontSize: 20}}>No query run yet.</h4>
          <div className="note">Hit <span className="kbd">Run novelty check</span> to see a verdict.</div>
        </div>
      )}

      {result && (
        <div>
          {/* Verdict */}
          <div className="novelty-verdict">
            <div>
              <div className="novelty-label">Novelty score</div>
              <div className="novelty-big">{result.score}<span className="unit">/100</span></div>
              <div style={{marginTop: 14, fontSize: 13, color:'var(--ink-3)', fontFamily:'var(--mono)'}}>
                moderate · combinatorial
              </div>
            </div>
            <div>
              <div className="novelty-label">Verdict</div>
              <div className="serif" style={{fontSize: 26, lineHeight: 1.3, fontWeight: 400, letterSpacing:'-0.01em', maxWidth: 680, marginBottom: 28, textWrap: 'pretty'}}>
                <span className="italic">Combinatorial novelty</span>: your three components each have prior art, but their <em style={{fontStyle:'normal', fontWeight:600}}>composition</em> appears unpublished.
              </div>
              <div>
                <div className="stacked-bar">
                  <span className="s-novel" style={{width:'22%'}}/>
                  <span className="s-onestep" style={{width:'40%'}}/>
                  <span className="s-notnovel" style={{width:'38%'}}/>
                </div>
                <div className="row" style={{marginTop: 10, gap: 22, fontSize: 12, color:'var(--ink-3)'}}>
                  <span><span className="dot" style={{background:'var(--accent)'}}/> &nbsp;Novel <b style={{color:'var(--ink)'}}>22%</b></span>
                  <span><span className="dot" style={{background:'oklch(70% 0.11 var(--accent-h))'}}/> &nbsp;One step away <b style={{color:'var(--ink)'}}>40%</b></span>
                  <span><span className="dot" style={{background:'var(--hair-2)'}}/> &nbsp;Not novel <b style={{color:'var(--ink)'}}>38%</b></span>
                </div>
              </div>
            </div>
          </div>

          {/* Breakdown */}
          <div className="grid-2" style={{marginTop: 32}}>
            <div className="card" style={{padding: 24}}>
              <div className="eyebrow" style={{color:'var(--accent-ink)', marginBottom:14}}>What's novel</div>
              <ol style={{paddingLeft: 22, margin: 0, fontSize: 14, lineHeight: 1.55, color: 'var(--ink)'}}>
                {result.novel.map((n,i) => <li key={i} style={{marginBottom: 12}}>{n}</li>)}
              </ol>
            </div>
            <div className="card" style={{padding: 24}}>
              <div className="eyebrow" style={{marginBottom: 14}}>Already established</div>
              <div className="col" style={{gap:14}}>
                {result.not_novel.map((n,i) => (
                  <div key={i}>
                    <div style={{fontSize: 14, color: 'var(--ink)'}}>{n.point}</div>
                    {n.paper && (
                      <div className="row" style={{marginTop:6, gap:6, fontSize: 12, color:'var(--ink-3)'}}>
                        <span>→</span>
                        <button className="btn ghost sm" onClick={()=>onOpenPaper(n.paper)} style={{padding:'2px 6px', fontSize:12}}>
                          <span style={{color:'var(--ink-2)', fontStyle:'italic'}}>{n.paper.title.slice(0, 60)}{n.paper.title.length > 60 ? '…' : ''}</span>
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Closest prior work */}
          <div className="section">
            <SectionHeader
              title="Closest prior work"
              hint="Ranked by embedding similarity then reranked by method/problem/benchmark overlap."
            />
            <div className="card">
              {result.closest.map((p,i) => (
                <div key={p.id} className="sota" onClick={() => onOpenPaper(p)}>
                  <div className="rank">{String(i+1).padStart(2, '0')}</div>
                  <div>
                    <div className="sota-title">{p.title}</div>
                    <div className="sota-claim">{(p.brief||'').split('.')[0]}.</div>
                    <div className="sota-meta">
                      <AuthorLine paper={p}/>
                      <span className="divider"/>
                      <span className="mono" style={{fontSize:11, color:'var(--accent)'}}>sim 0.{(82-i*6)}</span>
                      <span className="divider"/>
                      {(p.methods||[]).slice(0,2).map(m => <Chip key={m} mono>{m}</Chip>)}
                    </div>
                  </div>
                  <Icon name="chevron" size={14}/>
                </div>
              ))}
            </div>
          </div>

          {/* Experimental requirements */}
          <div className="grid-2" style={{marginTop: 28}}>
            <div className="card" style={{padding: 24}}>
              <div className="eyebrow" style={{marginBottom: 14}}>If you run this — required baselines</div>
              <div className="col" style={{gap: 8}}>
                {result.baselines_needed.map(b => (
                  <div key={b} className="row" style={{padding:'8px 12px', background:'var(--surface-sunk)', borderRadius:'var(--radius-sm)', gap:10}}>
                    <Icon name="target" size={14}/>
                    <span style={{fontSize:13, fontWeight:500}}>{b}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="card" style={{padding: 24}}>
              <div className="eyebrow" style={{marginBottom: 14}}>Canonical benchmarks for the target subdomain</div>
              <div className="col" style={{gap: 8}}>
                {result.benchmarks_needed.map(b => (
                  <div key={b} className="row" style={{padding:'8px 12px', background:'var(--surface-sunk)', borderRadius:'var(--radius-sm)', gap:10}}>
                    <Icon name="flask" size={14}/>
                    <span style={{fontSize:13, fontWeight:500}}>{b}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="row" style={{marginTop: 32, gap: 10, justifyContent:'flex-end'}}>
            <button className="btn ghost"><Icon name="external" size={14}/> Export PDF</button>
            <button className="btn"><Icon name="book" size={14}/> Spawn notebook with these 5</button>
            <button className="btn primary">Save as shared report</button>
          </div>
        </div>
      )}
    </div>
  );
};

window.NoveltyCheck = NoveltyCheck;
