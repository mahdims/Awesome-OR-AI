// My Queue — personal reading state + team artifacts

const MyQueue = ({ onOpenPaper }) => {
  const unread = window.PAPERS.filter(p=>p.must_read).slice(0, 4);
  const reading = window.PAPERS.slice(4, 6);
  const read = window.PAPERS.slice(6, 9);

  const Col = ({ title, count, papers, tone }) => (
    <div>
      <div className="between" style={{marginBottom: 14, alignItems:'baseline'}}>
        <div className="row" style={{gap: 10}}>
          <span className="eyebrow">{title}</span>
          <span className="mono" style={{fontSize:11, color:'var(--ink-3)'}}>{count}</span>
        </div>
      </div>
      <div className="col" style={{gap: 10}}>
        {papers.map(p => (
          <div key={p.id} className="q-card" style={{cursor:'pointer'}} onClick={()=>onOpenPaper(p)}>
            <div style={{fontSize: 14, fontWeight: 600, marginBottom: 4, lineHeight: 1.35}}>{p.title}</div>
            <div style={{fontSize: 12, color:'var(--ink-3)'}}>
              <AuthorLine paper={p} compact={true}/>
            </div>
            <div className="row" style={{marginTop: 10, gap: 6}}>
              {p.must_read && <Chip kind="accent" mono>must-read</Chip>}
              <Chip mono>{window.SUBDOMAINS[p.subdomain]?.name.split(' ').slice(0,2).join(' ') || 'other'}</Chip>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="container" style={{paddingTop: 40, paddingBottom: 96}}>
      <div style={{maxWidth: 820, marginBottom: 28}}>
        <div className="row" style={{gap: 10, marginBottom: 14, alignItems:'center'}}>
          <div style={{width:32, height:32, borderRadius:'50%', background:'var(--accent)', color:'white', display:'flex', alignItems:'center', justifyContent:'center', fontSize:13, fontWeight:600, fontFamily:'var(--mono)'}}>M</div>
          <div className="eyebrow">Mahdi's queue</div>
        </div>
        <h1 className="h-display">Your <em>reading</em> state.</h1>
      </div>

      <div style={{display:'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 24, marginBottom: 48}}>
        <Col title="Unread" count={unread.length} papers={unread}/>
        <Col title="Reading" count={reading.length} papers={reading}/>
        <Col title="Done this week" count={read.length} papers={read}/>
      </div>

      {/* Shared artifacts */}
      <div className="section">
        <SectionHeader title="Shared team artifacts" hint="Everything you or a teammate saved. Attribution always visible."/>
        <div className="card">
          {[
            { kind: 'notebook', title: 'AlgoEvo reading list', who: 'Mahdi', when: '3d ago', n: '12 papers', icon: 'book' },
            { kind: 'report', title: 'Self-instrumenting deep-dive · novelty report', who: 'Team', when: '1w ago', n: 'score 62', icon: 'spark' },
            { kind: 'audio', title: 'Weekly: NL→Opt · Apr 21', who: 'Auto', when: '1d ago', n: '18 min', icon: 'headphones' },
            { kind: 'notebook', title: 'OR serving / GPU scheduling — Q2 survey', who: 'Collaborator A.', when: '2w ago', n: '9 papers', icon: 'book' },
            { kind: 'report', title: 'EVOM vs ORLM comparison', who: 'Mahdi', when: '3w ago', n: 'draft', icon: 'quote' },
          ].map((a, i) => (
            <div key={i} className="paper-row" style={{gridTemplateColumns:'40px 1fr auto auto'}}>
              <div style={{display:'flex', alignItems:'center', justifyContent:'center', width:40, height:40, borderRadius:8, background:'var(--surface-sunk)', color:'var(--ink-2)'}}>
                <Icon name={a.icon} size={16}/>
              </div>
              <div className="grow">
                <div style={{fontSize:14, fontWeight:600, lineHeight:1.35}}>{a.title}</div>
                <div style={{fontSize: 12, color:'var(--ink-3)', marginTop: 4}}>
                  {a.who} · {a.when} · <span className="mono">{a.n}</span>
                </div>
              </div>
              <Chip mono>{a.kind}</Chip>
              <Icon name="chevron" size={14}/>
            </div>
          ))}
        </div>
      </div>

      {/* Followed subdomains */}
      <div className="section">
        <SectionHeader title="Followed subdomains" hint="Get audio & digest for these."/>
        <div style={{display:'grid', gridTemplateColumns:'repeat(3, 1fr)', gap: 16}}>
          {['evo_llm_search', 'nl_to_opt', 'llm_serving_opt'].map(id => {
            const s = window.SUBDOMAINS[id];
            return (
              <div key={id} className="q-card">
                <div className="between" style={{marginBottom: 10}}>
                  <div className="eyebrow" style={{fontSize: 10}}>{s.category}</div>
                  <button className="btn ghost sm"><Icon name="bookmark" size={12}/> Following</button>
                </div>
                <div className="serif" style={{fontSize: 17, letterSpacing:'-0.005em', lineHeight:1.25}}>{s.name}</div>
                <div className="row" style={{marginTop: 12, gap: 18, fontSize: 12, color:'var(--ink-3)'}}>
                  <span><b className="mono" style={{color:'var(--ink)'}}>{s.weekly}</b> new · 7d</span>
                  <span><b className="mono" style={{color:'var(--ink)'}}>{(window.GAPS[id]||[]).length}</b> gaps</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

window.MyQueue = MyQueue;
