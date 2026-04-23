// Subdomains home grid

const SubdomainsHome = ({ onNav }) => {
  const sds = Object.entries(window.SUBDOMAINS);
  const total = window.PAPERS.length;
  const mustRead = window.PAPERS.filter(p => p.must_read).length;

  return (
    <div className="container" style={{paddingTop: 48, paddingBottom: 96}}>
      <div style={{maxWidth: 880, marginBottom: 48}}>
        <div className="eyebrow" style={{marginBottom: 14}}>Topics · curated taxonomy · {Object.keys(window.SUBDOMAINS).length} topics</div>
        <h1 className="h-display">
          Pick a <em>topic</em>, see the details.
        </h1>
        <p style={{marginTop: 18, fontSize: 17, color: 'var(--ink-2)', maxWidth: 680, lineHeight: 1.55}}>
          {total} papers across three categories, hand-assigned into topics by an LLM classifier
          with human review. Each card opens a working page: what's SOTA, which benchmarks and
          baselines everyone uses, and where the obvious gaps are.
        </p>
      </div>

      {/* KPI strip */}
      <div style={{display:'grid', gridTemplateColumns:'repeat(5, 1fr)', gap: 0, border:'1px solid var(--hair)', borderRadius:'var(--radius-lg)', overflow:'hidden', background:'var(--surface)', marginBottom: 48}}>
        {[
          {l:'Papers in corpus', v: total, d:'last 7 days: 2 new'},
          {l:'Must-reads', v: mustRead, d: 'flagged by Reader'},
          {l:'Topics', v: Object.keys(window.SUBDOMAINS).length, d:'curated, version-controlled'},
          {l:'Open gaps', v: Object.values(window.GAPS).flat().length, d:'across all topics'},
          {l:'Low-confidence queue', v: 7, d:'needs review'},
        ].map((k,i) => (
          <div key={i} style={{padding:'22px 24px', borderRight: i<4 ? '1px solid var(--hair)':'none'}}>
            <div style={{fontSize:11, color:'var(--ink-3)', textTransform:'uppercase', letterSpacing:'0.09em', fontWeight:600, marginBottom: 10}}>{k.l}</div>
            <div style={{fontFamily:'var(--mono)', fontSize:28, fontWeight:500, letterSpacing:'-0.02em'}}>{k.v}</div>
            <div style={{fontSize:12, color:'var(--ink-3)', marginTop:4}}>{k.d}</div>
          </div>
        ))}
      </div>

      {/* Subdomain cards, grouped by category */}
      {['LLMs for Algorithm Design', 'Generative AI for OR', 'OR for Generative AI'].map(cat => {
        const filtered = sds.filter(([,s]) => s.category === cat);
        if (!filtered.length) return null;
        return (
          <div key={cat} style={{marginBottom: 48}}>
            <div className="between" style={{marginBottom: 18, alignItems:'baseline'}}>
              <h3 className="serif" style={{fontSize: 26, letterSpacing:'-0.01em'}}>{cat}</h3>
              <div className="mono" style={{fontSize:12, color:'var(--ink-3)'}}>
                {filtered.reduce((a,[,s])=>a, 0) + window.PAPERS.filter(p=>p.category===cat).length} papers · {filtered.length} topics
              </div>
            </div>
            <div style={{display:'grid', gridTemplateColumns:'repeat(3, 1fr)', gap: 18}}>
              {filtered.map(([id, s], idx) => {
                const papers = window.papersIn(id);
                const mr = papers.filter(p => p.must_read).length;
                const gaps = (window.GAPS[id] || []).length;
                const isSpotlight = idx === 0 && cat === 'Generative AI for OR';
                return (
                  <div key={id} className={`sd-card ${isSpotlight?'spotlight':''}`} onClick={() => onNav('subdomain', id)}>
                    <div>
                      <div className="cat">{s.category}</div>
                      <div className="sd-title" style={{marginTop: 8}}>{s.name}</div>
                      <div className="sd-body" style={{marginTop: 12}}>{s.tagline}</div>
                    </div>
                    <div className="sd-stats">
                      <div className="sd-stat"><div className="v">{papers.length}</div><div className="l">Papers</div></div>
                      <div className="sd-stat"><div className="v">{s.weekly}</div><div className="l">7-day</div></div>
                      <div className="sd-stat"><div className="v">{mr}</div><div className="l">Must-read</div></div>
                      <div className="sd-stat"><div className="v">{gaps}</div><div className="l">Gaps</div></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}

      {/* Low-confidence queue banner */}
      <div className="card" style={{padding:'28px 32px', display:'grid', gridTemplateColumns:'1fr auto', gap: 24, alignItems:'center', marginTop: 32}}>
        <div>
          <div className="eyebrow" style={{color:'var(--warn)', marginBottom: 6}}>Needs your input</div>
          <div className="serif" style={{fontSize: 22, letterSpacing:'-0.01em'}}>7 papers below 0.7 classifier confidence</div>
          <div style={{fontSize: 14, color:'var(--ink-2)', marginTop: 8, maxWidth: 640}}>
            Five minutes of triage a week keeps the taxonomy trustworthy. Overrides outrank the classifier and persist forever.
          </div>
        </div>
        <button className="btn primary">Open review queue <Icon name="arrow" size={14}/></button>
      </div>
    </div>
  );
};

window.SubdomainsHome = SubdomainsHome;
