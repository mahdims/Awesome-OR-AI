// Feed — cross-subdomain chronological

const Feed = ({ onOpenPaper }) => {
  const [cat, setCat] = React.useState('all');
  const [sig, setSig] = React.useState('all');
  const [q, setQ] = React.useState('');

  let papers = [...window.PAPERS].sort((a,b)=>(b.date||'').localeCompare(a.date||''));
  if (cat !== 'all') papers = papers.filter(p => p.category === cat);
  if (sig === 'must') papers = papers.filter(p => p.must_read);
  if (sig === 'changes') papers = papers.filter(p => p.changes_thinking);
  if (q) {
    const qq = q.toLowerCase();
    papers = papers.filter(p => (p.title + ' ' + p.brief).toLowerCase().includes(qq));
  }

  return (
    <div className="container" style={{paddingTop: 40, paddingBottom: 96}}>
      <div style={{maxWidth: 820, marginBottom: 28}}>
        <div className="eyebrow" style={{marginBottom: 14}}>Cross-subdomain feed · chronological</div>
        <h1 className="h-display">The <em>whole corpus</em>, reverse-chronological.</h1>
      </div>

      <div className="card" style={{padding:'14px 18px', marginBottom: 24, display:'grid', gridTemplateColumns:'1fr auto', gap:16, alignItems:'center'}}>
        <div className="search-wrap">
          <Icon name="search" size={16}/>
          <input className="input" placeholder="Search titles and briefs…" value={q} onChange={e=>setQ(e.target.value)}/>
        </div>
        <div className="row gap-12">
          <div className="seg">
            {[['all','All'],['must','Must-read'],['changes','Changes thinking']].map(([id,l]) => (
              <button key={id} className={sig===id?'on':''} onClick={()=>setSig(id)}>{l}</button>
            ))}
          </div>
        </div>
      </div>

      <div className="filters" style={{border:'none', padding:'0 0 16px 0'}}>
        <div className="row gap-12">
          <span className="mono" style={{fontSize:11, color:'var(--ink-3)', textTransform:'uppercase', letterSpacing:'0.08em'}}>Category</span>
          {['all','LLMs for Algorithm Design','Generative AI for OR','OR for Generative AI'].map(c => (
            <button key={c} className={`filter-pill ${cat===c?'on':''}`} onClick={()=>setCat(c)}>{c === 'all' ? 'All' : c}</button>
          ))}
        </div>
        <div style={{marginLeft:'auto', fontSize:12, color:'var(--ink-3)'}}>
          Showing <b style={{color:'var(--ink)'}}>{papers.length}</b> papers
        </div>
      </div>

      <div className="card">
        {papers.slice(0, 20).map(p => <PaperRow key={p.id} paper={p} onOpen={onOpenPaper}/>)}
      </div>
    </div>
  );
};

window.Feed = Feed;
