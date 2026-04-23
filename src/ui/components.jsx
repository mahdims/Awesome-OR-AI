// Shared components

const Icon = ({ name, size = 16, stroke = 1.5 }) => {
  const paths = {
    search: <><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></>,
    filter: <><path d="M3 5h18M6 12h12M10 19h4"/></>,
    chevron: <><path d="M9 6l6 6-6 6"/></>,
    chevronDown: <><path d="M6 9l6 6 6-6"/></>,
    close: <><path d="M18 6L6 18M6 6l12 12"/></>,
    external: <><path d="M14 3h7v7M21 3L10 14M15 21H5a2 2 0 01-2-2V9"/></>,
    book: <><path d="M4 4a2 2 0 012-2h14v20H6a2 2 0 01-2-2zM4 19h16"/></>,
    arrow: <><path d="M5 12h14M13 6l6 6-6 6"/></>,
    plus: <><path d="M12 5v14M5 12h14"/></>,
    check: <><path d="M5 12l5 5L20 7"/></>,
    spark: <><path d="M12 2l2.4 7.4H22l-6.2 4.5L18.2 22 12 17.3 5.8 22l2.4-8.1L2 9.4h7.6z"/></>,
    dot: <><circle cx="12" cy="12" r="3"/></>,
    bookmark: <><path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/></>,
    flask: <><path d="M9 2h6M10 2v7L4 20a2 2 0 002 2h12a2 2 0 002-2L14 9V2"/></>,
    mic: <><rect x="9" y="3" width="6" height="12" rx="3"/><path d="M5 11a7 7 0 0014 0M12 18v3"/></>,
    headphones: <><path d="M3 12a9 9 0 0118 0v7a2 2 0 01-2 2h-2v-8h4M3 19a2 2 0 002 2h2v-8H3z"/></>,
    quote: <><path d="M7 7h4v4H7zM7 11c0 3-1 5-3 5M17 7h4v4h-4zM17 11c0 3-1 5-3 5"/></>,
    grid: <><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></>,
    list: <><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></>,
    target: <><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/></>,
    waves: <><path d="M2 12c2-2 4-2 6 0s4 2 6 0 4-2 6 0M2 7c2-2 4-2 6 0s4 2 6 0 4-2 6 0M2 17c2-2 4-2 6 0s4 2 6 0 4-2 6 0"/></>,
    alert: <><path d="M12 9v4M12 17h.01"/><circle cx="12" cy="12" r="9"/></>,
    trend: <><path d="M3 17l6-6 4 4 8-8M14 7h7v7"/></>,
    star: <><path d="M12 3l2.5 6.5L21 10l-5 4.5L17.5 21 12 17.5 6.5 21 8 14.5 3 10l6.5-.5z"/></>,
  };
  const p = paths[name] || paths.dot;
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round"
      style={{ flexShrink: 0, display: 'inline-block', verticalAlign: '-3px' }}>
      {p}
    </svg>
  );
};

const Chip = ({ children, kind = '', className = '', mono = false, ...rest }) => (
  <span className={`chip ${kind} ${mono?'mono-chip':''} ${className}`} {...rest}>{children}</span>
);

// Priority score pill
const PriorityBadge = ({ score }) => {
  const s = Math.round((score||0) * 10) / 10;
  return <span className="mono" style={{
    fontSize: 11, fontWeight: 600,
    color: s >= 8 ? 'var(--accent)' : s >= 6 ? 'var(--ink-2)' : 'var(--ink-3)',
  }}>p{s.toFixed(1)}</span>;
};

// Small circular score ring
const ScoreRing = ({ value, max = 10, label }) => {
  const pct = Math.min(1, value/max);
  const R = 18, C = 2*Math.PI*R;
  return (
    <div className="score" title={`${label}: ${value}/${max}`}>
      <svg width="44" height="44">
        <circle className="bg" cx="22" cy="22" r={R} fill="none" strokeWidth="3"/>
        <circle className="fg" cx="22" cy="22" r={R} fill="none" strokeWidth="3"
          strokeDasharray={C} strokeDashoffset={C*(1-pct)} strokeLinecap="round"/>
      </svg>
      <span>{value}</span>
    </div>
  );
};

const ConfidenceDot = ({ level = 'hi' }) => {
  const n = level === 'hi' ? 3 : level === 'lo' ? 1 : 2;
  return (
    <span className={`conf-dot ${level}`} title={`Reader confidence: ${level === 'hi' ? 'high' : 'low'}`}>
      {[0,1,2].map(i => <i key={i} className={i < n ? 'on' : ''}/>)}
    </span>
  );
};

// Short-form authors line ("Alexander et al. · Google DeepMind · 2025-06")
const AuthorLine = ({ paper, compact = false }) => {
  const authorStr = Array.isArray(paper.authors) && paper.authors.length > 0
    ? paper.authors[0] + (paper.authors.length > 1 || !paper.authors[0].endsWith('.') ? ' et al.' : '')
    : 'Anonymous';
  const aff = (paper.affiliations||'').split(',')[0].trim();
  return (
    <span>{authorStr}{!compact && aff ? ` · ${aff}` : ''} · <span className="mono" style={{fontSize:'0.9em'}}>{paper.date}</span></span>
  );
};

// ArXiv id chip
const ArxivChip = ({ id }) => (
  <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>
    arXiv:{id}
  </span>
);

// Significance markers
const SigMarkers = ({ paper }) => (
  <>
    {paper.must_read && <Chip kind="accent" mono>must-read</Chip>}
    {paper.changes_thinking && <Chip kind="pos" mono>changes thinking</Chip>}
    {paper.team_discussion && <Chip kind="ghost" mono>discuss</Chip>}
  </>
);

// Section header with serif title
const SectionHeader = ({ title, hint, right }) => (
  <div className="section-hd">
    <div>
      <h3>{title}</h3>
    </div>
    <div className="row gap-16">
      {hint && <div className="hint">{hint}</div>}
      {right}
    </div>
  </div>
);

// Paper row used in feeds
const PaperRow = ({ paper, onOpen }) => {
  return (
    <div className="paper-row" onClick={() => onOpen(paper)}>
      <div className="col" style={{gap:6, alignItems:'flex-start'}}>
        <ScoreRing value={Math.round(paper.priority || 0)} label="priority"/>
      </div>
      <div className="grow">
        <div className="title">{paper.title}</div>
        <div className="meta">
          <AuthorLine paper={paper}/>
          {paper.framework_lineage && <><span className="divider"/><span className="mono">lineage:{paper.framework_lineage}</span></>}
          {paper.new_benchmark && <><span className="divider"/><Chip kind="ghost" mono>new benchmark</Chip></>}
          {paper.code_url && <><span className="divider"/><span className="mono" style={{color:'var(--pos)'}}>code</span></>}
        </div>
        {paper.brief && <div style={{
          fontSize: 'var(--fs-sm)', color: 'var(--ink-2)',
          marginTop: 8, lineHeight: 1.5,
          display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden'
        }}>{paper.brief.split('.')[0]}.</div>}
        <div className="tag-list" style={{marginTop: 10}}>
          <SigMarkers paper={paper}/>
          {(paper.methods||[]).slice(0,3).map(m => <Chip key={m} mono>{m}</Chip>)}
        </div>
      </div>
      <div className="col" style={{alignItems:'flex-end', gap: 8, minWidth: 90}}>
        <ConfidenceDot level={paper.confidence_results}/>
        <Icon name="chevron" size={14}/>
      </div>
    </div>
  );
};

window.Icon = Icon;
window.Chip = Chip;
window.PriorityBadge = PriorityBadge;
window.ScoreRing = ScoreRing;
window.ConfidenceDot = ConfidenceDot;
window.AuthorLine = AuthorLine;
window.ArxivChip = ArxivChip;
window.SigMarkers = SigMarkers;
window.SectionHeader = SectionHeader;
window.PaperRow = PaperRow;
