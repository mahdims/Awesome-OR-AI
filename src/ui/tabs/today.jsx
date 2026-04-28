// Today — landing / morning briefing dispatcher

const Today = ({ onNav, onOpenPaper }) => {
  const [newMenuOpen, setNewMenuOpen] = React.useState(false);
  const [profileOpen, setProfileOpen] = React.useState(false);
  const userName = (window.ME?.name || window.ME?.email || 'there').split('@')[0];

  // Pinned topics (subdomains) for this user — server-driven, persisted via /api/me/pins.
  // Falls back to first 4 available subdomains for guests / new users.
  const allSdIds = Object.keys(window.SUBDOMAINS || {});
  const pinnedIds = (window.PINS && window.PINS.length) ? window.PINS : allSdIds.slice(0, 4);
  // last visit = 2 days ago
  const since = new Date(Date.now() - 2 * 86400000).toISOString().slice(0, 10);

  const newSinceUser = (sdId) =>
    window.papersIn(sdId).filter(p => (p.date || '') > since);

  // Pending-for-you counters
  const mustReadsSince = window.PAPERS.filter(p => p.must_read && (p.date || '') > since).slice(0, 4);
  const assigned = window.PAPERS.slice(6, 8); // synthetic "assigned by Jun"
  const queueUnread = window.PAPERS.slice(4, 10);

  // Total papers added since last visit (across whole corpus)
  const newTotal = window.PAPERS.filter(p => (p.date || '') > since).length;
  const daysAgo = 2;

  // Audio — topics with weekly digest ready
  const audioReady = pinnedIds
    .map(id => ({ id, ...window.SUBDOMAINS[id], newCount: newSinceUser(id).length }))
    .filter(x => x.newCount >= 3)
    .slice(0, 3);

  const Kpi = ({ value, label, sub, onClick, accent }) => {
    if (value === 0) {
      return (
        <div className="kpi-empty">
          <span className="mono">—</span> <span>{label}</span>
        </div>
      );
    }
    return (
      <button className={`kpi-card ${accent ? 'accent' : ''}`} onClick={onClick}>
        <div className="kpi-num">{value}</div>
        <div className="kpi-label">{label}</div>
        <div className="kpi-sub">
          {sub} <span className="kpi-arrow">→</span>
        </div>
      </button>
    );
  };

  const TopicCard = ({ sdId }) => {
    const s = window.SUBDOMAINS[sdId];
    const all = window.papersIn(sdId);
    const newly = newSinceUser(sdId);
    const mr = newly.filter(p => p.must_read).length;
    const gaps = (window.GAPS[sdId] || []).length;
    const signals = window.SIGNALS[sdId] || [];
    // Emerging signal: first "keyword"-kind signal, if any, for dense display
    const spike = signals.find(x => x.kind === 'keyword');

    return (
      <div className="topic-card" onClick={() => onNav('subdomain', sdId)}>
        <div className="topic-hd">
          <div className="topic-pin" title="Pinned">★</div>
          <div className="topic-title">{s.name}</div>
        </div>
        <div className="topic-global">
          {all.length} papers · {s.weekly} this week
        </div>
        <div className="topic-personal">
          <div className="topic-row">
            <span className={`tdot ${newly.length > 0 ? 'on' : ''}`}/>
            {newly.length > 0 ? <><b>{newly.length}</b> new since your visit</> : <span className="muted">— no new</span>}
          </div>
          <div className="topic-row">
            <span className={`tdot ${mr > 0 ? 'on accent' : ''}`}/>
            {mr > 0 ? <><b>{mr}</b> must-read</> : <span className="muted">—</span>}
          </div>
          <div className="topic-row">
            <span className={`tdot ${gaps > 0 ? 'on' : ''}`}/>
            {gaps > 0 ? <><b>{gaps}</b> open gap{gaps>1?'s':''}</> : <span className="muted">—</span>}
          </div>
        </div>
        {spike && (
          <div className="topic-spike">
            <span className="spike-arrow">▲</span>
            <div>
              <div className="spike-label">spike</div>
              <div className="spike-body">{spike.body}</div>
            </div>
          </div>
        )}
        <div className="topic-actions">
          <button className="btn sm primary" onClick={(e)=>{e.stopPropagation(); onNav('subdomain', sdId);}}>
            Open <Icon name="arrow" size={12}/>
          </button>
          <button className="btn sm ghost" onClick={(e)=>{e.stopPropagation(); onNav('notebook');}}>
            <Icon name="plus" size={11} stroke={2}/> Notebook
          </button>
        </div>
      </div>
    );
  };

  // Team feed (immutable, newest first)
  const teamFeed = [
    {
      kind: 'audio', icon: 'headphones',
      title: 'New audio: LLM Evo Search weekly roundup',
      body: '12 min · 7 papers · generated 4h ago',
      action: 'Play',
      action_nav: () => onNav('subdomain', pinnedIds[0] || allSdIds[0]),
    },
    {
      kind: 'novelty', icon: 'spark',
      who: 'Jun',
      title: 'ran a novelty check',
      body: '"RL-guided crossover in LLM evolutionary search" · 3 close precedents, 1 candidate gap',
      action: 'Open report',
      action_nav: () => onNav('novelty'),
    },
    {
      kind: 'notebook', icon: 'book',
      who: 'Elena',
      title: 'built a notebook',
      body: '"Stochastic VRP with LLM priors" · 8 papers · audio ready',
      action: 'Open notebook',
      action_nav: () => onNav('notebook'),
    },
    {
      kind: 'flag', icon: 'quote',
      who: `${userName} (you)`,
      title: 'flagged BEAM for team discussion',
      body: 'No responses yet',
      muted: true,
      action: 'Open paper',
      action_nav: () => onOpenPaper(window.PAPERS[0]),
    },
    {
      kind: 'gap', icon: 'target',
      who: 'Jun',
      title: 'marked a gap as explored',
      body: 'Self-instrumenting agents × IndustryOR · notes attached',
      action: 'View',
      action_nav: () => onNav('gaps'),
    },
  ];

  return (
    <>
      {/* Page */}
      <div className="today-wrap">
        {/* Top strip — New… + profile sit in the page, NOT in the global nav (they apply to "Today" context) */}
        <div className="today-actionbar">
          <div className="today-actionbar-left">
            <span className="eyebrow">Landing</span>
          </div>
          <div className="today-actionbar-right">
            <div className="menu-wrap">
              <button className="btn primary" onClick={() => setNewMenuOpen(!newMenuOpen)}>
                <Icon name="plus" size={12} stroke={2.5}/> New <Icon name="chevronDown" size={12} stroke={2}/>
              </button>
              {newMenuOpen && (
                <div className="menu-pop" onClick={() => setNewMenuOpen(false)}>
                  <div className="menu-item" onClick={() => onNav('novelty')}>
                    <Icon name="spark" size={14}/>
                    <div>
                      <div className="mi-t">Novelty check</div>
                      <div className="mi-s">Is this idea already published?</div>
                    </div>
                  </div>
                  <div className="menu-item" onClick={() => onNav('notebook')}>
                    <Icon name="book" size={14}/>
                    <div>
                      <div className="mi-t">Notebook</div>
                      <div className="mi-s">Curate a shareable paper set</div>
                    </div>
                  </div>
                  <div className="menu-item" onClick={() => onNav('gaps')}>
                    <Icon name="target" size={14}/>
                    <div>
                      <div className="mi-t">Gap explorer</div>
                      <div className="mi-s">Find unexplored benchmark × method combos</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div className="menu-wrap">
              <button className="profile-btn" onClick={()=>setProfileOpen(!profileOpen)}>
                <span className="avatar">A</span> {userName}
                <Icon name="chevronDown" size={11} stroke={2}/>
              </button>
              {profileOpen && (
                <div className="menu-pop" style={{right:0, left:'auto', minWidth: 220}} onClick={() => setProfileOpen(false)}>
                  <div className="menu-item"><Icon name="book" size={14}/><div><div className="mi-t">Reading history</div></div></div>
                  <div className="menu-item"><Icon name="bookmark" size={14}/><div><div className="mi-t">Pinned topics</div></div></div>
                  <div className="menu-item"><Icon name="filter" size={14}/><div><div className="mi-t">Notification prefs</div></div></div>
                  <div className="menu-item"><Icon name="close" size={14}/><div><div className="mi-t">Sign out</div></div></div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Morning briefing */}
        <div className="today-hero">
          <h1 className="today-greet">
            Good morning, <em>{userName}</em>.
          </h1>
          <div className="today-meta">
            Last visit: {daysAgo} days ago · <b>{newTotal}</b> new papers since.
          </div>

          <div className="kpi-row">
            <Kpi
              value={mustReadsSince.length}
              label="Must-reads"
              sub="since you left"
              accent
              onClick={() => onNav('queue')}
            />
            <Kpi
              value={assigned.length}
              label="Assigned by Jun"
              sub="due this week"
              onClick={() => onNav('queue')}
            />
            <Kpi
              value={queueUnread.length}
              label="In your queue"
              sub="unread"
              onClick={() => onNav('queue')}
            />
          </div>

          {audioReady.length > 0 && (
            <div className="audio-strip">
              <Icon name="headphones" size={16}/>
              <span><b>{audioReady.length}</b> topic{audioReady.length>1?'s have':' has'} fresh weekly audio ready</span>
              <span className="audio-topics">
                {audioReady.map(a => a.name).join(' · ')}
              </span>
              <button className="btn sm ghost" style={{marginLeft:'auto'}}>
                <Icon name="headphones" size={12}/> Play all
              </button>
            </div>
          )}
        </div>

        {/* Your topics */}
        <section className="today-section">
          <div className="today-section-hd">
            <div>
              <h2 className="today-h2">Your topics</h2>
              <div className="today-section-sub">Pinned — these are your navigation.</div>
            </div>
            <div className="row" style={{gap: 8}}>
              <button className="btn ghost sm">Reorder</button>
              <button className="btn ghost sm" onClick={() => onNav('subdomains')}>Browse all →</button>
            </div>
          </div>
          <div className="topics-grid">
            {pinnedIds.map(id => <TopicCard key={id} sdId={id}/>)}
            <button className="topic-card topic-pin-more" onClick={() => onNav('subdomains')}>
              <div className="tpm-plus"><Icon name="plus" size={18} stroke={2}/></div>
              <div className="tpm-label">Pin another topic</div>
              <div className="tpm-sub">{Object.keys(window.SUBDOMAINS).length - pinnedIds.length} available</div>
            </button>
          </div>
        </section>

        {/* Team feed */}
        <section className="today-section">
          <div className="today-section-hd">
            <div>
              <h2 className="today-h2">This week on the team</h2>
              <div className="today-section-sub">Immutable, link-shareable · reverse chronological</div>
            </div>
          </div>
          <div className="team-feed">
            {teamFeed.map((f, i) => (
              <div key={i} className={`feed-row ${f.muted?'muted':''}`}>
                <div className="feed-ico"><Icon name={f.icon} size={16}/></div>
                <div className="feed-body">
                  <div className="feed-title">
                    {f.who && <b className="feed-who">{f.who}</b>}
                    {f.who && ' '}
                    {f.title}
                  </div>
                  <div className="feed-sub">{f.body}</div>
                </div>
                <button className="btn ghost sm" onClick={f.action_nav}>
                  {f.action} →
                </button>
              </div>
            ))}
          </div>
        </section>

        {/* Resume footer */}
        <div className="resume-strip">
          <div className="resume-item">
            <span className="resume-label">Continue reading</span>
            <span className="resume-body">"PRIME: Training-free proactive reasoning…"</span>
            <span className="mono" style={{fontSize:11, color:'var(--accent)'}}>45% through</span>
          </div>
          <div className="resume-item">
            <span className="resume-label">Your drafts</span>
            <span className="resume-body">1 novelty check in progress</span>
            <button className="btn ghost sm" onClick={() => onNav('novelty')}>Resume →</button>
          </div>
        </div>
      </div>
    </>
  );
};

window.Today = Today;
