/* api_client.js — replaces data.jsx.
 *
 * Loaded as a regular <script> (not type="text/babel") so it runs before
 * the JSX scripts. Fetches /api/init once, populates window.PAPERS /
 * SUBDOMAINS / GAPS / SIGNALS / NAV / ME, and exposes the same helper
 * functions the rest of the UI reads off `window.*`.
 *
 * App.jsx awaits `window.__ri_init_promise` before calling ReactDOM.render
 * so the React tree always sees populated data.
 */
(() => {
  'use strict';

  // Static UI config (used to live in data.jsx).
  window.NAV = [
    { id: 'today',       label: 'Today' },
    { id: 'subdomains',  label: 'Topics' },
    { id: 'subdomain',   label: 'Topic Detail' },
    { id: 'feed',        label: 'Feed' },
    { id: 'novelty',     label: 'Novelty Check' },
    { id: 'notebook',    label: 'Notebook Builder' },
    { id: 'gaps',        label: 'Gap Explorer' },
    { id: 'queue',       label: 'My Queue' },
  ];

  // Map API paper shape -> UI's expected shape (mock used `id`/`date`/`priority`).
  function paperFromApi(p) {
    return Object.assign({}, p, {
      id: p.arxiv_id,
      date: p.published_date,
      priority: p.priority_score,
      // `subdomain` (singular) is set when M3's per-paper assignment lands.
      // For M1b every paper.subdomain is undefined → papersIn() returns [].
    });
  }

  // Helper functions previously baked into data.jsx. Behavior is identical;
  // they all read off window.PAPERS so they keep working after the cutover.
  window.papersIn      = (sd) => (window.PAPERS || []).filter(p => p.subdomain === sd);
  window.getPaper      = (id) => (window.PAPERS || []).find(p => p.id === id);
  window.relPercent    = (v)  => Math.min(100, Math.max(0, (v / 10) * 100));
  window.sotaFor       = (sd) => window.papersIn(sd).sort((a, b) => (b.priority || 0) - (a.priority || 0)).slice(0, 4);

  window.benchmarksFor = (sd) => {
    const counts = {};
    window.papersIn(sd).forEach(p => {
      (p.benchmarks || []).forEach(b => {
        if (!counts[b]) counts[b] = { name: b, count: 0, latestDate: '', latestPaper: null };
        counts[b].count++;
        if (p.date > counts[b].latestDate) {
          counts[b].latestDate = p.date;
          counts[b].latestPaper = p.id;
        }
      });
    });
    return Object.values(counts).sort((a, b) => b.count - a.count);
  };

  window.baselinesFor = (sd) => {
    const counts = {};
    window.papersIn(sd).forEach(p => {
      (p.baselines || []).forEach(b => {
        const key = b.replace(/\s*\(.+\)/, '').trim();
        if (!counts[key]) counts[key] = { name: key, count: 0 };
        counts[key].count++;
      });
    });
    return Object.values(counts).sort((a, b) => b.count - a.count);
  };

  window.labsFor = (sd) => {
    const counts = {};
    window.papersIn(sd).forEach(p => {
      (p.affiliations || '').split(',').map(s => s.trim()).filter(Boolean).forEach(a => {
        counts[a] = (counts[a] || 0) + 1;
      });
    });
    return Object.entries(counts).map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count);
  };

  // ---- write helpers (used by paper drawer, follow/pin, prefs) -----------

  async function jsonFetch(path, opts = {}) {
    const headers = Object.assign({ 'Content-Type': 'application/json' }, opts.headers || {});
    const res = await fetch(path, Object.assign({}, opts, { headers, credentials: 'same-origin' }));
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`${res.status} ${res.statusText}: ${body}`);
    }
    return res.status === 204 ? null : res.json();
  }

  window.api = {
    setPaperState: (arxivId, status, notes) =>
      jsonFetch(`/api/me/papers/${encodeURIComponent(arxivId)}`, {
        method: 'PATCH',
        body: JSON.stringify({ status, notes }),
      }),
    follow:   (sdId) => jsonFetch(`/api/me/follows/${encodeURIComponent(sdId)}`, { method: 'POST' }),
    unfollow: (sdId) => jsonFetch(`/api/me/follows/${encodeURIComponent(sdId)}`, { method: 'DELETE' }),
    pin:      (sdId) => jsonFetch(`/api/me/pins/${encodeURIComponent(sdId)}`,    { method: 'POST' }),
    unpin:    (sdId) => jsonFetch(`/api/me/pins/${encodeURIComponent(sdId)}`,    { method: 'DELETE' }),
    setPrefs: (patch) => jsonFetch('/api/me/prefs', { method: 'PATCH', body: JSON.stringify(patch) }),
    getPaper: (arxivId) => jsonFetch(`/api/papers/${encodeURIComponent(arxivId)}`),
    logout:   () => jsonFetch('/auth/logout', { method: 'POST' }),
  };

  // ---- bootstrap ---------------------------------------------------------

  window.__ri_init_promise = (async () => {
    try {
      const data = await fetch('/api/init', { credentials: 'same-origin' }).then(r => r.json());

      window.PAPERS = (data.papers || []).map(paperFromApi);

      // Mock had SUBDOMAINS as an object keyed by id; API returns an array.
      window.SUBDOMAINS = {};
      (data.subdomains || []).forEach(sd => {
        window.SUBDOMAINS[sd.id] = {
          name: sd.name,
          tagline: sd.tagline,
          category: sd.category,
          weekly: sd.paper_count || 0,
        };
      });

      window.GAPS = data.gaps && Object.keys(data.gaps).length ? data.gaps : {};
      window.SIGNALS = data.signals && Object.keys(data.signals).length ? data.signals : {};
      window.ME = data.me || null;
      window.QUEUE = data.queue || [];
      window.FOLLOWS = (data.follows && data.follows.ids) || [];
      window.PINS = (data.pins && data.pins.ids) || [];
      window.PREFS = data.prefs || { density: 'balanced', theme: 'default' };
      window.CATEGORIES = data.categories || [];

      // Sensible default for tabs that need a "current subdomain":
      // pinned > followed > first available > null.
      const sdIds = Object.keys(window.SUBDOMAINS);
      window.DEFAULT_SDID =
        window.PINS[0] || window.FOLLOWS[0] || sdIds[0] || null;
    } catch (err) {
      console.error('[api_client] /api/init failed; falling back to empty state', err);
      // Don't break the React render — give the app empty collections so it
      // can show a clear "couldn't load" message instead of stack-tracing.
      window.PAPERS = [];
      window.SUBDOMAINS = {};
      window.GAPS = {};
      window.SIGNALS = {};
      window.ME = null;
      window.QUEUE = [];
      window.FOLLOWS = [];
      window.PINS = [];
      window.PREFS = { density: 'balanced', theme: 'default' };
      window.CATEGORIES = [];
      window.DEFAULT_SDID = null;
      window.__ri_init_error = err;
    }
  })();
})();
