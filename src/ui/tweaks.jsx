// Tweaks panel

const DEFAULTS = /*EDITMODE-BEGIN*/{
  "density": "balanced",
  "theme": "default"
}/*EDITMODE-END*/;

const TweaksPanel = () => {
  const [open, setOpen] = React.useState(false);
  const [editMode, setEditMode] = React.useState(false);
  const [state, setState] = React.useState(DEFAULTS);

  React.useEffect(() => {
    document.documentElement.setAttribute('data-density', state.density);
    document.documentElement.setAttribute('data-theme', state.theme);
  }, [state]);

  React.useEffect(() => {
    const onMsg = (e) => {
      if (!e.data || !e.data.type) return;
      if (e.data.type === '__activate_edit_mode') { setEditMode(true); setOpen(true); }
      if (e.data.type === '__deactivate_edit_mode') { setEditMode(false); setOpen(false); }
    };
    window.addEventListener('message', onMsg);
    window.parent.postMessage({type:'__edit_mode_available'}, '*');
    return () => window.removeEventListener('message', onMsg);
  }, []);

  const setKey = (k, v) => {
    const next = { ...state, [k]: v };
    setState(next);
    window.parent.postMessage({type:'__edit_mode_set_keys', edits: {[k]: v}}, '*');
  };

  if (!editMode && !open) return null;

  return (
    <>
      <button className="tweaks-btn" onClick={()=>setOpen(!open)}>
        <Icon name="spark" size={14}/> Tweaks
      </button>
      {open && (
        <div className="tweaks-panel">
          <h5>Density</h5>
          <div className="seg" style={{marginBottom: 16}}>
            {[['compact','Compact'],['balanced','Balanced'],['airy','Airy']].map(([id,l]) => (
              <button key={id} className={state.density===id?'on':''} onClick={()=>setKey('density',id)}>{l}</button>
            ))}
          </div>
          <h5>Color theme</h5>
          <div className="seg">
            {[['default','Blue'],['neutral','Neutral'],['warm','Warm']].map(([id,l]) => (
              <button key={id} className={state.theme===id?'on':''} onClick={()=>setKey('theme',id)}>{l}</button>
            ))}
          </div>
        </div>
      )}
    </>
  );
};

window.TweaksPanel = TweaksPanel;
