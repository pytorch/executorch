(function() {
  const OBS = window.__observatory;
  const state = OBS.state;
  const { escapeHtml } = OBS.utils;

  // Restore pinned state from localStorage
  let sidebarPinned = false;
  try { sidebarPinned = localStorage.getItem('obs_sidebar_pinned') === '1'; } catch(e) {}

  function renderLayout() {
    const icon = state.theme === 'dark' ? '☀️' : '🌙';
    OBS.app.innerHTML = `
      <header>
        <div class="header-content">
          <h1>${escapeHtml(state.data.title || 'Observatory Report')}</h1>
          <div class="header-meta">
            <span>Generated: ${escapeHtml(state.data.generated_at || '')}</span>
          </div>
        </div>
        <div>
          <button class="theme-toggle" onclick="toggleTheme()" title="Toggle dark mode">
            <span class="theme-icon">${icon}</span>
          </button>
        </div>
      </header>
      <div class="container">
        <div class="index-pane-trigger"></div>
        <button class="sidebar-toggle-btn ${sidebarPinned ? 'open' : ''}"
                id="sidebar-toggle-btn"
                onclick="toggleSidebar()"
                title="${sidebarPinned ? 'Close panel' : 'Open panel'}">
          ${sidebarPinned ? '&#8249;' : '&#8250;'}
        </button>
        <nav class="index-pane ${sidebarPinned ? 'pinned' : ''}" id="index-pane">
          <div class="index-header" id="index-header">
            <h2>Collected Graphs (${(state.data.records || []).length})</h2>
          </div>
          <ul id="index-list" class="index-list"></ul>
        </nav>
        <main id="main-pane" class="main-pane"></main>
      </div>
    `;
    updateIndexHeader();
  }

  window.toggleSidebar = function() {
    sidebarPinned = !sidebarPinned;
    try { localStorage.setItem('obs_sidebar_pinned', sidebarPinned ? '1' : '0'); } catch(e) {}

    const pane = document.getElementById('index-pane');
    const btn  = document.getElementById('sidebar-toggle-btn');
    if (pane) pane.classList.toggle('pinned', sidebarPinned);
    if (btn)  {
      btn.classList.toggle('open', sidebarPinned);
      btn.innerHTML = sidebarPinned ? '&#8249;' : '&#8250;';
      btn.title = sidebarPinned ? 'Close panel' : 'Open panel';
    }
  };

  function updateIndexHeader() {
    const header = document.getElementById('index-header');
    if (!header) return;

    if (state.selectionMode) {
      const total = (state.data.records || []).length;
      const allSelected = state.selectedIndices.size === total;
      header.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <span style="font-size:0.9rem;font-weight:600;">Selected: ${state.selectedIndices.size}</span>
          <div>
            <button class="btn-sm" onclick="toggleSelectAll()">${allSelected ? 'Unselect All' : 'Select All'}</button>
            <button class="btn-sm" onclick="toggleSelectionMode()">Cancel</button>
          </div>
        </div>
      `;
      return;
    }

    const treeMode = OBS.layout && OBS.layout.isTreeView();
    header.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <h2>Collected Graphs (${(state.data.records || []).length})</h2>
        <div>
          <button class="btn-sm" onclick="toggleTreeView()" title="Toggle region tree grouping">${treeMode ? '☰ Flat' : '🌳 Tree'}</button>
          <button class="btn-sm" onclick="toggleSelectionMode()">Select Items</button>
        </div>
      </div>
    `;
  }

  // -- Tree-view state ------------------------------------------------------
  // Records carry an optional `region_stack: List[str]` snapshotted at
  // collect() time (RFC §4.5). When tree mode is on, the left panel
  // groups records by that path into collapsible region nodes. Toggling
  // is purely visual — the underlying record order and indices are
  // preserved, so all click handlers (selectRecord, compare etc.) keep
  // working unchanged.
  let treeView = false;
  try { treeView = localStorage.getItem('obs_tree_view') === '1'; } catch(e) {}

  // Per-region collapsed state, keyed by the joined region path (e.g.
  // "session/edge/etrecord"). Defaults to expanded.
  const collapsedRegions = new Set();
  try {
    const stored = localStorage.getItem('obs_collapsed_regions');
    if (stored) {
      JSON.parse(stored).forEach((k) => collapsedRegions.add(k));
    }
  } catch(e) {}

  function persistCollapsed() {
    try {
      localStorage.setItem(
        'obs_collapsed_regions',
        JSON.stringify(Array.from(collapsedRegions))
      );
    } catch(e) {}
  }

  function toggleTreeView() {
    treeView = !treeView;
    try { localStorage.setItem('obs_tree_view', treeView ? '1' : '0'); } catch(e) {}
    renderIndex();
  }

  function toggleRegionCollapse(key) {
    if (collapsedRegions.has(key)) collapsedRegions.delete(key);
    else collapsedRegions.add(key);
    persistCollapsed();
    renderIndex();
  }

  // Expose for inline onclick handlers in the rendered HTML.
  window.toggleTreeView = toggleTreeView;
  window.toggleRegionCollapse = toggleRegionCollapse;

  // Build the HTML for one record row. Used by both flat and tree-view
  // rendering paths so click semantics stay identical (selectRecord,
  // selection-mode checkboxes, badges, diff-separators all work the
  // same). The optional `treeMode` flag suppresses the diff_index
  // separator (2-way compare summary) which only makes sense between
  // adjacent records in flat order — in tree view two consecutive
  // records can sit in different regions, so the diff is misleading.
  function renderRecordItem(rec, idx, treeMode) {
    const isSelected = state.selectedIndices.has(idx);
    const checkbox = state.selectionMode
      ? `<input type="checkbox" class="selection-checkbox" ${isSelected ? 'checked' : ''} onclick="toggleSelect(${idx}, event)" style="margin-right:0.5rem;">`
      : '';

    let activeClass = '';
    if (state.selectionMode) {
      if (isSelected) activeClass = 'selected';
    } else if (typeof state.activeRecordIndex === 'object' && state.activeRecordIndex !== null) {
      if (state.activeRecordIndex.pool && state.activeRecordIndex.pool.includes(idx)) activeClass = 'selected';
      if (state.activeRecordIndex.base === idx) activeClass = 'diff-base';
      if (state.activeRecordIndex.new === idx) activeClass = 'diff-new';
    } else if (state.activeRecordIndex === idx) {
      activeClass = 'active';
    }

    const badges = (rec.badges || [])
      .map((b) => {
        const badgeClass = b.class || 'badge';
        const title = escapeHtml(b.title || b.label || '');
        const label = escapeHtml(b.label || '');
        return `<span class="badge ${badgeClass}" title="${title}">${label}</span>`;
      })
      .join('');

    let html = '';
    if (
      !treeMode &&
      rec.diff_index &&
      Object.keys(rec.diff_index).length > 0 &&
      idx > 0
    ) {
      const rows = Object.entries(rec.diff_index)
        .map(([key, val]) => {
          const text = String(val);
          const plusMatch  = text.match(/\+((?:\d+\.?\d*)|(?:\.\d+))/);
          const minusMatch = text.match(/-((?:\d+\.?\d*)|(?:\.\d+))/);
          let stats = '';
          if (plusMatch || minusMatch) {
            if (plusMatch) stats += `<span class="stat-add">+${plusMatch[1]}</span>`;
            if (minusMatch) stats += `<span class="stat-rem">-${minusMatch[1]}</span>`;
          } else {
            stats = `<span>${escapeHtml(text)}</span>`;
          }
          return `
            <div class="diff-row">
              <span class="diff-label">${escapeHtml(key)}</span>
              <span class="diff-stats">${stats}</span>
            </div>
          `;
        })
        .join('');

      html += `
        <li class="diff-separator" onclick="showCompareView(${idx - 1}, ${idx})">
          <div class="diff-content">${rows}</div>
        </li>
      `;
    }

    html += `
      <li class="index-item ${activeClass}" onclick="selectRecord(${idx})">
        <div style="display:flex;align-items:center;overflow:hidden;flex:1;">
          ${checkbox}
          <div style="flex:1;min-width:0;">
            <div style="text-overflow:ellipsis;overflow:hidden;white-space:nowrap;">${escapeHtml(rec.name || '')}</div>
          </div>
        </div>
        <div class="badges">${badges}</div>
      </li>
    `;
    return html;
  }

  function _isActiveSession(sessionId) {
    const a = state.activeRecordIndex;
    if (a && typeof a === 'object' && a.sessionDashboard === sessionId) return true;
    // Sentinel: -1 means "first session" (the default landing).
    if (a === -1) {
      const first = (state.data.sessions || [])[0];
      return Boolean(first && first.id === sessionId);
    }
    return false;
  }

  function _renderSessionDashboardLink(session) {
    const active = _isActiveSession(session.id) ? 'active' : '';
    const label = session.name || session.id || '(unnamed session)';
    return `
      <li class="index-item session-dashboard-link ${active}"
          onclick="selectSession('${escapeHtml(session.id)}')">
        <span>📊 Session Dashboard: ${escapeHtml(label)}</span>
      </li>
    `;
  }

  function renderIndexFlat(records, sessionFilter) {
    let html = '';
    const filtered = sessionFilter
      ? records.map((r, i) => ({rec: r, idx: i})).filter(x => x.rec.session_id === sessionFilter)
      : records.map((r, i) => ({rec: r, idx: i}));
    filtered.forEach(({rec, idx}) => {
      html += renderRecordItem(rec, idx, false);
    });
    return html;
  }

  // Tree-view: group records by their `region_stack`. Records keep their
  // original time ordering — only the visual structure changes. Each
  // region becomes a collapsible header bearing the region name. The
  // diff_index "2-way compare" separator that flat view shows between
  // adjacent records is suppressed in tree view (renderRecordItem
  // receives treeMode=true) since adjacent records in the records[]
  // array may not be visually adjacent under the tree.
  function renderIndexTree(records, sessionFilter) {
    let html = '';

    // Walk records preserving order. Track open region path; when the
    // next record's region_stack differs, close the differing tail and
    // open the new tail. This produces correctly-nested <ul> blocks.
    const openPath = [];

    function closeTo(depth) {
      while (openPath.length > depth) {
        openPath.pop();
        html += `</ul></li>`;
      }
    }

    function openRegion(name, fullKey) {
      const collapsed = collapsedRegions.has(fullKey);
      const caret = collapsed ? '▶' : '▼';
      html += `
        <li class="region-header" onclick="toggleRegionCollapse('${escapeHtml(fullKey)}')">
          <span class="region-caret">${caret}</span>
          <span class="region-name">${escapeHtml(name)}</span>
        </li>
        <ul class="region-children" style="${collapsed ? 'display:none;' : ''}">
      `;
    }

    records.forEach((rec, idx) => {
      if (sessionFilter && rec.session_id !== sessionFilter) return;
      const stack = Array.isArray(rec.region_stack) ? rec.region_stack : [];

      // Compute longest matching prefix between openPath and stack.
      let common = 0;
      while (
        common < openPath.length &&
        common < stack.length &&
        openPath[common] === stack[common]
      ) {
        common++;
      }
      // Close anything below the common prefix.
      closeTo(common);
      // Open the new tail.
      for (let i = common; i < stack.length; i++) {
        openPath.push(stack[i]);
        // Scope collapsed-region keys by session so two sessions with
        // identical region names keep independent open/closed state.
        const fullKey = (sessionFilter ? sessionFilter + '::' : '') + openPath.join('/');
        openRegion(stack[i], fullKey);
      }

      html += renderRecordItem(rec, idx, true);
    });

    closeTo(0);
    return html;
  }

  function _renderArchiveColumn(archive, sessions, records, useTree) {
    const sessionsInArchive = sessions.filter((s) => s.archive === archive.label);
    let inner = '';
    sessionsInArchive.forEach((session) => {
      inner += _renderSessionDashboardLink(session);
      inner += useTree
        ? renderIndexTree(records, session.id)
        : renderIndexFlat(records, session.id);
    });
    return `
      <li class="archive-column">
        <div class="archive-column-header" title="${escapeHtml(archive.label)}">${escapeHtml(archive.label)}</div>
        <ul class="archive-sessions">${inner}</ul>
      </li>
    `;
  }

  function renderIndex() {
    const list = document.getElementById('index-list');
    if (!list) return;

    const records = state.data.records || [];
    const sessions = state.data.sessions || [];
    const archives = state.data.archives || [];
    const useTree = treeView && records.some((r) => Array.isArray(r.region_stack) && r.region_stack.length > 0);

    if (sessions.length === 0) {
      // Defensive fallback: payloads without a sessions list (very old
      // archives) render as a single ungrouped list.
      list.classList.remove('compare-mode');
      list.style.removeProperty('--archive-cols');
      list.innerHTML = useTree ? renderIndexTree(records) : renderIndexFlat(records);
      updateIndexHeader();
      return;
    }

    // Compare mode: more than one archive => N-column grid, one per archive.
    if (archives.length > 1) {
      list.classList.add('compare-mode');
      list.style.setProperty('--archive-cols', String(archives.length));
      let html = '';
      archives.forEach((archive) => {
        html += _renderArchiveColumn(archive, sessions, records, useTree);
      });
      list.innerHTML = html;
      updateIndexHeader();
      return;
    }

    // Single-archive: flat list of session dashboards + records.
    list.classList.remove('compare-mode');
    list.style.removeProperty('--archive-cols');
    let html = '';
    sessions.forEach((session) => {
      html += _renderSessionDashboardLink(session);
      html += useTree ? renderIndexTree(records, session.id) : renderIndexFlat(records, session.id);
    });
    list.innerHTML = html;
    updateIndexHeader();
  }

  OBS.layout = {
    renderLayout,
    renderIndex,
    updateIndexHeader,
    isTreeView: () => treeView,
  };
})();
