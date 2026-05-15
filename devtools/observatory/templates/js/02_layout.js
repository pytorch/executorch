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

    header.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <h2>Collected Graphs (${(state.data.records || []).length})</h2>
        <button class="btn-sm" onclick="toggleSelectionMode()">Select Items</button>
      </div>
    `;
  }

  function renderIndex() {
    const list = document.getElementById('index-list');
    if (!list) return;

    const records = state.data.records || [];
    let html = `
      <li class="index-item ${state.activeRecordIndex === -1 ? 'active' : ''}" onclick="selectRecord(-1)">
        <span>📊 Run Dashboard</span>
      </li>
    `;

    records.forEach((rec, idx) => {
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

      if (rec.diff_index && Object.keys(rec.diff_index).length > 0 && idx > 0) {
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
    });

    list.innerHTML = html;
    updateIndexHeader();
  }

  OBS.layout = {
    renderLayout,
    renderIndex,
    updateIndexHeader,
  };
})();
