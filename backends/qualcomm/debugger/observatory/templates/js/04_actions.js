(function() {
  const OBS = window.__observatory;
  const state = OBS.state;
  const { renderIndex, updateIndexHeader } = OBS.layout;
  const { renderMain } = OBS.render;

  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    state.theme = theme;
    localStorage.setItem('graphCollectorTheme', theme);
    const icon = document.querySelector('.theme-icon');
    if (icon) icon.textContent = theme === 'dark' ? '☀️' : '🌙';

    // Sync theme to all mounted fx_viewer instances
    for (const viewer of (state.mountedViewers || [])) {
      try {
        if (typeof viewer.setTheme === 'function') viewer.setTheme(theme);
      } catch (_) {}
    }
    for (const compare of (state.mountedCompares || [])) {
      try {
        if (typeof compare.setSync === 'function') compare.setSync({ theme: false });
        for (const v of (compare.viewers || [])) {
          if (typeof v.setTheme === 'function') v.setTheme(theme);
        }
      } catch (_) {}
    }
  }

  function toggleTheme() {
    setTheme(state.theme === 'light' ? 'dark' : 'light');
  }

  function showCompareView() {
    const indices = Array.from(arguments).filter((n) => Number.isInteger(n));
    state.activeRecordIndex = { pool: indices };
    renderIndex();
    renderMain();
  }

  function selectRecord(index, forceNavigate) {
    if (forceNavigate && state.selectionMode) {
      state.selectionMode = false;
      state.selectedIndices.clear();
    }

    if (state.selectionMode && index !== -1) {
      toggleSelect(index);
      return;
    }

    state.activeRecordIndex = index;
    renderIndex();
    renderMain();
  }

  function toggleSelectionMode() {
    state.selectionMode = !state.selectionMode;
    if (state.selectionMode) {
      state.selectedIndices.clear();
      if (typeof state.activeRecordIndex === 'number' && state.activeRecordIndex !== -1) {
        state.selectedIndices.add(state.activeRecordIndex);
      } else if (
        typeof state.activeRecordIndex === 'object' &&
        state.activeRecordIndex &&
        Array.isArray(state.activeRecordIndex.pool)
      ) {
        for (const idx of state.activeRecordIndex.pool) state.selectedIndices.add(idx);
      }
    } else {
      state.selectedIndices.clear();
    }

    updateIndexHeader();
    renderIndex();
    renderMain();
  }

  function toggleSelectAll() {
    const records = state.data.records || [];
    if (state.selectedIndices.size === records.length) {
      state.selectedIndices.clear();
    } else {
      state.selectedIndices = new Set(records.map((_, i) => i));
    }
    renderIndex();
    renderMain();
  }

  function toggleSelect(idx, event) {
    if (event) event.stopPropagation();
    if (state.selectedIndices.has(idx)) {
      state.selectedIndices.delete(idx);
    } else {
      state.selectedIndices.add(idx);
    }
    renderIndex();
    renderMain();
  }

  OBS.actions = {
    setTheme,
    toggleTheme,
    showCompareView,
    selectRecord,
    toggleSelectionMode,
    toggleSelectAll,
    toggleSelect,
  };

  window.setTheme = setTheme;
  window.toggleTheme = toggleTheme;
  window.showCompareView = showCompareView;
  window.selectRecord = selectRecord;
  window.toggleSelectionMode = toggleSelectionMode;
  window.toggleSelectAll = toggleSelectAll;
  window.toggleSelect = toggleSelect;
})();
