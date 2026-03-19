(function() {
  const OBS = window.__observatory;
  const state = OBS.state;
  const {
    safeStr,
    escapeHtml,
    copyTable,
    resolveFunction,
    getLensBlocks,
    toArraySet,
    buildViewerPayload,
    destroyGraphRuntime,
  } = OBS.utils;

  function createSection(title, storageKey, collapsible) {
    const isCollapsible = collapsible !== false;
    const isCollapsed = isCollapsible && state.viewPrefs[storageKey] === false;

    const section = document.createElement('div');
    section.className = 'toggle-section';

    const header = document.createElement('div');
    header.className = `toggle-header ${isCollapsed ? 'collapsed' : ''}`;

    const titleSpan = document.createElement('span');
    titleSpan.className = 'toggle-title';
    titleSpan.textContent = title;
    header.appendChild(titleSpan);

    const content = document.createElement('div');
    content.className = `toggle-content ${isCollapsed ? 'hidden' : ''}`;

    if (isCollapsible) {
      header.onclick = () => {
        content.classList.toggle('hidden');
        header.classList.toggle('collapsed');
        state.viewPrefs[storageKey] = content.classList.contains('hidden') ? false : true;
        localStorage.setItem('graphCollectorViewPrefs', JSON.stringify(state.viewPrefs));
      };
    }

    section.appendChild(header);
    section.appendChild(content);

    return { section, header, content };
  }

  function renderTableContent(content, data) {
    const table = document.createElement('table');
    table.className = 'kv-table';
    const tbody = document.createElement('tbody');

    const entries = Object.entries(data || {});
    if (entries.length === 0) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = 2;
      td.textContent = '(empty)';
      tr.appendChild(td);
      tbody.appendChild(tr);
    } else {
      for (const [key, val] of entries) {
        const tr = document.createElement('tr');
        const th = document.createElement('th');
        th.textContent = key;
        const td = document.createElement('td');
        td.textContent = safeStr(val);
        tr.appendChild(th);
        tr.appendChild(td);
        tbody.appendChild(tr);
      }
    }

    table.appendChild(tbody);
    content.appendChild(table);
    return table;
  }

  function resolveViewerCtor() {
    if (typeof FXGraphViewer !== 'undefined') return FXGraphViewer;
    if (window && window.FXGraphViewer) return window.FXGraphViewer;
    return null;
  }

  function resolveGraphRef(graphRecord, fallbackGraphRef) {
    if (!graphRecord || typeof graphRecord !== 'object') return fallbackGraphRef || '';
    return (
      graphRecord.graph_ref ||
      graphRecord.graphRef ||
      graphRecord.record_name ||
      graphRecord.recordName ||
      fallbackGraphRef ||
      ''
    );
  }

  function mountGraphViewer(root, graphRecord, viewerOptions, fallbackGraphRef) {
    const ViewerCtor = resolveViewerCtor();
    const graphRef = resolveGraphRef(graphRecord, fallbackGraphRef);

    if (!ViewerCtor || !graphRef) {
      const reason = !ViewerCtor ? 'FXGraphViewer unavailable' : 'graph_ref missing';
      root.innerHTML = `<div style="color:red">${reason}.</div>`;
      return null;
    }

    const payload = buildViewerPayload(graphRef);
    const defaultLayers = Array.isArray(graphRecord.default_layers) ? graphRecord.default_layers : [];
    const defaultColorBy = graphRecord.default_color_by || (defaultLayers.length > 0 ? defaultLayers[0] : 'base');

    const layoutMode = (viewerOptions || {}).layout_mode || 'full';
    let preset = 'split';
    if (layoutMode === 'compare_compact') preset = 'compact';
    if (layoutMode === 'headless') preset = 'headless';

    const viewer = ViewerCtor.create({
      payload,
      mount: { root },
      layout: { preset },
      state: {
        activeExtensions: defaultLayers,
        colorBy: defaultColorBy,
      },
    });
    viewer.init();

    if ((viewerOptions || {}).sidebar_mode === 'hidden' && typeof viewer.setLayout === 'function') {
      try {
        viewer.setLayout({ panels: { sidebar: { visible: false } } });
      } catch (_e) {}
    }
    if ((viewerOptions || {}).minimap_mode === 'off' && typeof viewer.setUIVisibility === 'function') {
      try {
        viewer.setUIVisibility({ minimapToggle: false });
      } catch (_e) {}
    }

    state.mountedViewers.push(viewer);
    return viewer;
  }

  function renderRecordBlock(container, lensName, block, context, analysis) {
    const storageKey = `${lensName}:${block.id}`;
    const title = block.title || block.id || lensName;
    const { section, header, content } = createSection(title, storageKey, block.collapsible);

    if (block.type === 'table') {
      const table = renderTableContent(content, block.record && block.record.data);
      const copyBtn = document.createElement('button');
      copyBtn.className = 'copy-btn';
      copyBtn.innerText = 'Copy';
      copyBtn.onclick = (e) => {
        e.stopPropagation();
        copyTable(table);
      };
      header.appendChild(copyBtn);
    } else if (block.type === 'html') {
      content.innerHTML = (block.record && block.record.content) || '';
    } else if (block.type === 'custom') {
      const jsFunc = block.record && block.record.js_func;
      const fn = resolveFunction(jsFunc);
      if (!fn) {
        content.innerHTML = `<div style="color:red">Function ${escapeHtml(jsFunc || '')} not found</div>`;
      } else {
        try {
          fn(content, (block.record && block.record.args) || {}, context, analysis);
        } catch (err) {
          content.innerHTML = `<div style="color:red">JS Error: ${escapeHtml(err.message || String(err))}</div>`;
        }
      }
    } else if (block.type === 'graph') {
      const graphRoot = document.createElement('div');
      graphRoot.style.height = '640px';
      graphRoot.style.minHeight = '400px';
      graphRoot.style.border = '1px solid var(--border-color)';
      graphRoot.style.borderRadius = '8px';
      graphRoot.style.overflow = 'hidden';
      content.appendChild(graphRoot);
      const fallbackGraphRef = (context && context.record && context.record.name) || '';
      mountGraphViewer(graphRoot, block.record || {}, (block.record && block.record.viewer_options) || {}, fallbackGraphRef);
    } else {
      content.innerHTML = `<div style="color:red">Unsupported block type: ${escapeHtml(block.type || '')}</div>`;
    }

    container.appendChild(section);
  }

  function renderDashboard(container) {
    container.innerHTML = '<h2>Run Dashboard</h2>';

    const dashboard = state.data.dashboard || {};
    let hasContent = false;

    for (const [lensName, viewList] of Object.entries(dashboard)) {
      const blocks = Array.isArray(viewList && viewList.blocks)
        ? viewList.blocks.slice().sort((a, b) => Number(a.order || 0) - Number(b.order || 0))
        : [];
      if (blocks.length === 0) continue;

      const analysis = state.data.analysis_results && state.data.analysis_results[lensName];
      const context = {
        start: (state.data.session && state.data.session.start_data && state.data.session.start_data[lensName]) || {},
        end: (state.data.session && state.data.session.end_data && state.data.session.end_data[lensName]) || {},
        records: state.data.records || [],
      };

      for (const block of blocks) {
        renderRecordBlock(container, lensName, block, context, analysis);
        hasContent = true;
      }
    }

    if (!hasContent) container.innerHTML += '<p>No dashboard data available.</p>';
  }

  function renderTableCompare(content, entries) {
    const allKeys = new Set();
    for (const entry of entries) {
      const data = entry.block && entry.block.record && entry.block.record.data;
      if (!data || typeof data !== 'object') continue;
      Object.keys(data).forEach((k) => allKeys.add(k));
    }

    if (allKeys.size === 0) {
      content.innerHTML = '<p>No table data to compare.</p>';
      return null;
    }

    const table = document.createElement('table');
    table.className = 'kv-table comparison-table';

    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    const th0 = document.createElement('th');
    th0.textContent = 'Property';
    headerRow.appendChild(th0);

    for (const entry of entries) {
      const th = document.createElement('th');
      const span = document.createElement('span');
      span.className = 'clickable-name';
      span.textContent = entry.record.name || `record_${entry.idx}`;
      span.onclick = (e) => {
        e.stopPropagation();
        window.selectRecord(entry.idx, true);
      };
      th.appendChild(span);
      headerRow.appendChild(th);
    }

    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    for (const key of Array.from(allKeys).sort()) {
      const tr = document.createElement('tr');
      const th = document.createElement('th');
      th.textContent = key;
      tr.appendChild(th);

      for (const entry of entries) {
        const td = document.createElement('td');
        const data = entry.block && entry.block.record && entry.block.record.data;
        td.textContent = data && data[key] !== undefined ? safeStr(data[key]) : '-';
        tr.appendChild(td);
      }

      tbody.appendChild(tr);
    }

    table.appendChild(tbody);
    content.appendChild(table);
    return table;
  }

  function renderHtmlCompare(content, entries) {
    const split = document.createElement('div');
    split.className = 'split-view';

    for (const entry of entries) {
      const pane = document.createElement('div');
      pane.className = 'split-pane';

      const h3 = document.createElement('h3');
      const span = document.createElement('span');
      span.className = 'clickable-name';
      span.textContent = entry.record.name || `record_${entry.idx}`;
      span.onclick = () => window.selectRecord(entry.idx, true);
      h3.appendChild(span);
      pane.appendChild(h3);

      const html = (entry.block && entry.block.record && entry.block.record.content) || '';
      const div = document.createElement('div');
      div.innerHTML = html;
      pane.appendChild(div);
      split.appendChild(pane);
    }

    content.appendChild(split);
  }

  function renderCustomCompare(content, entries, compareSpec, sampleBlock, lensName, blockId) {
    const recordJsFunc = sampleBlock && sampleBlock.record && sampleBlock.record.js_func;
    const jsFunc = compareSpec.js_func || recordJsFunc || '';
    const fn = resolveFunction(jsFunc);
    if (!fn) {
      content.innerHTML = `<div style="color:red">Function ${escapeHtml(jsFunc)} not found</div>`;
      return;
    }

    const context = {
      indices: entries.map((e) => e.idx),
      names: entries.map((e) => e.record.name),
      records: entries.map((e) => e.record),
      blocks: entries.map((e) => e.block),
      lens: lensName,
      block_id: blockId,
    };

    try {
      fn(content, compareSpec.args || {}, context, state.data.analysis_results && state.data.analysis_results[lensName]);
    } catch (err) {
      content.innerHTML = `<div style="color:red">JS Error: ${escapeHtml(err.message || String(err))}</div>`;
    }
  }

  function renderGraphCompare(content, entries, compareSpec) {
    const maxParallel = Math.max(1, Number(compareSpec.max_parallel || 2));
    const syncEnabledByDefault = compareSpec.sync_toggle !== false;
    const selected = entries.slice(0, maxParallel);

    const controls = document.createElement('div');
    controls.style.cssText = 'display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px;';
    const syncId = `sync_${Math.random().toString(36).slice(2)}`;
    controls.innerHTML = `
      <label style="display:flex;align-items:center;gap:6px;">
        <input id="${syncId}" type="checkbox" ${syncEnabledByDefault ? 'checked' : ''} />
        Sync Selection
      </label>
    `;
    content.appendChild(controls);

    const split = document.createElement('div');
    split.className = 'split-view';
    content.appendChild(split);

    const viewers = [];
    for (const entry of selected) {
      const pane = document.createElement('div');
      pane.className = 'split-pane';

      const h3 = document.createElement('h3');
      const span = document.createElement('span');
      span.className = 'clickable-name';
      span.textContent = entry.record.name || `record_${entry.idx}`;
      span.onclick = () => window.selectRecord(entry.idx, true);
      h3.appendChild(span);
      pane.appendChild(h3);

      const root = document.createElement('div');
      root.style.height = '520px';
      root.style.minHeight = '360px';
      root.style.border = '1px solid var(--border-color)';
      root.style.borderRadius = '8px';
      root.style.overflow = 'hidden';
      pane.appendChild(root);

      const options = Object.assign({}, (entry.block && entry.block.record && entry.block.record.viewer_options) || {}, compareSpec.viewer_options_compare || {});
      const fallbackGraphRef = (entry.record && entry.record.name) || '';
      const viewer = mountGraphViewer(root, entry.block.record || {}, options, fallbackGraphRef);
      if (viewer) viewers.push(viewer);

      split.appendChild(pane);
    }

    const hasCompareCtor = typeof FXGraphCompare !== 'undefined' || !!(window && window.FXGraphCompare);
    if (viewers.length > 1 && hasCompareCtor) {
      const CompareCtor = typeof FXGraphCompare !== 'undefined' ? FXGraphCompare : window.FXGraphCompare;
      const syncConfig = {
        selection: syncEnabledByDefault,
        camera: false,
        theme: false,
        layers: false,
      };
      const compare = CompareCtor.create({
        viewers,
        layout: { columns: Math.min(maxParallel, viewers.length), compact: true },
        sync: syncConfig,
      });
      state.mountedCompares.push(compare);

      const syncInput = document.getElementById(syncId);
      if (syncInput) {
        syncInput.addEventListener('change', () => {
          const enabled = syncInput.checked;
          if (typeof compare.setSync === 'function') {
            compare.setSync({ selection: enabled, camera: false, theme: false, layers: false });
          }
        });
      }
    }
  }

  function defaultCompareMode(block) {
    if (!block) return 'disabled';
    if (block.type === 'table' || block.type === 'html' || block.type === 'graph') return 'auto';
    return 'disabled';
  }

  function renderCompareLens(recordView, lensName, indices) {
    const records = state.data.records || [];
    const entriesByBlock = new Map();

    for (const idx of indices) {
      const record = records[idx];
      if (!record) continue;
      const blocks = getLensBlocks(record, lensName);
      for (const block of blocks) {
        const id = block.id || `${lensName}_${block.type}`;
        if (!entriesByBlock.has(id)) entriesByBlock.set(id, []);
        entriesByBlock.get(id).push({ idx, record, block });
      }
    }

    const blockEntries = Array.from(entriesByBlock.values())
      .filter((list) => list.length > 0)
      .sort((a, b) => Number((a[0].block && a[0].block.order) || 0) - Number((b[0].block && b[0].block.order) || 0));

    for (const entries of blockEntries) {
      if (entries.length < 2) continue;

      const sample = entries[0].block;
      const compareSpec = sample.compare || {};
      const mode = compareSpec.mode || defaultCompareMode(sample);
      if (mode === 'disabled') continue;

      const blockLabel = sample.title || sample.id || sample.type;
      const sectionKey = `cmp:${lensName}:${sample.id}`;
      const { section, header, content } = createSection(`Comparison: ${lensName} / ${blockLabel}`, sectionKey, sample.collapsible);

      if (sample.type === 'table' && mode === 'auto') {
        const table = renderTableCompare(content, entries);
        if (table) {
          const copyBtn = document.createElement('button');
          copyBtn.className = 'copy-btn';
          copyBtn.innerText = 'Copy';
          copyBtn.onclick = (e) => {
            e.stopPropagation();
            copyTable(table);
          };
          header.appendChild(copyBtn);
        }
      } else if (sample.type === 'html' && mode === 'auto') {
        renderHtmlCompare(content, entries);
      } else if (sample.type === 'graph' && mode === 'auto') {
        renderGraphCompare(content, entries, compareSpec);
      } else if (mode === 'custom') {
        renderCustomCompare(content, entries, compareSpec, sample, lensName, sample.id);
      } else {
        content.innerHTML = `<p>Compare mode '${escapeHtml(mode)}' for block type '${escapeHtml(sample.type)}' is not supported in minimal runtime.</p>`;
      }

      recordView.appendChild(section);
    }
  }

  function renderUnifiedView(container, indices) {
    const records = (state.data.records || []).filter((_, idx) => indices.includes(idx));
    const isSingle = indices.length === 1;
    const title = isSingle
      ? records[0].name
      : `Comparison (${indices.map((i) => (state.data.records || [])[i].name).join(' vs ')})`;

    container.innerHTML = `<div class="record-view"><h2>${escapeHtml(title)}</h2></div>`;
    const recordView = container.querySelector('.record-view');

    if (isSingle) {
      const idx = indices[0];
      const record = (state.data.records || [])[idx];
      let hasContent = false;

      for (const lensName of Object.keys((record && record.views) || {})) {
        const blocks = getLensBlocks(record, lensName);
        const analysis = state.data.analysis_results && state.data.analysis_results[lensName];
        const context = { index: idx, record };
        for (const block of blocks) {
          renderRecordBlock(recordView, lensName, block, context, analysis);
          hasContent = true;
        }
      }

      if (!hasContent) recordView.innerHTML += '<p>No views available for this record.</p>';
      return;
    }

    const allLenses = new Set();
    for (const idx of indices) {
      const record = (state.data.records || [])[idx];
      for (const lensName of Object.keys((record && record.views) || {})) {
        allLenses.add(lensName);
      }
    }

    for (const lensName of allLenses) {
      renderCompareLens(recordView, lensName, indices);
    }
  }

  function renderMain() {
    destroyGraphRuntime();

    const container = document.getElementById('main-pane');
    if (!container) return;
    container.innerHTML = '';

    if (state.selectionMode) {
      const indices = toArraySet(state.selectedIndices);
      if (indices.length === 0) {
        container.innerHTML = '<div class="loading">Select items to compare...</div>';
      } else {
        renderUnifiedView(container, indices);
      }
      return;
    }

    if (state.activeRecordIndex === -1) {
      renderDashboard(container);
      return;
    }

    if (typeof state.activeRecordIndex === 'object' && state.activeRecordIndex !== null) {
      const indices = state.activeRecordIndex.pool
        ? state.activeRecordIndex.pool
        : [state.activeRecordIndex.base, state.activeRecordIndex.new].filter((x) => Number.isInteger(x));
      renderUnifiedView(container, indices);
      return;
    }

    renderUnifiedView(container, [state.activeRecordIndex]);
  }

  OBS.render = {
    renderMain,
    renderDashboard,
    renderUnifiedView,
    mountGraphViewer,
  };
})();
