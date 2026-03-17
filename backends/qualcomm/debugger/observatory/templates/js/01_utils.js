(function() {
  const OBS = window.__observatory;
  const state = OBS.state;

  function safeStr(val) {
    if (val === null || val === undefined) return '';
    if (typeof val === 'object') return JSON.stringify(val);
    return String(val);
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text == null ? '' : String(text);
    return div.innerHTML;
  }

  function showToast(message, type = 'success') {
    const existingToast = document.querySelector('.toast');
    if (existingToast) existingToast.remove();

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }

  function copyTable(tableEl) {
    const rows = Array.from(tableEl.querySelectorAll('tr'));
    const csv = rows
      .map((row) => {
        const cols = Array.from(row.querySelectorAll('td, th'));
        return cols
          .map((col) => `"${col.innerText.replace(/"/g, '""')}"`)
          .join(',');
      })
      .join('\n');

    const html = `<style>table, th, td { border: 1px solid black; border-collapse: collapse; padding: 4px; }</style><table>${tableEl.innerHTML}</table>`;

    try {
      const blobHTML = new Blob([html], { type: 'text/html' });
      const blobText = new Blob([csv], { type: 'text/plain' });
      const clipboardItem = new ClipboardItem({
        'text/html': blobHTML,
        'text/plain': blobText,
      });
      navigator.clipboard
        .write([clipboardItem])
        .then(() => showToast('Copied to clipboard!', 'success'))
        .catch(() => showToast('Failed to copy', 'error'));
    } catch (_e) {
      navigator.clipboard
        .writeText(csv)
        .then(() => showToast('Copied to clipboard!', 'success'))
        .catch(() => showToast('Failed to copy', 'error'));
    }
  }

  function resolveFunction(path) {
    if (!path || typeof path !== 'string') return null;
    const parts = path.split('.');
    let fn = window;
    for (const p of parts) fn = fn && fn[p];
    return typeof fn === 'function' ? fn : null;
  }

  function getLensBlocks(record, lensName) {
    const lensView = (record.views || {})[lensName];
    if (!lensView || !Array.isArray(lensView.blocks)) return [];
    return lensView.blocks.slice().sort((a, b) => Number(a.order || 0) - Number(b.order || 0));
  }

  function toArraySet(maybeSet) {
    return Array.from(maybeSet || []).sort((a, b) => a - b);
  }

  function buildViewerPayload(graphRef) {
    const assets = state.data.graph_assets || {};
    const layers = state.data.graph_layers || {};
    const asset = assets[graphRef] || {};
    return {
      base: asset.base || { legend: [], nodes: [], edges: [] },
      extensions: layers[graphRef] || {},
    };
  }

  function destroyGraphRuntime() {
    for (const compare of state.mountedCompares) {
      try {
        if (compare && typeof compare.destroy === 'function') compare.destroy();
      } catch (_e) {}
    }
    state.mountedCompares = [];

    for (const viewer of state.mountedViewers) {
      try {
        if (viewer && typeof viewer.destroy === 'function') viewer.destroy();
      } catch (_e) {}
    }
    state.mountedViewers = [];
  }

  OBS.utils = {
    safeStr,
    escapeHtml,
    showToast,
    copyTable,
    resolveFunction,
    getLensBlocks,
    toArraySet,
    buildViewerPayload,
    destroyGraphRuntime,
  };
})();
