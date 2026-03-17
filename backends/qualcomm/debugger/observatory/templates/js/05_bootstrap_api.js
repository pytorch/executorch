(function() {
  const OBS = window.__observatory;
  const state = OBS.state;
  const { renderLayout, renderIndex } = OBS.layout;
  const { renderMain, mountGraphViewer } = OBS.render;
  const { showToast } = OBS.utils;
  const actions = OBS.actions;

  function wrapGraphHandle(viewer) {
    return {
      setLayers(layerIds) {
        if (viewer && typeof viewer.setLayers === 'function') viewer.setLayers(layerIds || []);
      },
      setColorBy(layerId) {
        if (viewer && typeof viewer.setColorBy === 'function') viewer.setColorBy(layerId);
      },
      updateLayerNodeStyle(layerId, nodeId, patch) {
        if (!viewer || typeof viewer.patchLayerNodes !== 'function') return;
        const payload = {};
        payload[nodeId] = patch || {};
        viewer.patchLayerNodes(layerId, payload);
      },
      selectNode(nodeId, opts) {
        if (viewer && typeof viewer.selectNode === 'function') viewer.selectNode(nodeId, opts || {});
      },
      zoomToFit() {
        if (viewer && typeof viewer.zoomToFit === 'function') viewer.zoomToFit();
      },
      setSyncEnabled(enabled) {
        if (viewer && typeof viewer.setState === 'function') {
          try {
            viewer.setState({ syncSelection: !!enabled });
          } catch (_e) {}
        }
      },
      enterFullscreen() {
        if (viewer && typeof viewer.enterFullscreen === 'function') viewer.enterFullscreen();
      },
      exitFullscreen() {
        if (viewer && typeof viewer.exitFullscreen === 'function') viewer.exitFullscreen();
      },
      onNodeSelected(callback) {
        if (viewer && typeof viewer.on === 'function') {
          viewer.on('selectionchange', callback);
        }
      },
      destroy() {
        if (viewer && typeof viewer.destroy === 'function') viewer.destroy();
      },
      _viewer: viewer,
    };
  }

  window.ObservatoryAPI = {
    mountGraph(container, graphRef, options) {
      let root = container;
      if (typeof container === 'string') root = document.querySelector(container);
      if (!root) throw new Error('mountGraph: container not found');

      const graphRecord = {
        graph_ref: graphRef,
        default_layers: (options && options.default_layers) || [],
        default_color_by: options && options.default_color_by,
        viewer_options: (options && options.viewer_options) || {},
      };

      const viewer = mountGraphViewer(root, graphRecord, graphRecord.viewer_options);
      if (!viewer) throw new Error('mountGraph: failed to mount viewer');
      return wrapGraphHandle(viewer);
    },

    selectRecord(index) {
      actions.selectRecord(index, true);
    },

    openCompare(indices) {
      if (!Array.isArray(indices) || indices.length === 0) return;
      state.activeRecordIndex = { pool: indices.slice() };
      renderIndex();
      renderMain();
    },

    showSingleRecord(index) {
      actions.selectRecord(index, true);
    },

    showToast(message, type) {
      showToast(message, type || 'success');
    },

    getContext() {
      return {
        activeRecordIndex: state.activeRecordIndex,
        selectionMode: state.selectionMode,
        selectedIndices: Array.from(state.selectedIndices),
        records: state.data.records || [],
      };
    },
  };

  function setupDelegatedActions() {
    document.body.addEventListener('click', (event) => {
      const target = event.target && event.target.closest && event.target.closest('[data-ob-action]');
      if (!target) return;

      const action = target.getAttribute('data-ob-action');
      if (action === 'select-record') {
        const rec = Number(target.getAttribute('data-ob-record'));
        if (Number.isInteger(rec)) actions.selectRecord(rec, true);
        return;
      }

      if (action === 'open-compare') {
        const raw = target.getAttribute('data-ob-indices') || '';
        const indices = raw
          .split(',')
          .map((v) => Number(v.trim()))
          .filter((v) => Number.isInteger(v));
        if (indices.length > 0) window.ObservatoryAPI.openCompare(indices);
        return;
      }

      if (action === 'graph-focus-node') {
        const nodeId = target.getAttribute('data-ob-node-id');
        if (!nodeId) return;
        const firstViewer = state.mountedViewers && state.mountedViewers[0];
        if (firstViewer && typeof firstViewer.selectNode === 'function') {
          firstViewer.selectNode(nodeId, { center: true, animate: true });
        }
      }
    });
  }

  function init() {
    actions.setTheme(state.theme);
    renderLayout();
    renderIndex();
    renderMain();
    setupDelegatedActions();
  }

  init();
})();
