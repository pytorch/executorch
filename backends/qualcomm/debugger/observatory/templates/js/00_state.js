(function() {
  const OBS = (window.__observatory = window.__observatory || {});

  OBS.state = {
    data: window.OBSERVATORY_DATA || {},
    activeRecordIndex: -1,
    theme: localStorage.getItem('graphCollectorTheme') || 'light',
    viewPrefs: JSON.parse(localStorage.getItem('graphCollectorViewPrefs') || '{}'),
    selectionMode: false,
    selectedIndices: new Set(),
    mountedViewers: [],
    mountedCompares: [],
    viewerCache: new Map(),
    compareStateCache: new Map(),
  };

  OBS.app = document.getElementById('app');
})();
