"""Testcase catalog for the unified fx_viewer API harness.

This file intentionally orders cases from simple to advanced so the harness doubles
as a tutorial.
"""

from __future__ import annotations

from typing import Any


def build_testcases(*, include_qualcomm: bool) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = [
        {
            "id": "js_01_create_init_destroy",
            "title": "JS 01: Create / Init / Destroy",
            "description": "Smallest viewer lifecycle example.",
            "html": """
<div style="display:grid;grid-template-rows:auto 1fr;gap:10px;height:100%;">
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
    <button id="c1_create">Create + Init</button>
    <button id="c1_destroy">Destroy</button>
    <span style="font-size:12px;color:#6b7280;">Target APIs: create, init, destroy, getState</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 320px;gap:10px;min-height:0;">
    <div id="c1_view" style="border:1px solid #d1d5db;border-radius:8px;overflow:hidden;"></div>
    <pre id="c1_state" style="margin:0;border:1px solid #d1d5db;border-radius:8px;padding:8px;overflow:auto;font-size:12px;"></pre>
  </div>
</div>
""".strip(),
            "js": """
let viewer = null;

function renderState() {
  const stateEl = document.getElementById('c1_state');
  if (!viewer) {
    stateEl.textContent = 'viewer = null';
    return;
  }
  const s = viewer.getState();
  stateEl.textContent = JSON.stringify({
    theme: s.theme,
    colorBy: s.colorBy,
    selectedNodeId: s.selectedNodeId,
    activeExtensions: s.activeExtensions,
  }, null, 2);
}

function createViewer() {
  if (viewer) viewer.destroy();
  viewer = FXGraphViewer.create({
    payload: api.payloads.structural,
    mount: { root: '#c1_view' },
    layout: { preset: 'split' },
  });
  viewer.init();
  renderState();
  api.log('Created + initialized viewer');
}

function destroyViewer() {
  if (!viewer) return;
  viewer.destroy();
  viewer = null;
  renderState();
  api.log('Destroyed viewer');
}

document.getElementById('c1_create').addEventListener('click', createViewer);
document.getElementById('c1_destroy').addEventListener('click', destroyViewer);

createViewer();
api.setCleanup(() => destroyViewer());
""".strip(),
        },
        {
            "id": "js_02_state_theme",
            "title": "JS 02: State + Theme",
            "description": "Learn getState/setState/setTheme with visible state snapshot.",
            "html": """
<div style="display:grid;grid-template-rows:auto 1fr;gap:10px;height:100%;">
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
    <label>Theme
      <select id="c2_theme"><option value="light">light</option><option value="dark">dark</option></select>
    </label>
    <button id="c2_toggle_highlight">Toggle highlightAncestors</button>
    <span style="font-size:12px;color:#6b7280;">Target APIs: getState, setState, setTheme</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 340px;gap:10px;min-height:0;">
    <div id="c2_view" style="border:1px solid #d1d5db;border-radius:8px;overflow:hidden;"></div>
    <pre id="c2_state" style="margin:0;border:1px solid #d1d5db;border-radius:8px;padding:8px;overflow:auto;font-size:12px;"></pre>
  </div>
</div>
""".strip(),
            "js": """
const viewer = FXGraphViewer.create({
  payload: api.payloads.structural,
  mount: { root: '#c2_view' },
  layout: { preset: 'split' },
  state: { theme: 'light' },
});
viewer.init();
api.registerViewer(viewer);

const stateEl = document.getElementById('c2_state');
const themeSel = document.getElementById('c2_theme');

themeSel.addEventListener('change', () => {
  viewer.setTheme(themeSel.value);
  renderState();
});

document.getElementById('c2_toggle_highlight').addEventListener('click', () => {
  const s = viewer.getState();
  viewer.setState({ highlightAncestors: !s.highlightAncestors });
  renderState();
});

function renderState() {
  const s = viewer.getState();
  stateEl.textContent = JSON.stringify({
    theme: s.theme,
    highlightAncestors: s.highlightAncestors,
    colorBy: s.colorBy,
    activeExtensions: s.activeExtensions,
    camera: s.camera,
  }, null, 2);
}

renderState();
api.log('Use theme dropdown and highlight toggle, then inspect getState output.');
""".strip(),
        },
        {
            "id": "js_03_selection_camera",
            "title": "JS 03: Selection + Camera",
            "description": "Control navigation APIs explicitly from custom host buttons.",
            "html": """
<div style="display:grid;grid-template-rows:auto 1fr;gap:10px;height:100%;">
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
    <button id="c3_select_first">Select First</button>
    <button id="c3_select_mid">Animate To Middle</button>
    <button id="c3_pan_last">Pan To Last</button>
    <button id="c3_zoom_fit">Zoom To Fit</button>
    <button id="c3_clear">Clear Selection</button>
    <span style="font-size:12px;color:#6b7280;">Target APIs: selectNode, animateToNode, panToNode, zoomToFit, clearSelection</span>
  </div>
  <div id="c3_view" style="border:1px solid #d1d5db;border-radius:8px;overflow:hidden;"></div>
</div>
""".strip(),
            "js": """
const viewer = FXGraphViewer.create({
  payload: api.payloads.structural,
  mount: { root: '#c3_view' },
  layout: { preset: 'split' },
});
viewer.init();
api.registerViewer(viewer);

const ids = viewer.store.baseData.nodes.map((n) => n.id);
const firstId = ids[0];
const midId = ids[Math.floor(ids.length / 2)];
const lastId = ids[ids.length - 1];

document.getElementById('c3_select_first').addEventListener('click', () => {
  viewer.selectNode(firstId, { center: true });
});

document.getElementById('c3_select_mid').addEventListener('click', () => {
  viewer.selectNode(midId, { animate: true, center: true });
});

document.getElementById('c3_pan_last').addEventListener('click', () => {
  viewer.panToNode(lastId);
});

document.getElementById('c3_zoom_fit').addEventListener('click', () => viewer.zoomToFit());
document.getElementById('c3_clear').addEventListener('click', () => viewer.clearSelection());

api.log('Use buttons to see how camera + selection APIs differ.');
""".strip(),
        },
        {
            "id": "js_04_layers_colorby",
            "title": "JS 04: Layers + ColorBy",
            "description": "Learn extension activation and color source switching.",
            "html": """
<div style="display:grid;grid-template-columns:280px 1fr;gap:10px;height:100%;">
  <div style="border:1px solid #d1d5db;border-radius:8px;padding:10px;overflow:auto;">
    <div style="font-weight:600;margin-bottom:8px;">Layer Controls</div>
    <label style="display:block;margin-bottom:8px;"><input id="c4_layer_type" type="checkbox" checked /> color_by_type</label>
    <label style="display:block;margin-bottom:8px;"><input id="c4_layer_topo" type="checkbox" checked /> topological_order</label>
    <div style="height:8px;"></div>
    <label>Color By
      <select id="c4_colorby">
        <option value="base">base</option>
        <option value="color_by_type">color_by_type</option>
        <option value="topological_order">topological_order</option>
      </select>
    </label>
  </div>
  <div id="c4_view" style="border:1px solid #d1d5db;border-radius:8px;overflow:hidden;"></div>
</div>
""".strip(),
            "js": """
const viewer = FXGraphViewer.create({
  payload: api.payloads.structural,
  mount: { root: '#c4_view' },
  layout: { preset: 'split' },
  state: { activeExtensions: ['color_by_type', 'topological_order'], colorBy: 'topological_order' },
});
viewer.init();
api.registerViewer(viewer);

const layerType = document.getElementById('c4_layer_type');
const layerTopo = document.getElementById('c4_layer_topo');
const colorBy = document.getElementById('c4_colorby');

function applyLayers() {
  const layers = [];
  if (layerType.checked) layers.push('color_by_type');
  if (layerTopo.checked) layers.push('topological_order');
  viewer.setLayers(layers);
}

layerType.addEventListener('change', applyLayers);
layerTopo.addEventListener('change', applyLayers);
colorBy.addEventListener('change', () => viewer.setColorBy(colorBy.value));

api.log('Toggle layers and colorBy to observe legend/canvas updates.');
""".strip(),
        },
        {
            "id": "js_05_runtime_mutation",
            "title": "JS 05: Runtime Layer Mutation",
            "description": "Create, patch, recolor, relabel, and remove a dynamic layer at runtime.",
            "html": """
<div style="display:grid;grid-template-columns:300px 1fr;gap:10px;height:100%;">
  <div style="border:1px solid #d1d5db;border-radius:8px;padding:10px;overflow:auto;">
    <div style="font-weight:600;margin-bottom:8px;">Runtime Mutation Controls</div>
    <label>Threshold <input id="c5_threshold" type="range" min="0" max="100" value="50" style="width:100%;" /></label>
    <div id="c5_threshold_value" style="margin-top:6px;font-family:monospace;"></div>
    <div style="height:8px;"></div>
    <button id="c5_rename">Rename Layer</button>
    <button id="c5_rule" style="margin-left:6px;">Apply Color Rule Fn</button>
    <button id="c5_remove" style="margin-left:6px;">Remove Layer</button>
  </div>
  <div id="c5_view" style="border:1px solid #d1d5db;border-radius:8px;overflow:hidden;"></div>
</div>
""".strip(),
            "js": """
const layerId = 'runtime_score';
const viewer = FXGraphViewer.create({
  payload: api.payloads.structural,
  mount: { root: '#c5_view' },
  layout: { preset: 'split' },
  state: { activeExtensions: ['color_by_type', 'topological_order'], colorBy: 'base' },
});
viewer.init();
api.registerViewer(viewer);

const allNodes = viewer.store.baseData.nodes.slice(0, 140);
const runtimeNodes = {};
allNodes.forEach((n, idx) => {
  runtimeNodes[n.id] = {
    info: { runtime_score: idx / Math.max(1, allNodes.length - 1) },
    label_append: [`r=${(idx / Math.max(1, allNodes.length - 1)).toFixed(2)}`],
    fill_color: '#93c5fd',
  };
});

viewer.upsertLayer(layerId, {
  name: 'Runtime Score',
  legend: [
    { label: 'low', color: '#93c5fd' },
    { label: 'high', color: '#b91c1c' },
  ],
  nodes: runtimeNodes,
});
viewer.setLayers(['color_by_type', 'topological_order', layerId]);
viewer.setColorBy(layerId);

const slider = document.getElementById('c5_threshold');
const valueEl = document.getElementById('c5_threshold_value');

function applyThreshold() {
  const t = Number(slider.value) / 100;
  valueEl.textContent = `threshold=${t.toFixed(2)}`;
  const patch = {};
  Object.entries(runtimeNodes).forEach(([nodeId, nodeData]) => {
    const score = Number((nodeData.info && nodeData.info.runtime_score) || 0);
    patch[nodeId] = {
      fill_color: score >= t ? '#b91c1c' : '#93c5fd',
      label_append: [`r=${score.toFixed(2)}`],
    };
  });
  viewer.patchLayerNodes(layerId, patch);
  viewer.setColorBy(layerId);
}

slider.addEventListener('input', applyThreshold);
applyThreshold();

document.getElementById('c5_rename').addEventListener('click', () => {
  viewer.setLayerLabel(layerId, 'Runtime Score (renamed)');
});

document.getElementById('c5_rule').addEventListener('click', () => {
  viewer.setColorRule(layerId, (nodeData) => {
    const s = Number((nodeData.info && nodeData.info.runtime_score) || 0);
    return s > 0.7 ? '#14532d' : '#fef08a';
  });
  viewer.setColorBy(layerId);
});

document.getElementById('c5_remove').addEventListener('click', () => {
  viewer.removeLayer(layerId);
  viewer.setColorBy('base');
});

api.log('This case targets runtime mutation APIs end-to-end.');
""".strip(),
        },
        {
            "id": "js_06_layout_slots",
            "title": "JS 06: Layout + External Slots",
            "description": "Mount viewer pieces into external host divs and control layout/UI visibility.",
            "html": """
<div style="display:grid;grid-template-rows:auto 1fr;gap:10px;height:100%;">
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
    <button id="c6_toggle_info">Toggle Info Panel</button>
    <button id="c6_toggle_minimap">Toggle Minimap</button>
    <button id="c6_toggle_chrome">Toggle Toolbar Chrome</button>
    <span style="font-size:12px;color:#6b7280;">Target APIs: mount.slots, setLayout, setUIVisibility</span>
  </div>
  <div id="c6_hidden_root" style="display:none;"></div>
  <div style="display:grid;grid-template-columns:220px 1fr 280px;gap:10px;min-height:0;">
    <div id="c6_toolbar" style="border:1px solid #d1d5db;border-radius:8px;padding:8px;overflow:auto;"></div>
    <div style="position:relative;border:1px solid #d1d5db;border-radius:8px;overflow:hidden;">
      <div id="c6_canvas" style="position:absolute;inset:0;"></div>
      <div id="c6_legend" style="position:absolute;right:8px;bottom:8px;"></div>
    </div>
    <div style="display:grid;grid-template-rows:1fr 220px;gap:10px;min-height:0;">
      <div id="c6_info" style="border:1px solid #d1d5db;border-radius:8px;overflow:auto;"></div>
      <div id="c6_minimap" style="border:1px solid #d1d5db;border-radius:8px;overflow:hidden;"></div>
    </div>
  </div>
</div>
""".strip(),
            "js": """
let showInfo = true;
let showMinimap = true;
let showChrome = true;

const viewer = FXGraphViewer.create({
  payload: api.payloads.structural,
  mount: {
    root: '#c6_hidden_root',
    slots: {
      canvas: '#c6_canvas',
      toolbar: '#c6_toolbar',
      info: '#c6_info',
      minimap: '#c6_minimap',
      legend: '#c6_legend',
    },
  },
  layout: {
    preset: 'headless',
    panels: {
      info: { visible: true },
      minimap: { visible: true, height: 220, resizable: false },
      legend: { visible: true },
    },
  },
  ui: {
    controls: {
      toolbar: true,
      search: true,
      layers: true,
      colorBy: true,
      theme: true,
      legend: true,
      zoomButtons: true,
      highlightButton: true,
      fullscreenButton: false,
    },
  },
  state: { activeExtensions: ['topological_order'], colorBy: 'topological_order' },
});
viewer.init();
api.registerViewer(viewer);

document.getElementById('c6_toggle_info').addEventListener('click', () => {
  showInfo = !showInfo;
  viewer.setLayout({ panels: { info: { visible: showInfo } } });
});

document.getElementById('c6_toggle_minimap').addEventListener('click', () => {
  showMinimap = !showMinimap;
  viewer.setLayout({ panels: { minimap: { visible: showMinimap } } });
});

document.getElementById('c6_toggle_chrome').addEventListener('click', () => {
  showChrome = !showChrome;
  viewer.setUIVisibility({
    toolbar: showChrome,
    search: showChrome,
    layers: showChrome,
    theme: showChrome,
  });
});

api.log('This case demonstrates host-owned slots and runtime layout toggles.');
""".strip(),
        },
        {
            "id": "js_07_events",
            "title": "JS 07: Events and Subscriptions",
            "description": "Observe and unsubscribe viewer events from host code.",
            "html": """
<div style="display:grid;grid-template-rows:auto 1fr;gap:10px;height:100%;">
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
    <button id="c7_theme">Toggle Theme</button>
    <button id="c7_select">Select First Node</button>
    <button id="c7_clear">Clear</button>
    <button id="c7_layout">Toggle Minimap</button>
    <button id="c7_unsub">Unsubscribe Events</button>
  </div>
  <div style="display:grid;grid-template-columns:1fr 360px;gap:10px;min-height:0;">
    <div id="c7_view" style="border:1px solid #d1d5db;border-radius:8px;overflow:hidden;"></div>
    <pre id="c7_log" style="margin:0;border:1px solid #d1d5db;border-radius:8px;padding:8px;overflow:auto;font-size:12px;"></pre>
  </div>
</div>
""".strip(),
            "js": """
const viewer = FXGraphViewer.create({
  payload: api.payloads.structural,
  mount: { root: '#c7_view' },
  layout: { preset: 'split' },
});
viewer.init();
api.registerViewer(viewer);

const logEl = document.getElementById('c7_log');
const firstNode = viewer.store.baseData.nodes[0].id;
let minimapVisible = true;

function log(line) {
  logEl.textContent = `${line}\n${logEl.textContent}`.slice(0, 3000);
}

const offState = viewer.on('statechange', (evt) => log(`statechange source=${evt.source}`));
const offSel = viewer.on('selectionchange', (evt) => log(`selection ${evt.prevSelection} -> ${evt.nextSelection}`));
const offTheme = viewer.on('themechange', (evt) => log(`theme ${evt.prevTheme} -> ${evt.nextTheme}`));
const offLayout = viewer.on('layoutchange', () => log('layoutchange'));

document.getElementById('c7_theme').addEventListener('click', () => {
  viewer.setTheme(viewer.getState().theme === 'light' ? 'dark' : 'light');
});
document.getElementById('c7_select').addEventListener('click', () => viewer.selectNode(firstNode, { animate: true, center: true }));
document.getElementById('c7_clear').addEventListener('click', () => viewer.clearSelection());
document.getElementById('c7_layout').addEventListener('click', () => {
  minimapVisible = !minimapVisible;
  viewer.setLayout({ panels: { minimap: { visible: minimapVisible } } });
});
document.getElementById('c7_unsub').addEventListener('click', () => {
  offState(); offSel(); offTheme(); offLayout();
  log('All event listeners unsubscribed');
});

api.log('Trigger controls and inspect event stream in the right log panel.');
""".strip(),
        },
        {
            "id": "js_08_compare_basics",
            "title": "JS 08: 3-Graph Compare + Auto debug_handle Sync",
            "description": "Three-view compare (Reference, Candidate A, Candidate B) using Map API. Default sync is 'Auto (handle→id)' — tries debug_handle first, falls back to node name.",
            "html": """
<div style="display:grid;grid-template-rows:auto 1fr;gap:10px;height:100%;">
  <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
    <span style="font-size:12px;color:#6b7280;">3-graph compare with auto debug_handle sync. Sidebar: Auto (handle→id) | ID only | Ext: per_layer_accuracy.debug_handle | Don't sync</span>
  </div>
  <div id="c8_grid" style="min-height:0;">
    <div id="c8_ref_mount" style="display:none;"></div>
    <div id="c8_cand1_mount" style="display:none;"></div>
    <div id="c8_cand2_mount" style="display:none;"></div>
  </div>
</div>
""".strip(),
            "js": """
let compare = null;

function buildCompare() {
  if (compare) { compare.destroy(); compare = null; }

  const ref = FXGraphViewer.create({
    payload: api.payloads.accuracy_reference,
    mount: { root: '#c8_ref_mount' },
    layout: { preset: 'split' },
    state: { activeExtensions: ['per_layer_accuracy', 'color_by_type'], colorBy: 'color_by_type' },
  });
  ref.init();
  api.registerViewer(ref);

  const cand1 = FXGraphViewer.create({
    payload: api.payloads.accuracy_candidate,
    mount: { root: '#c8_cand1_mount' },
    layout: { preset: 'split' },
    state: { activeExtensions: ['per_layer_accuracy'], colorBy: 'per_layer_accuracy' },
  });
  cand1.init();
  api.registerViewer(cand1);

  const cand2Payload = api.payloads.accuracy_candidate_2 || api.payloads.accuracy_candidate;
  const cand2 = FXGraphViewer.create({
    payload: cand2Payload,
    mount: { root: '#c8_cand2_mount' },
    layout: { preset: 'split' },
    state: { activeExtensions: ['per_layer_accuracy'], colorBy: 'per_layer_accuracy' },
  });
  cand2.init();
  api.registerViewer(cand2);

  compare = FXGraphCompare.create({
    viewers: new Map([
      ['Reference', ref],
      ['Candidate A', cand1],
      ['Candidate B', cand2],
    ]),
    layout: { container: '#c8_grid' },
    // sync defaults to { mode: 'auto' }
  });
  api.registerCompare(compare);

  api.log('3-graph compare ready. Sidebar shows Auto (handle→id) sync by default.');
}

buildCompare();
api.setCleanup(() => { if (compare) compare.destroy(); });
""".strip(),
        },
        {
            "id": "adv_01_accuracy_dynamic",
            "title": "ADV 01: Per-layer Accuracy Controls",
            "description": "Interesting combo: real per-layer metrics + dynamic threshold + theme + focus.",
            "html": """
<div style="display:grid;grid-template-columns:280px 1fr;gap:10px;height:100%;">
  <div style="border:1px solid #d1d5db;border-radius:8px;padding:10px;overflow:auto;">
    <div style="font-weight:600;margin-bottom:8px;">Accuracy Controls</div>
    <label>Theme
      <select id="acc_theme"><option value="light">light</option><option value="dark">dark</option></select>
    </label>
    <div style="height:8px;"></div>
    <label>Severity Percentile <input id="acc_threshold" type="range" min="0" max="100" value="25" style="width:100%;" /></label>
    <div id="acc_threshold_value" style="margin-top:6px;font-family:monospace;"></div>
    <button id="acc_focus_worst" style="margin-top:10px;">Focus Highest Severity Node</button>
  </div>
  <div id="acc_view" style="border:1px solid #d1d5db;border-radius:8px;overflow:hidden;"></div>
</div>
""".strip(),
            "js": """
const viewer = FXGraphViewer.create({
  payload: api.payloads.accuracy_candidate,
  mount: { root: '#acc_view' },
  layout: { preset: 'split', panels: { sidebar: { width: 420 } } },
  state: {
    activeExtensions: ['per_layer_accuracy', 'topological_order', 'color_by_type'],
    colorBy: 'per_layer_accuracy',
    theme: 'light',
  },
});
viewer.init();
api.registerViewer(viewer);

const extId = 'per_layer_accuracy';
const nodes = viewer.store.extensions[extId].nodes;
const severities = Object.values(nodes)
  .map((n) => Number((n.info && n.info.severity_score) || 0))
  .filter(Number.isFinite)
  .sort((a, b) => a - b);

const slider = document.getElementById('acc_threshold');
const label = document.getElementById('acc_threshold_value');
const themeSel = document.getElementById('acc_theme');
const focusBtn = document.getElementById('acc_focus_worst');

function quantile(q) {
  if (severities.length === 0) return 0;
  const i = Math.min(severities.length - 1, Math.max(0, Math.floor(q * (severities.length - 1))));
  return severities[i];
}

function applyThreshold() {
  const p = Number(slider.value) / 100;
  const threshold = quantile(p);
  label.textContent = `percentile=${slider.value}, threshold=${threshold.toExponential(3)}`;
  const patch = {};
  Object.entries(nodes).forEach(([nodeId, nodeData]) => {
    const s = Number((nodeData.info && nodeData.info.severity_score) || 0);
    patch[nodeId] = {
      fill_color: s >= threshold ? '#991b1b' : '#fecaca',
      label_append: [`sev=${s.toExponential(2)}`],
    };
  });
  viewer.patchLayerNodes(extId, patch);
  viewer.setColorBy(extId);
}

slider.addEventListener('input', applyThreshold);
themeSel.addEventListener('change', () => viewer.setTheme(themeSel.value));
focusBtn.addEventListener('click', () => {
  let worst = null;
  let worstScore = -Infinity;
  Object.entries(nodes).forEach(([nodeId, nodeData]) => {
    const s = Number((nodeData.info && nodeData.info.severity_score) || 0);
    if (s > worstScore) {
      worstScore = s;
      worst = nodeId;
    }
  });
  if (worst) viewer.selectNode(worst, { animate: true, center: true });
});

applyThreshold();
api.log(`Loaded real accuracy payload. worst_sample_index=${api.payloads.meta.worst_sample_index}`);
""".strip(),
        },
        {
            "id": "adv_02_headless_slots_slider",
            "title": "ADV 02: Headless Slots + Slider",
            "description": "Interesting combo: custom host layout + external slots + dynamic recoloring.",
            "html": """
<div id="adv2_case_view" style="width:100%;height:100%;">
  <div id="adv2_headless_mount" style="display:none;"></div>
  <div style="display:grid;grid-template-columns:220px 1fr 280px;gap:10px;height:100%;">
    <div style="border:1px solid #d1d5db;border-radius:8px;padding:10px;overflow:auto;">
      <div style="font-weight:600;margin-bottom:8px;">Custom Controls</div>
      <label>Threshold <input id="adv2_topo_threshold" type="range" min="0" max="100" value="20" style="width:100%;" /></label>
      <div id="adv2_topo_threshold_value" style="margin-top:6px;font-family:monospace;"></div>
      <div id="adv2_slot_legend" style="margin-top:12px;"></div>
    </div>
    <div style="position:relative;border:1px solid #d1d5db;border-radius:8px;overflow:hidden;">
      <div id="adv2_slot_canvas" style="position:absolute;inset:0;"></div>
    </div>
    <div style="display:grid;grid-template-rows:1fr 1fr;gap:10px;min-height:0;">
      <div id="adv2_slot_minimap" style="border:1px solid #d1d5db;border-radius:8px;overflow:hidden;min-height:0;"></div>
      <div id="adv2_slot_info" style="border:1px solid #d1d5db;border-radius:8px;overflow:hidden;min-height:0;"></div>
    </div>
  </div>
</div>
""".strip(),
            "js": """
const viewer = FXGraphViewer.create({
  payload: api.payloads.structural,
  mount: {
    root: '#adv2_headless_mount',
    slots: {
      canvas: '#adv2_slot_canvas',
      info: '#adv2_slot_info',
      minimap: '#adv2_slot_minimap',
      legend: '#adv2_slot_legend',
    },
  },
  layout: {
    preset: 'headless',
    panels: {
      minimap: { visible: true, height: 220, resizable: false },
      info: { visible: true },
      legend: { visible: true },
    },
  },
  ui: {
    controls: {
      toolbar: false,
      search: false,
      layers: false,
      colorBy: false,
      theme: false,
      legend: true,
      zoomButtons: false,
      highlightButton: false,
      fullscreenButton: false,
    },
  },
  state: { activeExtensions: ['topological_order'], colorBy: 'topological_order' },
});
viewer.init();
api.registerViewer(viewer);

const slider = document.getElementById('adv2_topo_threshold');
const valueEl = document.getElementById('adv2_topo_threshold_value');
const nodes = viewer.store.extensions['topological_order'].nodes;
const maxTopo = Math.max(...Object.values(nodes).map((n) => Number(n.info.topo_index || 0)));
slider.max = String(maxTopo);

function renderThreshold() {
  const threshold = Number(slider.value);
  valueEl.textContent = `threshold=${threshold} / ${maxTopo}`;
  const patch = {};
  Object.entries(nodes).forEach(([nodeId, nodeData]) => {
    const idx = Number(nodeData.info.topo_index || 0);
    patch[nodeId] = { fill_color: idx >= threshold ? '#b91c1c' : '#93c5fd' };
  });
  viewer.patchLayerNodes('topological_order', patch);
  viewer.setColorBy('topological_order');
}

slider.addEventListener('input', renderThreshold);
renderThreshold();
api.log('Headless slot composition active. Move slider and inspect recoloring.');
""".strip(),
        },
        {
            "id": "adv_03_fullscreen_toolbar",
            "title": "ADV 03: Fullscreen + Toolbar API",
            "description": "Interesting combo: fullscreen button in toolbar + direct fullscreen APIs.",
            "html": """
<div style="display:grid;grid-template-columns:220px 1fr;gap:10px;height:100%;">
  <div style="border:1px solid #d1d5db;border-radius:8px;padding:10px;overflow:auto;">
    <div style="font-weight:600;margin-bottom:8px;">Fullscreen Controls</div>
    <button id="adv3_enter_fs">Enter Fullscreen (API)</button>
    <button id="adv3_exit_fs" style="margin-left:8px;">Exit Fullscreen (API)</button>
    <p style="font-size:12px;color:#6b7280;">Taskbar also has a fullscreen toggle button in this case.</p>
  </div>
  <div id="adv3_view" style="border:1px solid #d1d5db;border-radius:8px;overflow:hidden;"></div>
</div>
""".strip(),
            "js": """
const viewer = FXGraphViewer.create({
  payload: api.payloads.structural,
  mount: { root: '#adv3_view' },
  layout: { preset: 'split', fullscreen: { enabled: true, button: true } },
  state: { activeExtensions: ['color_by_type'], colorBy: 'color_by_type' },
});
viewer.init();
api.registerViewer(viewer);

document.getElementById('adv3_enter_fs').addEventListener('click', () => viewer.enterFullscreen());
document.getElementById('adv3_exit_fs').addEventListener('click', () => viewer.exitFullscreen());

api.log('Use taskbar fullscreen button or side controls to validate API + UI integration.');
""".strip(),
        },
        {
            "id": "adv_04_tiled_compare",
            "title": "ADV 04: 3-Graph Compare + Extension Sync Key",
            "description": "Three-graph compare demonstrating set_sync_key('debug_handle') on per_layer_accuracy. Sidebar shows 'Ext: per_layer_accuracy.debug_handle' as an explicit sync option.",
            "html": """
<div id="adv4_grid" style="width:100%;height:100%;">
  <div id="adv4_ref_mount" style="display:none;"></div>
  <div id="adv4_left_mount" style="display:none;"></div>
  <div id="adv4_right_mount" style="display:none;"></div>
</div>
""".strip(),
            "js": """
const ref = FXGraphViewer.create({
  payload: api.payloads.accuracy_reference,
  mount: { root: '#adv4_ref_mount' },
  layout: { preset: 'split' },
  state: { activeExtensions: ['per_layer_accuracy', 'topological_order'], colorBy: 'topological_order' },
});
ref.init();
api.registerViewer(ref);

const left = FXGraphViewer.create({
  payload: api.payloads.accuracy_candidate,
  mount: { root: '#adv4_left_mount' },
  layout: { preset: 'split' },
  state: {
    activeExtensions: ['per_layer_accuracy', 'topological_order'],
    colorBy: 'per_layer_accuracy',
  },
});
left.init();
api.registerViewer(left);

const cand2Payload = api.payloads.accuracy_candidate_2 || api.payloads.accuracy_candidate;
const right = FXGraphViewer.create({
  payload: cand2Payload,
  mount: { root: '#adv4_right_mount' },
  layout: { preset: 'split' },
  state: {
    activeExtensions: ['per_layer_accuracy', 'topological_order'],
    colorBy: 'per_layer_accuracy',
  },
});
right.init();
api.registerViewer(right);

const compare = FXGraphCompare.create({
  viewers: new Map([
    ['Reference', ref],
    ['Candidate A', left],
    ['Candidate B', right],
  ]),
  layout: { container: '#adv4_grid' },
  sync: { mode: 'layer', layer: 'per_layer_accuracy', field: 'debug_handle' },
});
api.registerCompare(compare);

api.log('ADV04: 3-graph compare with extension sync key per_layer_accuracy.debug_handle active.');
""".strip(),
        },
        {
            "id": "js_99_combo_mixed",
            "title": "JS 99: Mixed Combo Demo",
            "description": "Current mixed demo: compare + sync + runtime mutation + events + themed controls.",
            "html": """
<div style="display:grid;grid-template-columns:320px 1fr;gap:10px;height:100%;">
  <div style="border:1px solid #d1d5db;border-radius:8px;padding:10px;overflow:auto;">
    <div style="font-weight:600;margin-bottom:8px;">Combo Controls</div>
    <label>Theme
      <select id="c99_theme"><option value="light">light</option><option value="dark">dark</option></select>
    </label>
    <div style="height:8px;"></div>
    <label>Severity Percentile
      <input id="c99_threshold" type="range" min="0" max="100" value="30" style="width:100%;" />
    </label>
    <div id="c99_threshold_text" style="margin-top:6px;font-family:monospace;"></div>
    <div style="height:8px;"></div>
    <label><input id="c99_sync_sel" type="checkbox" checked /> Sync selection</label><br />
    <div style="height:8px;"></div>
    <button id="c99_focus_worst">Focus Worst Node</button>
    <button id="c99_sequence" style="margin-left:6px;">Run Scripted Sequence</button>
    <div style="height:10px;"></div>
    <pre id="c99_log" style="margin:0;border:1px solid #d1d5db;border-radius:8px;padding:8px;min-height:180px;max-height:280px;overflow:auto;font-size:12px;"></pre>
  </div>
  <div id="c99_grid" style="min-height:0;">
    <div id="c99_left_mount" style="display:none;"></div>
    <div id="c99_right_mount" style="display:none;"></div>
  </div>
</div>
""".strip(),
            "js": """
const left = FXGraphViewer.create({
  payload: api.payloads.accuracy_reference,
  mount: { root: '#c99_left_mount' },
  layout: { preset: 'split' },
  state: { activeExtensions: ['color_by_type'], colorBy: 'color_by_type', theme: 'light' },
});
left.init();
api.registerViewer(left);

const right = FXGraphViewer.create({
  payload: api.payloads.accuracy_candidate,
  mount: { root: '#c99_right_mount' },
  layout: { preset: 'split' },
  state: {
    activeExtensions: ['per_layer_accuracy', 'topological_order', 'color_by_type'],
    colorBy: 'per_layer_accuracy',
    theme: 'light',
  },
});
right.init();
api.registerViewer(right);

const compare = FXGraphCompare.create({
  viewers: [left, right],
  layout: { columns: 2, container: '#c99_grid' },
  sync: { mode: 'id' },
});
api.registerCompare(compare);

const logEl = document.getElementById('c99_log');
function log(msg) {
  logEl.textContent = `${new Date().toLocaleTimeString()} ${msg}\n${logEl.textContent}`.slice(0, 4000);
}

const offSel = right.on('selectionchange', (evt) => log(`selection ${evt.prevSelection} -> ${evt.nextSelection}`));
const offTheme = right.on('themechange', (evt) => log(`theme ${evt.prevTheme} -> ${evt.nextTheme}`));

api.setCleanup(() => {
  offSel();
  offTheme();
});

const extId = 'per_layer_accuracy';
const nodes = right.store.extensions[extId].nodes;
const severities = Object.values(nodes)
  .map((n) => Number((n.info && n.info.severity_score) || 0))
  .filter(Number.isFinite)
  .sort((a, b) => a - b);

function quantile(q) {
  if (severities.length === 0) return 0;
  const i = Math.min(severities.length - 1, Math.max(0, Math.floor(q * (severities.length - 1))));
  return severities[i];
}

function applyThreshold() {
  const slider = document.getElementById('c99_threshold');
  const threshold = quantile(Number(slider.value) / 100);
  document.getElementById('c99_threshold_text').textContent =
    `percentile=${slider.value} threshold=${threshold.toExponential(3)}`;
  const patch = {};
  Object.entries(nodes).forEach(([nodeId, nodeData]) => {
    const s = Number((nodeData.info && nodeData.info.severity_score) || 0);
    patch[nodeId] = {
      fill_color: s >= threshold ? '#991b1b' : '#fecaca',
      label_append: [`sev=${s.toExponential(2)}`],
    };
  });
  right.patchLayerNodes(extId, patch);
  right.setColorBy(extId);
}

function focusWorst() {
  let worst = null;
  let worstScore = -Infinity;
  Object.entries(nodes).forEach(([nodeId, nodeData]) => {
    const s = Number((nodeData.info && nodeData.info.severity_score) || 0);
    if (s > worstScore) {
      worstScore = s;
      worst = nodeId;
    }
  });
  if (worst) {
    right.selectNode(worst, { animate: true, center: true });
    log(`focus worst node=${worst} score=${worstScore.toExponential(3)}`);
  }
}

document.getElementById('c99_theme').addEventListener('change', (e) => {
  left.setTheme(e.target.value);
  right.setTheme(e.target.value);
});
document.getElementById('c99_threshold').addEventListener('input', applyThreshold);
document.getElementById('c99_sync_sel').addEventListener('change', (e) => compare.setSync({ mode: e.target.checked ? 'id' : 'none' }));
document.getElementById('c99_focus_worst').addEventListener('click', focusWorst);
document.getElementById('c99_sequence').addEventListener('click', () => {
  log('scripted sequence start');
  left.setTheme('dark');
  applyThreshold();
  focusWorst();
  left.zoomToFit();
  right.zoomToFit();
  log('scripted sequence done');
});

applyThreshold();
log('Mixed combo demo ready.');
api.log('Final mixed demo: compare + sync + mutation + events + controls.');
""".strip(),
        },
    ]

    if include_qualcomm:
        cases.append(
            {
                "id": "qualcomm_metadata",
                "title": "QUALCOMM: PTQ Metadata",
                "description": "Qualcomm-specific payload metadata from real QNN PTQ path.",
                "html": """
<div style="display:grid;grid-template-columns:360px 1fr;gap:10px;height:100%;">
  <div style="border:1px solid #d1d5db;border-radius:8px;padding:10px;overflow:auto;">
    <h3 style="margin-top:0;">Qualcomm Metadata</h3>
    <pre id="qnn_meta" style="white-space:pre-wrap;font-size:12px;"></pre>
  </div>
  <div id="qnn_view" style="border:1px solid #d1d5db;border-radius:8px;overflow:hidden;"></div>
</div>
""".strip(),
                "js": """
const viewer = FXGraphViewer.create({
  payload: api.payloads.accuracy_candidate,
  mount: { root: '#qnn_view' },
  layout: { preset: 'split' },
  state: { activeExtensions: ['per_layer_accuracy'], colorBy: 'per_layer_accuracy' },
});
viewer.init();
api.registerViewer(viewer);

document.getElementById('qnn_meta').textContent = JSON.stringify(api.payloads.meta, null, 2);
api.log('Rendered Qualcomm PTQ payload + metadata snapshot.');
""".strip(),
            }
        )

    return cases
