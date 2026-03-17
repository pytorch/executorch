"""Testcase catalog for the unified fx_viewer API harness.

This file keeps testcase definitions separate from payload generation so we can:
1) Reuse the same UI harness template.
2) Reuse the same testcase list across portable and Qualcomm profiles.
3) Keep JS snippets educational and easy to evolve.
"""

from __future__ import annotations

from typing import Any


def build_testcases(*, include_qualcomm: bool) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = [
        {
            "id": "topology_split",
            "title": "Topology + Type Layers (Split)",
            "description": "Baseline split layout. Demonstrates setLayers/setColorBy + legend updates.",
            "html": """
<div id=\"case_view\" style=\"width:100%;height:100%;border:1px solid #d1d5db;border-radius:8px;\"></div>
""".strip(),
            "js": """
// Educational: create a standard split viewer and activate two extension layers.
const viewer = FXGraphViewer.create({
  payload: api.payloads.structural,
  mount: { root: '#case_view' },
  layout: { preset: 'split' },
  state: { theme: 'light' },
});
viewer.init();

// Show both structural layers, then color by topological order.
viewer.setLayers(['color_by_type', 'topological_order']);
viewer.setColorBy('topological_order');

api.registerViewer(viewer);
api.log('Loaded structural payload with color_by_type + topological_order');
""".strip(),
        },
        {
            "id": "headless_slots",
            "title": "Headless Slots + Dynamic Slider",
            "description": "Demonstrates mount.slots ownership + patchLayerNodes for dynamic recoloring.",
            "html": """
<div id=\"case_view\" style=\"width:100%;height:100%;\">
  <div id=\"headless_mount\" style=\"display:none;\"></div>
  <div style=\"display:grid;grid-template-columns:220px 1fr 280px;gap:10px;height:100%;\">
    <div style=\"border:1px solid #d1d5db;border-radius:8px;padding:10px;overflow:auto;\">
      <div style=\"font-weight:600;margin-bottom:8px;\">Custom Controls</div>
      <label>Threshold <input id=\"topo_threshold\" type=\"range\" min=\"0\" max=\"100\" value=\"20\" style=\"width:100%;\" /></label>
      <div id=\"topo_threshold_value\" style=\"margin-top:6px;font-family:monospace;\"></div>
      <div id=\"slot_legend\" style=\"margin-top:12px;\"></div>
    </div>
    <div style=\"position:relative;border:1px solid #d1d5db;border-radius:8px;overflow:hidden;\">
      <div id=\"slot_canvas\" style=\"position:absolute;inset:0;\"></div>
    </div>
    <div style=\"display:grid;grid-template-rows:1fr 1fr;gap:10px;\">
      <div id=\"slot_minimap\" style=\"border:1px solid #d1d5db;border-radius:8px;overflow:hidden;\"></div>
      <div id=\"slot_info\" style=\"border:1px solid #d1d5db;border-radius:8px;overflow:auto;\"></div>
    </div>
  </div>
</div>
""".strip(),
            "js": """
// Educational: in headless mode we mount viewer shell to a hidden root,
// and dock renderers into external slots controlled by host HTML.
const viewer = FXGraphViewer.create({
  payload: api.payloads.structural,
  mount: {
    root: '#headless_mount',
    slots: {
      canvas: '#slot_canvas',
      info: '#slot_info',
      minimap: '#slot_minimap',
      legend: '#slot_legend',
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
      clearButton: false,
      highlightButton: false,
      fullscreenButton: false,
    },
  },
  state: { activeExtensions: ['topological_order'], colorBy: 'topological_order' },
});
viewer.init();
api.registerViewer(viewer);

// Educational: dynamic patch by node id + re-apply colorBy.
const slider = document.getElementById('topo_threshold');
const valueEl = document.getElementById('topo_threshold_value');
const nodes = viewer.store.extensions['topological_order'].nodes;
const maxTopo = Math.max(...Object.values(nodes).map(n => Number(n.info.topo_index || 0)));
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
            "id": "accuracy_dynamic",
            "title": "Per-layer Accuracy Controls",
            "description": "Real per-layer accuracy payload with dynamic threshold + theme sync.",
            "html": """
<div style=\"display:grid;grid-template-columns:280px 1fr;gap:10px;height:100%;\">
  <div style=\"border:1px solid #d1d5db;border-radius:8px;padding:10px;overflow:auto;\">
    <div style=\"font-weight:600;margin-bottom:8px;\">Accuracy Controls</div>
    <label>Theme
      <select id=\"acc_theme\"><option value=\"light\">light</option><option value=\"dark\">dark</option></select>
    </label>
    <div style=\"height:8px;\"></div>
    <label>Severity Percentile <input id=\"acc_threshold\" type=\"range\" min=\"0\" max=\"100\" value=\"25\" style=\"width:100%;\" /></label>
    <div id=\"acc_threshold_value\" style=\"margin-top:6px;font-family:monospace;\"></div>
    <button id=\"acc_focus_worst\" style=\"margin-top:10px;\">Focus Highest Severity Node</button>
  </div>
  <div id=\"acc_view\" style=\"border:1px solid #d1d5db;border-radius:8px;overflow:hidden;\"></div>
</div>
""".strip(),
            "js": """
// Educational: this payload includes real per-layer accuracy metrics from capture outputs.
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
  .map(n => Number((n.info && n.info.severity_score) || 0))
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
            "id": "fullscreen_toolbar",
            "title": "Fullscreen Button API",
            "description": "Taskbar fullscreen button (layout.fullscreen.button) + programmatic fullscreen API.",
            "html": """
<div style=\"display:grid;grid-template-columns:220px 1fr;gap:10px;height:100%;\">
  <div style=\"border:1px solid #d1d5db;border-radius:8px;padding:10px;overflow:auto;\">
    <div style=\"font-weight:600;margin-bottom:8px;\">Fullscreen Controls</div>
    <button id=\"api_enter_fs\">Enter Fullscreen (API)</button>
    <button id=\"api_exit_fs\" style=\"margin-left:8px;\">Exit Fullscreen (API)</button>
    <p style=\"font-size:12px;color:#6b7280;\">Taskbar also has a fullscreen toggle button in this case.</p>
  </div>
  <div id=\"fs_view\" style=\"border:1px solid #d1d5db;border-radius:8px;overflow:hidden;\"></div>
</div>
""".strip(),
            "js": """
// Educational: fullscreen button is enabled by layout.fullscreen.button.
const viewer = FXGraphViewer.create({
  payload: api.payloads.structural,
  mount: { root: '#fs_view' },
  layout: { preset: 'split', fullscreen: { enabled: true, button: true } },
  state: { activeExtensions: ['color_by_type'], colorBy: 'color_by_type' },
});
viewer.init();
api.registerViewer(viewer);

document.getElementById('api_enter_fs').addEventListener('click', () => viewer.enterFullscreen());
document.getElementById('api_exit_fs').addEventListener('click', () => viewer.exitFullscreen());

api.log('Use taskbar fullscreen button or side controls to validate API + UI integration.');
""".strip(),
        },
        {
            "id": "compare_sync",
            "title": "Compare + Sync",
            "description": "Native FXGraphCompare orchestration with sync and compact toggles.",
            "html": """
<div style=\"display:flex;align-items:center;gap:12px;margin-bottom:8px;\">
  <label><input id=\"cmp_sync\" type=\"checkbox\" checked /> Sync selection</label>
  <label><input id=\"cmp_compact\" type=\"checkbox\" checked /> Compact mode</label>
</div>
<div id=\"cmp_grid\" style=\"display:grid;grid-template-columns:1fr 1fr;gap:10px;height:calc(100% - 42px);\">
  <div id=\"cmp_left\" style=\"border:1px solid #d1d5db;border-radius:8px;overflow:hidden;\"></div>
  <div id=\"cmp_right\" style=\"border:1px solid #d1d5db;border-radius:8px;overflow:hidden;\"></div>
</div>
""".strip(),
            "js": """
const left = FXGraphViewer.create({
  payload: api.payloads.accuracy_reference,
  mount: { root: '#cmp_left' },
  layout: { preset: 'split' },
});
left.init();
api.registerViewer(left);

const right = FXGraphViewer.create({
  payload: api.payloads.accuracy_candidate,
  mount: { root: '#cmp_right' },
  layout: { preset: 'split' },
  state: { activeExtensions: ['per_layer_accuracy'], colorBy: 'per_layer_accuracy' },
});
right.init();
api.registerViewer(right);

const compare = FXGraphCompare.create({
  viewers: [left, right],
  layout: { columns: 2, compact: true, container: '#cmp_grid' },
  sync: { selection: true },
});
api.registerCompare(compare);

document.getElementById('cmp_sync').addEventListener('change', (e) => {
  compare.setSync({ selection: e.target.checked });
});

document.getElementById('cmp_compact').addEventListener('change', (e) => {
  compare.setCompact(e.target.checked);
});

api.log('Select nodes in either pane to verify synced focus behavior.');
""".strip(),
        },
    ]

    if include_qualcomm:
        cases.append(
            {
                "id": "qualcomm_metadata",
                "title": "Qualcomm PTQ Metadata",
                "description": "Qualcomm-specific payload metadata from real QNN PTQ path.",
                "html": """
<div style=\"display:grid;grid-template-columns:360px 1fr;gap:10px;height:100%;\">
  <div style=\"border:1px solid #d1d5db;border-radius:8px;padding:10px;overflow:auto;\">
    <h3 style=\"margin-top:0;\">Qualcomm Metadata</h3>
    <pre id=\"qnn_meta\" style=\"white-space:pre-wrap;font-size:12px;\"></pre>
  </div>
  <div id=\"qnn_view\" style=\"border:1px solid #d1d5db;border-radius:8px;overflow:hidden;\"></div>
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
