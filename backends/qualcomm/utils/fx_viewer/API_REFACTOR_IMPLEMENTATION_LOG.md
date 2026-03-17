# FX Viewer API Refactor Implementation Log

Date: 2026-03-13
Owner: Codex

## Step 0: Scope and guardrails

Goal:
1. Implement the RFC v1-style JS API surface in current `fx_viewer` templates.
2. Validate with two example families: topological extension demo and per-layer accuracy demo.
3. Build a single HTML testcase harness that reuses one JS bundle and shared payloads.

Constraints/decisions:
1. Keep payload schema (`base` + `extensions`) unchanged.
2. Allow a compatibility constructor path for now (`new FXGraphViewer(containerId, payload)`), while adding the new factory-style API.
3. Prioritize deterministic behavior and explicit precedence over broad feature scope.

## Step 1: Architecture audit (completed)

Files inspected:
1. `templates/fx_graph_viewer.js`
2. `templates/view_controller.js`
3. `templates/ui_manager.js`
4. `templates/graph_data_store.js`
5. `templates/minimap_renderer.js`
6. `exporter.py`
7. `examples/demo_fx_viewer_extensions.py`
8. `examples/demo_per_layer_accuracy_fx.py`

Key findings:
1. UI synchronization gap exists: external state changes for theme/layer/colorBy are not fully reflected in controls.
2. Layout ownership is hardcoded to split shell; minimap/info mounting is coupled to sidebar.
3. Compare behavior in accuracy demo is achieved by demo-side monkey patching of `selectNode`.
4. Data-layer runtime mutation APIs do not exist (only static extension payload at init).

Implementation plan selected:
1. Add normalized config model with precedence rules inside `FXGraphViewer`.
2. Add API methods categorized in RFC.
3. Add explicit UI reconciliation (`syncControlsFromState`) in `UIManager`.
4. Add runtime layer mutation helpers in `GraphDataStore`.
5. Add small compare orchestrator class (`FXGraphCompare`).

## Step 2: v1 API implementation (completed)

Implemented modules:
1. `templates/fx_graph_viewer.js`
2. `templates/view_controller.js`
3. `templates/ui_manager.js`
4. `templates/graph_data_store.js`
5. `templates/minimap_renderer.js`

### 2.1 Decisions

1. Keep constructor compatibility (`new FXGraphViewer(containerId, payload)`) while adding v1 factory (`FXGraphViewer.create(config)`).
2. Implement `preset` as baseline defaults (`split`, `compact`, `headless`, `custom`) merged with explicit `layout`.
3. Enforce slot precedence by resolving mount slots before layout shell usage.
4. Add explicit event system and categorized APIs now; defer strict schema validator to a follow-up.

### 2.2 Implemented API surface

1. Construction:
   - `FXGraphViewer.create(config)`
2. State/events:
   - `getState`, `setState`, `replaceState`, `batch`
   - `on`, `off` with `statechange`, `selectionchange`, `themechange`, `layoutchange`
3. Appearance/control:
   - `setTheme`, `setLayers`, `setColorBy`, `setUIVisibility`, `setLayout`
4. Navigation:
   - `selectNode`, `clearSelection`, `search`, `zoomToFit`, `panToNode`, `animateToNode`
5. Runtime layer mutation:
   - `upsertLayer`, `removeLayer`, `patchLayerNodes`, `setLayerLabel`, `setColorRule`
6. Layout/lifecycle:
   - `enterFullscreen`, `exitFullscreen`, `destroy`
7. Compare:
   - `FXGraphCompare.create({ viewers, layout, sync })`
   - `setColumns`, `setCompact`, `setSync`, `destroy`

### 2.3 UI synchronization fix

Added `UIManager.syncControlsFromState()` and called it from controller state updates so external JS changes reflect in:
1. Theme select
2. Layer checkboxes
3. Color-by radios
4. Highlight toggle

This resolves the previously observed drift issue.

## Step 3: unified testcase harness (completed)

Created:
1. `backends/qualcomm/utils/fx_viewer/examples/generate_api_test_harness.py`
2. Generated output:
   - `backends/qualcomm/utils/fx_viewer/examples/fx_viewer_api_test_harness.html`

Harness characteristics:
1. Single shared JS bundle from exporter.
2. Shared payload pool generated once per run.
3. Testcase selector with source panes (HTML input + JS input) and live outcome panel.
4. Runtime log pane for each testcase.

Included testcases:
1. `topology_split`
2. `headless_slots`
3. `accuracy_dynamic`
4. `compare_sync`

## Step 4: environment and import path resolution (completed)

Observed issue:
1. `demo_per_layer_accuracy_fx.py` could import `fx_viewer` from `.venv/site-packages` when source tree path was not resolved correctly.
2. Site-packages copy lacked `templates/*.js`, causing `FileNotFoundError` during `export_html`.

Fix applied:
1. Updated source-path bootstrap in demos to add repo parent (`~/`) semantics correctly for local source imports.
2. Validated with required run convention:
   - source venv
   - source QAIRT env
   - `set -px PYTHONPATH ~/`

## Step 5: validation results (completed)

### 5.1 Static checks

1. `node --check` passed for all modified JS template files.
2. `python3 -m py_compile` passed for modified Python scripts.

### 5.2 Runtime checks

1. `demo_per_layer_accuracy_fx.py` with `--pipeline both` succeeded (fake_quant + qualcomm_ptq):
   - Output root: `/tmp/fx_viewer_acc_api_regress6`
2. `demo_fx_viewer_extensions.py --model swin` succeeded:
   - Output: `/tmp/fx_viewer_ext_regress/swin_graph_v3_extensions.html`
3. Harness generation succeeded:
   - `backends/qualcomm/utils/fx_viewer/examples/fx_viewer_api_test_harness.html`

## Notes

1. The warning from backend opinfo adapter is environment/version related and non-blocking for these demos.
2. Follow-up work recommended for strict runtime config validation and more granular compare sync modes.

## Step 6: testcase bugfixes from review (completed)

User-reported issues addressed:

1. `headless_slots` testcase failed with `root mount not found: #case_view`.
   - Cause: testcase HTML did not include `#case_view` while JS mounted to that id.
   - Fix: wrapped headless grid HTML in `<div id="case_view">...</div>`.

2. Sidebar splitter between info/minimap appeared at top.
   - Cause: `resizerH` was appended before info/minimap order was established.
   - Fix: create `resizerH` first but insert it into DOM only after minimap mount, positioned before minimap container so it sits between info panel and minimap.

3. Outcome panel requested resizable behavior to validate container resize handling.
   - Fix: added resizable `#outcomeHost` (`resize: both`) in harness and kept viewer mount in nested `#sandbox`.
   - Supporting runtime behavior: `CanvasRenderer` now observes container size via `ResizeObserver` and triggers `resize()`.

## Step 7: harness UX refinements (completed)

User-requested updates:
1. Full-screen default harness layout.
2. Run executes edited HTML/JS, not forced template reset.

Changes in `generate_api_test_harness.py`:
1. Harness page grid changed to full viewport (`height: 100vh`).
2. Body split uses responsive width (`minmax(360px, 36vw) 1fr`).
3. Testcase container heights switched to `height: 100%` to consume outcome pane.
4. Added per-testcase draft storage (`caseDrafts`) so edits persist while switching cases.
5. `runCurrentCase()` now executes `htmlCode.value` and `jsCode.value` directly without calling `renderCaseMeta()`.
