# RFC: fx_viewer API Interface v1 (Breaking)

Date: 2026-03-13  
Status: Draft for review  
Owners: Qualcomm Executorch debugging team  
Scope: `backends/qualcomm/utils/fx_viewer` public JavaScript API and embedding contract

## 1. Abstract

This RFC defines a breaking v1 API for `fx_viewer` so it can be used consistently in:

1. Standalone graph/accuracy demos.
2. Observatory GraphView integration.

The v1 design is state-driven, layout-composable, compare-native, and explicit about ownership/override rules.

## 2. Background

### 2.1 What is `fx_viewer`

`fx_viewer` renders FX graph payloads (`base` graph + optional extension layers) with interactive controls (selection, search, theme, color-by, minimap, info panel).

### 2.2 What is Observatory (`debugging_utils`)

Observatory captures and organizes multiple debug records across compilation/runtime stages using a lens framework and report UI.

### 2.3 Why this RFC

Current integration works but requires ad-hoc JS/DOM coupling for layout, compare sync, and dynamic per-layer coloring. v1 makes these capabilities first-class.

## 3. Problem Statement

1. External API updates can desync built-in controls.
2. Layout shell and graph core are tightly coupled.
3. Host custom controls (sliders/jump links) lack stable contracts.
4. Compare orchestration is duplicated per demo.

## 4. Goals

1. Single source of truth state.
2. Unified layout config with strict precedence rules.
3. Explicit host/viewer ownership rules.
4. Clear JS API categories.
5. First-class runtime data/layer mutation.
6. First-class N-view compare + sync.

## 5. Non-Goals

1. Backward compatibility with old APIs.
2. Observatory backend schema details.
3. Large-graph performance redesign.

## 6. Unified Config Model

`FXGraphViewer.create(...)` accepts one normalized config shape.

```ts
const viewer = FXGraphViewer.create({
  payload,
  mount: {
    root: "#graph-root",
    slots: {
      canvas: "#canvas-slot",
      toolbar: "#toolbar-slot",
      info: "#info-slot",
      minimap: "#minimap-slot",
      legend: "#legend-slot",
    },
  },
  layout: {
    preset: "split", // split | compact | headless | custom
    panels: {
      sidebar: { visible: true, width: 420, resizable: true, collapsible: true },
      info: { visible: true, dock: "sidebar" },
      minimap: { visible: true, dock: "sidebar", height: 240, resizable: true },
      legend: { visible: true, dock: "toolbar" },
    },
    fullscreen: { enabled: true, button: true },
  },
  ui: {
    controls: {
      toolbar: true,
      search: true,
      layers: true,
      colorBy: true,
      theme: true,
      minimapToggle: true,
      zoomButtons: true,
      clearSelection: true,
    },
  },
  state: {
    theme: "light",
    activeExtensions: ["per_layer_accuracy"],
    colorBy: "per_layer_accuracy",
    selectedNodeId: null,
    searchQuery: "",
    highlightAncestors: true,
  },
});
```

## 7. Precedence and Ownership Rules

## 7.1 Precedence (highest to lowest)

1. Explicit `mount.slots.*` (placement owner).
2. Explicit `layout.*` fields (behavior/visibility/dock defaults).
3. `layout.preset` defaults.
4. Internal built-in defaults.

Interpretation:

1. If `mount.slots.info` is given, info panel mounts there even if preset is `split`.
2. Preset cannot override explicit slot placement.
3. Explicit layout fields override preset behavior values.

## 7.2 Ownership

1. Host owns slot nodes (`#info-slot`, `#minimap-slot`, etc).
2. Viewer never overwrites host attributes/classes/styles on slot nodes.
3. Viewer creates and owns child nodes mounted inside slot nodes.
4. If slot is absent, viewer creates and owns container nodes in preset/custom layout shell.

## 7.3 HTML attribute behavior

1. Host HTML attributes are preserved.
2. Viewer style/classes apply to viewer-owned descendants.
3. Host can style slot containers; viewer guarantees stable mount points.

## 8. Presets

`preset` is a layout baseline recipe only.

1. `split`: canvas + sidebar defaults.
2. `compact`: minimal chrome, graph-first.
3. `headless`: no shell; host provides slots.
4. `custom`: no defaults; all structure explicit.

Rule: preset fills missing layout values only.

## 9. Public JS API (Categorized)

## 9.1 State APIs

```ts
viewer.getState(): ViewerState;
viewer.setState(patch: Partial<ViewerState>, opts?: { source?: "api" | "ui" | "system" }): void;
viewer.replaceState(next: ViewerState, opts?: { source?: "api" | "ui" | "system" }): void;
viewer.batch(fn: () => void): void; // coalesce redraw/events
```

## 9.2 Data/Layer Mutation APIs (runtime)

```ts
viewer.upsertLayer(layerId: string, layerPayload: LayerPayload): void;
viewer.removeLayer(layerId: string): void;
viewer.patchLayerNodes(layerId: string, patchByNodeId: Record<string, NodePatch>): void;
viewer.setColorRule(layerId: string, colorRule: ColorRule): void;
viewer.setLayerLabel(layerId: string, label: string): void;
```

Use these for slider-driven threshold coloring or dynamic labels. Do not mutate transient renderer-only state.

## 9.3 Selection/Camera APIs

```ts
viewer.selectNode(nodeId: string, opts?: { animate?: boolean; center?: boolean; durationMs?: number }): void;
viewer.clearSelection(): void;
viewer.panToNode(nodeId: string): void;
viewer.animateToNode(nodeId: string, opts?: { durationMs?: number; easing?: string; k?: number }): void;
viewer.zoomToFit(): void;
```

## 9.4 Appearance/UI APIs

```ts
viewer.setTheme(themeName: string): void;
viewer.setLayers(layerIds: string[]): void;
viewer.setColorBy(layerId: string): void;
viewer.setUIVisibility(flags: Partial<UIVisibility>): void;
```

## 9.5 Layout APIs

```ts
viewer.setLayout(layoutPatch: Partial<LayoutConfig>): void;
viewer.enterFullscreen(): void;
viewer.exitFullscreen(): void;
```

## 9.6 Lifecycle and Events

```ts
viewer.on("statechange", (e) => {});
viewer.on("selectionchange", (e) => {});
viewer.on("layoutchange", (e) => {});
viewer.on("themechange", (e) => {});
viewer.on("error", (e) => {});
viewer.destroy(): void;
```

All events include `source`, `timestamp`, and relevant previous/next snapshots.

## 10. Runtime Mutation Semantics (Dynamic threshold use case)

Goal: support real-time slider callbacks that update node color/severity by node id.

Rules:

1. Mutate layer registry (`upsertLayer`, `patchLayerNodes`, `setColorRule`), not transient active node cache.
2. Layer toggle off/on must preserve patched values.
3. `setColorBy(layerId)` should immediately use latest patched values.
4. Prefer `batch(...)` for smooth updates.

Example:

```ts
slider.oninput = (threshold) => {
  const patch = computePatch(threshold); // nodeId -> { value, color, label }
  viewer.batch(() => {
    viewer.patchLayerNodes("per_layer_accuracy", patch);
    viewer.setColorBy("per_layer_accuracy");
  });
};
```

## 11. UI Synchronization Contract

Built-in UI is a pure state adapter.

1. API updates must always reflect in UI controls.
2. UI interaction must always dispatch state mutations, never direct renderer mutations.
3. No hidden UI-only state for theme/layers/colorBy/search.

This closes the current drift issue.

## 12. Compare API

```ts
const compare = FXGraphCompare.create({
  viewers: [viewerA, viewerB],
  layout: { columns: 2, tiled: true, container: '#compare-root' },
  sharedTaskbar: {
    enabled: true,
    controls: { theme: true, layers: true, zoomFit: true, fullscreen: true, tiledToggle: true, syncMode: true },
  },
  sync: { mode: 'id' },  // 'none' | 'id' | 'layer'
  // For layer sync: sync: { mode: 'layer', layer: 'ext_name', field: 'field_name' }
});

compare.setColumns(3);
compare.setTiled(false);
compare.setSync({ mode: 'layer', layer: 'topological_order', field: 'topo_index' });
compare.destroy();
```

Semantics:

1. Source-guarded propagation avoids loops.
2. `sync.mode: 'id'` — select matching node id in other viewers.
3. `sync.mode: 'layer'` — match by `extensions[layer].nodes[id].info[field]` value; topologically last on ties.
4. `layout.tiled: true` (default) — vertical stack: minimap top, canvas middle, info bottom.
5. `sharedTaskbar.enabled: true` (default) — shared taskbar above grid; per-viewer toolbars hide except search.
6. `setCompact()` is a deprecated alias for `setTiled()`.
7. Viewers remain independently usable outside compare orchestration.

## 13. Breaking Changes

Removed in v1:

1. Legacy ad-hoc constructor mutation paths.
2. Direct DOM poking as integration contract.
3. Non-state-backed control update paths.

## 14. Implementation Plan

1. Add `ViewerStateStore` (schema + reducer/actions + event bus).
2. Add `LayoutManager` (preset resolution + ownership + docking + splitters).
3. Refactor `UIManager` into state subscriber/dispatcher.
4. Add `LayerRegistry` mutation APIs.
5. Add `FXGraphCompare` orchestrator.
6. Migrate templates to v1-only paths.

## 15. Testing Plan

1. Precedence tests: slot/layout/preset conflict resolution.
2. Ownership tests: host attributes preserved; viewer children mounted correctly.
3. UI sync tests: API-driven theme/layer/colorBy reflected in controls.
4. Runtime mutation tests: slider-style patch updates persist across layer toggles.
5. Compare tests: sync propagation and loop prevention.
6. E2E demo tests: standalone accuracy view + split/compact/headless embeddings.

## 16. Risks and Mitigations

1. Risk: complexity increase from flexibility.
   Mitigation: strict normalized config + validation errors.
2. Risk: regressions during migration.
   Mitigation: interaction snapshot coverage for default exports.
3. Risk: unclear host/viewer responsibilities.
   Mitigation: explicit ownership rules and precedence docs (Sections 7 and 6).

## 17. Expected Outcome

1. Lens and demo authors get one clear API model.
2. Dynamic per-layer coloring/labels from JS becomes official and stable.
3. Layout composition is powerful without DOM hacks.
4. Compare/sync capabilities are reusable instead of reimplemented.

## Appendix A: Integration Recipes

### A.1 Standalone Split Viewer

```ts
const viewer = FXGraphViewer.create({
  payload,
  mount: { root: "#graph-root" },
  layout: { preset: "split" },
  state: { theme: "light" },
});
viewer.init();
```

### A.2 Headless Embedding with External Slots

```ts
const viewer = FXGraphViewer.create({
  payload,
  mount: {
    root: "#root",
    slots: {
      canvas: "#canvas-slot",
      info: "#info-slot",
      minimap: "#minimap-slot",
      legend: "#legend-slot",
    },
  },
  layout: {
    preset: "headless",
    panels: {
      minimap: { visible: true, height: 220 },
      info: { visible: true },
      legend: { visible: true },
    },
  },
  ui: { controls: { toolbar: false, search: false, layers: false, colorBy: false, theme: false } },
});
viewer.init();
```

### A.3 Compare with Sync Toggle

```ts
const compare = FXGraphCompare.create({
  viewers: [leftViewer, rightViewer],
  layout: { columns: 2, tiled: true },
  sync: { mode: 'id' },
});

syncCheckbox.onchange = (e) => compare.setSync({ mode: e.target.checked ? 'id' : 'none' });
tiledCheckbox.onchange = (e) => compare.setTiled(e.target.checked);
```

## Appendix B: Runtime Threshold Coloring Recipe

```ts
slider.oninput = (e) => {
  const threshold = Number(e.target.value);
  const patch = {};
  Object.entries(nodes).forEach(([nodeId, nodeData]) => {
    const sev = Number(nodeData.info.severity_score || 0);
    patch[nodeId] = {
      fill_color: sev >= threshold ? "#991b1b" : "#fecaca",
      label_append: [`sev=${sev.toExponential(2)}`],
    };
  });

  viewer.batch(() => {
    viewer.patchLayerNodes("per_layer_accuracy", patch);
    viewer.setColorBy("per_layer_accuracy");
  });
};
```

## Appendix C: Precedence Quick Reference

1. `mount.slots.*` decides where modules mount.
2. `layout.*` decides behavior for mounted modules.
3. `layout.preset` fills only missing layout values.
4. Internal defaults apply only when nothing else is specified.
