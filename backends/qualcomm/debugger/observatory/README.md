# Observatory (GraphView Minimal)

This directory contains a new, review-focused observatory runtime under:

`backends/qualcomm/debugger/observatory`

The implementation is intentionally breaking and typed.

## 1. Goals

1. Keep runtime behavior close to legacy observatory UI.
2. Use strict dataclass contracts instead of raw dict APIs.
3. Make graph rendering first-class via `GraphView` + `GraphHub`.
4. Split JS runtime into topic files for easier review.
5. Support ETRecord auto-collection while context is enabled.

## 2. Main Entry Points

1. `__init__.py`: exports `Observatory`.
2. `observatory.py`: runtime lifecycle and report assembly.
3. `interfaces.py`: typed API and contract validation.
4. `graph_hub.py`: graph asset/layer merge logic.
5. `auto_collect.py`: ETRecord monkey-patch auto collection.

## 3. Quick Start

```bash
source ~/executorch/.venv/bin/activate
source ~/executorch/qairt/2.37.0.250724/bin/envsetup.sh
export PYTHONPATH=~/
```

```python
import torch
from executorch.backends.qualcomm.debugger.observatory import Observatory

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 8)
    def forward(self, x):
        return self.fc(x)

model = M().eval()
graph = torch.fx.symbolic_trace(model)

Observatory.clear()
with Observatory.enable_context():
    Observatory.collect("step_0", graph)

Observatory.export_html_report("/tmp/observatory_report.html")
```

## 4. Typed Frontend Example

```python
from executorch.backends.qualcomm.debugger.observatory.interfaces import (
    Frontend,
    TableBlock,
    TableRecordSpec,
    ViewList,
)

class MyFrontend(Frontend):
    def record(self, digest, analysis, context):
        return ViewList(
            blocks=[
                TableBlock(
                    id="summary",
                    title="Summary",
                    record=TableRecordSpec(data={"nodes": 42}),
                    order=0,
                )
            ]
        )
```

## 5. GraphView Example

```python
from executorch.backends.qualcomm.debugger.observatory.interfaces import GraphView

graph_block = GraphView(
    id="acc_graph",
    title="Accuracy Graph",
    graph_ref="record_0",
    default_layers=["accuracy/error"],
    default_color_by="accuracy/error",
).as_block()
```

## 6. Analyze-Step Graph Layer Contribution

Use `RecordAnalysis.graph_layers` from `analyze()`.

```python
from executorch.backends.qualcomm.debugger.observatory.interfaces import (
    AnalysisResult,
    RecordAnalysis,
)
from executorch.backends.qualcomm.utils.fx_viewer import (
    GraphExtensionPayload,
    GraphExtensionNodePayload,
)

payload = GraphExtensionPayload(
    id="error",
    name="Accuracy Error",
    legend=[{"label": "Low", "color": "#93c5fd"}],
    nodes={"node_0": GraphExtensionNodePayload(fill_color="#93c5fd")},
)

record_analysis = RecordAnalysis(data={"max_mse": 0.1})
record_analysis.add_graph_layer("error", payload)

return AnalysisResult(per_record_data={"step_1": record_analysis})
```

## 7. ETRecord Auto-Collection

While inside `enable_context()`, observatory patches ETRecord methods:

1. `ETRecord.add_exported_program`
2. `ETRecord.add_edge_dialect_program`
3. `ETRecord.add_extra_export_modules`

These calls auto-trigger `Observatory.collect(...)`.

## 8. Demos

1. `examples/demo_graphview_accuracy_compare.py`
2. `examples/demo_etrecord_auto_collect.py`
3. `examples/generate_ui_test_harness.py`

## 9. Tests

```bash
pytest -q backends/qualcomm/debugger/observatory/tests
```


## 10. Contract Tables

The tables below define the actual contracts across lens code, report JSON,
frontend rendering, and custom JS callbacks.

### 10.1 End-to-End Stage Matrix

| Stage | Entrypoint / Signature | Primary input | Primary output | Output JSON path |
| --- | --- | --- | --- | --- |
| Runtime capture | `Observatory.collect(name: str, artifact: Any)` | `artifact`, `ObservationContext(config, shared_state)` | `RecordDigest(name, timestamp, data)` | `records[i].digests` |
| Session hooks | `Lens.on_session_start/end(context)` | `ObservationContext` | lens-scoped session payload | `session.start_data[lens]`, `session.end_data[lens]` |
| Analyze | `Lens.analyze(records, config) -> AnalysisResult` | `List[RecordDigest]`, merged config | `global_data`, `per_record_data[name]` | `analysis_results[lens]` |
| Graph merge | `GraphHub.add_analysis_layers(graph_ref, lens_name, record_analysis)` | `RecordAnalysis.graph_layers` | namespaced layer map | `graph_layers[graph_ref]["<lens>/<key>"]` |
| Frontend assembly | `Frontend.dashboard(...)`, `Frontend.record(...)` | digest/session/analysis values | `ViewList(blocks=[...])` | `dashboard[lens]`, `records[i].views[lens]` |
| Browser render | `renderMain()/renderUnifiedView()` | full report payload | DOM sections / graph viewers | runtime only |
| Custom JS invoke | `fn(container, args, context, analysis)` | block spec + runtime context | user DOM changes | runtime only |

### 10.2 Lens Lifecycle Signatures by Stage

| Stage | Method | Signature | Return / Side Effect |
| --- | --- | --- | --- |
| Registration | `get_name` | `@classmethod get_name() -> str` | stable lens key |
| Registration | `setup` | `@classmethod setup() -> None` | one-time setup |
| Session | `on_session_start` | `@classmethod on_session_start(context: ObservationContext) -> Optional[Serializable]` | stored in `session.start_data[lens]` |
| Runtime | `observe` | `@classmethod observe(artifact: Any, context: ObservationContext) -> Any` | transient observation |
| Runtime | `digest` | `@classmethod digest(observation: Any, context: ObservationContext) -> Serializable` | persisted digest in `RecordDigest.data[lens]` |
| Session | `on_session_end` | `@classmethod on_session_end(context: ObservationContext) -> Optional[Serializable]` | stored in `session.end_data[lens]` |
| Analyze | `analyze` | `@staticmethod analyze(records: List[RecordDigest], config: Dict[str, Any]) -> AnalysisResult` | global + per-record derived data |
| Frontend | `get_frontend_spec` | `@staticmethod get_frontend_spec() -> Frontend` | returns strategy object |
| Reset | `clear` | `@classmethod clear() -> None` | clears lens internal state |

### 10.3 Frontend Stage Signatures and Input Sources

| Method | Signature | Python-side argument source | Serialized destination |
| --- | --- | --- | --- |
| `resources` | `resources() -> Dict[str, str]` | lens frontend implementation | `resources.js[]`, `resources.css[]` |
| `dashboard` | `dashboard(start, end, analysis, records) -> Optional[ViewList]` | `start=session.start_data[lens]`, `end=session.end_data[lens]`, `analysis=analysis_results[lens].global_data`, `records=List[RecordDigest]` | `dashboard[lens]` |
| `record` | `record(digest, analysis, context) -> Optional[ViewList]` | `digest=record.data[lens]`, `analysis={"global": global_data, "record": per_record_data[name].data}`, `context={"index", "name"}` | `records[i].views[lens]` |
| `check_badges` | `check_badges(digest, analysis) -> List[Dict[str, str]]` | current digest + `global_data` | `records[i].badges[]` |
| `check_index_diffs` | `check_index_diffs(prev_digest, curr_digest, analysis) -> Dict[str, str]` | previous/current digest + `global_data` | `records[i].diff_index` |

### 10.4 View Block Contracts (Frontend Output)

| Block type | Python dataclass | Required record fields | Compare modes | JS renderer path |
| --- | --- | --- | --- | --- |
| Table | `TableBlock(record=TableRecordSpec)` | `record.data: Dict[str, Serializable]` | `auto`, `disabled` | `renderTableContent` |
| HTML | `HtmlBlock(record=HtmlRecordSpec)` | `record.content: str` | `auto`, `disabled` | `content.innerHTML` |
| Custom | `CustomBlock(record=CustomRecordSpec)` | `record.js_func: str`, `record.args: dict` | `custom`, `disabled` | `resolveFunction(js_func)` then callback |
| Graph | `GraphBlock(record=GraphRecordSpec)` | `record.graph_ref: str` | `auto`, `custom`, `disabled` | `mountGraphViewer` |

| Common block fields | Type | Notes |
| --- | --- | --- |
| `id` | `str` | must be non-empty, unique inside one `ViewList` |
| `title` | `str` | section header text |
| `order` | `int` | stable sort key for rendering |
| `collapsible` | `bool` | section open/close behavior |

### 10.5 Information Object Map Across Boundaries

| Python object | Produced in Python stage | JSON path in report | JS access pattern |
| --- | --- | --- | --- |
| `RecordDigest` | `collect` | `records[i]` | `state.data.records[i]` |
| `RecordDigest.data[lens]` | `digest` | `records[i].digests[lens]` | `context.record.digests[lens]` in record custom JS |
| `SessionResult.start_data[lens]` | `on_session_start` | `session.start_data[lens]` | dashboard custom context `start` |
| `SessionResult.end_data[lens]` | `on_session_end` | `session.end_data[lens]` | dashboard custom context `end` |
| `AnalysisResult.global_data` | `analyze` | `analysis_results[lens].global_data` | `analysis.global_data` in custom JS |
| `AnalysisResult.per_record_data[name].data` | `analyze` | `analysis_results[lens].per_record_data[name].data` | `analysis.per_record_data?.[recordName]?.data` |
| `AnalysisResult.per_record_data[name].graph_layers` | `analyze` | merged into `graph_layers[graph_ref]` | included via viewer `extensions` |
| `ViewList` | frontend callbacks | `dashboard[lens].blocks` / `records[i].views[lens].blocks` | `getLensBlocks(record, lensName)` |

### 10.6 Custom JS Callback Signatures

| Callback stage | Invocation site | Signature | `context` shape |
| --- | --- | --- | --- |
| Record block render | `renderRecordBlock(..., context={ index, record }, analysis)` | `fn(container, args, context, analysis)` | `{ index: number, record: SerializedRecord }` |
| Dashboard block render | `renderDashboard(..., context={ start, end, records }, analysis)` | `fn(container, args, context, analysis)` | `{ start: object, end: object, records: SerializedRecord[] }` |
| Compare render (`mode="custom"`) | `renderCustomCompare(..., context, analysis)` | `fn(container, args, context, analysis)` | `{ indices, names, records, blocks, lens, block_id }` |

| Callback arg | Runtime value | Source contract |
| --- | --- | --- |
| `container` | target block DOM container | JS renderer internals |
| `args` | `block.record.args` or `block.compare.args` | `CustomRecordSpec.args` / `CustomCompareSpec.args` |
| `analysis` | `state.data.analysis_results[lensName]` | serialized `AnalysisResult` (`global_data`, `per_record_data`) |

Example (record view):

```javascript
function renderRecord(container, args, context, analysis) {
  const lensName = "accuracy";
  const digest = context.record?.digests?.[lensName] || {};
  const perRecord = analysis?.per_record_data?.[context.record?.name]?.data || {};
  const global = analysis?.global_data || {};
  container.textContent = `${args.title}: mse_max=${perRecord.max_mse ?? "n/a"}`;
}
```

### 10.7 Graph Pipeline (Observatory -> Viewer)

| Step | Python/JS API | Payload shape | Notes |
| --- | --- | --- | --- |
| Base graph capture | `GraphLens.observe` | `{ graph_ref, base, meta }` | `base` comes from `FXGraphExporter.generate_json_payload()["base"]` |
| Base graph registration | `GraphHub.register_asset(graph_ref, base, meta)` | `graph_assets[graph_ref] = { base, meta }` | one asset per record name by default |
| Layer authoring | `RecordAnalysis.add_graph_layer(key, extension)` | `RecordAnalysis.graph_layers[key]` | `extension` accepts `GraphExtensionPayload` or `GraphExtension` |
| Layer merge | `GraphHub.add_analysis_layers(graph_ref, lens_name, analysis)` | `graph_layers[graph_ref]["<lens>/<key>"]` | namespaced IDs prevent cross-lens collisions |
| Viewer payload build | `buildViewerPayload(graphRef)` in `01_utils.js` | `{ base, extensions }` | `base <- graph_assets`, `extensions <- graph_layers` |
| Viewer mount | `FXGraphViewer.create({ payload, mount, layout, state })` | viewer instance | called from `mountGraphViewer` |
| Compare mount | `FXGraphCompare.create({ viewers, layout, sync })` | compare controller | called from `renderGraphCompare` |

### 10.8 fx_viewer Type Bridge (Python Side)

| Observatory usage point | fx_viewer type/API | Field-level mapping |
| --- | --- | --- |
| Base graph export | `FXGraphExporter.generate_json_payload()` | uses `.base.legend/nodes/edges` for `graph_assets[graph_ref].base` |
| Layer helper | `GraphExtension(id, name)` | accumulates `nodes_data[node_id]` then `build_payload()` |
| Stable layer payload | `GraphExtensionPayload` | `id`, `name`, `legend`, `nodes[node_id]` |
| Per-node layer payload | `GraphExtensionNodePayload` | `info`, `tooltip`, `label_append`, `fill_color` |
| Observatory conversion | `GraphLayerContribution.to_payload()` | converts `GraphExtension` to `GraphExtensionPayload` and applies overrides |

### 10.9 fx_viewer Runtime API Used by Observatory (JS Side)

| Runtime API | Called from observatory JS | Purpose |
| --- | --- | --- |
| `FXGraphViewer.create(config)` | `mountGraphViewer` | mount single graph view |
| `viewer.init()` | `mountGraphViewer` | initialize renderer/UI |
| `viewer.setLayout(patch)` | `mountGraphViewer` | hide sidebar in compact compare layouts |
| `viewer.setUIVisibility(flags)` | `mountGraphViewer` | hide minimap toggle in compare |
| `viewer.setLayers(layerIds)` | `ObservatoryAPI` graph handle | switch active extension layers |
| `viewer.setColorBy(layerId)` | `ObservatoryAPI` graph handle | set color source layer |
| `viewer.patchLayerNodes(layerId, patch)` | `ObservatoryAPI` graph handle | patch node style/info in a layer |
| `viewer.selectNode(nodeId, opts)` | `ObservatoryAPI` + delegated actions | focus node |
| `viewer.zoomToFit()` | `ObservatoryAPI` | reset camera to graph bounds |
| `viewer.enterFullscreen()/exitFullscreen()` | `ObservatoryAPI` | fullscreen control |
| `viewer.on('selectionchange', cb)` | `ObservatoryAPI`, `FXGraphCompare` | selection events + compare sync |
| `FXGraphCompare.create(config)` | `renderGraphCompare` | multi-view synchronization |
| `compare.setSync(patch)` | compare sync checkbox handler | toggle selection sync on/off |

### 10.10 Graph Layer Naming and Selection Rules

| Rule | Contract | Example |
| --- | --- | --- |
| Author key (lens analyze code) | free-form local key in `RecordAnalysis.graph_layers` | `"error"` |
| Report-level namespaced key | `"<lens_name>/<layer_key>"` | `"accuracy/error"` |
| Graph block default layers | `GraphRecordSpec.default_layers: list[str]` | `default_layers=["accuracy/error"]` |
| Graph block default color | `GraphRecordSpec.default_color_by: str | None` | `default_color_by="accuracy/error"` |
| Compare viewer options merge | `Object.assign({}, record.viewer_options, compare.viewer_options_compare)` | compare options override record defaults without mutating record options |

### 10.11 Python-to-JS Dataflow Cheat Sheet

| You define in Python | Appears in report | Read in JS callback |
| --- | --- | --- |
| `CustomRecordSpec.args` | `block.record.args` | callback `args` |
| `CustomCompareSpec.args` | `block.compare.args` | callback `args` (compare mode) |
| `Frontend.record(..., context={"index","name"})` | selection metadata + serialized record list | callback `context.index`, `context.record` |
| `AnalysisResult.global_data` | `analysis_results[lens].global_data` | callback `analysis.global_data` |
| `AnalysisResult.per_record_data[name].data` | `analysis_results[lens].per_record_data[name].data` | callback `analysis.per_record_data?.[context.record.name]?.data` |


## 11. Embedded References (Single Source)

This README now embeds the former standalone references so contributors can
review runtime behavior and API contracts in one place.

### 11.1 Architecture Reference


Observatory is a whitebox debugging runtime for ExecuTorch compilation and execution flows.
This implementation is intentionally graph-native and typed.

#### 1. Core Principles

1. Strict contracts over implicit dicts.
2. Runtime capture separated from offline analysis.
3. Graph assets shared by reference via `graph_ref`.
4. Graph overlays produced in `analyze()` and merged centrally.
5. UI runtime split into topic JS modules for reviewability.

#### 2. Four-Phase Lifecycle

##### Phase 1: Runtime Capture

1. User enters `Observatory.enable_context(...)`.
2. Lenses run `observe()` and `digest()` during `collect(name, artifact)`.
3. Output is persisted as `RecordDigest`.

##### Phase 2: Session Hooks

1. Outermost context entry triggers `on_session_start`.
2. Outermost context exit triggers `on_session_end`.
3. ETRecord monkey-patch auto-collection is installed/uninstalled on outermost boundaries.

##### Phase 3: Analysis

1. Each lens runs `analyze(records, config)`.
2. Global results go to `AnalysisResult.global_data`.
3. Per-record results go to `AnalysisResult.per_record_data[record_name]` as `RecordAnalysis`.
4. Graph overlay contributions are attached in `RecordAnalysis.graph_layers`.

##### Phase 4: Report Assembly + Rendering

1. Frontend blocks are produced from typed `ViewList` contracts.
2. `GraphHub` merges base assets and analysis-time graph overlays.
3. Report payload is exported to JSON and HTML.

#### 3. Graph-Native Runtime Model

##### 3.1 Graph Asset Source

1. `GraphLens` builds one canonical fx_viewer base payload per record.
2. Payload stored in report-level `graph_assets[graph_ref]`.

##### 3.2 Graph Overlay Source

1. Lenses attach layers in `analyze()` per record using `RecordAnalysis.graph_layers`.
2. Each layer uses typed fx_viewer payloads (`GraphExtensionPayload`) or `GraphExtension` authoring helper.
3. `GraphHub` namespaces internal layer IDs as `<lens_name>/<layer_key>`.

##### 3.3 GraphView Consumption

1. `GraphBlock.record.graph_ref` resolves base graph.
2. `graph_layers[graph_ref]` provides merged overlay layers.
3. Compare mode renders side-by-side viewers with optional selection sync.

#### 4. UI Runtime Topology

JS modules under `templates/js`:

1. `00_state.js`: report state bootstrap.
2. `01_utils.js`: utility + viewer payload helpers.
3. `02_layout.js`: header/sidebar/index rendering.
4. `03_blocks.js`: block renderers + compare behavior.
5. `04_actions.js`: navigation/selection/theme actions.
6. `05_bootstrap_api.js`: app init + `window.ObservatoryAPI`.

#### 5. Auto-Collection Architecture (ETRecord)

1. `enable_context` installs ETRecord wrappers.
2. Wrapped ETRecord calls invoke `Observatory.collect(...)` transparently.
3. No manual observe points are required for ETRecord graph capture.
4. Wrappers are restored on outermost context exit.

#### 6. Breaking API Policy

This observatory path is intentionally breaking:

1. Frontend methods must return typed `ViewList` block contracts.
2. Analyze-time graph layers use typed dataclasses, not raw dict hooks.
3. Legacy compatibility shims are intentionally not maintained.

### 11.2 Python API Reference


#### 1. Entry Point

Import:

```python
from executorch.backends.qualcomm.debugger.observatory import Observatory
```

#### 2. Session Lifecycle

##### `Observatory.enable_context(config: dict | None = None)`

Context manager enabling collection.

Behavior:

1. Registers default lenses lazily on first use.
2. Installs ETRecord auto-collection wrappers on outermost entry.
3. Calls `on_session_start` hooks on outermost entry.
4. Calls `on_session_end` hooks on outermost exit.
5. Uninstalls ETRecord wrappers on outermost exit.

Example:

```python
with Observatory.enable_context(config={"profiling": {"enabled": True}}):
    ...
```

##### Nested config behavior

1. Configs are shallow-merged by key.
2. Nested dict values are merged per top-level lens key.
3. Inner context values override outer context values.

#### 3. Capture APIs

##### `Observatory.collect(name: str, artifact: Any) -> None`

Captures one record across all registered lenses.

Behavior:

1. No-op if context disabled.
2. Populates `ObservationContext.shared_state["record_name"]`.
3. Executes each lens `observe -> digest` pipeline.
4. Stores `RecordDigest` keyed by `name`.

##### `Observatory.ignore_graphs(names: list[str]) -> None`

1. Marks matching names ignored for future collect calls.
2. Removes existing records with matching names.

##### `Observatory.list_collected() -> list[str]`

Returns all collected record names.

##### `Observatory.get(name: str) -> RecordDigest | None`

Returns one collected record by name.

#### 4. Lens Registration and Reset

##### `Observatory.register_lens(lens_cls)`

Registers a custom lens and runs `lens_cls.setup()` once.

##### `Observatory.clear() -> None`

1. Clears all records.
2. Clears session data.
3. Uninstalls ETRecord wrappers.
4. Calls `clear()` on every registered lens.

#### 5. Export APIs

##### `Observatory.export_html_report(output_path, title="Observatory Report", config=None)`

Builds analysis + frontend payload and emits interactive HTML.

##### `Observatory.export_json(output_path)`

Exports raw records + session payload only.

##### `Observatory.generate_html_from_json(json_path, html_path, title="Observatory Report", config=None)`

Reconstructs HTML report from exported raw JSON and current lens frontend/analyze logic.

#### 6. Minimal End-to-End Example

```python
import torch
from executorch.backends.qualcomm.debugger.observatory import Observatory

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.fc(x)

model = M().eval()
graph = torch.fx.symbolic_trace(model)

Observatory.clear()
with Observatory.enable_context():
    Observatory.collect("step_0", graph)

Observatory.export_html_report("/tmp/observatory_report.html")
```

#### 7. Notes

1. This observatory path is breaking-by-design and does not support legacy dict block contracts.
2. Frontend returns must use typed `ViewList` dataclass API.
3. Graph layers must be attached via `AnalysisResult.per_record_data[record].graph_layers`.

### 11.3 Interface Reference


This document defines strict dataclass contracts used by observatory lenses and UI rendering.

Source of truth:

- `backends/qualcomm/debugger/observatory/interfaces.py`

#### 1. Frontend Block Contracts

All frontend methods return:

```python
ViewList(blocks=[...])
```

where each block is one of:

1. `TableBlock`
2. `HtmlBlock`
3. `CustomBlock`
4. `GraphBlock`

##### 1.1 TableBlock

Fields:

1. base: `id`, `title`, `order`, `collapsible`
2. `record: TableRecordSpec(data: dict[str, Serializable])`
3. `compare: TableCompareSpec(mode: "auto" | "disabled")`

##### 1.2 HtmlBlock

Fields:

1. base: `id`, `title`, `order`, `collapsible`
2. `record: HtmlRecordSpec(content: str)`
3. `compare: HtmlCompareSpec(mode: "auto" | "disabled")`

##### 1.3 CustomBlock

Fields:

1. base: `id`, `title`, `order`, `collapsible`
2. `record: CustomRecordSpec(js_func: str, args: dict)`
3. `compare: CustomCompareSpec(mode: "custom" | "disabled", js_func: str | None, args: dict)`

Rules:

1. `record.js_func` must be non-empty.
2. `compare.mode == "custom"` uses `compare.js_func` if set; otherwise falls back to `record.js_func`.

##### 1.4 GraphBlock

Fields:

1. base: `id`, `title`, `order`, `collapsible`
2. `record: GraphRecordSpec`
3. `compare: GraphCompareSpec`

`GraphRecordSpec` fields:

1. `graph_ref: str` (required)
2. `default_layers: list[str]`
3. `default_color_by: str | None`
4. `layer_scope: "all" | "lens_only" | list[str]`
5. `viewer_options: dict`
6. `controls: dict`
7. `fullscreen: dict`

`GraphCompareSpec` fields:

1. `mode: "auto" | "disabled" | "custom"`
2. `max_parallel: int >= 1`
3. `sync_toggle: bool`
4. `viewer_options_compare: dict`
5. `js_func: str | None`
6. `args: dict`

#### 2. Validation API

Utilities:

1. `validate_view_block(block)`
2. `validate_view_list(view_list)`

Validation checks:

1. Non-empty block id/title.
2. Unique block ids in one `ViewList`.
3. CustomBlock function requirements.
4. GraphBlock `graph_ref` and `max_parallel` constraints.

#### 3. Analysis Contracts

##### 3.1 `AnalysisResult`

Fields:

1. `global_data: dict[str, Serializable]`
2. `per_record_data: dict[str, RecordAnalysis]`

##### 3.2 `RecordAnalysis`

Fields:

1. `data: dict[str, Serializable]`
2. `graph_layers: dict[str, GraphLayerContribution]`

##### 3.3 `GraphLayerContribution`

Fields:

1. `extension`: `GraphExtensionPayload | GraphExtension`
2. `id_override: str | None`
3. `name_override: str | None`

Method:

1. `to_payload() -> GraphExtensionPayload`

Notes:

1. `GraphExtensionPayload` is preferred for stable serialization.
2. `GraphExtension` is accepted as authoring helper and converted lazily.

#### 4. Runtime Core Contracts

##### 4.1 ObservationContext

1. `config: dict`
2. `shared_state: dict`

##### 4.2 RecordDigest

1. `name: str`
2. `timestamp: float`
3. `data: dict[str, Serializable]`

##### 4.3 SessionResult

1. `start_data: dict`
2. `end_data: dict`

#### 5. Lens Protocol Summary

Each lens may implement:

1. `setup()`
2. `on_session_start(context)`
3. `observe(artifact, context)`
4. `digest(observation, context)`
5. `on_session_end(context)`
6. `clear()`
7. `analyze(records, config) -> AnalysisResult`
8. `get_frontend_spec() -> Frontend`

No separate `contribute_graph_layers` hook is used in this architecture.
Graph layers are contributed through `analyze()` via typed `RecordAnalysis`.

### 11.4 JavaScript API Reference


This document describes the host/runtime JS contracts exposed by observatory reports.

#### 1. CustomBlock JS Contract

`CustomBlock.record.js_func` signature:

```javascript
function renderRecord(container, args, context, analysis) {
  // container: HTMLElement
  // args: static serializable args from Python
  // context: { index, record }
  // analysis: { global_data, per_record_data }
}
```

`CustomBlock.compare` with `mode="custom"` signature:

```javascript
function renderCompare(container, args, context, analysis) {
  // container: HTMLElement
  // args: static compare args
  // context: {
  //   indices: number[],
  //   names: string[],
  //   records: object[],
  //   blocks: object[],
  //   lens: string,
  //   block_id: string,
  // }
  // analysis: { global_data, per_record_data }
}
```

Behavior:

1. If `compare.js_func` is set, it is used.
2. If `compare.js_func` is not set, runtime falls back to `record.js_func`.

#### 2. `window.ObservatoryAPI`

Defined in `templates/js/05_bootstrap_api.js`.

##### 2.1 `mountGraph(container, graphRef, options)`

Mount graph viewer into host container.

Arguments:

1. `container`: selector string or `HTMLElement`.
2. `graphRef`: key into report `graph_assets`.
3. `options`:
   - `default_layers`
   - `default_color_by`
   - `viewer_options`

Returns `GraphHandle` with methods:

1. `setLayers(layerIds)`
2. `setColorBy(layerId)`
3. `updateLayerNodeStyle(layerId, nodeId, patch)`
4. `selectNode(nodeId, opts)`
5. `zoomToFit()`
6. `setSyncEnabled(enabled)`
7. `enterFullscreen()`
8. `exitFullscreen()`
9. `onNodeSelected(callback)`
10. `destroy()`

##### 2.2 Navigation helpers

1. `selectRecord(index)`
2. `openCompare(indices)`
3. `showSingleRecord(index)`

##### 2.3 Utility helpers

1. `showToast(message, type)`
2. `getContext()`

#### 3. Delegated HTML Actions

Supported action attributes:

1. `data-ob-action="select-record" data-ob-record="N"`
2. `data-ob-action="open-compare" data-ob-indices="A,B"`
3. `data-ob-action="graph-focus-node" data-ob-node-id="node_name"`

#### 4. Minimal Example

```html
<div id="graph-slot" style="height: 400px"></div>
<button data-ob-action="select-record" data-ob-record="1">Go Record 1</button>
```

```javascript
const handle = window.ObservatoryAPI.mountGraph('#graph-slot', 'record_1', {
  default_layers: ['accuracy/error'],
  default_color_by: 'accuracy/error',
});

handle.zoomToFit();
```

### 11.5 Lens-to-GraphHub Guide


This guide explains how lenses contribute graph overlays through `analyze()`.

#### 1. Architectural Rule

Graph layers are derived data and must be attached in analysis results.

Do:

1. Build per-record graph overlays in `analyze()`.
2. Store them in `RecordAnalysis.graph_layers`.

Do not:

1. Use separate runtime hooks for layer contribution.
2. Return raw dict layer payloads in user-facing lens APIs.

#### 2. Preferred Payload Types

Use fx_viewer typed payload API:

1. `GraphExtensionPayload` (preferred persisted form)
2. `GraphExtension` (authoring helper, converted lazily)

Imports:

```python
from executorch.backends.qualcomm.utils.fx_viewer import (
    GraphExtension,
    GraphExtensionPayload,
    GraphExtensionNodePayload,
)
```

#### 3. Pattern A: Build payload directly

```python
from executorch.backends.qualcomm.debugger.observatory.interfaces import (
    AnalysisResult,
    RecordAnalysis,
)
from executorch.backends.qualcomm.utils.fx_viewer import (
    GraphExtensionPayload,
    GraphExtensionNodePayload,
)

@staticmethod
def analyze(records, config):
    per_record = {}
    for record in records:
        payload = GraphExtensionPayload(
            id="error",
            name="Accuracy Error",
            legend=[{"label": "Low", "color": "#93c5fd"}, {"label": "High", "color": "#b91c1c"}],
            nodes={
                "node_0": GraphExtensionNodePayload(
                    info={"mse": 0.12},
                    label_append=["mse=0.12"],
                    fill_color="#b91c1c",
                )
            },
        )

        analysis = RecordAnalysis(data={"max_mse": 0.12})
        analysis.add_graph_layer("error", payload)
        per_record[record.name] = analysis

    return AnalysisResult(per_record_data=per_record)
```

#### 4. Pattern B: Use GraphExtension helper

```python
from executorch.backends.qualcomm.utils.fx_viewer import GraphExtension, NumericColorRule

ext = GraphExtension(id="latency", name="Layer Latency")
ext.add_node_data("node_0", {"latency_ms": 1.23})
ext.set_color_rule(NumericColorRule(attribute="latency_ms"))

analysis = RecordAnalysis(data={"p95_ms": 1.23})
analysis.add_graph_layer("latency", ext)
```

`GraphHub` converts `GraphExtension` to `GraphExtensionPayload` internally.

#### 5. Layer ID Policy

User-facing key in `RecordAnalysis`:

1. `graph_layers["error"] = ...`

Internal report layer ID by GraphHub:

1. `<lens_name>/<layer_key>`
2. Example: `accuracy/error`

This keeps lens APIs free from hardcoded namespacing rules.

#### 6. How GraphHub Resolves Target Graph

1. Graph assets are registered by `graph_ref` from graph digest.
2. During payload assembly, observatory reads each record's `RecordAnalysis`.
3. `GraphHub.add_analysis_layers(graph_ref, lens_name, record_analysis)` merges layers for that graph.

#### 7. Frontend Binding

`GraphBlock.record.graph_ref` selects which graph asset/layers to render.

Example:

```python
GraphView(
    id="acc_graph",
    title="Accuracy Graph",
    graph_ref="Candidate FakeQuant",
    default_layers=["accuracy/error"],
    default_color_by="accuracy/error",
)
```

### 11.6 UI Testcases

See:
1. `examples/OBSERVATORY_UI_TESTCASES.md`
