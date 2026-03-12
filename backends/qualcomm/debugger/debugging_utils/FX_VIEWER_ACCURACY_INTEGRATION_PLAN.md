# Observatory GraphView Integration Plan (Breaking API)

Date: 2026-03-12  
Status: Approved design direction for initial release  
Scope: `debugging_utils` + `fx_viewer` unified developer experience

## 1. Goal

Deliver one coherent framework where lens authors can build graph-native debugging views without duplicating graph data, and users can inspect single-record and compare-record graph behavior with minimal friction.

## 2. Non-Negotiable Decisions

1. This is a **breaking API change**. No backward compatibility layer will be maintained.
2. Graph visualization is a dedicated framework capability through `GraphHub` + `GraphLens`.
3. `graph_code` is no longer the graph host path.
4. All lens frontend rendering is based on composable `ViewBlock` objects.
5. Compare behavior is declared per `ViewBlock`.

## 3. Why This Change

Existing Observatory frontend contracts are flexible but not graph-first. They make it hard to:

1. Share one graph payload across multiple lens sections.
2. Add lens-specific graph overlays in a consistent way.
3. Compose controls + graph + HTML actions cleanly.
4. Generalize compare behavior for graph views.

The new model makes graph analysis a first-class primitive while keeping lens author ergonomics simple.

## 4. Architecture

## 4.1 GraphHub (framework-level)

GraphHub is a new report-generation subsystem that:

1. Stores canonical per-record base graph payloads.
2. Collects and merges layer contributions from lenses.
3. Serves graph assets to all views by reference (`graph_ref`).
4. Owns graph compare orchestration settings.

## 4.2 GraphLens (built-in)

GraphLens is the canonical producer of base graph payload:

1. Captures artifact graph data once per record.
2. Exports fx_viewer base payload (`base.nodes`, `base.edges`, legend).
3. Registers graph metadata (`graph_id`, node count, mappings).
4. Provides default graph section in each record view.

## 4.3 Lens layer contributions

Any lens can contribute fx_viewer layers through a hook:

```python
@classmethod
def contribute_graph_layers(
    cls,
    digest: Any,
    context: dict,
    graph_context: dict,
) -> list[dict]:
    ...
```

Contribution payload is fx_viewer extension-compatible and namespaced as `<lens>/<layer>`.

## 5. New Frontend Contract (Breaking)

## 5.1 Old contract removed

Removed behavior:

1. Returning single `Table`/`HTML`/`Custom` directly from `record()` and `compare()`.
2. Implicit framework auto-conversion from legacy return shapes.

## 5.2 New required contract

Each lens returns structured blocks.

```python
@dataclass
class ViewBlock:
    id: str
    title: str
    type: Literal["table", "html", "custom", "graph"]
    record: dict
    compare: dict
    order: int = 0
    collapsible: bool = True
```

```python
@dataclass
class ViewList:
    blocks: list[ViewBlock]
```

Each block owns its own record and compare behavior.

## 5.3 GraphView block

`type="graph"` `record` payload supports:

1. `graph_ref`
2. `layer_scope` (`all`, `lens_only`, explicit IDs)
3. `default_layers`
4. `default_color_by`
5. `viewer_options`
6. `controls` (simple declarative controls)
7. `fullscreen` settings

`compare` payload supports:

1. `mode` (`auto`, `disabled`, `custom`)
2. `max_parallel` (default 2)
3. `sync_toggle` (default true)
4. `viewer_options_compare`

## 6. Compare Rules

## 6.1 Type-based defaults

1. `table`: `mode=auto` side-by-side.
2. `html`: `mode=auto` side-by-side.
3. `custom`: default `mode=disabled`; lens opts in to `auto` or `custom`.
4. `graph`: default `mode=auto`, side-by-side graphs with sync toggle.

## 6.2 Graph compare layout profile

Default `viewer_options_compare`:

1. `layout_mode="compare_compact"`
2. `sidebar_mode="hidden"`
3. `minimap_mode="off"`
4. `info_mode="external"`

This preserves graph canvas area and keeps details in host-controlled panels.

## 7. Controls and Custom Composition

## 7.1 Simple path

Lens authors use declarative `controls` in `GraphView.record`.

1. Controls are local to each pane.
2. No cross-pane control synchronization in v1.

## 7.2 Advanced path

Lens authors use `custom` blocks plus JS SDK to wire arbitrary UI and graph behavior.

Recommended composition:

1. `ViewBlock(type="custom")` for sliders/actions.
2. `ViewBlock(type="graph")` for shared graph rendering.

## 8. Observatory JS SDK

Expose `window.ObservatoryAPI`:

1. `mountGraph(container, graphRef, options) -> GraphHandle`
2. `selectRecord(index)`
3. `openCompare(indices)`
4. `showSingleRecord(index)`
5. `showToast(message, type)`
6. `getContext()`

`GraphHandle`:

1. `setLayers(layerIds)`
2. `setColorBy(layerId)`
3. `updateLayerNodeStyle(layerId, nodeId, patch)`
4. `selectNode(nodeId, opts)`
5. `zoomToFit()`
6. `setSyncEnabled(bool)`
7. `enterFullscreen()`
8. `exitFullscreen()`
9. `onNodeSelected(callback)`

## 9. HTML Action Integration

Framework supports delegated action attributes so lens authors can create jumpable HTML without internal template knowledge.

Supported actions:

1. `data-ob-action="select-record" data-ob-record="N"`
2. `data-ob-action="open-compare" data-ob-indices="A,B"`
3. `data-ob-action="graph-focus-node" data-ob-node-id="node_name"`

## 10. Accuracy + Graph Story

Accuracy lens extends digest with:

1. Global metrics (`top_1`, `psnr`, etc.).
2. `layer_metrics` keyed by node identity.
3. `top_k_worst` summaries.

Implementation path:

1. Use `IntermediateOutputCapturer` for candidate and reference paths.
2. Align by debug handle.
3. Resolve to node IDs.
4. Emit one or more graph layers (`accuracy/error`, `accuracy/cosine`, flags).

## 11. Payload Shape

```json
{
  "graph_assets": {
    "record_0": {"base": {}, "meta": {"record_name": "step_0"}}
  },
  "graph_layers": {
    "record_0": {
      "accuracy/error": {"name": "Accuracy Error", "legend": [], "nodes": {}},
      "profiling/latency": {"name": "Latency", "legend": [], "nodes": {}}
    }
  },
  "records": [
    {
      "name": "step_0",
      "views": {
        "accuracy": {
          "blocks": [
            {"id": "acc_summary", "type": "table", "record": {}, "compare": {"mode": "auto"}},
            {"id": "acc_graph", "type": "graph", "record": {"graph_ref": "record_0"}, "compare": {"mode": "auto", "max_parallel": 2, "sync_toggle": true}}
          ]
        }
      }
    }
  ]
}
```

## 12. Implementation Plan

## Phase A: Core model

1. Add `GraphHub` internals to report payload generation.
2. Add `GraphLens` and graph asset registration.
3. Add layer contribution hook to `Lens` base protocol.

## Phase B: Frontend contract migration (breaking)

1. Replace old frontend return handling with `ViewBlock`/`ViewList` contract.
2. Update built-in lenses to new return model.
3. Remove legacy parsing paths from renderer.

## Phase C: Graph runtime

1. Implement graph block renderer using fx_viewer payload references.
2. Add compare auto-render for graph blocks.
3. Add sync toggle and source-guarded selection sync.

## Phase D: SDK + actions

1. Add `ObservatoryAPI` and `GraphHandle`.
2. Add delegated HTML actions.
3. Add fullscreen expand behavior in graph blocks.

## Phase E: Accuracy integration demo

1. Extend accuracy lens to produce layer metrics.
2. Register accuracy layers via contribution hook.
3. Ship demo script + sample report.

## 13. Testing Plan

1. Unit tests for GraphHub assembly and layer merge.
2. Unit tests for ViewBlock contract validation.
3. Integration tests for per-block compare behavior.
4. JS tests for graph sync and fullscreen state retention.
5. Accuracy alignment tests (debug handle to node).
6. End-to-end report test for multi-lens composite graph usage.

## 14. Deliverables

1. Updated framework contracts (`interfaces.py`, renderer pipeline).
2. New GraphLens + GraphHub.
3. fx_viewer API updates for compare and events.
4. Built-in lens migration to ViewBlock.
5. Accuracy + graph overlay demo report.

## 15. Success Criteria

1. Lens authors can add graph overlays with minimal code.
2. Same base graph is reused across multiple sections.
3. Graph compare works automatically and predictably.
4. Slider/custom controls can drive live graph updates via SDK.
5. No legacy contract code remains in framework.

