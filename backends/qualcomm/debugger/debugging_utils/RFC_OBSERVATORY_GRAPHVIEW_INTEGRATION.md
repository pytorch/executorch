# RFC: Graph-Native Observatory with fx_viewer (Breaking API)

Date: 2026-03-12  
Status: Draft for review  
Authors: Qualcomm Executorch debugging team  
Target release: Initial public release of `debugging_utils` + `fx_viewer` integration

## 1. Abstract

This RFC proposes a breaking redesign of Observatory frontend and graph integration APIs so that graph visualization becomes a first-class capability instead of an ad-hoc per-lens customization.

The redesign introduces:

1. `GraphHub` (framework-level graph asset and layer manager).
2. `GraphLens` (canonical base graph producer per record).
3. A new composable `ViewBlock` contract for lens frontend rendering.
4. A dedicated `GraphView` block type with built-in auto compare behavior.
5. A small Observatory JS SDK for controls, actions, fullscreen, and synchronization.

No backward compatibility with old lens frontend return types will be provided.

## 2. Background (for readers new to both projects)

## 2.1 What is `debugging_utils` (Observatory)?

Observatory is a whitebox debugging framework for Executorch compilation and execution workflows. It lets developers capture intermediate artifacts at multiple pipeline points, analyze them, and render a report.

At a high level, it does four things:

1. Observe runtime artifacts (graphs, tensors, metadata).
2. Digest those artifacts into serializable record data.
3. Analyze records globally and per-step.
4. Present report views (dashboard, record, compare).

Observatory is lens-based: each lens focuses on one concern (metadata, graph code, profiling, accuracy, etc.).

## 2.2 What is `fx_viewer`?

`fx_viewer` is a standalone, dependency-light FX graph visualizer.

It provides:

1. Export pipeline from graph module to payload (`base` graph + `extensions` overlays).
2. Fast interactive viewer (canvas + minimap + search + info panel).
3. Layer toggles and color-by controls for overlays.
4. Embeddable JS API suitable for integration into custom reports.

## 2.3 Why combine them?

Observatory already has rich multi-record lifecycle and lens composition. `fx_viewer` already has rich node-level graph interaction. Combining them gives a unified debugging surface:

1. Capture pipeline events and metrics with Observatory.
2. Visualize structural and per-node details with `fx_viewer`.
3. Compare records with synchronized graph interaction.

## 3. Problem Statement

Current integration patterns are fragmented and not graph-native at framework level.

Pain points:

1. Graph payload generation is not centralized and reusable by reference.
2. Lens authors cannot contribute node overlays through a standard framework API.
3. Rendering contract is too coarse for composing controls + graph + HTML actions cleanly.
4. Compare behavior for graph-like views is inconsistent and lens-specific.
5. Reusing graph data across multiple sections often implies duplication.

## 4. Goals

1. Make graph rendering a first-class framework capability.
2. Keep one canonical base graph payload per record.
3. Let any lens contribute fx_viewer layers in a standard way.
4. Provide composable frontend blocks with explicit per-block compare behavior.
5. Enable simple advanced interactivity via a small stable JS SDK.
6. Deliver strong demo capability for per-layer accuracy + graph compare.

## 5. Non-Goals

1. Preserving old frontend return APIs.
2. Supporting every legacy renderer format during transition.
3. Cross-pane declarative control synchronization in v1.
4. Solving every large-graph performance edge case in this RFC.

## 6. Proposal

## 6.1 New framework components

### GraphHub

GraphHub is added to Observatory report assembly and is responsible for:

1. Registering base graph assets by `graph_ref`.
2. Merging layer contributions from all lenses for each record.
3. Serving graph assets/layers to all graph views by reference.

### GraphLens

GraphLens is a built-in lens that:

1. Produces canonical base graph payload once per record.
2. Adds graph metadata and index mappings for layer contributors.
3. Provides default graph section in record views.

## 6.2 New lens extension hook

All lenses may optionally provide graph layers:

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

Returned layers must follow fx_viewer extension schema and be namespaced (`<lens>/<layer>`).

## 6.3 New frontend rendering contract (breaking)

Lens frontend now returns a `ViewList` of `ViewBlock` objects only.

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

@dataclass
class ViewList:
    blocks: list[ViewBlock]
```

No old single-object return formats are accepted.

## 6.4 GraphView block

`type="graph"` supports:

1. `graph_ref`
2. `default_layers`
3. `default_color_by`
4. `layer_scope`
5. `viewer_options`
6. `controls`
7. `fullscreen`

Compare section supports:

1. `mode` (`auto`, `disabled`, `custom`)
2. `max_parallel`
3. `sync_toggle`
4. `viewer_options_compare`

## 6.5 Compare behavior

Default behavior by block type:

1. table -> auto side-by-side.
2. html -> auto side-by-side.
3. custom -> disabled by default.
4. graph -> auto graph compare with sync toggle.

Graph compare defaults to compact profile:

1. hidden sidebar,
2. minimap off,
3. external info rendering.

## 6.6 Observatory JS SDK

Expose `window.ObservatoryAPI` for lens authors:

1. `mountGraph(...)` and returned `GraphHandle`.
2. navigation helpers (`selectRecord`, `openCompare`, `showSingleRecord`).
3. utility helpers (`showToast`, `getContext`).

`GraphHandle` supports layer toggling, recoloring, node selection, sync toggling, zoom, and fullscreen.

## 6.7 HTML action attributes

Add delegated action mechanism for simple navigable HTML:

1. `select-record`
2. `open-compare`
3. `graph-focus-node`

This avoids requiring lens authors to know internal template code.

## 7. Detailed API and Payload

## 7.1 Report payload additions

```json
{
  "graph_assets": {
    "record_0": {
      "base": {"legend": [], "nodes": [], "edges": []},
      "meta": {"record_name": "step_0", "node_count": 123}
    }
  },
  "graph_layers": {
    "record_0": {
      "accuracy/error": {"name": "Accuracy Error", "legend": [], "nodes": {}},
      "profiling/latency": {"name": "Latency", "legend": [], "nodes": {}}
    }
  }
}
```

## 7.2 View block example

```json
{
  "id": "accuracy_graph",
  "title": "Accuracy Graph",
  "type": "graph",
  "record": {
    "graph_ref": "record_0",
    "default_layers": ["accuracy/error"],
    "default_color_by": "accuracy/error",
    "viewer_options": {"layout_mode": "full"}
  },
  "compare": {
    "mode": "auto",
    "max_parallel": 2,
    "sync_toggle": true,
    "viewer_options_compare": {"layout_mode": "compare_compact"}
  }
}
```

## 8. Breaking Changes

This RFC intentionally breaks API at lens frontend boundary.

### Removed

1. Returning `Table`, `HTML`, `Custom` directly from frontend methods.
2. Legacy renderer auto-normalization logic.
3. Ad-hoc graph embedding patterns in unrelated lens code.

### Required migration

1. Every built-in and custom lens migrates to `ViewList/ViewBlock`.
2. Graph rendering must use `type="graph"` block or SDK mount path.
3. Compare behavior must be declared in each block.

## 9. Alternatives Considered

## 9.1 Keep old API + add wrappers

Rejected because:

1. Preserves ambiguity in rendering semantics.
2. Creates long-term maintenance debt.
3. Delays a clean graph-native model.

## 9.2 Put graph entirely inside each lens custom JS

Rejected because:

1. Duplicates base graph payload and integration code.
2. Prevents reusable layer composition.
3. Makes compare behavior inconsistent across lenses.

## 9.3 Keep graph in `graph_code` lens

Rejected because:

1. Couples code rendering and graph rendering unnecessarily.
2. Limits extensibility and ownership boundaries.

## 10. Implementation Plan

## Phase 1: Core contracts

1. Add `ViewBlock/ViewList/GraphView` dataclasses.
2. Remove legacy return handling.
3. Update built-in lenses to new return model.

## Phase 2: GraphHub + GraphLens

1. Add graph asset registration.
2. Add layer contribution hook and merge flow.
3. Wire graph payload into report output.

## Phase 3: Frontend and compare

1. Add graph block renderer.
2. Add auto compare for graph blocks.
3. Add sync toggle and compact compare profile.

## Phase 4: SDK + actions

1. Add `ObservatoryAPI` and `GraphHandle` methods.
2. Add delegated HTML actions.
3. Add fullscreen support.

## Phase 5: Accuracy integration demo

1. Extend accuracy lens layer metrics and overlays.
2. Register accuracy layers through hook.
3. Produce demo report and docs.

## 11. Risk Analysis

1. Migration burden for existing lenses.
Mitigation: provide migration template and lint/check tooling.

2. Contract churn during initial release.
Mitigation: freeze schema after Phase 2 and add conformance tests.

3. Sync behavior bugs in compare mode.
Mitigation: source-tagged events and reentrancy guard.

4. Complexity creep in control system.
Mitigation: keep declarative controls local-only in v1.

## 12. Test Strategy

1. Unit tests for block schema validation.
2. Unit tests for GraphHub merge and namespacing.
3. Integration tests for mixed block layouts.
4. Compare-mode tests for graph sync toggle.
5. Accuracy-to-layer mapping correctness tests.
6. End-to-end smoke test that builds report and mounts graph blocks.

## 13. Rollout and Adoption

1. Land breaking contract and update all built-in lenses in one branch.
2. Add migration note for external/custom lenses.
3. Ship one official example lens using `custom + graph` composition.
4. Ship one official per-layer accuracy demo.

## 14. Open Questions

1. Should `max_parallel` for graph compare allow >2 in initial release?
2. Should compact compare show optional per-pane minimap toggle?
3. Should GraphHub expose a per-record graph cache key for incremental report regen?

## 15. Expected Outcome

After this RFC, lens developers get a simple and scalable way to build graph-driven debugging experiences, and users get an integrated report where graph analysis, metric overlays, and record comparison work coherently out of the box.

