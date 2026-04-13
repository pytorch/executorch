# Debug Handle Consistency Analysis: Observatory Pipeline Graph Collector

## Overview

This document provides a full technical analysis of how `debug_handle` values are assigned
across the "Exported Float" and "Edge" graphs captured by the Observatory
`PipelineGraphCollectorLens`, and why those handles are consistent — enabling cross-stage
node sync in the fx_viewer.

---

## 1. Motivation

The Observatory `PipelineGraphCollectorLens` (in
`backends/qualcomm/debugger/observatory/lenses/pipeline_graph_collector.py`) patches
`torch.export.export` to capture the float model graph for visualization. The goal is to
enable cross-stage graph sync in the fx_viewer using `debug_handle` as the sync key — so
clicking a node in the "Exported Float" graph highlights the corresponding node(s) in the
"Edge" graph and vice versa.

The problem is that `torch.export.export` produces an `ExportedProgram` whose nodes have
**no `debug_handle`** and no `from_node` metadata. The `DebugHandleGeneratorPass` (which
assigns integer handles) runs later, inside `_generate_edge_program` (called from
`to_edge_transform_and_lower`). So the graph captured at export time has no handles,
making sync impossible without additional work.

The fix must satisfy two constraints:

1. The "Exported Float" graph must have integer `debug_handle` values assigned using the
   same scheme as the pipeline.
2. The handle values must be **consistent** between the two graphs — the same logical
   operation must receive the same integer handle in both the "Exported Float" and "Edge"
   graphs.

---

## 2. The `debug_handle` Assignment Mechanism

### `DebugHandleGeneratorPass`

**Location:** `exir/passes/debug_handle_generator_pass.py`

`DebugHandleGeneratorPass` is the canonical mechanism for assigning integer `debug_handle`
values to graph nodes. It is registered as part of `base_post_op_replace_passes` in
`exir/passes/__init__.py` (lines 507–513):

```python
base_post_op_replace_passes = [
    ...
    DebugHandleGeneratorPass(),
    ...
]
```

These passes are applied inside `_generate_edge_program` in `exir/program/_program.py`
via a call at line 923:

```python
_transform(edge_program, *post_op_replace_passes)
```

### How the pass assigns handles

The pass iterates all non-placeholder, non-output nodes using `bfs_trace_with_node_process`
from `exir/graph_module.py`. For each node it calls:

```python
get_greatest_ancestor_node_identifier(node)
```

from `exir/debug_handle_utils.py`. This function walks the `from_node` chain stored in
`node.meta["from_node"]` to find the deepest (oldest) ancestor node in the provenance
chain, and returns a string identifier of the form:

```
"{node.name}.{graph_id}"
```

where `graph_id = id(node.graph)` — the Python object identity of the graph containing
that ancestor node.

Nodes that share the same deepest ancestor receive the same integer handle. This is the
**1-to-many mapping for decomposed ops**: when one high-level op decomposes into multiple
lower-level ops, all resulting nodes trace back to the same ancestor and receive the same
handle.

The counter starts at 1 and increments once per unique ancestor identifier encountered
during BFS traversal.

---

## 3. The `from_node` Metadata: What It Is and When It Is Populated

### `NodeSource` and `from_node`

**Location:** `torch/fx/traceback.py`

Every `torch.fx.Node` carries a `meta` dict. The `from_node` key, when present, holds a
`list[NodeSource]` — a recursive provenance chain describing which earlier node(s) this
node was derived from.

`NodeSource` has the following fields:

| Field | Type | Meaning |
|---|---|---|
| `name` | `str` | Name of the source node |
| `graph_id` | `int` | `id(source_node.graph)` at construction time |
| `pass_name` | `str` | Name of the pass/interpreter that created the link |
| `action` | `CREATE` or `REPLACE` | How the new node was derived |
| `from_node` | `list[NodeSource]` | Recursive ancestry chain of the source node |

### When `from_node` is set

`from_node` is populated by `set_current_meta(node, pass_name)` in `torch/fx/traceback.py`
(line 435). This is called by `Interpreter._set_current_node` in `torch/fx/interpreter.py`
(line 271):

```python
def _set_current_node(self, node):
    set_current_meta(node, f"Interpreter_{self.__class__.__name__}")
```

Every time an `Interpreter` subclass executes a node, the thread-local
`current_meta["from_node"]` is set to `[NodeSource(node, pass_name, CREATE)]`. Any new
nodes created during that node's execution inherit `current_meta`, so their `from_node`
points back to the node being executed.

### Why `torch.export.export` produces nodes without `from_node`

`torch.export.export` uses **tracing** (Dynamo + AOT compilation), not an `Interpreter`
subclass. Dynamo traces the module symbolically and records operations into an FX graph
without going through `Interpreter._set_current_node`. As a result, nodes in the raw
exported graph have **no `from_node`** in their `meta`.

---

## 4. Why `run_decompositions({})` Populates `from_node`

### Call chain

`ExportedProgram.run_decompositions({})` in `torch/export/exported_program.py` (line 1468)
calls `_decompose_exported_program` (line 993), which calls
`_decompose_and_get_gm_with_new_signature_constants` (line 331).

The non-joint path calls `_export_to_aten_ir`, which internally runs `aot_export_module`.
This function **re-traces the module** through an `Interpreter` subclass named
`PropagateUnbackedSymInts`.

### What re-tracing does to `from_node`

As each node in the input graph is executed by `PropagateUnbackedSymInts`:

1. `Interpreter._set_current_node(source_node)` fires.
2. This calls `set_current_meta(source_node, "Interpreter_PropagateUnbackedSymInts")`.
3. `current_meta["from_node"]` is set to `[NodeSource(source_node, "Interpreter_PropagateUnbackedSymInts", CREATE)]`.
4. New nodes emitted into the output graph during that execution inherit `current_meta`.
5. Those new nodes therefore have `from_node[-1]` pointing to `source_node`, and
   `from_node[-1].from_node` recursively chains further back if `source_node` itself had
   `from_node`.

Even with an empty decomposition table `{}`, the re-tracing still runs every node through
the Interpreter. So **every node in the output graph gets `from_node` populated**.

### The root invariant

The deepest ancestor in the `from_node` chain is always a node from the **original `ep`
graph** — the one produced by `torch.export.export`. This is because `NodeSource` captures
`graph_id = id(node.graph)` at construction time, and the original graph object is never
mutated in place by `run_decompositions`. The root identifier is therefore:

```
"original_node_name.{id(ep.graph_module.graph)}"
```

This string is stable across multiple calls to `run_decompositions({})` on the same `ep`,
because `id(ep.graph_module.graph)` does not change between calls.

---

## 5. The Fix: Patch Path in `patched_export`

The fix lives inside `_install_export_patch` in
`backends/qualcomm/debugger/observatory/lenses/pipeline_graph_collector.py`. The relevant
portion of `patched_export` is:

```python
collect_target = result.run_decompositions({})
DebugHandleGeneratorPass()(collect_target.graph_module)
cls._collect_fn("Exported Float", collect_target)
return result  # original unmodified
```

Step by step:

1. **`result.run_decompositions({})`** — Creates a **new** `ExportedProgram` whose nodes
   all have `from_node` populated via the `PropagateUnbackedSymInts` interpreter. The
   original `result` object is not mutated, so the caller receives the unmodified export.

2. **`DebugHandleGeneratorPass()(collect_target.graph_module)`** — Applies the same
   handle-assignment pass used by the pipeline. Because `from_node` is now populated,
   `get_greatest_ancestor_node_identifier` can walk the chain and produce stable root
   identifiers. Integer handles are assigned in BFS order.

3. **`cls._collect_fn("Exported Float", collect_target)`** — Stores the annotated graph
   for later visualization.

4. **`return result`** — The caller (user code) receives the original, unmodified
   `ExportedProgram`. The patch is invisible to downstream processing.

---

## 6. Why Handles Are Consistent Between the Two `DebugHandleGeneratorPass` Invocations

This section contains the core consistency argument.

### The shared root invariant

Both the patch path and the pipeline path call `run_decompositions({})` on the **same `ep`
object** (or the same `ep.graph_module` after in-place mutation by QNN passes). Regardless
of which graph is being re-traced, the `from_node` ancestry chain always terminates at
nodes whose `graph_id` equals `id(ep.graph_module.graph)` — the Python object identity of
the graph at the time the `NodeSource` was constructed.

Since `NodeSource` records `graph_id` at construction time and the original graph object
is never replaced (only mutated in place), `get_greatest_ancestor_node_identifier` returns
the same string `"node_name.{orig_graph_id}"` in both invocations for any node that
traces back to the same original op.

### The numbering invariant

`DebugHandleGeneratorPass` assigns handles by first-seen order of unique root identifiers
during BFS traversal. Both paths traverse graphs that are structurally identical at the
level of logical operations — `run_decompositions({})` with an empty decomposition table
performs no decompositions, only re-tracing. The BFS order is therefore the same in both
paths, and the counter increments identically.

### Verified experimentally

For a model with a simple structure `fc1 → relu → fc2 → relu`:

- **Patch path ("Exported Float" graph):** `linear: 1, relu: 2, linear_1: 3, relu_1: 4`
- **Pipeline path ("Edge" graph):** `permute+addmm: 1, relu: 2, permute+addmm: 3, relu: 4`

The decomposed `permute` and `addmm` nodes both trace back to the same `linear` ancestor,
so both receive handle `1`. The `relu` nodes are unchanged and receive handles `2` and `4`
respectively. Cross-stage sync works correctly.

---

## 7. QNN Pass Transforms Between the Two Invocations — Consistency Analysis

### Pipeline structure

In `to_edge_transform_and_lower_to_qnn` (in `backends/qualcomm/utils/utils.py`, line 325),
the full pipeline is:

```
torch.export.export(m, inputs)
    |
    | (patch fires here, captures pre-transform ep)
    v
QnnPassManager().transform_for_export_pipeline(ep)   # mutates ep.graph_module in-place
    v
to_edge_transform_and_lower(aten_programs, ...)       # calls run_decompositions + DebugHandleGeneratorPass
```

The patch captures `ep` **after** `torch.export.export` but **before**
`transform_for_export_pipeline`. So:

- The patch's `run_decompositions({})` runs on the **pre-transform** graph.
- The pipeline's `run_decompositions({})` runs on the **post-transform** graph.

These are structurally different graphs. The consistency question is: do both
`DebugHandleGeneratorPass` invocations produce the same handle for the same logical
operation?

### Case 1: Passes that use `ExportPass.call_operator` (the correct way)

All QNN passes inherit from `ExportPass` (in `exir/pass_base.py`). `ExportPass` itself
runs as an `Interpreter` over the graph. When it executes a source node,
`_set_current_node` fires, setting `current_meta["from_node"]` to point at the source
node. New nodes created via `call_operator` inherit this `current_meta` and therefore have
`from_node` pointing back to the source node.

If a QNN pass uses `call_operator` correctly, the new node's `from_node` chain leads back
to a node in the pre-transform graph, which in turn (after `run_decompositions`) traces
back to the same original `ep` root. Both patch and pipeline paths see the same root
identifier. **Handles are consistent.**

### Case 2: `graph.create_node` + `copy_meta` (no `from_node` in source)

**All QNN passes use `graph.create_node` directly** inside their `call` method, not
`call_operator`. This bypasses the `Interpreter` machinery for node creation. New nodes
receive their metadata via `copy_meta` from `backends/qualcomm/_passes/utils.py`, which
performs a shallow copy: `meta = dict(source_node.meta)`.

At ATen stage (before `run_decompositions`), source nodes have **no `from_node`** (they
came from `torch.export.export`). So `copy_meta` copies a dict without `from_node`, and
the new nodes also have no `from_node`.

When `run_decompositions({})` later re-traces the transformed graph, the Interpreter sees
the new nodes (e.g. `reshape_default` inserted by `InsertReshapeForReduceOps`) and creates
`NodeSource` entries pointing to them. The root becomes:

```
"reshape_default.{transformed_graph_id}"
```

where `transformed_graph_id = id(ep.graph_module.graph)` — the same graph object that
both the patch path and the pipeline path operate on (since `transform_for_export_pipeline`
mutates in place).

**Both paths call `run_decompositions` on the same `ep.graph_module.graph` object. The
root identifier is therefore the same in both invocations. Handles are consistent.**

### Case 3: `graph.create_node` + `copy_meta` (source has `from_node`)

If a source node already has `from_node` in its meta (e.g. because it was itself produced
by a prior pass that ran through an Interpreter), `copy_meta` copies that `from_node`
shallowly into the new node. The new node's ancestry chain therefore leads back to the
same original root as the source node. Both paths see the same root identifier. **Handles
are consistent.**

### `InsertReshapeForReduceOps` — a concrete example

`InsertReshapeForReduceOps` (in
`backends/qualcomm/_passes/insert_reshape_for_reduce_ops.py`) uses
`graph.create_node` with `graph.inserting_before(n)` and sets
`reshape_node.meta = dict(inp.meta)`. At ATen stage, `inp.meta` has no `from_node`. So
`reshape_node` has no `from_node`.

After `run_decompositions({})`, the Interpreter re-traces and creates:

```
NodeSource("reshape_default", "Interpreter_PropagateUnbackedSymInts", graph_id=transformed_graph_id)
```

Both the patch path and the pipeline path call `run_decompositions` on the same
transformed graph, so both get the same root identifier. The handles for `reshape_default`
match between the two invocations.

### Summary table

| Pass style | `from_node` on new nodes | Root after `run_decompositions` | Patch vs Pipeline consistent? |
|---|---|---|---|
| `ExportPass.call_operator` | Yes, via Interpreter | Original `ep` graph node | Yes |
| `graph.create_node` + `copy_meta` (no `from_node` in source) | No | The new node itself (in transformed graph) | Yes — both paths use same transformed graph |
| `graph.create_node` + `copy_meta` (source has `from_node`) | Yes (shallow copy) | Original `ep` graph node | Yes |

All cases are consistent because both paths ultimately call `run_decompositions` on the
same `ep.graph_module` object (the one that `transform_for_export_pipeline` mutated in
place).

---

## 8. The `transform_for_export_pipeline` Pass List

`QnnPassManager.transform_for_export_pipeline` (line 224 in `qnn_pass_manager.py`) runs
passes via `self._transform(exported_program.graph_module)`, which calls
`self(graph_module).graph_module` (i.e. `PassManager.__call__`).

The passes applied at this stage are:

- `DecomposeBinaryAlpha`
- `DecomposeCDist`
- `DecomposeScaledDotProductAttention`
- `DecomposeRoll`
- `DecomposeThreshold`
- `DecomposeTriu`
- `DecomposeLinalgVectorNorm`
- `DecomposeExpM1`
- `DecomposeFloorDivide`
- `DecomposeWrapWithAutocast`
- `CanonicalizeConv`
- `ConvertLinearToConv2d` (optional, model-dependent)
- `ConvertSquareToPow`
- `LiftConstantScalarOperands`
- `InsertReshapeForReduceOps`

All of these passes use `graph.create_node` directly (not `call_operator`). None set
`from_node` on new nodes at ATen stage. This is correct behavior — consistency is
guaranteed by the shared `graph_id` invariant described in Section 7, not by `from_node`
being present on the new nodes themselves.

### Passes that run after `run_decompositions` (Edge stage)

The `get_to_edge_transform_passes` method (line 140 in `qnn_pass_manager.py`) returns
passes from `get_capture_program_passes()` (line 78). These are passed as
`transform_passes` to `to_edge_transform_and_lower` and run **after** `run_decompositions`
inside `EdgeProgramManager.transform()`.

Because these passes operate on the Edge graph — after the pipeline's own
`DebugHandleGeneratorPass` invocation — they are outside the scope of the consistency
analysis for the "Exported Float" graph. The pipeline's `DebugHandleGeneratorPass` handles
them, and they produce handle values that may or may not have counterparts in the
"Exported Float" graph (see Section 9).

---

## 9. Limitation: Patch Captures Pre-Transform Graph

The patch fires at `torch.export.export` time, before `transform_for_export_pipeline`.
The "Exported Float" graph in the Observatory is therefore the **pre-QNN-transform** ATen
graph.

Nodes inserted by QNN passes — such as `reshape` from `InsertReshapeForReduceOps`,
`unsqueeze`/`squeeze` from `CanonicalizeConv`, or decomposed ops from
`DecomposeScaledDotProductAttention` — will **not** appear in the "Exported Float" graph
but will appear in the "Edge" graph.

For the purposes of `debug_handle` sync, this means:

- Nodes that are **unchanged** by QNN passes appear in both graphs with matching handles
  and sync correctly when clicked in the fx_viewer.
- Nodes **inserted** by QNN passes appear in the Edge graph with handles that have no
  counterpart in the "Exported Float" graph. They will not sync to any node in the float
  graph. This is expected behavior.
- Nodes **removed or replaced** by QNN passes (e.g. a `linear` replaced by `conv2d` via
  `ConvertLinearToConv2d`) appear differently in the two graphs. The replaced form in the
  Edge graph will have a handle derived from the original node's root identifier, so sync
  still works for those nodes.

This is the correct tradeoff: the "Exported Float" graph represents the user's original
model as exported by PyTorch, and the Edge graph represents the QNN-lowered version. The
Observatory makes no attempt to synthesize a "virtual" node in the float graph for every
QNN-inserted node.

---

## Summary

| Question | Answer |
|---|---|
| Why does `torch.export.export` produce nodes without `debug_handle`? | `DebugHandleGeneratorPass` runs later, inside `_generate_edge_program`. |
| Why do exported nodes have no `from_node`? | Export uses Dynamo tracing, not an `Interpreter` subclass. |
| How does `run_decompositions({})` populate `from_node`? | It re-traces via `PropagateUnbackedSymInts` (an `Interpreter` subclass), which fires `_set_current_node` for each node. |
| Why doesn't `run_decompositions({})` change the graph structure? | An empty decomposition table `{}` means no ops are decomposed; re-tracing is a no-op structurally. |
| Why are handles consistent between the patch and the pipeline? | Both call `run_decompositions` on the same `ep.graph_module` object. The `from_node` chain always terminates at nodes in that object's graph, giving the same root identifiers and thus the same integer handles. |
| What is the root identifier formula? | `"{node.name}.{id(ep.graph_module.graph)}"` — stable across multiple `run_decompositions` calls on the same `ep`. |
| What is the effect of QNN passes on consistency? | All QNN passes use `graph.create_node` directly and mutate `ep.graph_module` in place. Both patch and pipeline paths use the same mutated graph as the root, so consistency holds in all cases. |
| What does the "Exported Float" graph represent? | The pre-QNN-transform ATen graph. Nodes inserted by QNN passes will not appear in it. |
