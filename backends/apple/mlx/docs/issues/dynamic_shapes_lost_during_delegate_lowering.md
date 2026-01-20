# Dynamic Shapes Lost During Delegate Lowering

**Component**: ExecuTorch Backend Infrastructure
**Affects**: All delegates (MLX, MPS, QNN, etc.)
**Severity**: High - Blocks dynamic shape support in delegates

## Summary

When ExecuTorch lowers a subgraph to a delegate backend, symbolic shapes (SymInts) are replaced with their concrete example values. This prevents delegates from supporting dynamic shapes, even when the delegate's runtime fully supports them.

## Reproduction

```python
import torch
from torch.export import Dim, export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.apple.mlx import MLXPartitioner

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)

    def forward(self, x):
        seq_len = x.size(1)
        batch = x.size(0)
        x = x.view(batch * seq_len, 64)
        x = self.linear(x)
        x = x.view(batch, seq_len, 64)
        return x

model = SimpleModel()
seq_len = Dim('seq_len', min=1, max=256)
dynamic_shapes = {'x': {1: seq_len}}

# Export with dynamic shapes
ep = export(model, (torch.randn(1, 3, 64),), dynamic_shapes=dynamic_shapes)
print("After export:")
print(f"  Range constraints: {ep.range_constraints}")
# Output: Range constraints: {s27: VR[1, 256]}

# Lower to edge with delegate
edge = to_edge_transform_and_lower(ep, partitioner=[MLXPartitioner()])
final_ep = edge.exported_program()
print("\nAfter delegate lowering:")
print(f"  Range constraints: {final_ep.range_constraints}")
# Output: Range constraints: {3: VR[3, 3]}  <-- PROBLEM: Dynamic shape lost!
```

## Expected Behavior

After delegate lowering, the range constraints should still be `{s27: VR[1, 256]}`, and the delegate subgraph should receive symbolic shapes that can be resolved at runtime.

## Actual Behavior

After delegate lowering:
- Range constraints become `{3: VR[3, 3]}` (fixed to the example value)
- The delegate's `preprocess()` receives a subgraph with concrete shapes
- `sym_size` nodes are not included in the delegate subgraph
- View/reshape operations have hardcoded shape values instead of symbolic references

## Root Cause Analysis

The issue occurs in the delegate subgraph extraction process:

### 1. Partitioning Works Correctly

When all nodes are supported by the delegate, the partitioner correctly includes `sym_size` in the partition along with its users. The partition node list shows symbolic references preserved:
```
Partition 0:
  sym_size: aten.sym_size.int
  aten_view_copy_default: args=(x, [sym_size, 64])  # Symbolic!
```

### 2. `fuse_as_graphmodule` Works Correctly

The PyTorch `fuse_as_graphmodule` function correctly preserves symbolic references in the fused subgraph:
```python
def forward(self, x, p_linear_weight, p_linear_bias):
    sym_size = torch.ops.aten.sym_size.int(x, 1)
    aten_view_copy_default = view_copy(x, [sym_size, 64])  # Still symbolic!
```

### 3. `create_exported_program_from_submodule` Breaks It

The bug is in `create_exported_program_from_submodule()` in `lowered_backend_module.py`. When creating the `ExportedProgram` from the fused submodule, symbolic values are concretized:
```python
# What preprocess() receives:
range_constraints: {3: VR[3, 3]}  # Should be {s27: VR[1, 256]}
args=(x, [3, 64])                  # Should be (x, [sym_size, 64])
```

### 3. Where Concretization Happens

In the subgraph extraction, node arguments that reference nodes outside the subgraph get their values evaluated. For symbolic values, this means:
```python
# Original graph has:
#   sym_size_1 = aten.sym_size.int(x, 1)  # returns SymInt s27
#   view = aten.view(x, [sym_size_1, 64])  # uses symbolic s27

# After subgraph extraction (if sym_size is outside):
#   view = aten.view(x, [3, 64])  # s27 evaluated to concrete value 3
```

### 4. Evidence from Debugging

When tracing through the MLX builder with debug prints:

```
# Before Edge lowering (in ops_to_not_decompose check):
[DEBUG view_handler] shape=[Slot(SymInt), 64]  # Symbolic!

# After Edge lowering (in actual preprocess):
[DEBUG view_handler] shape=[3, 64]  # Concrete!
```

## Files Involved

- `/executorch/exir/lowered_backend_module.py`
  - `create_exported_program_from_submodule()` - Creates the delegate subgraph
  - `create_submodule_from_nodes()` - Extracts nodes into a submodule

- `/executorch/exir/backend/backend_api.py`
  - `_partition_and_lower_one_graph_module()` - Orchestrates partitioning
  - `to_backend()` - Calls `preprocess()` with the subgraph

- `torch/fx/passes/utils/fuser_utils.py` (PyTorch core)
  - `fuse_as_graphmodule()` - Creates the submodule, evaluates external references

## Proposed Solutions

### Option 1: Include `sym_size` Nodes in Partitions

Modify the partitioning logic to automatically include `sym_size` nodes when any of their users are in the partition.

**Pros**: Minimal changes, preserves existing flow
**Cons**: May include unnecessary nodes, doesn't handle all symbolic expression cases

### Option 2: Pass Symbolic Values as Subgraph Inputs

When creating the delegate subgraph, symbolic values that are used within the subgraph should become inputs to the subgraph (as SymInt inputs).

```python
# Current: sym_size output is outside subgraph, gets concretized
# Proposed: sym_size value becomes a SymInt input to the subgraph

# Delegate subgraph would have:
#   def forward(self, x: Tensor, s0: SymInt):
#       view = aten.view(x, [s0, 64])
```

**Pros**: Clean solution, works for all symbolic expressions
**Cons**: Requires changes to subgraph signature generation

### Option 3: Preserve Symbolic Expressions in Subgraph

Modify `fuse_as_graphmodule` to preserve symbolic expressions instead of evaluating them to concrete values when extracting subgraphs.

**Pros**: Most complete solution
**Cons**: Requires changes to PyTorch core

## Workarounds

Currently, there is no workaround that preserves dynamic shapes in delegates. Users must either:
1. Use static shapes (limiting flexibility)
2. Keep dynamic operations on CPU (limiting performance)

## Impact

This issue blocks:
- LLM inference with variable sequence lengths in delegates
- Any model with batch size flexibility in delegates
- KV cache implementations that need dynamic position indexing

## Related Issues

- Similar issues may exist in other backends (MPS, QNN, XNNPACK)
- The XNNPACK delegate appears to have some dynamic shape support, which may provide a reference implementation

## Test Case

A minimal test case is included in the reproduction section above. To run:

```bash
conda activate et-mlx
python -c "
# ... (paste reproduction code)
"
```

The test passes if `range_constraints` after lowering still contains symbolic dimensions (e.g., `{s27: VR[1, 256]}`) instead of concrete values (e.g., `{3: VR[3, 3]}`).
