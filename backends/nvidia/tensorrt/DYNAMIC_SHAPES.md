# TRT dynamic shapes — progress & next steps

## Completed

### 1. expand_copy / expand converter (shape tensor API)

**Files:** `converters/expand.py`

The converter now has a two-path branch: static (all dims concrete) and
dynamic (any `-1`).  The dynamic path uses `add_shape` → `add_gather` →
`add_concatenation` → `set_input(2, shape_tensor)` so TRT can prove all
dims are positive through the optimization profile.  Both `aten.expand.default`
and `aten.expand_copy.default` are marked `supports_dynamic_shapes=True`.

### 2. constant_pad_nd converter

**Files:** `converters/comparison.py`

Same pattern: `add_shape` + `add_elementwise(SUM)` to compute
`output_shape = input_shape + pad_offset` at runtime.  Marked
`supports_dynamic_shapes=True`.

### 3. view / view_copy converter (shape tensor API)

**Files:** `converters/reshape.py`

The multi-dynamic-dim path builds a shape tensor from `target_shape` args,
resolving FX Nodes through `input_map`.  Marked `supports_dynamic_shapes=True`.

### 4. slice / slice_copy converter

**Files:** `converters/slice.py`

Marked `supports_dynamic_shapes=True` (dynamic path was already committed).

### 5. Partitioner gating

**Files:** `partitioner/operator_support.py`

Two new checks in `is_node_supported`:

- **Symbolic scalar args check:** If any non-tensor arg is a symbolic FX
  Node and the converter hasn't declared `supports_dynamic_shapes=True`,
  reject the node.  This prevents static converters from receiving
  runtime-computed values they can't handle.

- **Stale reshape check (`_is_stale_reshape`):** Rejects view/reshape
  nodes where the output metadata has SymInt dims but the `target_shape`
  arg is all concrete ints.  These concrete ints were captured at trace
  time and won't adapt to different input sizes, causing TRT volume
  mismatches at non-trace-time profile points.

### 6. Optimization profiles

**Files:** `backend.py`

- `_add_network_inputs` converts SymInt dims to `-1` for TRT.
- `_eval_symint_range` evaluates derived SymInt expressions (e.g.
  `s77 + 3`, `s77 // 2`) at min/max bounds via sympy substitution.
- `_add_optimization_profile` creates a TRT optimization profile from the
  edge program's `range_constraints`, using `set_shape` for regular
  tensor inputs and `set_shape_input` for shape tensor inputs.
- Handles `int_oo` (unbounded ranges from `Dim.AUTO`) gracefully by
  capping at the trace-time value.
- Recovers symbol ranges after partitioning via `shape_env.var_to_val`
  matching, since SymInt expressions get concretized during partitioning.
- Uses `_eval_symint_range_from_shape_env` to resolve derived expressions
  (e.g. `s18 // 2`) by proportional scaling of the base symbol range.

### 7. unsqueeze / unsqueeze_copy converter

**Files:** `converters/reshape.py`

Applied `resolve_shape()` to map SymInts to `-1`.  When multiple dynamic
dims exist, builds a shape tensor via `add_shape` + `add_gather` (for each
kept input dim) + constant `1` at the inserted position + `add_concatenation`
→ `set_input(1, shape_tensor)`.  Marked `supports_dynamic_shapes=True`.

### 8. squeeze / squeeze_copy converter

**Files:** `converters/reshape.py`

Same pattern.  Tracks `kept_dims` (input indices not squeezed) and gathers
only those from the runtime shape.  Marked `supports_dynamic_shapes=True`.

### 9. flatten converter

**Files:** `converters/reshape.py`

Three-segment output shape: pre-range, flattened, post-range.  When the
flattened range contains dynamic dims, computes the runtime product via
iterative `add_elementwise(PROD)`.  Marked `supports_dynamic_shapes=True`.

### 10. select / select_copy converter

**Files:** `converters/reshape.py`

Two layers fixed:
- **Slice layer:** builds shape tensor with `dim` replaced by constant `1`,
  uses `set_input(2, shape_tensor)`.
- **Squeeze shuffle:** gathers all dims except `dim` from slice output,
  uses `set_input(1, shape_tensor)`.
Marked `supports_dynamic_shapes=True`.

### 11. Conv1d dynamic shape wrappers

**Files:** `converters/conv2d.py`

Conv1d operations require 3D→4D unsqueeze before and 4D→3D squeeze after
the TRT convolution.  These wrappers now use the shape tensor API when
the input has multiple dynamic dims.  Also added a shape-restoring
shuffle before convolution when the FX metadata has more concrete dims
than the TRT tensor (which happens when preceding shape-tensor-based
reshapes make all dims appear dynamic in TRT).

### 12. Tests

**Files:** `test/test_expand_dynamic.py`

15 tests covering registry flags for all dynamic-shapes converters,
partitioner acceptance for expand/unsqueeze/squeeze/select with dynamic
batch dim, and the attention mask broadcast pattern.

All 41 TRT backend tests pass.

---

## Recently completed: Conv1d & single-partition lowering

### 13. Conv1d after shape-tensor reshape (FIXED)

**Files:** `converters/conv2d.py`, `converters/comparison.py`

Two bugs fixed:

- **`constant_pad_nd` -1 arithmetic:** The pad converter computed
  `output_shape[dim] = input_shape[dim] + pad` which produced small
  positive ints (e.g. 7) when `input_shape[dim]` was -1 (dynamic).
  Downstream convolutions saw a concrete spatial dim of 7 instead of
  dynamic.  Fix: keep -1 when input dim is -1 and use per-component
  shape tensor (constants for concrete dims, gather+add for dynamic).

- **Conv restore with `zero_is_placeholder`:** When the FX metadata has
  >1 dynamic dim but some concrete dims (e.g. `[-1, 256, -1]`), the
  restore now uses `zero_is_placeholder=True` with 0 for dynamic dims
  and concrete values for known dims.  Previously it only worked for ≤1
  dynamic dim.

Also marked `aten.convolution.default` as `supports_dynamic_shapes=True`.

### 14. Stale reshape relaxation

**Files:** `partitioner/operator_support.py`

The `_is_stale_reshape` check rejected 95 `view_copy` nodes that had
dynamic output dims but all-concrete target shape args.  When ≤1 output
dim is dynamic, the converter uses `trt.Dims([-1, ...])` with concrete
values from metadata (not the stale args), so the reshape adapts
correctly.  Relaxed the check to only reject when >1 dynamic dim AND
all-concrete target args.

### 15. Scalar shape arithmetic converters

**Files:** `converters/shape_ops.py`, `partitioner/operator_support.py`

New converters for scalar operations that compute derived sequence
lengths (e.g. `(s18 - 1) // 8 + 1`):

- `aten.sym_size.int` → `add_shape` + `add_gather`
- `add` / `sub` / `mul` / `floordiv` (operator.*) → `add_elementwise`
  on int32 shape tensors

These bring the shape computation inside the TRT partition so derived
dimensions (masks, position encodings) are computed from the same source
as the encoder's internal sequence length.  All registered with
`supports_dynamic_shapes=True`.

### 16. Dynamic arange / full / repeat

**Files:** `converters/comparison.py`, `converters/expand.py`

Marked `aten.arange.start_step`, `aten.full.default`, and
`aten.repeat.default` as `supports_dynamic_shapes=True`.  Updated the
`full` converter to build a shape tensor from mixed FX Node / concrete
size args so it handles symbolic sizes from the scalar shape chain.

### 17. Stashed SymInt expressions for exact profile ranges

**Files:** `partitioner/__init__.py`, `backend.py`

After partitioning, partition-boundary placeholder nodes lose their
symbolic expressions (SymInt expr becomes a constant like 625).  The
profile code's proportional scaling approximation was off by 1 for
expressions like `(s18-1)//8+1`: at s18=161 it gave 20 instead of 21.

Fix: the partitioner stashes the original SymInt expressions in
`node.meta["trt_symint_exprs"]` before partitioning.  The profile code
checks this stash and evaluates the exact sympy expression at the base
symbol's min/max bounds via `expr.subs({s18: 161})`.

### 18. Multi-dynamic-dim view converter

**Files:** `converters/reshape.py`

The multi-dynamic-dim path in `convert_view` previously used the
`target_shape` args directly, which could contain stale trace-time
concrete ints (e.g. `2*S-1` evaluated to 1249).  Now uses `output_shape`
from metadata to decide which dims are dynamic, gathers those from input
runtime shape, and infers remaining unknowns from `input_volume /
product_of_known_dims`.

### 19. logical_and Bool/Float cast

**Files:** `converters/comparison.py`

TRT's AND requires Bool inputs.  Added casts for non-Bool operands.

### 20. slice_copy with FX Node start/end and dim adjustment (FIXED)

**Files:** `converters/slice.py`

Two issues fixed:

- **FX Node start/end args:** The position encoding slice uses start/end
  that are FX Nodes from scalar shape ops (`sub`, `sub_1`).  The
  converter now detects FX Node args, gets their TRT shape tensors from
  `input_map`, and computes slice size as `end_trt - start_trt` via
  `add_elementwise(SUB)`.  Sets start and size via `set_input(1, ...)`
  and `set_input(2, ...)`.

- **Missing `input_dim - start` for non-FX-Node slices:** When the
  sliced dim is dynamic but start is a concrete int > 0 (e.g.
  `tensor[:, :, 1:, :]`), the output size is `input_dim - start`.
  The rewritten dynamic path lost this adjustment.  Restored as
  `add_elementwise(SUB)` of gathered input dim and start constant.

- **Size clamp:** All dynamic slice output dims are clamped to
  `max(size, 1)` so TRT can prove the output is non-empty.

### 21. Cask convolution fix: bring arange/full inside the partition

**Root cause:** When arange/full/repeat are partition-boundary inputs
(outside TRT), their dynamic shapes appear in the optimization profile
as independent dimensions.  TRT's Cask optimizer entangles these with
the conv chain's dynamic dims via `BROADCAST_SIZE`, creating a complex
shape expression that fails `isOpConsistent`.  A single-layer repro of
the same conv builds fine — the bug is the shape entanglement.

**Fix:** Bring the entire shape computation chain inside the partition:
- Scalar ops (`sym_size`, `add`, `sub`, `mul`, `floordiv`) as TRT
  shape tensor operations (`add_elementwise` on int32 scalars).
- `arange`, `full`, `repeat` with `supports_dynamic_shapes=True` and
  shape-tensor-driven output sizes.

With everything inside, all dynamic dims derive from `audio_signal`
through internal TRT computation — no independent profile shapes to
entangle.

---

## Current state: Parakeet exports successfully (single partition)

```bash
python examples/models/parakeet/export_parakeet_tdt.py \
    --backend tensorrt --output-dir ./parakeet_tensorrt
```

- **0 TRT errors, 0 warnings, 0 rejections**
- **3 TRT engines** (encoder + other methods), down from 27
- **2443 MB** saved as `model.pte`
- Encoder is a single ~4000-node TRT delegate partition
- 4 user inputs: `audio_signal`, `length`, 2 lifted constants

### Previous Cask bug (resolved)

The Cask `isOpConsistent` assertion on `convolution_default_3` was NOT
(4000+ nodes) containing depthwise conv2d + dynamic shapes.  It does NOT
occur when the same conv is in a smaller partition (~200 nodes).  The
root cause was that arange inputs in the optimization profile created
`BROADCAST_SIZE` shape expressions that Cask couldn't validate.  Moving
arange inside the partition eliminated the independent profile shapes.

### Next steps

1. **Test transcription accuracy** with test audio to verify numerics
   match eager/CUDA.

2. **Remove debug logging:** Verbose TRT logging is enabled by default.
   Switch back to `trt.Logger.WARNING` for production.
