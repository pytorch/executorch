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

## Remaining: Conv1d after shape-tensor reshape

The Conv1d converter uses FX metadata to restore concrete dims before
calling `add_convolution_nd`, but this only works when `resolve_shape`
produces ≤1 dynamic dim.  When the Conformer encoder has nodes with
multiple SymInt dims in their metadata (e.g. after view operations that
reshape with both batch and sequence as symbolic), the restore fails and
TRT cannot determine the conv output shape.

The error manifests as:
```
ITensor::getDimensions: Error Code 4: Tensor conv1d_..._output has axis 3
with inherently negative length.  Proven upper bound is -1.
```

### Root cause

When a shape-tensor-based shuffle (`set_input(1, shape_tensor)`) feeds
into a Conv1d wrapper, TRT reports all output dims as `-1`.  The Conv1d
unsqueeze creates `[-1, -1, 1, -1]` (three dynamic dims).  TRT then
cannot prove the spatial output of the conv is non-negative.

### Possible fixes

1. **Avoid shape tensor API when ≤1 dim is dynamic:** The FX metadata
   from `get_node_shape` + `resolve_shape` should produce shapes with
   concrete batch/channel dims.  If the preceding reshape uses the
   simple `trt.Dims()` path (≤1 dynamic dim), the conv input will have
   concrete dims.  Investigate why some reshape nodes produce >1 dynamic
   dim when only the sequence length is truly dynamic.

2. **Add optimization profile before network building:** If TRT had
   profile ranges available during `add_convolution_nd`, it could prove
   the spatial dim is positive.  This would require restructuring the
   build flow (currently profile is set after all layers are added).

3. **Explicit output shape:** Use the conv node's FX metadata to compute
   the expected output shape and apply it via a reshape after the conv.

### Unused shape inputs

The encoder partition has 5–7 unused scalar inputs (`sym_size`, `add_1`,
`sub`, etc.) that were pulled in as partition boundary values.  The
converters use `add_shape` instead.  TRT warns about them but builds
successfully.  To clean these up:

- Option A: post-process the network to remove inputs with zero consumers.
- Option B: filter them out in `_add_network_inputs` (requires tracking
  which placeholder nodes are actually referenced by converter code).
- Low priority — they don't block engine building.

### Verification

```bash
python examples/models/parakeet/export_parakeet_tdt.py \
    --backend tensorrt --output-dir ./parakeet_tensorrt
```

The encoder is a single TRT delegate partition.  Once the conv issue is
resolved, test transcription with 7.4s audio to verify it produces the
same text as eager/CUDA.
