# TRT Backend: Dynamic Shape Support — Debugging Progress

## Status

**Int EValue crash: FIXED.** The pybind runtime crashed (`EValue is not a Tensor`)
when TRT delegates had dynamic shapes, because ExecuTorch passes `sym_size` and
derived dimension values as Int EValues while the backend called `toTensor()`.
The fix in `TensorRTBackend::execute()` checks `args[i]->isInt()` and routes
Int args through the shape-tensor path.

**NaN outputs with dynamic shapes: FIXED.** `setInputShape` for shape tensors
was passing the tensor's *dimensions* (e.g., `{1}` for a scalar) instead of
its *values* (e.g., `{376}`). For TRT shape tensors, `setInputShape` expects
`dims.d[]` to contain the actual integer values, not the tensor shape. Fixed
in both `execute()` (runtime) and `allocate_gpu_buffers()` (pre-allocation).
The 24-layer encoder now produces max_diff ≈ 0.20 with dynamic shapes.

## What was changed

### `TensorRTBackend.cpp`
- **Input handling:** Added `|| args[i]->isInt()` fallback alongside
  `is_input_shape_tensor()`. When ExecuTorch passes an Int EValue (e.g.
  `sym_size`), we read it with `toInt()`, store as int32, and hand the
  host pointer to the executor.
- **Output handling:** Added `is_output_shape_tensor()` guards in the output
  extraction, dtype-conversion, and resize loops so shape-tensor outputs
  don't trigger `toTensor()`.

### `TensorRTExecutor.{h,cpp}`
- Added `is_output_shape_tensor(size_t output_index)` to mirror the existing
  `is_input_shape_tensor()`.

## Reproduction

### Setup (one-time)
```bash
# Build the pybind extension with the TRT backend changes:
cd /home/dev/executorch/pip-out/temp.linux-x86_64-cpython-313/cmake-out
cmake --build . --target tensorrt_backend -j$(nproc)
cmake --build . --target portable_lib -j$(nproc)
cp _portable_lib.cpython-313-x86_64-linux-gnu.so \
   /home/dev/miniconda3/envs/executorch/lib/python3.13/site-packages/executorch/extension/pybindings/
```

### Verify the Int EValue fix (should print max_diff < 0.001)
```bash
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate executorch
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python -c "
import os, torch, logging
os.environ['TRT_LOG_LEVEL'] = '3'
logging.getLogger('executorch.backends.nvidia.tensorrt').setLevel(logging.ERROR)
from torch.export import Dim, export
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass
from executorch.runtime import Runtime
from executorch.backends.nvidia.tensorrt.compile_spec import TensorRTCompileSpec, TensorRTPrecision
from executorch.backends.nvidia.tensorrt.partitioner import TensorRTPartitioner

class MatMulTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1024, 1024))
    def forward(self, x):
        return torch.matmul(x, self.weight.t())

module = MatMulTest().eval()
x = torch.randn(1, 376, 1024)
ep = export(module, (), kwargs={'x': torch.randn(1, 376, 1024)},
            dynamic_shapes={'x': {1: Dim.AUTO}}, strict=False)
specs = TensorRTCompileSpec(precision=TensorRTPrecision.FP32).to_compile_specs()
part = [TensorRTPartitioner(compile_specs=specs)]
et_prog = to_edge_transform_and_lower({'forward': ep}, partitioner={'forward': part},
    compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True))
et = et_prog.to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=True,
    memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=True)))
runtime = Runtime.get()
prog = runtime.load_program(et.buffer)
method = prog.load_method('forward')
with torch.no_grad(): eager_out = module(x)
result = method.execute([x])
et_out = torch.tensor(result[0].numpy())
diff = (eager_out - et_out).abs()
print(f'Dynamic MatMul: max_diff={diff.max():.6f}  (expect < 0.001)')
"
```

### Reproduce the NaN (encoder with dynamic shapes)
```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \
  python examples/models/parakeet/export_parakeet_tdt.py \
    --dtype fp32 --output-dir /home/dev/models/parakeet_trt_fp32 --backend tensorrt

LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python repro_joint.py
# Encoder f_proj will show NaN
```

### Verify static shapes work (max_diff ≈ 2.7, no NaN)
```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python debug_conformer_ops.py single_layer
# Expect max_diff < 0.001 for a single conformer layer
```

## Architecture of the problem

When `torch.export` traces the encoder with `Dim.AUTO`, it inserts `sym_size.int`
nodes for the dynamic mel-frame dimension. These, plus derived arithmetic
(`add`, `sub`, `floordiv`), produce Int EValues that become inputs to the TRT
delegate. The partitioned graph looks like:

```
placeholder: audio_signal          ← Tensor [1, 128, T]
placeholder: length                ← Tensor [1]
sym_size  → 3001                   ← Int (mel frames)
add_1     → 1501                   ← Int (after subsample stage 1)
add_2     → 751                    ← Int (after subsample stage 2)
add_3     → 376                    ← Int (conformer output length)
sub       → 4624                   ← Int (padding related)
sub_1     → 5375                   ← Int (padding related)
add_5     → 751                    ← Int (intermediate)
executorch_call_delegate(...)      ← TRT engine
getitem_144 → f_proj [1, 625, 640] ← Tensor output
getitem_145 → enc_len [1]          ← Tensor output
```

The TRT engine has 9 inputs (2 tensors + 7 shape tensors) and 2 tensor outputs.
Shape tensors are `kHOST`-location inputs that carry dimension values. The
runtime correctly passes them as int32 host pointers and `enqueueV3` succeeds,
but the output is NaN.

## Dynamic shapes with different export/runtime sizes: IN PROGRESS

**Same-shape export/runtime**: Works. Export with N mel frames, run with N →
correct output (max_diff ≈ 0.2 vs eager, within TRT FP32 bounds).

**Different-shape export/runtime**: Still produces NaN. Root cause: TRT's
Myelin optimizer constant-folds all shape arithmetic when `opt == max` in the
optimization profile. With `Dim.AUTO` the upper bound is `int_oo` (unbounded),
which gets capped to the trace-time value, making `opt == max`. TRT then
builds a fully static engine optimized for exactly the trace-time shape.

**Workaround**: Export with the SAME mel frame count as the runtime input.

### What was done

**1. SymInt placeholders → internal TRT arithmetic (backend.py)**
SymInt delegate inputs (`sym_size`, `add_1`, `add_3`, `sub`, `sub_1`,
`add_5`) are no longer added as TRT network inputs. Instead:
- The base symbol (`s18 = audio_signal.shape[2]`) is extracted at runtime
  via `network.add_shape()` + `add_gather(dim=2)`.
- Derived expressions like `floor((s18-1)/8) + 1` are converted to TRT
  elementwise arithmetic via `_symint_expr_to_trt()`, which handles
  `sympy.Add`, `sympy.Mul`, `sympy.floor`, and torch's custom
  `FloorDiv` class (`torch.utils._sympy.functions.FloorDiv`).
- These TRT tensors are placed in `input_map` so converters that reference
  SymInt FX Nodes (e.g., `view_copy(tensor, [1, N:add_3, 4096])`) use the
  dynamic values.

This eliminates shape tensor inputs entirely. The TRT engine only has
tensor inputs (`audio_signal`, `length`).

**2. C++ executor: skip Int EValue args (TensorRTBackend.cpp)**
ExecuTorch's delegate framework still passes SymInt EValues as args to the
delegate's `execute()`. The C++ backend now counts all input args
(tensors + SymInts) to find the correct output offset:
`num_all_input_args = args.size() - num_outputs`. Int EValue args are
skipped when populating `input_buffers`.

**3. C++ executor: fix setInputShape for shape tensors (TensorRTExecutor.cpp)**
For shape tensor inputs, `setInputShape` must receive the tensor's own
dimensions (e.g., `(1,)` for a scalar), NOT the values. Values go through
`setTensorAddress` only. Both `allocate_gpu_buffers()` and `execute()` were
fixed.

**4. Converters: dynamic shape propagation**
- `input_has_dynamic_dims()` + `build_reshape_shape_tensor()` utilities
  compute output shapes from input element count when metadata is concrete.
- `convert_view` multi-dynamic path uses `input_map[node]` for FX Node
  shape args, creating TRT layer dependencies.
- `convert_expand`, `convert_slice`: enter dynamic path when input has
  dynamic dims; gather from runtime shape for non-broadcast dims.

**5. Python arithmetic ops for SymInt graph nodes**
`_process_graph_nodes` handles `operator.add/sub/mul/floordiv` on int32
TRT tensors, so SymInt arithmetic FX nodes (if present in the delegate
subgraph) produce dynamic TRT results.

### Why different-shape still produces NaN

TRT's Myelin optimizer constant-folds ALL shape arithmetic — including
`add_shape()` + `gather()` + elementwise ops — regardless of the
optimization profile's min/opt/max spread. Even with
`min=(1,128,161), opt=(1,128,2580), max=(1,128,5000)` (opt < max),
the engine output is still `[1, 625, 640]` (trace-time shape) with NaN.

The fundamental issue: TRT's optimizer treats shape computations as
compile-time evaluable. The `add_shape` layer returns the input tensor's
shape, but TRT pre-computes all possible shapes across the profile range
and embeds them as lookup tables or conditional branches, rather than
keeping the shape arithmetic as runtime-evaluated layers. The SymInt
arithmetic (`floor_div`, `sum`) gets folded into these pre-computed tables.

Explicit `Dim("mel_frames", min=161, max=5000)` in the export script
fails with `ConstraintViolationError` because the conformer encoder's
internal logic has modular arithmetic guards that aren't satisfied for
all values in [161, 5000].

### What would actually fix it: torch-tensorrt's `add_select` pattern

The upstream torch-tensorrt project uses `get_shape_with_dynamic_shape()`
(in `torch_tensorrt/dynamo/conversion/impl/shape.py`) which does:

```python
# 1. Get runtime input shape
input_shape = net.add_shape(input_val).get_output(0)
# 2. Create target shape constant (with -1 for dynamic dims)
scale = net.add_constant(shape, target_shape_with_neg1)
# 3. Find which dims are -1
condition = elementwise(scale < zeros, LESS)
# 4. Replace -1 dims with actual runtime values
result = net.add_select(condition, input_shape, scale)
# 5. Use as shuffle shape
layer.set_input(1, result)
```

The `add_select` creates a **data-dependent** shape that TRT cannot
constant-fold, because the selection depends on runtime tensor values.
This is the idiomatic TRT pattern for dynamic reshapes.

**Status**: Implemented in `convert_view` using per-dim TRT tensor
concatenation (like torch-tensorrt's `impl.shuffle.reshape`). The engine
builds cleanly and runs without crashes, but TRT still constant-folds the
shape tensors. The `add_select` pattern only prevents folding when the
select inputs have truly data-dependent values (not just shape-dependent).
Since all our shape values derive from `add_shape()` which TRT evaluates
at build time from the profile, the select is also folded.

### Latest progress: shape arithmetic now INSIDE TRT partition

Fixed the `TensorRTPartitioner` to keep `sym_size.int`, `operator.add`,
`operator.floordiv` etc. inside the TRT partition instead of extracting
them as SymInt placeholder inputs. The partitioner's `_produces_tensor_output`
check was rejecting these ops because they produce SymInt outputs, but
they're intermediate nodes consumed by tensor ops.

With this fix, the TRT network has 11868 layers (up from ~5000) because
the shape arithmetic (`add_shape → gather → floor_div → sum`) is now
inside the network, connected to the actual `audio_signal` tensor input.

**Static profile (min=opt=max=5000)**: Builds and runs, but static.

**Dynamic profile (min < opt)**: Fails with `Cask convolution isConsistent`
error. Even min=2500, opt=5000 triggers this. This is a TRT limitation
with convolutions that need to handle a wide range of spatial sizes with
dynamic shape tensors in the same network. Investigation needed:
- May need conv1d→conv2d decomposition (like the CUDA backend uses)
- May need to restrict the dynamic range to a narrower band
- May be related to how our conv converter interacts with dynamic shapes

### Changed files
- `TensorRTExecutor.cpp`: Fixed `setInputShape` for shape tensors
- `TensorRTBackend.cpp`: Skip Int EValue args, correct output offset
- `backend.py`: `_symint_expr_to_trt` (sympy → TRT arithmetic with
  `FloorDiv` support); SymInt placeholders as internal TRT constants/
  arithmetic; `operator.add/sub/mul/floordiv` handlers; profile range
  computation
- `converter_utils.py`: `input_has_dynamic_dims()`, `build_reshape_shape_tensor()`
- `converters/reshape.py`: view, reshape, flatten, squeeze, unsqueeze
- `converters/expand.py`: Enter dynamic path for dynamic input
- `converters/slice.py`: Enter dynamic path for dynamic input; unflatten

## Encoder precision analysis

The encoder shows max_diff ≈ 2.2 vs eager for a single conformer layer.
Root cause: the **subsampling output has extreme magnitudes** (range
[-6257, 5011]) which amplifies small numerical differences in TRT's
fused kernels (Myelin) for layer norm and attention.

| Component                | max_diff | Notes                           |
|--------------------------|----------|---------------------------------|
| Subsampling only (0L)    | 0.002    | Near-perfect                    |
| 1 conformer layer        | 2.208    | Input from subsample: [-6257, 5011] |
| Full encoder (24L)       | 0.197    | After normalization: [-180, 202] |

Investigated and ruled out:
- **TF32**: `clear_flag(TF32)` is applied; no change with/without
- **OBEY_PRECISION_CONSTRAINTS**: Identical results — TRT already uses FP32
- **Dynamic vs static shapes**: Identical results (max_diff difference < 0.001)

The relative error is ~0.05% for the full 24-layer output, which is within
expected TRT FP32 bounds. The high absolute max_diff for partial layers is
an artifact of large pre-normalization magnitudes, not a real precision defect.

### `STRICT_TYPES` fix
The `BuilderFlag.STRICT_TYPES` flag was renamed to `OBEY_PRECISION_CONSTRAINTS`
in TRT 10+. Fixed in `backend.py` to use `hasattr` fallback.

Debug scripts: `debug_encoder_trt.py` (layer bisection),
`debug_conformer_ops.py` (individual op testing).
