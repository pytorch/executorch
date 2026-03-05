# TRT Backend: Dynamic Shape Support — Debugging Progress

## Status

**Int EValue crash: FIXED.** The pybind runtime crashed (`EValue is not a Tensor`)
when TRT delegates had dynamic shapes, because ExecuTorch passes `sym_size` and
derived dimension values as Int EValues while the backend called `toTensor()`.
The fix in `TensorRTBackend::execute()` checks `args[i]->isInt()` and routes
Int args through the shape-tensor path.

**NaN outputs with dynamic shapes: OPEN.** The full encoder produces all-NaN
output when exported with `Dim.AUTO`. Static shapes work (max_diff ≈ 2.7 vs
eager — a separate precision issue). The shape tensor values themselves are
correct (verified via logging), and `enqueueV3` returns success, but TRT
produces NaN.

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

## Debugging next steps

### 1. Check shape tensor value consistency
The TRT engine was built with `max_mel_frames=5000` as the trace-time example.
The shape tensor profile ranges may not cover the test-time values. Add logging
in `TensorRTExecutor::execute()` after `setInputShape` to verify TRT's
`context_->allInputShapesSpecified()` returns true and
`context_->allInputDimensionsSpecified()` returns true.

### 2. Test with trace-time shape
Export with 5000 mel frames and run with 5000 mel frames (match exactly).
If this eliminates NaN, the issue is profile range mismatch.

### 3. Enable TRT error logging during execution
Set `TRT_LOG_LEVEL=0` (verbose) and look for TRT warnings/errors during
`enqueueV3`. NaN from TRT typically means:
- Shape tensor values outside the optimization profile range
- Output buffer too small for the inferred shape
- Internal kernel error due to unsupported dynamic shape combination

### 4. Compare with tensorrt_executor_runner (C++ runner)
Build the C++ runner and test the same .pte:
```bash
cd /home/dev/executorch/pip-out/temp.linux-x86_64-cpython-313/cmake-out
cmake --build . --target tensorrt_executor_runner
./backends/nvidia/tensorrt/tensorrt_executor_runner \
  --model_path /home/dev/models/parakeet_trt_fp32/model.pte
```
If the C++ runner also produces NaN, the issue is in the engine/executor.
If it works, the issue is in how pybind passes args.

## Separate issue: full encoder precision (max_diff ≈ 2.7)

Even with static shapes (no dynamic dims, no shape tensors), the full 24-layer
encoder has max_diff=2.68 vs eager. Individual ops are fine:

| Component           | max_diff |
|---------------------|----------|
| LayerNorm (each)    | < 0.00003 |
| FeedForward 1       | 0.002884 |
| FeedForward 2       | 0.002679 |
| SelfAttn            | 0.000484 |
| ConvModule          | 0.000916 |
| Full single layer   | 0.000061 |
| **Full encoder (24L)** | **2.676** |

This suggests TRT's engine-level optimizations (tactic selection, kernel fusion)
degrade precision when the entire encoder is a single partition. Possible fixes:
- Force `strict_type_constraints=True` in the compile spec
- Split the encoder into multiple smaller TRT partitions
- Investigate if TF32 is still being used despite `clear_flag(TF32)`

Debug scripts: `debug_encoder_trt.py` (layer bisection),
`debug_conformer_ops.py` (individual op testing).
