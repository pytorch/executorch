---
title: Arm Backend Overview
category: BACKEND_CONSTRAINT
backends: [Arm]
last_validated: 2026-04-05
source_issues: [1004, 1110, 1161, 1163, 1230, 11913, 12237, 12306, 10899, 10964, 12270, 12447, 12627, 12959, 12991, 13022, 13399, 13557, 13842, 13901, 15805, 15870, 16090, 16225, 16244, 16426, 16541, 16628, 16629, 16739, 16779, 16784, 16864, 16899, 16902, 17157, 17211, 17241, 17397, 17437, 17489, 17651, 17653, 17667, 17668, 17753, 17902, 18306, 18319, 18320, 18491, 18500]
---

# Arm Backend (Ethos-U / Cortex-M)

The Arm backend delegates model execution to Arm Ethos-U NPUs (U55, U85) for acceleration, with fallback to Cortex-M CPUs. It uses TOSA (Tensor Operator Set Architecture) as the intermediate representation and the Vela compiler to generate Ethos-U command streams.

## Architecture

The delegation flow:
```
PyTorch model → torch.export → PT2E quantization → to_edge →
ArmPartitioner/TOSAPartitioner → TOSA IR → Vela compiler → .pte with Ethos-U command stream
```

Three output modes:
1. **Vela output** (default): Generates a binary with Ethos-U command stream for NPU execution
2. **TOSA output**: For reference checking and debugging (not runnable on hardware)
3. **No delegate**: Runs entirely on Cortex-M CPU via portable ExecuTorch kernels [Source: #1161]

### TOSA Representation

TOSA requires NHWC (channel-last) memory format. PyTorch uses NCHW (channel-first). The `Permute_Memory_Format_Pass` handles this conversion during lowering, though it was historically WIP with shape update issues. [Source: #1110]

## Hardware Targets

| Target | Description | Use Case |
|--------|-------------|----------|
| Ethos-U55 | Micro NPU for Cortex-M | Ultra-low-power edge inference |
| Ethos-U85 (MAC-256) | Higher-performance micro NPU | Corstone-320 based systems |
| Cortex-M55 | CPU with Helium (MVE) | Fallback for unsupported ops |

## Setup

### Environment setup

```bash
cd examples/arm
./setup.sh
```

Note: The `--skip-fvp-setup` flag is ignored because `setup.sh` unconditionally calls `install_reference_model.sh` which requires FVP binaries. [Source: #12306 area]

### Submodule issues

The Arm backend depends on submodules hosted on `git.mlplatform.org` which has historically been unstable:
- SSL certificate verification failures
- HTTP 500 errors from the server

**Workaround:** To disable the Ethos-U driver dependency, deinit the corresponding third-party submodule. See upstream docs for current command syntax. [Source: #1004, #1163]

These submodules have been moved to backend-specific install scripts in later versions. [Source: #1004]

## Export Example

```python
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.backends.arm.quantizer.arm_quantizer import (
    ArmQuantizer, get_symmetric_quantization_config, TOSAQuantizer
)

# Quantize
quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-0.80+BI+u55"))
quantizer.set_global(get_symmetric_quantization_config())
prepared = prepare_pt2e(exported_model, quantizer)
# ... calibration ...
quantized = convert_pt2e(prepared)

# Export with delegation
exported = torch.export.export_for_training(quantized, example_inputs)
edge = to_edge_transform_and_lower(
    exported,
    partitioner=[TOSAPartitioner(...)],
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
)
et_program = edge.to_executorch()
```

## Quantization

- Use `TOSAQuantizer` with `get_symmetric_quantization_config()` for Arm targets
- Use `XNNPACKQuantizer` only if targeting XNNPACK fallback (different numerics) [Source: #1161]
- For fused quantized operators: use `quantization_tag` during annotation, or `SubgraphMatcher` for pattern matching Q/DQ nodes [Source: #1230]
- Reference integer decomposition available via `convert_pt2e` with `use_reference_representation=True` for TOSA numeric matching [Source: #1230]

## Running on FVP

Models are compiled to `.pte` files, then converted to C headers for embedding in the Cortex-M firmware:
```bash
python backends/cadence/utils/gen_header.py
```

The `method_allocator_pool` size in `runner.cpp` controls tensor arena space. Increase it for larger models:
```cpp
__attribute__((section(".sram.data"), aligned(16)))
uint8_t method_allocator_pool[136 * 1024U];
```
[Source: #1161]

## Cross-Compilation for Cortex-M

Cross-compiling ExecuTorch for Arm Cortex-M hardware (e.g., Raspberry Pi Pico 2) has known friction points:

1. Documentation needs Cortex-M specific sections
2. The build for Pico 2 may not produce the final binary
3. Using ET as a third-party project in embedded CMake is difficult
4. `libportable_kernels` for Arm baremetal may not support selective build

Use `-DEXECUTORCH_SELECT_OPS_FROM_MODEL="<path>.pte"` and `-DEXECUTORCH_DTYPE_SELECTIVE_BUILD=ON` for smaller binaries. [Source: #11913]

## Dynamic Shapes

The Arm backend does **not** support models with dynamic shapes. SymFloat objects in the graph will cause errors like:
```
TypeError: Expected a FakeTensor in meta["val"] of node _local_scalar_dense_2, but got SymFloat
```

**Workaround:** Fix input sizes at export time. YOLO models in particular require removing dynamic anchor computation. [Source: #12237]

## YOLO on Ethos-U

YOLO models require special handling:
- Use `strict=False` in `export_for_training` (attribute mutation not supported in strict mode)
- Fix input sizes to avoid dynamic shapes
- YOLOv12 example available at `examples/models/yolo12`
- Successfully tested on Ethos-U85 MAC-256 / Corstone-320 with fixed input sizes [Source: #12237]

## INT16 Extended Profile

The Arm backend supports INT16 quantization for the extended TOSA profile. Supported INT16 ops include:
- Linear (FCNode), Add, Mul, Sub
- Sigmoid, Tanh, Slice
- View/Transpose, Cat, Rescale
- Quantize/Dequantize nodes

Use `get_symmetric_a16w8_quantization_config()` for 16-bit activations with 8-bit weights. [Source: #13840, #13635]

**Known issue:** `FuseQuantizedActivationPass` fails for INT16 symmetric quantization in some cases. [Source: #17437]

## CMSIS-NN Integration (Cortex-M)

The `backends/cortex_m` module provides CMSIS-NN optimized operators for Cortex-M CPUs (M33, M55, M85):

| Op | CMSIS-NN Status |
|---|---|
| quantized_add | Supported [#13506] |
| quantized_linear (per tensor/channel) | Supported [#13708] |
| quantized_conv2d | Supported [#13707] |
| quantized_avg_pool2d | Supported [#13709] |
| quantized_max_pool2d | Supported [#13710] |
| quantized_relu / hardtanh | Supported [#13711, #13712] |
| quantized_sub | In progress [#13706] |
| depthwise_conv2d | Supported [#16105] |
| transpose_conv2d | Supported [#16106] |
| max_pool | Supported [#16107] |
| batch_matmul | Supported [#16109] |
| SVDF | Supported [#16110] |
| Pad | Supported [#16111] |
| LSTM | Not yet [#16108] |

### Benchmark: CMSIS-NN vs TFLite Micro (Alif E8 Cortex-M55)

MobileNetV2 int8 benchmarks on Alif E8 HP Cortex-M55 (SRAM arenas, MRAM model):

| Framework | Inference Time | Notes |
|---|---|---|
| TFLite Micro + CMSIS-NN | Baseline | Reference |
| ExecuTorch + CMSIS-NN | ~Comparable | Improving with each release |
| ExecuTorch (portable ops) | Much slower | Use CMSIS-NN for production |

Key insight: Q/DQ nodes outside delegation add overhead. Use `QuantizeInputs`/`QuantizeOutputs` passes to keep I/O as int8 when possible. [Source: #17157, #17651, #16899]

## Zephyr RTOS Support

ExecuTorch can be built as a Zephyr external module. Documentation and sample apps are under `zephyr/samples/`.

- Ethos-U delegation works via Zephyr
- Cortex-M only (no Ethos-U) also supported
- See `zephyr/README.md` for build instructions [Source: #13508, #17618, #17653]

## Cortex-A Support

ExecuTorch runs on Cortex-A CPUs (Android, Linux, macOS, iOS) using XNNPACK for acceleration. No special Arm-specific backend is needed for Cortex-A — use XNNPACK or Vulkan delegates. [Source: #12627]

## GRU / RNN Support

GRU layers are not directly supported by the Arm Ethos-U backend. The decomposition of `torch.nn.GRU` fails during lowering. Manual decomposition of GRU into its component ops may work but requires careful handling. [Source: #12270, #17753]

## BatchNorm2d Without Preceding Conv

Standalone `BatchNorm2d` (not immediately following a convolution) is not supported for Ethos-U delegation. In TFLite→Vela flow this works, but in ExecuTorch it fails.

**Workaround:** Manually decompose the BatchNorm into equivalent operations (mul + add with running mean/var). [Source: #17241, #17397]

## Softmax Performance on Ethos-U

Softmax decomposition for Ethos-U uses `aten::amax` which runs on the elementwise engine (not MACs), causing poor performance. The Vela performance estimator is not accurate for cycle counts — always validate on FVP or real hardware. [Source: #18319]

## LayerNorm and Quantization Accuracy

LayerNorm quantization is sensitive to epsilon values. The default epsilon (1e-5) can cause accuracy issues in int8. For DeiT-tiny and similar transformer models, accuracy drops in the TOSA quantized pipeline may be caused by LayerNorm sensitivity. The `--stable_softmax` flag enables a numerically stable algorithm. [Source: #16426, #18306, #18316]

## PReLU Not Supported on Ethos-U

`torch.nn.PReLU` is not supported for Ethos-U delegation. The forward call decomposes to `torch.where(x>0, x, self.weights * x)` which isn't handled by the backend. [Source: #16902]

## Object Lifetime Bug in ARM Runner

`arm_executor_runner.cpp` had a major object lifetime error where `BufferCleanup` used `free()` on memory not allocated by `malloc()` (e.g., from `ArmMemoryAllocator` using static pools). This was hidden by ARM FVP but crashes on real hardware. Fixed in PR #16339. [Source: #16225]

## Quantizer Issues

### SharedQuantizationSpec recursion

Using `SharedQuantizationSpec` in certain graph topologies (e.g., `minimum → eq` chains) causes infinite recursion. Fixed upstream in pytorch/ao#3011. [Source: #13842]

### Conv-ReLU + Residual observer sharing bug

The Arm Ethos quantizer incorrectly shares/derives observers across `add`, `permute`, and `relu` operations at Conv-ReLU + residual junctions. The root cause is in `quantization_annotator.py` where `add` node annotation doesn't properly handle shared quantization specs. [Source: #12959]

### LeakyReLU device placement

ARM quantizers fail on `nn.LeakyReLU` due to the `negative_slope` constant being placed on the wrong device. XNNPACK quantizer doesn't have this issue. [Source: #16541]

### set_module_name doesn't apply to ops

`VgfQuantizer.set_module_name()` targets submodules (`torch.nn.Module`), not raw operators. To apply module-level quantization config to an op like `add`, wrap it in a `torch.nn.Module` subclass. [Source: #16542]

### Name filter doesn't match nodes

`arm_quantizer.py` name filter assumes module names start with `"L['self']."` which may not be present, causing filters to miss target nodes. Fixed on main. [Source: #15870]

## Vela Compiler Issues

### Custom Vela config file

Custom Vela configuration files may crash due to a parsing bug; fixed in Vela 4.5.0. [Source: #15805]

### `--optimise Size` result mismatch

Running Vela with `--optimise Size` can produce different (incorrect) results compared to the default optimization. [Source: #16864]

### reduce_mean not fully delegated

The operator support checks for views/reshapes are overly pessimistic — they assume transposes are always needed, rejecting view nodes with axis-product > 65536 even when no transpose is required. [Source: #16779]

### Ethos-U crashes on conv→relu→permute→reshape(5D)

Specific graph pattern `conv → relu → permute → reshape(5D)` crashes during partitioning for Ethos-U. [Source: #16739]

## Setup Issues

### Dependency conflicts in setup.sh

`examples/arm/setup.sh` has known dependency conflicts:
- `ethos-u-vela` requires `flatbuffers==24.12.23` but `tosa-tools` requires `flatbuffers==23.5.26`
- `executorch` requires `numpy>=2.0` but `tosa-tools` requires older numpy

These conflicts are known and the backend still works despite pip warnings. [Source: #10899, #12991]

### tosa module import failure

`from executorch.backends.arm.tosa.partitioner import TOSAPartitioner` may fail with `No module named 'tosa'` after `pip install executorch`. You must also run `examples/arm/setup.sh` to install tosa dependencies. A future `pip install executorch[ethos-u]` flow is planned. [Source: #13901]

### ARM GitLab instability

Arm-hosted GitLab (`git.gitlab.arm.com`) for tosa-serialization had recurring access issues causing CI failures. This has been resolved with improved IP access management. [Source: #13557]

### Cross-compilation flatc issues

When cross-compiling for ARM targets, `flatc` compilation may fail because it tries to build for the target instead of the host. On newer ExecuTorch versions, remove manual `FLATBUFFERS_FLATC_EXECUTABLE` and `EXECUTORCH_BUILD_FLATC` args — the build system handles host flatc compilation automatically. [Source: #10964]
