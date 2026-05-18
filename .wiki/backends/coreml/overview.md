---
title: CoreML Backend Overview
category: BACKEND_CONSTRAINT
backends: [CoreML]
last_validated: 2026-04-05
source_issues: [1020, 10014, 10066, 10151, 10179, 10451, 10549, 11221, 11427, 11541, 11615, 11687, 11714, 11718, 11719, 11723, 11738, 11753, 12059, 12306, 12408, 12906, 13305, 14474, 14692, 14809, 15833, 16484, 16492, 17537]
---

# CoreML Backend

The CoreML backend delegates model execution to Apple's CoreML framework, targeting Apple Neural Engine (ANE), GPU (Metal/MPS), and CPU on iOS, macOS, and other Apple platforms.

## Architecture

The ET CoreML delegate is a lightweight wrapper around `coremltools`. It converts the exported PyTorch graph into a CoreML `.mlpackage` via the `torch.export.export` path (not `torch.jit.trace`). [Source: #10179]

Key architectural points:
- Uses `coremltools` under the hood for model conversion
- Supports `ct.TensorType` and `ct.StateType` (inferred from the ExportedProgram)
- Does **not** support `ct.ImageType` — users must handle normalization (scale/bias) within the model itself [Source: #10179]
- You can extract the `.mlpackage` from a `.pte` file for inspection — see `docs/source/backends/coreml/coreml-overview.md#extracting-the-mlpackage` [Source: #10179]

## Installation and Setup

### pip install (v0.6+)

As of v0.6, CoreML export is supported out of the box via pip on macOS:
```bash
pip install executorch
# coremltools is installed automatically as a dependency
```
No need to build from source for CoreML + XNNPACK export. [Source: #10066]

### Building from source (for MPS backend)

MPS backend is **not** included in the PyPI package. To use MPS, you must clone the repo and build from source. The iOS demo app requires MPS for one of its inference modes. [Source: #10066]

```bash
./install_executorch.sh
# or for subsequent installs:
pip install -e . --no-build-isolation
```

### iOS Framework Build

Build Apple frameworks from source:
```bash
./scripts/build_apple_frameworks.sh
```

Common issue: missing `zstd` module. Run `install_executorch.sh` first to ensure pip deps are installed. [Source: #10014]

After building, use the generated xcconfig for linking. [Source: #11753]

## Hardware Requirements

| Feature | Minimum Requirement |
|---------|-------------------|
| MPS backend | Apple Silicon (M1+), macOS Sonoma+, Xcode 15+ |
| MPS runtime | Apple Silicon only (x86 support added later via PR #1655 with caveats) |
| CoreML export | Any macOS with pip install |
| iOS deployment | iOS 17+ for MPS; CoreML works on earlier versions |

MPS backend requires Apple Silicon. On x86 Macs, the Metal device init fails with: `assert failed: _mtl_device != nil`. [Source: #1020]

## Export Patterns

### Basic CoreML export

```python
from executorch.backends.apple.coreml.partition import CoreMLPartitioner

model = models.mobilenet_v3_small(weights="DEFAULT").eval()
sample_inputs = (torch.randn(1, 3, 224, 224),)

et_program = to_edge_transform_and_lower(
    torch.export.export(model, sample_inputs),
    partitioner=[CoreMLPartitioner()],
).to_executorch()
```

### Using the export script (recommended)

```bash
python3 -m executorch.examples.apple.coreml.scripts.export --model_name=mv3
```

The export script includes patches and workarounds that the raw API does not. [Source: #10451]

## Known Issues and Workarounds

### dim_order_copy not supported by CoreML

**Status:** Active issue as of v0.6

When dim order is enabled (now the default), models contain `_to_dim_order_copy` ops that CoreML/coremltools does not recognize. The partitioner skips these nodes, leading to scalar inputs being passed to the delegate, which causes runtime crashes. [Source: #10451]

**Workaround:** Disable dim order during export:
```python
EdgeCompileConfig(_skip_dim_order=True)
```

The export script at `examples/apple/coreml/scripts/export.py` has a `--use_partitioner` flag; CI uses the older `to_backend` API which doesn't hit this path. [Source: #10451]

### Accuracy drops with CoreML

If you see significant accuracy drops when comparing CoreML-delegated models vs direct coremltools conversion, check:

1. **Input normalization**: The ET CoreML delegate does not support `ct.ImageType` with scale/bias. You must apply normalization within the model's forward method. [Source: #10179]
2. **Export path difference**: ET uses `torch.export.export` path, not `torch.jit.trace`. Results may differ from direct coremltools conversion via trace. [Source: #10179]
3. **fp16 conversion**: CoreML may convert to fp16 by default, which can cause precision loss for some models. [Source: #10179]

### Decomposition warnings during export

Warnings like "ET ignoring decomposition requests from CoreML" are benign — they indicate ops that don't have decompositions anyway and are not related to CoreML rejecting ops. [Source: #10179]

### iOS Xcode build — undefined symbols

If you see undefined symbol errors for `load_tokenizer` or `create_text_llm_runner` when building the iOS LLaMA demo:

1. Ensure submodules are initialized: `git submodule update --init --recursive`
2. Use the latest main branch — text LLM runner APIs change frequently
3. Use the correct xcconfig with proper `-force_load` linker flags [Source: #11753]

### Kernel registration on iOS

All kernel libraries (`kernels_optimized`, `kernels_quantized`, `kernels_custom`) require `--force_load` linker flags because they use static initialization. This is a known UX pain point. A `register_<lib_name>_kernels()` API is being developed to allow explicit registration without force-load. [Source: #11221]

### cpuinfo core detection

`executorch::extension::cpuinfo::get_num_performant_cores()` may report all cores on iPhone 14 and Pixel 8 due to cpuinfo not correctly parsing newer CPU topologies. Fixed in PR #11268 after cpuinfo patch. [Source: #10549]

## Multi-Entry Point Models

Multi-entry point export with shared mutable state is not fully supported with delegates (including CoreML). Current workarounds involve overriding `forward()` which is fragile. A better approach uses `torch.ao.quantization.pt2e.export_utils` wrapper functions, but this is still not great. [Source: #11738]

XNNPACK handles constant weight sharing across methods via a weight cache, but mutable state sharing across entry points is an active area of development. [Source: #11738]

## Quantization

### PT2E quantization requires iOS 17+ deployment target

When using PT2E quantization with CoreML, the minimum deployment target is iOS 17 (CoreML7). If `minimum_deployment_target` is set to `None` (defaults to iOS 15), quantization will fail with:
```
ValueError: No available version for quantize in the coremltools.target.iOS15 opset.
Please update the minimum_deployment_target to at least coremltools.target.iOS17
```

This is a known UX gap — fp32 models work on older targets but quantized models silently require iOS 17+. [Source: #13305, #12059]

### Palletization via quantize_

CoreML supports palletization (weight clustering) via the `quantize_` API using torchao. Available from ET 0.7+. [Source: #12923]

### torchao quantizer migration

CoreML's quantizer has migrated from `torch.ao.quantization.quantizer` to `torchao.quantization.pt2e.quantizer` (as of ExecuTorch v1.0+). If you see import errors related to the deprecated module, ensure you have a compatible version of torchao installed. [Source: #16484]

## Additional Known Issues and Workarounds

### macOS 26 / iOS 26 ANE regression

fp16 LLaMA inference on macOS 26.1 / iOS 26 produces inf/nan values on the Apple Neural Engine. This is caused by a regression in CoreML's handling of SDPA (Scaled Dot-Product Attention). Does not affect macOS 15.x.

**Workaround:** A decomposition-based workaround is available to avoid the problematic SDPA path. [Source: #15833]

### CoreML GPU crash on iPhone (works on macOS)

Some models crash with `shape.count = 0 != strides.count = 2` assertion failure when run on iPhone GPU but work fine on iPhone CPU or macOS GPU. Fixed in macOS 15.6 / iOS 18.6. [Source: #11541]

### CoreML segfault with pybindings

CoreML-delegated models can segfault when run via pybindings. ASAN reveals global-buffer-overflow in executor code. The issue was traced to PR #11391 changing backend options handling. The pybindings module definition is outdated vs the extension module. [Source: #12408]

### CoreML ignores add/sub alpha parameter

The CoreML backend ignores the `alpha` parameter in `aten::add.Tensor` and `aten::sub.Tensor`, producing incorrect results. A temporary fix is available in PR #13023; the upstream fix is tracked in coremltools #2573. [Source: #11687]

### CoreML floor_divide crashes process

`torch.floor_divide` on CoreML causes a process crash due to dtype mishandling. Fixed by PR #13018 in ExecuTorch. [Source: #11714]

### CoreML diagonal gives wrong outputs

`torch.diagonal` on CoreML produces incorrect outputs or crashes due to memory corruption. Temporary fix in PR #13023. [Source: #11718]

### CoreML model with no inputs fails to load

Models that produce output without any inputs (e.g., `return torch.ones(...)`) fail to load at runtime. Fixed in PR #13053. [Source: #11719]

### torch.split fails in to_edge (aliasing error)

`torch.split` errors out during `to_edge` with an aliasing complaint. Workaround: add `split` and `split_copy` to `replace_broken_ops_with_function_ops_pass.py`. [Source: #11723]

### CoreML cached model produces garbage output

When a CoreML model is cached to disk, subsequent runs can produce corrupted outputs for certain models (observed with stories110M). Clearing the model cache and re-compiling fixes it. [Source: #16492]

### CoreML segfault with aten::where (single-input form)

Models containing `aten::where(x)` (single-input) or `aten::nonzero_numpy` segfault at runtime. Export and lowering succeed; the crash is an underlying CoreML bug involving dynamic shapes. [Source: #17537]

### efficient_sam model issues on CoreML

efficient_sam fails to run on CoreML with MPS-related errors about missing resources. Loading eventually succeeds on CPU/ANE but with extremely long load times. Fixed in macOS 15.6. [Source: #12906]

### CoreML export fails with "Metadata is invalid or missing"

If you see `Metadata is invalid or missing` when exporting custom models with CoreMLPartitioner, try upgrading to ET 1.0+. This was seen with EfficientViT and other custom architectures. [Source: #14692]

### Rank > 5 tensors not supported

CoreML does not support tensors with rank greater than 5. The partitioner should (but doesn't always) exclude these, causing lowering errors. [Source: #11694]

### Unsupported op partitioner gaps

Several ops are partitioned to CoreML but fail during lowering. The error messages are clear but these should ideally be partitioner constraints:

| Op | Issue |
|---|---|
| avg_pool2d with divisor_override | Fails to lower [#11695] |
| max_pool with dilation > 1 | Only dilation=1 supported [#11697] |
| topk with sorted=False | iOS < 16 only [#11698] |
| PixelUnshuffle | iOS < 16 only [#11711] |
| ConvTranspose with output_padding | Not supported [#11705] |
| maxpool with indices | Not supported [#11706] |
| CircularPad1/2/3d | Not supported [#11710] |
| Convolution with circular padding | Not supported [#11703] |
| asinh/acosh | Internal error, temp fix in PR #13023 [#11712] |
| integer ReLU | Not supported [#11693] |
| BatchNorm3d | Crashes process [#11701] |
| ReflectionPad3d | Fails to load [#11708] |
| ReplicationPad3d | Fails to load [#11709] |

## ANE (Apple Neural Engine) Scheduling

### Simple indexing prevents ANE scheduling

Using simple indexing patterns (like `tensor[0]`) can generate `ios18.gather` ops that cannot be scheduled on the ANE, forcing GPU/CPU fallback. This is a coremltools limitation. [Source: #11615]

### ANE compile OOMs on certain shapes

Certain input shapes can cause the ANE compiler to run out of memory. No general workaround — try adjusting input dimensions. [Source: #8439]

### CPU overhead after ANE execution

There can be significant CPU overhead after ANE execution completes, impacting end-to-end latency beyond just the ANE compute time. [Source: #8445]

## LLaVA / Large Model Memory

LLaVA models require ~6 GB of RAM on iOS, which exceeds the memory limit on most iPhones. The PTE file itself is ~3.8 GB (7B model at ~4 effective bits), plus ~2.3 GB activation memory. XNNPACK weight cache does not release original weights, contributing to the overhead. [Source: #14474]

## SwiftPM / iOS Integration

### SwiftPM version compatibility

SwiftPM binary distributions may fail with "Error 32 (NotFound)" when loading methods. If you encounter this:
1. Try `1.0.0` or later versions
2. Avoid `-all_load` linker flag (causes 88 duplicate symbols)
3. Use the xcconfig from the Benchmark target as reference [Source: #14809]

### MPS delegate crashes on iOS 26

MPS delegate crashes on iOS 26 simulator with `insertObject:atIndex: object cannot be nil`. [Source: #11655]

### In-place activations alter graph outputs

Using in-place ops like `relu(inplace=True)` adds extra outputs for USER_INPUT_MUTATION. This is expected behavior — the mutated input must be output to keep the graph functional. [Source: #11700]

## CoreML + Preserved Ops

The `to_edge_with_preserved_ops` API (experimental) allows preserving ops like `aten.rms_norm` from decomposition. CoreML requests that view ops (view, transpose, permute) be preserved when consumed by the backend. The API is being promoted to official status with `preserve_ops` added to `EdgeCompileConfig`. [Source: #12306]

## CoreML Export on Linux

CoreML export now works on Linux (as of v0.6+). The `coremltools` package can run on Linux for AOT compilation, though runtime execution still requires macOS/iOS. [Source: #9800]
