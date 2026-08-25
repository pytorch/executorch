---
title: Cadence Backend Overview
category: BACKEND_CONSTRAINT
backends: [Cadence]
last_validated: 2026-04-05
source_issues: [12306, 4812, 5508, 7081, 8237, 8900, 10499, 11050, 14208, 14701, 16898, 18181]
---

# Cadence Backend

The Cadence backend targets Cadence/Xtensa DSP processors. It does not use delegation in the traditional sense — instead of `to_edge_transform_and_lower`, it relies on `to_edge_with_preserved_ops` to prevent decomposition of ops it handles natively.

## Key Characteristics

- **No delegation**: Unlike other backends, Cadence does not use the partitioner/delegate pattern
- **Preserved ops**: Relies on `to_edge_with_preserved_ops` to keep ops like `aten.rms_norm` intact rather than decomposing them and fusing later
- **Pattern matching avoidance**: Preserving ops avoids the need for brittle pattern matching to re-fuse decomposed operations [Source: #12306]

## API Status

The `to_edge_with_preserved_ops` API is experimental. An effort is underway to promote it to official status by adding `preserve_ops` to `EdgeCompileConfig`. Key decisions:

- View/mutation ops should not be preserved if they remain in the portable graph (breaks functional graph assumptions)
- View/mutation ops CAN be preserved if consumed by a delegate backend
- The `_core_aten_ops_exception_list` should eventually be eliminated — non-core ATen ops should be explicitly listed in `preserve_ops` instead [Source: #12306]

## Usage Pattern

Note: `to_edge_with_preserved_ops` is not a public API and only exists in test code. Use `to_edge_transform_and_lower` with a preserved ops list instead:

```python
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

edge = to_edge_transform_and_lower(
    exported_program,
    compile_config=EdgeCompileConfig(
        preserve_ops=[torch.ops.aten.rms_norm.default],
    ),
)
```

Future API (once promoted):
```python
edge = to_edge(
    exported_program,
    compile_config=EdgeCompileConfig(
        preserve_ops=[torch.ops.aten.rms_norm.default],
    ),
)
```

## Setup and Build

### Prerequisites

Before running Cadence examples, ensure you run the full setup:
```bash
cd executorch
rm -rf pip-out
git submodule sync
git submodule update --init --recursive
./install_requirements.sh
./install_executorch.sh
./backends/cadence/install_requirements.sh
```

The Cadence backend support is still maturing — tutorials may not succeed fully without these steps. [Source: #11050]

### Build Errors

#### Missing `cadence_kernels` link error

When building Cadence examples, you may see:
```
/usr/bin/ld: cannot find -lcadence_kernels: No such file or directory
```

The `cadence_kernels` target is only built when the appropriate NNLib libraries are present. Ensure your local `hifi/third-party/nnlib/` is up to date by running `backends/cadence/install_requirements.sh` which clones the nnlib repositories. [Source: #11050, #18181]

#### Outdated paths in examples/cadence CMake config

The examples/cadence CMake config references legacy operator paths; consult the current tree for authoritative structure. [Source: #16898]

#### NNLib kernel removal breaks builds

When NNLib kernels are reorganized (e.g., `xa_nn_elm_add_broadcast_4D_f32xf32_f32` moved from local kernels to the NNLib submodule), builds break if the local nnlib is out of date. Update your local nnlib: https://github.com/foss-xtensa/nnlib-hifi4 [Source: #18181]

### C10_USING_CUSTOM_GENERATED_MACROS

If you see `c10/macros/cmake_macros.h file not found` when consuming ExecuTorch as a C++ dependency, define `C10_USING_CUSTOM_GENERATED_MACROS` in your CMakeLists.txt. [Source: #15922]

## ConvertToLinearPass Bug

`ConvertToLinearPass` (shared with XNNPACK) is not sound when transposes are elided. If `const_propagation` or `RemoveRedundantTransposes` removes the permute before `addmm`, the pass incorrectly reconstructs the linear op, causing dimension mismatches:
```
RuntimeError: a and b must have same reduction dim, but got [1, 4] X [8, 4]
```

The Cadence backend avoids this by not decomposing linear at all (using `preserve_ops` instead of pattern matching to reconstruct it). [Source: #10499]

## Xtensa Platform Limitations

### No `pread()` support

The Xtensa platform does not support `pread()`, requiring a workaround in `FileDataLoader` for multi-threaded file access. The workaround uses seeking with a mutex instead. [Source: #4812]

### kTensorDimensionLimit assumption

Many kernels and utilities assume tensors will never exceed `kTensorDimensionLimit` dimensions (currently 16). This is a runtime limitation, not an export-time check. [Source: #8237]

## Cadence Custom Ops

Running Cadence custom ops (e.g., `cadence::quantized_conv`, `cadence::quantized_relu`) requires the Cadence runner (`cadence_runner`), not the standard CPU executor runner. The CPU backend cannot dispatch these custom ops. [Source: #5508, #8900]

## Bare-Metal MCU Support

ExecuTorch can run on bare-metal MCUs (Cortex-M, ESP32-S3, etc.) using the portable kernel library. For MCUs without specialized backends:
1. Use portable ops with selective build to reduce binary size
2. Use `BufferDataLoader` for XIP (execute-in-place) from flash memory
3. Set appropriate `method_allocator_pool` size for the available SRAM [Source: #14208, #14701, #3585]

## NNLib Size

The `nnlib-hifi4` submodule is large (~700MB). It was moved to `backends/cadence/install_requirements.sh` to keep the default install fast. Only install it if you need the Cadence HiFi backend. [Source: #7081]
