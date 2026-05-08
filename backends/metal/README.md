# ExecuTorch Metal Backend (`metal_v2`)

A Metal-based GPU backend for ExecuTorch on Apple silicon. Targets both
the Metal 3 dispatch path (with optional MPSGraph fallback) and the
Metal 4 dispatch path (residency-set + cooperative tensors).

## Directory layout

```
backends/metal/
├── core/         # Pure Metal infrastructure — no op knowledge
├── ops/          # Op implementations + per-op-family helpers
│   ├── registry/ # Op base class, registry, shared op-side helpers
│   └── mlx_jit/  # MLX-style JIT kernel-loading infrastructure
├── kernels/      # Shader-side helpers (host-included MSL fragments)
├── tests/        # GoogleTest unit tests
├── CMakeLists.txt
├── MlxJit.cmake
├── TARGETS       # Buck rules
└── README.md
```

### `core/` — Metal infrastructure

Self-contained Metal runtime. No file in `core/` includes anything from
`ops/` or `kernels/`. Could in principle be reused by a different op
model.

| File | Responsibility |
|---|---|
| `MetalStream` | Composes the per-thread subsystems; orchestrates `flush`/`wait`/`sync` |
| `MetalAllocator` | Owns per-stream memory; routes through pool/heap; tracks External/Owned |
| `MetalCommandRecorder` | Encoder + dispatch lifecycle (RAII `Dispatch`); auto-flush; hazard tracking; per-CB residency binds |
| `IComputeBackend` (+ `MetalMTL3Backend`, `MetalMTL4Backend`) | MTL3 / MTL4 dispatch backends, swapped by `useMTL4()` |
| `MpsInterop` | MPSGraph encode bridge with side-door residency declaration |
| `MetalKernelCompiler` + `MetalKernelCache` + `MetalKernel` | MSL compile, process-wide PSO cache, per-kernel slot metadata |
| `MetalAllocator` co-tenants: `BufferRegistry`, `MetalBufferPool`, `MetalHeap`, `ResidencyManager` | Memory subsystem internals |
| `HazardTracker` | RAW/WAW barrier-skip analysis on the MTL4 path |
| `MetalDeviceInfo` | Apple-silicon tier classification |
| `MetalTypes` | `uvec3` and shared scalar types |

### `ops/registry/` — Op layer infrastructure

The contract every op implementation participates in. Read once; rarely
changed.

| File | Responsibility |
|---|---|
| `MetalOp` | Op base class (`name`, `dispatch`, `kernelSource`, `getKernel`) |
| `MetalOpRegistry` | Process-wide name → op singleton; constructs all known ops at first use |
| `OpUtils` | Output resize, broadcast strides, dtype suffix helpers |
| `StrideUtils` | Stride math (collapsing, broadcast computation) |
| `Elementwise` | Binary/unary variant detection (SS/SV/VS/VV/G) |
| `GridDims` | Grid + threadgroup sizing helpers |

### `ops/` (top level) — Op implementations

Each op pair `XOp.{h,mm}` extends `MetalOp`. Per-op-family inline
helpers (`MatMulCommon.h`, `MatMulMlxJit.h`, `SdpaMlxJit.h`,
`AffineQuantizedLinearMlxJit.h`, `MatMulKernels.metal.h`) live alongside
the op that uses them.

### `ops/mlx_jit/` — MLX JIT infrastructure

Loader + snippet system for MLX-style per-shape JIT-compiled kernels
(`KernelLoader`, `Snippets`, `TemplateGen`, plus per-family snippet
sources). Consumed by the per-family JIT helpers above.

### `kernels/` — Shader-side helpers

`Accessors.h` and `TileLoad.h` are MSL fragments included in raw-string
kernel sources. Distinct from C++ host headers.

---

## Using the core API

The canonical API surface is **subsystem accessors on `MetalStream`** plus
the **RAII `Dispatch` API** for kernel binding + dispatch.

### Stream lifecycle

```cpp
#include <executorch/backends/metal/core/MetalStream.h>

using executorch::backends::metal_v2::MetalStream;

// Per-thread default stream (recommended).
MetalStream* stream = MetalStream::get();

// Or create an independent stream with explicit ownership.
auto stream = MetalStream::create();
```

### Memory

```cpp
auto& alloc = stream->allocator();
void* p = alloc.alloc(N);                  // pool-backed allocation
alloc.registerExternalBuffer(p, N);        // wrap externally-owned memory
auto bind = alloc.bufferForPtr(p, N);      // {MTLBuffer, offset} resolution
alloc.free(p);
```

### Dispatch (RAII)

```cpp
auto& rec = stream->recorder();

// Chain form (preferred for unconditional bindings):
rec.beginDispatch(kernel)
    .setInput(0, A.const_data_ptr(), A.nbytes())
    .setInput(1, B.const_data_ptr(), B.nbytes())
    .setOutput(2, C.mutable_data_ptr(), C.nbytes())
    .setBytes<int32_t>(3, M)
    .run(grid, block);

// Lvalue form (preferred when bindings are conditional):
auto d = rec.beginDispatch(kernel);
d.setInput(0, A.const_data_ptr(), A.nbytes());
d.setInput(1, B.const_data_ptr(), B.nbytes());
if (hasBias) d.setInput(2, bias.const_data_ptr(), bias.nbytes());
d.run(grid, block);
```

The `Dispatch` object enforces three properties at the type level:
- **`[[nodiscard]]`** — forgetting `.run(...)` is a compile-time warning.
- **Captured kernel** — bindings cannot drift to a different kernel.
- **Move-only** — a single `Dispatch` cannot be `.run()` twice.

### Compile

```cpp
MetalKernel* kernel = stream->compiler()->compile(
    msl_source_string, "kernel_name", /*function_constants=*/nullptr);
```

For shape-stable use, prefer `MetalKernelCache::shared().findOrInsert(
key, factory)` so concurrent threads share the same compiled PSO.

### MPSGraph (when `ET_METAL_USE_MPSGRAPH=1`)

```cpp
stream->mps().encodeWithLegacyCommandBuffer(
    [&](MPSCommandBuffer* mpsCB) {
      [graph encodeToCommandBuffer:mpsCB ...];
    },
    binds.data(), binds.size());  // declare side-door binds
```

Side-door binds are required so the MTL4 residency set covers buffers
MPSGraph encoded outside our typed-setter path.

### Stream-level orchestration

```cpp
stream->flush();   // submit + start the pending CB without waiting
stream->wait();    // block until the GPU has drained submitted work
stream->sync();    // flush + wait
```

`sync()` is the right call after a single computation finishes and you
need the result on CPU.

---

## Build configuration

Two mutually exclusive build modes are selected via the
`METAL_V2_USE_MTL4` CMake option:

| Mode | Defines | Notes |
|---|---|---|
| `METAL_V2_USE_MTL4=ON` (default) | `ET_METAL4_ENABLE=1`, `ET_METAL_USE_MPSGRAPH=0` | Metal 4 dispatch + residency set |
| `METAL_V2_USE_MTL4=OFF` | `ET_METAL4_ENABLE=0`, `ET_METAL_USE_MPSGRAPH=1` | Metal 3 + MPSGraph fallback for unimplemented ops |

Runtime selection between MTL3 and MTL4 (within the MTL4 build) is
controlled by `useMTL4()` in `MetalStream.h`, which combines the
compile-time gate with an OS availability check.

---

## Adding a new op

1. Create `ops/MyOp.h` and `ops/MyOp.mm` extending `MetalOp`. Implement:
   - `name()` → unique op name (asserted unique at registry construction)
   - `kernelSource()` → MSL string returned via stable pointer
   - `dispatch(stream, inputs, outputs)` → RAII `Dispatch` with
     `setInput`/`setOutput`/`setBytes`/`.run()`
2. Add `registerOp(std::make_unique<MyOp>())` to the `MetalOpRegistry`
   constructor in `ops/registry/MetalOpRegistry.mm`.
3. Add `${_metal_dir}/ops/MyOp.mm` to `_metal_arc_srcs` in
   `CMakeLists.txt`.
4. Write a unit test under `tests/`.

---

## Testing

```sh
cd backends/metal/tests
./build_and_run_tests.sh                       # both MTL3 + MTL4
./build_and_run_tests.sh --config mtl4         # MTL4 only
./build_and_run_tests.sh --config mtl3 -t metal_v2_test_buffer_registry
```

Tests under `tests/` use the same canonical API: subsystem accessors +
RAII `Dispatch`. New tests should follow the same convention.
