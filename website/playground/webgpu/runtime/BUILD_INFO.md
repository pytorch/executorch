# ExecuTorch Wasm + WebGPU runtime

`executorch_wasm_lib.js` and `executorch_wasm_lib.wasm` are the ExecuTorch Wasm
bindings (`extension/wasm`) with the WebGPU backend, linked into one web bundle.
The `.js` loader ships in this folder. The `.wasm` is hosted on Hugging Face
(`executorch-community/ExecuTorch-WebGPU-Runtime`, pinned in `src/config/runtime.js`),
downloaded and checked against the sha256 below like a model file.
Every demo in the playground shares this one runtime (`src/runtime/executorch.js`).
The JS file defines `createExecuTorch()`. After `await et.initWebGPU()`, a `.pte`
loads with `et.Module.load(path)` and `loadMethod('forward')`, and runs with an
async `forward([tensors])`.

## Source

Built from the ExecuTorch source tree of 2026-10-16 (`version.txt` 1.6.0) with the two
changes that make WebGPU run in the Wasm build, both merged upstream:

1. pytorch/executorch#23938: the `EXECUTORCH_BUILD_WASM_WEBGPU` option, which links the
   WebGPU backend into `extension/wasm`, and the `initWebGPU()` binding.
2. pytorch/executorch#23914: `WebGPUGraph::propagate_resize` resizes every operand before
   its consumer in one pass, so a multi-token LLM prefill works.

A build from ExecuTorch main that contains both, with the commands below, should give the
same files. The emdawnwebgpu override (`EXECUTORCH_WEBGPU_EMDAWNWEBGPU_PORT`) was only
needed because the build machine could not reach GitHub; without it, Emscripten downloads
the same emdawnwebgpu version itself.

## Toolchain

- Emscripten 4.0.19, official release build `8b01e2ec3f33e6b94842096d7312ce4ef5f33f6c`.
- emdawnwebgpu v20251002.162335, the version Emscripten 4.0.19 pins. It was rebuilt
  from Dawn source at `01940842b667a7812d0e4ca0ef4367fbec294241`, using Dawn's
  own packaging recipe.
- CMake 3.31, Ninja 1.13. The configure step needs a Python with `torch`
  (for `torchgen`).

## Build

A small consumer project links the bindings into a distributable bundle, as
described under "Building for Web" in `extension/wasm/README.md`:

```cmake
cmake_minimum_required(VERSION 3.29)
project(executorch_web)
add_subdirectory(${ET_SRC} executorch)
add_executable(executorch_wasm_lib)
target_link_libraries(executorch_wasm_lib PRIVATE executorch_wasm)
target_link_options(executorch_wasm_lib PRIVATE
  -sMODULARIZE=1 -sEXPORT_NAME=createExecuTorch -sENVIRONMENT=web,worker
  -sFORCE_FILESYSTEM=1 -sEXPORTED_RUNTIME_METHODS=FS -sINITIAL_MEMORY=64MB
  -sEXPORT_EXCEPTION_HANDLING_HELPERS=1)
```

```bash
emcmake cmake -S <consumer project> -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DET_SRC=<executorch> \
  -DEXECUTORCH_BUILD_WASM=ON -DEXECUTORCH_BUILD_WEBGPU=ON -DEXECUTORCH_BUILD_WASM_WEBGPU=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON -DEXECUTORCH_LOG_LEVEL=Info \
  -DEXECUTORCH_WEBGPU_EMDAWNWEBGPU_PORT=<emdawnwebgpu_pkg>/emdawnwebgpu.port.py \
  "-DCMAKE_C_FLAGS=-ffile-prefix-map=<executorch>=executorch" \
  "-DCMAKE_CXX_FLAGS=-ffile-prefix-map=<executorch>=executorch" \
  -DPYTHON_EXECUTABLE=<python with torch>
cmake --build build --target executorch_wasm_lib
```

The WebGPU Wasm option itself adds `-sASYNCIFY -sASYNCIFY_STACK_SIZE=65536
-sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -fexceptions`. The full portable
kernel library is linked (no selective build). Logging is at Info, so a failed
load or run prints the reason to the console. `-ffile-prefix-map` keeps
build-machine paths out of the binary.

## Files

| File | Bytes | sha256 |
|---|---:|---|
| `executorch_wasm_lib.js` | 165,877 | `8a631b1537927f8f2ef184d9f1b343061f47df0a32ddb8f5190d74e80d40618e` |
| `executorch_wasm_lib.wasm` | 6,562,188 | `35a9087d0b6cd86c279e7111adfd54a58156bd993928b20f63348f1043652eb4` |

## Checked with this build

Headless Chromium 141, WebGPU on SwiftShader (no `shader-f16`), with the
playground's models compared against ExecuTorch CPU references:

- Depth Anything V2 Small, MODNet, Real-ESRGAN, D-FINE S: max abs difference
  between 0 and 4.6e-3.
- SAM 2.1 Hiera Tiny: mask IoU 1.000 on a click, a second click, and a box prompt.

The playground's own end-to-end checks, through the real UI, cover Qwen3 0.6B,
Llama 3.2 1B, SAM 2.1, Depth Anything V2, D-FINE S and Whisper base. Real GPUs
and the `shader-f16` kernel paths have not been tested with this build.
