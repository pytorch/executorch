# Native runtime CMake targets

Enable `EXECUTORCH_BUILD_NATIVE=ON` to build the shared native runtime, or
`EXECUTORCH_BUILD_NATIVE_VULKAN=ON` to build it together with the Vulkan engine.
`EXECUTORCH_BUILD_NATIVE_GGML=ON` builds it together with the GGML engine.
All three options default to `OFF`; the engine options can be enabled together.

The native runtime requires C++20, a host `flatc`, the vendored FlatBuffers and
nlohmann JSON headers, and an installed libzip package with the `libzip::zip`
CMake target. Set `CMAKE_PREFIX_PATH` to the libzip installation if necessary.
Native Vulkan also requires the Vulkan submodules and `glslc`, as described in
the [Vulkan build instructions](../vulkan/README.md).

From the ExecuTorch checkout:

```sh
cmake -S . -B cmake-out -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_NATIVE_VULKAN=ON
cmake --build cmake-out --target native_api native_vulkan_engine
cmake --install cmake-out --prefix install
```

## Shared targets

| Target | Contents |
| --- | --- |
| `native_graph` | Graph IR, graph utilities, and memory planning |
| `native_package` | PTN archive, safetensors, and owned byte readers |
| `native_runtime` | Program deserialization, validation, and method metadata |
| `native_engine` | Engine host, context, and executable interfaces |
| `native_api` | Engine-independent `Model`, `Session`, and tensor API |
| `native_vulkan_passes` | Native graph rewrites for Vulkan |
| `native_vulkan_engine` | Vulkan host and executable implementation |
| `native_ggml_engine` | GGML host, graph lowering, and executable implementation |
| `vulkan_compute_api` | Vulkan device, allocator, and command API |
| `vulkan_graph_runtime` | ComputeGraph, operator registrations, and shaders |

The common native targets have no dependency on the ExecuTorch executor or a
specific engine. New engines link `native_engine` and implement its interfaces.
Applications link `native_api` and the engines they use. The Vulkan engine and
the existing `vulkan_backend` delegate share `vulkan_graph_runtime`; its static
operator and shader registrations are retained automatically when linking.

All library targets are installed in `ExecuTorchTargets`. For example:

```cmake
find_package(executorch CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE native_api native_vulkan_engine)
```

With `EXECUTORCH_BUILD_TESTS=ON`, `native_runtime_test` covers the shared runtime
and API. `native_vulkan_passes_test` covers the Vulkan graph rewrites without
requiring a GPU. When either runner below is enabled,
`native_runner_utils_test` covers their shared runner utilities. Run these with
`ctest --test-dir cmake-out -R native_`.

## GGML engine

Native GGML requires an installed GGML 0.24 or newer package providing
`ggml::ggml`, built with `GGML_CPU=ON` and `GGML_BACKEND_DL=OFF`. For example,
build the GGML sources from llama.cpp revision
`60081bb2b5b3294165a4d67c5cbeebe74c868014` (the version used by the Buck build):

```sh
cmake -S /path/to/llama.cpp/ggml -B ggml-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/ggml-install" \
  -DGGML_CPU=ON -DGGML_BACKEND_DL=OFF \
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
cmake --build ggml-build --parallel
cmake --install ggml-build
cmake -S . -B cmake-out -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_NATIVE_GGML=ON -DEXECUTORCH_BUILD_TESTS=ON \
  -DCMAKE_PREFIX_PATH="$PWD/ggml-install"
cmake --build cmake-out --target native_api native_ggml_engine native_ggml_engine_test
ctest --test-dir cmake-out -R native_ggml_engine_test --output-on-failure
cmake --install cmake-out --prefix install
```

Applications link `native_api` and `native_ggml_engine`, then create a
`ptn::GgmlEngineHost`. Its default device is CPU. For GPU execution, build GGML
with the desired backend (for example, `GGML_CUDA=ON`, `GGML_HIP=ON`, or
`GGML_METAL=ON`) and set `ptn::GgmlEngineOptions::device` to
`ptn::GgmlDevice::Gpu`. The GGML package supplies the backend libraries and their
dependencies. CPU support is required even when selecting a GPU.

`EXECUTORCH_BUILD_TESTS=ON` adds `native_ggml_engine_test`, which runs the existing
CPU engine tests, including quantized linear execution, graph caching, and
mutable buffers.

## Vision classification runner

Enable `EXECUTORCH_BUILD_NATIVE_VISION_RUNNER=ON` together with
`EXECUTORCH_BUILD_NATIVE_VULKAN=ON` to build `run_classification` from
`examples/vision`. The runner additionally requires gflags; an OSS build uses
the vendored copy, and other builds resolve an installed gflags package.
Running the runner requires a Vulkan device.

```sh
cmake -S . -B cmake-out -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_NATIVE_VULKAN=ON \
  -DEXECUTORCH_BUILD_NATIVE_VISION_RUNNER=ON
cmake --build cmake-out --target run_classification
cmake --install cmake-out --prefix install
install/bin/run_classification \
  --package mv2.ptn --input mv2_input.bin \
  --expected mv2_expected.bin --categories categories.txt
```

Export the `.ptn`, input, reference logits, and labels with
[`export.py`](../../examples/native/vision/export.py). The runner prints the
top-5 logits and, when `--expected` is given, exits nonzero if the logits
differ from the reference beyond `--rtol`/`--atol` (both default to `1e-3`).

## Native LLM runner

`EXECUTORCH_BUILD_NATIVE_LLM_RUNNER=ON` builds `run_hf_llm_vulkan` with Native
Vulkan and `run_hf_llm_ggml` with Native GGML. It requires gflags and
`tokenizers::tokenizers` (the vendored tokenizer is built automatically in an
OSS checkout; otherwise provide an installed tokenizers package with regex
lookahead support). It accepts Llama 3 TikToken tokenizer files.

To build the text runner for GGML CPU, first install GGML as above:

```sh
cmake -S . -B cmake-out -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_NATIVE_GGML=ON \
  -DEXECUTORCH_BUILD_NATIVE_LLM_RUNNER=ON \
  -DCMAKE_PREFIX_PATH="$PWD/ggml-install"
cmake --build cmake-out --target run_hf_llm_ggml
cmake --install cmake-out --prefix install
install/bin/run_hf_llm_ggml --device cpu --cpu_threads 4 \
  --package llama.ptn --tokenizer tokenizer.model \
  --prompt "The capital of France is" --max_new_tokens 32
```

The GGML runner accepts `--device cpu|gpu`, `--device_index`, `--cpu_threads`,
and `--max_cached_graphs` (default 1, 0 rebuilds each call). GPU execution uses
the backend linked into the installed GGML package. Buck provides separate
`run_hf_llm_ggml_{cpu,cuda,hip,metal}` targets; the Metal target uses xplat.

With `EXECUTORCH_BUILD_TESTS=ON`, `native_llm_runner_utils_test` exercises the
LLM generation utilities.
