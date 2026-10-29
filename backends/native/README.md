# ExecuTorch Native

> [!NOTE]
> Experimental. The Python API, the `.ptn` format, and operator coverage may change.

ExecuTorch Native lowers a PyTorch model to a single `.ptn` file and runs it on
one of several engines. In a `.pte`, delegates lower the graph ahead of time for
specific backends. A `.ptn` keeps the whole graph without backend-specific
information, and each engine lowers it when compiling a method.

```
.pte:  export -> [quantize] -> to_edge -> to_backend -> to_executorch -> .pte (+ .ptd)
.ptn:  export -> [quantize] -> to_native -> .ptn
```

## Export

```python
import torch
from executorch.exir.native import to_native

ep = torch.export.export(model, example_inputs)
to_native(ep).save("model.ptn")

# Several methods in one file.
to_native({"forward": ep_forward, "prefill": ep_prefill}).save("model.ptn")
```

The partitioner accepts core ATen ops plus `linear`, `matmul`, `addmm`,
`scaled_dot_product_attention`, `rms_norm`, `native::rope`, and TorchAO affine
quantize/dequantize ops. TorchAO int4 group-wise weights of `linear` and
`embedding`, and GGUF k-quant weights from `executorch.extension.llm.export.gguf`, are
stored packed with their scheme in tensor metadata. Each method must lower wholly
to one delegate; `to_native` raises otherwise.

Immutable constant keys are namespaced as `<method>/<key>`, even for a single
method. A key mutated by any method becomes `state/<key>` in every method, so
methods share its runtime state. Method names may not be `state` or start with
`state/`. `to_native` raises if namespaced keys collide.

A `.ptn` is an uncompressed zip with fixed entry names:

| Entry | Contents |
| --- | --- |
| `program.ptg` | Methods as graphs; FlatBuffer defined by `serialization/native_graph.fbs`. |
| `program.safetensors` | Constants. Byte-identical immutable tensors are stored once. |
| `aliases.json` | Deduplicated key to owner key; present only when needed. |

## Samples (optional)

A `.ptn` may carry recorded method calls: inputs, and optionally the outputs and
intermediate values they produced. A sample is an ordered sequence of calls, each
naming the method it runs, so a stateful sequence such as an LLM's prefill
followed by decode steps can be checked whether it uses one method or several.
Samples serve reference checks, layer-by-layer numerical debugging, calibration
and benchmarking. Program loaders ignore them.

```python
from executorch.exir.native import Call, read_samples, Sample, SampleSet, to_native, write_samples

to_native(ep).save("model.ptn")
golden = SampleSet([Sample([Call("forward", [x], [model(x)])])])
write_samples("model.ptn", "golden", golden)
read_samples("model.ptn")       # {"golden": SampleSet(...)}
```

`write_samples` appends a set to any existing `.ptn`, not only one this process
saved, and replaces the file only once the set is complete. `read_samples`
validates every set against the rules below, except that each call's method
exists in the program and its intermediate names are values of that method.

### Layout

```
samples/<set>/index.json
samples/<set>/data.safetensors
```

- `<set>` is an identifier with no meaning to the format, matching
  `[A-Za-z0-9_-]+`.
- Each set has exactly these two members, and nothing else lives under `samples/`.
- Members are stored (uncompressed). The `data.safetensors` data section starts
  64-byte aligned in the file, like `program.safetensors`.
- Adding a set only appends members; removing one means rewriting the zip.

### `index.json`

```
{ "version":     "1.0",                required, "MAJOR.MINOR"
  "description": str,                  optional
  "source":      str,                  optional, producer of the outputs
  "tol":         [atol, rtol],         optional, default for every call
  "samples":     [Sample, ...] }       required, non-empty

Sample = { "calls":         [Call, ...],         required, non-empty, run in order
           "tol":           [atol, rtol] }       optional, overrides the set's

Call   = { "method":        str,                 required, a program method
           "in":            [Value, ...],        required, positional arguments
           "out":           [Value, ...],        optional, positional outputs
           "intermediates": {name: Key, ...},    optional
           "tol":           [atol, rtol] }       optional, overrides the sample's

Value  = Key | number | true | false | [Value, ...]
Key    = str naming a tensor in the set's data.safetensors
```

- Each sample starts from the program's initial state. Samples are independent and
  may run in any order; the calls within a sample run in order with no reset
  between them, so each sees the state the previous one left, whichever method
  that call ran.
- A tensor's key is its JSON path in the index, e.g. `samples[0].calls[1].in[2]`.
  Below a call that is `.in[k]`, `.out[k]` or `.intermediates.<name>`, with
  `[l]` appended for each level of list nesting. Every key resolves, and every
  tensor is referenced exactly once.
- A JSON integer is an int scalar, any other number a float.
- Intermediate names are top-level SSA value names of the call's method
  (`[A-Za-z_][A-Za-z0-9_]*`).
  Producers record what they can; consumers compare the names both sides have.
- Tolerances are finite and non-negative, and advisory: a consumer may override them.
  `write_samples` warns when atol or rtol is above 1e-3.
- Readers reject another major `version` and accept any minor one: a minor bump
  only adds fields, and readers ignore unknown fields and the tensors only those
  fields reference.

Not in v1:

- Tensors shared across sets.
- Values inside nested graphs.
- Quantized method inputs and outputs (a `QuantSpec` on graph I/O). Integer and
  float I/O of quantized models is supported; torch quantized dtypes and tensor
  subclasses are not.

TODO: specify quantized I/O as the storage form the method binds (packed for
sub-byte), with quant parameters read from the program rather than the sample.

## Runtime

The C++ runtime reads a `.ptn` into an in-memory graph and hands each method to
an engine through the interfaces in `runtime/engine/Engine.h`:

- `EngineHost`: process-wide device resources.
- `EngineContext`: state for one loaded program, including mutable buffers.
- `EngineExecutable`: one compiled method (`set_input`, `execute`, `get_output`).

An engine rejects a method whose semantics it cannot fully honor.

| Engine | Source | Notes |
| --- | --- | --- |
| Vulkan | `runtime/vulkan/` | Builds on the ET-VK `ComputeGraph` and shaders. |
| ggml | `runtime/engine/ggml/` | CPU, CUDA, HIP, and Metal variants. |

## Building with CMake

Enable `EXECUTORCH_BUILD_NATIVE=ON` to build the shared native runtime. With it,
`EXECUTORCH_BUILD_NATIVE_VULKAN=ON` adds the Vulkan engine and
`EXECUTORCH_BUILD_NATIVE_GGML=ON` adds the GGML engine. All three options default
to `OFF`; the engine options are ignored unless `EXECUTORCH_BUILD_NATIVE=ON` and
can be enabled together.

The native runtime requires C++20, a host `flatc`, the vendored FlatBuffers and
nlohmann JSON headers, and an installed libzip package with the `libzip::zip`
CMake target. Set `CMAKE_PREFIX_PATH` to the libzip installation if necessary.
Native Vulkan also requires the Vulkan submodules and `glslc`, as described in
the [Vulkan build instructions](../vulkan/README.md).

From the ExecuTorch checkout:

```sh
cmake -S . -B cmake-out -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_NATIVE=ON -DEXECUTORCH_BUILD_NATIVE_VULKAN=ON
cmake --build cmake-out --target native_api native_vulkan_engine
cmake --install cmake-out --prefix install
```

### Shared targets

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

The `Test Native Engines` CI workflow has separate Linux jobs for the shared
runtime, GGML, and Vulkan, with each test suite run once. To reproduce with
dependencies installed, run `bash .ci/scripts/test-native.sh runtime`,
`bash .ci/scripts/test-native.sh ggml`, or
`bash .ci/scripts/test-native.sh vulkan` from the OSS checkout. The GGML job
also runs `native_llm_runner_utils_test` and executes `run_hf_llm_ggml` on a tiny
seeded Hugging Face Llama with a generated byte tokenizer. Its eight generated
token IDs must match eager PyTorch, covering prefill and KV-cache decoding without
downloading a model. The GGML job requires the Python export dependencies
installed by the `Test Native Engines` workflow. The Vulkan job builds
`run_classification` and `run_hf_llm_vulkan`; its tests do not require a GPU.

### GGML engine

Native GGML builds the GGML subtree of the llama.cpp submodule at
`backends/native/third-party/llama.cpp`, pinned to
`620d2225cc436ab0b001e3200e590e3b286ef9ab`. This matches the Buck dependency
used to validate this CMake integration. The build requires `GGML_CPU=ON`
and `GGML_BACKEND_DL=OFF`, which are the defaults.

```sh
git submodule update --init backends/native/third-party/llama.cpp
cmake -S . -B cmake-out -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_NATIVE=ON -DEXECUTORCH_BUILD_NATIVE_GGML=ON \
  -DEXECUTORCH_BUILD_TESTS=ON
cmake --build cmake-out --target native_api native_ggml_engine native_ggml_engine_test
ctest --test-dir cmake-out -R native_ggml_engine_test --output-on-failure
cmake --install cmake-out --prefix install
```

Applications link `native_api` and `native_ggml_engine`, then create a
`ptn::GgmlEngineHost`. Its default device is CPU. For GPU execution, configure
ExecuTorch with the desired GGML backend (for example, `GGML_CUDA=ON`,
`GGML_HIP=ON`, or `GGML_METAL=ON`) and set `ptn::GgmlEngineOptions::device` to
`ptn::GgmlDevice::Gpu`. CPU support is required even when selecting a GPU.

The CMake installation includes the GGML libraries and package configuration.
It also copies the submodule's upstream MIT license, including its copyright
notice, to `share/licenses/executorch/ggml/LICENSE` under the installation
prefix (or the corresponding `CMAKE_INSTALL_DATADIR` location).

`EXECUTORCH_BUILD_TESTS=ON` adds `native_ggml_engine_test`, which runs the existing
CPU engine tests, including quantized linear execution, graph caching, and
mutable buffers.

### Vision classification runner

Enable `EXECUTORCH_BUILD_NATIVE_VISION_RUNNER=ON` together with
`EXECUTORCH_BUILD_NATIVE_VULKAN=ON` to build `run_classification` from
`examples/vision`. The runner additionally requires gflags; an OSS build uses
the vendored copy, and other builds resolve an installed gflags package.
Running the runner requires a Vulkan device.

```sh
cmake -S . -B cmake-out -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_NATIVE=ON -DEXECUTORCH_BUILD_NATIVE_VULKAN=ON \
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

To build the text runner for GGML CPU, initialize the llama.cpp submodule as above:

```sh
cmake -S . -B cmake-out -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_NATIVE=ON -DEXECUTORCH_BUILD_NATIVE_GGML=ON \
  -DEXECUTORCH_BUILD_NATIVE_LLM_RUNNER=ON
cmake --build cmake-out --target run_hf_llm_ggml
cmake --install cmake-out --prefix install
install/bin/run_hf_llm_ggml --device cpu --cpu_threads 4 \
  --package llama.ptn --tokenizer tokenizer.model \
  --prompt "The capital of France is" --max_new_tokens 32
```

The GGML runner accepts `--device cpu|gpu`, `--device_index`, `--cpu_threads`,
and `--max_cached_graphs` (default 1, 0 rebuilds each call). For GPU execution,
enable the desired `GGML_*` backend when configuring ExecuTorch, as above.

With `EXECUTORCH_BUILD_TESTS=ON`, `native_llm_runner_utils_test` exercises the
LLM generation utilities.
