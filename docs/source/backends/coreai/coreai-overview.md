# Core AI Backend

The Core AI delegate is the ExecuTorch backend for Apple's Core AI framework, available on macOS 27 and iOS 27. At export time, it converts the parts of your model that Core AI supports into a Core AI model, using [coreai-torch](https://pypi.org/project/coreai-torch/), and embeds that model in the `.pte` file. At runtime, Core AI compiles the model for the device it's running on, a step called *specialization*, and runs it on the CPU, GPU or Neural Engine.

::::{warning}
The Core AI delegate is experimental and under active development. It has no backward- or forward-compatibility guarantees: APIs, compile specs, runtime options and the serialized format may change between ExecuTorch versions.
::::

## Features

- Runs on the CPU, GPU and Neural Engine. A runtime option restricts Core AI to the CPU or makes it prefer one compute unit.
- FP32 and FP16 models.
- INT8 weight and activation quantization; INT8, INT4, INT2, FP8 and FP4 weight-only quantization; and 4-, 6- and 8-bit palettization, through [coreai-opt](https://pypi.org/project/coreai-opt/).
- Dynamic shapes, and enumerated shapes, which compile a dynamic dimension to a fixed set of sizes.
- Multi-method models whose methods share one Core AI model, so weights they have in common are stored once and can be loaded once.
- Optional ahead-of-time compilation for specific device architectures.
- A persistent cache of specialized models, so only the first load of a model pays for specialization.

## Target Requirements

One of:

- [macOS](https://developer.apple.com/macos) 27 or later on Apple silicon.
- [iOS](https://developer.apple.com/ios) 27 or later on a physical device. The iOS simulator doesn't include Core AI.

## Development Requirements

- A Mac with [Xcode](https://developer.apple.com/xcode/) 27, which provides the Apple SDK 27 and its Swift toolchain.
- CMake 3.31 or later, with the Ninja or Xcode generator.
- The Core AI Python packages (`coreai-core`, `coreai-torch` and `coreai-opt`), at the versions pinned by the backend:

  ```bash
  pip install -r backends/apple/coreai/requirements.txt
  ```

- For [ahead-of-time compilation](coreai-partitioner.md#ahead-of-time-compilation) only, the Metal Toolchain, which provides `xcrun coreai-build`:

  ```bash
  xcodebuild -downloadComponent MetalToolchain
  ```

----

## Using the Core AI Backend

To lower a model to Core AI, pass a `CoreAIPartitioner` to `to_edge_transform_and_lower`, along with the backend's default transform passes and compile config. The example below lowers MobileNet V2:

```python
import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.apple.coreai import (
    CoreAIPartitioner,
    get_default_compile_config,
    get_default_passes,
)
from executorch.exir import to_edge_transform_and_lower

mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224),)

et_program = to_edge_transform_and_lower(
    torch.export.export(mobilenet_v2, sample_inputs),
    transform_passes=get_default_passes(),
    partitioner=[CoreAIPartitioner()],
    compile_config=get_default_compile_config(),
).to_executorch()

with open("mv2_coreai.pte", "wb") as file:
    et_program.write_to_file(file)
```

Always pass both `get_default_passes()` and `get_default_compile_config()`. The passes adapt the graph to what coreai-torch can convert, for example by casting 64-bit inputs to 32 bits. The compile config turns off ExecuTorch features that coreai-torch doesn't support, such as dim-order ops.

Ops that Core AI can't run stay on ExecuTorch's CPU kernels. See [Partitioner API](coreai-partitioner.md) for the partitioner's options, including ahead-of-time compilation, enumerated shapes and multi-method models.

### Precision

Core AI runs the model in the dtype it was exported with: unlike Core ML, it doesn't convert FP32 models to FP16. To run in FP16, either cast the exported program with coreai-opt's `cast_fp32_to_fp16` (from `coreai_opt.casting`) before lowering, or use the FP16 export recipe:

```python
from executorch.backends.apple.coreai.recipes import CoreAIRecipeType
from executorch.export import ExportRecipe, export

session = export(
    model=mobilenet_v2,
    example_inputs=[sample_inputs],
    export_recipe=ExportRecipe.get_recipe(CoreAIRecipeType.FP16),
)
session.save_to_pte("mv2_coreai_fp16.pte")
```

The FP16 recipe casts the whole program, so the model's inputs and outputs become FP16 too; pass FP16 tensors when you run it. Values that FP16 can't represent stay FP32. `CoreAIRecipeType.FP32` lowers the model in FP32.

----

## Quantization

The Core AI backend supports INT8 weight and activation quantization, INT8, INT4, INT2, FP8 and FP4 weight-only quantization, and palettization. All are provided by coreai-opt and applied before `torch.export`. See [Core AI Quantization](coreai-quantization.md) for details.

----

## Runtime Integration

### Python (pybindings)

Build the Python runtime with the Core AI delegate, on macOS 27, from the repository root:

```bash
CMAKE_GENERATOR=Ninja \
  CMAKE_ARGS="-DEXECUTORCH_BUILD_COREAI=ON -DEXECUTORCH_BUILD_COREML=OFF -DEXECUTORCH_BUILD_MLX=OFF -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0" \
  python install_executorch.py --editable
pip install -r backends/apple/coreai/requirements.txt
```

Then load and run a Core AI `.pte` with the ExecuTorch runtime API:

```python
import torch
from executorch.runtime import Runtime

program = Runtime.get().load_program("mv2_coreai.pte")
method = program.load_method("forward")
outputs = method.execute([torch.randn(1, 3, 224, 224)])
```

### C++ (CMake)

Configure ExecuTorch with `EXECUTORCH_BUILD_COREAI=ON` and an OS 27 deployment target. The backend also requires `EXECUTORCH_BUILD_EXTENSION_DATA_LOADER`:

```bash
cmake -S . -B cmake-out -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/cmake-out/install" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0 \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DEXECUTORCH_BUILD_COREAI=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON
cmake --build cmake-out --target install
```

For iOS, also set `CMAKE_SYSTEM_NAME=iOS` and `CMAKE_OSX_SYSROOT=iphoneos`. Only arm64 is supported, and each Ninja build directory builds one architecture.

In your app's CMake project, find the installed package (for example by passing `-DCMAKE_PREFIX_PATH=<executorch>/cmake-out/install`) and link the `coreaidelegate` target:

```cmake
find_package(executorch REQUIRED)
target_link_libraries(my_target PRIVATE executorch coreaidelegate)
```

Link the `coreaidelegate` target, not its archive file. The target registers the backend with ExecuTorch and pulls in the Swift runtime and the Core AI framework from the SDK and toolchain your project uses. No other setup is needed: a Core AI `.pte` runs on the registered backend.

With `EXECUTORCH_BUILD_SHARED=ON`, the delegate is a shared library linked against the `executorch_shared` runtime, and your app must link that same runtime. Swift Package Manager and XCFramework distribution aren't available yet.

### Calling from Swift

Core AI model loading, inference and cache operations are synchronous. In Swift, run them on a background `DispatchQueue` or a dedicated thread. Don't call them directly inside a Swift `Task` or `Task.detached`, which can hang your app.

From async code, submit the whole ExecuTorch call with `DispatchQueue.async` and await its result with a continuation. Don't block the task with `DispatchQueue.sync` or a semaphore. Keep input tensors alive and unchanged until the call finishes, even if the awaiting task is cancelled.

----

## Runtime Options

The Core AI backend accepts four optional load-time options, passed under the `CoreAIBackend` key of the load-time backend options:

| Option | Default | Effect |
|---|---|---|
| `coreai_cache_dir` | `executorch_coreai` in the app's Caches directory | Where the backend keeps its bookmarks to Core AI's cached models. |
| `coreai_compute_unit` | `default` | Which compute units Core AI may use: `default`, `cpu_only`, `prefer_cpu`, `prefer_gpu` or `prefer_neural_engine`. |
| `coreai_expect_frequent_reshapes` | `false` | Optimize for input shapes that change between calls. |
| `coreai_share_multifunction_model` | `false` | Load a multi-method model once and share it across its methods. |

Core AI caches each specialized model, so only a model's first load pays for specialization. See [Runtime Options and Cache](coreai-runtime.md) for how to pass the options, what each one does, how the cache works and how to clear it.

## Reference

**→{doc}`/backends/coreai/coreai-troubleshooting` — Debug common issues.**

**→{doc}`/backends/coreai/coreai-partitioner` — Partitioner options.**

**→{doc}`/backends/coreai/coreai-runtime` — Runtime options and cache.**

**→{doc}`/backends/coreai/coreai-quantization` — Supported quantization schemes.**

**→{doc}`/backends/coreai/coreai-op-support` — Supported operators.**

```{toctree}
:maxdepth: 2
:hidden:
:caption: Core AI Backend
coreai-troubleshooting
coreai-partitioner
coreai-runtime
coreai-quantization
coreai-op-support
```
