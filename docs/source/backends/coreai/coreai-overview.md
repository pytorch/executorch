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

The Core AI backend accepts these options when a model is loaded. All are optional.

| Option | Type | Default | Effect |
|---|---|---|---|
| `coreai_cache_dir` | string | `executorch_coreai` in the app's Caches directory | Where the backend keeps its bookmarks to Core AI's cached models. See [Cache](#cache). |
| `coreai_compute_unit` | string | `default` | Which compute units Core AI may use. See [Compute units](#compute-units). |
| `coreai_expect_frequent_reshapes` | bool | `false` | Optimize for input shapes that change between calls. See [Frequent reshapes](#frequent-reshapes). |
| `coreai_share_multifunction_model` | bool | `false` | Load a multi-method model once and share it across its methods. See [Sharing a multi-method model](#sharing-a-multi-method-model). |

Options are grouped by backend; Core AI's are read from the `CoreAIBackend` entry. In C++, pass them to `Module::load` through a `LoadBackendOptionsMap`:

```cpp
#include <executorch/extension/module/module.h>
#include <executorch/runtime/backend/options.h>

using executorch::extension::Module;
using namespace executorch::runtime;

Error load_model(Module& module) {
  BackendOptions<2> options;
  ET_CHECK_OK_OR_RETURN_ERROR(
      options.set_option("coreai_compute_unit", "prefer_neural_engine"));
  ET_CHECK_OK_OR_RETURN_ERROR(
      options.set_option("coreai_expect_frequent_reshapes", true));
  LoadBackendOptionsMap map;
  ET_CHECK_OK_OR_RETURN_ERROR(map.set_options("CoreAIBackend", options.view()));
  return module.load(map);
}
```

Here `module` is a `Module` constructed from the `.pte` path. String values are limited to 255 bytes.

In Python, pass them to `load_program`:

```python
program = Runtime.get().load_program(
    "mv2_coreai.pte",
    backend_options={"CoreAIBackend": {"coreai_compute_unit": "cpu_only"}},
)
```

Options apply to the model being loaded, so different models in one app can use different options.

### Compute units

`coreai_compute_unit` chooses which compute units Core AI may run the model on. Each value maps to one of the configurations the Core AI SDK provides:

| Value | Allowed compute units | Core AI `SpecializationOptions` |
|---|---|---|
| `default` | CPU, GPU, Neural Engine | `.default` |
| `cpu_only` | CPU | `.cpuOnly` |
| `prefer_cpu` | CPU, GPU, Neural Engine, preferring the CPU | `init(preferredComputeUnitKind: .cpu)` |
| `prefer_gpu` | CPU, GPU, Neural Engine, preferring the GPU | `init(preferredComputeUnitKind: .gpu)` |
| `prefer_neural_engine` | CPU, GPU, Neural Engine, preferring the Neural Engine | `init(preferredComputeUnitKind: .neuralEngine)` |

- **`cpu_only` is the only restriction.** The `prefer_*` values make Core AI run as much of the model as it can on that unit, and fall back to the other units for operations the preferred unit can't run.
- **There are no other combinations.** The SDK doesn't offer, for example, CPU and Neural Engine only.
- **Core AI only uses the hardware the device has.** On a machine without a GPU or Neural Engine, `default` and `cpu_only` behave the same.

An unknown value fails the load with `InvalidArgument`. Each value is specialized and cached separately.

### Frequent reshapes

Set `coreai_expect_frequent_reshapes` for models with dynamic inputs whose shapes change from call to call. It sets Core AI's `SpecializationOptions.expectFrequentReshapes`, in addition to the compute unit chosen above. Each value is specialized and cached separately.

For [ahead-of-time compiled](coreai-partitioner.md#ahead-of-time-compilation) models, also set the matching export option, `AOTCompileConfig(expect_frequent_reshapes=True)`.

### Sharing a multi-method model

In a [multi-method export](coreai-partitioner.md#multi-method-models), every method's Core AI delegate uses the same Core AI model. By default, though, each delegate loads its own copy, with its own weights in memory. With `coreai_share_multifunction_model` set, the first delegate to load creates the model, and the others reuse it until the last one is destroyed.

Resident memory after running each delegate once, for delegates of the same 33.6 MB model on an M4 Max:

| Delegates | Not shared | Shared |
|---|---|---|
| 1 | +91 MB | +91 MB |
| 2 | +157 MB | +92 MB |
| 4 | +288 MB | +92 MB |
| 8 | +551 MB | +92 MB |

Before turning it on, note:

- **Calls run one at a time.** Delegates that share a model don't execute concurrently, because the Core AI SDK doesn't document concurrent execution of one model as safe.
- **Sharing is process-wide.** Separate `Module`s that load the same `.pte` with the option set share one model, so their `execute` calls wait for each other.
- **Only identical specializations share.** Loads share a model only if they use the same bookmark directory (or a symlink to it), the same `coreai_compute_unit` and the same `coreai_expect_frequent_reshapes`. Otherwise each loads its own.
- **It requires a multi-method export.** For a model not exported with `CoreAIPartitioner(multifunction=True)`, the option fails the load with `InvalidArgument`.

### Cache

Specialization can take much longer than the rest of a load, so its result is cached. For example, a small model that took 118 ms to load the first time took 7 ms afterwards. Two directories are involved:

| | Core AI's model cache | The backend's bookmark directory |
|---|---|---|
| Holds | The specialized models | One *bookmark* per specialized model, plus lock files |
| Managed by | Core AI | The ExecuTorch backend |
| Location | `coreai-cache` in the app's Caches directory, such as `~/Library/Caches/coreai-cache` for a command-line tool on macOS | `executorch_coreai` in the app's Caches directory, or `coreai_cache_dir` |
| Configurable | No | Yes, with `coreai_cache_dir` |

On a model's first load, Core AI specializes it and stores the result in its own model cache. Core AI then returns a bookmark, a small handle that identifies the model in its cache, and the backend saves it in the bookmark directory. Later loads, including from new processes, pass the bookmark back to Core AI to restore the specialized model without specializing it again.

`coreai_cache_dir` only moves the bookmarks: Core AI always stores specialized models in its own cache, and ExecuTorch can't change where. Don't modify Core AI's cache directly; use [`clear_cache`](#clearing-the-cache) to remove models from it.

Each combination of model, platform, device architecture, `coreai_compute_unit` and `coreai_expect_frequent_reshapes` is specialized and cached separately. The methods of a multi-method export share one entry.

The backend lets the system purge Core AI's model cache under storage pressure. The next load of a purged model specializes it again, which needs the model data from the `.pte`, so keep the `.pte` (and any external data files) available.

#### Bookmark directory

The bookmark directory defaults to `executorch_coreai` in the app's Caches directory. Besides bookmarks and lock files, it briefly holds a copy of a model's Core AI bundle, extracted from the `.pte`, while that model is being specialized. To use another directory, set `coreai_cache_dir`:

- Use an absolute path, on a local APFS volume, that only this backend uses.
- Don't rename, replace or delete the directory while models are loading, running or being cleared.
- If you load a model with a custom directory, pass the same directory when clearing its cache.

The backend excludes the directory from backups. It creates new directories with `NSFileProtectionCompleteUntilFirstUserAuthentication`, so on iOS the cache stays usable while the device is locked, once it has been unlocked after starting up.

#### Clearing the cache

The backend provides C++ functions to remove cached models, declared in `executorch/backends/apple/coreai/runtime/coreai_cache.h`:

```cpp
namespace executorch::backends::coreai {
runtime::Error clear_cache(const char* coreai_cache_dir = nullptr);
runtime::Error clear_cache_for_pte(
    runtime::DataLoader& pte, const char* coreai_cache_dir = nullptr);
runtime::Error clear_cache_for_pte(
    const char* pte_path, const char* coreai_cache_dir = nullptr);
}
```

- `clear_cache` removes every model that has a bookmark in the bookmark directory.
- `clear_cache_for_pte` removes the models for one `.pte`, for every device, architecture and option combination they were loaded with.

Both use each bookmark to remove the specialized model from Core AI's model cache, then delete the bookmark. They only reach models with a bookmark in the given directory. Without a directory argument, they use the default bookmark directory. They run synchronously, and require that nothing else is using the affected models: unload them first, and make sure no process loads, runs or clears Core AI models in that cache directory at the same time. The next load of a cleared model specializes it again.

## Reference

- {doc}`/backends/coreai/coreai-troubleshooting`: Debug common issues.
- {doc}`/backends/coreai/coreai-partitioner`: Partitioner options, ahead-of-time compilation, enumerated shapes and multi-method models.
- {doc}`/backends/coreai/coreai-quantization`: Supported quantization schemes.
- {doc}`/backends/coreai/coreai-op-support`: Supported operators.

```{toctree}
:maxdepth: 2
:hidden:
:caption: Core AI Backend
coreai-troubleshooting
coreai-partitioner
coreai-quantization
coreai-op-support
```
