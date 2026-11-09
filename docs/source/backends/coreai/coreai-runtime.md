# Runtime Options and Cache

This page covers the Core AI backend's load-time options and how it caches specialized models. See the [overview](coreai-overview.md#runtime-integration) for building the runtime and loading a `.pte`.

## Options

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

## Compute Units

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

An unknown value fails the load with `InvalidArgument`. Each value is specialized and cached separately.

## Frequent Reshapes

Set `coreai_expect_frequent_reshapes` for models with dynamic inputs whose shapes change from call to call. It sets Core AI's `SpecializationOptions.expectFrequentReshapes`, in addition to the compute unit chosen above. Each value is specialized and cached separately.

For [ahead-of-time compiled](coreai-partitioner.md#ahead-of-time-compilation) models, also set the matching export option, `AOTCompileConfig(expect_frequent_reshapes=True)`.

## Sharing a Multi-Method Model

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

## Cache

Specialization can take much longer than the rest of a load, so its result is cached. For example, a small model that took 118 ms to load the first time took 7 ms afterwards. Two directories are involved:

| | Core AI's model cache | The backend's bookmark directory |
|---|---|---|
| Holds | The specialized models | One *bookmark* per specialized model, plus lock files |
| Managed by | Core AI | The ExecuTorch backend |
| Location | Chosen by Core AI. Currently `coreai-cache` in the app's Caches directory, for example `~/Library/Caches/coreai-cache` for a command-line tool on macOS. | `executorch_coreai` in the app's Caches directory, or `coreai_cache_dir` |
| Configurable | No | Yes, with `coreai_cache_dir` |

On a model's first load, Core AI specializes it and stores the result in its own model cache. Core AI then returns a bookmark, a small handle that identifies the model in its cache, and the backend saves it in the bookmark directory. Later loads, including from new processes, pass the bookmark back to Core AI to restore the specialized model without specializing it again.

`coreai_cache_dir` only moves the bookmarks: Core AI always stores specialized models in its own cache, and ExecuTorch can't change where. Don't modify Core AI's cache directly; use [`clear_cache`](#clearing-the-cache) to remove models from it.

Each combination of model, platform, device architecture, `coreai_compute_unit` and `coreai_expect_frequent_reshapes` is specialized and cached separately. The methods of a multi-method export share one entry.

The backend lets the system purge Core AI's model cache under storage pressure. The next load of a purged model specializes it again, which needs the model data from the `.pte`, so keep the `.pte` (and any external data files) available.

### Bookmark Directory

The bookmark directory defaults to `executorch_coreai` in the app's Caches directory. Besides bookmarks and lock files, it briefly holds a copy of a model's Core AI bundle, extracted from the `.pte`, while that model is being specialized. To use another directory, set `coreai_cache_dir`:

- Use an absolute path, on a local APFS volume, that only this backend uses.
- Don't rename, replace or delete the directory while models are loading, running or being cleared.
- If you load a model with a custom directory, pass the same directory when clearing its cache.

The backend excludes the directory from backups. It creates new directories with `NSFileProtectionCompleteUntilFirstUserAuthentication`, so on iOS the cache stays usable while the device is locked, once it has been unlocked after starting up.

### Clearing the Cache

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
