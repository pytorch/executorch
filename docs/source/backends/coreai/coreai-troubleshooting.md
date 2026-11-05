# Troubleshooting

This page describes common issues when using the Core AI backend and how to debug them.

## Debug Logging

### At export

The partitioner logs at `INFO` level which ops it leaves out of the delegate, and why. Turn on Python logging to see them:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### At runtime

Build ExecuTorch with `EXECUTORCH_ENABLE_LOGGING=ON` and `EXECUTORCH_LOG_LEVEL=Info` to log what happens to each model in the [cache](coreai-runtime.md#cache):

- cache lookups, and whether the model was restored from its bookmark or missed (no bookmark, or Core AI couldn't restore it)
- specialization starting and finishing, and the bookmark being saved
- a shared model being reused
- cache clearing

Each message includes the model's cache key, so you can follow one model across several delegates. These messages don't log model data or individual inferences.

## Export Errors

| Error message contains | Cause and fix |
|---|---|
| `CoreAIPartitioner must be used with to_edge_transform_and_lower()` | The `to_edge()` + `to_backend()` workflow isn't supported. Use `to_edge_transform_and_lower()`. |
| `multifunction=True must be set on the CoreAIPartitioner of every method lowered to Core AI` | In a [multi-method export](coreai-partitioner.md#multi-method-models), some methods' partitioners don't set `multifunction=True`. The message lists them. |
| `multifunction requires identical Core AI compile specs in every partition` | In a multi-method export, the methods' partitioners use different `aot_compile_config` or `min_deployment_version` values. Make them match; only `input_enumerations` can differ. |
| `generates the function name '<name>' twice` | Two partitions in a multi-method export would get the same Core AI function name. Rename one of the methods. |
| `Core AI needs every dim to be positive` | An [enumerated size](coreai-partitioner.md#enumerated-shapes) makes another dimension computed from it zero or negative. Remove that size from `input_enumerations`. |
| `Ninja is required to load C++ extensions` | coreai-opt's palettizer builds a C++ extension the first time it runs. Install `ninja` (`pip install ninja`) and make sure it's on your `PATH`. |

## Loading Fails

### The device's architecture is missing

A model compiled [ahead of time](coreai-partitioner.md#ahead-of-time-compilation) only runs on the device architectures it was compiled for. On any other device, loading fails with:

```
Core AI AOT has no bundle for device architecture <arch>; available architectures: [<archs>]; re-export the model for this device architecture
```

To fix it, either add that architecture to `AOTCompileConfig(architectures=...)`, leave `architectures` unset to compile for every architecture the platform supports, or export without `aot_compile_config` to use the portable format, which runs on any device.

### Loading fails with `NotSupported`: stateful functions

A model that updates a buffer in place, such as a KV cache, exported with the default `take_over_mutable_buffer=True` produces a stateful Core AI model, which the runtime doesn't run yet. Loading fails, and the log shows:

```
Stateful functions are not supported
```

Re-export with `CoreAIPartitioner(take_over_mutable_buffer=False)`, which keeps the buffers and their updates in ExecuTorch. See [Stateful models](coreai-partitioner.md#stateful-models).

### Loading fails with `InvalidArgument`

Check the runtime options:

- **`coreai_share_multifunction_model` is set, but the model isn't a multi-method export.** Sharing requires a model exported with `CoreAIPartitioner(multifunction=True)`. The log shows:

  ```
  coreai_share_multifunction_model requires a multi-function export (CoreAIPartitioner(multifunction=True))
  ```

- **`coreai_compute_unit` has an unknown value.** Use `default`, `cpu_only`, `prefer_cpu`, `prefer_gpu` or `prefer_neural_engine`.
- **`coreai_cache_dir` isn't an absolute path.**

## Loading Is Slow

The first load of a model is slow because Core AI specializes it for the device. Later loads restore it from the [cache](coreai-runtime.md#cache) and are much faster. If every load is slow:

- **Check the bookmark directory.** Loads only find each other's bookmarks if they use the same `coreai_cache_dir`. A temporary or per-launch directory means no load finds a bookmark to restore from.
- **Check the options.** Each `coreai_compute_unit` and `coreai_expect_frequent_reshapes` value is cached separately.
- **Look for cache misses in the [runtime logs](#at-runtime).** The system may have purged Core AI's cache under storage pressure, which forces the next load to specialize again.

Models with [enumerated shapes](coreai-partitioner.md#enumerated-shapes) also load more slowly the more shapes they enumerate, since the delegate prepares one function per shape.

## A Call Fails

### With `InvalidArgument`

A model exported with `input_enumerations` only accepts the exact shapes it enumerates. A call with any other shape fails before anything runs, and the log shows the shapes it received:

```
Core AI inputs with shapes <shapes> match none of the <N> enumerated shapes this delegate was exported with
```

Call the model with one of the enumerated shapes, or re-export with the size you need added.

### With `NotSupported`

Delegate inputs and outputs must be `float16`, `float32`, `int32` or `bool` tensors that are contiguous, use the default dim order and are in CPU memory. The log names the requirement that wasn't met, for example:

```
Core AI runtime supports FP16, FP32, INT32 and BOOL tensors only
```

See [Delegate boundary](coreai-op-support.md#delegate-boundary).

## The App Hangs When Calling from Swift

Core AI loading, inference and cache operations are synchronous, and can hang your app if called directly inside a Swift `Task` or `Task.detached`. Run them on a background `DispatchQueue` or a dedicated thread instead. See [Calling from Swift](coreai-overview.md#calling-from-swift).

## Recovering from Repeated Core AI Errors

When Core AI fails to restore or remove a cached model, the backend keeps that model's bookmark so a later attempt can retry; it can't tell whether the error is temporary. If loading or [clearing](coreai-runtime.md#clearing-the-cache) one model keeps failing:

1. Stop every process that uses the bookmark directory.
2. Find the model's cache key in the [runtime logs](#at-runtime). Move `bookmarks/<key>.bookmark` and the `sources/<key>` directory, if present, out of the bookmark directory, and keep them as a backup. Don't touch the `locks` directory.
3. Load the model again. The backend specializes it from scratch and saves a new bookmark.

This doesn't remove the old entry from Core AI's cache or free its disk space. Keep the backup: its bookmark may be the only remaining handle to that entry. Only do this while no model is loading, running or being cleared.
