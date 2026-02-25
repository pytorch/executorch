# Android SDK Tech Debt Tracker

A summary of known issues in `extension/android/` and the progress made so far. Contributions and feedback are welcome.

## Resolved

### Correctness Fixes

| Issue | Impact | PR |
|-------|--------|-----|
| JNI read `mTypeCode` instead of `mData` for non-tensor EValues | Incorrect inference results for int/double/bool types | [#17603](https://github.com/pytorch/executorch/pull/17603) |
| Global `token_buffer` shared across LLM generate calls | Data corruption when multiple LlmModule instances run concurrently | [#17608](https://github.com/pytorch/executorch/pull/17608) |
| Unchecked `method_meta().get()` in `getUsedBackends()` JNI | Undefined behavior / native crash on error | [#17631](https://github.com/pytorch/executorch/pull/17631) |
| `EValue.toByteArray()` used `toString()` instead of `toStr()` for strings | String serialization round-trip was broken | [#17609](https://github.com/pytorch/executorch/pull/17609) |
| `Tensor.numel()` computed product in `int` despite `long` return type | Silent overflow for large tensors | [#17633](https://github.com/pytorch/executorch/pull/17633) |
| Check-after-use in JNI scalar type lookup (`.count()` after `.at()`) | Potential C++ exception / crash | [#17636](https://github.com/pytorch/executorch/pull/17636) |

### Build & Size Improvements

| Issue | Impact | PR |
|-------|--------|-----|
| 11,000+ unnecessary dynamic symbol exports, missing `--gc-sections` | Exported symbols: 11,403 → 6. AAR size (2 ABIs): 12.4 MB → 6.9 MB (**-44%**) | [#17614](https://github.com/pytorch/executorch/pull/17614) |
| Deprecated `jcenter()` repository | Dependency resolution failures on clean builds | [#17606](https://github.com/pytorch/executorch/pull/17606) |
| Duplicate `log.cpp` in training and llama CMake targets | Redundant compilation, ODR risk | [#17634](https://github.com/pytorch/executorch/pull/17634) |
| Build config: Java 8 target, outdated test deps, dead Gradle properties | Toolchain modernization (Java 11, compileSdk 35, latest test deps) | [#17636](https://github.com/pytorch/executorch/pull/17636) |

### Code Quality & API

| Issue | Impact | PR |
|-------|--------|-----|
| `getMethodMetadata()` was a side-effecting getter that called native JNI and mutated state on every invocation | Thread-safety race + surprising API behavior | [#17652](https://github.com/pytorch/executorch/pull/17652) |
| File validation boilerplate duplicated across Module, LlmModule, TrainingModule | Inconsistent error handling | [#17648](https://github.com/pytorch/executorch/pull/17648) |
| `utf8_check_validity()` copy-pasted in llama and ASR JNI layers | Code duplication across JNI files | [#17648](https://github.com/pytorch/executorch/pull/17648) |
| QNN AAR published but README didn't mention it | Users didn't know about QNN Maven dependency | [#17681](https://github.com/pytorch/executorch/pull/17681) |

---

## To Do

### Thread Safety

**`Module.java` ReentrantLock coverage is incomplete** — The lock covers `execute()` and `loadMethod()`, which protects the most critical paths. However, several native calls remain unprotected: `getMethods()`, `getUsedBackends()`, `readLogBuffer()`, `etdump()`. Additionally, `destroy()` uses `tryLock()` which silently does nothing if the lock is held — the caller has no way to know resources weren't freed.

**No thread safety in `LlmModule`** — No synchronization at all, yet `generate()` (long-running), `stop()` (cross-thread cancellation), and `resetContext()` can be called from different threads. ReentrantLock is the wrong pattern here because `stop()` must be callable while `generate()` holds the lock. An `AtomicBoolean` guard is likely the right approach.

### Build & Configuration

- **`compileSdk` should be updated to 35** (Android 15)
- **Maven publish plugin version mismatch** — root `build.gradle` declares `0.34.0`, module declares `0.31.0`
- **Monolithic .so** — the default Android CMake preset enables nearly all features (XNNPACK, quantized kernels, LLM, ASR, training, devtools). Everything links into one `libexecutorch.so` with no flavor mechanism for a slim AAR
- **Build script hardcodes debug AAR** — `build_android_library.sh` always copies `executorch_android-debug.aar`
- **No validation that .so exists before Gradle** — running Gradle without the CMake phase produces an AAR with no native libraries and no error
- **.so rename obscures debugging** — CMake produces `libexecutorch_jni.so`, the build script renames to `libexecutorch.so`, mismatching native crash stacks
- **Android build requires host PyTorch install for C++ headers** — the Android preset triggers `find_package_torch_headers()` which runs `python -c "import torch; ..."`. A cross-compilation build should not depend on a host Python package for headers
- **Redundant CMake configure on every build** — fix is ready in branch `android/skip-redundant-cmake-configure` (saves ~37s per ABI on incremental builds)

### Code Duplication

- **`NativeLoader` init block** — identical static initializer copied in `Module.java`, `ExecuTorchRuntime.java`, and `TrainingModule.java`
- **`TensorHybrid`/`JEValue` class definitions** — duplicated between `jni_layer.cpp` and `jni_layer_training.cpp`. If the main definitions change, training JNI will silently get out of sync

### API Design

- **Load mode as raw `int`** — `LOAD_MODE_FILE = 0` etc. should be an enum; callers can currently pass any int
- **Duplicate model type constants** — `LlmModule` and `LlmModuleConfig` both define `MODEL_TYPE_TEXT`, `MODEL_TYPE_TEXT_VISION`, `MODEL_TYPE_MULTIMODAL`, and `TEXT_VISION == MULTIMODAL == 2` is confusing
- **Constructor explosion in `LlmModule`** — 9 constructors with subtle parameter differences. `LlmModuleConfig` builder exists but most constructors bypass it
- **`generate()` overload explosion** — 8+ overloads. `LlmGenerationConfig` builder exists but isn't the primary entry point
- **`LlmModuleConfig.getDataPath()` returns `String`** but `LlmModule` internally needs `List<String>`, forcing conversion at every call site
- **`EValue.TYPE_NAMES` is a non-static instance field** — every `EValue` instance allocates its own copy of the same string array
- **`Module.execute()` silently returns empty array on destroyed module** — should throw `IllegalStateException`
- **`MethodMetadata` missing input/output tensor shape info** — C++ `MethodMeta` exposes `input_tensor_meta(index)` → `TensorInfo` → `sizes()`, `scalar_type()`, but the Java `MethodMetadata` only exposes `name` and `backends`. Tensor shape and dtype should be surfaced through JNI so Java callers can validate inputs without hardcoding shapes

### Test Coverage

Only `EValue` and `Tensor` have unit tests. Missing coverage for: `Module`, `ExecuTorchRuntime`, `DType`, `MethodMetadata`, `LlmModule`, `LlmModuleConfig`, `LlmGenerationConfig`, `TrainingModule`, `SGD`, `AsrModule`, `AsrTranscribeConfig`.

### Architecture

- **fbjni → raw JNI migration** — `Module`, `LlmModule`, `TrainingModule` use fbjni `HybridClass`/`HybridData`; `AsrModule` already uses standard JNI. HybridData ties C++ destruction to Java GC, which is wrong for ML models holding hundreds of MB of native memory. Standard JNI would also remove `libfbjni.so` from the AAR and reduce contributor friction. Incremental migration path: Runtime → Module → LlmModule → Training.
- **Mixed Java/Kotlin** — core classes are Java, ASR and tests are Kotlin. The BUCK file declares `language = "JAVA"` for all targets, so Kotlin sources aren't buildable via BUCK.
- **BUCK file references internal targets** — should be removed or clearly annotated for the OSS repo

### Minor Cleanup

- `Tensor.java:921,979`: typo "supoprt" → "support"
- `EValue.java:210`: error message says "Unknown Tensor dtype" but it's an EValue issue
