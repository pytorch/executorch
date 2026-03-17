---
name: android
description: Build, test, or develop the Android Java/Kotlin bindings and JNI layer. Use when working on extension/android/, running Android tests, building the AAR, or integrating ExecuTorch into Android apps.
---

# Android

## Architecture

```
Java/Kotlin API (Module, Tensor, EValue, LlmModule, AsrModule)
    ↓  facebook::jni HybridClass
JNI layer (jni_layer.cpp, jni_layer_llama.cpp, jni_layer_asr.cpp)
    ↓
C++ ExecuTorch runtime (extension_module, extension_tensor, backends)
```

The AAR bundles `libexecutorch.so` (C++ runtime + JNI) with the Java/Kotlin API jar.

## Key files

| File | Purpose |
|------|---------|
| `extension/android/executorch_android/src/main/java/org/pytorch/executorch/Module.java` | Core model loading and inference API |
| `extension/android/executorch_android/src/main/java/org/pytorch/executorch/Tensor.java` | Tensor creation (`fromBlob`), data access, dtype support |
| `extension/android/executorch_android/src/main/java/org/pytorch/executorch/EValue.java` | Tagged union for inputs/outputs (Tensor, String, Int, Double, Bool) |
| `extension/android/executorch_android/src/main/java/org/pytorch/executorch/ExecuTorchRuntime.java` | Singleton runtime — registered ops/backends, file validation |
| `extension/android/executorch_android/src/main/java/org/pytorch/executorch/DType.java` | Data type enum |
| `extension/android/executorch_android/src/main/java/org/pytorch/executorch/MethodMetadata.java` | Method metadata (name, backends) |
| `extension/android/executorch_android/src/main/java/org/pytorch/executorch/extension/llm/LlmModule.java` | LLM text generation with streaming callbacks |
| `extension/android/executorch_android/src/main/java/org/pytorch/executorch/extension/llm/LlmCallback.java` | Callback interface: `onToken`, `onFinish`, `onError` |
| `extension/android/executorch_android/src/main/java/org/pytorch/executorch/extension/llm/LlmModuleConfig.java` | LLM configuration (model path, tokenizer, temperature) |
| `extension/android/executorch_android/src/main/java/org/pytorch/executorch/extension/llm/LlmGenerationConfig.java` | Generation parameters (seq_len, echo, temperature) |
| `extension/android/executorch_android/src/main/java/org/pytorch/executorch/extension/asr/AsrModule.kt` | Speech recognition wrapper (Kotlin) |
| `extension/android/executorch_android/src/main/java/org/pytorch/executorch/training/TrainingModule.java` | On-device training support |
| `extension/android/jni/jni_layer.cpp` | Main JNI bridge — `ExecuTorchJni` HybridClass, tensor/EValue marshalling |
| `extension/android/jni/jni_layer_llama.cpp` | LLM JNI bindings (token generation, prefill, multimodal) |
| `extension/android/jni/jni_layer_asr.cpp` | ASR JNI bindings |
| `extension/android/jni/jni_layer_runtime.cpp` | Runtime native methods (registered ops/backends) |
| `extension/android/jni/jni_layer_training.cpp` | Training JNI bindings (conditional compilation) |
| `extension/android/jni/jni_helper.h` | JNI utilities — error propagation, UTF-8 validation |
| `extension/android/CMakeLists.txt` | JNI shared library build — backend selection, fbjni download |
| `extension/android/executorch_android/build.gradle` | Gradle library config — dependencies, publishing, Spotless |
| `scripts/build_android_library.sh` | End-to-end build: CMake native → Gradle AAR packaging |

## Java/Kotlin API

### Module (core inference)
```java
// Load
Module module = Module.load("model.pte");                         // FILE mode
Module module = Module.load("model.pte", Module.LOAD_MODE_MMAP);  // MMAP mode
Module module = Module.load("model.pte", Module.LOAD_MODE_MMAP, 4); // 4 threads

// Execute
EValue[] results = module.forward(EValue.from(inputTensor));
EValue[] results = module.execute("custom_method", EValue.from(inputTensor));

// Inspect
String[] methods = module.getMethods();
MethodMetadata meta = module.getMethodMetadata("forward");
String[] backends = meta.getBackends();  // e.g. ["XnnpackBackend"]

// Cleanup
module.destroy();
```

Load modes: `LOAD_MODE_FILE` (0), `LOAD_MODE_MMAP` (1), `LOAD_MODE_MMAP_USE_MLOCK` (2), `LOAD_MODE_MMAP_USE_MLOCK_IGNORE_ERRORS` (3).

### Tensor
```java
// Create from arrays (copies data)
Tensor t = Tensor.fromBlob(new float[]{1.0f, 2.0f, 3.0f}, new long[]{1, 3});

// Create from direct buffers (zero-copy, must be native byte order)
FloatBuffer buf = Tensor.allocateFloatBuffer(3);
buf.put(new float[]{1.0f, 2.0f, 3.0f});
Tensor t = Tensor.fromBlob(buf, new long[]{1, 3});

// Read data
float[] data = t.getDataAsFloatArray();
long[] shape = t.shape();
DType dtype = t.dtype();
```

Supported dtypes: `UINT8`, `INT8`, `INT32`, `INT64`, `FLOAT16`, `FLOAT32`, `FLOAT64`.

### EValue
```java
EValue.from(tensor)     // Tensor
EValue.from("hello")    // String
EValue.from(42L)        // Int64
EValue.from(3.14)       // Double
EValue.from(true)       // Bool

// Extract
Tensor out = result.toTensor();
```

### LlmModule (text generation)
```java
LlmModule llm = new LlmModule("model.pte", "tokenizer.bin", 0.7f);
llm.generate("What is AI?", new LlmCallback() {
    @Override public void onToken(String token) { /* stream token */ }
    @Override public void onFinish(String results) { /* done */ }
    @Override public void onError(Throwable error) { /* handle */ }
});
llm.stop();         // interrupt generation
llm.resetContext(); // clear KV cache
```

Multimodal prefill (vision):
```java
llm.prefillImages(pixelData, width, height, channels);
llm.prefillPrompt("Describe this image");
llm.generate("", config, callback);
```

## JNI conventions

- **HybridClass methods** (called from fbjni): use `throwExecutorchException(errorCode, details)` for errors.
- **Plain JNIEXPORT functions**: use `setExecutorchPendingException(env, code, details)` instead — defined in `jni_helper.h`.
- `Module.java` uses `ReentrantLock` for thread safety; the C++ `ExecuTorchJni` is not thread-safe.
- Kotlin `jvmTarget = "11"` with no `-Xjvm-default` flag — Kotlin interface methods with bodies are not JVM default methods for Java callers.
- Symbol visibility controlled by `jni/version_script.txt` (only JNI symbols exported).

## Build

### Full AAR build
```bash
export ANDROID_NDK=/path/to/ndk    # required
export ANDROID_SDK=/path/to/sdk    # required for Gradle
export ANDROID_ABIS="arm64-v8a"    # default: arm64-v8a x86_64
export BUILD_AAR_DIR=aar-out       # optional: copy final AAR here
mkdir -p ${BUILD_AAR_DIR}
sh scripts/build_android_library.sh
```

Output: `extension/android/executorch_android/build/outputs/aar/executorch_android-debug.aar`

### What the build script does
1. **CMake native build** per ABI using NDK toolchain and `android-<ABI>` preset
2. Copies `libexecutorch.so` to `cmake-out-android-so/<ABI>/`
3. Strips `.so` in Release mode via `llvm-strip`
4. **Gradle build** (`./gradlew build`) — compiles Java/Kotlin, packages AAR
5. Runs unit tests as sanity check (`./gradlew :executorch_android:testDebugUnitTest`)

### Backend selection (CMake flags in build script)
| Flag | Default | Backend |
|------|---------|---------|
| `EXECUTORCH_BUILD_XNNPACK` | ON (via preset) | XNNPACK CPU |
| `EXECUTORCH_BUILD_QNN` | ON if `QNN_SDK_ROOT` set | Qualcomm QNN |
| `EXECUTORCH_BUILD_VULKAN` | OFF | Vulkan GPU |
| `EXECUTORCH_BUILD_NEURON` | ON if `NEURON_BUFFER_ALLOCATOR_LIB` set | MediaTek Neuron |
| `EXECUTORCH_BUILD_EXTENSION_LLM` | ON | LLM runner (Llama) |
| `EXECUTORCH_BUILD_LLAMA_JNI` | ON | LLM + ASR JNI bindings |
| `EXECUTORCH_BUILD_EXTENSION_TRAINING` | ON | Training support |
| `EXECUTORCH_ANDROID_PROFILING` | OFF | ETDump profiling |

### Custom JNI library
```bash
cmake ... -DEXECUTORCH_JNI_CUSTOM_LIBRARY=my_custom_ops
```
Linked with `--whole-archive` to ensure operator registration.

## Testing

### Unit tests (no device required)
```bash
cd extension/android
ANDROID_HOME="${ANDROID_SDK}" ./gradlew :executorch_android:testDebugUnitTest
```
Tests: `src/test/java/org/pytorch/executorch/TensorTest.kt`, `EValueTest.kt`

### Instrumentation tests (device/emulator required)
```bash
# Download test models first
sh executorch_android/android_test_setup.sh

# Run on connected device
ANDROID_HOME="${ANDROID_SDK}" ./gradlew :executorch_android:connectedAndroidTest
```
Tests: `src/androidTest/java/org/pytorch/executorch/ModuleE2ETest.kt`, `ModuleInstrumentationTest.kt`, `RuntimeInstrumentationTest.kt`, `LlmModuleInstrumentationTest.kt`

### Test pattern (E2E golden model)
```kotlin
val input = loadFloatArrayFromResource("model_input.bin")
val expected = loadFloatArrayFromResource("model_expected_output.bin")
val tensor = Tensor.fromBlob(input, longArrayOf(1, 3, 224, 224))
val module = Module.load(pteFile.absolutePath)
val output = module.forward(EValue.from(tensor))[0].toTensor().getDataAsFloatArray()
assertOutputsClose(output, expected, atol = 1e-3f)
module.destroy()
```

### Code formatting
```bash
cd extension/android
./gradlew spotlessCheck   # check Kotlin formatting (ktfmt)
./gradlew spotlessApply   # auto-fix
```

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| `com.facebook.fbjni:fbjni` | 0.7.0 | JNI bridge (HybridClass, type marshalling) |
| `com.facebook.soloader:nativeloader` | 0.10.5 | Native `.so` loading |
| `androidx.core:core-ktx` | (catalog) | Kotlin Android extensions |
| `com.qualcomm.qti:qnn-runtime` | (optional) | QNN backend runtime |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ANDROID_NDK` not set | Set `export ANDROID_NDK=/path/to/ndk` (NDK 25+ recommended) |
| Gradle build fails with SDK error | Set `export ANDROID_SDK=/path/to/sdk` or `ANDROID_HOME` |
| `UnsatisfiedLinkError` at runtime | Verify AAR includes correct ABI; check `ANDROID_ABIS` build variable |
| fbjni version mismatch | Ensure same fbjni version in `CMakeLists.txt` and `build.gradle` (currently 0.7.0) |
| Missing operators at runtime | Link with `--whole-archive` or use `EXECUTORCH_JNI_CUSTOM_LIBRARY` |
| `IllegalArgumentException: buffer is not direct` | Use `Tensor.allocateFloatBuffer()` or `ByteBuffer.allocateDirect()` with native byte order |
| Thread safety issues | `Module` uses `ReentrantLock`; don't share `Module` across threads without synchronization |
| Kotlin interface default methods not visible from Java | `kotlinOptions` only sets `jvmTarget=11` — no `-Xjvm-default` flag |
