---
name: qualcomm
description: Build, test, or develop the QNN (Qualcomm AI Engine Direct) backend. Use when working on backends/qualcomm/, building QNN (use backends/qualcomm/scripts/build.sh), adding new ops or passes, running QNN delegate
  tests, or exporting models for Qualcomm HTP/GPU targets.
---

# QNN (Qualcomm AI Engine Direct) Backend

## Advanced Topics

When the user's request falls into one of these areas, read the corresponding file before proceeding:

| Topic | File | When to read |
|---|---|---|
| Export / lowering / quantization options / pass pipelines | `lowering_export.md` | User asks about exporting, lowering, quantization config, QuantDtype, QuantRecipe, pass pipelines |
| New op development | `new_op_development.md` | User asks to add/implement a new op or op builder |
| Model enablement | `model_enablement.md` | User asks to enable a new model end-to-end |
| Profiling & debugging | `profiling.md` | User asks about profiling, optrace, QHAS, QAIRT Visualizer *(file TBD)* |

## Building

Use `backends/qualcomm/scripts/build.sh`. Linux only (macOS not supported).

**Environment variables:**
- `QNN_SDK_ROOT` — path to QNN SDK (auto-downloaded if not set)
- `ANDROID_NDK_ROOT` — path to Android NDK (auto-downloaded if not set)

**Build targets:**

| Target | Default | Build dir |
|---|---|---|
| x86_64 (Python interface + host tools) | enabled | `build-x86/` |
| Android arm64-v8a (device runner) | enabled | `build-android/` |
| Hexagon DSP (direct mode) | disabled | `build-hexagon/` |
| OE Linux embedded | disabled | `build-oe-linux/` |

**Common build commands:**

```bash
# Full build (x86_64 + Android)
./backends/qualcomm/scripts/build.sh

# x86_64 only (faster, for Python interface development)
./backends/qualcomm/scripts/build.sh --skip_linux_android

# Android only (skip x86_64)
./backends/qualcomm/scripts/build.sh --skip_x86_64

# Incremental build (skip clean)
./backends/qualcomm/scripts/build.sh --no_clean

# Enable Hexagon DSP direct mode (requires HEXAGON_SDK_ROOT, HEXAGON_TOOLS_ROOT, DSP_VERSION)
./backends/qualcomm/scripts/build.sh --enable_hexagon

# OE Linux embedded target (requires TOOLCHAIN_ROOT_HOST, TOOLCHAIN_ROOT_TARGET)
./backends/qualcomm/scripts/build.sh --enable_linux_embedded

# Release build
./backends/qualcomm/scripts/build.sh --release

# Control parallelism
./backends/qualcomm/scripts/build.sh --job_number 8
```

**After x86_64 build**, the Python interface `.so` files are copied to `backends/qualcomm/python/` automatically.

## Testing

```bash
QNN_SDK_ROOT=/path/to/qnn_sdk \
ANDROID_NDK_ROOT=/path/to/android_ndk \
LD_LIBRARY_PATH=/path/to/executorch/build-x86/lib:/path/to/qnn_sdk/lib/x86_64-linux-clang \
PYTHONPATH=$(dirname $EXECUTORCH_ROOT) \
python backends/qualcomm/tests/test_qnn_delegate.py \
    TestQNNFloatingPointOperator.test_qnn_backend_abs \
    -H $HOST -s $DEVICE_SERIAL -m SM8850 -b build-android -a /path/to/artifacts
```

> **Note (build from source):** Set `PYTHONPATH` to the parent directory of the executorch repo root. Required because `executorch.examples.qualcomm` lives in the source tree and is not installed into site-packages.

Required flags: `-m` (SoC model), `-b` (Android build dir). Optional: `-s` (device serial), `-H` (host), `-a` (artifact dir), `-c` (compile only), `-x` (run on x86_64).

**Test classes:**

| Class | Description |
|---|---|
| `TestQNNFloatingPointOperator` | FP16 operator tests |
| `TestQNNQuantizedOperator` | Quantized operator tests |
| `TestQNNFloatingPointModel` | FP16 model-level tests |
| `TestQNNQuantizedModel` | Quantized model-level tests |
| `TestQNNFloatingPointUtils` | FP16 utility tests |
| `TestQNNQuantizedUtils` | Quantized utility tests |
| `TestExampleLLMScript` | LLM script tests |
| `TestExampleMultimodalityScript` | Multimodality script tests |
| `TestExampleOssScript` | OSS model script tests |
| `TestExampleScript` | General example script tests |
| `TestUtilsScript` | Utility script tests |
