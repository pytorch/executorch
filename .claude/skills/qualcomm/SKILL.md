---
name: qualcomm
description: Build, test, or develop the QNN (Qualcomm AI Engine Direct) backend. Use when working on backends/qualcomm/, building QNN (use backends/qualcomm/scripts/build.sh), adding new ops or passes, running QNN delegate tests, or exporting models for Qualcomm HTP/GPU targets. Also exposes a Buck-vs-CMake parity workflow — invoke as `/qualcomm buck-fix`, `/qualcomm buck-cmake fix`, `/qualcomm buck-parity`, or any user request to fix `test-qnn-buck-build-linux` CI failures or check buck/cmake drift in backends/qualcomm/. Also covers QNN intermediate-output / per-layer accuracy debugging — trigger on phrases like "QNN accuracy issue", "QNN output doesn't match CPU", "debug per-layer for QNN", "find which QNN layer is wrong".
---

# QNN (Qualcomm AI Engine Direct) Backend

## Slash command argument routing

When this skill is invoked with arguments (e.g. `/qualcomm <args>`), classify the args FIRST and route before doing anything else:

| If args contain any of… | Route to |
|---|---|
| `buck-fix`, `buck-cmake`, `buck cmake`, `buck-parity`, `buck parity`, `buck ci`, `qnn buck`, `fix qnn ci`, `test-qnn-buck-build-linux`, or any natural-language request to fix QNN buck CI / catch buck-cmake drift | Read `buck_parity.md` and follow it end-to-end. Default mode: full iterative-fix loop. If the args also contain `check` or `diagnose`, run buck once and report only — do not apply fixes. |
| (no args) or any other args | Stay in this file; treat as a normal `/qualcomm` discovery request and use the Advanced Topics table below. |

## Advanced Topics

When the user's request falls into one of these areas, read the corresponding file before proceeding:

| Topic | File | When to read |
|---|---|---|
| Export / lowering / quantization options / pass pipelines | `lowering_export.md` | User asks about exporting, lowering, quantization config, QuantDtype, QuantRecipe, pass pipelines |
| New op development | `new_op_development.md` | User asks to add/implement a new op or op builder |
| Custom op enablement via QNN op packages | `custom_op_enablement.md` | User asks to add a custom PyTorch op with their own kernel, mentions op packages / `qnn-op-package-generator` / `QnnCustomOpPackageBuilder`, or needs an op QNN has no equivalent for and that cannot be composed from existing QNN ops. Covers HTP and LPAI/eNPU. |
| Model enablement | `model_enablement.md` | User asks to enable a new model end-to-end |
| Buck vs CMake parity (pre-PR or fix red CI) | `buck_parity.md` | User changed BUCK / TARGETS / `targets.bzl` or `CMakeLists.txt` under `backends/qualcomm/`, added new `.cpp` / `.h` / `#include` there, is preparing to push a PR that touches QNN, **or** the `test-qnn-buck-build-linux` CI check on their PR is red and they want to fix it locally. Direct trigger: `/qualcomm buck-fix`. |
| Profiling & debugging | `profiling.md` | User asks about profiling, optrace, QHAS, QAIRT Visualizer *(file TBD)* |
| QNN intermediate-output / per-layer accuracy debugging | `qnn_intermediate_debugger.md` | User reports QNN-vs-CPU accuracy divergence, asks to debug per-layer / intermediate output for QNN, mentions `QNNIntermediateDebugger` / `QcomNumericalComparator`, or wants to find which layer causes a QNN accuracy drop. Workflow generates a new debug script from the user's existing example script. |

## Building

Use `backends/qualcomm/scripts/build.sh`. Linux only (macOS not supported).

**Environment variables:**
- `QNN_SDK_ROOT` — path to QNN SDK (auto-downloaded if not set)
- `ANDROID_NDK_ROOT` — path to Android NDK (auto-downloaded if not set)

**Build targets:**

| Target | Default | Build dir | Flag |
|---|---|---|---|
| x86_64 (Python interface + host tools) | enabled | `build-x86/` | (on by default; `--skip_x86_64` to disable) |
| Android arm64-v8a (device runner) | enabled | `build-android/` | (on by default; `--skip_linux_android` to disable) |
| Direct mode (LPAI ADSP or Hexagon CDSP) | disabled | `build-direct/` | `--build_direct_mode <0\|3> --soc_model <model>` |
| OE Linux embedded | disabled | `build-oe-linux/` | `--enable_linux_embedded` |

Direct mode takes the DSP type as its argument: **`0` = ADSP/LPAI**, **`3` = CDSP/HTP**.
`--soc_model` is required with it. When the DSP type is `0`, the build also signs
the runtime libraries by calling `sign_library.sh --direct_mode`.

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

# Direct mode on the ADSP/LPAI (requires HEXAGON_SDK_ROOT, HEXAGON_TOOLS_ROOT)
./backends/qualcomm/scripts/build.sh --build_direct_mode 0 --soc_model SM8850

# Direct mode on the CDSP/HTP
./backends/qualcomm/scripts/build.sh --build_direct_mode 3 --soc_model SM8750

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
    --host $HOST --device $DEVICE_SERIAL --soc_model SM8850 --build_folder build-android -a /path/to/artifacts
```

> **Note (build from source):** Set `PYTHONPATH` to the parent directory of the executorch repo root. Required because `executorch.examples.qualcomm` lives in the source tree and is not installed into site-packages.

Required: `--soc_model`, `--build_folder` (Android build dir). Optional: `--device`
(serial), `--host`, `--artifact_dir` / `-a`, `--compile_only`, `--enable_x86_64`,
`--backend <htp|gpu|lpai>`, `--direct_build_folder <dir>` (direct mode; required
for LPAI op package tests).

> Most flags are **long-form only** — they come from
> `setup_common_args_and_variables()` in `backends/qualcomm/export_utils.py`,
> which defines no short aliases. Only the test file's own arguments have them
> (`-r`/`--executorch_root`, `-a`/`--artifact_dir`, `-i`/`--image_dataset`,
> `-p`/`--pretrained_weight`, `-n`/`--model_name`, `-e`/`--error_only`,
> `-d`/`--op_package_dir`).

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
