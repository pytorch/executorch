---
name: custom_op_enablement
description: Enable a user-defined PyTorch operator on QNN via a QNN Op Package (custom kernel). Use when the op has no QNN equivalent and cannot be composed from existing QNN ops, or when overriding a built-in QNN op with your own kernel. Covers HTP and LPAI/eNPU.
---

# Custom Op Enablement — QNN Op Packages

An **op package** is a shared library holding your own kernel, registered with the
QNN backend at context-creation time. Use it when the delegate cannot express the
op with QNN primitives.

## Decision Tree

1. QNN has a native op → **native builder**, see `new_op_development.md`
2. No native op, composable from several QNN ops → **decompose pass**, see `new_op_development.md`
3. No native op, needs your own kernel → **op package** (this file)
4. Built-in QNN op exists but you want to replace its implementation → **op package**

Prefer 1 and 2. An op package means owning C/C++ kernel code, a per-target
build, and (on LPAI) code signing.

> Reference examples: `examples/qualcomm/custom_op/` — `custom_ops_1.py` (HTP,
> single output), `custom_ops_2.py` (HTP, multi output), `custom_ops_lpai.py`
> (LPAI/eNPU, direct mode). Tutorial prose lives in that folder's `README.md`;
> this file carries the decisions and traps.

## HTP vs LPAI

The programming models differ substantially. Pick the column before writing code.

| | HTP | LPAI (eNPU) |
|---|---|---|
| Interface | `QnnOpPackage_Interface_t` v1, `DEF_PACKAGE_OP` macros | OpPackage **v1.4**: `init`/`terminate`/`getInfo`/`validateOpConfig` |
| Files per op | `{Op}.cpp` | `{Op}_inference.c` **and** `{Op}_compiler.cpp` |
| Kernel entry | registered impl via macro | `executeOp` on `QnnLpaiOpPackage_OperationInfo_t` |
| Tensor access | `TensorType&` wrappers | `QnnLpaiOpPackage_GlobalInfrastructure_t` accessors |
| Make targets | `htp_x86 htp_aarch64 htp_v<arch>` | `lpai_x86`, `lpai_hexagon_v79` |
| Build output | `build/` | `libs/` |
| On-device transport | FastRPC (normal runner) | **direct mode only** |
| Code signing | not required | **required** |
| Arch lookup | `_soc_info_table[chipset].htp_info.htp_arch` | `get_soc_to_lpai_hw_ver_map()[soc_model]` |
| SDK floor | 2.37 verified | **2.49** (2.48 first ships the headers; 2.49 needed for direct mode) |

LPAI has no FastRPC path: the LPAI team does not support registering an op
package over FastRPC, so on-device execution requires direct mode, where the
delegate is compiled for Hexagon and calls `registerOpPackage()` inside the DSP
process.

## Workflow

1. **Define the PyTorch op** with `torch.library`, including the `.out` variant
   (ExecuTorch export requires it). See README Step 1.
2. **Write the XML OpDef.** `PackageName` determines the library name
   (`libQnn<PackageName>.so`).
3. **Generate the skeleton:**
   ```bash
   qnn-op-package-generator -p path/to/config.xml -o <output_dir>
   ```
4. **Fill in the `TODO` blocks.** On LPAI that is the kernel in
   `{Op}_inference.c` and `validateOp` in `{Op}_compiler.cpp`.
5. **Build**, **register**, **quantize** (below).

> `QNN_SDK_ROOT` must be exported *before* sourcing `envsetup.sh`, or the script
> picks whichever SDK it finds first and the package is generated against the
> wrong headers.

### LPAI: one source set, two builds

`LPAI_INFERENCE_ONLY` selects which half is compiled:

* **unset** → x86_64 host library, containing the compiler-side callbacks
  (`validateOp`, `getTempBufferSize`, `getLayoutSupportFlag`) **and** the
  inference implementation. Enough to compile a model and to run the x86 simulator.
* **set** → inference-only DSP skel, plus the island image.

The `lpai_hexagon_v79` target emits **two** objects that both carry the kernel:
`libQnn<Pkg>.so` and `libLpaiOpPackageIsland.so`. Rebuild, sign, and deploy them
together — updating one leaves the DSP running the older kernel.

> `HEXAGON_TOOLS_VERSION` (bare version, e.g. `8.8.06`, resolved under
> `$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/`) selects the op package compiler.
> `HEXAGON_TOOLS_ROOT` (full path, must be **19.0**) is for the direct-mode
> runtime. Different variables; setting one does not satisfy the other.

> The QNN LPAI headers `#include <cstdint>`, so the C++ compiler must ship the
> C++ standard library headers. Pass `CC=gcc CXX=g++` if your default `clang`
> lacks libstdc++ headers.

## Registration

```python
from executorch.backends.qualcomm.custom_op.interface import QnnCustomOpPackageBuilder
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchOpPackagePlatform,  # UNKNOWN, X86_64, AARCH64_ANDROID, HEXAGON
    QnnExecuTorchOpPackageTarget,    # UNKNOWN, CPU, HTP, LPAI
)

op_package_config = QnnCustomOpPackageBuilder(
    xml_path=f"{op_package_dir}/config/my_package.xml",
    torch_op_name_map={"ExampleCustomOp": torch.ops.my_ops.mul3.default},
)
op_package_config.register_implementation(target=..., platform=..., op_package_path=...)
op_package_options = op_package_config.get_op_package_options()
```

`torch_op_name_map` maps the XML `<Name>` to the PyTorch target; a key missing
from the parsed package raises `ValueError`. Pass `op_package_options` to
`build_executorch_binary`.

Which combinations to register, and what `op_package_path` means:

| Backend | target / platform | `op_package_path` |
|---|---|---|
| HTP | `HTP` / `AARCH64_ANDROID` | on-device path to `libQnn<Pkg>_HTP.so` |
| HTP | `CPU` / `AARCH64_ANDROID` | on-device path to `libQnn<Pkg>.so` |
| HTP | `CPU` / `X86_64` | host path to the x86 build |
| LPAI | `LPAI` / `X86_64` | host path — used for graph compilation (preprocess) |
| LPAI | `LPAI` / `HEXAGON` | **bare file name**, e.g. `libQnn<Pkg>.so` |

Two traps here:

* **HTP needs the hexagon object copied to a distinct name.** The HTP and CPU
  registrations must not share a path, so `custom_ops_1.py` copies
  `build/hexagon-v<arch>/libQnn<Pkg>.so` → `..._HTP.so` after building.
* **LPAI/HEXAGON takes a base name, not a path.** In direct mode there is no
  64-bit AP process; the object is resolved through `ADSP_LIBRARY_PATH`.

### Quantization

Op-package ops are quantized like any other, via `CustomOpsQuantAnnotator` +
`IOQuantConfig` (`input_quant_specs` / `output_quant_specs`, indexed per
input/output). Indices you omit stay unquantized — that is how integer index
outputs of a multi-output op are handled. See README Step 4.

## Partitioning of op-package ops needs no bypass

Op-package ops go through the ordinary `is_node_supported` path in
`partition/qnn_partitioner.py`: `CustomOp.define_node` builds the op wrapper and
`QnnBackend.validateOpConfig` accepts it. Look for `my_ops.mul3.default | True`
in the log (not `| Forced Passed`).

This works because the op package is registered *before* partitioning, not at
preprocess time. `to_edge_transform_and_lower_to_qnn` opens a
`QnnManagerContext` around the whole flow, and `InitBackend` →
`GetOrCreateBackendBundle` → `QnnBackend::Configure` →
`BackendRegisterOpPackage` runs the registration from the
`op_package_options` carried in the compiler spec. `QnnOperatorSupport` obtains
its manager from that same context via `get_current_qnn_manager`, so it
validates against a backend that already has the package.

An earlier revision force-passed these nodes on the (incorrect) assumption that
registration happened later; that has been removed. If a custom op ever does get
rejected here, find out why rather than re-adding a bypass — a silent
force-pass hides genuine validation failures, and the cached-per-backend-type
bundle in `QnnManagerRegistry` is a plausible culprit (a bundle created earlier
without `op_package_options` would be reused).

`LpaiPartitionFallbackSupport` needs no separate check — it calls the same
`is_node_supported`.

## LPAI direct mode

Build the runtime (this also signs the runtime libraries when DSP type is 0):

```bash
backends/qualcomm/scripts/build.sh --build_direct_mode 0 --soc_model SM8850
```

Five things that are easy to get wrong:

* **`platform=HEXAGON` is mandatory.** In direct mode the delegate is neither
  `__x86_64__` nor `__ANDROID__`; before `__hexagon__` was handled,
  `current_platform` stayed `UNKNOWN` and every registration was silently
  skipped — no error, just an op that never ran.
* **`--domain_id 0`** selects the ADSP. The default `3` is the CDSP, where LPAI
  is absent. `SimpleADB` passes this automatically when `direct_build_folder` is
  set; a hand-written invocation must not omit it.
* **`registerOpPackage`'s 4th argument is backend specific.** CPU/HTP take a
  processor target name (`"CPU"`, `"HTP"`); LPAI takes an *optional target memory
  pool*. The runtime passes `nullptr` for LPAI so the backend picks its default.
* **Island mode and op packages are currently mutually exclusive.**
  `LpaiContextCustomConfig` sets `QNN_LPAI_CONTEXT_SET_CFG_ENABLE_ISLAND` under
  `#ifndef __hexagon__`, with a pre-existing `TODO: support graph based execution
  in island mode`. Direct mode *is* `__hexagon__`, so island is compiled out
  there — and direct mode is the only way to reach an op package. The island
  image is built, signed and deployed, but the kernel does not execute from it.
  Everything below was validated in non-island mode.
* **The loader searches `<workspace>/adsp/` before `<workspace>/`.** If you
  populate both, keep them identical or you will run a stale kernel.

Everything the drivers need follows from `direct_build_folder` in `QnnConfig`:
`SimpleADB` selects `qnn_executor_direct_runner`, appends `--domain_id`, and
`get_lpai_target_env()` returns `kAdsp`. Forgetting to forward that one field
produces a non-direct `.pte` run by the FastRPC runner — which fails on the DSP,
not at compile time.

### Signing

Everything the aDSP loads must be signed.

| What | Who signs it |
|---|---|
| Direct-mode runtime libs | `build.sh --build_direct_mode 0` (calls `sign_library.sh --direct_mode`) |
| Op package DSP objects | `custom_ops_lpai.py` after `make lpai_hexagon_v79` |
| Either, without rebuilding | `sign_library.sh` by hand |

```bash
# op package only
backends/qualcomm/scripts/sign_library.sh --lpai_arch v6 \
  --op_package_dir path/to/MyOpPackage
```

Signed output lands in `$QNN_SDK_ROOT/lib/lpai-v<hw_ver>/signed`, and that is
what gets deployed. `--skip_sign_op_package` deploys the raw build outputs
instead — only loadable on a device that does not enforce signing.

> `elfsigner.py` imports `imp`, removed in Python 3.12, so it needs an older
> interpreter. The driver prepends its own interpreter's directory to `PATH` for
> the subprocess. Never suppress `elfsigner` output — a silent failure here looks
> like a stale-kernel bug later.

## eNPU quantization conventions

The values an LPAI kernel receives do **not** follow the
`value = scale * (q - zero_point)` rule the surrounding Q/DQ nodes use. For 8a8w
on SM8850:

* **`offset = -128` while the graph's zero point is `0`.** The offset biases the
  values *as stored*: `code = stored - offset` on the way in,
  `stored = code + offset` on the way out, `value = scale * code`. Doing only one
  of the two shifts the tensor by 128 codes.
* **Saturate the code, not the stored byte.** For `1.0` with `scale = 1/255` the
  code is 255 and the stored byte 127, so a clamp on the byte never fires — yet
  code 256 is stored as the innocuous byte 128 and read back as `256-256 = 0`.
  Clamping the wrong quantity yielded `1.494` (`= 3/255 * 127`) instead of `3.0`.
* **The reported dtype's sign is unusable.** The package declares
  `UFIXED_POINT_8` and the tensors are unsigned, but `getTensorDataType()`
  reports `INT_8`. Take only the element *width*; treating storage as signed
  reads `0x80` as `-128` and collapses the tensor to the zero point.
* **Scale can arrive as `scale / 2^shift` with `shift > 31`** (e.g. `scale =
  538976320, shift = 37` for `1/255`). `1u << shift` is UB for `shift >= 32` and
  evaluates to 0 here, so the division yields `+inf` and every value becomes
  `NaN`. Use `ldexpf(scale, -shift)`.

Element addressing uses `layoutOrder` / `layoutStride`, so input and output may
use different memory layouts. Do not assume they match.

## Debugging on the DSP

Host logs say nothing about the kernel. Enable FARF and read `logcat`:

```bash
adb shell "echo 0x1f > $W/qnn_executor_direct_runner.farf"
adb logcat -d -v time | grep 'ADSP:\[DS\]'
```

* **DSP `printf` garbles long argument lists** — a dozen args prints plausible
  but nonsensical filler. Use several small `printf`s of one to three args.
* **Transient `EAI_ERR: Failed to reset global enpu clock level`** (often with
  `FTQ driver invoke method (273) failed`) means the kernel never ran and the
  output buffer is untouched. Retry; do not read the zeros as a measurement.
* **Stale-library trap.** Verify md5 local → remote → every on-device path, and
  grep the built `.so` for a marker string, before trusting any on-device result.
* `QnnContextCustomProtocol expected magic number ... but get: ...` is
  **informational** — a format probe that falls through to the Dlc path. Not an
  error, and unrelated to op packages.

## Verification

```bash
# LPAI, x86 simulator
python examples/qualcomm/custom_op/custom_ops_lpai.py \
  --build_folder build-x86 --backend lpai --soc_model SM8850 \
  --op_package_dir examples/qualcomm/custom_op/example_op_package_lpai/ExampleLpaiOpPackage \
  --build_op_package --enable_x86_64

# LPAI, on device (direct mode); builds + signs + deploys + compares
python backends/qualcomm/tests/test_qnn_delegate.py \
  TestUtilsScript.test_custom_op_lpai \
  --executorch_root . --artifact_dir ./custom_op_lpai \
  --build_folder build-android --direct_build_folder build-direct \
  --backend lpai --soc_model SM8850 --device <serial> [--host <host>]
```

A healthy on-device run shows `my_ops.mul3.default | True` (validated against the
backend) for your op, `_dom=adsp` in the runner's domain URI, and a non-trivial
`qnn_executorch_execute_all` duration.

HTP equivalents are `TestUtilsScript.test_custom_op_1` / `test_custom_op_2` with
`--op_package_dir`; see README "Running the Examples".

> Confidence: the LPAI path above is verified end-to-end on SM8850. The HTP
> material is derived from the committed examples and has not been re-run as part
> of that work.
