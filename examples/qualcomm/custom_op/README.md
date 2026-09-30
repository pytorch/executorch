# Custom Operator Support

The Qualcomm AI Engine Direct Backend in ExecuTorch supports custom PyTorch operators via the Qualcomm AI Engine Direct Op Package mechanism. Custom PyTorch operators, utilizing the `torch.library` API, can be successfully delegated and supported through user-written op packages. Additionally, built-in PyTorch nodes can be overridden by these op packages.

Note: The Qualcomm AI Engine Direct SDK is required to compile an op package.

This folder contains examples demonstrating the end-to-end flow for adding a custom op: defining the PyTorch op, writing the QNN op package, registering it with the ExecuTorch backend, and quantizing it.

## Prerequisite

- Please finish tutorial [Setting up executorch](https://pytorch.org/executorch/stable/getting-started-setup).

- Please finish [setup QNN backend](../../../docs/source/backends-qualcomm.md). This example is verified with QNN SDK 2.37.0.

- Please follow [the instructions to install proper version of Hexagon SDK and Hexagon Tools.](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/linux_setup.html#htp-and-dsp)

  The required Hexagon SDK and tools versions depend on your QNN SDK version. Check the `Makefile` in the op package directory for the exact combination — `HEXAGON_SDK_ROOT_V<arch>` and `HEXAGON_TOOLS_VERSION_V<arch>` specify the SDK and tools version per target.

  For the examples in this folder (verified with QNN SDK 2.37.0, for SM8650):

  | Target | Hexagon SDK | Tools version |
  |--------|-------------|---------------|
  | `htp_v75` (SM8650 on-device) | hexagon-sdk-5.4.0 | 8.7.03 (bundled) |
  | `htp_x86` (x86 emulator) | hexagon-sdk-6.0.0 | 8.8.02 (install separately) |

  For each target you intend to build, install the corresponding Hexagon SDK:
  ```bash
  # example: hexagon-sdk-5.4.0 for v75 target (bundled with Hexagon tools 8.7.03)
  qpm-cli --install hexagonsdk5.x --version 5.4.0.3 --path /path/to/Qualcomm/Hexagon_SDK/hexagon-sdk-5.4.0
  # example: hexagon-sdk-6.0.0 for x86 target
  qpm-cli --install hexagonsdk6.x --version 6.0.0.2 --path /path/to/Qualcomm/Hexagon_SDK/hexagon-sdk-6.0.0
  ```

  **Note:** The tools version required by the Makefile (`HEXAGON_TOOLS_VERSION_V<arch>`) may differ from the version bundled inside the Hexagon SDK. If the required tools version is not present under `hexagon-sdk-<version>/tools/HEXAGON_Tools/`, install it   separately:
    > ```bash
    > # example: tools 8.8.02 for x86 target
    > qpm-cli --extract hexagon8.8 --version 8.8.02.1 \
    >   --path /path/to/Qualcomm/Hexagon_SDK/hexagon-sdk-6.0.0/tools/HEXAGON_Tools/8.8.02
    > ```

## Setup environment variables

`$QNN_SDK_ROOT` refers to the root of the Qualcomm AI Engine Direct SDK.

`$HEXAGON_SDK_ROOT` refers to the root of the specified version of Hexagon SDK, i.e., the directory containing `readme.txt`.

`$X86_CXX` refers to the clang++ compiler, verified with clang++14.

```bash
export HEXAGON_SDK_ROOT=/path/to/Qualcomm/Hexagon_SDK/hexagon-sdk-5.4.0
export X86_CXX=/path/to/clang-14.0.0/bin/clang++

# Source the QNN environment setup script to make op package tools available
source $QNN_SDK_ROOT/bin/envsetup.sh
```

---

## End-to-End Custom Op Flow

Adding a custom op involves four steps:

1. [Define the PyTorch custom op](#step-1-define-the-pytorch-custom-op)
2. [Write the QNN op package](#step-2-write-the-qnn-op-package)
3. [Register the op package with ExecuTorch](#step-3-register-the-op-package-with-executorch)
4. [Annotate the op for quantization (optional)](#step-4-annotate-the-op-for-quantization)

---

### Step 1: Define the PyTorch custom op

Use `torch.library` to register the custom op and its `out` variant. The `out` variant is required for ExecuTorch export.

**Single-output op:**
```python
from torch.library import impl, Library

my_op_lib = Library("my_ops", "DEF")
my_op_lib.define("mul3(Tensor input) -> Tensor")

@impl(my_op_lib, "mul3", dispatch_key="CompositeExplicitAutograd")
def mul3_impl(a: torch.Tensor) -> torch.Tensor:
    return a * 3

my_op_lib.define("mul3.out(Tensor input, *, Tensor(a!) output) -> Tensor(a!)")

@impl(my_op_lib, "mul3.out", dispatch_key="CompositeExplicitAutograd")
def mul3_out_impl(a: torch.Tensor, *, output: torch.Tensor) -> torch.Tensor:
    output.copy_(a * 3)
    return output
```

**Multi-output op** (returns a tuple of tensors):
```python
my_op_lib.define("split_custom(Tensor input) -> (Tensor, Tensor)")

@impl(my_op_lib, "split_custom", dispatch_key="CompositeExplicitAutograd")
def split_custom_impl(x: torch.Tensor):
    half = x.shape[-1] // 2
    return x[..., :half], x[..., half:]

my_op_lib.define(
    "split_custom.out("
    "Tensor input, "
    "*, Tensor(a!) first_half, Tensor(b!) second_half"
    ") -> (Tensor(a!), Tensor(b!))"
)

@impl(my_op_lib, "split_custom.out", dispatch_key="CompositeExplicitAutograd")
def split_custom_out_impl(x, *, first_half, second_half):
    half = x.shape[-1] // 2
    first_half.copy_(x[..., :half])
    second_half.copy_(x[..., half:])
    return first_half, second_half
```

---

### Step 2: Write the QNN op package

An op package consists of an XML config file and C++ implementation files.

#### 2a. Define the XML OpDef config

Create an XML file describing the package name, domain, version, and the operations it contains. The `PackageName` in the XML determines the library name (`libQnn<PackageName>.so`).

```xml
<OpDefCollection
    PackageName="ExampleOpPackage"
    Domain="aisw"
    Version="1.0.0">
  <OpDefList>
    <OpDef>
      <Name>ExampleCustomOp</Name>
      ...
    </OpDef>
  </OpDefList>
</OpDefCollection>
```

Refer to [the example XML config](example_op_package_htp/ExampleOpPackage/config/example_op_package_htp.xml) for a complete example. Consult the [Qualcomm AI Engine Direct op package documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/op_def_schema.html) for the full schema.

#### 2b. Generate skeleton code

Pass the XML to `qnn-op-package-generator` to generate the C++ skeleton:

```bash
qnn-op-package-generator --config_path path/to/config.xml
```

Detailed instructions to use `qnn-op-package-generator` can be found here: https://docs.qualcomm.com/doc/80-63442-10/topic/op_package_gen_example.html

#### 2c. Implement the op

Fill in the generated C++ source files. The interface file generally does not require changes. The op source file (e.g., `src/ops/ExampleCustomOp.cpp`) contains the kernel implementation. Refer to [the example implementation](example_op_package_htp/ExampleOpPackage/src/ops/ExampleCustomOp.cpp) for details.

#### Op package I/O format

The op package I/O must align with the PyTorch op schema:

- **Inputs** `in[0]…in[m-1]`: one tensor per input argument in the PyTorch op
- **Outputs** `out[0]…out[n-1]`: one tensor per output in the PyTorch op
- **Parameters**: optional scalar/tensor parameters matching the op schema

#### 2d. Build the op package

The generated `Makefile` supports building for all required targets:

```bash
cd path/to/ExampleOpPackage
make htp_x86 htp_aarch64 htp_v<arch>
```


---

### Step 3: Register the op package with ExecuTorch

Use `QnnCustomOpPackageBuilder` to parse the XML config and register target/platform/path combinations. It reads the package name and interface provider from the XML automatically.

```python
from executorch.backends.qualcomm.custom_op.interface import QnnCustomOpPackageBuilder
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchOpPackagePlatform,
    QnnExecuTorchOpPackageTarget,
)

# Parse the XML and map QNN op type names to PyTorch op targets
op_package_config = QnnCustomOpPackageBuilder(
    xml_path="path/to/ExampleOpPackage/config/example_op_package_htp.xml",
    torch_op_name_map={"ExampleCustomOp": torch.ops.my_ops.mul3.default},
)

# Register entry for (target, platform)
op_package_config.register_implementation(
    target=QnnExecuTorchOpPackageTarget.HTP,
    platform=QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
    op_package_path="/path/to/op_package",  # on-device path
)
op_package_config.register_implementation(
    target=QnnExecuTorchOpPackageTarget.CPU,
    platform=QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
    op_package_path="/path/to/op_package",  # on-device path
)
op_package_config.register_implementation(
    target=QnnExecuTorchOpPackageTarget.CPU,
    platform=QnnExecuTorchOpPackagePlatform.X86_64,
    op_package_path="/path/to/op_package",
)

op_package_options = op_package_config.get_op_package_options()
```

`torch_op_name_map` maps each QNN op type name (as it appears in the XML `<Name>` field) to the corresponding PyTorch op target. A `ValueError` is raised if any key is not found in the parsed package.

Pass `op_package_options` to `build_executorch_binary`:

```python
build_executorch_binary(
    model,
    sample_input,
    soc_model,
    output_path,
    sample_input,
    op_package_options=op_package_options,
    ...
)
```

---

### Step 4: Annotate the op for quantization (optional)

Use `CustomOpsQuantAnnotator` to declare quantization specs for custom op inputs and outputs.

```python
from executorch.backends.qualcomm.custom_op.annotator import (
    CustomOpsQuantAnnotator,
    IOQuantConfig,
)
from executorch.backends.qualcomm.quantizer.qconfig import get_ptq_per_channel_quant_config

quant_cfg = get_ptq_per_channel_quant_config()
annotator = CustomOpsQuantAnnotator()
annotator.register_annotation(
    torch.ops.my_ops.mul3.default,
    IOQuantConfig(
        input_quant_specs={0: quant_cfg.input_activation},
        output_quant_specs={0: quant_cfg.output_activation},
    ),
)
annotate_fn = annotator.build_annotation_fn()

quantizer = make_quantizer(
    quant_dtype=QuantDtype.use_8a8w,
    custom_annotations=(annotate_fn,),
    backend=get_backend_type(args.backend),
    soc_model=args.soc_model,
)
```

`IOQuantConfig` takes two optional dicts:
- `input_quant_specs`: maps input index → `QuantizationSpec`
- `output_quant_specs`: maps output index → `QuantizationSpec`

**Multi-output ops** return a tuple of tensors. Specify one entry per output index; indices not listed are left unquantized (e.g., integer index outputs):

```python
annotator.register_annotation(
    torch.ops.my_ops.split_custom.default,
    IOQuantConfig(
        input_quant_specs={0: quant_cfg.input_activation},
        output_quant_specs={
            0: quant_cfg.output_activation,  # first output tensor
            1: quant_cfg.output_activation,  # second output tensor
        },
    ),
)
```

Multiple ops can be registered on the same annotator before calling `build_annotation_fn()`.

---

## Running the Examples

### Example 1: Single-output custom op (`custom_ops_1.py`)

Registers `torch.ops.my_ops.mul3.default` (multiply by 3) and delegates it via `ExampleOpPackage`.

**On-device (Android):**
```bash
python3 examples/qualcomm/custom_op/custom_ops_1.py \
  --build_folder build-android \
  --device <device_serial> \
  --host <host> \
  --soc_model SM8650 \
  --op_package_dir examples/qualcomm/custom_op/example_op_package_htp/ExampleOpPackage \
  --build_op_package
```

**x86 emulator:**
```bash
python3 examples/qualcomm/custom_op/custom_ops_1.py \
  --build_folder build-x86 \
  --soc_model SM8650 \
  --op_package_dir examples/qualcomm/custom_op/example_op_package_htp/ExampleOpPackage \
  --build_op_package \
  --enable_x86_64
```

### Example 2: Multi-output custom op (`custom_ops_2.py`)

Registers `torch.ops.my_ops.split_custom.default` (splits a tensor into two halves) and delegates it via `SplitCustomOpPackage`.

**On-device (Android):**
```bash
python3 examples/qualcomm/custom_op/custom_ops_2.py \
  --build_folder build-android \
  --device <device_serial> \
  --host <host> \
  --soc_model SM8650 \
  --op_package_dir examples/qualcomm/custom_op/example_op_package_htp_multi_output/SplitCustomOpPackage \
  --build_op_package
```

**x86 emulator:**
```bash
python3 examples/qualcomm/custom_op/custom_ops_2.py \
  --build_folder build-x86 \
  --soc_model SM8650 \
  --op_package_dir examples/qualcomm/custom_op/example_op_package_htp_multi_output/SplitCustomOpPackage \
  --build_op_package \
  --enable_x86_64
```

### Example 3: LPAI (eNPU) custom op (`custom_ops_lpai.py`)

Registers the same `torch.ops.my_ops.mul3.default` operator, but delegates it to
the **LPAI** backend via `ExampleLpaiOpPackage`.

> **Requires Qualcomm AI Engine Direct SDK >= 2.49.** SDK 2.48 is the first
> release that ships `include/QNN/LPAI/QnnLpaiOpPackage.h`,
> `include/QNN/LPAI/QnnLpaiOpPackageInfrastructure.h` and
> `share/QNN/OpPackageGenerator/makefiles/LPAI/`; on older SDKs (e.g. 2.47) the
> LPAI Mako templates are present but the headers and makefiles they depend on
> are not, so `qnn-op-package-generator` fails and the generated sources cannot
> be compiled. 2.49 is required for the on-device flow, which needs the direct
> mode runtime.
>
> `QNN_SDK_ROOT` must be exported *before* sourcing `envsetup.sh`, otherwise the
> script silently selects whichever SDK it finds first, and the op package is
> then generated against the wrong headers.

> **On-device execution requires direct mode.** The LPAI team does not support
> registering an op package over FastRPC, so the DSP-side kernel is only reachable
> when the delegate itself is compiled for Hexagon and runs inside the DSP
> process. See [Running on device](#running-on-device-direct-mode) below.

#### How the LPAI op package differs from HTP

| | HTP | LPAI |
| --- | --- | --- |
| Interface version | `QnnOpPackage_Interface_t` v1 (`DEF_PACKAGE_OP` macros) | QNN OpPackage **v1.4** (`init` / `terminate` / `getInfo` / `validateOpConfig`) |
| Files per op | `{OpName}.cpp` | `{OpName}_inference.c` **and** `{OpName}_compiler.cpp` |
| Kernel entry point | `DEF_PACKAGE_OP(...)` registered impl | `executeOp` callback on `QnnLpaiOpPackage_OperationInfo_t` |
| Tensor access | `TensorType&` wrappers | `QnnLpaiOpPackage_GlobalInfrastructure_t` accessors (`getInputTensor`, `getTensorData`, `getTensorLayout`, ...) |
| Build outputs | one lib per hexagon arch + x86 + aarch64 | x86 host lib (compiler **and** inference side) + `hexagon-v79` inference-only skel and island image |

The same sources are compiled twice, selected by the `LPAI_INFERENCE_ONLY` define:

* **without** the define -> host / compiler side library
  (`libs/x86_64-linux-clang/libQnnExampleLpaiOpPackage.so`). This library
  contains the compiler-side callbacks (`validateOp`, `getTempBufferSize`,
  `getLayoutSupportFlag`) **and** the inference implementation, so it is all that
  is needed to compile a model and to run it through the LPAI x86_64 simulator.
* **with** the define -> inference-only skel for the DSP
  (`libs/hexagon-v79/libQnnExampleLpaiOpPackage.so`, plus the island image
  `libs/hexagon-v79/libLpaiOpPackageIsland.so`). Building this target
  additionally requires `HEXAGON_SDK_ROOT` and `HEXAGON_TOOLS_VERSION`.

> **`HEXAGON_TOOLS_VERSION` and `HEXAGON_TOOLS_ROOT` are different variables.**
> The op package Makefile selects the compiler with `HEXAGON_TOOLS_VERSION` (a
> bare version number such as `8.8.06`, resolved under
> `$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/`), while the direct mode runtime build
> uses `HEXAGON_TOOLS_ROOT` (a full path, and it must point at **19.0**, the only
> version that ships v81). Both are needed for the on-device flow and setting one
> does not satisfy the other.

Note that the `hexagon-v79` target produces **two** objects and both contain the
kernel: the op package itself and `libLpaiOpPackageIsland.so`, which is linked
with the uImage linker script the SDK ships in
`share/QNN/OpPackageGenerator/makefiles/LPAI/island` for always-resident
(island) memory. Both must be rebuilt, signed and deployed together; updating
only one of them leaves the DSP running the older kernel.

> **Island mode is not exercised by this example.** The island image is built,
> signed and deployed, but the kernel is not executed from it. LPAI enables
> island only outside of direct mode — `LpaiContextCustomConfig` sets
> `QNN_LPAI_CONTEXT_SET_CFG_ENABLE_ISLAND` under `#ifndef __hexagon__`, and it
> carries a pre-existing `TODO: support graph based execution in island mode`.
> Since an op package is only reachable in direct mode (where `__hexagon__` *is*
> defined), the two are currently mutually exclusive, and the flow below was
> validated in non-island mode.

#### Generating the op package skeleton

The committed sources under `example_op_package_lpai/` were produced with:

```bash
qnn-op-package-generator \
  -p examples/qualcomm/custom_op/example_op_package_lpai/ExampleLpaiOpPackage/config/example_op_package_lpai.xml \
  -o <output_dir>
```

and then the two `TODO` blocks were filled in:
* `ExampleCustomOp_inference.c` - the `mul3` kernel. Because LPAI is a fixed
  point accelerator, the kernel reads the per-tensor quantization parameters and
  requantizes. Element addressing uses `layoutOrder` / `layoutStride`, so the
  input and output are free to use different memory layouts. See
  [Quantization conventions](#quantization-conventions-on-the-enpu) for the
  arithmetic, which has several non-obvious pitfalls.
* `ExampleCustomOp_compiler.cpp` - `validateOp`, which checks that the input and
  output share a data type and are rank 4 with matching dimensions.

#### Quantization conventions on the eNPU

The values an LPAI kernel is handed do not follow the plain
`value = scale * (q - zero_point)` rule that the surrounding ExecuTorch
quantize / dequantize nodes use, and the differences are easy to get wrong.

The numbers below were measured with the committed example (8a8w, `mul3`,
`--calibration_value 1.0`) by printing the values `getPerTensorQuantParams()` and
`getTensorDataType()` return, plus the first stored byte, from inside the kernel:

```
in  offset=-128  dataType=3 (INT_8)  scale.type=1 (INT)  scale=538976320  shift=37
out offset=-128  dataType=3 (INT_8)  scale.type=1 (INT)  scale=808464448  shift=36
stored_in[0]=127
```

`538976320 / 2^37 = 1/255` and `808464448 / 2^36 = 3/255`, as expected for input
range `[0, 1]` and output range `[0, 3]`.

* **`getPerTensorQuantParams()` reports `offset = -128`** (for input *and*
  output), while the Q/DQ nodes in the ExecuTorch graph use a zero point of `0`.
  The offset biases the values *as they sit in memory*:

  ```
  code   = stored - offset          // remove the bias to get the graph's code
  stored = code   + offset          // re-apply it before writing
  value  = scale * code
  ```

  The measurement above confirms this: the graph's code for `1.0` is `255`, and
  the byte the kernel actually reads is `255 + (-128) = 127`. The kernel must
  therefore un-bias on the way in **and** re-bias on the way out; doing only one
  of the two shifts the whole tensor by 128 codes.

* **Un-biasing must wrap modulo the storage width.** With `offset = -128` the
  codes `0..127` are stored as the bytes `128..255`, so `stored - offset`
  produces `256..383` rather than `0..127`, and the value then saturates. Mask
  the result (`& 0xFF`, or `& 0xFFFF` for 16 bit) before scaling. The output
  direction gets this for free from the narrowing store, which is why the bug
  only affects the input path — and why it only shows up for *small* values:
  calibrating on `1.0` and running on `1.0` always yields the stored byte `127`,
  which takes the correct branch. Running on `0.25` yields the stored byte `192`
  and returns `3.0` instead of `0.75`. Both cases are covered by
  `--calibration_value` / `--inference_value`.

* **Saturate the code, not the stored byte.** For input `1.0` with
  `scale = 1/255` the code is `255` and the stored byte is `127`, so a clamp
  applied to the byte never triggers — yet a code of `256` (one past the maximum)
  is stored as the innocent looking byte `128` and read back by the consumer as
  code `256 - 256 = 0`. Overflow has to be caught while the value is still a
  code. Note that this particular op cannot overflow: its output range is exactly
  three times its input range, so `requantScale` is `1`. A kernel whose ranges
  are not related that way can, hence the clamp.

* **The reported data type's *sign* is unusable.** The op package declares
  `QNN_DATATYPE_UFIXED_POINT_8` and the tensors really are unsigned, but
  `getTensorDataType()` reports `LPAI_CUSTOM_OP_DATATYPE_INT_8` (enum value `3`,
  see the measurement above). Take only the element *width* from the reported
  type; treating the storage as signed reads `0x80` as `-128` instead of `128`
  and collapses the tensor to the zero point.

* **The scale arrives as `scale / 2^shift` with `shift > 31`** — `shift` is `37`
  and `36` above. Computing `1u << shift` is undefined behaviour for
  `shift >= 32` and evaluates to `0` on this DSP, so the division silently yields
  `+inf` and every requantized value becomes `NaN`. Use `ldexpf(scale, -shift)`.

> These conventions are not documented in the QNN/LPAI SDK documentation as far
> as we can tell, which is why they are spelled out (and measured) here.

> **Note:** the QNN LPAI headers `#include <cstdint>`, so the C++ compiler must
> provide the C++ standard library headers. Pass `CC=gcc CXX=g++` (as the example
> script does) if your default `clang` install does not ship libstdc++ headers.

**x86 emulator:**
```bash
python3 examples/qualcomm/custom_op/custom_ops_lpai.py \
  --build_folder build-x86 \
  --backend lpai \
  --soc_model SM8850 \
  --op_package_dir examples/qualcomm/custom_op/example_op_package_lpai/ExampleLpaiOpPackage \
  --build_op_package \
  --enable_x86_64
```

##### Exercising the requantization edge cases

The calibration input and the inference input are separate
(`--calibration_value` / `--inference_value`). This matters: when both are the
same tensor, the input always sits at the top of the calibrated range (code
`255`, stored byte `127`), and the quantization pitfalls listed above are never
reached. Two extra runs cover them, and neither needs a device:

```bash
# Low codes: calibrate on 1.0, run on 0.25 -> code 64, stored byte 192.
# A kernel that un-biases without wrapping returns 3.0 instead of 0.75.
python3 examples/qualcomm/custom_op/custom_ops_lpai.py \
  --build_folder build-x86 --backend lpai --soc_model SM8850 \
  --op_package_dir examples/qualcomm/custom_op/example_op_package_lpai/ExampleLpaiOpPackage \
  --enable_x86_64 --calibration_value 1.0 --inference_value 0.25

# Above the calibrated range: run on 2.0. The graph's quantize node clamps to
# the calibrated maximum, so the expected result is the saturated 3.0 rather
# than 6.0, which has to be stated explicitly.
python3 examples/qualcomm/custom_op/custom_ops_lpai.py \
  --build_folder build-x86 --backend lpai --soc_model SM8850 \
  --op_package_dir examples/qualcomm/custom_op/example_op_package_lpai/ExampleLpaiOpPackage \
  --enable_x86_64 --calibration_value 1.0 --inference_value 2.0 --expected_value 3.0
```

#### Running on device (direct mode)

On-device execution goes through **direct mode**: the delegate is compiled for
Hexagon and runs inside the DSP process, so it registers the op package locally
instead of forwarding the registration over FastRPC. Build the runtime with:

```bash
backends/qualcomm/scripts/build.sh --build_direct_mode 0 --soc_model SM8850
```

Everything that runs on the DSP must be **signed**, including the op package.
This is handled for you in two places:

* the build command above ends by calling
  `sign_library.sh --direct_mode`, which signs the direct mode **runtime**
  libraries;
* `custom_ops_lpai.py` calls `sign_library.sh --op_package_dir` whenever it
  builds the `hexagon-v79` target (i.e. with `--build_op_package` and without
  `--enable_x86_64`), which signs every `.so` the **op package** produces. The
  signed objects are then deployed from
  `$QNN_SDK_ROOT/lib/lpai-v<hw_ver>/signed`. Pass `--skip_sign_op_package` to
  deploy the raw build outputs instead, which only load on a device that does
  not enforce code signing.

To re-sign without rebuilding, invoke the script directly:

```bash
# runtime libraries and op package together
backends/qualcomm/scripts/sign_library.sh \
  --direct_mode --htp_arch v81 --lpai_arch v6 \
  --op_package_dir examples/qualcomm/custom_op/example_op_package_lpai/ExampleLpaiOpPackage
```

Note that `elfsigner.py` uses the `imp` module, which was removed in Python 3.12,
so it has to run under an older interpreter. The example script takes care of
this by putting its own interpreter first on `PATH`.

Generate the `.pte` and push the artifacts:

```bash
python3 examples/qualcomm/custom_op/custom_ops_lpai.py \
  --build_folder build-android \
  --backend lpai \
  --device <device_serial> \
  --host <host> \
  --soc_model SM8850 \
  --op_package_dir examples/qualcomm/custom_op/example_op_package_lpai/ExampleLpaiOpPackage \
  --build_op_package
```

On device, the DSP libraries must be present in **both** the workspace root and
its `adsp/` subdirectory, because the loader searches `<path>/adsp/` first and
then `<path>/`. The runner also needs `--domain_id 0` to select the ADSP; the
default of `3` selects the CDSP, where LPAI is not present:

```bash
W=/data/local/tmp/executorch/custom_qnn_lpai
ADSP_LIBRARY_PATH=$W LD_LIBRARY_PATH=$W ./qnn_executor_direct_runner \
  --model_path $W/custom_qnn_lpai.pte \
  --input_list_path $W/input_list.txt \
  --output_folder_path $W \
  --domain_id 0
```

Supported LPAI SoCs are those with an `LpaiInfo` entry in
`backends/qualcomm/serialization/qc_schema.py` (e.g. `SM8850`, `SAR2230P`).

#### Debugging on the DSP

Host-side logs say nothing about what happens inside the kernel. Enable FARF
logging and read the DSP output from `logcat`:

```bash
adb shell "echo 0x1f > $W/qnn_executor_direct_runner.farf"
adb logcat -d -v time | grep 'ADSP:\[DS\]'
```

Two things worth knowing when reading that output:

* **`printf` on the DSP garbles long argument lists.** A single call with a dozen
  arguments prints repeating ASCII filler that looks like real but nonsensical
  data. Use several small `printf`s with one to three arguments each.
* A transient `EAI_ERR: Failed to reset global enpu clock level` (usually with
  `FTQ driver invoke method (273) failed`) means the kernel never ran and the
  output buffer was left untouched. It clears on a retry a few seconds later; do
  not read the all-zero buffer as a measurement.

#### Registering the op package for LPAI

Registration is identical to HTP apart from the target enum:

```python
op_package_config.register_implementation(
    target=QnnExecuTorchOpPackageTarget.LPAI,
    platform=QnnExecuTorchOpPackagePlatform.X86_64,
    op_package_path=x86_op_package_path,
)
```

Note that the fourth argument of QNN's `registerOpPackage` is backend specific:
for CPU / HTP it is the processor target name (`"CPU"`, `"HTP"`), whereas for
LPAI it is an *optional target memory pool* string. The ExecuTorch runtime
therefore passes `nullptr` for LPAI so that the backend selects its default pool.
