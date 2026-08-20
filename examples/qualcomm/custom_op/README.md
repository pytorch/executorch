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
    soc_model=args.model,
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
  -s <device_serial> \
  -H <host> \
  -m SM8650 \
  --op_package_dir examples/qualcomm/custom_op/example_op_package_htp/ExampleOpPackage \
  --build_op_package
```

**x86 emulator:**
```bash
python3 examples/qualcomm/custom_op/custom_ops_1.py \
  --build_folder build-x86 \
  -m SM8650 \
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
  -s <device_serial> \
  -H <host> \
  -m SM8650 \
  --op_package_dir examples/qualcomm/custom_op/example_op_package_htp_multi_output/SplitCustomOpPackage \
  --build_op_package
```

**x86 emulator:**
```bash
python3 examples/qualcomm/custom_op/custom_ops_2.py \
  --build_folder build-x86 \
  -m SM8650 \
  --op_package_dir examples/qualcomm/custom_op/example_op_package_htp_multi_output/SplitCustomOpPackage \
  --build_op_package \
  --enable_x86_64
```
