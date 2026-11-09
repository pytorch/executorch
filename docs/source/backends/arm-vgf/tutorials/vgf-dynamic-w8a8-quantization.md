# Dynamic W8A8 Quantization with the Arm VGF Backend

Tutorial we recommend you complete before this:
* [VGF Getting Started Tutorial](vgf-getting-started.md)

## What you will learn in this tutorial:

In this tutorial you will learn how to use dynamic W8A8 quantization for
Linear layers with the ExecuTorch Arm VGF backend, lower the quantized model
to VGF, and run the resulting `.pte`.

Dynamic W8A8 quantization keeps model weights statically quantized to INT8
while computing activation quantization parameters from the input tensor at
runtime.

This can be useful when activation ranges vary significantly between inputs.
Unlike static activation quantization, the activation scale is not fixed from
calibration data. Instead, it is recomputed when the model executes.

For a dynamically quantized Linear operation, the VGF lowering conceptually
performs:

```text
FP32 activation
      |
      v
choose runtime scale
      |
      v
quantize to INT8
      |
      v
 INT8 MATMUL <---- static INT8 weight
      |
    INT32 <--- accumulator
      |
      v
* activation_scale
* weight_scale
      |
      v
    FP32
      |
 + FP32 bias
      |
      v
 FP32 output
```

```{note}
Dynamic W8A8 support currently targets the dynamic quantization pattern used
by `Linear`/`AddMM`.

Activations use per-tensor symmetric INT8 quantization with the range
`[-127, 127]` and zero point `0`. Weights are statically quantized and can use
either per-tensor or per-channel symmetric INT8 quantization.
```

## Static W8A8 and Dynamic W8A8

The main difference is when activation quantization parameters are determined.

| | Static W8A8 | Dynamic W8A8 |
|---|---|---|
| Weight quantization | Static | Static |
| Activation quantization | Static | Dynamic |
| Activation scale | Determined during calibration | Computed at runtime |
| Activation zero point | Fixed | `0` for the supported symmetric configuration |
| Linear accumulation | INT32 | INT32 |
| Linear bias | Quantized according to the static configuration | Applied in FP32 after accumulator rescaling |
| Useful when | Activation ranges are predictable | Activation ranges vary between inputs |

## Prerequisites

### Hardware

To follow the VGF flow you need a Linux machine with an `aarch64` or `x86_64`
processor architecture, or a macOS machine with Apple Silicon.

This tutorial can use the ML SDK for Vulkan emulation layer, so physical
VGF-capable target hardware is not required.

### Software

Install ExecuTorch and the VGF dependencies as described in the
[VGF Getting Started Tutorial](vgf-getting-started.md).

From the ExecuTorch repository, run:

```bash
./examples/arm/setup.sh \
    --i-agree-to-the-contained-eula \
    --disable-ethos-u-deps \
    --enable-mlsdk-deps
```

Source the generated environment:

```bash
source examples/arm/arm-scratch/setup_path.sh
```

Verify the Ahead-of-Time environment:

```bash
python -m executorch.backends.arm.vgf.check_env --aot
```

## Quantize a Linear Layer with Dynamic W8A8

We first use a small Linear model to demonstrate the quantization flow.

```python
import torch


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)

    def forward(self, x):
        return self.linear(x)


model = LinearModel().eval()
example_inputs = (torch.randn(2, 16),)
```

Export the model before applying PT2E quantization:

```python
exported_program = torch.export.export(model, example_inputs)
graph_module = exported_program.module(check_guards=False)
```

### Configure the VGF quantizer

Create a `VgfQuantizer` and configure dynamic W8A8 for Linear layers:

```python
from executorch.backends.arm.quantizer import (
    VgfQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.vgf import VgfCompileSpec


compile_spec = VgfCompileSpec()

dynamic_w8a8_config = get_symmetric_quantization_config(
    is_per_channel=True,
    is_dynamic=True,
    act_qmin=-127,
    act_qmax=127,
)

quantizer = VgfQuantizer(compile_spec)

# Leave operators in floating point by default.
quantizer.set_global(None)

# Apply dynamic W8A8 to Linear operations.
quantizer.set_module_type(torch.nn.Linear, dynamic_w8a8_config)
```

`is_dynamic=True` enables runtime activation quantization.

The `[-127, 127]` activation range selects the symmetric dynamic INT8
representation supported by the Arm backend.

`is_per_channel=True` selects per-channel weight quantization. Per-tensor
weight quantization can instead be selected by setting `is_per_channel=False`.

Using `set_global(None)` and explicitly configuring `torch.nn.Linear` limits
dynamic quantization to the operation currently supported by the dynamic W8A8
lowering.

### Prepare and convert the model

Use the standard PT2E quantization flow:

```python
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
)


prepared_model = prepare_pt2e(graph_module, quantizer)

# Run representative inputs so static weight observers are populated.
prepared_model(*example_inputs)

quantized_model = convert_pt2e(prepared_model)
```

The weights are statically quantized, while activation quantization parameters
are computed dynamically. The activation range observed from `example_inputs`
does not become a fixed activation range for inference.

You can inspect the converted graph with:

```python
print(quantized_model.graph)
```

For a dynamically quantized Linear operation, the graph contains dynamic
qparam calculation followed by quantize/dequantize operations. Conceptually,
this includes:

```text
choose_qparams_symmetric
quantize_per_tensor
dequantize_per_tensor
linear
```

These operations are recognized and lowered by the Arm backend during
delegation.

## Lower the Model to VGF

Export the quantized model again:

```python
quantized_exported_program = torch.export.export(
    quantized_model,
    example_inputs,
)
```

Create the VGF partitioner and lower the model:

```python
from executorch.backends.arm.vgf import VgfPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.extension.export_util.utils import save_pte_program


partitioner = VgfPartitioner(compile_spec)

edge_program_manager = to_edge_transform_and_lower(
    quantized_exported_program,
    partitioner=[partitioner],
    compile_config=EdgeCompileConfig(
        _check_ir_validity=False,
    ),
)

executorch_program_manager = edge_program_manager.to_executorch(
    config=ExecutorchBackendConfig(
        extract_delegate_segments=False,
    )
)

save_pte_program(
    executorch_program_manager,
    "dynamic_w8a8_linear.pte",
)
```

The generated `dynamic_w8a8_linear.pte` contains the VGF delegated program.

During Arm lowering, the dynamic PT2E representation is converted
approximately as follows:

```text
choose_qparams_symmetric(x)
              |
              v
 scale = max(amax(abs(x)) / 127, eps)
              |
              v
       INT8 activation
              |
              v
        INT8 MATMUL
              |
              v
       INT32 accumulator
              |
              v
      FP32 rescaling
              |
              v
       FP32 bias add
```

The activation scale remains a runtime graph value rather than becoming a
compile-time constant.

## Using Dynamic W8A8 with DeiT-Tiny

Dynamic activation quantization can be useful for transformer models where
activation distributions can change between inputs and between transformer
blocks.

The same flow can be applied to DeiT-Tiny.

Install `timm` if it is not already available:

```bash
pip install timm
```

Replace the small example model with:

```python
import timm
import torch


model = timm.models.deit.deit_tiny_patch16_224(
    pretrained=True
).eval()

example_inputs = (
    torch.randn(1, 3, 224, 224),
)
```

Keep the same Linear-specific quantizer configuration:

```python
dynamic_w8a8_config = get_symmetric_quantization_config(
    is_per_channel=True,
    is_dynamic=True,
    act_qmin=-127,
    act_qmax=127,
)

quantizer = VgfQuantizer(compile_spec)

quantizer.set_global(None)
quantizer.set_module_type(
    torch.nn.Linear,
    dynamic_w8a8_config,
)
```

Then run the same sequence:

```text
torch.export
    |
    v
prepare_pt2e
    |
    v
convert_pt2e
    |
    v
torch.export
    |
    v
VgfPartitioner
    |
    v
to_edge_transform_and_lower
    |
    v
to_executorch
    |
    v
.pte
```

The Arm dynamic W8A8 implementation has been validated with the rank-3
transformer Linear operations used by DeiT-Tiny, including query, key, value,
attention projection, and MLP Linear operations.

```{tip}
For model accuracy evaluation, use the normal DeiT/ImageNet preprocessing and
compare the quantized model against the FP32 reference over a representative
validation dataset.

A random tensor such as the one above is sufficient for demonstrating export
and compilation, but it is not suitable for measuring model accuracy.
```

## Why Dynamic Quantization Can Help

With static activation quantization, the activation scale is determined from
calibration data.

For example, suppose calibration observed values approximately in:

```text
[-1, 1]
```

but a runtime input produces values in:

```text
[-20, 20]
```

A static scale derived from the smaller range can cause substantial clipping
or quantization error.

Dynamic W8A8 instead computes a new scale from the current tensor:

```text
scale = max(amax(abs(x)) / 127, eps)
```

The quantization range therefore adapts to the current activation values.

This requires additional runtime work for calculating the activation scale, so
dynamic and static quantization represent different performance and accuracy
trade-offs.

## Build the VGF Runtime

If you have not already built the runtime while completing the VGF Getting
Started Tutorial, configure it with:

```bash
cmake \
  -DCMAKE_INSTALL_PREFIX=cmake-out \
  -DCMAKE_BUILD_TYPE=Debug \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_XNNPACK=OFF \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_VGF=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DPYTHON_EXECUTABLE=python \
  -Bcmake-out .
```

Build the executor runner:

```bash
cmake --build cmake-out --target executor_runner
```

## Run the Model

Using the ML SDK for Vulkan emulation environment, run:

```bash
./cmake-out/executor_runner \
    -model_path dynamic_w8a8_linear.pte
```

For an application integrating ExecuTorch directly, load and execute the
`.pte` in the same way as other VGF delegated models. Dynamic activation
qparams are evaluated as part of execution and do not require
application-side quantization.

## Troubleshooting

### `choose_qparams_symmetric` is not present in the converted graph

Check that the quantization configuration uses all of:

```python
is_dynamic=True
act_qmin=-127
act_qmax=127
```

The supported dynamic W8A8 activation representation is symmetric INT8.

### Linear is not dynamically quantized

Make sure the dynamic configuration is assigned to the Linear operation:

```python
quantizer.set_module_type(
    torch.nn.Linear,
    dynamic_w8a8_config,
)
```

If `set_global(None)` is used, operators without an explicit override remain
in floating point.

### Only part of the model is delegated

Dynamic W8A8 support does not by itself make every operator in a model
VGF-compatible. Delegation of the rest of the graph still depends on normal
VGF operator support.

See [VGF operator support](../VGF_op_support.md) for the currently supported
operations.

### VGF environment validation fails

Run:

```bash
python -m executorch.backends.arm.vgf.check_env --aot
```

and verify that the TOSA serialization tools and ML SDK Model Converter are
available in the environment.

You can also check:

```bash
which model-converter
```

## Takeaways

In this tutorial you learned how to:

* configure the VGF quantizer for dynamic W8A8;
* dynamically quantize Linear activations while keeping weights statically
  INT8 quantized;
* use per-channel or per-tensor INT8 weight quantization;
* export the PT2E dynamic quantization representation;
* lower dynamic Linear operations through the Arm/TOSA pipeline to VGF;
* generate a VGF-backed ExecuTorch `.pte`;
* apply the same flow to transformer models such as DeiT-Tiny.

Dynamic W8A8 is useful when runtime activation ranges vary enough that a
single calibration-derived activation scale is not appropriate. The
activation scale is recomputed from the runtime tensor while the model weights
remain statically quantized.

For additional information about VGF quantization options, see
[VGF Quantization](../arm-vgf-quantization.md).

If you encounter any bugs or issues following this tutorial, please file an
issue on the
[ExecuTorch GitHub repository](https://github.com/pytorch/executorch/issues/new).
