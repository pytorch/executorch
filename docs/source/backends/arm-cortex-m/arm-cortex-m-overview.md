# Arm Cortex-M Backend

:::{note}
This backend is in **beta**. It has been validated with a set of small models (e.g. MLPerf Tiny, MobileNetV2) and provides broad operator coverage through CMSIS-NN accelerated kernels with portable-ops fallback.
:::

The Arm&reg; Cortex&reg;-M backend accelerates quantized model execution on Arm Cortex-M CPUs using [CMSIS-NN](https://arm-software.github.io/CMSIS-NN/latest/) optimized kernels. Unlike delegate-based backends, it operates as an operator library: quantized subgraphs are replaced with CMSIS-NN accelerated kernels during the pass-lowering stage, while unsupported operators fall back to portable fp32 kernels.

## Target Support

The backend targets Arm Cortex-M CPUs via CMSIS-NN, which provides optimized kernel implementations for three instruction set variants:

| Variant      | Description                 | Example CPUs       | Supported |
|--------------|-----------------------------|--------------------|-----------|
| MVE (Helium) | M-profile Vector extensions | Cortex-M55, M85    | ✅        |
| DSP          | DSP extension instructions  | Cortex-M4, M7, M33 | ⬜        |
| Pure C       | Reference C implementation  | Any Cortex-M       | ⬜        |

DSP and pure C variants use the same CMSIS-NN API and may work, but have not been tested.

## CMSIS-NN Supported Operators

The backend pass pipeline replaces quantized ATen operators with [CMSIS-NN](https://arm-software.github.io/CMSIS-NN/latest/) kernel calls. See the [CMSIS-NN API documentation](https://arm-software.github.io/CMSIS-NN/latest/modules.html) for the full list of available kernels.

| ATen Op                        | CMSIS-NN Kernel        | 8w8a | 8w16a | 4w8a |
|--------------------------------|------------------------|------|-------|------|
| `aten.convolution`             | `arm_convolve`         | ✅   | ⬜    | ⬜   |
| `aten.convolution` (depthwise) | `arm_depthwise_conv`   | ✅   | ⬜    | ⬜   |
| `aten.convolution` (transposed)| `arm_transpose_conv`   | ✅   | ⬜    | ⬜   |
| `aten.linear`                  | `arm_fully_connected`  | ✅   | ⬜    | ⬜   |
| `aten.bmm`                     | `arm_batch_matmul`     | ✅   | ⬜    | ⬜   |
| `aten.add`                     | `arm_elementwise_add`  | ✅   | ⬜    | N/A  |
| `aten.mul`                     | `arm_elementwise_mul`  | ✅   | ⬜    | N/A  |
| `aten.max_pool2d`              | `arm_max_pool`         | ✅   | ⬜    | N/A  |
| `aten.avg_pool2d`              | `arm_avgpool`          | ✅   | ⬜    | N/A  |
| `aten._softmax`                | `arm_softmax`          | ✅   | ⬜    | N/A  |
| `aten.minimum`                 | `arm_minimum`          | ✅   | ⬜    | N/A  |
| `aten.maximum`                 | `arm_maximum`          | ✅   | ⬜    | N/A  |
| `aten.permute_copy`            | `arm_transpose`        | ✅   | ⬜    | N/A  |
| `aten.constant_pad_nd`         | `arm_pad`              | ✅   | ⬜    | N/A  |
| —                              | LSTM                   | ⬜   | ⬜    | ⬜   |
| —                              | SVDF                   | ⬜   | ⬜    | ⬜   |

## Quantization Support

The Cortex-M backend currently implements **symmetric INT8 (8w8a)** quantization:
- **Per-channel** quantization for convolution operators.
- **Per-tensor** quantization for all other supported operators.
- **Shared quantization parameters** for data-movement operators (e.g. reshape, permute) to avoid unnecessary requantization.

CMSIS-NN also supports INT4 weights with INT8 activations (4w8a), INT8 weights with INT16 activations (8w16a), and per-channel quantization for fully connected layers, but the corresponding quantizer configurations and operator implementations are not yet integrated.

## Tutorial

### Prerequisites

Install the ExecuTorch pip package:
```bash
./install_executorch.sh
```

For cross-compilation and running on simulated hardware:
- [Arm GNU Toolchain](https://developer.arm.com/Tools%20and%20Software/GNU%20Toolchain) for cross compilation.
- [Arm&reg; Corstone&trade; SSE-300 FVP](https://developer.arm.com/documentation/100966/1128/Arm--Corstone-SSE-300-FVP) or [SSE-320 FVP](https://developer.arm.com/documentation/109760/0000/SSE-320-FVP) for simulation.

:::{tip}
All cross-compilation tools can be downloaded and added to the path:
```bash
examples/arm/setup.sh --i-agree-to-the-contained-eula
source examples/arm/arm-scratch/setup_path.sh
```
:::

### 1. Export and quantize

Export the model, then quantize using `CortexMQuantizer` with the PT2E quantization flow:

```python
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()

example_input = torch.randn(1, 3, 224, 224).to(memory_format=torch.channels_last)
exported_program = torch.export.export(model, (example_input,))
graph_module = exported_program.module()

quantizer = CortexMQuantizer()
prepared = prepare_pt2e(graph_module, quantizer)

# Calibrate with representative data
for calibration_input in calibration_data:
    prepared(calibration_input)

quantized = convert_pt2e(prepared)
quantized_exported_program = torch.export.export(quantized, (example_input,))
```

### 2. Lower to edge and apply Cortex-M passes

Lower to the edge dialect with a custom `EdgeCompileConfig`, then run the `CortexMPassManager` to replace quantized subgraphs with CMSIS-NN operator implementations:

```python
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager

config = EdgeCompileConfig(
    preserve_ops=[
        torch.ops.aten.linear.default,
        torch.ops.aten.hardsigmoid.default,
        torch.ops.aten.hardsigmoid_.default,
        torch.ops.aten.hardswish.default,
        torch.ops.aten.hardswish_.default,
    ],
    _check_ir_validity=False,
    _core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default],
)

edge_program_manager = to_edge(quantized_exported_program, compile_config=config)

pass_manager = CortexMPassManager(edge_program_manager.exported_program())
edge_program_manager._edge_programs["forward"] = pass_manager.transform()
```

### 3. Serialize to .pte

```python
executorch_program = edge_program_manager.to_executorch(
    config=ExecutorchBackendConfig(extract_delegate_segments=False)
)

with open("model.pte", "wb") as f:
    f.write(executorch_program.buffer)
```

### 4. Cross-compile and run

Cross-compile the ExecuTorch runtime, Cortex-M kernels, and the example runner application. The first cmake invocation builds the ExecuTorch libraries for Arm baremetal. The second builds the [arm_executor_runner](https://github.com/pytorch/executorch/blob/main/examples/arm/executor_runner/) and links it against those libraries with the `.pte` model baked in.

```bash
# Build ExecuTorch libraries for Arm baremetal
cmake --preset arm-baremetal \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_DEVTOOLS=ON \
  -Bcmake-out-arm
cmake --build cmake-out-arm --target install -j$(nproc)

# Build the executor runner, linking the .pte into the binary
cmake -DCMAKE_TOOLCHAIN_FILE=$(pwd)/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DET_PTE_FILE_PATH=$(pwd)/model.pte \
      -DTARGET_CPU=cortex-m55 \
      -Bbuild \
      examples/arm/executor_runner
cmake --build build -j$(nproc) -- arm_executor_runner
```

Run on a simulated Cortex-M target:

```bash
backends/arm/scripts/run_fvp.sh --elf=build/arm_executor_runner --target=ethos-u55-128
```

For a complete end-to-end walkthrough including dataset setup, calibration, and result validation, see the [Cortex-M MobileNetV2 notebook](https://github.com/pytorch/executorch/blob/main/examples/arm/cortex_m_mv2_example.ipynb).

## Testing with Bundled I/O

The tutorial above produces a plain `.pte`. For programmatic testing,
`aot_arm_compiler --bundleio` instead produces a bundled (`.bpte`) program
that embeds reference inputs and expected outputs; the Cortex-M test runner
loads the bundle via semihosting and self-checks its outputs against the
embedded references, emitting `Test_result: PASS` or `Test_result: FAIL`
on the UART.

The driver for this flow is `examples/arm/run.sh`, which exports the model,
builds the Cortex-M test runner, launches the Corstone-300 FVP with
semihosting enabled, and checks the bundled output. Run it from the
ExecuTorch repo root after `./install_executorch.sh`:

```bash
# One-time: install the Arm toolchain + FVP.
examples/arm/setup.sh --i-agree-to-the-contained-eula
source examples/arm/arm-scratch/setup_path.sh

# Per model: export, build, and run on the FVP in one step.
# (Quantization is the default for the cortex-m55+int8 target.)
examples/arm/run.sh \
    --model_name=<model> \
    --target=cortex-m55+int8 \
    --bundleio
```

Replace `<model>` with any of the validated-model names in the table
below. Without `--calibration_data`, calibration falls back to the model's
`get_example_inputs()` (random data) — enough for bundled-I/O numerical
parity, but not for task-accuracy claims. On `Test_result: FAIL`, inspect
the FVP UART log for the per-tensor diff; supplying a representative
calibration dataset via `--calibration_data=<dir>` often resolves
mismatches caused by random-input calibration.

:::{important}
Bundled I/O checks INT8 **numerical parity** between the exported `.bpte`
and the eager-mode quantized model on reference inputs; it does not
validate task accuracy (VWW / KWS / ImageNet).
:::

## Validated Models

The following models are exported, INT8 quantized, lowered, and validated
end-to-end on the Corstone-300 FVP:

| Model              | Task                               | Input shape   | Source                                            | Test                                                     |
|--------------------|------------------------------------|---------------|---------------------------------------------------|----------------------------------------------------------|
| `mv2`              | Image classification               | `1x3x224x224` | `examples/models/mobilenet_v2/`                   | `backends/cortex_m/test/models/test_mobilenet_v2.py`     |
| `mv3`              | Image classification               | `1x3x224x224` | `examples/models/mobilenet_v3/`                   | `backends/cortex_m/test/models/test_mobilenet_v3.py`     |
| `ds_cnn`           | Keyword spotting (MLPerf Tiny)     | `1x1x49x10`   | `examples/models/mlperf_tiny/ds_cnn.py`           | `backends/cortex_m/test/models/test_ds_cnn.py`           |
| `mobilenet_v1_025` | Visual Wake Words (MLPerf Tiny)    | `1x3x96x96`   | `examples/models/mlperf_tiny/mobilenet_v1_025.py` | `backends/cortex_m/test/models/test_mobilenet_v1_025.py` |
| `resnet8`          | Image classification (MLPerf Tiny) | `1x3x32x32`   | `examples/models/mlperf_tiny/resnet8.py`          | `backends/cortex_m/test/models/test_resnet8.py`          |
| `deep_autoencoder` | Anomaly detection (MLPerf Tiny)    | `1x640`       | `examples/models/mlperf_tiny/deep_autoencoder.py` | `backends/cortex_m/test/models/test_deep_autoencoder.py` |

:::{note}
`mobilenet_v1_025` is the MLPerf Tiny Visual Wake Words benchmark
(MobileNetV1 with width multiplier 0.25) — the canonical person-detection
reference model for TinyML.
:::
