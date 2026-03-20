# Arm Cortex-M Backend

The Arm&reg; Cortex&reg;-M backend accelerates quantized model execution on Arm Cortex-M CPUs using [CMSIS-NN](https://arm-software.github.io/CMSIS-NN/latest/) optimized kernels. Unlike delegate-based backends, it operates as an operator library: quantized subgraphs are replaced with CMSIS-NN accelerated kernels during the pass-lowering stage, while unsupported operators fall back to portable fp32 kernels.

## Target Support

The backend targets Arm Cortex-M CPUs via CMSIS-NN, which provides optimized kernel implementations for three instruction set variants:

| Variant | Description | Example CPUs |
|---------|-------------|--------------|
| MVE (Helium) | Vector extensions for Arm-M | Cortex-M55, Cortex-M85 |
| DSP | DSP extension instructions | Cortex-M4, Cortex-M7, Cortex-M33 |
| Pure C | Reference C implementation | Any Cortex-M |

Testing has only been done with MVE targets (Cortex-M55, Cortex-M85). DSP and pure C CMSIS-NN kernels might work as well since the same CMSIS-NN API is used across all variants, but is unverified at this point.

## CMSIS-NN Supported Operators

| Operator | 8w8a | 8w16a | 4w8a |
|---|---|---|---|
| Conv2D | ✅ | ⬜ | ⬜ |
| DepthwiseConv2D | ✅ | ⬜ | ⬜ |
| TransposeConv2D | ✅ | ⬜ | ⬜ |
| Fully Connected | ✅ | ⬜ | ⬜ |
| Batch Matmul | ✅ | ⬜ | ⬜ |
| Add | ✅ | ⬜ | N/A |
| Mul | ✅ | ⬜ | N/A |
| MaxPooling | ✅ | ⬜ | N/A |
| AvgPooling | ✅ | ⬜ | N/A |
| Softmax | ✅ | ⬜ | N/A |
| Pad | ✅ | ⬜ | N/A |
| LSTM | ⬜ | ⬜ | ⬜ |
| SVDF | ⬜ | ⬜ | ⬜ |

## Quantization Support

The Cortex-M backend currently implements **symmetric INT8 (8w8a)** quantization:
- **Per-channel** quantization for convolution operators.
- **Per-tensor** quantization for all other supported operators.
- **Shared quantization parameters** for data-movement operators (e.g. reshape, permute) to avoid unnecessary requantization.

CMSIS-NN also supports INT4 weights with INT8 activations (4w8a) and INT8 weights with INT16 activations (8w16a), but the corresponding quantizer configuration and operator implementations are not yet integrated.

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
