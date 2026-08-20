# ExecuTorch CMSIS Pack

## Overview

ExecuTorch is the PyTorch Edge Runtime, enabling efficient on-device AI inference. This pack is the Cortex-M / bare-metal distribution.

This pack provides:
- **Core Runtime**: Program loading and execution
- **Portable Operators**: Platform-independent operator implementations
- **Quantized Operators**: Optimized quantized inference
- **Ethos-U Backend**: Hardware acceleration for Arm Ethos-U NPU
- **Cortex-M Backend**: CMSIS-NN optimized operators

## Getting Started

### Basic Usage

1. Add the pack to your csolution:
   ```yaml
   packs:
     - pack: PyTorch::ExecuTorch
   ```

2. Add required components:
   ```yaml
   components:
     # Core runtime (always required)
     - component: Machine Learning:ExecuTorch:Runtime
     - component: Machine Learning:ExecuTorch:Kernel Utils
     
     # Backend (choose one or more)
     - component: Machine Learning:ExecuTorch:Backend::EthosU
     # or
     - component: Machine Learning:ExecuTorch:Backend::CortexM
   ```

3. Include ExecuTorch headers in your code:
   ```cpp
   #include <executorch/runtime/executor/program.h>
   #include <executorch/runtime/executor/method.h>
   ```

### Model Integration

Operators are included as individual CMSIS components, one per
`op_*.cpp`. Add the ones your model uses:

```yaml
components:
  - component: Machine Learning:ExecuTorch Operators:Portable add
  - component: Machine Learning:ExecuTorch Operators:Portable mul
  # ... other operators used by your model
```

You can also generate this list automatically: a Model Pack produced
from your `.pte` file declares dependencies on the operators the model
needs, and the toolchain resolves them against this pack's components.

## Memory Requirements

| Component | Flash (approx) | RAM (approx) |
|-----------|----------------|--------------|
| Runtime | 50-100 KB | 4-8 KB |
| Per Operator | 1-10 KB | minimal |
| Ethos-U Backend | 20-40 KB | 2-4 KB |

Actual requirements depend on:
- Selected operators
- Model complexity
- Tensor sizes

## Compiler Support

Tested with `arm-none-eabi-gcc` 13.x+ via the avh-mlops vcpkg toolchain.
Other Arm-Cortex-M toolchains (Arm Compiler 6, LLVM Embedded) are expected
to work but are not currently exercised by the pack's CI.

## Dependencies

- ARM::CMSIS (core headers)
- ARM::CMSIS-NN (for Cortex-M backend)
- ARM::ethos-u-core-driver (for Ethos-U backend)

## Resources

- [ExecuTorch Documentation](https://pytorch.org/executorch/)
- [CMSIS-Pack Specification](https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/index.html)
- [Issue Tracker](https://github.com/pytorch/executorch/issues)
