# ExecuTorch CMSIS Pack

## Overview

ExecuTorch is the PyTorch Edge Runtime, enabling efficient on-device AI inference on Arm Cortex-M processors.

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

For automatic operator selection, use a Model Pack generated from your `.pte` file. The model pack declares dependencies on required operators, which are automatically resolved.

Alternatively, manually add required operators:
```yaml
components:
  - component: Machine Learning:ExecuTorch:Operators::Portable::add
  - component: Machine Learning:ExecuTorch:Operators::Portable::mul
  # ... other operators used by your model
```

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

- **GCC**: arm-none-eabi-gcc 13.x+
- **Arm Compiler 6**: armclang 6.24+
- **LLVM/Clang**: Arm Compiler for Embedded

## Dependencies

- ARM::CMSIS (core headers)
- ARM::CMSIS-NN (for Cortex-M backend)
- ARM::ethos-u-core-driver (for Ethos-U backend)

## Resources

- [ExecuTorch Documentation](https://pytorch.org/executorch/)
- [CMSIS-Pack Specification](https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/index.html)
- [Issue Tracker](https://github.com/pytorch/executorch/issues)
