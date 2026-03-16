(advanced-topics-section)=

# Advanced

Deep dive into ExecuTorch's advanced features for optimization, customization, and integration.

This section covers advanced concepts for developers who need to customize ExecuTorch for specific use cases, optimize performance, or integrate with custom hardware backends.

## Quantization & Optimization

Techniques for model compression and performance optimization.

**→ {doc}`quantization-optimization` — Quantization strategies and performance optimization**

Key topics:

- Quantization strategies and techniques
- Performance profiling and optimization

## Model Export

Learn the core ExecuTorch workflow, exporting PyTorch models to the `.pte` format for edge deployment.

**→ {doc}`using-executorch-export`** - Model Export & Lowering

Key topics:

- Export and Lowering Workflow
- Hardware Backend Selection & Optimization
- Dynamic Shapes & Advanced Model Features


## Kernel Library

Deep dive into ExecuTorch's kernel implementation and customization.

**→ {doc}`kernel-library-advanced` — Kernel library deep dive and customization**

Key topics:

- Kernel library architecture
- Custom kernel implementation
- Selective build and optimization

## Backend & Delegates

**→ {doc}`backend-delegate-advanced` — Backend delegate integration**

Key topics:

- Learn how to integrate Backend Delegate into ExecuTorch and more
- XNNPACK Delegate Internals
- Debugging Delegation


## Runtime & Integration

Advanced runtime features and backend integration.

**→ {doc}`runtime-integration-advanced` — Runtime customization and backend integration**

Key topics:

- Backend delegate implementation
- Platform abstraction layer
- Custom runtime integration

## Compiler & IR

Advanced compiler features and intermediate representation details.

**→ {doc}`compiler-ir-advanced` — Compiler passes and IR specification**

Key topics:

- Custom compiler passes
- Memory planning strategies
- Backend dialect and EXIR
- Ops set definition


## File Formats

ExecuTorch file format specifications and internals.

**→ {doc}`file-formats-advanced` — PTE and PTD file format specifications**

Key topics:

- PTE file format internals
- PTD file format specification
- Custom file format handling

## Next Steps

After exploring advanced topics:

- **{doc}`tools-sdk-section`** - Developer tools for debugging and profiling
- **{doc}`api-section`** - Complete API reference documentation

```{toctree}
:hidden:
:maxdepth: 2
:caption: Advanced Topics

quantization-optimization
using-executorch-export
kernel-library-advanced
backend-delegate-advanced
runtime-integration-advanced
compiler-ir-advanced
file-formats-advanced
