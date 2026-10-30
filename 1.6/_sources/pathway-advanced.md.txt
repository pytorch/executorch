(pathway-advanced)=
# Advanced Pathway

This pathway is for engineers building production-grade deployments, implementing custom backends, optimizing for constrained hardware, or working with large language models on edge devices. It assumes familiarity with the ExecuTorch export pipeline and at least one successful model deployment.

---

## Advanced Topic Areas

Select the area most relevant to your current work. Each section provides a curated sequence of documentation with dependencies noted.

---

### Quantization and Optimization

Quantization is the most impactful optimization available in ExecuTorch, reducing model size by 2–8× and improving latency significantly on supported backends.

::::{grid} 2
:gutter: 2

:::{grid-item-card} Quantization Overview
:link: quantization
:link-type: doc

Introduction to ExecuTorch's quantization framework, including supported schemes (INT8, INT4, FP16) and the relationship between quantization and backend selection.

**Difficulty:** Intermediate
:::

:::{grid-item-card} Quantization & Optimization (Advanced)
:link: quantization-optimization
:link-type: doc

Advanced quantization techniques including mixed-precision, per-channel quantization, and calibration workflows for production models.

**Difficulty:** Advanced
:::

:::{grid-item-card} Model Export and Lowering
:link: using-executorch-export
:link-type: doc

Full reference for `to_edge_transform_and_lower`, including quantization integration, dynamic shapes, and multi-backend lowering.

**Difficulty:** Advanced
:::

:::{grid-item-card} Backend Dialect
:link: compiler-backend-dialect
:link-type: doc

Understanding the Backend Dialect IR and how it differs from Edge Dialect — essential for backend developers and advanced export customization.

**Difficulty:** Advanced
:::

::::

---

### Computer Vision Models

Computer vision apps need a precise contract for image resizing, crop behavior, tensor layout, dtype conversion, normalization, and output decoding.

::::{grid} 2
:gutter: 2

:::{grid-item-card} Working with Computer Vision Models
:link: working-with-cv-models
:link-type: doc

Guidance for preprocessing placement, Android and iOS image-to-tensor conversion, and classifier or segmentation output interpretation.

**Difficulty:** Intermediate
:::

:::{grid-item-card} Getting Started with ExecuTorch
:link: getting-started
:link-type: doc

End-to-end MobileNet V2 export, validation, and mobile runtime links for a first image classification workflow.

**Difficulty:** Beginner
:::

::::

---

### Memory Planning and Runtime Optimization

Memory planning is critical for constrained devices. ExecuTorch provides ahead-of-time memory planning to eliminate runtime allocations.

::::{grid} 2
:gutter: 2

:::{grid-item-card} Memory Planning
:link: compiler-memory-planning
:link-type: doc

How ExecuTorch plans tensor memory at compile time, including memory hierarchy, buffer reuse strategies, and how to customize the planner.

**Difficulty:** Advanced
:::

:::{grid-item-card} Memory Planning Inspection
:link: memory-planning-inspection
:link-type: doc

Tools for inspecting memory plans and diagnosing memory-related issues in exported programs.

**Difficulty:** Advanced
:::

:::{grid-item-card} Managing Tensor Memory in C++
:link: extension-tensor
:link-type: doc

The `TensorPtr` and `from_blob` APIs for zero-copy tensor management in C++ runtime integrations.

**Difficulty:** Intermediate
:::

:::{grid-item-card} Portable C++ Programming
:link: portable-cpp-programming
:link-type: doc

Guidelines for writing ExecuTorch C++ code that runs on bare-metal and RTOS environments without dynamic allocation or standard library dependencies.

**Difficulty:** Advanced
:::

::::

---

### Custom Compiler Passes and Kernel Registration

ExecuTorch's compiler pass interface allows you to transform the exported graph before lowering, enabling model-specific optimizations and operator fusion.

::::{grid} 2
:gutter: 2

:::{grid-item-card} Custom Compiler Passes
:link: compiler-custom-compiler-passes
:link-type: doc

Writing and registering custom graph transformation passes that run during the export and lowering pipeline.

**Difficulty:** Advanced
:::

:::{grid-item-card} Kernel Registration
:link: kernel-library-custom-aten-kernel
:link-type: doc

Registering custom ATen kernel implementations to replace or supplement the portable operator library with hardware-optimized versions.

**Difficulty:** Advanced
:::

:::{grid-item-card} Kernel Library Overview
:link: kernel-library-overview
:link-type: doc

Architecture of ExecuTorch's kernel library system, including the portable library, custom kernels, and selective build.

**Difficulty:** Advanced
:::

:::{grid-item-card} Selective Build
:link: kernel-library-selective-build
:link-type: doc

Reduce binary size by including only the operators required by your specific model using the selective build system.

**Difficulty:** Advanced
:::

::::

---

### Backend Delegate Development

Implementing a new hardware backend for ExecuTorch requires understanding the delegate interface, partitioner API, and runtime integration.

::::{grid} 2
:gutter: 2

:::{grid-item-card} Backend Development Guide
:link: backend-development
:link-type: doc

Complete guide to implementing a new ExecuTorch backend delegate, including the `BackendInterface`, `preprocess`, and `execute` methods.

**Difficulty:** Advanced
:::

:::{grid-item-card} Integrating a Backend Delegate
:link: backend-delegates-integration
:link-type: doc

Step-by-step walkthrough of integrating an existing backend delegate into the ExecuTorch build system and runtime.

**Difficulty:** Beginner (for integration) / Advanced (for implementation)
:::

:::{grid-item-card} Delegate and Partitioner
:link: compiler-delegate-and-partitioner
:link-type: doc

The `Partitioner` interface for selecting which subgraphs to delegate, including pattern matching and constraint specification.

**Difficulty:** Advanced
:::

:::{grid-item-card} Backend Delegate Implementation and Linking
:link: runtime-backend-delegate-implementation-and-linking
:link-type: doc

Linking backend delegate implementations into the ExecuTorch runtime, including static and dynamic registration patterns.

**Difficulty:** Advanced
:::

:::{grid-item-card} Lowering a Model as a Delegate
:link: examples-end-to-end-to-lower-model-to-delegate
:link-type: doc

End-to-end example of using `to_backend` to lower a model subgraph to a custom delegate.

**Difficulty:** Advanced
:::

:::{grid-item-card} Debugging Backend Delegates
:link: debug-backend-delegate
:link-type: doc

Techniques for debugging delegate execution, including intermediate output comparison and delegate-specific logging.

**Difficulty:** Advanced
:::

::::

---

### C++ Runtime Integration

For embedded, mobile native, and server deployments, the C++ runtime APIs provide full control over model loading, execution, and memory management.

::::{grid} 2
:gutter: 2

:::{grid-item-card} Module Extension (High-Level API)
:link: extension-module
:link-type: doc

The `Module` class provides a high-level C++ API for loading and running `.pte` files with minimal boilerplate. Recommended for most C++ integrations.

**Difficulty:** Intermediate
:::

:::{grid-item-card} Detailed C++ Runtime APIs
:link: running-a-model-cpp-tutorial
:link-type: doc

Low-level C++ runtime APIs for fine-grained control over memory allocation, operator dispatch, and execution planning. Required for bare-metal targets.

**Difficulty:** Intermediate
:::

:::{grid-item-card} Using ExecuTorch with C++
:link: using-executorch-cpp
:link-type: doc

CMake integration, target linking, cross-compilation setup, and C++ API reference for production deployments.

**Difficulty:** Advanced
:::

:::{grid-item-card} Runtime Platform Abstraction Layer
:link: runtime-platform-abstraction-layer
:link-type: doc

The PAL interface for porting ExecuTorch to new operating systems and bare-metal environments.

**Difficulty:** Advanced
:::

::::

---

### Large Language Models on Edge

Deploying LLMs to edge devices involves additional complexity around quantization, tokenization, KV-cache management, and platform-specific optimizations.

::::{grid} 2
:gutter: 2

:::{grid-item-card} LLM Overview
:link: llm/working-with-llms
:link-type: doc

Complete overview of the ExecuTorch LLM workflow, supported models, and platform-specific deployment paths.

**Difficulty:** Intermediate
:::

:::{grid-item-card} Exporting LLMs
:link: llm/export-llm
:link-type: doc

The `export_llm` module for exporting supported LLMs (Llama, Qwen, Phi, SmolLM) with quantization and optimization.

**Difficulty:** Intermediate
:::

:::{grid-item-card} Exporting Custom LLMs
:link: llm/export-custom-llm
:link-type: doc

Adapting the export pipeline for custom LLM architectures beyond the officially supported models, using nanoGPT as a worked example.

**Difficulty:** Intermediate
:::

:::{grid-item-card} Running LLMs with C++
:link: llm/run-with-c-plus-plus
:link-type: doc

C++ runtime integration for LLM inference, including tokenizer setup, KV-cache configuration, and streaming output.

**Difficulty:** Intermediate
:::

:::{grid-item-card} Llama on Qualcomm Android
:link: llm/build-run-llama3-qualcomm-ai-engine-direct-backend
:link-type: doc

Deploying Llama 3 3B Instruct on Android using the Qualcomm AI Engine Direct backend with hardware acceleration.

**Difficulty:** Advanced
:::

:::{grid-item-card} ExecuTorch on Raspberry Pi
:link: raspberry_pi_llama_tutorial
:link-type: doc

Deploying Llama models on Raspberry Pi 4/5 edge devices using the ExecuTorch runtime.

**Difficulty:** Intermediate
:::

::::

---

### Developer Tools and Debugging

ExecuTorch provides a comprehensive suite of profiling and debugging tools for diagnosing performance and correctness issues.

::::{grid} 2
:gutter: 2

:::{grid-item-card} Developer Tools Overview
:link: devtools-overview
:link-type: doc

Overview of the ExecuTorch developer tools suite, including ETRecord, ETDump, and the Inspector API.

**Difficulty:** Intermediate
:::

:::{grid-item-card} Profiling a Model
:link: devtools-tutorial
:link-type: doc

Step-by-step tutorial for profiling model execution using ETRecord and ETDump to identify performance bottlenecks.

**Difficulty:** Intermediate
:::

:::{grid-item-card} Profiling and Debugging
:link: using-executorch-troubleshooting
:link-type: doc

Comprehensive debugging guide covering numerical debugging, operator-level profiling, and common failure modes.

**Difficulty:** Advanced
:::

:::{grid-item-card} Delegate Debugging
:link: delegate-debugging
:link-type: doc

Techniques specific to debugging backend delegate execution, including output comparison and delegate-level tracing.

**Difficulty:** Advanced
:::

::::

---

### IR and Compiler Internals

For contributors and advanced backend developers who need to understand ExecuTorch's compiler internals.

::::{grid} 2
:gutter: 2

:::{grid-item-card} Export Overview
:link: export-overview
:link-type: doc

The complete export pipeline from `torch.export` to `.pte`, including the role of each compilation stage.

**Difficulty:** Intermediate
:::

:::{grid-item-card} Compiler Entry Points
:link: compiler-entry-points
:link-type: doc

The public API surface for the ExecuTorch compiler, including `to_edge`, `to_edge_transform_and_lower`, and `to_executorch`.

**Difficulty:** Intermediate
:::

:::{grid-item-card} IR Specification
:link: ir-specification
:link-type: doc

Formal specification of the ExecuTorch IR, including operator semantics, type system, and serialization format.

**Difficulty:** Advanced
:::

:::{grid-item-card} Compiler & IR (Advanced)
:link: compiler-ir-advanced
:link-type: doc

Advanced IR topics including graph transformations, custom dialects, and the relationship between Export IR and Edge Dialect.

**Difficulty:** Advanced
:::

::::

---

## Contributing to ExecuTorch

If you are working on ExecuTorch internals or want to contribute upstream, start with the contributor guide.

::::{grid} 2
:gutter: 2

:::{grid-item-card} New Contributor Guide
:link: new-contributor-guide
:link-type: doc

Development environment setup, code style, testing requirements, and the pull request process for ExecuTorch contributors.

**Difficulty:** Advanced
:::

:::{grid-item-card} API Life Cycle and Deprecation Policy
:link: api-life-cycle
:link-type: doc

How ExecuTorch manages API stability, deprecation timelines, and backward compatibility across releases.

**Difficulty:** Intermediate
:::

::::

---

## Advanced Learning Sequence

If you prefer a structured progression rather than topic-based navigation, follow this sequence for a comprehensive advanced curriculum.

```{list-table}
:header-rows: 1
:widths: 10 30 60

* - **Order**
  - **Topic**
  - **Goal**
* - 1
  - {doc}`export-overview`
  - Understand the full compilation pipeline
* - 2
  - {doc}`using-executorch-export`
  - Master advanced export options
* - 3
  - {doc}`working-with-cv-models`
  - Define image preprocessing, tensor layout, and output decoding for CV apps
* - 4
  - {doc}`quantization-optimization`
  - Apply production-grade quantization
* - 5
  - {doc}`compiler-memory-planning`
  - Optimize memory for constrained devices
* - 6
  - {doc}`compiler-custom-compiler-passes`
  - Write custom graph transformations
* - 7
  - {doc}`backend-development`
  - Implement a custom backend delegate
* - 8
  - {doc}`running-a-model-cpp-tutorial`
  - Master the low-level C++ runtime
* - 9
  - {doc}`devtools-tutorial`
  - Profile and debug production models
```
