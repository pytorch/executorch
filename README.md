# ExecuTorch

**ExecuTorch** is an end-to-end solution for enabling on-device inference
capabilities across mobile and edge devices including wearables, embedded
devices and microcontrollers. It is part of the PyTorch Edge ecosystem and
enables efficient deployment of PyTorch models to edge devices.

Key value propositions of ExecuTorch are:

- **Portability:** Compatibility with a wide variety of computing platforms,
  from high-end mobile phones to highly constrained embedded systems and
  microcontrollers.
- **Productivity:** Enabling developers to use the same toolchains and SDK from
  PyTorch model authoring and conversion, to debugging and deployment to a wide
  variety of platforms.
- **Performance:** Providing end users with a seamless and high-performance
  experience due to a lightweight runtime and utilizing full hardware
  capabilities such as CPUs, NPUs, and DSPs.

For a comprehensive technical overview of ExecuTorch and step-by-step tutorials,
please visit our documentation website [for the latest release](https://pytorch.org/executorch/stable/index.html) (or the [main branch](https://pytorch.org/executorch/main/index.html)).

## Important: This is a preview release

This is a preview version of ExecuTorch and should be used for testing and
evaluation purposes only. It is not recommended for use in production settings.
We welcome any feedback, suggestions, and bug reports from the community to help
us improve the technology. Please use the [PyTorch
Forums](https://discuss.pytorch.org/c/executorch) for discussion and feedback
about ExecuTorch using the **ExecuTorch** category, and our [GitHub
repository](https://github.com/pytorch/executorch/issues) for bug reporting.

The ExecuTorch code and APIs are still changing quickly, and there are not yet
any guarantees about forward/backward source compatibility. We recommend using
the latest `v#.#.#` release tag from the
[Releases](https://github.com/pytorch/executorch/releases) page when
experimenting with this preview release.

## Directory Structure

```
executorch
├── backends                        #  Backend delegate implementations.
├── build                           #  Utilities for managing the build system.
├── bundled_program                 #  Utilities for attaching reference inputs and outputs to models. TODO move to extension
├── codegen                         #  Tooling to autogenerate bindings between kernels and the runtime. TODO move to tool
├── configurations                  #  TODO delete this
├── docs                            #  Static docs tooling
├── examples                        #  Examples of various user flows, such as model export, delegates, and runtime execution.
├── exir                            #  Ahead of time library, model capture and lowering apis.
|   ├── _serialize                  #  Serialize final export artifact.
|   ├── backend                     #  Backend delegate ahead of time APIs
|   ├── capture                     #  Program capture.
|   ├── dialects                    #  Op sets for various dialects in the export process.
|   ├── emit                        #  Conversion from ExportedProgram to ExecuTorch execution instructions.
|   ├── passes                      #  Built-in compiler passes.
|   ├── program                     #  Export artifacts.
|   ├── verification                #  IR verification.
├── extension                       #  Extensions built on top of the runtime.
|   ├── aten_util
|   ├── data_loader                 #  1st party data loader implementations.
|   ├── memory_allocator            #  1st party memory allocator implementations.
|   ├── pybindings                  #  Python api for executorch runtime.
|   ├── pytree                      #  C++ and Python flattening and unflattening lib for pytrees.
|   ├── testing_util
├── kernels                         #  1st party kernel implementations.
|   ├── aten
|   ├── optimized
|   ├── portable                    #  Reference implementations of ATen operators.
|   ├── prim_ops                    #  Special ops used in executorch runtime for control flow and symbolic primitives.
|   ├── quantized
├── profiler                        #  Utilities for profiling. TODO delete in favor of ETDump in sdk/
├── runtime                         #  core cpp runtime of executorch
|   ├── backend                     #  Backend delegate runtime APIs
|   ├── core                        #  Core structures used across all levels of the runtime
|   ├── executor                    #  Model loading, initalization, and execution.
|   ├── kernel                      #  Kernel registration and management.
|   ├── platform                    #  Layer between architecture specific code and user calls.
├── schema                          #  ExecuTorch program definition, TODO move under serialization/
├── scripts                         #  Utility scripts for size management, dependency management, etc.
├── sdk                             #  Model profiling, debugging, and introspection.
├── shim                            #  Compatibility layer between OSS and Internal builds
├── test                            #  Broad scoped end2end tests
├── third-party                     #  third-party dependencies
├── util                            #  TODO delete this
```

## License
ExecuTorch is BSD licensed, as found in the LICENSE file.
