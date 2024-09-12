# ExecuTorch

**ExecuTorch** is an end-to-end solution for enabling on-device inference
capabilities across mobile and edge devices including wearables, embedded
devices and microcontrollers. It is part of the PyTorch Edge ecosystem and
enables efficient deployment of PyTorch models to edge devices.

Key value propositions of ExecuTorch are:

- **Portability:** Compatibility with a wide variety of computing platforms,
  from high-end mobile phones to highly constrained embedded systems and
  microcontrollers.
- **Productivity:** Enabling developers to use the same toolchains and Developer
  Tools from PyTorch model authoring and conversion, to debugging and deployment
  to a wide variety of platforms.
- **Performance:** Providing end users with a seamless and high-performance
  experience due to a lightweight runtime and utilizing full hardware
  capabilities such as CPUs, NPUs, and DSPs.

For a comprehensive technical overview of ExecuTorch and step-by-step tutorials,
please visit our documentation website [for the latest release](https://pytorch.org/executorch/stable/index.html) (or the [main branch](https://pytorch.org/executorch/main/index.html)).

Check out the [Getting Started](https://pytorch.org/executorch/stable/getting-started-setup.html#quick-setup-colab-jupyter-notebook-prototype) page for a quick spin.

## Feedback

We welcome any feedback, suggestions, and bug reports from the community to help
us improve our technology. Please use the [PyTorch
Forums](https://discuss.pytorch.org/c/executorch) for discussion and feedback
about ExecuTorch using the **ExecuTorch** category, and our [GitHub
repository](https://github.com/pytorch/executorch/issues) for bug reporting.

We recommend using the latest release tag from the
[Releases](https://github.com/pytorch/executorch/releases) page when developing.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details about issues, PRs, code
style, CI jobs, and other development topics.

## Directory Structure

```
executorch
├── backends                        #  Backend delegate implementations.
├── build                           #  Utilities for managing the build system.
├── codegen                         #  Tooling to autogenerate bindings between kernels and the runtime.
├── configurations
├── docs                            #  Static docs tooling.
├── examples                        #  Examples of various user flows, such as model export, delegates, and runtime execution.
├── exir                            #  Ahead-of-time library: model capture and lowering APIs.
|   ├── _serialize                  #  Serialize final export artifact.
|   ├── backend                     #  Backend delegate ahead of time APIs
|   ├── capture                     #  Program capture.
|   ├── dialects                    #  Op sets for various dialects in the export process.
|   ├── emit                        #  Conversion from ExportedProgram to ExecuTorch execution instructions.
|   ├── operator                    #  Operator node manipulation utilities.
|   ├── passes                      #  Built-in compiler passes.
|   ├── program                     #  Export artifacts.
|   ├── serde                       #  Graph module
serialization/deserialization.
|   ├── verification                #  IR verification.
├── extension                       #  Extensions built on top of the runtime.
|   ├── android                     #  ExecuTorch wrappers for Android apps.
|   ├── apple                       #  ExecuTorch wrappers for iOS apps.
|   ├── aten_util                   #  Converts to and from PyTorch ATen types.
|   ├── data_loader                 #  1st party data loader implementations.
|   ├── evalue_util                 #  Helpers for working with EValue objects.
|   ├── gguf_util                   #  Tools to convert from the GGUF format.
|   ├── kernel_util                 #  Helpers for registering kernels.
|   ├── memory_allocator            #  1st party memory allocator implementations.
|   ├── module                      #  A simplified C++ wrapper for the runtime.
|   ├── parallel                    #  C++ threadpool integration.
|   ├── pybindings                  #  Python API for executorch runtime.
|   ├── pytree                      #  C++ and Python flattening and unflattening lib for pytrees.
|   ├── runner_util                 #  Helpers for writing C++ PTE-execution
tools.
|   ├── testing_util                #  Helpers for writing C++ tests.
|   ├── training                    #  Experimental libraries for on-device training
├── kernels                         #  1st party kernel implementations.
|   ├── aten
|   ├── optimized
|   ├── portable                    #  Reference implementations of ATen operators.
|   ├── prim_ops                    #  Special ops used in executorch runtime for control flow and symbolic primitives.
|   ├── quantized
├── profiler                        #  Utilities for profiling runtime execution.
├── runtime                         #  Core C++ runtime.
|   ├── backend                     #  Backend delegate runtime APIs.
|   ├── core                        #  Core structures used across all levels of the runtime.
|   ├── executor                    #  Model loading, initalization, and execution.
|   ├── kernel                      #  Kernel registration and management.
|   ├── platform                    #  Layer between architecture specific code and portable C++.
├── schema                          #  ExecuTorch PTE file format flatbuffer
schemas.
├── scripts                         #  Utility scripts for size management, dependency management, etc.
├── devtools                        #  Model profiling, debugging, and introspection.
├── shim                            #  Compatibility layer between OSS and Internal builds
├── test                            #  Broad scoped end-to-end tests.
├── third-party                     #  Third-party dependencies.
├── util                            #  Various helpers and scripts.
```

## License
ExecuTorch is BSD licensed, as found in the LICENSE file.
