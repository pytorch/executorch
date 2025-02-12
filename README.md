<div align="center">
  <img src="./docs/source/_static/img/et-logo.png" alt="Logo" width="200">
  <h1 align="center">ExecuTorch: A powerful on-device AI Framework</h1>
</div>


<div align="center">
  <a href="https://github.com/pytorch/executorch/graphs/contributors"><img src="https://img.shields.io/github/contributors/pytorch/executorch?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/pytorch/executorch/stargazers"><img src="https://img.shields.io/github/stars/pytorch/executorch?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://discord.gg/Dh43CKSAdc"><img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a>
  <a href="https://pytorch.org/executorch/stable/index.html"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <hr>
</div>

**ExecuTorch** is an end-to-end solution for on-device inference and training. It powers much of Meta's on-device AI experiences across Facebook, Instagram, Meta Quest, Ray-Ban Meta Smart Glasses, WhatsApp, and more.

It supports a wide range of models including LLMs (Large Language Models), CV (Computer Vision), ASR (Automatic Speech Recognition), and TTS (Text to Speech).

Platform Support:
- Operating Systems:
  - iOS
  - Mac
  - Android
  - Linux
  - Microcontrollers

- Hardware Acceleration:
  - Apple
  - Arm
  - Cadence
  - MediaTek
  - OpenVINO
  - Qualcomm
  - Vulkan
  - XNNPACK

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

## Getting Started
To get started you can:

- Visit the [Step by Step Tutorial](https://pytorch.org/executorch/main/index.html) on getting things running locally and deploy a model to a device
- Use this [Colab Notebook](https://pytorch.org/executorch/stable/getting-started-setup.html#quick-setup-colab-jupyter-notebook-prototype) to start playing around right away
- Jump straight into LLMs use cases by following specific instructions for [Llama](./examples/models/llama/README.md) and [Llava](./examples/models/llava/README.md)

## Feedback and Engagement

We welcome any feedback, suggestions, and bug reports from the community to help
us improve our technology. Check out the [Discussion Board](https://github.com/pytorch/executorch/discussions) or chat real time with us on [Discord](https://discord.gg/Dh43CKSAdc)

## Contributing

We welcome contributions. To get started review the [guidelines](CONTRIBUTING.md) and chat with us on [Discord](https://discord.gg/Dh43CKSAdc)


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
|   ├── executor                    #  Model loading, initialization, and execution.
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
