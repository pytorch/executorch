# Examples


The series of demos featured in this directory exemplify a broad spectrum of workflows for deploying ML models on edge devices using ExecuTorch. These demos offer practical insights into key processes such as model exporting, quantization, backend delegation, module composition, memory planning, program saving and  loading for inference on ExecuTorch runtime.

ExecuTorch's extensive support spans from simple modules like "Add" to comprehensive models like `MobileNet V3`, `Wav2Letter`, `Llama 2`, and more, showcasing its versatility in enabling the deployment of a wide spectrum of models across various edge AI applications.


## Directory structure
```
examples
├── llm_manual                        # A storage place for the files that [LLM Maunal](https://pytorch.org/executorch/main/llm/getting-started.html) needs
├── models                            # Contains a set of popular and representative PyTorch models
├── portable                          # Contains end-to-end demos for ExecuTorch in portable mode
├── selective_build                   # Contains demos of selective build for optimizing the binary size of the ExecuTorch runtime
├── devtools                          # Contains demos of BundledProgram and ETDump
├── demo-apps                         # Contains demo apps for Android and iOS
├── xnnpack                           # Contains end-to-end ExecuTorch demos with first-party optimization using XNNPACK
├── apple
|   |── coreml                        # Contains demos of Apple's Core ML backend
|   └── mps                           # Contains end-to-end demos of MPS backend
├── arm                               # Contains demos of the Arm TOSA and Ethos-U NPU flows
├── qualcomm                          # Contains demos of Qualcomm QNN backend
├── cadence                           # Contains demos of exporting and running a simple model on Xtensa DSPs
├── third-party                       # Third-party libraries required for working on the demos
└── README.md                         # This file
```


## Using the examples

A user's journey may commence by exploring the demos located in the [`portable/`](./portable) directory. Here, you will gain insights into the fundamental end-to-end workflow to generate a binary file from a ML model in [portable mode](../docs/source/concepts.md##portable-mode-lean-mode) and run it on the ExecuTorch runtime.

## Demos Apps

Explore mobile apps with ExecuTorch models integrated and deployable on [Android](./demo-apps/android) and [iOS]((./demo-apps/apple_ios)). This provides end-to-end instructions on how to export Llama models, load on device, build the app, and run it on device.

For specific details related to models and backend, you can explore the various subsections.

### Llama Models

[This page](./models/llama2/README.md) demonstrates how to run Llama 3.2 (1B, 3B), Llama 3.1 (8B), Llama 3 (8B), and Llama 2 7B models on mobile via ExecuTorch. We use XNNPACK, QNNPACK, MediaTek, and MPS to accelerate the performance and 4-bit groupwise PTQ quantization to fit the model on Android and iOS mobile phones.

### Llava1.5 7B

[This page](./models/llava/README.md) demonstrates how to run [Llava 1.5 7B](https://github.com/haotian-liu/LLaVA) model on mobile via ExecuTorch. We use XNNPACK to accelerate the performance and 4-bit groupwise PTQ quantization to fit the model on Android and iOS mobile phones.

### Selective Build

To understand how to deploy the ExecuTorch runtime with optimization for binary size, explore the demos available in the [`selective_build/`](./selective_build) directory. These demos are specifically designed to illustrate the [Selective Build](../docs/source/kernel-library-selective_build.md), offering insights into reducing the binary size while maintaining efficiency.

### Developer Tools

You will find demos of [ExecuTorch Developer Tools](./devtools/) in the [`devtools/`](./devtools/) directory. The examples focuses on exporting and executing BundledProgram for ExecuTorch model verification and ETDump for collecting profiling and debug data.

### XNNPACK delegation

The demos in the [`xnnpack/`](./xnnpack) directory provide valuable insights into the process of lowering and executing an ExecuTorch model with built-in performance enhancements. These demos specifically showcase the workflow involving [XNNPACK backend](https://github.com/pytorch/executorch/tree/main/backends/xnnpack) delegation and quantization.

### Apple Backend

You will find demos of [ExecuTorch Core ML Backend](./apple/coreml/) in the [`apple/coreml/`](./apple/coreml) directory and [MPS Backend](./apple/mps/) in the [`apple/mps/`](./apple/mps) directory.

### ARM Cortex-M55 + Ethos-U55 Backend

The [`arm/`](./arm) directory contains scripts to help you run a PyTorch model on a ARM Corstone-300 platform via ExecuTorch.

### QNN Backend

You will find demos of [ExecuTorch QNN Backend](./qualcomm) in the [`qualcomm/`](./qualcomm) directory.

### Cadence HiFi4 DSP

The [`Cadence/`](./cadence) directory hosts a demo that showcases the process of exporting and executing a model on Xtensa Hifi4 DSP. You can utilize [this tutorial](../docs/source/build-run-xtensa.md) to guide you in configuring the demo and running it.

## Dependencies

Various models and workflows listed in this directory have dependencies on some other packages. You need to follow the setup guide in [Setting up ExecuTorch from GitHub](https://pytorch.org/executorch/stable/getting-started-setup) to have appropriate packages installed.

# Disclaimer

The ExecuTorch Repository Content is provided without any guarantees about performance or compatibility. In particular, ExecuTorch makes available model architectures written in Python for PyTorch that may not perform in the same manner or meet the same standards as the original versions of those models. When using the ExecuTorch Repository Content, including any model architectures, you are solely responsible for determining the appropriateness of using or redistributing the ExecuTorch Repository Content and assume any risks associated with your use of the ExecuTorch Repository Content or any models, outputs, or results, both alone and in combination with any other technologies. Additionally, you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models, weights, data, or other technologies, and you are solely responsible for complying with all such obligations.
