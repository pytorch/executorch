(home)=
# Welcome to the ExecuTorch Documentation

**ExecuTorch** is PyTorch's solution to training and inference on the
Edge.

## Key Value Propositions

- **Portability:** Compatibility with a wide variety of computing
  platforms, from high-end mobile phones to highly constrained
  embedded systems and microcontrollers.
- **Productivity:** Enabling developers to use the same toolchains and
  Developer Tools from PyTorch model authoring and conversion, to
  debugging and deployment to a wide variety of platforms.
- **Performance:** Providing end users with a seamless and
  high-performance experience due to a lightweight runtime and
  utilizing full hardware capabilities such as CPUs, NPUs, and DSPs.

ExecuTorch provides support for:

* **Strong Model Support** LLMs (Large Language Models),
  CV (Computer Vision), ASR (Automatic Speech Recognition), TTS (Text To Speech)
* **All Major Platforms** Android, Mac, Linux, Windows
* **Rich Acceleration Support** Apple, Arm, Cadence, MediaTek, NXP, OpenVino, Qualcomm, Vulkan, XNNPACK

### Documentation Navigation
#### Introduction
- [Overview](intro-overview)
- [How it Works](intro-how-it-works)
- [Getting Started with Architecture](getting-started-architecture)
- [Concepts](concepts)
#### Usage
- [Getting Started](getting-started)
- [Using Executorch Export](using-executorch-export)
- [Using Executorch on Android](using-executorch-android)
- [Using Executorch on iOS](using-executorch-ios)
- [Using Executorch with C++](using-executorch-cpp)
- [Runtime Integration](using-executorch-runtime-integration)
- [Troubleshooting](using-executorch-troubleshooting)
- [Building from Source](using-executorch-building-from-source)
- [Quantization](quantization-overview)
- [FAQs](using-executorch-faqs)
#### Examples
- [Android Demo Apps](https://github.com/meta-pytorch/executorch-examples/tree/main/dl3/android/DeepLabV3Demo#executorch-android-demo-app)
- [iOS Demo Apps](https://github.com/meta-pytorch/executorch-examples/tree/main/mv3/apple/ExecuTorchDemo)
- [Hugging Face Models](https://github.com/huggingface/optimum-executorch/blob/main/README.md)
#### Backends
- [Overview](backends-overview)
- [XNNPACK](backends-xnnpack)
- [Core ML](backends-coreml)
- [MPS](backends-mps)
- [Vulkan](backends-vulkan)
- [ARM Ethos-U](backends-arm-ethos-u)
- [ARM VGF](backends-arm-vgf)
- [Qualcomm](backends-qualcomm)
- [MediaTek](backends-mediatek)
- [Cadence](backends-cadence)
- [OpenVINO](build-run-openvino)
- [NXP](backend-nxp)
#### Developer Tools
- [Overview](devtools-overview)
- [Bundled IO](bundled-io)
- [ETRecord](etrecord)
- [ETDump](etdump)
- [Runtime Profiling](runtime-profiling)
- [Model Debugging](model-debugging)
- [Model Inspector](model-inspector)
- [Memory Planning Inspection](memory-planning-inspection)
- [Delegate Debugging](delegate-debugging)
- [Tutorial](devtools-tutorial)
#### Runtime
- [Overview](runtime-overview)
- [Extension Module](extension-module)
- [Extension Tensor](extension-tensor)
- [Detailed C++ Runtime APIs Tutorial](running-a-model-cpp-tutorial)
- [Backend Delegate Implementation and Linking](runtime-backend-delegate-implementation-and-linking)
- [Platform Abstraction Layer](runtime-platform-abstraction-layer)
#### Portable C++ Programming
- [PTE File Format](pte-file-format)
- [PTD File Format](ptd-file-format)
#### API Reference
- [Export to Executorch API Reference](export-to-executorch-api-reference)
- [Executorch Runtime API Reference](executorch-runtime-api-reference)
- [Runtime Python API Reference](runtime-python-api-reference)
- [API Life Cycle](api-life-cycle)
- [Javadoc](https://pytorch.org/executorch/main/javadoc/)
#### Kernel Library
- [Overview](kernel-library-overview)
- [Custom ATen Kernel](kernel-library-custom-aten-kernel)
- [Selective Build](kernel-library-selective-build)
#### Working with LLMs
- [Getting Started](llm/getting-started.md)
- [Exporting LLMs](llm/export-llm.md)
- [Exporting custom LLMs](llm/export-custom-llm.md)
- [Running with C++](llm/run-with-c-plus-plus.md)
- [Running on Android (XNNPack)](https://github.com/meta-pytorch/executorch-examples/tree/main/llm/android)
- [Running on Android (QNN)](llm/build-run-llama3-qualcomm-ai-engine-direct-backend.md)
- [Running on iOS](llm/run-on-ios.md)
#### Backend Development
- [Delegates Integration](backend-delegates-integration)
- [XNNPACK Reference](backend-delegates-xnnpack-reference)
- [Dependencies](backend-delegates-dependencies)
- [Compiler Delegate and Partitioner](compiler-delegate-and-partitioner)
- [Debug Backend Delegate](debug-backend-delegate)
#### IR Specification
- [EXIR](ir-exir)
- [Ops Set Definition](ir-ops-set-definition)
#### Compiler Entry Points
- [Backend Dialect](compiler-backend-dialect)
- [Custom Compiler Passes](compiler-custom-compiler-passes)
- [Memory Planning](compiler-memory-planning)
#### Contributing
- [Contributing](contributing)

```{toctree}
:glob:
:maxdepth: 1
:hidden:

intro
usage
examples
backends
developer-tools
runtime
api
quantization
kernel-library
llm/working-with-llms
backend-development
ir-specification
compiler-entry-points
contributing
```
