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
- [Running on Android (XNNPack)](llm/llama-demo-android.md)
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
:caption: Introduction
:hidden:

intro-overview
intro-how-it-works
getting-started-architecture
concepts
```

```{toctree}
:glob:
:maxdepth: 1
:caption: Usage
:hidden:

getting-started
using-executorch-export
using-executorch-android
using-executorch-ios
using-executorch-cpp
using-executorch-runtime-integration
using-executorch-troubleshooting
using-executorch-building-from-source
using-executorch-faqs
```

```{toctree}
:glob:
:maxdepth: 1
:caption: Examples
:hidden:

Building an ExecuTorch Android Demo App <https://github.com/pytorch-labs/executorch-examples/tree/main/dl3/android/DeepLabV3Demo#executorch-android-demo-app>
Building an ExecuTorch iOS Demo App <https://github.com/meta-pytorch/executorch-examples/tree/main/mv3/apple/ExecuTorchDemo>
tutorial-arm.md
```

```{toctree}
:glob:
:maxdepth: 1
:caption: Backends
:hidden:

backends-overview
backends-xnnpack
backends-coreml
backends-mps
backends-vulkan
backends-arm-ethos-u
backends-qualcomm
backends-mediatek
backends-cadence
OpenVINO Backend <build-run-openvino>
backends-nxp
```

```{toctree}
:glob:
:maxdepth: 1
:caption: Developer Tools
:hidden:

devtools-overview
bundled-io
etrecord
etdump
runtime-profiling
model-debugging
model-inspector
memory-planning-inspection
delegate-debugging
devtools-tutorial
```

```{toctree}
:glob:
:maxdepth: 1
:caption: Runtime
:hidden:

runtime-overview
extension-module
extension-tensor
running-a-model-cpp-tutorial
runtime-backend-delegate-implementation-and-linking
runtime-platform-abstraction-layer
portable-cpp-programming
pte-file-format
ptd-file-format
```

```{toctree}
:glob:
:maxdepth: 1
:caption: API Reference
:hidden:

export-to-executorch-api-reference
executorch-runtime-api-reference
runtime-python-api-reference
api-life-cycle
Javadoc <https://pytorch.org/executorch/main/javadoc/>
```

```{toctree}
:glob:
:maxdepth: 1
:caption: Quantization
:hidden:

quantization-overview
```

```{toctree}
:glob:
:maxdepth: 1
:caption: Kernel Library
:hidden:

kernel-library-overview
kernel-library-custom-aten-kernel
kernel-library-selective-build
```

```{toctree}
:glob:
:maxdepth: 2
:caption: Working with LLMs
:hidden:

Getting Started <llm/getting-started>
Exporting LLMs with export_llm <llm/export-llm>
Exporting custom LLMs <llm/export-custom-llm>
Running with C++ <llm/run-with-c-plus-plus>
Running on Android <XNNPack> <llm/llama-demo-android>
Running on Android <QNN> <llm/build-run-llama3-qualcomm-ai-engine-direct-backend>
Running on iOS <llm/run-on-ios>
```

```{toctree}
:glob:
:maxdepth: 1
:caption: Backend Development
:hidden:

backend-delegates-integration
backend-delegates-xnnpack-reference
backend-delegates-dependencies
compiler-delegate-and-partitioner
debug-backend-delegate
```

```{toctree}
:glob:
:maxdepth: 1
:caption: IR Specification
:hidden:

ir-exir
ir-ops-set-definition
```

```{toctree}
:glob:
:maxdepth: 1
:caption: Compiler Entry Points
:hidden:

compiler-backend-dialect
compiler-custom-compiler-passes
compiler-memory-planning
```

```{toctree}
:glob:
:maxdepth: 1
:caption: Contributing
:hidden:

contributing
```
