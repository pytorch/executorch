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
* **Rich Acceleration Support** Apple, Arm, Cadence, MediaTek,
  Qualcomm, Vulkan, XNNPACK

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

demo-apps-android.md
demo-apps-ios.md
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

Llama <llm/llama>
Llama on Android <llm/llama-demo-android>
Llama on iOS <llm/llama-demo-ios>
Llama on Android via Qualcomm backend <llm/build-run-llama3-qualcomm-ai-engine-direct-backend>
Intro to LLMs in Executorch <llm/getting-started>
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
