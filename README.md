<div align="center">
  <img src="docs/source/_static/img/et-logo.png" alt="ExecuTorch logo mark" width="200">
  <h1>ExecuTorch</h1>
  <p><strong>On-device AI inference powered by PyTorch</strong></p>
</div>

<div align="center">
  <a href="https://pypi.org/project/executorch/"><img src="https://img.shields.io/pypi/v/executorch?style=for-the-badge&color=blue" alt="PyPI - Version"></a>
  <a href="https://github.com/pytorch/executorch/graphs/contributors"><img src="https://img.shields.io/github/contributors/pytorch/executorch?style=for-the-badge&color=blue" alt="GitHub - Contributors"></a>
  <a href="https://github.com/pytorch/executorch/stargazers"><img src="https://img.shields.io/github/stars/pytorch/executorch?style=for-the-badge&color=blue" alt="GitHub - Stars"></a>
  <a href="https://discord.gg/Dh43CKSAdc"><img src="https://img.shields.io/badge/Discord-Join%20Us-blue?logo=discord&logoColor=white&style=for-the-badge" alt="Discord - Chat with Us"></a>
  <a href="https://docs.pytorch.org/executorch/main/index.html"><img src="https://img.shields.io/badge/Documentation-blue?logo=googledocs&logoColor=white&style=for-the-badge" alt="Documentation"></a>
</div>

**ExecuTorch** is PyTorch's unified solution for deploying AI models on-device—from smartphones to microcontrollers—built for privacy, performance, and portability. It powers Meta's on-device AI across **Instagram, WhatsApp, Quest 3, Ray-Ban Meta Smart Glasses**, and [more](https://docs.pytorch.org/executorch/main/success-stories.html).

Deploy **LLMs, vision, speech, and multimodal models** with the same PyTorch APIs you already know—accelerating research to production with seamless model export, optimization, and deployment. No manual C++ rewrites. No format conversions. No vendor lock-in.

<details>
  <summary><strong>📘 Table of Contents</strong></summary>

- [Why ExecuTorch?](#why-executorch)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Export and Deploy in 3 Steps](#export-and-deploy-in-3-steps)
  - [Run on Device](#run-on-device)
  - [LLM Example: Llama](#llm-example-llama)
- [Platform & Hardware Support](#platform--hardware-support)
- [Production Deployments](#production-deployments)
- [Examples & Models](#examples--models)
- [Key Features](#key-features)
- [Documentation](#documentation)
- [Community & Contributing](#community--contributing)
- [License](#license)

</details>

## Why ExecuTorch?

- **🔒 Native PyTorch Export** — Direct export from PyTorch. No .onnx, .tflite, or intermediate format conversions. Preserve model semantics.
- **⚡ Production-Proven** — Powers billions of users at [Meta with real-time on-device inference](https://engineering.fb.com/2025/07/28/android/executorch-on-device-ml-meta-family-of-apps/).
- **💾 Tiny Runtime** — 50KB base footprint. Runs on microcontrollers to high-end smartphones.
- **🚀 [12+ Hardware Backends](https://docs.pytorch.org/executorch/main/backends-overview.html)** — Open-source acceleration for Apple, Qualcomm, ARM, MediaTek, Vulkan, and more.
- **🎯 One Export, Multiple Backends** — Switch hardware targets with a single line change. Deploy the same model everywhere.

## How It Works

ExecuTorch uses **ahead-of-time (AOT) compilation** to prepare PyTorch models for edge deployment:

1. **🧩 Export** — Capture your PyTorch model graph with `torch.export()`
2. **⚙️ Compile** — Quantize, optimize, and partition to hardware backends → `.pte`
3. **🚀 Execute** — Load `.pte` on-device via lightweight C++ runtime

Models use a standardized [Core ATen operator set](https://docs.pytorch.org/executorch/main/compiler-ir-advanced.html#intermediate-representation). [Partitioners](https://docs.pytorch.org/executorch/main/compiler-delegate-and-partitioner.html) delegate subgraphs to specialized hardware (NPU/GPU) with CPU fallback.

Learn more: [How ExecuTorch Works](https://docs.pytorch.org/executorch/main/intro-how-it-works.html) • [Architecture Guide](https://docs.pytorch.org/executorch/main/getting-started-architecture.html)

## Quick Start

### Installation

```bash
pip install executorch
```

For platform-specific setup (Android, iOS, embedded systems), see the [Quick Start](https://docs.pytorch.org/executorch/main/quick-start-section.html) documentation for additional info.

### Export and Deploy in 3 Steps

```python
import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# 1. Export your PyTorch model
model = MyModel().eval()
example_inputs = (torch.randn(1, 3, 224, 224),)
exported_program = torch.export.export(model, example_inputs)

# 2. Optimize for target hardware (switch backends with one line)
program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()]  # CPU | CoreMLPartitioner() for iOS | QnnPartitioner() for Qualcomm
).to_executorch()

# 3. Save for deployment
with open("model.pte", "wb") as f:
    f.write(program.buffer)

# Test locally via ExecuTorch runtime's pybind API (optional)
from executorch.runtime import Runtime
runtime = Runtime.get()
method = runtime.load_program("model.pte").load_method("forward")
outputs = method.execute([torch.randn(1, 3, 224, 224)])
```

### Run on Device

**[C++](https://docs.pytorch.org/executorch/main/using-executorch-cpp.html)**
```cpp
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

Module module("model.pte");
auto tensor = make_tensor_ptr({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
auto outputs = module.forward(tensor);
```

**[Swift (iOS)](https://docs.pytorch.org/executorch/main/ios-section.html)**
```swift
import ExecuTorch

let module = Module(filePath: "model.pte")
let input = Tensor<Float>([1.0, 2.0, 3.0, 4.0], shape: [2, 2])
let outputs = try module.forward(input)
```

**[Kotlin (Android)](https://docs.pytorch.org/executorch/main/android-section.html)**
```kotlin
val module = Module.load("model.pte")
val inputTensor = Tensor.fromBlob(floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f), longArrayOf(2, 2))
val outputs = module.forward(EValue.from(inputTensor))
```

### LLM Example: Llama

Export Llama models using the [`export_llm`](https://docs.pytorch.org/executorch/main/llm/export-llm.html) script or [Optimum-ExecuTorch](https://github.com/huggingface/optimum-executorch):

```bash
# Using export_llm
python -m executorch.extension.llm.export.export_llm --model llama3_2 --output llama.pte

# Using Optimum-ExecuTorch
optimum-cli export executorch \
  --model meta-llama/Llama-3.2-1B \
  --task text-generation \
  --recipe xnnpack \
  --output_dir llama_model
```

Run on-device with the LLM runner API:

**[C++](https://docs.pytorch.org/executorch/main/llm/run-with-c-plus-plus.html)**
```cpp
#include <executorch/extension/llm/runner/text_llm_runner.h>

auto runner = create_llama_runner("llama.pte", "tiktoken.bin");
executorch::extension::llm::GenerationConfig config{
    .seq_len = 128, .temperature = 0.8f};
runner->generate("Hello, how are you?", config);
```

**[Swift (iOS)](https://docs.pytorch.org/executorch/main/llm/run-on-ios.html)**
```swift
import ExecuTorchLLM

let runner = TextRunner(modelPath: "llama.pte", tokenizerPath: "tiktoken.bin")
try runner.generate("Hello, how are you?", Config {
    $0.sequenceLength = 128
}) { token in
    print(token, terminator: "")
}
```

**Kotlin (Android)** — [API Docs](https://docs.pytorch.org/executorch/main/javadoc/org/pytorch/executorch/extension/llm/package-summary.html) • [Demo App](https://github.com/meta-pytorch/executorch-examples/tree/main/llm/android/LlamaDemo)
```kotlin
val llmModule = LlmModule("llama.pte", "tiktoken.bin", 0.8f)
llmModule.load()
llmModule.generate("Hello, how are you?", 128, object : LlmCallback {
    override fun onResult(result: String) { print(result) }
    override fun onStats(stats: String) { }
})
```

For multimodal models (vision, audio), use the [MultiModal runner API](extension/llm/runner) which extends the LLM runner to handle image and audio inputs alongside text. See [Llava](examples/models/llava/README.md) and [Voxtral](examples/models/voxtral/README.md) examples.

See [examples/models/llama](examples/models/llama/README.md) for complete workflow including quantization, mobile deployment, and advanced options.

**Next Steps:**
- 📖 [Step-by-step tutorial](https://docs.pytorch.org/executorch/main/getting-started.html) — Complete walkthrough for your first model
- ⚡ [Colab notebook](https://colab.research.google.com/drive/1qpxrXC3YdJQzly3mRg-4ayYiOjC6rue3?usp=sharing) — Try ExecuTorch instantly in your browser
- 🤖 [Deploy Llama models](examples/models/llama/README.md) — LLM workflow with quantization and mobile demos

## Platform & Hardware Support

| **Platform**     | **Supported Backends**                                   |
|------------------|----------------------------------------------------------|
| Android          | XNNPACK, Vulkan, Qualcomm, MediaTek, Samsung Exynos      |
| iOS              | XNNPACK, MPS, CoreML (Neural Engine)                     |
| Linux / Windows  | XNNPACK, OpenVINO, CUDA *(experimental)*                 |
| macOS            | XNNPACK, MPS, Metal *(experimental)*                     |
| Embedded / MCU   | XNNPACK, ARM Ethos-U, NXP, Cadence DSP                   |

See [Backend Documentation](https://docs.pytorch.org/executorch/main/backends-overview.html) for detailed hardware requirements and optimization guides.

## Production Deployments

ExecuTorch powers on-device AI at scale across Meta's family of apps, VR/AR devices, and partner deployments. [View success stories →](https://docs.pytorch.org/executorch/main/success-stories.html)

## Examples & Models

**LLMs:** [Llama 3.2/3.1/3](examples/models/llama/README.md), [Qwen 3](examples/models/qwen3/README.md), [Phi-4-mini](examples/models/phi_4_mini/README.md), [LiquidAI LFM2](examples/models/lfm2/README.md)

**Multimodal:** [Llava](examples/models/llava/README.md) (vision-language), [Voxtral](examples/models/voxtral/README.md) (audio-language), [Gemma](examples/models/gemma3) (vision-language)

**Vision/Speech:** [MobileNetV2](https://github.com/meta-pytorch/executorch-examples/tree/main/mv2), [DeepLabV3](https://github.com/meta-pytorch/executorch-examples/tree/main/dl3), [Whisper](https://github.com/meta-pytorch/executorch-examples/tree/main/whisper/android/WhisperApp)

**Resources:** [`examples/`](examples/) directory • [executorch-examples](https://github.com/meta-pytorch/executorch-examples) out-of-tree demos • [Optimum-ExecuTorch](https://github.com/huggingface/optimum-executorch) for HuggingFace models

## Key Features

ExecuTorch provides advanced capabilities for production deployment:

- **Quantization** — Built-in support via [torchao](https://docs.pytorch.org/ao) for 8-bit, 4-bit, and dynamic quantization
- **Memory Planning** — Optimize memory usage with ahead-of-time allocation strategies
- **Developer Tools** — ETDump profiler, ETRecord inspector, and model debugger
- **Selective Build** — Strip unused operators to minimize binary size
- **Custom Operators** — Extend with domain-specific kernels
- **Dynamic Shapes** — Support variable input sizes with bounded ranges

See [Advanced Topics](https://docs.pytorch.org/executorch/main/advanced-topics-section.html) for quantization techniques, custom backends, and compiler passes.

## Documentation

- [**Documentation Home**](https://docs.pytorch.org/executorch/main/index.html) — Complete guides and tutorials
- [**API Reference**](https://docs.pytorch.org/executorch/main/api-section.html) — Python, C++, Java/Kotlin APIs
- [**Backend Integration**](https://docs.pytorch.org/executorch/main/backend-delegates-integration.html) — Build custom hardware backends
- [**Troubleshooting**](https://docs.pytorch.org/executorch/main/support-section.html) — Common issues and solutions

## Community & Contributing

We welcome contributions from the community!

- 💬 [**GitHub Discussions**](https://github.com/pytorch/executorch/discussions) — Ask questions and share ideas
- 🎮 [**Discord**](https://discord.gg/Dh43CKSAdc) — Chat with the team and community
- 🐛 [**Issues**](https://github.com/pytorch/executorch/issues) — Report bugs or request features
- 🤝 [**Contributing Guide**](CONTRIBUTING.md) — Guidelines and codebase structure

## License

ExecuTorch is BSD licensed, as found in the [LICENSE](LICENSE) file.

<br><br>

---

<div align="center">
  <p><strong>Part of the PyTorch ecosystem</strong></p>
  <p>
    <a href="https://github.com/pytorch/executorch">GitHub</a> •
    <a href="https://docs.pytorch.org/executorch">Documentation</a>
  </p>
</div>
