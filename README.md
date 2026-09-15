<div align="center">
  <img src="docs/source/_static/img/et-logo.png" alt="ExecuTorch logo mark" width="200">
  <h1>ExecuTorch</h1>
  <p><strong>PyTorch-native AI inference from phones and laptops to microcontrollers</strong></p>
</div>

<div align="center">
  <a href="https://pypi.org/project/executorch/"><img src="https://img.shields.io/pypi/v/executorch?style=for-the-badge&color=blue" alt="PyPI - Version"></a>
  <a href="https://github.com/pytorch/executorch/graphs/contributors"><img src="https://img.shields.io/github/contributors/pytorch/executorch?style=for-the-badge&color=blue" alt="GitHub - Contributors"></a>
  <a href="https://github.com/pytorch/executorch/stargazers"><!-- @lint-ignore GitHub serves 404 on /stargazers to non-browser clients --><img src="https://img.shields.io/github/stars/pytorch/executorch?style=for-the-badge&color=blue" alt="GitHub - Stars"></a>
  <a href="https://discord.gg/Dh43CKSAdc"><img src="https://img.shields.io/badge/Discord-Join%20Us-blue?logo=discord&logoColor=white&style=for-the-badge" alt="Discord - Chat with Us"></a>
  <a href="https://docs.pytorch.org/executorch/main/index.html"><img src="https://img.shields.io/badge/Documentation-blue?logo=googledocs&logoColor=white&style=for-the-badge" alt="Documentation"></a>
</div>

**ExecuTorch** is PyTorch's open source stack for running AI locally on phones,
wearables, laptops, browsers, embedded systems, and microcontrollers. Start with
a PyTorch model, capture it with `torch.export`, optimize it for target hardware,
and run it through C++, Python, Swift/Objective-C, Kotlin/Java, or JavaScript
APIs.

ExecuTorch powers on-device experiences across **Instagram, WhatsApp, Facebook,
and Messenger**, serving billions of people. It also runs AI features on
**Meta Quest and Ray-Ban Meta devices**.
[See where ExecuTorch is shipping.](https://docs.pytorch.org/executorch/main/success-stories.html)

> [!IMPORTANT]
> **Release channels:** use the
> [latest release](https://github.com/pytorch/executorch/releases/latest) with
> the [stable documentation](https://docs.pytorch.org/executorch/stable/).
> This README tracks `main`; features labeled **Main / nightly** may change
> before release. Use the [main documentation](https://docs.pytorch.org/executorch/main/)
> with a source checkout or nightly package.

## Built for current edge workloads

| Workload | Representative capabilities | Start here |
|---|---|---|
| **Local LLM and agent building blocks** | Quantized text models, long context, tool calling, speculative decoding, multi-session execution, and an experimental OpenAI-compatible local server | [Muse Glimmer](examples/models/muse-glimmer/README.md) · [LLM guide](https://docs.pytorch.org/executorch/main/llm/working-with-llms.html) · [LLM server](examples/llm_server/README.md) |
| **Voice** | Streaming and offline speech recognition, speech synthesis, voice activity detection, and speaker diarization | [Voxtral Realtime](examples/models/voxtral_realtime/README.md) · [Parakeet](examples/models/parakeet/README.md) · [Sortformer diarization](examples/models/sortformer/README.md) · [Voxtral TTS](examples/models/voxtral_tts/README.md) |
| **Multimodal** | Text, image, and audio runners; mobile vision-language models | [Gemma 4](examples/models/gemma4/README.md) · [Multimodal runner](extension/llm/runner/README.md) |
| **Computer vision** | Image classification, object detection, semantic segmentation, and promptable segmentation | [MobileNet V2](https://docs.pytorch.org/executorch/main/getting-started.html#preparing-the-model) · [YOLO26](examples/models/yolo26/README.md) · [DeepLabV3](https://github.com/meta-pytorch/executorch-examples/tree/main/dl3/android/DeepLabV3Demo) · [EfficientSAM](examples/models/efficient_sam/README.md) |
| **Embedded AI** | Cortex-M CPU kernels, Ethos-U and NXP NPUs, Cadence DSPs, Zephyr, Arduino, and Raspberry Pi Pico workflows | [Embedded guide](https://docs.pytorch.org/executorch/main/embedded-section.html) · [Cortex-M](https://docs.pytorch.org/executorch/main/backends/arm-cortex-m/arm-cortex-m-overview.html) · [Arduino](examples/arduino/README.md) |

These are starting points, not a compatibility list. A model does not need to
appear here to run with ExecuTorch: use the
[standard export guide](https://docs.pytorch.org/executorch/main/using-executorch-export.html)
or adapt the closest [model example](examples/models/). Deployment depends on
`torch.export` capture plus operator, runtime, and selected-backend coverage;
validate numerical accuracy, memory use, and performance on the target. For a
new language-model architecture, see the
[custom LLM guide](https://docs.pytorch.org/executorch/main/llm/export-custom-llm.html).

## Why ExecuTorch

- **PyTorch-native workflow:** work directly from `torch.export`, with PyTorch
  program metadata and source mappings available to export and debugging tools.
- **Partitioned hardware acceleration:** delegate supported graph regions to
  CPU, GPU, NPU, or DSP backends while retaining portable CPU kernels for
  fallback.
- **One source model, explicit targets:** reuse the PyTorch model and export
  flow, while producing a backend-specific `.pte` for each target that needs
  hardware specialization.
- **A runtime you can right-size:** link only the operators, kernels, and
  delegates a deployment needs;
  [selective build](https://docs.pytorch.org/executorch/main/kernel-library-selective-build.html)
  keeps only the required operator kernels.
- **Inspect, profile, and extend:** use
  [ETDump](https://docs.pytorch.org/executorch/main/etdump.html),
  [ETRecord](https://docs.pytorch.org/executorch/main/etrecord.html), and
  [numeric debugging](https://docs.pytorch.org/executorch/main/model-debugging.html),
  or add custom operators and backends.
- **Versioned deployment contract:** a `.pte` created with stable APIs is
  guaranteed to load and execute for at least one following non-patch runtime
  release; see the
  [runtime compatibility](runtime/COMPATIBILITY.md) and
  [API lifecycle](https://docs.pytorch.org/executorch/main/api-life-cycle.html) policies.

## Install

Install the latest stable Python package in a Python 3.10–3.14 environment:

```bash
pip install executorch
```

Install a nightly built from `main` to use the newest features:

```bash
pip install --upgrade --pre executorch torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

`torch` is explicit because nightly ExecuTorch wheels do not declare it as a
dependency. This command installs a CPU-only PyTorch nightly. On Linux with an
NVIDIA GPU, select the nightly command matching your CUDA version in the
[PyTorch installation selector](https://pytorch.org/get-started/locally/) and
use its `nightly/cu*` index instead; otherwise, pip can replace a CUDA-enabled
PyTorch installation with the CPU build. ExecuTorch CUDA wheels are published
for Linux only.

Backend export tools can require optional dependencies. For example, use
`pip install 'executorch[ethos_u]'` for Ethos-U AOT export. Embedded
toolchains, simulators, and target runtimes are installed separately.

For platform-specific setup (Android, iOS, embedded systems), see the
[Quick Start](https://docs.pytorch.org/executorch/main/quick-start-section.html)
documentation for additional information.

The prebuilt Python wheel is published for Linux x86-64, Linux AArch64, macOS
arm64, and Windows x86-64. Build from source for other hosts or custom
configurations. Native integration is also available through:

- [Prebuilt C++ libraries, headers, and CMake package in current main/nightly Linux and macOS wheels](https://docs.pytorch.org/executorch/main/using-executorch-cpp.html#using-the-prebuilt-libraries-from-the-pip-package)
- [Android AAR from Maven Central](https://docs.pytorch.org/executorch/main/using-executorch-android.html)
- [Apple frameworks through Swift Package Manager](https://docs.pytorch.org/executorch/main/using-executorch-ios.html)
- [Source builds and cross-compilation](https://docs.pytorch.org/executorch/main/using-executorch-building-from-source.html)

## Five-minute export and run

This complete example, from the
[quick-start pathway](https://docs.pytorch.org/executorch/main/pathway-quickstart.html),
exports a small model and lowers it to XNNPACK:

```python
import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime

class Add(torch.nn.Module):
    def forward(self, x, y):
        return x + y

model = Add().eval()
sample_inputs = (torch.ones(1), torch.ones(1))

et_program = to_edge_transform_and_lower(
    torch.export.export(model, sample_inputs),
    partitioner=[XnnpackPartitioner()]
).to_executorch()

with open("add.pte", "wb") as f:
    f.write(et_program.buffer)

runtime = Runtime.get()
runtime_program = runtime.load_program("add.pte")
method = runtime_program.load_method("forward")
output = method.execute(sample_inputs)[0]

torch.testing.assert_close(output, model(*sample_inputs))
print("Output:", output)
```

Expected output: `Output: tensor([2.])`. This verifies export, XNNPACK lowering,
serialization, loading, and execution on the host; measure performance again on
the target device.

The resulting `add.pte` is specialized for the selected backend. Targeting
Core ML, Qualcomm, or another accelerator requires that backend's dependencies,
configuration, and a separate export, not a blind partitioner substitution.
Continue with the [Python runtime](https://docs.pytorch.org/executorch/main/getting-started.html#testing-the-model),
[C++ Module API](https://docs.pytorch.org/executorch/main/using-executorch-cpp.html),
[Android](https://docs.pytorch.org/executorch/main/android-section.html), or
[iOS](https://docs.pytorch.org/executorch/main/ios-section.html).

### Choose an export path

| Starting point | Recommended path | Scope |
|---|---|---|
| PyTorch `nn.Module` | [Standard export and lowering](https://docs.pytorch.org/executorch/main/using-executorch-export.html) | Full backend choice; some models need decompositions or custom operators |
| Hugging Face `PreTrainedModel` | [Transformers ExecuTorch exporter](https://huggingface.co/docs/transformers/en/exporters) | Experimental programmatic XNNPACK/CUDA export; generation components require application orchestration |
| Optimized text or multimodal generation | [`export_llm`](https://docs.pytorch.org/executorch/main/llm/export-llm.html) or [Optimum ExecuTorch](https://huggingface.co/docs/optimum-executorch/) | Tested recipes, quantization, tokenizers, and runner-specific metadata |

Already have a compatible `.pte`? Browse the
[ExecuTorch Community](https://huggingface.co/executorch-community),
[Arm AI model catalog filtered to ExecuTorch](https://developer.arm.com/ai/models?runtime=executorch),
or, where available, the linked model pages above. Match the model
configuration, precision, backend, and runtime; a `.pte` built for one hardware
delegate is not a universal model file.

## Runtime and LLM APIs

Use C++, Python, Java/Kotlin, Swift/Objective-C, or JavaScript/WebAssembly for
general `.pte` execution. Higher-level text and multimodal runners are available
for C++, Python, Android, and Apple platforms.

[Compare the runtime and LLM APIs](https://docs.pytorch.org/executorch/main/api-section.html)
for language-specific entry points and API maturity.

## Platforms and hardware backends

Choose a backend based on target hardware, operator coverage, and deployment
constraints. The linked guides document setup, supported hardware, and known
limitations. These are representative paths; the
[backend documentation](https://docs.pytorch.org/executorch/main/backends-overview.html)
is authoritative. See also the [Desktop guide](desktop/README.md).

| Target | Backends and integrations |
|---|---|
| Android | [XNNPACK](https://docs.pytorch.org/executorch/main/backends/xnnpack/xnnpack-overview.html) CPU; [Vulkan](https://docs.pytorch.org/executorch/main/backends/vulkan/vulkan-overview.html) GPU; [Qualcomm](https://docs.pytorch.org/executorch/main/backends-qualcomm.html), [MediaTek](https://docs.pytorch.org/executorch/main/backends-mediatek.html), [Arm VGF](https://docs.pytorch.org/executorch/main/backends/arm-vgf/arm-vgf-overview.html), and [Samsung Exynos](https://docs.pytorch.org/executorch/main/backends/samsung/samsung-overview.html) accelerators |
| iOS / iPadOS | [XNNPACK](https://docs.pytorch.org/executorch/main/backends/xnnpack/xnnpack-overview.html) CPU; [Core ML](https://docs.pytorch.org/executorch/main/backends/coreml/coreml-overview.html); [MLX](https://docs.pytorch.org/executorch/main/backends/mlx/mlx-overview.html) on physical devices *(experimental)* |
| macOS | [XNNPACK](https://docs.pytorch.org/executorch/main/backends/xnnpack/xnnpack-overview.html); [Core ML](https://docs.pytorch.org/executorch/main/backends/coreml/coreml-overview.html); experimental [MLX](https://docs.pytorch.org/executorch/main/backends/mlx/mlx-overview.html), [Metal/AOTInductor](backends/apple/metal/README.md), and [WebGPU](https://docs.pytorch.org/executorch/main/backends/webgpu/webgpu-overview.html) paths |
| Linux | [XNNPACK](https://docs.pytorch.org/executorch/main/backends/xnnpack/xnnpack-overview.html); [OpenVINO](https://docs.pytorch.org/executorch/main/build-run-openvino.html); experimental [CUDA/AOTInductor](https://docs.pytorch.org/executorch/main/backends/cuda/cuda-overview.html), [Vulkan](https://docs.pytorch.org/executorch/main/backends/vulkan/vulkan-overview.html), and [WebGPU](https://docs.pytorch.org/executorch/main/backends/webgpu/webgpu-overview.html) desktop paths |
| Windows | [XNNPACK](https://docs.pytorch.org/executorch/main/backends/xnnpack/xnnpack-overview.html); experimental [CUDA/AOTInductor](https://docs.pytorch.org/executorch/main/backends/cuda/cuda-overview.html) and [Vulkan](https://docs.pytorch.org/executorch/main/backends/vulkan/vulkan-overview.html) desktop paths |
| Browser / WebAssembly | [Portable WebAssembly runtime](extension/wasm/README.md) and [WebGPU](https://docs.pytorch.org/executorch/main/backends/webgpu/webgpu-overview.html) *(both experimental)* |
| Embedded / MCU | [Arm Cortex-M with CMSIS-NN](https://docs.pytorch.org/executorch/main/backends/arm-cortex-m/arm-cortex-m-overview.html) *(beta)*; [Arm Ethos-U](https://docs.pytorch.org/executorch/main/backends/arm-ethos-u/arm-ethos-u-overview.html); [NXP eIQ Neutron](https://docs.pytorch.org/executorch/main/backends/nxp/nxp-overview.html); [Cadence DSP](https://docs.pytorch.org/executorch/main/backends-cadence.html); [Zephyr](zephyr/README.md) and [Arduino](examples/arduino/README.md) integrations |

## Documentation

- [Get started](https://docs.pytorch.org/executorch/main/getting-started.html)
- [Choose a platform](https://docs.pytorch.org/executorch/main/edge-platforms-section.html)
- [Choose a backend](https://docs.pytorch.org/executorch/main/backends-overview.html)
- [Export and lower a model](https://docs.pytorch.org/executorch/main/using-executorch-export.html)
- [Work with LLMs](https://docs.pytorch.org/executorch/main/llm/working-with-llms.html)
- [Profile and debug](https://docs.pytorch.org/executorch/main/tools-section.html)
- [Runtime and LLM APIs](https://docs.pytorch.org/executorch/main/api-section.html)
- [PTE/runtime compatibility policy](runtime/COMPATIBILITY.md)
- [LLM benchmark dashboard](https://hud.pytorch.org/benchmark/llms?repoName=pytorch%2Fexecutorch) <!-- @lint-ignore -->
- [Troubleshooting](https://docs.pytorch.org/executorch/main/support-section.html)

## Community and contributing

- [GitHub Discussions](https://github.com/pytorch/executorch/discussions): ask questions and share ideas
- [Discord](https://discord.gg/Dh43CKSAdc): chat with maintainers and the community
- [Issues](https://github.com/pytorch/executorch/issues): report bugs or request features
- [Contributing guide](CONTRIBUTING.md): development setup, testing, and review guidelines

## Citing ExecuTorch

Read the [MLSys 2026 paper (PDF)](https://proceedings.mlsys.org/paper_files/paper/2026/file/236f915dd02af4f11927f67330b21d4b-Paper-Conference.pdf).

If you use ExecuTorch in research, please cite:

```bibtex
@article{executorch2026,
    title={{ExecuTorch} - A Unified {PyTorch} Solution to Run {AI} Models On-Device},
    author={Nachin, Mergen and Desai, Digant and Jia, Sicheng Stephen and Lai, Chen and Liu, Mengwei and Szwejbka, Jacob and Alvarez, Raziel and Ascani, RJ and Bort, Dave and Candales, Manuel and
  others},
    journal={arXiv preprint arXiv:2605.08195},
    url={https://arxiv.org/abs/2605.08195},
    year={2026}
  }
```

## License

ExecuTorch is BSD licensed. See [LICENSE](LICENSE).

---

<div align="center">
  <p><strong>Part of the PyTorch ecosystem</strong></p>
  <p>
    <a href="https://github.com/pytorch/executorch">GitHub</a> •
    <a href="https://docs.pytorch.org/executorch">Documentation</a>
  </p>
</div>
