<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<div align="center">
  <img src="docs/source/_static/img/et-logo.png" alt="ExecuTorch logo mark" width="200">
  <h1>ExecuTorch</h1>
  <p><strong>从手机、笔记本电脑到微控制器，原生支持 PyTorch 的端侧 AI 推理引擎</strong></p>
</div>

<div align="center">
  <a href="https://pypi.org/project/executorch/"><img src="https://img.shields.io/pypi/v/executorch?style=for-the-badge&color=blue" alt="PyPI - Version"></a>
  <a href="https://github.com/pytorch/executorch/graphs/contributors"><img src="https://img.shields.io/github/contributors/pytorch/executorch?style=for-the-badge&color=blue" alt="GitHub - Contributors"></a>
  <a href="https://github.com/pytorch/executorch/stargazers"><!-- @lint-ignore GitHub serves 404 on /stargazers to non-browser clients --><img src="https://img.shields.io/github/stars/pytorch/executorch?style=for-the-badge&color=blue" alt="GitHub - Stars"></a>
  <a href="https://discord.gg/Dh43CKSAdc"><img src="https://img.shields.io/badge/Discord-Join%20Us-blue?logo=discord&logoColor=white&style=for-the-badge" alt="Discord - Chat with Us"></a>
  <a href="https://docs.pytorch.org/executorch/main/index.html"><img src="https://img.shields.io/badge/Documentation-blue?logo=googledocs&logoColor=white&style=for-the-badge" alt="Documentation"></a>
</div>

**ExecuTorch** 是 PyTorch 官方推出的开源技术栈，用于在手机、可穿戴设备、笔记本电脑、浏览器、嵌入式系统和微控制器（MCU）上本地运行 AI。从 PyTorch 模型出发，使用 `torch.export` 捕获计算图，针对目标硬件深度优化，并通过 C++、Python、Swift/Objective-C、Kotlin/Java 或 JavaScript API 高效运行。

ExecuTorch 为 **Instagram、WhatsApp、Facebook 和 Messenger** 上的端侧体验提供强劲支持，服务全球数十亿用户。它还驱动着 **Meta Quest 和 Ray-Ban Meta 智能眼镜设备** 上的端侧 AI 功能。
[查看 ExecuTorch 的量产落地案例。](https://docs.pytorch.org/executorch/main/success-stories.html)

> [!IMPORTANT]
> **发布渠道说明**：推荐在生产环境中使用[最新正式版本 (latest release)](https://github.com/pytorch/executorch/releases/latest) 并配合[稳定版官方文档 (stable documentation)](https://docs.pytorch.org/executorch/stable/)。
> 本 README 跟踪 `main` 分支；标记为 **Main / nightly** 的特性在正式发布前可能发生变化。若使用源码检出或 Nightly 安装包，请参考[开发主线文档 (main documentation)](https://docs.pytorch.org/executorch/main/)。

## 专为现代边缘工作负载打造

| 工作负载 | 代表性功能特性 | 入门指引 |
|---|---|---|
| **端侧大语言模型 (LLM) 与智能体构建基石** | 量化文本模型、长上下文、工具调用、投机推测解码 (Speculative Decoding)、多会话执行，以及实验性兼容 OpenAI 协议的本地服务 | [Muse Glimmer](examples/models/muse-glimmer/README.md) · [LLM 开发指南](https://docs.pytorch.org/executorch/main/llm/working-with-llms.html) · [LLM 本地服务端](examples/llm_server/README.md) |
| **语音处理 (Voice)** | 流式与离线语音识别、语音合成 (TTS)、语音活动检测 (VAD) 以及说话人日志识别 (Speaker Diarization) | [Voxtral Realtime](examples/models/voxtral_realtime/README.md) · [Parakeet](examples/models/parakeet/README.md) · [Sortformer 说话人日志](examples/models/sortformer/README.md) · [Voxtral TTS](examples/models/voxtral_tts/README.md) |
| **多模态 (Multimodal)** | 文本、图像和音频执行器；移动端视觉语言模型 (VLM) | [Gemma 4](examples/models/gemma4/README.md) · [多模态执行器](extension/llm/runner/README.md) |
| **计算机视觉 (Computer vision)** | 图像分类、目标检测、语义分割以及可提示分割 (Promptable Segmentation) | [MobileNet V2](https://docs.pytorch.org/executorch/main/getting-started.html#preparing-the-model) · [YOLO26](examples/models/yolo26/README.md) · [DeepLabV3](https://github.com/meta-pytorch/executorch-examples/tree/main/dl3/android/DeepLabV3Demo) · [EfficientSAM](examples/models/efficient_sam/README.md) |
| **嵌入式 AI (Embedded AI)** | Cortex-M CPU 算子核、Ethos-U 与 NXP NPU、Cadence DSP、Zephyr、Arduino 以及 Raspberry Pi Pico 工作流 | [嵌入式指南](https://docs.pytorch.org/executorch/main/embedded-section.html) · [Cortex-M](https://docs.pytorch.org/executorch/main/backends/arm-cortex-m/arm-cortex-m-overview.html) · [Arduino](examples/arduino/README.md) |

上述模型均为典型起点，而非排他性的兼容性列表。模型无需显式列在此处即可在 ExecuTorch 上运行：请参阅[标准导出指南](https://docs.pytorch.org/executorch/main/using-executorch-export.html)或参考最相近的[模型示例](examples/models/)。最终部署取决于 `torch.export` 图捕获以及算子、运行时和所选后端的覆盖范围；请在目标设备上验证数值精度、内存占用及实际性能。对于全新语言模型架构，请参阅[自定义 LLM 指南](https://docs.pytorch.org/executorch/main/llm/export-custom-llm.html)。

## 为什么选择 ExecuTorch

- **PyTorch 原生工作流**：直接从 `torch.export` 开始工作，PyTorch 程序元数据与源文件映射信息在导出和调试工具中完全透明可用。
- **分区硬件加速 (Partitioned hardware acceleration)**：将受支持的子图区域委托下沉至 CPU、GPU、NPU 或 DSP 专用加速后端，同时保留高可移植的 CPU 算子核作为无缝回退（Fallback）。
- **一份源码模型，明确目标导出**：复用同一 PyTorch 模型定义与导出流程，针对每个需要硬件特化的目标设备生成专用的 `.pte` 文件。
- **尺寸可精确定制的运行时**：仅链接特定部署所需的算子、计算核与硬件委托器；[选择性构建 (Selective Build)](https://docs.pytorch.org/executorch/main/kernel-library-selective-build.html) 能够精确剔除所有不需要的算子内核。
- **探查、性能剖析与扩展**：利用 [ETDump](https://docs.pytorch.org/executorch/main/etdump.html)、[ETRecord](https://docs.pytorch.org/executorch/main/etrecord.html) 以及[数值精度调试工具](https://docs.pytorch.org/executorch/main/model-debugging.html)，亦或扩展自定义算子与加速后端。
- **带版本约束的部署契约**：使用稳定版 API 构建的 `.pte` 文件，保证至少能够在紧随其后的一个非补丁版本运行时中正常加载并执行；详见[运行时兼容性策略](runtime/COMPATIBILITY.md)与 [API 生命周期规范](https://docs.pytorch.org/executorch/main/api-life-cycle.html)。

## 安装指引

在 Python 3.10–3.14 环境中安装最新稳定版 Python 包：

```bash
pip install executorch
```

如需体验来自 `main` 分支的最新开发特性，可安装 Nightly 预构建版本：

```bash
pip install --upgrade --pre executorch torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

此处之所以显式指定 `torch`，是因为 ExecuTorch 的 Nightly Wheel 包并未显式声明对其的依赖。上述命令安装的是仅支持 CPU 的 PyTorch Nightly。在带有 NVIDIA GPU 的 Linux 环境下，请在 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)中选取匹配您 CUDA 版本的安装命令，并替换为对应的 `nightly/cu*` 源索引；否则，pip 可能会将已有的 CUDA 版 PyTorch 替换为 CPU 版本。ExecuTorch CUDA Wheel 仅针对 Linux 平台发布。

后端导出工具可能需要额外的可选依赖。例如，使用 `pip install 'executorch[ethos_u]'` 安装 Ethos-U AOT 导出依赖。嵌入式交叉工具链、模拟器及目标硬件运行时需单独安装。

有关特定平台（Android、iOS、嵌入式系统）的环境配置，请参阅[快速入门 (Quick Start)](https://docs.pytorch.org/executorch/main/quick-start-section.html) 文档获取补充信息。

预编译的 Python Wheel 包已针对 Linux x86-64、Linux AArch64、macOS arm64 及 Windows x86-64 发布。对于其他宿主架构或定制配置，可选择源码构建。原生平台集成亦可通过以下方式获取：

- [主流 Linux 与 macOS Wheel 中附带的预编译 C++ 库、头文件与 CMake 配置文件](https://docs.pytorch.org/executorch/main/using-executorch-cpp.html#using-the-prebuilt-libraries-from-the-pip-package)
- [Maven Central 上的 Android AAR 包](https://docs.pytorch.org/executorch/main/using-executorch-android.html)
- [通过 Swift Package Manager 提供的 Apple Framework](https://docs.pytorch.org/executorch/main/using-executorch-ios.html)
- [源码构建与交叉编译支持](https://docs.pytorch.org/executorch/main/using-executorch-building-from-source.html)

## 五分钟快速上手：模型导出与推理

以下完整示例来源于[快速上手实践路径 (Quickstart Pathway)](https://docs.pytorch.org/executorch/main/pathway-quickstart.html)，展示了如何导出微型模型并将其下沉编译至 XNNPACK 后端：

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

预期终端输出：`Output: tensor([2.])`。该步骤验证了模型在宿主机上的导出、XNNPACK 下沉编译、序列化、加载与执行链路；随后可在目标设备上进一步评测实际运行性能。

生成的 `add.pte` 针对所选后端进行了专门优化。若要面向 Core ML、Qualcomm 或其他加速硬件，需要该后端专用的依赖项与配置并进行单独导出，而非盲目直接替换图分区器。
后续进阶指引请参阅 [Python 运行时](https://docs.pytorch.org/executorch/main/getting-started.html#testing-the-model)、[C++ Module API](https://docs.pytorch.org/executorch/main/using-executorch-cpp.html)、[Android 部署](https://docs.pytorch.org/executorch/main/android-section.html) 或 [iOS 部署](https://docs.pytorch.org/executorch/main/ios-section.html)。

### 选择适合的导出路径

| 起始类型 | 推荐导出路径 | 覆盖范围与适用场景 |
|---|---|---|
| PyTorch `nn.Module` | [标准导出与下沉编译](https://docs.pytorch.org/executorch/main/using-executorch-export.html) | 支持所有后端选择；部分模型可能需要算子分解（Decompositions）或自定义算子 |
| Hugging Face `PreTrainedModel` | [Transformers ExecuTorch 导出器](https://huggingface.co/docs/transformers/en/exporters) | 实验性程序化 XNNPACK/CUDA 导出；自回归生成组件需在应用层进行编排 |
| 优化后的文本或多模态生成大模型 | [`export_llm`](https://docs.pytorch.org/executorch/main/llm/export-llm.html) 或 [Optimum ExecuTorch](https://huggingface.co/docs/optimum-executorch/) | 经过严谨测试的训练配方、量化支持、分词器与面向执行器的专有元数据 |

如果您已经拥有兼容的 `.pte` 文件，欢迎浏览 [ExecuTorch Community](https://huggingface.co/executorch-community)、[针对 ExecuTorch 筛选的 Arm AI 模型目录](https://developer.arm.com/ai/models?runtime=executorch)，或上文中链接的模型页面。请确保模型配置、数值精度、加速后端及运行时版本精准匹配；针对特定硬件委托构建的 `.pte` 并非通用跨平台文件。

## 运行时与大模型 API

支持使用 C++、Python、Java/Kotlin、Swift/Objective-C 或 JavaScript/WebAssembly 执行通用 `.pte` 模型。面向 C++、Python、Android 及 Apple 平台还提供了更高层级的文本与多模态专用 Runner 执行器。

参阅[运行时与 LLM API 对比指南](https://docs.pytorch.org/executorch/main/api-section.html)了解各语言接入点及 API 成熟度。

## 目标平台与硬件加速后端

请结合目标硬件架构、算子覆盖度以及具体部署约束选择合适的加速后端。各链接详细记录了环境搭建、支持硬件及已知局限。下表为典型路径示例，具体请以[后端官方文档](https://docs.pytorch.org/executorch/main/backends-overview.html)为准。另请参阅[桌面端平台指南 (Desktop guide)](desktop/README.md)。

| 目标平台 | 加速后端与深度集成 |
|---|---|
| Android | [XNNPACK](https://docs.pytorch.org/executorch/main/backends/xnnpack/xnnpack-overview.html) CPU；[Vulkan](https://docs.pytorch.org/executorch/main/backends/vulkan/vulkan-overview.html) GPU；[Qualcomm 高通](https://docs.pytorch.org/executorch/main/backends-qualcomm.html)、[MediaTek 联发科](https://docs.pytorch.org/executorch/main/backends-mediatek.html)、[Arm VGF](https://docs.pytorch.org/executorch/main/backends/arm-vgf/arm-vgf-overview.html) 以及 [Samsung Exynos 三星](https://docs.pytorch.org/executorch/main/backends/samsung/samsung-overview.html) 加速器 |
| iOS / iPadOS | [XNNPACK](https://docs.pytorch.org/executorch/main/backends/xnnpack/xnnpack-overview.html) CPU；[Core ML](https://docs.pytorch.org/executorch/main/backends/coreml/coreml-overview.html)；真机运行实验性 [MLX](https://docs.pytorch.org/executorch/main/backends/mlx/mlx-overview.html) *(实验性)* |
| macOS | [XNNPACK](https://docs.pytorch.org/executorch/main/backends/xnnpack/xnnpack-overview.html)；[Core ML](https://docs.pytorch.org/executorch/main/backends/coreml/coreml-overview.html)；实验性 [MLX](https://docs.pytorch.org/executorch/main/backends/mlx/mlx-overview.html)、[Metal/AOTInductor](backends/apple/metal/README.md) 以及 [WebGPU](https://docs.pytorch.org/executorch/main/backends/webgpu/webgpu-overview.html) 路径 |
| Linux | [XNNPACK](https://docs.pytorch.org/executorch/main/backends/xnnpack/xnnpack-overview.html)；[OpenVINO](https://docs.pytorch.org/executorch/main/build-run-openvino.html)；实验性桌面端 [CUDA/AOTInductor](https://docs.pytorch.org/executorch/main/backends/cuda/cuda-overview.html)、[Vulkan](https://docs.pytorch.org/executorch/main/backends/vulkan/vulkan-overview.html) 以及 [WebGPU](https://docs.pytorch.org/executorch/main/backends/webgpu/webgpu-overview.html) 路径 |
| Windows | [XNNPACK](https://docs.pytorch.org/executorch/main/backends/xnnpack/xnnpack-overview.html)；实验性桌面端 [CUDA/AOTInductor](https://docs.pytorch.org/executorch/main/backends/cuda/cuda-overview.html) 与 [Vulkan](https://docs.pytorch.org/executorch/main/backends/vulkan/vulkan-overview.html) 路径 |
| 浏览器 / WebAssembly | [轻量化可移植 WebAssembly 运行时](extension/wasm/README.md) 与 [WebGPU](https://docs.pytorch.org/executorch/main/backends/webgpu/webgpu-overview.html) *(均为实验性)* |
| 嵌入式 / MCU | [基于 CMSIS-NN 的 Arm Cortex-M](https://docs.pytorch.org/executorch/main/backends/arm-cortex-m/arm-cortex-m-overview.html) *(Beta)*；[Arm Ethos-U](https://docs.pytorch.org/executorch/main/backends/arm-ethos-u/arm-ethos-u-overview.html)；[NXP eIQ Neutron](https://docs.pytorch.org/executorch/main/backends/nxp/nxp-overview.html)；[Cadence DSP](https://docs.pytorch.org/executorch/main/backends-cadence.html)；[Zephyr](zephyr/README.md) 与 [Arduino](examples/arduino/README.md) 集成 |

## 官方文档与资源

- [新手快速入门 (Get started)](https://docs.pytorch.org/executorch/main/getting-started.html)
- [选择目标平台 (Choose a platform)](https://docs.pytorch.org/executorch/main/edge-platforms-section.html)
- [选择硬件加速后端 (Choose a backend)](https://docs.pytorch.org/executorch/main/backends-overview.html)
- [模型导出与下沉编译 (Export and lower a model)](https://docs.pytorch.org/executorch/main/using-executorch-export.html)
- [大语言模型实战 (Work with LLMs)](https://docs.pytorch.org/executorch/main/llm/working-with-llms.html)
- [性能剖析与精度调试 (Profile and debug)](https://docs.pytorch.org/executorch/main/tools-section.html)
- [运行时与 LLM API (Runtime and LLM APIs)](https://docs.pytorch.org/executorch/main/api-section.html)
- [PTE 模型与运行时兼容性政策](runtime/COMPATIBILITY.md)
- [LLM 性能基准看板 (Benchmark Dashboard)](https://hud.pytorch.org/benchmark/llms?repoName=pytorch%2Fexecutorch) <!-- @lint-ignore -->
- [故障排查与技术支持 (Troubleshooting)](https://docs.pytorch.org/executorch/main/support-section.html)

## 社区互动与参与贡献

- [GitHub Discussions 讨论区](https://github.com/pytorch/executorch/discussions)：提问交流与分享灵感
- [Discord 社区](https://discord.gg/Dh43CKSAdc)：与核心维护者和社区开发者即时探讨
- [GitHub Issues](https://github.com/pytorch/executorch/issues)：提交缺陷反馈或功能需求
- [贡献指南 (Contributing guide)](CONTRIBUTING.md)：本地开发环境搭建、单元测试与代码评审规范

## 引用 ExecuTorch

阅读发表于 [MLSys 2026 上的学术论文 (PDF)](https://proceedings.mlsys.org/paper_files/paper/2026/file/236f915dd02af4f11927f67330b21d4b-Paper-Conference.pdf)。

如果您在学术研究中使用了 ExecuTorch，请使用以下 BibTeX 条目进行引用：

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

## 开源协议

ExecuTorch 遵循 BSD 开源许可证。详见 [LICENSE](LICENSE)。

---

<div align="center">
  <p><strong>PyTorch 生态重要组成部分</strong></p>
  <p>
    <a href="https://github.com/pytorch/executorch">GitHub 仓库</a> •
    <a href="https://docs.pytorch.org/executorch">官方文档</a>
  </p>
</div>

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月27日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
