# ExecuTorch examples

Use this directory to find an end-to-end workflow for a task or hardware
target. The model entries below are representative starting points, not a
compatibility list. A model does not need a dedicated example: adapt the
closest workflow, then validate `torch.export` capture, ExecuTorch runtime and
operator coverage, and the selected backend.

## Start here

| Goal | Example |
|---|---|
| Learn the export-to-runtime flow | [Portable runtime](portable/) |
| Export your own PyTorch model | [Export guide](../docs/source/using-executorch-export.md) |
| Export a Hugging Face model | [Transformers exporter](https://huggingface.co/docs/transformers/en/exporters) *(experimental)* or [Optimum ExecuTorch](../docs/source/llm/export-llm-optimum.md) |
| Understand an LLM runner end to end | [Minimal LLM](llm_manual/) |
| Profile and debug a program | [Developer tools](devtools/) |
| Reduce runtime size | [Selective build](selective_build/) |
| Find a nearby model pattern | [Model implementations](models/) |

## Find an example by task

These are representative entry points, not an exhaustive compatibility list.
“Released” means the example appeared in a stable ExecuTorch release; it does
not imply that every backend or configuration is supported.

| Task | Representative examples | Availability |
|---|---|---|
| Text generation | [Llama](models/llama/), [Qwen3.5](models/qwen3_5/), [Qwen3.5 MoE](models/qwen3_5_moe/), [Gemma 4 31B](models/gemma4_31b/) | Released; consult each backend matrix |
| Multimodal generation | [Gemma 4](models/gemma4/), [Voxtral](models/voxtral/) | Released |
| Streaming speech recognition | [Voxtral Realtime](models/voxtral_realtime/), [Parakeet](models/parakeet/) | Released |
| Speaker and speech detection | [Sortformer](models/sortformer/), [Silero VAD](models/silero_vad/) | Released |
| Text-to-speech | [Voxtral TTS](models/voxtral_tts/), [Supertonic](models/supertonic/) | Voxtral TTS released; Supertonic is on main/nightly |
| Computer vision | [DINOv2](models/dinov2/), [EfficientSAM](models/efficient_sam/), [YOLO26](models/yolo26/) | Model-specific |
| On-device adaptation | [PTE fine-tuning](llm_pte_finetuning/) | Experimental workflow |
| Local OpenAI-compatible serving | [LLM server](llm_server/) | Experimental workflow |

## Find an example by platform or backend

| Target | Examples | Maturity notes |
|---|---|---|
| Portable CPU | [Portable](portable/), [XNNPACK](xnnpack/) | General-purpose starting points |
| Android GPU and NPU | [Vulkan](vulkan/), [Qualcomm](qualcomm/), [MediaTek](mediatek/), [Samsung](samsung/) | Hardware and SDK requirements vary |
| Apple devices | [Core ML](apple/coreml/), [MLX LLM](../backends/mlx/examples/llm/) | MLX is experimental |
| Desktop acceleration | [CUDA](cuda/), [Vulkan](vulkan/), [OpenVINO](openvino/) | CUDA and Vulkan are experimental; OpenVINO setup is Linux-only |
| Browser and WebAssembly | [WebAssembly](wasm/), [WebGPU guide](../docs/source/backends/webgpu/webgpu-overview.md) | WebGPU is experimental |
| Arm microcontrollers | [Arm](arm/), [Arduino](arduino/), [Raspberry Pi Pico 2](raspberry_pi/pico2/), [Zephyr/Alif](../docs/source/zephyr_alif_tutorial.md) | Cortex-M backend is beta |
| Embedded and edge SoCs | [NXP](nxp/), [Cadence](cadence/), [Espressif](espressif/), [RISC-V](riscv/) | Backend-specific toolchains required |

For complete mobile applications, see
[executorch-examples](https://github.com/meta-pytorch/executorch-examples/tree/main)
and the local [demo apps](demo-apps/).

## Before running an example

- Follow the example's own README; dependencies, model assets, quantization,
  and build steps differ by target.
- Start with `pip install executorch` for supported host platforms. Use the
  [build-from-source guide](../docs/source/using-executorch-building-from-source.md)
  when an example requires a custom backend or build option.
- Treat downloaded model weights and generated `.pte`/`.ptd` files as trusted
  inputs. Third-party model licenses and terms still apply.

## Disclaimer

The ExecuTorch Repository Content is provided without any guarantees about performance or compatibility. In particular, ExecuTorch makes available model architectures written in Python for PyTorch that may not perform in the same manner or meet the same standards as the original versions of those models. When using the ExecuTorch Repository Content, including any model architectures, you are solely responsible for determining the appropriateness of using or redistributing the ExecuTorch Repository Content and assume any risks associated with your use of the ExecuTorch Repository Content or any models, outputs, or results, both alone and in combination with any other technologies. Additionally, you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models, weights, data, or other technologies, and you are solely responsible for complying with all such obligations.
