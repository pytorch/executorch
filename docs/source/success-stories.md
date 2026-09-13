(success-stories)=

# Success Stories

See where ExecuTorch is shipping today, how organizations integrate it, and
what current reference implementations demonstrate. Labels on this page are
intentional: production claims link to primary sources, while prototypes and
experimental backends are identified as such.

---

## Production Deployments

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} **Meta's Family of Apps**
:class-header: bg-primary text-white

- **Status:** Production
- **Platforms:** Android and iOS
- **Scale:** Features serving billions of people

ExecuTorch runs on-device models behind Instagram Cutouts, WhatsApp bandwidth
estimation, Messenger language identification and encrypted experiences, and
Facebook Stories music recommendations.

[Read the engineering story →](https://engineering.fb.com/2025/07/28/android/executorch-on-device-ml-meta-family-of-apps/)
:::

:::{grid-item-card} **Meta Reality Labs**
:class-header: bg-primary text-white

- **Status:** Production
- **Devices:** Meta Quest 3 and 3S; Ray-Ban Meta, Oakley Meta Vanguard, and Meta
  Ray-Ban Display glasses

ExecuTorch powers capabilities including hand and controller tracking,
persistent room memory, live translation, visual captions, and on-device OCR.

[Read the product story →](https://ai.meta.com/blog/executorch-reality-labs-on-device-ai/)
:::

:::{grid-item-card} **LM Studio Voice Transcription**
:class-header: bg-primary text-white

- **Status:** Production
- **Platforms:** macOS and Windows
- **Model:** NVIDIA Parakeet TDT

LM Studio ships local voice transcription powered by ExecuTorch. The same model
and application layer target Apple GPUs on macOS and NVIDIA GPUs on Windows.

[Read the case study →](https://pytorch.org/blog/building-voice-agents-with-executorch-a-cross-platform-foundation-for-on-device-audio/)
:::

::::

---

## Customer and Product Case Studies

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} **Liquid AI: Hybrid Models on the Edge**
:class-header: bg-info text-white

- **Status:** Published customer case study
- **Hardware tested:** AMD Ryzen AI 9 HX 370 and Samsung Galaxy S24 CPUs
- **Reported result:** Up to 2× higher CPU throughput than selected similarly
  sized models, with reduced memory use

Liquid AI adopted ExecuTorch for LFM2 models from 350M to 4B parameters. Its
case study highlights support for hybrid attention and recurrent architectures,
portable model packaging, and integration with the LEAP platform.

[Read Liquid AI's case study →](https://www.liquid.ai/blog/how-liquid-ai-uses-executorch-to-power-efficient-flexible-on-device-intelligence) <!-- @lint-ignore -->
:::

:::{grid-item-card} **Private Mind: A Fully Local AI Assistant**
:class-header: bg-info text-white

- **Status:** Shipping and open source
- **Platforms:** iOS and Android
- **Workloads:** Chat, document retrieval, images, and speech input

Software Mansion built Private Mind with React Native ExecuTorch. After models
are downloaded, conversations, retrieval, embeddings, and inference stay on the
device.

[Source →](https://github.com/software-mansion-labs/private-mind) •
[App Store →](https://apps.apple.com/us/app/private-mind-local-ai/id6746713439) •
[Google Play →](https://play.google.com/store/apps/details?id=com.swmansion.privatemind)
:::

::::

---

## Ecosystem Integrations

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} **Hugging Face and torchao**
:class-header: bg-secondary text-white

Transformers includes an experimental generic ExecuTorch exporter, Optimum
ExecuTorch adds tested task-level recipes and model wrappers, and torchao supplies
PyTorch-native quantization recipes and kernels used throughout ExecuTorch.

[Transformers exporters →](https://huggingface.co/docs/transformers/en/exporters) •
[Optimum ExecuTorch →](https://huggingface.co/docs/optimum-executorch/en/index) •
[torchao →](https://docs.pytorch.org/ao/)
:::

:::{grid-item-card} **React Native ExecuTorch**
:class-header: bg-secondary text-white

A React Native library with task-level APIs, a pre-exported model catalog, and
XNNPACK, Core ML, MLX, and Vulkan acceleration.

[Documentation →](https://executorch.swmansion.com/) •
[Gallery →](https://github.com/software-mansion-labs/react-native-executorch-gallery)
:::

:::{grid-item-card} **Ultralytics**
:class-header: bg-secondary text-white

Ultralytics provides a first-class ExecuTorch export path for YOLO26 across its
seven vision tasks, targeting mobile and edge applications.

[Integration guide →](https://docs.ultralytics.com/integrations/executorch/)
:::

:::{grid-item-card} **NimbleEdge DeliteAI**
:class-header: bg-secondary text-white

The open-source DeliteAI SDK offers ExecuTorch as one of its model runtimes for
Python-orchestrated agent workflows in Android and iOS applications.

[Source →](https://github.com/NimbleEdge/deliteAI)
:::

:::{grid-item-card} **Arm, Alif, and Arduino**
:class-header: bg-secondary text-white

The Arm ML Embedded Evaluation Kit supports Cortex-M and Ethos-U targets. Alif
has demonstrated generative AI and real-time speech-to-text on its Ensemble E8,
and the ExecuTorch Arduino library is hardware-verified on Arduino UNO Q.

[Arm kit →](https://gitlab.arm.com/artificial-intelligence/ethos-u/ml-embedded-evaluation-kit) •
[Alif demonstration →](https://alifsemi.com/press-release/alif-semiconductor-elevates-generative-ai-with-support-for-executorch-runtime/) •
[Arduino library →](https://github.com/meta-pytorch/executorch-arduino)
:::

:::{grid-item-card} **OpenVINO from Intel**
:class-header: bg-secondary text-white

The OpenVINO delegate supports Intel CPU, GPU, and NPU deployment, with
end-to-end examples for YOLO26, Llama, and Stable Diffusion.

[Examples →](https://github.com/pytorch/executorch/tree/main/examples/openvino) •
[Intel guide →](https://www.intel.com/content/www/us/en/developer/articles/community/optimizing-executorch-on-ai-pcs.html)
:::

::::

---

## Technical Showcases

These are reference implementations, not claims of production deployment.
Performance figures apply only to the linked model, device, and measurement
setup.

[Browse representative model examples →](https://github.com/pytorch/executorch/tree/main/examples/models)

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} **Muse Glimmer 30B Agentic AI**
:class-header: bg-success text-white

- **Maturity:** Reference implementation
- **Platforms:** NVIDIA CUDA and Apple silicon through MLX

Run text and image input, 128K-token context, OpenAI-compatible serving, GGUF
K-quant weights, and DFlash speculative decoding. The published M5 Pro test
improved decode from 21.6 to 33.0 tokens/s (52.8%) without quality regression.

[Try it →](https://github.com/pytorch/executorch/tree/main/examples/models/muse-glimmer) •
[Read the benchmark →](https://pytorch.org/blog/fast-ondevice-agentic-ai-with-executorch/)
:::

:::{grid-item-card} **Gemma 4 Multimodal on Mobile**
:class-header: bg-success text-white

- **Maturity:** Reference implementation
- **Device measured:** Samsung Galaxy S25

Gemma 4 E2B and E4B examples combine audio transcription, translation, image
understanding, and text generation. The E2B 4-bit audio configuration reports a
0.71 real-time factor, 6 tokens/s generation, and 2,251 MB peak memory on the
documented 23-second sample.

[Try it and see the measurement setup →](https://github.com/pytorch/executorch/tree/main/examples/models/gemma4)
:::

:::{grid-item-card} **Voice Agent Building Blocks**
:class-header: bg-success text-white

**Maturity:** Reference implementations; one production adopter

Build transcription, streaming ASR, diarization, and voice activity detection
with Parakeet TDT, Voxtral Realtime, Whisper, Sortformer, and Silero VAD across
CPU, GPU, and NPU backends.

[Read the overview →](https://pytorch.org/blog/building-voice-agents-with-executorch-a-cross-platform-foundation-for-on-device-audio/) •
[Explore the models →](https://github.com/pytorch/executorch/tree/main/examples/models)
:::

:::{grid-item-card} **Current Text-to-Speech Models**
:class-header: bg-success text-white

**Maturity:** Reference implementations

Voxtral TTS 2603 supports CPU and CUDA; its documented RTX 5080 configuration
runs at 0.31× real-time factor, more than 3× real time. Its model weights and
voice embeddings are CC BY-NC 4.0. Supertonic 3 adds a dynamic FP16
text-to-speech path for the experimental MLX delegate.

[Voxtral TTS →](https://github.com/pytorch/executorch/tree/main/examples/models/voxtral_tts) •
[Supertonic 3 →](https://github.com/pytorch/executorch/tree/main/examples/models/supertonic)
:::

:::{grid-item-card} **Qwen 3.5**
:class-header: bg-success text-white

**Maturity:** Early model support

The dense 0.8B, 2B, and 4B path currently targets FP32 static-shape XNNPACK.
The 35B-A3B mixture-of-experts reference adds INT4 export, CUDA and MLX runners,
and OpenAI-compatible serving. Review each example's limitations before use.

[Dense models →](https://github.com/pytorch/executorch/tree/main/examples/models/qwen3_5) •
[Mixture of experts →](https://github.com/pytorch/executorch/tree/main/examples/models/qwen3_5_moe)
:::

:::{grid-item-card} **WebGPU in the Browser**
:class-header: bg-success text-white

**Maturity:** Experimental

The WebGPU backend demonstrates language, vision, retrieval, audio, and
on-device training workflows. A July 2026 M4 Pro measurement reports 188.3
decode tokens/s for a 4-bit Llama 3.2 1B artifact at 128-token context.

[See the benchmark and workflows →](https://github.com/pytorch/executorch/tree/main/backends/webgpu#performance)
:::

::::

---

## Community Showcase

At the June 2026 ExecuTorch Hackathon, more than 100 participants across over 20
teams built on-device prototypes on Snapdragon-powered Samsung Galaxy S25 Ultra
phones. The winning prototypes explored real-time visual safety, haptic
navigation for blind and low-vision users, and private gait analysis. These are
community prototypes, not production deployments.

[See the hackathon projects →](https://pytorch.org/blog/building-the-future-of-on-device-ai-at-the-executorch-hackathon/)

Want your project considered for this page? [Submit a success story](https://github.com/pytorch/executorch/issues/new?title=%5BSuccess%20story%5D%20)
with its deployment status, model, device and backend, reproducible measurements
and baseline, a public primary source, and permission to use any supplied logo or
image.
