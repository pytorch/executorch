(success-stories)=

# Success Stories

Discover how organizations are leveraging ExecuTorch to deploy AI models at scale on edge devices.

---

## Featured Success Stories

::::{grid} 1
:gutter: 3

:::{grid-item-card} **Meta's Family of Apps**
:class-header: bg-primary text-white

**Industry:** Social Media & Messaging
**Hardware:** Android & iOS Devices
**Impact:** Billions of users, latency reduction

Powers Instagram, WhatsApp, Facebook, and Messenger with real-time on-device AI for content ranking, recommendations, and privacy-preserving features at scale.

[Read Blog →](https://engineering.fb.com/2025/07/28/android/executorch-on-device-ml-meta-family-of-apps/)
:::

:::{grid-item-card} **Meta Quest & Ray-Ban Smart Glasses**
:class-header: bg-success text-white

**Industry:** AR/VR & Wearables
**Hardware:** Quest 3, Ray-Ban Meta Smart Glasses, Meta Ray-Ban Display

Enables real-time computer vision, hand tracking, voice commands, and translation on power-constrained wearable devices.
[Read Blog →](https://ai.meta.com/blog/executorch-reality-labs-on-device-ai/)
:::

:::{grid-item-card} **Liquid AI: Efficient, Flexible On-Device Intelligence**
:class-header: bg-info text-white

**Industry:** Artificial Intelligence / Edge Computing
**Hardware:** CPU via PyTorch ExecuTorch
**Impact:** 2× faster inference, lower latency, seamless multimodal deployment

Liquid AI builds foundation models that make AI work where the cloud can't. In its LFM2 series, the team uses PyTorch ExecuTorch within the LEAP Edge SDK to deploy high-performance multimodal models efficiently across devices. ExecuTorch provides the flexibility to support custom architectures and processing pipelines while reducing inference latency through graph optimization and caching. Together, they enable faster, more efficient, privacy-preserving AI that runs entirely on the edge.

[Read Blog →](https://www.liquid.ai/blog/how-liquid-ai-uses-executorch-to-power-efficient-flexible-on-device-intelligence) <!-- @lint-ignore -->
:::

:::{grid-item-card} **PrivateMind: Complete Privacy with On-Device AI**
:class-header: bg-warning text-white

**Industry:** Privacy & Personal Computing
**Hardware:** iOS & Android Devices
**Impact:** 100% on-device processing

PrivateMind delivers a fully private AI assistant using ExecuTorch's .pte format. Built with React Native ExecuTorch, it supports LLaMA, Qwen, Phi-4, and custom models with offline speech-to-text and PDF chat capabilities.

[Visit →](https://privatemind.swmansion.com)
:::

:::{grid-item-card} **NimbleEdge: On-Device Agentic AI Platform**
:class-header: bg-danger text-white

**Industry:** AI Infrastructure
**Hardware:** iOS & Android Devices
**Impact:** 30% higher TPS on iOS, faster time-to-market with Qwen/Gemma models

NimbleEdge successfully integrated ExecuTorch with its open-source DeliteAI platform to enable agentic workflows orchestrated in Python on mobile devices. The extensible ExecuTorch ecosystem allowed implementation of on-device optimization techniques leveraging contextual sparsity. ExecuTorch significantly accelerated the release of "NimbleEdge AI" for iOS, enabling models like Qwen 2.5 with tool calling support and achieving up to 30% higher transactions per second.

[Visit →](https://nimbleedge.com) • [Blog →](https://www.nimbleedge.com/blog/meet-nimbleedge-ai-the-first-truly-private-on-device-assistant) • [iOS App →](https://apps.apple.com/in/app/nimbleedge-ai/id6746237456)
:::

::::

---

## Featured Ecosystem Integrations and Interoperability

::::{grid} 2 2 3 3
:gutter: 2

:::{grid-item-card} **Hugging Face Transformers**
:class-header: bg-secondary text-white

Popular models from Hugging Face easily export to ExecuTorch format for on-device deployment.

[Learn More →](https://github.com/huggingface/optimum-executorch/)
:::

:::{grid-item-card} **React Native ExecuTorch**
:class-header: bg-secondary text-white

Declarative toolkit for running AI models and LLMs in React Native apps with privacy-first, on-device execution.

[Explore →](https://docs.swmansion.com/react-native-executorch/) • [Blog →](https://expo.dev/blog/how-to-run-ai-models-with-react-native-executorch)
:::

:::{grid-item-card} **torchao**
:class-header: bg-secondary text-white

PyTorch-native quantization and optimization library for preparing efficient models for ExecuTorch deployment.

[Blog →](https://pytorch.org/blog/torchao-quantized-models-and-quantization-recipes-now-available-on-huggingface-hub/) • [Qwen Example →](https://huggingface.co/pytorch/Qwen3-4B-INT8-INT4) • [Phi Example →](https://huggingface.co/pytorch/Phi-4-mini-instruct-INT8-INT4) 
:::

:::{grid-item-card} **Unsloth**
:class-header: bg-secondary text-white

Optimize LLM fine-tuning with faster training and reduced VRAM usage, then deploy efficiently with ExecuTorch.

[Example Model →](https://huggingface.co/metascroy/Qwen3-4B-int8-int4-unsloth) • [Blog →](https://docs.unsloth.ai/new/quantization-aware-training-qat) • [Doc →](https://docs.unsloth.ai/new/deploy-llms-phone)
:::

:::{grid-item-card} **Ultralytics**
:class-header: bg-secondary text-white

Deploy on-device inference for Ultralytics YOLO models using ExecuTorch.

[Explore →](https://docs.ultralytics.com/integrations/executorch/) • [Blog →](https://www.ultralytics.com/blog/deploy-ultralytics-yolo-models-using-the-executorch-integration)
:::

:::{grid-item-card} **Arm ML Embedded Evaluation Kit**
:class-header: bg-secondary text-white

Build and deploy ML applications on Arm Cortex-M (M55, M85) and Ethos-U NPUs (U55, U65, U85) using ExecuTorch.

[Explore →](https://gitlab.arm.com/artificial-intelligence/ethos-u/ml-embedded-evaluation-kit)
:::

:::{grid-item-card} **Alif Semiconductor Ensemble**
:class-header: bg-secondary text-white

Run generative AI on Ensemble E4/E6/E8 MCUs with Arm Ethos-U85 NPU acceleration.

[Learn More →](https://alifsemi.com/press-release/alif-semiconductor-elevates-generative-ai-with-support-for-executorch-runtime/)
:::

:::{grid-item-card} **Digica AI SDK**
:class-header: bg-secondary text-white

Automate PyTorch model deployment to iOS, Android, and edge devices with ExecuTorch-powered SDK.

[Blog →](https://www.digica.com/blog/effortless-edge-deployment-of-ai-models-with-digicas-ai-sdk-feat-executorch.html)
:::

::::

---

## Featured Demos

- **Text and Multimodal LLM demo mobile apps** - Text (Llama, Qwen3, Phi-4) and multimodal (Gemma3, Voxtral) mobile demo apps. [Try →](https://github.com/meta-pytorch/executorch-examples/tree/main/llm)

- **Voxtral** - Deploy audio-text-input LLM on CPU (via XNNPACK) and on CUDA. [Try →](https://github.com/pytorch/executorch/blob/main/examples/models/voxtral/README.md)

- **Whisper** - Deploy OpenAI's Whisper speech recognition model on CUDA and Metal backends. [Try →](https://github.com/pytorch/executorch/blob/main/examples/models/whisper/README.md)

- **LoRA adapter** - Export two LoRA adapters that share a single foundation weight file, saving memory and disk space. [Try →](https://github.com/meta-pytorch/executorch-examples/tree/main/program-data-separation/cpp/lora_example)

- **OpenVINO from Intel** - Deploy [Yolo12](https://github.com/pytorch/executorch/tree/main/examples/models/yolo12), [Llama](https://github.com/pytorch/executorch/tree/main/examples/openvino/llama), and [Stable Diffusion](https://github.com/pytorch/executorch/tree/main/examples/openvino/stable_diffusion) on [OpenVINO from Intel](https://www.intel.com/content/www/us/en/developer/articles/community/optimizing-executorch-on-ai-pcs.html).

- **Audio Generation** - Generate audio from text prompts using Stable Audio Open Small on Arm CPUs with XNNPACK and KleidiAI. [Try →](https://github.com/Arm-Examples/ML-examples/tree/main/kleidiai-examples/audiogen-et) • [Video →](https://www.youtube.com/watch?v=q2P0ESVxhAY)

*Want to showcase your demo? [Submit here →](https://github.com/pytorch/executorch/issues)*
