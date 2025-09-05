<div align="center">
  <img src="docs/source/_static/img/et-logo.png" alt="Logo" width="200">
  <h1 align="center">ExecuTorch: A powerful on-device AI Framework</h1>
</div>


<div align="center">
  <a href="https://github.com/pytorch/executorch/graphs/contributors"><img src="https://img.shields.io/github/contributors/pytorch/executorch?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/pytorch/executorch/stargazers"><img src="https://img.shields.io/github/stars/pytorch/executorch?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://discord.gg/Dh43CKSAdc"><img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a>
  <a href="https://pytorch.org/executorch/main/index"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <hr>
</div>

**ExecuTorch** is an end-to-end solution for on-device inference and training. It powers much of Meta's on-device AI experiences across Facebook, Instagram, Meta Quest, Ray-Ban Meta Smart Glasses, WhatsApp, and more.

It supports a wide range of models including LLMs (Large Language Models), CV (Computer Vision), ASR (Automatic Speech Recognition), and TTS (Text to Speech).

Platform Support:
- Operating Systems:
  - iOS
  - MacOS (ARM64)
  - Android
  - Linux
  - Microcontrollers

- Hardware Acceleration:
  - Apple
  - Arm
  - Cadence
  - MediaTek
  - NXP
  - OpenVINO
  - Qualcomm
  - Vulkan
  - XNNPACK

Key value propositions of ExecuTorch are:

- **Portability:** Compatibility with a wide variety of computing platforms,
  from high-end mobile phones to highly constrained embedded systems and
  microcontrollers.
- **Productivity:** Enabling developers to use the same toolchains and Developer
  Tools from PyTorch model authoring and conversion, to debugging and deployment
  to a wide variety of platforms.
- **Performance:** Providing end users with a seamless and high-performance
  experience due to a lightweight runtime and utilizing full hardware
  capabilities such as CPUs, NPUs, and DSPs.

## Getting Started
To get started you can:

- Visit the [Step by Step Tutorial](https://pytorch.org/executorch/stable/getting-started.html) to get things running locally and deploy a model to a device
- Use this [Colab Notebook](https://colab.research.google.com/drive/1qpxrXC3YdJQzly3mRg-4ayYiOjC6rue3?usp=sharing) to start playing around right away
- Jump straight into LLM use cases by following specific instructions for popular open-source models such as [Llama](examples/models/llama/README.md), [Qwen 3](examples/models/qwen3/README.md), [Phi-4-mini](examples/models/phi_4_mini/README.md), [Llava](examples/models/llava/README.md), [Voxtral](examples/models/voxtral/README.md), and [LFM2](examples/models/lfm2/README.md).

## Feedback and Engagement

We welcome any feedback, suggestions, and bug reports from the community to help
us improve our technology. Check out the [Discussion Board](https://github.com/pytorch/executorch/discussions) or chat real time with us on [Discord](https://discord.gg/Dh43CKSAdc)

## Contributing

We welcome contributions. To get started review the [guidelines](CONTRIBUTING.md) and chat with us on [Discord](https://discord.gg/Dh43CKSAdc)


## Directory Structure

Please refer to the [Codebase structure](CONTRIBUTING.md#codebase-structure) section of the [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License
ExecuTorch is BSD licensed, as found in the LICENSE file.
