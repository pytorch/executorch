(home)=
# Welcome to the ExecuTorch Documentation

**ExecuTorch** is PyTorch's solution for efficient AI inference on edge devices — from mobile phones to embedded systems.

## Key Value Propositions

- **Portability:** Run on diverse platforms, from high-end mobile to constrained microcontrollers
- **Performance:** Lightweight runtime with full hardware acceleration (CPU, GPU, NPU, DSP)
- **Productivity:** Use familiar PyTorch tools from authoring to deployment

---

## Quick Navigation

::::{grid} 2

:::{grid-item-card} **Get Started**
:link: quick-start-section
:link-type: doc

New to ExecuTorch? Start here for installation and your first model deployment.
:::

:::{grid-item-card} **Deploy on Edge Platforms**
:link: edge-platforms-section
:link-type: doc

Deploy on Android, iOS, Laptops / Desktops and embedded platforms with optimized backends.
:::

:::{grid-item-card} **Work with LLMs**
:link: llm/working-with-llms
:link-type: doc

Export, optimize, and deploy Large Language Models on edge devices.
:::

:::{grid-item-card} 🔧 **Developer Tools**
:link: tools-sdk-section
:link-type: doc

Profile, debug, and inspect your models with comprehensive tooling.
:::

::::

---

### Documentation Sections

#### [Introduction](intro-section)

**Overview, architecture, and core concepts** — Understand how ExecuTorch works and its benefits

#### [Quick Start](quick-start-section)

**Get started with ExecuTorch** — Install, export your first model, and run inference

#### [Edge Platforms](edge-platforms-section)

**Android, iOS, Desktop, Embedded** — Platform-specific deployment guides and examples

#### [Backends](backends-section)

**CPU, GPU, NPU/Accelerator backends** — Hardware acceleration and backend selection

#### [Working with LLMs](llm/working-with-llms.md)

**LLM export, optimization, and deployment** — Complete LLM workflow for edge devices

#### [Advanced Topics](advanced-topics-section)

**Quantization, memory planning, custom passes** — Deep customization and optimization

#### [Tools & SDK](tools-sdk-section)

**Developer tools, profiling, debugging** — Comprehensive development and debugging suite

#### [API Reference](api-section)

**API Reference Usages & Examples** — Detailed Python, C++, and Java API references

#### 💬 [Support](support-section)

**FAQ, troubleshooting, contributing** — Get help and contribute to the project

---

## What's Supported

::::{grid} 3

:::{grid-item}
**Model Types**

- Large Language Models (LLMs)
- Computer Vision (CV)
- Speech Recognition (ASR)
- Text-to-Speech (TTS)
:::

:::{grid-item}
**Platforms**

- Android & iOS
- Linux, macOS, Windows
- Embedded & MCUs
:::

:::{grid-item}
**Acceleration**

- Apple (CoreML, MPS)
- Qualcomm, MediaTek
- ARM, Cadence, XNNPACK
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

intro-section
quick-start-section
edge-platforms-section
backends-section
llm/working-with-llms
advanced-topics-section
tools-sdk-section
api-section
support-section
