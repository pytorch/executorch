(home)=
# Welcome to the ExecuTorch Documentation

**ExecuTorch** is PyTorch's solution for efficient AI inference on edge devices — from mobile phones to embedded systems.

## Key Value Propositions

- **Portability:** Run on diverse platforms, from high-end mobile to constrained microcontrollers
- **Performance:** Lightweight runtime with full hardware acceleration (CPU, GPU, NPU, DSP)
- **Productivity:** Use familiar PyTorch tools from authoring to deployment

---

## 🗺️ Find Your Path

Not sure where to start? Use the guided pathways to navigate ExecuTorch based on your experience level, goal, and target platform.

::::{grid} 3
:gutter: 3

:::{grid-item-card} 🟢 New to ExecuTorch
:class-header: bg-success text-white
:link: pathway-beginner
:link-type: doc

Step-by-step learning sequence from installation to your first on-device deployment. Includes concept explanations and worked examples.

+++
**Beginner Pathway →**
:::

:::{grid-item-card} 🟡 Get Running Fast
:class-header: bg-warning text-dark
:link: pathway-quickstart
:link-type: doc

Skip the theory — get a model running in 15 minutes. Includes export cheat sheets, backend selection tables, and platform quick starts.

+++
**Quick Start Pathway →**
:::

:::{grid-item-card} 🔴 Production & Advanced
:class-header: bg-danger text-white
:link: pathway-advanced
:link-type: doc

Quantization, custom backends, C++ runtime, LLM deployment, and compiler internals for production-grade systems.

+++
**Advanced Pathway →**
:::

::::

::::{grid} 1

:::{grid-item-card} 🔀 Decision Matrix — Route by Goal, Platform & Model
:link: user-pathways
:link-type: doc

Not sure which pathway fits? The decision matrix routes you by experience level, target platform, model status, and developer role to the exact documentation you need.

+++
**View Decision Matrix →**
:::

::::

---

## 🎯 Wins & Success Stories

::::{grid} 1
:class-container: success-showcase
:::{grid-item-card}
:class-header: bg-primary text-white
:class-body: text-center
[View All Success Stories →](success-stories)
:::
::::

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
:link: tools-section
:link-type: doc

Profile, debug, and inspect your models with comprehensive tooling.
:::

::::

---

## Explore Documentation

::::{grid} 1
:::{grid-item-card} **Intro**
:link: intro-section
:link-type: doc

**Overview, architecture, and core concepts** — Understand how ExecuTorch works and its benefits
:::
::::

::::{grid} 1
:::{grid-item-card} **Quick Start**
:link: quick-start-section
:link-type: doc

**Get started with ExecuTorch** — Install, export your first model, and run inference
:::
::::

::::{grid} 1
:::{grid-item-card} **Edge**
:link: edge-platforms-section
:link-type: doc

**Android, iOS, Desktop, Embedded** — Platform-specific deployment guides and examples
:::
::::

::::{grid} 1
:::{grid-item-card} **Backends**
:link: backends-section
:link-type: doc

**CPU, GPU, NPU/Accelerator backends** — Hardware acceleration and backend selection
:::
::::

::::{grid} 1
:::{grid-item-card} **LLMs**
:link: llm/working-with-llms
:link-type: doc

**LLM export, optimization, and deployment** — Complete LLM workflow for edge devices
:::
::::

::::{grid} 1
:::{grid-item-card} **Advanced**
:link: advanced-topics-section
:link-type: doc

**Quantization, memory planning, custom passes** — Deep customization and optimization
:::
::::

::::{grid} 1
:::{grid-item-card} **Tools**
:link: tools-section
:link-type: doc

**Developer tools, profiling, debugging** — Comprehensive development and debugging suite
:::
::::

::::{grid} 1
:::{grid-item-card} **API**
:link: api-section
:link-type: doc

**API Reference Usages & Examples** — Detailed Python, C++, and Java API references
:::
::::

::::{grid} 1
:::{grid-item-card} **💬 Support**
:link: support-section
:link-type: doc

**FAQ, troubleshooting, contributing** — Get help and contribute to the project
:::
::::

---

## What's Supported

::::{grid} 3

:::{grid-item}
**Model Types**

- Large Language Models (LLMs)
- Computer Vision (CV)
- Speech Recognition (ASR)
- Text-to-Speech (TTS)
- More ...
:::

:::{grid-item}
**Platforms**

- Android & iOS
- Linux, macOS, Windows
- Embedded & MCUs
- Go **→ {doc}`edge-platforms-section`**
:::

:::{grid-item}
**Rich Acceleration**

- CPU
- GPU
- NPU
- DSP
- Go **→ {doc}`backends-section`**
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

intro-section
quick-start-section
user-pathways
edge-platforms-section
backends-section
llm/working-with-llms
advanced-topics-section
tools-section
api-section
support-section
