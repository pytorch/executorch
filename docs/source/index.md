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

New to ExecuTorch? Start here for installation and your first model deployment.

[**Get Started**](quick-start-section)
:::

:::{grid-item-card} **Deploy on Edge Platforms**

Deploy on Android, iOS, Laptops / Desktops and embedded platforms with optimized backends.

[**Deploy on Edge Platforms**](edge-platforms-section)
:::

:::{grid-item-card} **Work with LLMs**

Export, optimize, and deploy Large Language Models on edge devices.

[**Work with LLMs**](llm/working-with-llms)
:::

:::{grid-item-card} 🔧 **Developer Tools**

Profile, debug, and inspect your models with comprehensive tooling.

[**Developer Tools**](tools-section)
:::

::::

---

## Explore Documentation

::::{grid} 1
:::{grid-item-card} **Intro**

**Overview, architecture, and core concepts** — Understand how ExecuTorch works and its benefits

[:doc:`Intro`](intro-section)
:::
::::

::::{grid} 1
:::{grid-item-card} **Quick Start**

**Get started with ExecuTorch** — Install, export your first model, and run inference

[:doc:`Quick Start`](quick-start-section)
:::
::::

::::{grid} 1
:::{grid-item-card} **Edge**

**Android, iOS, Desktop, Embedded** — Platform-specific deployment guides and examples

[:doc:`Edge`](edge-platforms-section)
:::
::::

::::{grid} 1
:::{grid-item-card} **Backends**

**CPU, GPU, NPU/Accelerator backends** — Hardware acceleration and backend selection

[:doc:`Backends`](backends-section)
:::
::::

::::{grid} 1
:::{grid-item-card} **LLMs**

**LLM export, optimization, and deployment** — Complete LLM workflow for edge devices

[:doc:`LLMs`](llm/working-with-llms)
:::
::::

::::{grid} 1
:::{grid-item-card} **Advanced**

**Quantization, memory planning, custom passes** — Deep customization and optimization

[:doc:`Advanced`](advanced-topics-section)
:::
::::

::::{grid} 1
:::{grid-item-card} **Tools**

**Developer tools, profiling, debugging** — Comprehensive development and debugging suite

[:doc:`Tools`](tools-section)
:::
::::

::::{grid} 1
:::{grid-item-card} **API**

**API Reference Usages & Examples** — Detailed Python, C++, and Java API references

[:doc:`API`](api-section)
:::
::::

::::{grid} 1
:::{grid-item-card} **💬 Support**

**FAQ, troubleshooting, contributing** — Get help and contribute to the project

[:doc:`💬 Support`](support-section)
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
- Go **→ [:doc:`edge-platforms-section`](edge-platforms-section)**
:::

:::{grid-item}
**Rich Acceleration**

- CPU
- GPU
- NPU
- DSP
- Go **→ [:doc:`backends-section`](backends-section)**
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
