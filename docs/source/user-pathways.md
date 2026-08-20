(user-pathways)=
# Find Your Path

**ExecuTorch** serves a wide range of users — from ML engineers taking their first steps in on-device inference, to embedded systems developers targeting bare-metal microcontrollers, to researchers pushing the boundaries of LLM deployment. This page helps you navigate directly to the content most relevant to your experience level, goals, and target platform.

---

## Step 1: What best describes your experience?

::::{grid} 3
:gutter: 3

:::{grid-item-card} 🟢 New to ExecuTorch
:class-header: bg-success text-white
:link: pathway-beginner
:link-type: doc

**Beginner**

You are familiar with PyTorch but have not yet deployed a model to an edge device. You want a clear, guided path from installation to your first on-device inference.

+++
→ **Beginner Pathway**
:::

:::{grid-item-card} 🟡 I know the basics
:class-header: bg-warning text-dark
:link: pathway-quickstart
:link-type: doc

**Quick Start**

You have some experience with model export or mobile ML, and want to get a model running as fast as possible without reading through every concept first.

+++
→ **Quick Start Pathway**
:::

:::{grid-item-card} 🔴 Experienced / Production
:class-header: bg-danger text-white
:link: pathway-advanced
:link-type: doc

**Advanced**

You are building production systems, integrating custom backends, optimizing for constrained hardware, or working with LLMs on edge devices.

+++
→ **Advanced Pathway**
:::

::::

---

## Step 2: What is your primary goal?

Use the decision matrix below to jump directly to the most relevant section based on your goal and target platform.

```{list-table} ExecuTorch Decision Matrix
:header-rows: 1
:widths: 25 20 20 20 15

* - **Goal**
  - **Android**
  - **iOS / macOS**
  - **Desktop / Server**
  - **Embedded / MCU**
* - Run a pre-exported model quickly
  - {doc}`android-section`
  - {doc}`ios-section`
  - {doc}`getting-started`
  - {doc}`embedded-section`
* - Export my own PyTorch model
  - {doc}`using-executorch-export`
  - {doc}`using-executorch-export`
  - {doc}`getting-started`
  - {doc}`using-executorch-export`
* - Deploy an LLM (Llama, Phi, etc.)
  - {doc}`llm/llama`
  - {doc}`llm/run-on-ios`
  - {doc}`llm/working-with-llms`
  - —
* - Use hardware acceleration (NPU/GPU)
  - {doc}`android-qualcomm`
  - {doc}`ios-coreml`
  - {doc}`desktop-backends`
  - {doc}`embedded-backends`
* - Integrate a custom backend delegate
  - {doc}`backend-development`
  - {doc}`backend-development`
  - {doc}`backend-development`
  - {doc}`backend-development`
* - Profile and debug my model
  - {doc}`devtools-tutorial`
  - {doc}`devtools-tutorial`
  - {doc}`devtools-tutorial`
  - {doc}`devtools-tutorial`
* - Build from source
  - {doc}`using-executorch-building-from-source`
  - {doc}`using-executorch-building-from-source`
  - {doc}`using-executorch-building-from-source`
  - {doc}`using-executorch-building-from-source`
```

---

## Step 3: What is your role?

Different roles have different entry points into ExecuTorch. Select the one that best matches your background.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🤖 ML Engineer
:class-header: bg-primary text-white

You work primarily in Python, train models with PyTorch, and want to deploy them efficiently to edge devices.

**Start here:**
- {doc}`getting-started` — Installation and first export
- {doc}`using-executorch-export` — Model export and lowering
- {doc}`backends-overview` — Choose the right backend
- {doc}`quantization` — Reduce model size and latency
:::

:::{grid-item-card} 📱 Mobile Developer
:class-header: bg-primary text-white

You build Android or iOS applications and need to integrate an on-device ML model into your app.

**Start here:**
- {doc}`android-section` — Android integration guide
- {doc}`ios-section` — iOS integration guide
- {doc}`getting-started` — Export a model for your platform
- {doc}`using-executorch-android` — Full Android API reference
:::

:::{grid-item-card} ⚙️ Backend / Systems Developer
:class-header: bg-primary text-white

You are implementing a hardware backend, writing C++ runtime integrations, or contributing to ExecuTorch internals.

**Start here:**
- {doc}`backend-development` — Backend delegate development
- {doc}`backend-delegates-integration` — Integrating a backend
- {doc}`running-a-model-cpp-tutorial` — C++ runtime APIs
- {doc}`new-contributor-guide` — Contributing to ExecuTorch
:::

:::{grid-item-card} 🔌 Embedded Developer
:class-header: bg-primary text-white

You target microcontrollers, DSPs, or other resource-constrained hardware where memory and compute are tightly limited.

**Start here:**
- {doc}`embedded-section` — Embedded platforms overview
- {doc}`embedded-backends` — Available embedded backends
- {doc}`portable-cpp-programming` — Portable C++ for constrained devices
- {doc}`pico2_tutorial` — MNIST on Raspberry Pi Pico2
:::

::::

---

## Step 4: What is your model's status?

The right workflow depends on whether you are starting from scratch, using a supported model, or working with a custom architecture.

```{list-table} Model Status Routing
:header-rows: 1
:widths: 30 70

* - **Model Status**
  - **Recommended Path**
* - Using a supported LLM (Llama, Phi, Qwen, SmolLM)
  - Use the {doc}`llm/export-llm` script for a streamlined export with quantization and optimization built in. Pre-exported models are also available on `HuggingFace ExecuTorch Community <https://huggingface.co/executorch-community>`_.
* - Using a HuggingFace model
  - Use {doc}`llm/export-llm-optimum` (Optimum ExecuTorch) for broad HuggingFace model support with familiar APIs.
* - Using a custom PyTorch model
  - Follow {doc}`getting-started` for the standard export flow, then consult {doc}`using-executorch-export` for advanced lowering options.
* - Model requires dynamic shapes
  - See the dynamic shapes section in {doc}`using-executorch-export` and the {doc}`export-overview` for constraints.
* - Model uses unsupported operators
  - Consult {doc}`kernel-library-custom-aten-kernel` to register custom kernels, or {doc}`compiler-custom-compiler-passes` for graph-level transformations.
* - Pre-exported `.pte` file available
  - Skip export entirely and go directly to {doc}`getting-started` (Running on Device section) or your platform guide.
```

---

## Not sure where to start?

If you are completely new to ExecuTorch, the recommended entry point is the **{doc}`getting-started`** guide, which walks through installation, exporting a MobileNet V2 model, and running inference in under 15 minutes.

For a high-level conceptual overview before diving into code, read {doc}`intro-overview` and {doc}`intro-how-it-works`.

```{toctree}
:hidden:
:maxdepth: 1
:caption: User Pathways

pathway-beginner
pathway-quickstart
pathway-advanced
```
