(pathway-beginner)=
# Beginner Pathway

**Welcome to ExecuTorch.** This pathway is designed for engineers who are comfortable with PyTorch but are new to on-device deployment. You will follow a structured, step-by-step sequence that builds foundational knowledge before introducing more complex topics.

**Estimated time to complete:** 2–4 hours for the core sequence. Individual steps can be done independently.

---

## What You Will Learn

By following this pathway, you will be able to:

1. Understand what ExecuTorch is and why it exists
2. Install ExecuTorch and set up your development environment
3. Export a PyTorch model to the `.pte` format
4. Run inference using the Python runtime
5. Deploy a model to Android or iOS
6. Know where to go next based on your use case

---

## Core Learning Sequence

Work through these steps in order. Each step builds on the previous one.

### Step 1 — Understand ExecuTorch (15 min)

Before writing any code, read the conceptual overview to understand the ExecuTorch workflow and its key benefits.

::::{grid} 2
:gutter: 2

:::{grid-item-card} Overview of ExecuTorch
:link: intro-overview
:link-type: doc

High-level introduction to ExecuTorch's purpose, design principles, and where it fits in the PyTorch ecosystem.

**Difficulty:** Beginner
:::

:::{grid-item-card} How ExecuTorch Works
:link: intro-how-it-works
:link-type: doc

A technical walkthrough of the three-stage pipeline: export, compilation, and runtime execution.

**Difficulty:** Beginner
:::

::::

---

### Step 2 — Set Up Your Environment (20 min)

Install ExecuTorch and verify your setup before attempting to export a model.

::::{grid} 1

:::{grid-item-card} Getting Started with ExecuTorch
:link: getting-started
:link-type: doc

Install the ExecuTorch Python package, export a MobileNet V2 model using XNNPACK, and run your first inference. This is the canonical entry point for all new users.

**Difficulty:** Beginner | **Prerequisites:** Python 3.10–3.13, PyTorch, g++7+ or clang5+
:::

::::

> **Tip:** If you encounter build errors or platform-specific issues during installation, consult the {doc}`using-executorch-faqs` page before proceeding.

---

### Step 3 — Understand Core Concepts (20 min)

A brief review of the key concepts and terminology used throughout ExecuTorch documentation.

::::{grid} 1

:::{grid-item-card} Core Concepts and Terminology
:link: concepts
:link-type: doc

Definitions for Export IR, Edge Dialect, delegates, partitioners, `.pte` files, and other ExecuTorch-specific terms you will encounter throughout the documentation.

**Difficulty:** Beginner
:::

::::

---

### Step 4 — Export Your First Model (30 min)

Learn the standard export workflow using `torch.export` and `to_edge_transform_and_lower`.

::::{grid} 2
:gutter: 2

:::{grid-item-card} Model Export and Lowering
:link: using-executorch-export
:link-type: doc

The complete guide to exporting a PyTorch model for ExecuTorch, including backend selection, quantization basics, and handling dynamic shapes.

**Difficulty:** Intermediate | **Builds on:** Step 2
:::

:::{grid-item-card} Visualize Your Model
:link: visualize
:link-type: doc

Use ModelExplorer to inspect your exported model graph and verify the export result before deployment.

**Difficulty:** Beginner
:::

::::

---

### Step 5 — Deploy to Your Target Platform (30–60 min)

Choose the platform you are targeting and follow the appropriate guide.

::::{grid} 3
:gutter: 2

:::{grid-item-card} 🤖 Android
:link: android-section
:link-type: doc

Integrate ExecuTorch into an Android app using the Java/Kotlin bindings. Includes Gradle dependency setup and the `Module` API.

**Difficulty:** Intermediate
:::

:::{grid-item-card} 🍎 iOS
:link: ios-section
:link-type: doc

Add ExecuTorch to an iOS or macOS project via Swift Package Manager. Covers Objective-C and Swift integration.

**Difficulty:** Intermediate
:::

:::{grid-item-card} 💻 Desktop / Python
:link: getting-started
:link-type: doc

Run inference directly from Python using the ExecuTorch runtime bindings — the fastest way to validate a model before mobile deployment.

**Difficulty:** Beginner
:::

::::

---

### Step 6 — Explore a Complete Example (optional, 30 min)

Seeing a complete end-to-end example reinforces the concepts from the previous steps.

::::{grid} 2
:gutter: 2

:::{grid-item-card} Pico2: MNIST on a Microcontroller
:link: pico2_tutorial
:link-type: doc

A self-contained tutorial that exports an MNIST model and runs it on a Raspberry Pi Pico2. Excellent for understanding the full pipeline on constrained hardware.

**Difficulty:** Beginner (hardware required)
:::

:::{grid-item-card} MobileNet V2 — Colab Notebook
:link: https://colab.research.google.com/drive/1qpxrXC3YdJQzly3mRg-4ayYiOjC6rue3?usp=sharing
:link-type: url

An interactive Colab notebook covering the complete export, lowering, and verification workflow for MobileNet V2. No local setup required.

**Difficulty:** Beginner
:::

::::

---

## Frequently Encountered Issues

New users commonly encounter the following issues. Consult these resources before opening a support request.

```{list-table}
:header-rows: 1
:widths: 40 60

* - **Issue**
  - **Resource**
* - Installation fails or package not found
  - {doc}`using-executorch-faqs` — Installation section
* - Export fails with unsupported operator error
  - {doc}`using-executorch-export` — Operator support section
* - Model produces incorrect output after export
  - {doc}`devtools-tutorial` — Numerical debugging
* - Build errors on Windows
  - {doc}`getting-started` — Windows prerequisites note
* - Backend not accelerating as expected
  - {doc}`backends-overview` — Backend selection guide
```

---

## Where to Go Next

Once you have completed the core sequence, choose your next direction based on your use case.

::::{grid} 3
:gutter: 2

:::{grid-item-card} Work with LLMs
:link: llm/working-with-llms
:link-type: doc

Export and deploy Llama, Phi, Qwen, and other LLMs to mobile and edge devices.
:::

:::{grid-item-card} Hardware Acceleration
:link: backends-overview
:link-type: doc

Use XNNPACK, Core ML, Qualcomm, Vulkan, and other backends for hardware-accelerated inference.
:::

:::{grid-item-card} Advanced Topics
:link: pathway-advanced
:link-type: doc

Quantization, memory planning, custom compiler passes, and backend development.
:::

::::
