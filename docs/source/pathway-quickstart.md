(pathway-quickstart)=
# Quick Start Pathway

This pathway is for engineers who want to get a model running on a device as quickly as possible. It assumes you are familiar with PyTorch model development and have some prior exposure to mobile or edge deployment concepts. Steps are kept concise and link directly to the most actionable documentation.

**Estimated time to first inference:** 15–30 minutes.

---

## Choose Your Scenario

Select the scenario that most closely matches what you are trying to accomplish right now.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🚀 I have a PyTorch model and want to run it on device
:class-header: bg-primary text-white

**Fastest path: Export → Run**

1. Install: `pip install executorch`
2. Export with {doc}`getting-started` (Exporting section)
3. Run with Python runtime or deploy to {doc}`android-section` / {doc}`ios-section`

**Time:** ~15 min
:::

:::{grid-item-card} 📦 I want to use a pre-exported model
:class-header: bg-primary text-white

**Fastest path: Download → Run**

Pre-exported `.pte` files for Llama 3.2, MobileNet, and other models are available on [HuggingFace ExecuTorch Community](https://huggingface.co/executorch-community).

Skip export entirely and go directly to the runtime section of {doc}`getting-started`.

**Time:** ~10 min
:::

:::{grid-item-card} 🤗 I have a HuggingFace model
:class-header: bg-primary text-white

**Fastest path: Optimum ExecuTorch**

Use the `optimum-executorch` CLI for a one-command export of HuggingFace models.

See {doc}`llm/export-llm-optimum` for installation and usage.

**Time:** ~20 min
:::

:::{grid-item-card} 🦙 I want to run Llama on my phone
:class-header: bg-primary text-white

**Fastest path: Llama on ExecuTorch**

Follow the {doc}`llm/llama` guide for the complete Llama export and deployment workflow, including quantization and platform-specific setup.

**Time:** ~45 min (model download included)
:::

::::

---

## The 5-Minute Setup

If you have not yet installed ExecuTorch, run the following in a Python 3.10–3.13 virtual environment:

```bash
pip install executorch
```

Then verify the installation with a minimal export:

```python
import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# Define a simple model
class Add(torch.nn.Module):
    def forward(self, x, y):
        return x + y

model = Add()
sample_inputs = (torch.ones(1), torch.ones(1))

et_program = to_edge_transform_and_lower(
    torch.export.export(model, sample_inputs),
    partitioner=[XnnpackPartitioner()]
).to_executorch()

with open("add.pte", "wb") as f:
    f.write(et_program.buffer)

print("Export successful: add.pte created")
```

If this runs without error, your environment is correctly configured.

---

## Quick Reference: Export Cheat Sheet

```{list-table}
:header-rows: 1
:widths: 30 70

* - **Task**
  - **Code / Command**
* - Install ExecuTorch
  - `pip install executorch`
* - Export with XNNPACK (mobile CPU)
  - `to_edge_transform_and_lower(torch.export.export(model, inputs), partitioner=[XnnpackPartitioner()])`
* - Export with Core ML (iOS)
  - Replace `XnnpackPartitioner` with `CoreMLPartitioner` — see {doc}`ios-coreml`
* - Export with Qualcomm (Android NPU)
  - See {doc}`android-qualcomm` for QNN SDK setup and partitioner usage
* - Run from Python
  - `Runtime.get().load_program("model.pte").load_method("forward").execute([input])`
* - Run from C++
  - See {doc}`extension-module` for the high-level `Module` API
* - Export an LLM
  - `python -m executorch.examples.models.llama.export_llm ...` — see {doc}`llm/export-llm`
```

---

## Platform Quick Start Guides

Jump directly to the platform-specific setup guide for your target.

::::{grid} 2
:gutter: 2

:::{grid-item-card} Android Quick Start
:link: android-section
:link-type: doc

Gradle dependency, Java `Module` API, and XNNPACK / Vulkan / Qualcomm backend selection for Android.
:::

:::{grid-item-card} iOS Quick Start
:link: ios-section
:link-type: doc

Swift Package Manager setup, Objective-C runtime API, and Core ML / MPS / XNNPACK backend selection for iOS.
:::

:::{grid-item-card} Desktop / Linux / macOS
:link: desktop-section
:link-type: doc

Python runtime, C++ CMake integration, and XNNPACK / Core ML / MPS backends for desktop platforms.
:::

:::{grid-item-card} Embedded Systems
:link: embedded-section
:link-type: doc

Bare-metal and RTOS deployment, Arm Ethos-U, Cadence, NXP, and other embedded backends.
:::

::::

---

## Backend Selection Guide

Choosing the right backend has the largest impact on performance. Use this table to select the appropriate backend for your hardware.

```{list-table} Backend Selection by Platform and Hardware
:header-rows: 1
:widths: 20 20 20 40

* - **Platform**
  - **Hardware Target**
  - **Backend**
  - **Documentation**
* - Android
  - CPU (Arm/x86)
  - XNNPACK
  - {doc}`android-xnnpack`
* - Android
  - GPU (Vulkan)
  - Vulkan
  - {doc}`android-vulkan`
* - Android
  - Qualcomm NPU/DSP
  - QNN
  - {doc}`android-qualcomm`
* - Android
  - MediaTek APU
  - MediaTek
  - {doc}`android-mediatek`
* - iOS / macOS
  - Neural Engine / GPU
  - Core ML
  - {doc}`ios-coreml`
* - iOS / macOS
  - Metal GPU
  - MPS
  - {doc}`ios-mps`
* - iOS / macOS
  - CPU (Arm)
  - XNNPACK
  - {doc}`ios-xnnpack`
* - Desktop
  - Intel CPU/GPU/NPU
  - OpenVINO
  - {doc}`desktop-openvino`
* - Desktop
  - Apple Silicon
  - Core ML / MPS
  - {doc}`desktop-coreml`
* - Embedded
  - Arm Cortex-M / Ethos-U
  - Arm Ethos-U
  - {doc}`embedded-arm-ethos-u`
* - Embedded
  - Cadence DSP
  - Cadence
  - {doc}`embedded-cadence`
* - Embedded
  - NXP eIQ Neutron
  - NXP
  - {doc}`embedded-nxp`
```

---

## Troubleshooting Quick Fixes

```{list-table}
:header-rows: 1
:widths: 40 60

* - **Symptom**
  - **Quick Fix**
* - `ImportError: No module named executorch`
  - Run `pip install executorch` in your active virtual environment
* - Export fails with `torch._dynamo` error
  - Ensure your model is `export`-compatible; see {doc}`export-overview`
* - `.pte` file runs but produces wrong output
  - Use {doc}`devtools-tutorial` to compare intermediate activations
* - Android Gradle sync fails
  - Check `executorch_version` in `build.gradle.kts` matches your installed version
* - iOS build fails with missing xcframework
  - Verify the Swift PM branch name matches your ExecuTorch version (format: `swiftpm-X.Y.Z`)
```

---

## Going Deeper

Once your model is running, explore these topics to optimize performance and expand capabilities.

- {doc}`quantization` — Reduce model size and improve latency with INT8/INT4 quantization
- {doc}`using-executorch-troubleshooting` — Profiling and debugging tools
- {doc}`using-executorch-export` — Advanced export options including dynamic shapes
- {doc}`pathway-advanced` — Full advanced user pathway for production deployments
