(edge-platforms-section)=
# Edge

Deploy ExecuTorch on mobile, desktop, and embedded platforms with optimized backends for each.

ExecuTorch supports deployment across a wide variety of edge computing platforms, from high-end mobile devices to constrained embedded systems and microcontrollers.

## Android

Deploy ExecuTorch on Android devices with hardware acceleration support.

**→ {doc}`android-section` — Complete Android deployment guide**

Key features:

- Hardware acceleration support (CPU, GPU, NPU)
- Multiple backend options (XNNPACK, Vulkan, Qualcomm, MediaTek, ARM, Samsung)
- Comprehensive examples and demos

## iOS

Deploy ExecuTorch on iOS devices with Apple hardware acceleration.

**→ {doc}`ios-section` — Complete iOS deployment guide**

Key features:
- Apple hardware optimization (CoreML, MPS, XNNPACK)
- Swift and Objective-C integration
- LLM and computer vision examples

## Desktop & Laptop Platforms

Deploy ExecuTorch on Linux, macOS, and Windows with optimized backends.

**→ {doc}`desktop-section` — Complete desktop deployment guide**

Key features:
- Cross-platform C++ runtime
- Platform-specific optimization (OpenVINO, CoreML, MPS)
- CPU and GPU acceleration options

## Embedded Systems

Deploy ExecuTorch on constrained embedded systems and microcontrollers.

**→ {doc}`embedded-section` — Complete embedded deployment guide**

Key features:

- Resource-constrained deployment
- DSP and NPU acceleration (Cadence, ARM Ethos-U, NXP)
- Custom backend development support
- LLM and computer vision examples

## Troubleshooting & Support

- **{doc}`using-executorch-troubleshooting`** - Common issues and solutions across all platforms

## Next Steps

After choosing your platform:

- **{doc}`backends-section`** - Deep dive into backend selection and optimization
- **{doc}`llm/working-with-llms`** - Working with Large Language Models on edge devices

```{toctree}
:hidden:
:maxdepth: 2
:caption: Edge Platforms

android-section
ios-section
desktop-section
embedded-section
using-executorch-troubleshooting
