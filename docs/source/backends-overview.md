# Backends

## Backend Overview

ExecuTorch backends provide hardware acceleration for specific hardware targets, enabling models to run efficiently on devices ranging from mobile phones to embedded systems and DSPs. During the export and lowering process, ExecuTorch optimizes your model for the chosen backend, resulting in a `.pte` file specialized for that hardware. To support multiple platforms (e.g., Core ML on iOS, Arm CPU on Android), you typically generate a dedicated `.pte` file for each backend.

The choice of backend is informed by the hardware your model will run on. Each backend has its own hardware requirements and level of model/operator support. See the documentation for each backend for details.

As part of `.pte` file creation, ExecuTorch identifies model partitions supported by the backend. These are processed ahead of time for efficient execution. Operators not supported by the delegate are executed using the portable CPU fallback (e.g., XNNPACK), allowing for partial acceleration. You can also specify multiple partitioners in order of priority, so unsupported GPU ops can fall back to CPU, for example.

---

## Why Backends Matter

Backends are the bridge between your exported model and the hardware it runs on. Choosing the right backend ensures your model takes full advantage of device-specific acceleration, balancing performance, compatibility, and resource usage.

---

## Choosing a Backend

| Backend                                                      | Platform(s)   | Hardware Type | Typical Use Case                |
|--------------------------------------------------------------|---------------|---------------|---------------------------------|
| [XNNPACK](backends/xnnpack/xnnpack-overview.md)              | All           | CPU           | General-purpose, fallback       |
| [CUDA](/backends/cuda/cuda-overview.md)                      | Linux/Windows | GPU           | NVIDIA GPU acceleration         |
| [Core ML](/backends/coreml/coreml-overview.md)               | iOS, macOS    | NPU/GPU/CPU   | Apple devices, high performance |
| [Metal Performance Shaders](/backends/mps/mps-overview.md)   | iOS, macOS    | GPU           | Apple GPU acceleration          |
| [Vulkan ](/backends/vulkan/vulkan-overview.md)               | Android       | GPU           | Android GPU acceleration        |
| [Qualcomm](backends-qualcomm)                                | Android     | NPU           | Qualcomm SoCs                   |
| [MediaTek](backends-mediatek)                                | Android     | NPU           | MediaTek SoCs                   |
| [Arm Ethos-U](/backends/arm-ethos-u/arm-ethos-u-overview.md) | Embedded    | NPU           | Arm MCUs                        |
| [Arm VGF](/backends/arm-vgf/arm-vgf-overview.md)             | Android     | GPU           | Arm platforms                   |
| [OpenVINO](build-run-openvino)                               | Embedded    | CPU/GPU/NPU   | Intel SoCs                      |
| [NXP](backends/nxp/nxp-overview.md)                          | Embedded    | NPU           | NXP SoCs                        |
| [Cadence](backends-cadence)                                  | Embedded    | DSP           | DSP-optimized workloads         |
| [Samsung Exynos](/backends/samsung/samsung-overview.md)      | Android     | NPU           | Samsung SoCs                    |

**Tip:** For best performance, export a `.pte` file for each backend you plan to support.

---

## Best Practices

- **Test on all target devices:** Operator support may vary by backend.
- **Use fallback wisely:** If a backend doesn't support an operator, ExecuTorch will run it on CPU.
- **Consult backend docs:** Each backend has unique setup and tuning options.

---

```{toctree}
:maxdepth: 3
:hidden:
:caption: Backend Overview

backends/xnnpack/xnnpack-overview
backends/cuda/cuda-overview
backends/coreml/coreml-overview
backends/mps/mps-overview
backends/vulkan/vulkan-overview
backends-qualcomm
backends-mediatek
backends/arm-ethos-u/arm-ethos-u-overview
backends/arm-vgf/arm-vgf-overview
build-run-openvino
backends/nxp/nxp-overview
backends-cadence
backends/samsung/samsung-overview
