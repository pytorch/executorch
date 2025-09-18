# Backend Overview

ExecuTorch backends provide hardware acceleration for a specific hardware target. In order to achieve maximum performance on target hardware, ExecuTorch optimizes the model for a specific backend during the export and lowering process. This means that the resulting .pte file is specialized for the specific hardware. In order to deploy to multiple backends, such as Core ML on iOS and Arm CPU on Android, it is common to generate a dedicated .pte file for each.

The choice of hardware backend is informed by the hardware that the model is intended to be deployed on. Each backend has specific hardware requires and level of model support. See the documentation for each hardware backend for more details.

As part of the .pte file creation process, ExecuTorch identifies portions of the model (partitions) that are supported for the given backend. These sections are processed by the backend ahead of time to support efficient execution. Portions of the model that are not supported on the delegate, if any, are executed using the portable fallback implementation on CPU. This allows for partial model acceleration when not all model operators are supported on the backend, but may have negative performance implications. In addition, multiple partitioners can be specified in order of priority. This allows for operators not supported on GPU to run on CPU via XNNPACK, for example.

### Available Backends

Commonly used hardware backends are listed below. For mobile, consider using XNNPACK for Android and XNNPACK or Core ML for iOS. To create a .pte file for a specific backend, pass the appropriate partitioner class to `to_edge_transform_and_lower`. See the appropriate backend documentation for more information.

- [XNNPACK (Mobile CPU)](backends-xnnpack.md)
- [Core ML (iOS)](backends-coreml.md)
- [Metal Performance Shaders (iOS GPU)](backends-mps.md)
- [Vulkan (Android GPU)](backends-vulkan.md)
- [Qualcomm NPU](backends-qualcomm.md)
- [MediaTek NPU](backends-mediatek.md)
- [ARM Ethos-U NPU](backends-arm-ethos-u.md)
- [ARM VGF](backends-arm-vgf.md)
- [Cadence DSP](backends-cadence.md)
