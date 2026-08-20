# ExecuTorch: Inference on consumer Desktops/Laptops with GPUs

## Overview

ExecuTorch is a lightweight, flexible runtime designed for efficient AI inference, historically focused on mobile and embedded devices. With the growing demand for local inference on personal desktops and laptops—especially those equipped with consumer GPUs (e.g., gaming PCs with NVIDIA hardware)—ExecuTorch is experimenting on expanding its capabilities to support these platforms.

## Historical Context
- **Mobile and Embedded Focus**: ExecuTorch’s initial target market was mobile and embedded devices.
- **Desktop/Laptop Support**: Previously, desktop and laptop ("AI PC") inference was enabled through backends such as XNNPACK, OpenVino, and Qualcomm NPUs.
- **No CUDA Support**: For a long time, ExecuTorch did not offer a CUDA backend, limiting GPU acceleration on NVIDIA hardware.

## Recent Developments
With increased demand for local inference on consumer desktops and laptops, exemplified by popular runtimes like llama.cpp and MLX, ExecuTorch is now experimenting with CUDA and Metal support. This is achieved by leveraging Inductor compiler technology from PyTorch, specifically using Ahead-of-Time Inductor [AOTI](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html) to avoid reinventing the wheel.

## Key Benefits
- **Model Agnostic**: Validated on models such as [Voxtral](../examples/models/voxtral), [Gemma3-4b](../examples/models/gemma3), ResNet, and [Whisper](../examples/models/whisper/README.md). Theoretically, any model exportable via torch.export is supported.
- **PyTorch Ecosystem Integration**: Enables workflows for fine-tuning, quantization, and compilation within the PyTorch ecosystem.
- **No Python Runtime During Inference**: Ideal for native applications (e.g., written in C++) embedding AI capabilities.
- **No libtorch Dependency**: Reduces binary size, making deployment easier for resource-constrained applications.
- **Efficient GPU Support**: Uses AOTI-powered CUDA backend for efficient inference on NVIDIA GPUs.

## Backends

Backends leveraging AoTi
- [CUDA backend](../backends/cuda)
- [Metal backend](../backends/apple/metal)

## Roadmap & Limitations
- **Experimental Status**: CUDA and Metal backends via AoTi are currently experimental. Contributions and feedback are welcome!
- **Model Compatibility**: While most models exportable via torch.export should work, validation is ongoing for broader model support.
- **Portability**: Figuring out the balance and trade-off between performance, portability and model filesize.
- **Windows-native WIP**: On windows we only supports WSL right now. Native Windows support is WIP.
