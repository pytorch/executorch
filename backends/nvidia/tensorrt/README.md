# ExecuTorch TensorRT Delegate

This subtree contains the TensorRT Delegate implementation for ExecuTorch.
TensorRT is NVIDIA's high-performance deep learning inference optimizer and
runtime library. The delegate leverages TensorRT to accelerate model execution
on NVIDIA GPUs.

## Prerequisites

### TensorRT Installation

TensorRT is required for both ahead-of-time (AOT) compilation and runtime
execution. The installation method depends on your platform:

#### NVIDIA Jetson (aarch64)

TensorRT is **pre-installed** via JetPack SDK. No additional installation is
required.

To verify your TensorRT installation:
```bash
dpkg -l | grep -i tensorrt
```

> **Note:** Ensure you are using JetPack 6.x or later for TensorRT 10.x support.

#### Linux x86_64

Install TensorRT via pip:
```bash
pip install tensorrt>=10.3
```

Alternatively, download and install from the
[NVIDIA TensorRT Download Page](https://developer.nvidia.com/tensorrt).

#### Windows x86_64

Download and install from the
[NVIDIA TensorRT Download Page](https://developer.nvidia.com/tensorrt).

### Additional Requirements

- **CUDA Toolkit**: TensorRT requires a compatible CUDA installation
- **cuDNN**: Required for certain layer optimizations
- **NVIDIA GPU**: Compute capability 7.0 or higher recommended

## Supported Platforms

| Platform | Architecture | TensorRT Source |
|----------|-------------|-----------------|
| Linux | x86_64 | pip or NVIDIA installer |
| Linux (Jetson) | aarch64 | Pre-installed via JetPack |
| Windows | x86_64 | NVIDIA installer |
