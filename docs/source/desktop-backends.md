(desktop-backends)=
# Backends

Available hardware acceleration backends for desktop platforms.

## Linux Backends

- {doc}`desktop-xnnpack` — XNNPACK (CPU acceleration)
- {doc}`desktop-openvino` — OpenVINO (Intel hardware optimization)
- {doc}`backends/webgpu/webgpu-overview` — WebGPU acceleration (experimental)

## macOS Backends

- {doc}`backends/coreml/coreml-overview` — CoreML (recommended for Apple Silicon)
- {doc}`backends/mlx/mlx-overview` — MLX (Apple Silicon GPU)
- {doc}`backends/webgpu/webgpu-overview` — WebGPU acceleration (experimental)
- {doc}`desktop-xnnpack` — XNNPACK (CPU acceleration)

## Windows Backends

- {doc}`desktop-xnnpack` — XNNPACK (CPU acceleration)
- {doc}`desktop-openvino` — OpenVINO (Intel hardware optimization)

```{toctree}
:hidden:
desktop-xnnpack
desktop-openvino
backends/coreml/coreml-overview
backends/mlx/mlx-overview
backends/webgpu/webgpu-overview
```
