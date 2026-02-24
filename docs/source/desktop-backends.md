(desktop-backends)=
# Backends

Available hardware acceleration backends for desktop platforms.

## Linux Backends

- {doc}`desktop-xnnpack` — XNNPACK (CPU acceleration)
- {doc}`desktop-openvino` — OpenVINO (Intel hardware optimization)

## macOS Backends

- {doc}`backends/coreml/coreml-overview` — CoreML (recommended for Apple Silicon)
- {doc}`backends/mps/mps-overview` — Metal Performance Shaders (Apple Silicon GPU)
- {doc}`desktop-xnnpack` — XNNPACK (CPU acceleration)

## Windows Backends

- {doc}`desktop-xnnpack` — XNNPACK (CPU acceleration)
- {doc}`desktop-openvino` — OpenVINO (Intel hardware optimization)

```{toctree}
:hidden:
desktop-xnnpack
desktop-openvino
backends/coreml/coreml-overview
backends/mps/mps-overview
