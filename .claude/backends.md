# Backends

| Backend | Platform | Hardware | Location |
|---------|----------|----------|----------|
| XNNPACK | All | CPU | `backends/xnnpack/` |
| CUDA | Linux/Windows | GPU | `backends/cuda/` |
| CoreML | iOS, macOS | NPU/GPU/CPU | `backends/apple/coreml/` |
| MPS | iOS, macOS | GPU | `backends/apple/mps/` |
| Vulkan | Android | GPU | `backends/vulkan/` |
| QNN | Android | NPU | `backends/qualcomm/` |
| MediaTek | Android | NPU | `backends/mediatek/` |
| Arm Ethos-U | Embedded | NPU | `backends/arm/` |
| Cortex-M (CMSIS-NN) | Embedded | CPU | `backends/cortex_m/` |
| OpenVINO | Embedded | CPU/GPU/NPU | `backends/openvino/` |
| Cadence | Embedded | DSP | See `backends-cadence.md` |
| Samsung | Android | NPU | `backends/samsung/` |

## Partitioner imports
```python
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.vulkan.partition.vulkan_partitioner import VulkanPartitioner
```

## Usage pattern
```python
from executorch.exir import to_edge

edge = to_edge(exported_program)
edge = edge.to_backend(XnnpackPartitioner())  # or other partitioner
exec_prog = edge.to_executorch()
```

Unsupported ops fall back to portable CPU. Use multiple partitioners for priority fallback.
