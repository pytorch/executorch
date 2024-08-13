# ExecuTorch Core ML Delegate


This subtree contains the Core ML Delegate implementation for ExecuTorch.
Core ML is an optimized framework for running machine learning models on Apple devices. The delegate is the mechanism for leveraging the Core ML framework to accelerate operators when running on Apple devices.

## Layout
- `compiler/` : Lowers a module to Core ML backend.
- `partition/`: Partitions a module fully or partially to Core ML backend.
- `quantizer/`: Quantizes a module in Core ML favored scheme.
- `scripts/` : Scripts for installing dependencies and running tests.
- `runtime/`: Core ML delegate runtime implementation.
    - `inmemoryfs`: InMemory filesystem implementation used to serialize/de-serialize AOT blob.
    - `kvstore`: Persistent Key-Value store implementation.
    - `delegate`: Runtime implementation.
    - `include` : Public headers.
    - `sdk` : SDK implementation.
    - `tests` :  Unit tests.
    - `workspace` : Xcode workspace for the runtime.
- `third-party/`: External dependencies.

## Partition and Delegation

To delegate a Program to the **Core ML** backend, the client must call `to_backend` with the **CoreMLPartitioner**.

```python
import torch
import executorch.exir

from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

source_model = Model()
example_inputs = (torch.ones(1), )

# Export the source model to Edge IR representation
aten_program = torch.export.export(source_model, example_inputs)
edge_program_manager = executorch.exir.to_edge(aten_program)

# Delegate to Core ML backend
delegated_program_manager = edge_program_manager.to_backend(CoreMLPartitioner())

# Serialize delegated program
executorch_program = delegated_program_manager.to_executorch()
with open("model.pte", "wb") as f:
    f.write(executorch_program.buffer)
```

The module will be fully or partially delegated to **Core ML**, depending on whether all or part of ops are supported by the **Core ML** backend. User may force skip certain ops by `CoreMLPartitioner(skip_ops_for_coreml_delegation=...)`

The `to_backend` implementation is a thin wrapper over [coremltools](https://apple.github.io/coremltools/docs-guides/), `coremltools` is responsible for converting an **ExportedProgram** to a **MLModel**. The converted **MLModel** data is saved, flattened, and returned as bytes to **ExecuTorch**.

## Quantization

To quantize a Program in a Core ML favored way, the client may utilize **CoreMLQuantizer**.

```python
import torch
import executorch.exir

from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)

from executorch.backends.apple.coreml.quantizer import CoreMLQuantizer
from coremltools.optimize.torch.quantization.quantization_config import (
    LinearQuantizerConfig,
    QuantizationScheme,
)

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv(x)
        return self.relu(a)

source_model = Model()
example_inputs = (torch.randn((1, 3, 256, 256)), )

pre_autograd_aten_dialect = capture_pre_autograd_graph(model, example_inputs)

quantization_config = LinearQuantizerConfig.from_dict(
    {
        "global_config": {
            "quantization_scheme": QuantizationScheme.symmetric,
            "activation_dtype": torch.uint8,
            "weight_dtype": torch.int8,
            "weight_per_channel": True,
        }
    }
)
quantizer = CoreMLQuantizer(quantization_config)

# For post-training quantization, use `prepare_pt2e`
# For quantization-aware trainin,g use `prepare_qat_pt2e`
prepared_graph = prepare_pt2e(pre_autograd_aten_dialect, quantizer)

prepared_graph(*example_inputs)
converted_graph = convert_pt2e(prepared_graph)
```

The `converted_graph` is the quantized torch model, and can be delegated to **Core ML** similarly through **CoreMLPartitioner**

## Runtime

To execute a Core ML delegated program, the application must link to the `coremldelegate` library. Once linked there are no additional steps required, ExecuTorch when running the program would call the Core ML runtime to execute the Core ML delegated part of the program.

Please follow the instructions described in the [Core ML setup](/backends/apple/coreml/setup.md) to link the `coremldelegate` library.

## Help & Improvements
If you have problems or questions or have suggestions for ways to make
implementation and testing better, please create an issue on [github](https://www.github.com/pytorch/executorch/issues).
