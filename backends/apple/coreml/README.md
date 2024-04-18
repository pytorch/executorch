# ExecuTorch Core ML Delegate


This subtree contains the Core ML Delegate implementation for ExecuTorch.
Core ML is an optimized framework for running machine learning models on Apple devices. The delegate is the mechanism for leveraging the Core ML framework to accelerate operators when running on Apple devices.

## Layout
- `compiler/` : Lowers a module to Core ML backend.
- `scripts/` : Scripts for installing dependencies and running tests.
- `runtime/`: Core ML delegate runtime implementation.
    - `inmemoryfs`: InMemory filesystem implementation used to serialize/de-serialize AOT blob.
    - `kvstore`: Persistent Key-Value store implementation.
    - `delegate`: Runtime implementation.
    - `include` : Public headers.
    - `tests` :  Tests for Core ML delegate.
    - `workspace` : Xcode workspace for tests.
- `third-party/`: External dependencies.

## Help & Improvements
If you have problems or questions or have suggestions for ways to make
implementation and testing better, please create an issue on [github](https://www.github.com/pytorch/executorch/issues).

## Delegation

For delegating the Program to the **Core ML** backend, the client must be responsible for calling `to_backend` with the **CoreMLBackend** tag.

```python
import executorch.exir as exir
import torch

from torch.export import export

from executorch.exir import to_edge

from executorch.exir.backend.backend_api import to_backend

from executorch.backends.apple.coreml.compiler import CoreMLBackend

class LowerableSubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

# Convert the lowerable module to Edge IR Representation
to_be_lowered = LowerableSubModel()
example_input = (torch.ones(1), )
to_be_lowered_exir_submodule = to_edge(export(to_be_lowered, example_input))

# Lower to Core ML backend
lowered_module = to_backend('CoreMLBackend', to_be_lowered_exir_submodule.exported_program, [])
```

Currently, the **Core ML** backend delegates the whole module to **Core ML**. If a specific op is not supported by the **Core ML** backend then the `to_backend` call would throw an exception. We will be adding a **Core ML Partitioner** to resolve the issue.

The `to_backend` implementation is a thin wrapper over `coremltools`, `coremltools` is responsible for converting an **ExportedProgram** to a **MLModel**. The converted **MLModel** data is saved, flattened, and returned as bytes to **ExecuTorch**.

## Runtime

To execute a **Core ML** delegated **Program**, the client must link to the `coremldelegate` library. Once linked there are no additional steps required, **ExecuTorch** when running the **Program** would call the **Core ML** runtime to execute the **Core ML** delegated part of the **Program**.

Please follow the instructions described in the [Core ML setup](/backends/apple/coreml/setup.md) to link the `coremldelegate` library.
